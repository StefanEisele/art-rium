"""
Key-frame video generation — two workflow types:

  i2v_multi  Each image is animated independently (WanImageToVideo), clips concatenated.
             Supports 1–3 images with per-image prompts and frame counts.

  flf2v      Images are used as first/last key frames (WanFirstLastFrameToVideo).
             Requires 2–3 images with a shared prompt.

POST /api/video/generate            → enqueue job, return {video_id}
GET  /api/video/jobs/{id}           → poll status
GET  /api/video/jobs/{id}/progress  → lightweight progress (ComfyUI queue + phase)
GET  /api/video/thumb/{id}          → first-frame JPEG thumbnail
GET  /api/video/file/{fname}        → serve MP4
GET  /api/videos                    → list all videos
DELETE /api/video/{id}              → delete
"""
import asyncio
import logging
import random
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from core.auth import require_auth
from core.config import settings
from core.db import AsyncSessionLocal, get_db
from core.models import Image, Video
from services.comfy.client import (
    poll_history,
    post_workflow,
    queue_info,
    upload_image,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/video", dependencies=[Depends(require_auth)])

# ── Per-job progress (module-level, single-process safe) ──────────────────────
# str(video_id) → {"phase": str, "message": str, "pct": int}
_progress: dict[str, dict] = {}

# ── Constants ─────────────────────────────────────────────────────────────────

_NEG_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)
_CLIP_NAME  = "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
_UNET_HIGH  = "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"
_UNET_LOW   = "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"
_LORA_HIGH  = "wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors"
_LORA_LOW   = "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors"
_VAE_NAME   = "wan_2.1_vae.safetensors"
_RIFE_CKPT  = "rife49.pth"

POLL_INTERVAL = 15    # seconds between ComfyUI history polls
POLL_TIMEOUT  = 1800  # 30 minutes max
SAVE_NODE_ID  = "save"

# ── Pydantic ──────────────────────────────────────────────────────────────────

class GenerateVideoRequest(BaseModel):
    image_ids: list[uuid.UUID]
    workflow: str = "i2v_multi"    # "i2v_multi" | "flf2v"
    width:  int = 1088
    height: int = 1088
    frame_count: int = 25          # flf2v: frames per transition; i2v_multi: per-image fallback
    fps:    int = 24
    prompt: str = ""               # flf2v: shared prompt; i2v_multi: per-image fallback
    prompts: list[str] = []        # i2v_multi: per-image prompts (one per image_id)
    frame_counts: list[int] = []   # i2v_multi: per-image frame counts (one per image_id)


# ── FLF2V workflow builder (key-frame transitions) ────────────────────────────

def _transition_nodes(
    t: int,
    start_img_node: str,
    end_img_node: str,
    width: int, height: int, length: int,
    prompt: str, seed: int,
) -> tuple[dict, str]:
    """Build one Wan 2.2 FLF2V transition subgraph. Returns (nodes, decode_node_id)."""
    p = f"t{t}_"
    nodes = {
        p+"clip":   {"class_type": "CLIPLoader",          "inputs": {"clip_name": _CLIP_NAME, "type": "wan", "device": "default"}},
        p+"pos":    {"class_type": "CLIPTextEncode",      "inputs": {"clip": [p+"clip", 0], "text": prompt}},
        p+"neg":    {"class_type": "CLIPTextEncode",      "inputs": {"clip": [p+"clip", 0], "text": _NEG_PROMPT}},
        p+"vae":    {"class_type": "VAELoader",           "inputs": {"vae_name": _VAE_NAME}},
        p+"unet_h": {"class_type": "UNETLoader",          "inputs": {"unet_name": _UNET_HIGH, "weight_dtype": "default"}},
        p+"lora_h": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": [p+"unet_h", 0], "lora_name": _LORA_HIGH, "strength_model": 1}},
        p+"samp_h": {"class_type": "ModelSamplingSD3",    "inputs": {"model": [p+"lora_h", 0], "shift": 5}},
        p+"unet_l": {"class_type": "UNETLoader",          "inputs": {"unet_name": _UNET_LOW, "weight_dtype": "default"}},
        p+"lora_l": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": [p+"unet_l", 0], "lora_name": _LORA_LOW, "strength_model": 1}},
        p+"samp_l": {"class_type": "ModelSamplingSD3",    "inputs": {"model": [p+"lora_l", 0], "shift": 5}},
        p+"flf2v":  {"class_type": "WanFirstLastFrameToVideo", "inputs": {
            "positive":    [p+"pos",  0],
            "negative":    [p+"neg",  0],
            "vae":         [p+"vae",  0],
            "start_image": [start_img_node, 0],
            "end_image":   [end_img_node,   0],
            "width": width, "height": height, "length": length, "batch_size": 1,
        }},
        p+"ks_h":   {"class_type": "KSamplerAdvanced", "inputs": {
            "model":                     [p+"samp_h", 0],
            "add_noise":                 "enable",
            "noise_seed":                seed,
            "steps": 4, "cfg": 1, "sampler_name": "euler", "scheduler": "simple",
            "start_at_step": 0, "end_at_step": 2,
            "return_with_leftover_noise": "enable",
            "positive":     [p+"flf2v", 0],
            "negative":     [p+"flf2v", 1],
            "latent_image": [p+"flf2v", 2],
        }},
        p+"ks_l":   {"class_type": "KSamplerAdvanced", "inputs": {
            "model":                     [p+"samp_l", 0],
            "add_noise":                 "disable",
            "noise_seed":                0,
            "steps": 4, "cfg": 1, "sampler_name": "euler", "scheduler": "simple",
            "start_at_step": 2, "end_at_step": 10000,
            "return_with_leftover_noise": "disable",
            "positive":     [p+"flf2v", 0],
            "negative":     [p+"flf2v", 1],
            "latent_image": [p+"ks_h",  0],
        }},
        p+"decode": {"class_type": "VAEDecode", "inputs": {
            "samples": [p+"ks_l", 0],
            "vae":     [p+"vae",  0],
        }},
    }
    return nodes, p + "decode"


def _build_flf2v_workflow(
    comfy_filenames: list[str],
    width: int, height: int, frame_count: int, fps: int,
    prompt: str, vid_prefix: str,
) -> dict:
    """FLF2V: transitions between key frames → ImageBatch chain → RIFE ×3 → VHS h265."""
    n_img = len(comfy_filenames)
    wf: dict = {}

    img_ids: list[str] = []
    for i, fname in enumerate(comfy_filenames):
        nid = f"img{i + 1}"
        wf[nid] = {"class_type": "LoadImage", "inputs": {"image": fname, "upload": "image"}}
        img_ids.append(nid)

    decode_ids: list[str] = []
    for t in range(n_img - 1):
        seed = random.randint(0, 2**32 - 1)
        nodes, decode_id = _transition_nodes(
            t + 1, img_ids[t], img_ids[t + 1],
            width, height, frame_count, prompt, seed,
        )
        wf.update(nodes)
        decode_ids.append(decode_id)

    if len(decode_ids) == 1:
        all_frames = decode_ids[0]
    else:
        wf["batch_mid"] = {"class_type": "ImageBatch", "inputs": {
            "image1": [decode_ids[0], 0],
            "image2": [decode_ids[1], 0],
        }}
        all_frames = "batch_mid"

    # Append raw end frame so video lands cleanly on the final image
    wf["batch_final"] = {"class_type": "ImageBatch", "inputs": {
        "image1": [all_frames,  0],
        "image2": [img_ids[-1], 0],
    }}

    # RIFE VFI — 3× frame interpolation
    wf["rife"] = {"class_type": "RIFE VFI", "inputs": {
        "ckpt_name":                  _RIFE_CKPT,
        "clear_cache_after_n_frames": 10,
        "multiplier":                 3,
        "fast_mode":                  True,
        "ensemble":                   True,
        "scale_factor":               1,
        "dtype":                      "float32",
        "torch_compile":              False,
        "batch_size":                 1,
        "frames":                     ["batch_final", 0],
    }}

    wf[SAVE_NODE_ID] = {"class_type": "VHS_VideoCombine", "inputs": {
        "frame_rate":      fps,
        "loop_count":      0,
        "filename_prefix": vid_prefix,
        "format":          "video/h265-mp4",
        "pix_fmt":         "yuv420p10le",
        "crf":             22,
        "save_metadata":   False,
        "pingpong":        False,
        "save_output":     True,
        "images":          ["rife", 0],
    }}

    return wf


# ── i2v_multi workflow builder (independent clips per image) ──────────────────

def _i2v_segment(
    seg: int, img_node_id: str,
    prompt: str, frame_count: int,
    width: int, height: int, seed: int,
) -> dict:
    """One WanImageToVideo segment with 4-step LoRA, two-pass KSampler, RIFE ×2."""
    p = f"s{seg}_"
    return {
        p+"clip":   {"class_type": "CLIPLoader",          "inputs": {"clip_name": _CLIP_NAME, "type": "wan", "device": "default"}},
        p+"vae":    {"class_type": "VAELoader",           "inputs": {"vae_name": _VAE_NAME}},
        p+"unet_h": {"class_type": "UNETLoader",          "inputs": {"unet_name": _UNET_HIGH, "weight_dtype": "default"}},
        p+"unet_l": {"class_type": "UNETLoader",          "inputs": {"unet_name": _UNET_LOW,  "weight_dtype": "default"}},
        p+"lora_h": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": [p+"unet_h", 0], "lora_name": _LORA_HIGH, "strength_model": 1.0}},
        p+"lora_l": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": [p+"unet_l", 0], "lora_name": _LORA_LOW,  "strength_model": 1.0}},
        p+"samp_h": {"class_type": "ModelSamplingSD3",    "inputs": {"model": [p+"lora_h", 0], "shift": 5.0}},
        p+"samp_l": {"class_type": "ModelSamplingSD3",    "inputs": {"model": [p+"lora_l", 0], "shift": 5.0}},
        p+"pos":    {"class_type": "CLIPTextEncode",      "inputs": {"clip": [p+"clip", 0], "text": prompt}},
        p+"neg":    {"class_type": "CLIPTextEncode",      "inputs": {"clip": [p+"clip", 0], "text": _NEG_PROMPT}},
        p+"i2v":    {"class_type": "WanImageToVideo",     "inputs": {
            "width": width, "height": height, "length": frame_count, "batch_size": 1,
            "positive": [p+"pos", 0], "negative": [p+"neg", 0],
            "vae": [p+"vae", 0], "start_image": [img_node_id, 0],
        }},
        p+"ks_h":   {"class_type": "KSamplerAdvanced", "inputs": {
            "model": [p+"samp_h", 0], "add_noise": "enable", "noise_seed": seed,
            "steps": 4, "cfg": 1, "sampler_name": "euler", "scheduler": "simple",
            "start_at_step": 0, "end_at_step": 2, "return_with_leftover_noise": "enable",
            "positive": [p+"i2v", 0], "negative": [p+"i2v", 1], "latent_image": [p+"i2v", 2],
        }},
        p+"ks_l":   {"class_type": "KSamplerAdvanced", "inputs": {
            "model": [p+"samp_l", 0], "add_noise": "disable", "noise_seed": 0,
            "steps": 4, "cfg": 1, "sampler_name": "euler", "scheduler": "simple",
            "start_at_step": 2, "end_at_step": 10000, "return_with_leftover_noise": "disable",
            "positive": [p+"i2v", 0], "negative": [p+"i2v", 1], "latent_image": [p+"ks_h", 0],
        }},
        p+"decode": {"class_type": "VAEDecode", "inputs": {"samples": [p+"ks_l", 0], "vae": [p+"vae", 0]}},
        p+"rife":   {"class_type": "RIFE VFI", "inputs": {
            "ckpt_name": _RIFE_CKPT, "clear_cache_after_n_frames": 10, "multiplier": 2,
            "fast_mode": True, "ensemble": True, "scale_factor": 1,
            "dtype": "float32", "torch_compile": False, "batch_size": 1,
            "frames": [p+"decode", 0],
        }},
    }


def _build_i2v_multi_workflow(
    comfy_filenames: list[str],
    prompts: list[str],
    frame_counts: list[int],
    width: int, height: int, fps: int,
    vid_prefix: str,
) -> tuple[dict, str]:
    """i2v_multi: each image animated independently, RIFE clips concatenated."""
    n = len(comfy_filenames)
    wf: dict = {}
    rife_ids: list[str] = []

    for i, (fname, prompt, fc) in enumerate(zip(comfy_filenames, prompts, frame_counts)):
        img_id = f"img{i}"
        wf[img_id] = {"class_type": "LoadImage", "inputs": {"image": fname, "upload": "image"}}
        wf.update(_i2v_segment(i, img_id, prompt, fc, width, height, random.randint(0, 2**32 - 1)))
        rife_ids.append(f"s{i}_rife")

    if n == 1:
        frames_node = rife_ids[0]
    elif n == 2:
        wf["batch01"] = {"class_type": "ImageBatch", "inputs": {
            "image1": [rife_ids[0], 0], "image2": [rife_ids[1], 0],
        }}
        frames_node = "batch01"
    else:
        wf["batch01"] = {"class_type": "ImageBatch", "inputs": {
            "image1": [rife_ids[0], 0], "image2": [rife_ids[1], 0],
        }}
        wf["batch012"] = {"class_type": "ImageBatch", "inputs": {
            "image1": ["batch01", 0], "image2": [rife_ids[2], 0],
        }}
        frames_node = "batch012"

    save_id = "i2v_save"
    wf[save_id] = {"class_type": "VHS_VideoCombine", "inputs": {
        "frame_rate":      fps,
        "loop_count":      0,
        "filename_prefix": vid_prefix,
        "format":          "video/h265-mp4",
        "pix_fmt":         "yuv420p10le",
        "crf":             22,
        "save_metadata":   False,
        "pingpong":        False,
        "save_output":     True,
        "images":          [frames_node, 0],
    }}

    return wf, save_id


# ── Background generation task ────────────────────────────────────────────────

async def _run_generation(video_id: uuid.UUID, req: GenerateVideoRequest) -> None:
    vid_key = str(video_id)
    prefix  = f"artrium_{video_id.hex[:10]}"

    def _set(phase: str, message: str, pct: int) -> None:
        _progress[vid_key] = {"phase": phase, "message": message, "pct": pct}

    try:
        _set("uploading", "Uploading images to ComfyUI…", 5)

        async with AsyncSessionLocal() as db:
            video = await db.get(Video, video_id)
            if not video:
                return
            img_result = await db.execute(
                select(Image).where(Image.id.in_(req.image_ids))
            )
            images_by_id = {img.id: img for img in img_result.scalars().all()}

        ordered_images = [images_by_id[iid] for iid in req.image_ids if iid in images_by_id]
        if len(ordered_images) < 1:
            raise ValueError("Need at least 1 valid image")

        async with httpx.AsyncClient(timeout=60) as client:
            comfy_names: list[str] = []
            for i, img in enumerate(ordered_images):
                src = settings.storage_dir / img.filepath
                assigned_name = await upload_image(client, src, f"{prefix}_kf{i + 1}.png")
                comfy_names.append(assigned_name)
                pct = 5 + int(12 * (i + 1) / len(ordered_images))
                _set("uploading", f"Uploaded image {i+1}/{len(ordered_images)}", pct)
                logger.info("Uploaded %s → ComfyUI:%s", src.name, assigned_name)

            _set("submitting", "Submitting workflow to ComfyUI…", 20)
            n_imgs = len(comfy_names)

            if req.workflow == "flf2v":
                if n_imgs < 2:
                    raise ValueError("FLF2V requires at least 2 images")
                wf = _build_flf2v_workflow(
                    comfy_names, req.width, req.height, req.frame_count, req.fps, req.prompt, prefix,
                )
                active_save_node = SAVE_NODE_ID
            else:  # i2v_multi
                prompts_list = req.prompts if len(req.prompts) == n_imgs else [req.prompt] * n_imgs
                fc_list = req.frame_counts if len(req.frame_counts) == n_imgs else [req.frame_count] * n_imgs
                wf, active_save_node = _build_i2v_multi_workflow(
                    comfy_names, prompts_list, fc_list, req.width, req.height, req.fps, prefix,
                )

            prompt_id = await post_workflow(client, wf)
            logger.info("Video job %s → ComfyUI prompt %s", video_id, prompt_id)

            async with AsyncSessionLocal() as db:
                video = await db.get(Video, video_id)
                if video:
                    video.comfy_prompt_id = prompt_id
                    await db.commit()

            _set("queued", "Waiting in ComfyUI queue…", 25)
            outputs = await poll_history(
                client, prompt_id,
                timeout=POLL_TIMEOUT, interval=POLL_INTERVAL,
            )

        _set("finalizing", "Generation complete — saving video…", 92)

        save_out = outputs.get(active_save_node, {})
        gifs = save_out.get("gifs") or save_out.get("videos") or []
        if not gifs:
            raise RuntimeError(f"VHS_VideoCombine output missing from history: {save_out}")

        entry = gifs[0]
        comfy_vid_filename = entry["filename"]
        comfy_vid_subfolder = entry.get("subfolder", "")
        comfy_src = settings.comfyui_output_dir / comfy_vid_subfolder / comfy_vid_filename

        if not comfy_src.exists():
            raise FileNotFoundError(f"Video not found at {comfy_src}")

        settings.videos_dir.mkdir(parents=True, exist_ok=True)
        dest_name = f"{video_id}_{comfy_vid_filename}"
        dest = settings.videos_dir / dest_name
        await asyncio.to_thread(shutil.copy2, comfy_src, dest)
        rel_path = dest.relative_to(settings.storage_dir)
        logger.info("Video stored: %s", dest)

        # Generate first-frame thumbnail via ffmpeg
        _set("finalizing", "Generating thumbnail…", 96)
        thumb_dest = settings.videos_dir / f"{video_id}_thumb.jpg"
        try:
            proc = await asyncio.create_subprocess_exec(
                settings.ffmpeg_path,
                "-y", "-i", str(dest),
                "-frames:v", "1", "-q:v", "3",
                str(thumb_dest),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.wait()
        except Exception as e:
            logger.warning("Thumbnail generation failed: %s", e)

        async with AsyncSessionLocal() as db:
            video = await db.get(Video, video_id)
            if video:
                video.filename = dest_name
                video.filepath = str(rel_path)
                video.status   = "done"
                await db.commit()

        _progress.pop(vid_key, None)

    except Exception as exc:
        logger.error("Video generation %s failed: %s", video_id, exc)
        _progress.pop(vid_key, None)
        async with AsyncSessionLocal() as db:
            video = await db.get(Video, video_id)
            if video:
                video.status = "failed"
                video.error  = str(exc)[:1000]
                await db.commit()


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/generate", status_code=202)
async def generate_video(body: GenerateVideoRequest, db: AsyncSession = Depends(get_db)):
    n = len(body.image_ids)
    if body.workflow == "flf2v":
        if not (2 <= n <= 3):
            raise HTTPException(status_code=400, detail="flf2v requires 2 or 3 image IDs")
        if body.frame_count < 5 or body.frame_count > 81:
            raise HTTPException(status_code=400, detail="frame_count must be 5–81")
    else:
        if not (1 <= n <= 3):
            raise HTTPException(status_code=400, detail="Provide 1, 2, or 3 image IDs")
        for fc in body.frame_counts:
            if not (5 <= fc <= 81):
                raise HTTPException(status_code=400, detail="frame_counts values must be 5–81")

    # Summarise per-image prompts for display
    if body.workflow == "i2v_multi" and body.prompts:
        prompt_display = " | ".join(p for p in body.prompts if p) or body.prompt or None
    else:
        prompt_display = body.prompt or None

    video = Video(
        id=uuid.uuid4(),
        image_ids=body.image_ids,
        workflow=body.workflow,
        prompt=prompt_display,
        width=body.width,
        height=body.height,
        frame_count=body.frame_count,
        n_images=n,
        fps=body.fps,
        status="generating",
        created_at=datetime.now(timezone.utc),
    )
    db.add(video)
    await db.commit()
    await db.refresh(video)

    asyncio.create_task(_run_generation(video.id, body))
    logger.info("Queued video generation job %s (%s, %d images)", video.id, body.workflow, n)

    return {"video_id": str(video.id), "status": "generating"}


@router.get("/jobs/{video_id}/progress")
async def get_job_progress(video_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """Lightweight progress endpoint — reads module-level dict + optional ComfyUI queue check."""
    video = await db.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video job not found")

    if video.status == "done":
        return {"phase": "done", "message": "Complete", "pct": 100}
    if video.status == "failed":
        return {"phase": "failed", "message": video.error or "Generation failed", "pct": 0}

    prog = dict(_progress.get(str(video_id), {"phase": "processing", "message": "Processing…", "pct": 30}))

    # Enrich with live ComfyUI queue info when the prompt is submitted
    if video.comfy_prompt_id and prog["phase"] in ("queued", "submitting", "processing"):
        qi = await queue_info(video.comfy_prompt_id)
        if qi["status"] == "running":
            prog["phase"]   = "running"
            prog["message"] = "ComfyUI: generating frames…"
            prog["pct"]     = max(prog["pct"], 30)
        elif qi["status"] == "pending":
            pos = qi.get("position", "?")
            prog["phase"]   = "queued"
            prog["message"] = f"Queued in ComfyUI (position {pos})…"
            prog["pct"]     = 25
        prog["queue"] = qi

    return prog


@router.get("/jobs/{video_id}")
async def get_job(video_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    video = await db.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video job not found")
    return _serialize(video)


@router.get("/thumb/{video_id}")
async def video_thumbnail(video_id: uuid.UUID):
    p = settings.videos_dir / f"{video_id}_thumb.jpg"
    if p.exists():
        return FileResponse(p, media_type="image/jpeg")
    raise HTTPException(status_code=404, detail="Thumbnail not found")


@router.get("")
async def list_videos(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Video).order_by(desc(Video.created_at)))
    return [_serialize(v) for v in result.scalars().all()]


@router.delete("/{video_id}", status_code=204)
async def delete_video(video_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    video = await db.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if video.filepath:
        p = settings.storage_dir / video.filepath
        if p.exists():
            p.unlink(missing_ok=True)
    thumb = settings.videos_dir / f"{video_id}_thumb.jpg"
    thumb.unlink(missing_ok=True)
    await db.delete(video)
    await db.commit()


@router.get("/file/{filename}")
async def serve_video(filename: str):
    safe = Path(filename).name
    p = settings.videos_dir / safe
    if p.exists():
        return FileResponse(p, media_type="video/mp4")
    raise HTTPException(status_code=404, detail="Video not found")


# ── Serializer ────────────────────────────────────────────────────────────────

def _serialize(v: Video) -> dict:
    return {
        "id":          str(v.id),
        "status":      v.status,
        "workflow":    v.workflow or "flf2v",
        "filename":    v.filename,
        "url":         f"/api/video/file/{v.filename}" if v.filename else None,
        "thumb_url":   f"/api/video/thumb/{v.id}" if v.status == "done" else None,
        "image_ids":   [str(i) for i in v.image_ids] if v.image_ids else [],
        "prompt":      v.prompt,
        "width":       v.width,
        "height":      v.height,
        "frame_count": v.frame_count,
        "n_images":    v.n_images,
        "fps":         v.fps,
        "error":       v.error,
        "created_at":  v.created_at.isoformat(),
    }
