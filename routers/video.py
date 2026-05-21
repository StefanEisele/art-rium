"""
Key-frame video generation — two workflow types:

  i2v_multi  Each image is animated independently (WanImageToVideo), clips concatenated.
             Supports 1–6 images with per-image prompts and frame counts.

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
import json
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
from core.video_thumb import make_video_thumbnail
from services.comfy.client import (
    free_memory,
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
    rife_multiplier: int = 3       # i2v_multi: RIFE VFI frame interpolation factor (2/3/4)
    pingpong: bool = False         # i2v_multi: VHS_VideoCombine pingpong (boomerang) flag


class AssembleRequest(BaseModel):
    indices: list[int]             # segment indices to include, in playback order


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
    rife_multiplier: int,
) -> dict:
    """One WanImageToVideo segment — turbo path: 4-step LoRA, two-pass KSampler, RIFE ×N.

    Mirrors the `enable_turbo=true` branch of video_wan2_2_14B_i2v_reworked_API.json:
    LoRA-loaded UNETs, steps=4, cfg=1, split_step=2, shift=5. The ComfySwitchNode
    multiplexers from that file are dropped because turbo is hard-coded here.
    """
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
            "ckpt_name": _RIFE_CKPT, "clear_cache_after_n_frames": 10, "multiplier": rife_multiplier,
            "fast_mode": True, "ensemble": True, "scale_factor": 1,
            "dtype": "float32", "torch_compile": False, "batch_size": 1,
            "frames": [p+"decode", 0],
        }},
    }


def _build_i2v_single_workflow(
    comfy_filename: str,
    prompt: str,
    frame_count: int,
    width: int, height: int, fps: int,
    vid_prefix: str,
    rife_multiplier: int,
    pingpong: bool,
) -> tuple[dict, str]:
    """Single-image i2v segment with its own VHS save.

    One ComfyUI submission per segment keeps the VRAM peak independent of how
    many images the user picked: each prompt starts with a clean GPU state.
    Segments are stitched together server-side via ffmpeg concat.
    """
    wf: dict = {"img0": {"class_type": "LoadImage", "inputs": {"image": comfy_filename, "upload": "image"}}}
    wf.update(_i2v_segment(
        0, "img0", prompt, frame_count, width, height,
        random.randint(0, 2**32 - 1), rife_multiplier,
    ))

    save_id = "i2v_save"
    wf[save_id] = {"class_type": "VHS_VideoCombine", "inputs": {
        "frame_rate":      fps,
        "loop_count":      0,
        "filename_prefix": vid_prefix,
        "format":          "video/h265-mp4",
        "pix_fmt":         "yuv420p10le",
        "crf":             22,
        "save_metadata":   False,
        "pingpong":        pingpong,
        "save_output":     True,
        "images":          ["s0_rife", 0],
    }}
    return wf, save_id


# ── Segment storage helpers ───────────────────────────────────────────────────

def _segments_dir(video_id: uuid.UUID) -> Path:
    """Per-job directory holding individual segment MP4s, thumbnails and meta.json
    before the user picks which to assemble."""
    return settings.videos_dir / "segments" / str(video_id)


# ── Progress / failure helpers (shared by _run_generation + _assemble_video) ──

def _set_progress(vid_key: str, phase: str, message: str, pct: int) -> None:
    _progress[vid_key] = {"phase": phase, "message": message, "pct": pct}


async def _finalize_video_failure(video_id: uuid.UUID, exc: Exception, vid_key: str) -> None:
    """Common error path: clear progress, record exception on the Video row."""
    _progress.pop(vid_key, None)
    msg = str(exc).strip()
    err = f"{type(exc).__name__}: {msg}" if msg else type(exc).__name__
    async with AsyncSessionLocal() as db:
        video = await db.get(Video, video_id)
        if video:
            video.status = "failed"
            video.error  = err[:1000]
            await db.commit()


# ── Background generation task ────────────────────────────────────────────────

# httpx connection/read errors thrown when ComfyUI is mid-restart between
# segments (the VRAM-flush handoff occasionally crashes the server briefly).
_TRANSIENT_NET_ERRORS = (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError, ConnectionError)


async def _post_workflow_with_retry(
    client: httpx.AsyncClient, wf: dict, *, attempts: int = 3, base_delay: float = 4.0,
) -> str:
    """Submit a workflow, retrying briefly on transient network errors.

    Rationale: with per-segment submissions, ComfyUI sometimes momentarily
    refuses connections while it unloads/loads models between segments. Workflow
    validation errors (RuntimeError from post_workflow) are NOT retried.
    """
    for k in range(attempts):
        try:
            return await post_workflow(client, wf)
        except _TRANSIENT_NET_ERRORS as e:
            if k == attempts - 1:
                raise
            delay = base_delay * (k + 1)
            logger.warning(
                "post_workflow attempt %d/%d failed (%s: %s) — retry in %.1fs",
                k + 1, attempts, type(e).__name__, e, delay,
            )
            await asyncio.sleep(delay)
    raise RuntimeError("post_workflow retry loop exited without result")  # unreachable


async def _upload_images_to_comfy(
    client: httpx.AsyncClient,
    ordered_images: list[Image],
    prefix: str,
    vid_key: str,
) -> list[str]:
    """Upload each managed image to ComfyUI; return the assigned ComfyUI filenames."""
    comfy_names: list[str] = []
    n = len(ordered_images)
    for i, img in enumerate(ordered_images):
        src = settings.storage_dir / img.filepath
        assigned_name = await upload_image(client, src, f"{prefix}_kf{i + 1}.png")
        comfy_names.append(assigned_name)
        pct = 5 + int(12 * (i + 1) / n)
        _set_progress(vid_key, "uploading", f"Uploaded image {i+1}/{n}", pct)
        logger.info("Uploaded %s → ComfyUI:%s", src.name, assigned_name)
    return comfy_names


async def _save_comfy_prompt_id(video_id: uuid.UUID, prompt_id: str) -> None:
    async with AsyncSessionLocal() as db:
        video = await db.get(Video, video_id)
        if video:
            video.comfy_prompt_id = prompt_id
            await db.commit()


def _comfy_save_path(save_out: dict, segment_label: str) -> Path:
    """Resolve the ComfyUI-side output filename from a VHS_VideoCombine save node."""
    gifs = save_out.get("gifs") or save_out.get("videos") or []
    if not gifs:
        raise RuntimeError(f"{segment_label}: VHS_VideoCombine output missing: {save_out}")
    entry = gifs[0]
    comfy_src = settings.comfyui_output_dir / entry.get("subfolder", "") / entry["filename"]
    if not comfy_src.exists():
        raise FileNotFoundError(f"{segment_label} not found at {comfy_src}")
    return comfy_src


async def _run_flf2v(
    client: httpx.AsyncClient,
    video_id: uuid.UUID,
    comfy_names: list[str],
    req: GenerateVideoRequest,
    prefix: str,
    vid_key: str,
) -> Path:
    """Submit a single FLF2V workflow, poll, copy result into videos_dir. Returns the dest path."""
    if len(comfy_names) < 2:
        raise ValueError("FLF2V requires at least 2 images")

    _set_progress(vid_key, "submitting", "Submitting workflow to ComfyUI…", 20)
    wf = _build_flf2v_workflow(
        comfy_names, req.width, req.height, req.frame_count, req.fps, req.prompt, prefix,
    )
    prompt_id = await post_workflow(client, wf)
    logger.info("Video job %s → ComfyUI prompt %s", video_id, prompt_id)
    await _save_comfy_prompt_id(video_id, prompt_id)

    _set_progress(vid_key, "queued", "Waiting in ComfyUI queue…", 25)
    outputs = await poll_history(client, prompt_id, timeout=POLL_TIMEOUT, interval=POLL_INTERVAL)
    comfy_src = _comfy_save_path(outputs.get(SAVE_NODE_ID, {}), "VHS_VideoCombine output")

    dest = settings.videos_dir / f"{video_id}_{comfy_src.name}"
    await asyncio.to_thread(shutil.copy2, comfy_src, dest)
    return dest


async def _run_i2v_multi(
    client: httpx.AsyncClient,
    video_id: uuid.UUID,
    comfy_names: list[str],
    ordered_images: list[Image],
    req: GenerateVideoRequest,
    prefix: str,
    vid_key: str,
) -> Path | None:
    """Generate each image as an independent ComfyUI submission, persist segments
    as discrete files + meta.json. For n==1 returns the assembled mp4 path; for
    n>1 transitions the row to status=review and returns None (caller skips finalize)."""
    n_imgs = len(comfy_names)
    prompts_list = req.prompts if len(req.prompts) == n_imgs else [req.prompt] * n_imgs
    fc_list = req.frame_counts if len(req.frame_counts) == n_imgs else [req.frame_count] * n_imgs
    seg_dir = _segments_dir(video_id)
    seg_dir.mkdir(parents=True, exist_ok=True)
    segment_meta: list[dict] = []
    seg_band_lo, seg_band_hi = 20, 86
    seg_span = seg_band_hi - seg_band_lo

    for i, (fname, p_i, fc_i, img_obj) in enumerate(
        zip(comfy_names, prompts_list, fc_list, ordered_images)
    ):
        seg_prefix  = f"{prefix}_seg{i + 1}"
        pct_seg_lo  = seg_band_lo + int(seg_span * i           / n_imgs)
        pct_seg_mid = seg_band_lo + int(seg_span * (i + 0.3) / n_imgs)
        pct_seg_hi  = seg_band_lo + int(seg_span * (i + 1)   / n_imgs)

        _set_progress(vid_key, "submitting", f"Clip {i + 1}/{n_imgs} — submitting to ComfyUI…", pct_seg_lo)
        wf, save_node = _build_i2v_single_workflow(
            fname, p_i, fc_i, req.width, req.height, req.fps, seg_prefix,
            req.rife_multiplier, req.pingpong,
        )
        prompt_id = await _post_workflow_with_retry(client, wf)
        logger.info("Video job %s segment %d → ComfyUI prompt %s", video_id, i + 1, prompt_id)
        if i == 0:
            await _save_comfy_prompt_id(video_id, prompt_id)

        _set_progress(vid_key, "running", f"Clip {i + 1}/{n_imgs} — generating frames…", pct_seg_mid)
        seg_outputs = await poll_history(client, prompt_id, timeout=POLL_TIMEOUT, interval=POLL_INTERVAL)
        seg_src = _comfy_save_path(seg_outputs.get(save_node, {}), f"Segment {i + 1}")

        # Persist segment under a stable name so the review UI + assemble
        # endpoint can refer to it by index regardless of ComfyUI's output.
        seg_dest  = seg_dir / f"seg_{i}.mp4"
        seg_thumb = seg_dir / f"seg_{i}_thumb.jpg"
        await asyncio.to_thread(shutil.copy2, seg_src, seg_dest)
        await make_video_thumbnail(seg_dest, seg_thumb)
        segment_meta.append({
            "index":       i,
            "filename":    seg_dest.name,
            "thumb":       seg_thumb.name,
            "prompt":      p_i,
            "frame_count": fc_i,
            "image_id":    str(img_obj.id),
        })
        _set_progress(vid_key, "running", f"Clip {i + 1}/{n_imgs} — saved ✓", pct_seg_hi)

        # Before the next segment, force ComfyUI to fully unload models and free VRAM.
        # Without this, ComfyUI keeps Wan 14B fp8 partially evicted and hits an mmap
        # access violation in load_torch_file on the next prompt's partial-reload.
        if i < n_imgs - 1:
            _set_progress(vid_key, "running", f"Freeing GPU memory for clip {i + 2}/{n_imgs}…", pct_seg_hi)
            await free_memory(client)
            await asyncio.sleep(8)  # let CUDA actually release before the next cold load

    # Sidecar metadata — consumed by /jobs/{id}/segments and _assemble_video.
    meta = {
        "video_id":        str(video_id),
        "workflow":        "i2v_multi",
        "width":           req.width,
        "height":          req.height,
        "fps":             req.fps,
        "rife_multiplier": req.rife_multiplier,
        "pingpong":        req.pingpong,
        "segments":        segment_meta,
    }
    await asyncio.to_thread(
        (seg_dir / "meta.json").write_text, json.dumps(meta, indent=2), encoding="utf-8",
    )

    if n_imgs > 1:
        # Multi-segment: user picks which to merge — caller skips finalize.
        _set_progress(vid_key, "review", f"{n_imgs} clips ready — pick which to merge", 88)
        async with AsyncSessionLocal() as db:
            video = await db.get(Video, video_id)
            if video:
                video.status = "review"
                await db.commit()
        return None

    # n == 1: nothing to choose between, copy the single segment into place.
    dest = settings.videos_dir / f"{video_id}_artrium.mp4"
    await asyncio.to_thread(shutil.copy2, seg_dir / "seg_0.mp4", dest)
    return dest


async def _finalize_video_done(video_id: uuid.UUID, dest: Path, vid_key: str) -> None:
    """Common success path: thumbnail + persist filename/filepath + status='done'."""
    _set_progress(vid_key, "finalizing", "Saving video…", 94)
    rel_path = dest.relative_to(settings.storage_dir)
    logger.info("Video stored: %s", dest)

    _set_progress(vid_key, "finalizing", "Generating thumbnail…", 96)
    await make_video_thumbnail(dest, settings.videos_dir / f"{video_id}_thumb.jpg")

    async with AsyncSessionLocal() as db:
        video = await db.get(Video, video_id)
        if video:
            video.filename = dest.name
            video.filepath = str(rel_path)
            video.status   = "done"
            video.error    = None
            await db.commit()
    _progress.pop(vid_key, None)


async def _run_generation(video_id: uuid.UUID, req: GenerateVideoRequest) -> None:
    vid_key = str(video_id)
    prefix  = f"artrium_{video_id.hex[:10]}"

    try:
        _set_progress(vid_key, "uploading", "Uploading images to ComfyUI…", 5)

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
            comfy_names = await _upload_images_to_comfy(client, ordered_images, prefix, vid_key)
            settings.videos_dir.mkdir(parents=True, exist_ok=True)

            if req.workflow == "flf2v":
                dest = await _run_flf2v(client, video_id, comfy_names, req, prefix, vid_key)
            else:
                dest = await _run_i2v_multi(
                    client, video_id, comfy_names, ordered_images, req, prefix, vid_key,
                )
                if dest is None:  # multi-segment → review status, caller skips finalize
                    return

        await _finalize_video_done(video_id, dest, vid_key)

    except Exception as exc:
        logger.exception("Video generation %s failed", video_id)
        await _finalize_video_failure(video_id, exc, vid_key)


async def _assemble_video(video_id: uuid.UUID, indices: list[int]) -> None:
    """Concatenate selected segments into the final video and mark the job done.

    Runs as a background task after the user picks segments in the review UI.
    Reports progress through the same _progress dict as _run_generation so the
    existing polling endpoint keeps working without any client-side branching.
    """
    vid_key = str(video_id)

    try:
        seg_dir = _segments_dir(video_id)
        meta_path = seg_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Segment metadata missing for {video_id}")
        meta = json.loads(await asyncio.to_thread(meta_path.read_text, encoding="utf-8"))
        all_segments = meta.get("segments", [])
        seg_lookup = {s["index"]: s for s in all_segments}

        chosen_files: list[Path] = []
        for idx in indices:
            if idx not in seg_lookup:
                raise ValueError(f"Unknown segment index: {idx}")
            sf = seg_dir / seg_lookup[idx]["filename"]
            if not sf.exists():
                raise FileNotFoundError(f"Segment file missing: {sf}")
            chosen_files.append(sf)

        fps = int(meta.get("fps", 24))
        dest = settings.videos_dir / f"{video_id}_artrium.mp4"

        _set_progress(vid_key, "finalizing", f"Concatenating {len(chosen_files)} clip(s)…", 92)
        if len(chosen_files) == 1:
            await asyncio.to_thread(shutil.copy2, chosen_files[0], dest)
        else:
            # See _run_generation: concat *demuxer* with -c copy is unreliable for HEVC,
            # re-encode through the concat filter instead.
            cmd: list[str] = [settings.ffmpeg_path, "-y"]
            for sf in chosen_files:
                cmd += ["-i", str(sf)]
            n = len(chosen_files)
            filter_inputs = "".join(f"[{i}:v]" for i in range(n))
            cmd += [
                "-filter_complex", f"{filter_inputs}concat=n={n}:v=1:a=0[v]",
                "-map", "[v]",
                "-c:v",     "libx265",
                "-preset",  "medium",
                "-crf",     "22",
                "-pix_fmt", "yuv420p10le",
                "-tag:v",   "hvc1",
                "-r",       str(fps),
                str(dest),
            ]
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr_b = await proc.communicate()
            if proc.returncode != 0:
                tail = stderr_b.decode(errors="replace")[-1500:]
                raise RuntimeError(f"ffmpeg concat failed (rc={proc.returncode}): {tail}")

        await _finalize_video_done(video_id, dest, vid_key)
        logger.info("Video %s assembled from %d segment(s)", video_id, len(chosen_files))

    except Exception as exc:
        logger.exception("Video assembly %s failed", video_id)
        await _finalize_video_failure(video_id, exc, vid_key)


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
        if not (1 <= n <= 10):
            raise HTTPException(status_code=400, detail="i2v_multi requires 1–10 image IDs")
        for fc in body.frame_counts:
            if not (5 <= fc <= 81):
                raise HTTPException(status_code=400, detail="frame_counts values must be 5–81")
        if body.rife_multiplier not in (2, 3, 4):
            raise HTTPException(status_code=400, detail="rife_multiplier must be 2, 3 or 4")

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
    if video.status == "review":
        prog = _progress.get(str(video_id), {})
        return {
            "phase":   "review",
            "message": prog.get("message", "Clips ready — pick which to merge"),
            "pct":     prog.get("pct", 88),
        }

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


@router.get("/jobs/{video_id}/segments")
async def list_segments(video_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """Return per-segment metadata + URLs for the review UI. 404 if not yet generated."""
    video = await db.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    seg_dir = _segments_dir(video_id)
    meta_path = seg_dir / "meta.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail="No segments available for this video")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    segments = []
    for s in meta.get("segments", []):
        segments.append({
            "index":       s["index"],
            "filename":    s["filename"],
            "url":         f"/api/video/segments/{video_id}/{s['filename']}",
            "thumb_url":   f"/api/video/segments/{video_id}/{s['thumb']}",
            "prompt":      s.get("prompt", ""),
            "frame_count": s.get("frame_count"),
            "image_id":    s.get("image_id"),
        })
    return {
        "video_id":        str(video_id),
        "status":          video.status,
        "width":           meta.get("width"),
        "height":          meta.get("height"),
        "fps":             meta.get("fps"),
        "rife_multiplier": meta.get("rife_multiplier"),
        "pingpong":        meta.get("pingpong"),
        "segments":        segments,
    }


@router.post("/jobs/{video_id}/assemble", status_code=202)
async def assemble_video(
    video_id: uuid.UUID, body: AssembleRequest, db: AsyncSession = Depends(get_db),
):
    """Kick off concat for the chosen segment indices. Only valid in status=review."""
    video = await db.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if video.status != "review":
        raise HTTPException(
            status_code=400,
            detail=f"Video is in status '{video.status}' — assemble only valid from 'review'",
        )
    if not body.indices:
        raise HTTPException(status_code=400, detail="No segments selected")

    video.status = "assembling"
    await db.commit()

    asyncio.create_task(_assemble_video(video_id, body.indices))
    logger.info("Assembling video %s from segments %s", video_id, body.indices)
    return {"video_id": str(video_id), "status": "assembling", "n_selected": len(body.indices)}


@router.get("/segments/{video_id}/{filename}")
async def serve_segment(video_id: uuid.UUID, filename: str):
    """Serve a single segment MP4 or its thumbnail JPEG for the review UI."""
    safe = Path(filename).name
    p = _segments_dir(video_id) / safe
    if not p.exists():
        raise HTTPException(status_code=404, detail="Segment file not found")
    media = "image/jpeg" if safe.lower().endswith((".jpg", ".jpeg")) else "video/mp4"
    return FileResponse(p, media_type=media)


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
    seg_dir = _segments_dir(video_id)
    if seg_dir.exists():
        shutil.rmtree(seg_dir, ignore_errors=True)
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
        "id":                str(v.id),
        "status":            v.status,
        "workflow":          v.workflow or "flf2v",
        "filename":          v.filename,
        "url":               f"/api/video/file/{v.filename}" if v.filename else None,
        "thumb_url":         f"/api/video/thumb/{v.id}" if v.status == "done" else None,
        "image_ids":         [str(i) for i in v.image_ids] if v.image_ids else [],
        "prompt":            v.prompt,
        "width":             v.width,
        "height":            v.height,
        "frame_count":       v.frame_count,
        "n_images":          v.n_images,
        "fps":               v.fps,
        "error":             v.error,
        "youtube_video_id":  v.youtube_video_id,
        "youtube_url":       v.youtube_url,
        "youtube_uploaded":  bool(v.youtube_video_id),
        "created_at":        v.created_at.isoformat(),
    }
