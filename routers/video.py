"""
Key-frame video generation — three workflow types, all producing per-segment
CLIPS that land in a shared clip library (one "stack" per job):

  i2v_multi  Each image is animated independently (WanImageToVideo).
             Supports 1–10 images with per-image prompts and frame counts.
             Prompts can be auto-suggested via POST /api/video/suggest-i2v.

  ltx_i2v    Each image is animated independently via LTX-2.3 (native audio).
             Supports 1–6 images with per-image prompts and frame counts.

  flf2v      Each adjacent pair of key frames becomes its own independent
             transition clip (WanFirstLastFrameToVideo) — the same
             per-segment pattern as i2v_multi, but per PAIR. Supports
             2–20 images (1–19 transitions), each with its own prompt and
             frame count. Prompts can be auto-suggested in one VLM call via
             POST /api/video/suggest-transitions.

A job is "done" when all of its clips are rendered — there is no per-job
final file anymore. Final videos are created by merging clips (from any
number of jobs, in any order, mixed workflows allowed) via POST
/api/video/merge, which normalizes resolution/fps/audio and re-encodes into
a new Video row with workflow="merge". Sources can optionally be deleted
after a successful merge.

POST /api/video/generate            → enqueue job, return {video_id}
POST /api/video/suggest-transitions → VLM-suggested per-transition prompts (flf2v)
POST /api/video/suggest-i2v         → VLM-suggested surreal per-image prompts (i2v/ltx)
GET  /api/video/jobs/{id}           → poll status
GET  /api/video/jobs/{id}/progress  → lightweight progress (ComfyUI queue + phase)
GET  /api/video/clips               → all library clips (frontend groups by job)
DELETE /api/video/clips/{clip_id}   → delete one clip (empty source jobs are pruned)
POST /api/video/merge               → concat chosen clips (cross-job) into a new video
GET  /api/video/thumb/{id}          → first-frame JPEG thumbnail
GET  /api/video/file/{fname}        → serve MP4
GET  /api/videos                    → list all videos
DELETE /api/video/{id}              → delete a video/job (cascades to its clips)
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
from core.comfy import WORKFLOW_NAME as ZIMAGE_WORKFLOW_NAME
from core.config import settings
from core.db import AsyncSessionLocal, get_db
from core.imaging import prepare_jpg_for_web
from core.loras import ALLOWED_LORAS, DEFAULT_LORA, LORAS
from core.models import Image, Song, Video, VideoClip
from core.tasks import safe_create_task
from core.video_thumb import make_video_thumbnail
from services.comfy.client import (
    free_memory,
    poll_history,
    post_workflow,
    queue_info,
    upload_image,
)
from services.comfy.ingest import ingest_comfy_image
from services.comfy.zimage import ZIMAGE_SAVE_NODE, build_zimage_workflow
from services.ollama.analysis import (
    generate_i2v_motion_prompts,
    generate_transition_prompts,
)
from services.ollama.story_frames import (
    describe_image_for_story,
    generate_story_frame_prompts,
)
from services.ollama.zimage_enhance import get_zimage_style_block
from services.video.merge import MergeInput, merge_clips
from services.video.soundtrack import mux_soundtrack

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

# FLF2V sampler tuning — full-strength 4-step lightx2v distill on both
# experts, cfg=1 (the plain Lightning fast path). An 8-step asymmetric
# variant (weakened distill + cfg 3 on the high-noise expert, per
# https://huggingface.co/lightx2v/Wan2.2-Lightning/discussions/5) improved
# end-frame adherence but doubled render time per transition — too slow in
# practice, reverted 2026-07-09.
_FLF2V_STEPS             = 4     # total steps across both experts
_FLF2V_SPLIT_STEP        = 2     # high-noise expert covers steps 0..split
_FLF2V_LORA_HIGH_STRENGTH = 1.0  # distill LoRA at full strength on high-noise
_FLF2V_CFG_HIGH          = 1     # distilled guidance-free path on high-noise

# ── LTX-2.3 i2v (separate model family, native audio) ─────────────────────────
# Single checkpoint carries the diffusion model + pixel VAE + audio VAE. The
# distilled LoRA gives the few-step turbo path; the Gemma text encoder and the
# spatial upscaler are loaded separately. Combo strings verified against
# ComfyUI /object_info — the checkpoint lives in a `ltx\` subfolder.
_LTX_CKPT      = "ltx\\ltx-2.3-22b-dev-fp8.safetensors"
_LTX_LORA      = "ltx-2.3-22b-distilled-lora-384.safetensors"
_LTX_TEXT_ENC  = "gemma_3_12B_it_fp8_e4m3fn.safetensors"
_LTX_UPSCALER  = "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
_LTX_NEG       = "pc game, console game, video game, cartoon, childish, ugly"
# Two-stage sigma schedules baked into the source workflow: a longer low-res
# pass, then a short refinement pass on the 2×-upscaled latent.
_LTX_SIGMAS_LO = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
_LTX_SIGMAS_HI = "0.85, 0.7250, 0.4219, 0.0"

POLL_INTERVAL = 15    # seconds between ComfyUI history polls
POLL_TIMEOUT  = 1800  # 30 minutes max

# ── Pydantic ──────────────────────────────────────────────────────────────────

class GenerateVideoRequest(BaseModel):
    image_ids: list[uuid.UUID]
    workflow: str = "i2v_multi"    # "i2v_multi" | "ltx_i2v" | "flf2v"
    width:  int = 1088
    height: int = 1088
    frame_count: int = 49          # fallback frame count when prompts/frame_counts arrays are absent
    fps:    int = 24
    prompt: str = ""               # fallback prompt when `prompts` is absent/mismatched length
    prompts: list[str] = []        # i2v_multi/ltx_i2v: one per image; flf2v: one per transition (n-1)
    frame_counts: list[int] = []   # i2v_multi/ltx_i2v: one per image; flf2v: one per transition (n-1)
    rife_multiplier: int = 3       # i2v_multi/flf2v: RIFE VFI frame interpolation factor (2/3/4); unused by ltx_i2v
    pingpong: bool = False         # i2v_multi: VHS_VideoCombine pingpong (boomerang) flag; unused by flf2v
    end_on_keyframe: bool = False  # flf2v: append the raw end key frame after the diffused clip (pixel-exact landing, but reads as a cut when diffusion undershoots)


class MergeRequest(BaseModel):
    clip_ids: list[uuid.UUID]      # library clips to concatenate, in playback order (cross-job)
    delete_sources: bool = False   # delete the source clips (and empty source jobs) after a successful merge


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
        p+"lora_h": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": [p+"unet_h", 0], "lora_name": _LORA_HIGH, "strength_model": _FLF2V_LORA_HIGH_STRENGTH}},
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
            "steps": _FLF2V_STEPS, "cfg": _FLF2V_CFG_HIGH,
            "sampler_name": "euler", "scheduler": "simple",
            "start_at_step": 0, "end_at_step": _FLF2V_SPLIT_STEP,
            "return_with_leftover_noise": "enable",
            "positive":     [p+"flf2v", 0],
            "negative":     [p+"flf2v", 1],
            "latent_image": [p+"flf2v", 2],
        }},
        p+"ks_l":   {"class_type": "KSamplerAdvanced", "inputs": {
            "model":                     [p+"samp_l", 0],
            "add_noise":                 "disable",
            "noise_seed":                0,
            "steps": _FLF2V_STEPS, "cfg": 1,
            "sampler_name": "euler", "scheduler": "simple",
            "start_at_step": _FLF2V_SPLIT_STEP, "end_at_step": 10000,
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


def _build_flf2v_single_workflow(
    start_fname: str,
    end_fname: str,
    prompt: str,
    frame_count: int,
    width: int, height: int, fps: int,
    vid_prefix: str,
    rife_multiplier: int,
    append_end_frame: bool = False,
) -> tuple[dict, str]:
    """Single key-frame transition with its own VHS save.

    One ComfyUI submission per transition keeps the VRAM peak independent of
    how many key frames the user picked — mirrors _build_i2v_single_workflow.

    append_end_frame=True additionally appends the raw end key frame before
    RIFE so the clip lands pixel-exact on the chosen photo. Off by default:
    when the diffusion undershoots the end frame (distill-LoRA weakness), the
    appended photo IS the perceived hard cut — RIFE only puts multiplier-1
    interpolated frames (~1/12 s) between the last diffused frame and it.
    """
    wf: dict = {
        "img_start": {"class_type": "LoadImage", "inputs": {"image": start_fname, "upload": "image"}},
        "img_end":   {"class_type": "LoadImage", "inputs": {"image": end_fname,   "upload": "image"}},
    }
    nodes, decode_id = _transition_nodes(
        0, "img_start", "img_end", width, height, frame_count, prompt,
        random.randint(0, 2**32 - 1),
    )
    wf.update(nodes)

    rife_input = decode_id
    if append_end_frame:
        wf["batch_final"] = {"class_type": "ImageBatch", "inputs": {
            "image1": [decode_id, 0],
            "image2": ["img_end", 0],
        }}
        rife_input = "batch_final"

    wf["rife"] = {"class_type": "RIFE VFI", "inputs": {
        "ckpt_name":                  _RIFE_CKPT,
        "clear_cache_after_n_frames": 10,
        "multiplier":                 rife_multiplier,
        "fast_mode":                  True,
        "ensemble":                   True,
        "scale_factor":               1,
        "dtype":                      "float32",
        "torch_compile":              False,
        "batch_size":                 1,
        "frames":                     [rife_input, 0],
    }}

    save_id = "flf2v_save"
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
        "images":          ["rife", 0],
    }}

    return wf, save_id


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


# ── LTX-2.3 i2v workflow builder (native-audio, two-stage upscale) ────────────

def _build_ltx_single_workflow(
    comfy_filename: str,
    prompt: str,
    frame_count: int,
    width: int, height: int, fps: int,
    vid_prefix: str,
) -> tuple[dict, str]:
    """Single-image LTX-2.3 i2v segment producing an mp4 *with* generated audio.

    Faithful API-format translation of the `Image to Video (LTX-2.3)` subgraph:
      preprocess image → low-res AV sample → 2× latent upscale → high-res
      refine → tiled video decode + audio decode → VHS mux (h264 + audio).

    Like the Wan builder, one image == one ComfyUI submission so the VRAM peak
    is independent of how many images the batch holds. Returns (workflow,
    save_node_id) for the shared harvesting path.
    """
    # LTX latent compression is 32 spatially; the low-res pass runs at half the
    # final size and is upscaled 2×, so the full dims must be divisible by 64.
    width  = max(64, (width  // 64) * 64)
    height = max(64, (height // 64) * 64)
    half_w, half_h = width // 2, height // 2
    # Temporal compression is 8: valid lengths are k*8 + 1. Snap the requested
    # frame count to the nearest such value so EmptyLTXVLatentVideo validates.
    length = max(9, ((frame_count - 1 + 4) // 8) * 8 + 1)
    seed = random.randint(0, 2**32 - 1)

    p = "ltx_"
    wf: dict = {
        p+"load":    {"class_type": "LoadImage",               "inputs": {"image": comfy_filename, "upload": "image"}},
        # ── Model / encoders ──
        p+"ckpt":    {"class_type": "CheckpointLoaderSimple",  "inputs": {"ckpt_name": _LTX_CKPT}},
        p+"lora":    {"class_type": "LoraLoaderModelOnly",     "inputs": {"model": [p+"ckpt", 0], "lora_name": _LTX_LORA, "strength_model": 0.5}},
        p+"avae":    {"class_type": "LTXVAudioVAELoader",      "inputs": {"ckpt_name": _LTX_CKPT}},
        p+"te":      {"class_type": "LTXAVTextEncoderLoader",  "inputs": {"text_encoder": _LTX_TEXT_ENC, "ckpt_name": _LTX_CKPT, "device": "default"}},
        p+"upsmod":  {"class_type": "LatentUpscaleModelLoader","inputs": {"model_name": _LTX_UPSCALER}},
        # ── Conditioning ──
        p+"pos":     {"class_type": "CLIPTextEncode",  "inputs": {"clip": [p+"te", 0], "text": prompt}},
        p+"neg":     {"class_type": "CLIPTextEncode",  "inputs": {"clip": [p+"te", 0], "text": _LTX_NEG}},
        p+"cond":    {"class_type": "LTXVConditioning", "inputs": {"positive": [p+"pos", 0], "negative": [p+"neg", 0], "frame_rate": float(fps)}},
        # ── Image preprocess ──
        p+"scale":   {"class_type": "ImageScale",              "inputs": {"image": [p+"load", 0], "upscale_method": "lanczos", "width": width, "height": height, "crop": "center"}},
        p+"longer":  {"class_type": "ResizeImagesByLongerEdge","inputs": {"images": [p+"scale", 0], "longer_edge": 1536}},
        p+"pre":     {"class_type": "LTXVPreprocess",          "inputs": {"image": [p+"longer", 0], "img_compression": 18}},
        # ── Empty latents (low-res video + audio) ──
        p+"evid":    {"class_type": "EmptyLTXVLatentVideo", "inputs": {"width": half_w, "height": half_h, "length": length, "batch_size": 1}},
        p+"eaud":    {"class_type": "LTXVEmptyLatentAudio", "inputs": {"frames_number": length, "frame_rate": fps, "batch_size": 1, "audio_vae": [p+"avae", 0]}},
        # ── Low-res AV sampling pass ──
        p+"i2vlo":   {"class_type": "LTXVImgToVideoInplace", "inputs": {"vae": [p+"ckpt", 2], "image": [p+"pre", 0], "latent": [p+"evid", 0], "strength": 0.7, "bypass": False}},
        p+"catlo":   {"class_type": "LTXVConcatAVLatent",    "inputs": {"video_latent": [p+"i2vlo", 0], "audio_latent": [p+"eaud", 0]}},
        p+"noiselo": {"class_type": "RandomNoise",           "inputs": {"noise_seed": seed}},
        p+"samlo":   {"class_type": "KSamplerSelect",        "inputs": {"sampler_name": "euler_ancestral_cfg_pp"}},
        p+"siglo":   {"class_type": "ManualSigmas",          "inputs": {"sigmas": _LTX_SIGMAS_LO}},
        p+"gdlo":    {"class_type": "CFGGuider",             "inputs": {"model": [p+"lora", 0], "positive": [p+"cond", 0], "negative": [p+"cond", 1], "cfg": 1}},
        p+"kslo":    {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": [p+"noiselo", 0], "guider": [p+"gdlo", 0], "sampler": [p+"samlo", 0], "sigmas": [p+"siglo", 0], "latent_image": [p+"catlo", 0]}},
        p+"seplo":   {"class_type": "LTXVSeparateAVLatent",  "inputs": {"av_latent": [p+"kslo", 0]}},
        # ── 2× latent upscale + re-inject the image at full res ──
        p+"ups":     {"class_type": "LTXVLatentUpsampler",  "inputs": {"samples": [p+"seplo", 0], "upscale_model": [p+"upsmod", 0], "vae": [p+"ckpt", 2]}},
        p+"i2vhi":   {"class_type": "LTXVImgToVideoInplace","inputs": {"vae": [p+"ckpt", 2], "image": [p+"pre", 0], "latent": [p+"ups", 0], "strength": 1.0, "bypass": False}},
        p+"cathi":   {"class_type": "LTXVConcatAVLatent",   "inputs": {"video_latent": [p+"i2vhi", 0], "audio_latent": [p+"seplo", 1]}},
        # ── High-res refinement pass ──
        p+"noisehi": {"class_type": "RandomNoise",           "inputs": {"noise_seed": 42}},
        p+"samhi":   {"class_type": "KSamplerSelect",        "inputs": {"sampler_name": "euler_cfg_pp"}},
        p+"sighi":   {"class_type": "ManualSigmas",          "inputs": {"sigmas": _LTX_SIGMAS_HI}},
        p+"crop":    {"class_type": "LTXVCropGuides",        "inputs": {"positive": [p+"cond", 0], "negative": [p+"cond", 1], "latent": [p+"seplo", 0]}},
        p+"gdhi":    {"class_type": "CFGGuider",             "inputs": {"model": [p+"lora", 0], "positive": [p+"crop", 0], "negative": [p+"crop", 1], "cfg": 1}},
        p+"kshi":    {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": [p+"noisehi", 0], "guider": [p+"gdhi", 0], "sampler": [p+"samhi", 0], "sigmas": [p+"sighi", 0], "latent_image": [p+"cathi", 0]}},
        p+"sephi":   {"class_type": "LTXVSeparateAVLatent",  "inputs": {"av_latent": [p+"kshi", 0]}},
        # ── Decode video (tiled) + audio ──
        p+"vdec":    {"class_type": "VAEDecodeTiled",     "inputs": {"samples": [p+"sephi", 0], "vae": [p+"ckpt", 2], "tile_size": 768, "overlap": 64, "temporal_size": 4096, "temporal_overlap": 4}},
        p+"adec":    {"class_type": "LTXVAudioVAEDecode", "inputs": {"samples": [p+"sephi", 1], "audio_vae": [p+"avae", 0]}},
    }

    # VHS mux so the existing _comfy_save_path harvester finds the output, and
    # the segment carries LTX's generated audio (h264/yuv420p for broad
    # browser playback). Multi-clip assembly still concatenates video only.
    save_id = "ltx_save"
    wf[save_id] = {"class_type": "VHS_VideoCombine", "inputs": {
        "frame_rate":      fps,
        "loop_count":      0,
        "filename_prefix": vid_prefix,
        "format":          "video/h264-mp4",
        "pix_fmt":         "yuv420p",
        "crf":             19,
        "save_metadata":   False,
        "pingpong":        False,
        "save_output":     True,
        "images":          [p+"vdec", 0],
        "audio":           [p+"adec", 0],
    }}
    return wf, save_id


# ── Segment storage helpers ───────────────────────────────────────────────────

def _segments_dir(video_id: uuid.UUID) -> Path:
    """Per-job directory holding the job's clip MP4s + thumbnails (the files
    behind its VideoClip library rows)."""
    return settings.videos_dir / "segments" / str(video_id)


async def _persist_clip(
    video_id: uuid.UUID,
    idx: int,
    filename: str,
    thumb: str,
    prompt: str,
    frame_count: int,
    req: GenerateVideoRequest,
) -> None:
    """Insert one VideoClip library row for a freshly rendered segment, so the
    clip is browsable/mergeable the moment it exists (even if a later segment
    of the same job fails)."""
    async with AsyncSessionLocal() as db:
        db.add(VideoClip(
            video_id=video_id,
            idx=idx,
            filename=filename,
            thumb=thumb,
            prompt=prompt or None,
            frame_count=frame_count,
            workflow=req.workflow,
            width=req.width,
            height=req.height,
            fps=req.fps,
            has_audio=(req.workflow == "ltx_i2v"),
        ))
        await db.commit()


async def _finalize_clip_job(video_id: uuid.UUID, vid_key: str) -> None:
    """Generation jobs deliver clips, not a final file — mark the job done."""
    async with AsyncSessionLocal() as db:
        video = await db.get(Video, video_id)
        if video:
            video.status = "done"
            video.error = None
            await db.commit()
    _progress.pop(vid_key, None)


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


async def _run_flf2v_multi(
    client: httpx.AsyncClient,
    video_id: uuid.UUID,
    comfy_names: list[str],
    ordered_images: list[Image],
    req: GenerateVideoRequest,
    prefix: str,
    vid_key: str,
) -> None:
    """Generate each key-frame transition as an independent ComfyUI submission,
    persisting every transition as a VideoClip library row (the job's stack).
    Mirrors _run_i2v_multi exactly, but over adjacent image PAIRS (n_imgs-1
    transitions) instead of single images."""
    n_imgs = len(comfy_names)
    if n_imgs < 2:
        raise ValueError("FLF2V requires at least 2 images")
    n_trans = n_imgs - 1

    prompts_list = req.prompts if len(req.prompts) == n_trans else [req.prompt] * n_trans
    fc_list = req.frame_counts if len(req.frame_counts) == n_trans else [req.frame_count] * n_trans
    seg_dir = _segments_dir(video_id)
    seg_dir.mkdir(parents=True, exist_ok=True)
    seg_band_lo, seg_band_hi = 20, 86
    seg_span = seg_band_hi - seg_band_lo

    for i in range(n_trans):
        start_fname, end_fname = comfy_names[i], comfy_names[i + 1]
        p_i, fc_i = prompts_list[i], fc_list[i]
        seg_prefix  = f"{prefix}_seg{i + 1}"
        pct_seg_lo  = seg_band_lo + int(seg_span * i           / n_trans)
        pct_seg_mid = seg_band_lo + int(seg_span * (i + 0.3) / n_trans)
        pct_seg_hi  = seg_band_lo + int(seg_span * (i + 1)   / n_trans)

        _set_progress(vid_key, "submitting", f"Transition {i + 1}/{n_trans} — submitting to ComfyUI…", pct_seg_lo)
        wf, save_node = _build_flf2v_single_workflow(
            start_fname, end_fname, p_i, fc_i, req.width, req.height, req.fps, seg_prefix,
            req.rife_multiplier, append_end_frame=req.end_on_keyframe,
        )
        prompt_id = await _post_workflow_with_retry(client, wf)
        logger.info("Video job %s transition %d → ComfyUI prompt %s", video_id, i + 1, prompt_id)
        if i == 0:
            await _save_comfy_prompt_id(video_id, prompt_id)

        _set_progress(vid_key, "running", f"Transition {i + 1}/{n_trans} — generating frames…", pct_seg_mid)
        seg_outputs = await poll_history(client, prompt_id, timeout=POLL_TIMEOUT, interval=POLL_INTERVAL)
        seg_src = _comfy_save_path(seg_outputs.get(save_node, {}), f"Transition {i + 1}")

        # Persist the clip under a stable name and register it in the library
        # immediately — a crash on a later transition loses nothing.
        seg_dest  = seg_dir / f"seg_{i}.mp4"
        seg_thumb = seg_dir / f"seg_{i}_thumb.jpg"
        await asyncio.to_thread(shutil.copy2, seg_src, seg_dest)
        await make_video_thumbnail(seg_dest, seg_thumb)
        await _persist_clip(video_id, i, seg_dest.name, seg_thumb.name, p_i, fc_i, req)
        _set_progress(vid_key, "running", f"Transition {i + 1}/{n_trans} — saved ✓", pct_seg_hi)

        # Before the next transition, force ComfyUI to fully unload models and
        # free VRAM — same mmap-crash mitigation _run_i2v_multi requires.
        if i < n_trans - 1:
            _set_progress(vid_key, "running", f"Freeing GPU memory for transition {i + 2}/{n_trans}…", pct_seg_hi)
            await free_memory(client)
            await asyncio.sleep(8)  # let CUDA actually release before the next cold load

    await _finalize_clip_job(video_id, vid_key)


async def _run_i2v_multi(
    client: httpx.AsyncClient,
    video_id: uuid.UUID,
    comfy_names: list[str],
    ordered_images: list[Image],
    req: GenerateVideoRequest,
    prefix: str,
    vid_key: str,
) -> None:
    """Generate each image as an independent ComfyUI submission, persisting
    every clip as a VideoClip library row (the job's stack)."""
    n_imgs = len(comfy_names)
    prompts_list = req.prompts if len(req.prompts) == n_imgs else [req.prompt] * n_imgs
    fc_list = req.frame_counts if len(req.frame_counts) == n_imgs else [req.frame_count] * n_imgs
    seg_dir = _segments_dir(video_id)
    seg_dir.mkdir(parents=True, exist_ok=True)
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
        if req.workflow == "ltx_i2v":
            wf, save_node = _build_ltx_single_workflow(
                fname, p_i, fc_i, req.width, req.height, req.fps, seg_prefix,
            )
        else:
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

        # Persist the clip under a stable name and register it in the library
        # immediately — a crash on a later segment loses nothing.
        seg_dest  = seg_dir / f"seg_{i}.mp4"
        seg_thumb = seg_dir / f"seg_{i}_thumb.jpg"
        await asyncio.to_thread(shutil.copy2, seg_src, seg_dest)
        await make_video_thumbnail(seg_dest, seg_thumb)
        await _persist_clip(video_id, i, seg_dest.name, seg_thumb.name, p_i, fc_i, req)
        _set_progress(vid_key, "running", f"Clip {i + 1}/{n_imgs} — saved ✓", pct_seg_hi)

        # Before the next segment, force ComfyUI to fully unload models and free VRAM.
        # Without this, ComfyUI keeps Wan 14B fp8 partially evicted and hits an mmap
        # access violation in load_torch_file on the next prompt's partial-reload.
        if i < n_imgs - 1:
            _set_progress(vid_key, "running", f"Freeing GPU memory for clip {i + 2}/{n_imgs}…", pct_seg_hi)
            await free_memory(client)
            await asyncio.sleep(8)  # let CUDA actually release before the next cold load

    await _finalize_clip_job(video_id, vid_key)


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

            # Both runners persist their clips as they render and mark the
            # job done themselves — the clips ARE the deliverable.
            if req.workflow == "flf2v":
                await _run_flf2v_multi(
                    client, video_id, comfy_names, ordered_images, req, prefix, vid_key,
                )
            else:
                await _run_i2v_multi(
                    client, video_id, comfy_names, ordered_images, req, prefix, vid_key,
                )

    except Exception as exc:
        logger.exception("Video generation %s failed", video_id)
        await _finalize_video_failure(video_id, exc, vid_key)


async def _prune_empty_clip_job(db: AsyncSession, job_id: uuid.UUID) -> None:
    """Delete a generation-job Video row once its last clip is gone.

    Only pure clip jobs are pruned (no final file on the row); anything else —
    merge results, legacy assembled videos — is left alone. A FK RESTRICT
    (e.g. an ImprovSession referencing the row) keeps the row instead of
    failing the caller."""
    remaining = await db.execute(
        select(VideoClip.id).where(VideoClip.video_id == job_id).limit(1)
    )
    if remaining.first() is not None:
        return
    job = await db.get(Video, job_id)
    if not job or job.filename:
        return
    try:
        await db.delete(job)
        await db.commit()
    except Exception:
        await db.rollback()
        logger.warning("Could not prune empty clip job %s (still referenced?)", job_id)
        return
    shutil.rmtree(_segments_dir(job_id), ignore_errors=True)


async def _delete_clips(clip_ids: list[uuid.UUID]) -> None:
    """Delete clip files + rows, then prune source jobs that end up empty."""
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(VideoClip).where(VideoClip.id.in_(clip_ids)))
        clips = list(result.scalars().all())
        job_ids = {c.video_id for c in clips}
        for c in clips:
            seg_dir = _segments_dir(c.video_id)
            (seg_dir / c.filename).unlink(missing_ok=True)
            (seg_dir / c.thumb).unlink(missing_ok=True)
            await db.delete(c)
        await db.commit()
        for jid in job_ids:
            await _prune_empty_clip_job(db, jid)


async def _run_merge(
    video_id: uuid.UUID, clip_ids: list[uuid.UUID], delete_sources: bool,
) -> None:
    """Concatenate the chosen library clips into the merge Video's final file.

    Runs as a background task. Clips may come from different jobs/workflows;
    services.video.merge normalizes resolution/fps/audio in one ffmpeg pass.
    Reports progress through the same _progress dict as _run_generation so the
    existing polling endpoint keeps working without any client-side branching.
    """
    vid_key = str(video_id)

    try:
        async with AsyncSessionLocal() as db:
            result = await db.execute(select(VideoClip).where(VideoClip.id.in_(clip_ids)))
            by_id = {c.id: c for c in result.scalars().all()}
        clips = [by_id[cid] for cid in clip_ids if cid in by_id]
        if len(clips) != len(clip_ids):
            raise ValueError("One or more selected clips no longer exist")

        inputs: list[MergeInput] = []
        for c in clips:
            f = _segments_dir(c.video_id) / c.filename
            if not f.exists():
                raise FileNotFoundError(f"Clip file missing on disk: {f}")
            inputs.append(MergeInput(path=f, has_audio=c.has_audio))

        # Normalization target: the first selected clip decides.
        width  = clips[0].width  or 960
        height = clips[0].height or 960
        fps    = clips[0].fps    or 24
        dest = settings.videos_dir / f"{video_id}_artrium.mp4"

        _set_progress(vid_key, "finalizing", f"Merging {len(clips)} clip(s)…", 40)
        if len(clips) == 1:
            await asyncio.to_thread(shutil.copy2, inputs[0].path, dest)
        else:
            await merge_clips(
                inputs, dest, width, height, fps, ffmpeg_path=settings.ffmpeg_path,
            )

        await _finalize_video_done(video_id, dest, vid_key)
        logger.info("Video %s merged from %d clip(s)", video_id, len(clips))

    except Exception as exc:
        logger.exception("Video merge %s failed", video_id)
        await _finalize_video_failure(video_id, exc, vid_key)
        return

    if delete_sources:
        # The merge itself succeeded — a cleanup hiccup must not flip the
        # finished video back to failed, so this runs outside the main try.
        try:
            await _delete_clips(clip_ids)
            logger.info("Merge %s: deleted %d source clip(s)", video_id, len(clip_ids))
        except Exception:
            logger.exception("Merge %s: source-clip cleanup failed (video is fine)", video_id)


# ── Endpoints ─────────────────────────────────────────────────────────────────

_TRANSITION_MAX_EDGE = 512
_TRANSITION_JPG_QUALITY = 80
_TRANSITION_TIMEOUT_FLOOR = 180.0
_TRANSITION_TIMEOUT_PER_IMAGE = 20.0


class SuggestTransitionsRequest(BaseModel):
    image_ids: list[uuid.UUID]   # in the user's selected playback order
    context: str = ""            # optional story/narrative context (story-frames flow)


async def _load_suggest_jpgs(
    image_ids: list[uuid.UUID], db: AsyncSession,
) -> list[bytes]:
    """Resolve image IDs (order-preserving) and downscale each to a VLM-sized
    JPEG. Shared by both prompt-suggestion endpoints."""
    img_result = await db.execute(select(Image).where(Image.id.in_(image_ids)))
    images_by_id = {img.id: img for img in img_result.scalars().all()}
    ordered = [images_by_id[iid] for iid in image_ids if iid in images_by_id]
    if len(ordered) != len(image_ids):
        raise HTTPException(status_code=404, detail="One or more images not found")

    jpgs: list[bytes] = []
    for img in ordered:
        src = settings.storage_dir / img.filepath
        if not src.exists():
            raise HTTPException(status_code=404, detail=f"Image file not found on disk: {img.id}")
        jpg_bytes, _ = await prepare_jpg_for_web(
            src, max_edge=_TRANSITION_MAX_EDGE, quality=_TRANSITION_JPG_QUALITY,
        )
        jpgs.append(jpg_bytes)
    return jpgs


@router.post("/suggest-transitions")
async def suggest_transitions(
    body: SuggestTransitionsRequest, db: AsyncSession = Depends(get_db),
):
    """VLM-suggested per-transition prompts for flf2v — one vision call, N-1
    prompts back. Purely advisory: nothing is persisted here; the client
    fills its own per-transition textareas and the user can edit before
    calling /generate."""
    n = len(body.image_ids)
    if n < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 images to suggest transitions")
    if n > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 images")

    jpgs = await _load_suggest_jpgs(body.image_ids, db)
    logger.info(
        "Suggest-transitions: %d images, model=%s, payload=%dKB",
        n, settings.ollama_titler_model, sum(len(j) for j in jpgs) // 1024,
    )

    timeout = max(_TRANSITION_TIMEOUT_FLOOR, _TRANSITION_TIMEOUT_PER_IMAGE * n)
    try:
        prompts = await generate_transition_prompts(jpgs, context=body.context, timeout=timeout)
    except Exception as exc:
        logger.exception("Transition prompt suggestion failed for %d images", n)
        raise HTTPException(status_code=502, detail=f"Suggestion failed: {exc}")

    return {"prompts": prompts}


@router.post("/suggest-i2v")
async def suggest_i2v(
    body: SuggestTransitionsRequest, db: AsyncSession = Depends(get_db),
):
    """VLM-suggested per-image surreal animation prompts for i2v_multi/ltx_i2v
    — one vision call, N prompts back (one per image). Advisory only, exactly
    like /suggest-transitions."""
    n = len(body.image_ids)
    if n < 1:
        raise HTTPException(status_code=400, detail="Need at least 1 image to suggest prompts")
    if n > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 images")

    jpgs = await _load_suggest_jpgs(body.image_ids, db)
    logger.info(
        "Suggest-i2v: %d images, model=%s, payload=%dKB",
        n, settings.ollama_titler_model, sum(len(j) for j in jpgs) // 1024,
    )

    timeout = max(_TRANSITION_TIMEOUT_FLOOR, _TRANSITION_TIMEOUT_PER_IMAGE * n)
    try:
        prompts = await generate_i2v_motion_prompts(jpgs, context=body.context, timeout=timeout)
    except Exception as exc:
        logger.exception("i2v prompt suggestion failed for %d images", n)
        raise HTTPException(status_code=502, detail=f"Suggestion failed: {exc}")

    return {"prompts": prompts}


# ── Story key-frames (flf2v story mode) ───────────────────────────────────────
# One source image + a short story → N generated z-Image key frames that stay
# visually consistent with the source. The result is a plain list of Image
# rows; the client feeds [source, *frames] into the normal flf2v pipeline.

_STORY_MAX_FRAMES = 19          # source + N must stay within flf2v's 20-image cap
_STORY_JOBS_KEEP = 20           # in-memory job entries retained (newest first)
_STORY_POLL_INTERVAL = 2        # z-image turbo renders in seconds, poll tightly
_STORY_POLL_TIMEOUT = 900       # generous: first frame may pay a cold model load

# job_id → {status, message, pct, story, source_image_id, prompts, frames, error}
# In-memory like _progress: the generated frames themselves are ingested as
# Image rows the moment they exist, so a server restart only loses the job
# bookkeeping, never the images.
_story_jobs: dict[str, dict] = {}


class StoryFramesRequest(BaseModel):
    image_id: uuid.UUID            # source key frame
    story: str                     # short narrative to advance across the frames
    n_frames: int = 4              # additional frames to generate (1–19)
    beat_seconds: int = 10         # story time between consecutive frames (1–600)
    style: str | None = None       # z-Image enhancer style letter (A–D); None = derive from source
    width: int = 1024
    height: int = 1024
    # None = inherit the source image's own LoRA/strength (high consistency
    # across the story sequence); an explicit value still overrides it.
    lora_name: str | None = None
    lora_strength: float | None = None


def _prune_story_jobs() -> None:
    if len(_story_jobs) <= _STORY_JOBS_KEEP:
        return
    for job_id, _ in sorted(
        _story_jobs.items(), key=lambda kv: kv[1].get("created_at", "")
    )[: len(_story_jobs) - _STORY_JOBS_KEEP]:
        _story_jobs.pop(job_id, None)


def _story_update(job_id: str, status: str, message: str, pct: int) -> None:
    job = _story_jobs.get(job_id)
    if job is not None:
        job.update(status=status, message=message, pct=pct)


async def _run_story_frames(
    job_id: str,
    req: StoryFramesRequest,
    src_filepath: str,
    src_prompt: str | None,
    src_seed: int | None,
) -> None:
    """Background task: describe source → plan N frame prompts → generate each
    frame via z-Image Turbo and ingest it as a managed Image row."""
    n = req.n_frames
    try:
        _story_update(job_id, "describing", "Analyzing source image…", 4)
        jpg_bytes, _ = await prepare_jpg_for_web(
            settings.storage_dir / src_filepath,
            max_edge=_TRANSITION_MAX_EDGE, quality=_TRANSITION_JPG_QUALITY,
        )
        description = await describe_image_for_story(jpg_bytes)

        _story_update(job_id, "planning", f"Writing {n} frame prompt(s)…", 12)
        trigger = next(
            (lora["trigger"] for lora in LORAS if lora["filename"] == req.lora_name), None,
        )
        prompts = await generate_story_frame_prompts(
            story=req.story, n=n,
            description=description,
            source_prompt=src_prompt,
            trigger=trigger,
            beat_seconds=req.beat_seconds,
            style_block=get_zimage_style_block(req.style) if req.style else None,
        )
        _story_jobs[job_id]["prompts"] = prompts

        # One shared seed for all frames — reusing the source image's seed
        # (when known) keeps the initial noise identical across the sequence,
        # which pulls compositions toward the source. Prompts carry the story.
        seed = src_seed if src_seed is not None and src_seed >= 0 else random.randint(0, 2**32 - 1)
        batch_id = uuid.uuid4()

        async with httpx.AsyncClient(timeout=60) as client:
            for i, frame_prompt in enumerate(prompts):
                _story_update(
                    job_id, "generating",
                    f"Frame {i + 1}/{n} — generating…",
                    18 + int(78 * i / n),
                )
                wf = build_zimage_workflow(
                    frame_prompt, seed, req.width, req.height,
                    req.lora_name, req.lora_strength,
                )
                prompt_id = await _post_workflow_with_retry(client, wf)
                outputs = await poll_history(
                    client, prompt_id,
                    timeout=_STORY_POLL_TIMEOUT, interval=_STORY_POLL_INTERVAL,
                )
                images = (outputs.get(ZIMAGE_SAVE_NODE) or {}).get("images") or []
                if not images:
                    raise RuntimeError(f"Frame {i + 1}: SaveImage output missing")
                entry = images[0]
                rel = entry["filename"]
                if entry.get("subfolder"):
                    rel = f"{entry['subfolder']}/{entry['filename']}"

                dest, image_id = await ingest_comfy_image(
                    rel,
                    prompt=frame_prompt,
                    seed=seed,
                    width=req.width,
                    height=req.height,
                    lora_name=req.lora_name,
                    lora_strength=req.lora_strength,
                    workflow_name=ZIMAGE_WORKFLOW_NAME,
                    batch_id=batch_id,
                )
                if not dest or not image_id:
                    raise RuntimeError(f"Frame {i + 1}: ingest failed (see server log)")

                _story_jobs[job_id]["frames"].append({
                    "id":        image_id,
                    "filename":  dest.name,
                    "url":       f"/api/image/{dest.name}",
                    "thumb_url": f"/api/image/{dest.name}/thumb",
                    "prompt":    frame_prompt,
                })
                logger.info("Story job %s: frame %d/%d ingested as %s", job_id, i + 1, n, image_id)

            # Leave ComfyUI clean for the Wan 14B load that typically follows
            # (same partial-eviction mmap-crash mitigation as the segment loop).
            await free_memory(client)

        _story_update(job_id, "done", f"{n} story frame(s) ready", 100)

    except Exception as exc:
        logger.exception("Story frames job %s failed", job_id)
        msg = str(exc).strip()
        job = _story_jobs.get(job_id)
        if job is not None:
            job.update(
                status="failed",
                message="Generation failed",
                error=(f"{type(exc).__name__}: {msg}" if msg else type(exc).__name__)[:1000],
            )


@router.post("/story-frames", status_code=202)
async def create_story_frames(body: StoryFramesRequest, db: AsyncSession = Depends(get_db)):
    """Kick off story key-frame generation. Returns {job_id}; poll
    GET /api/video/story-frames/{job_id} until status is done/failed."""
    if not body.story.strip():
        raise HTTPException(status_code=400, detail="story is required")
    if not (1 <= body.n_frames <= _STORY_MAX_FRAMES):
        raise HTTPException(
            status_code=400,
            detail=f"n_frames must be 1–{_STORY_MAX_FRAMES} (source + frames ≤ 20 key frames)",
        )
    if not (1 <= body.beat_seconds <= 600):
        raise HTTPException(status_code=400, detail="beat_seconds must be 1–600")
    if body.style and get_zimage_style_block(body.style) is None:
        raise HTTPException(status_code=400, detail=f"Unknown style: {body.style}")

    image = await db.get(Image, body.image_id)
    if not image:
        raise HTTPException(status_code=404, detail="Source image not found")
    if not (settings.storage_dir / image.filepath).exists():
        raise HTTPException(status_code=404, detail="Source image file missing on disk")

    # Default LoRA/strength to the source image's own generation params —
    # keeps the story sequence visually consistent unless explicitly overridden.
    if body.lora_name is None:
        body.lora_name = image.lora_name or DEFAULT_LORA
    if body.lora_strength is None:
        body.lora_strength = float(image.lora_strength) if image.lora_strength is not None else 0.5
    if body.lora_name not in ALLOWED_LORAS:
        raise HTTPException(status_code=400, detail=f"Unknown LoRA: {body.lora_name}")

    job_id = str(uuid.uuid4())
    _story_jobs[job_id] = {
        "job_id":          job_id,
        "status":          "describing",
        "message":         "Queued…",
        "pct":             1,
        "story":           body.story.strip(),
        "source_image_id": str(body.image_id),
        "n_frames":        body.n_frames,
        "prompts":         [],
        "frames":          [],
        "error":           None,
        "created_at":      datetime.now(timezone.utc).isoformat(),
    }
    _prune_story_jobs()

    safe_create_task(
        _run_story_frames(job_id, body, image.filepath, image.prompt, image.seed),
        name=f"story_frames:{job_id}",
    )
    logger.info(
        "Queued story-frames job %s (source=%s, n=%d)", job_id, body.image_id, body.n_frames,
    )
    return {"job_id": job_id, "status": "describing"}


@router.get("/story-frames/{job_id}")
async def get_story_frames_job(job_id: str):
    job = _story_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Story job not found (server restarted?)")
    return job


@router.post("/generate", status_code=202)
async def generate_video(body: GenerateVideoRequest, db: AsyncSession = Depends(get_db)):
    n = len(body.image_ids)
    if body.workflow == "flf2v":
        if not (2 <= n <= 20):
            raise HTTPException(status_code=400, detail="flf2v requires 2–20 image IDs")
        if body.frame_count < 5 or body.frame_count > 81:
            raise HTTPException(status_code=400, detail="frame_count must be 5–81")
        for fc in body.frame_counts:
            if not (5 <= fc <= 81):
                raise HTTPException(status_code=400, detail="frame_counts values must be 5–81")
        if body.rife_multiplier not in (2, 3, 4):
            raise HTTPException(status_code=400, detail="rife_multiplier must be 2, 3 or 4")
    elif body.workflow == "ltx_i2v":
        if not (1 <= n <= 6):
            raise HTTPException(status_code=400, detail="ltx_i2v requires 1–6 image IDs")
        for fc in body.frame_counts:
            if not (5 <= fc <= 81):
                raise HTTPException(status_code=400, detail="frame_counts values must be 5–81")
    else:
        if not (1 <= n <= 10):
            raise HTTPException(status_code=400, detail="i2v_multi requires 1–10 image IDs")
        for fc in body.frame_counts:
            if not (5 <= fc <= 81):
                raise HTTPException(status_code=400, detail="frame_counts values must be 5–81")
        if body.rife_multiplier not in (2, 3, 4):
            raise HTTPException(status_code=400, detail="rife_multiplier must be 2, 3 or 4")

    # Summarise per-clip prompts for display
    if body.workflow in ("i2v_multi", "ltx_i2v", "flf2v") and body.prompts:
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

    safe_create_task(_run_generation(video.id, body), name=f"video_generation:{video.id}")
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


def _serialize_clip(c: VideoClip) -> dict:
    return {
        "id":          str(c.id),
        "video_id":    str(c.video_id),
        "idx":         c.idx,
        "url":         f"/api/video/segments/{c.video_id}/{c.filename}",
        "thumb_url":   f"/api/video/segments/{c.video_id}/{c.thumb}",
        "prompt":      c.prompt,
        "frame_count": c.frame_count,
        "workflow":    c.workflow,
        "width":       c.width,
        "height":      c.height,
        "fps":         c.fps,
        "has_audio":   c.has_audio,
        "created_at":  c.created_at.isoformat(),
    }


@router.get("/clips")
async def list_clips(db: AsyncSession = Depends(get_db)):
    """All library clips across every job — the client groups them into
    per-job stacks via video_id (job order comes from GET /api/video)."""
    result = await db.execute(
        select(VideoClip).order_by(VideoClip.video_id, VideoClip.idx)
    )
    return [_serialize_clip(c) for c in result.scalars().all()]


@router.delete("/clips/{clip_id}", status_code=204)
async def delete_clip(clip_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """Delete one clip (file + thumbnail + row). A generation job that loses
    its last clip is pruned entirely — an empty stack has nothing to show."""
    clip = await db.get(VideoClip, clip_id)
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")
    job_id = clip.video_id
    seg_dir = _segments_dir(job_id)
    (seg_dir / clip.filename).unlink(missing_ok=True)
    (seg_dir / clip.thumb).unlink(missing_ok=True)
    await db.delete(clip)
    await db.commit()
    await _prune_empty_clip_job(db, job_id)


@router.post("/merge", status_code=202)
async def merge_videos(body: MergeRequest, db: AsyncSession = Depends(get_db)):
    """Concatenate the chosen library clips — from any jobs, any workflows, in
    the given order — into a new final video (workflow='merge'). Resolution,
    fps and audio are normalized to make mixed selections always mergeable.
    With delete_sources=true the source clips are removed after success."""
    if not body.clip_ids:
        raise HTTPException(status_code=400, detail="No clips selected")
    if len(body.clip_ids) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 clips per merge")
    if len(set(body.clip_ids)) != len(body.clip_ids):
        raise HTTPException(status_code=400, detail="Duplicate clip IDs in selection")

    result = await db.execute(select(VideoClip).where(VideoClip.id.in_(body.clip_ids)))
    by_id = {c.id: c for c in result.scalars().all()}
    missing = [str(cid) for cid in body.clip_ids if cid not in by_id]
    if missing:
        raise HTTPException(status_code=404, detail=f"Clip(s) not found: {', '.join(missing)}")
    first = by_id[body.clip_ids[0]]

    video = Video(
        id=uuid.uuid4(),
        workflow="merge",
        status="assembling",
        width=first.width,
        height=first.height,
        fps=first.fps,
        n_images=len(body.clip_ids),   # for merges: number of source clips
        created_at=datetime.now(timezone.utc),
    )
    db.add(video)
    await db.commit()
    await db.refresh(video)

    safe_create_task(
        _run_merge(video.id, body.clip_ids, body.delete_sources),
        name=f"video_merge:{video.id}",
    )
    logger.info(
        "Queued merge %s from %d clip(s), delete_sources=%s",
        video.id, len(body.clip_ids), body.delete_sources,
    )
    return {"video_id": str(video.id), "status": "assembling"}


@router.get("/segments/{video_id}/{filename}")
async def serve_segment(video_id: uuid.UUID, filename: str):
    """Serve a single clip MP4 or its thumbnail JPEG (clip library files)."""
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


class VideoUpdate(BaseModel):
    title: str | None = None
    notes: str | None = None


@router.patch("/jobs/{video_id}")
async def update_video(
    video_id: uuid.UUID,
    body: VideoUpdate,
    db: AsyncSession = Depends(get_db),
):
    """Update user-editable fields. Empty string clears the field; null is ignored."""
    video = await db.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if "title" in body.model_fields_set:
        video.title = (body.title or "").strip()[:255] or None
    if "notes" in body.model_fields_set:
        video.notes = (body.notes or "").strip() or None
    await db.commit()
    await db.refresh(video)
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
    if video.muxed_filename:
        mp = settings.videos_dir / video.muxed_filename
        mp.unlink(missing_ok=True)
    thumb = settings.videos_dir / f"{video_id}_thumb.jpg"
    thumb.unlink(missing_ok=True)
    seg_dir = _segments_dir(video_id)
    if seg_dir.exists():
        shutil.rmtree(seg_dir, ignore_errors=True)
    await db.delete(video)
    await db.commit()


# ── Soundtrack (mux a generated Song onto a generated Video) ──────────────────

class SoundtrackAttach(BaseModel):
    song_id: uuid.UUID


async def _run_soundtrack_mux(video_id: uuid.UUID, song_id: uuid.UUID) -> None:
    """Background task: probe the video, mux the song's audio with fade-out,
    persist the muxed filename + FK. On failure write `error` on the Video row."""
    video_key = str(video_id)
    try:
        async with AsyncSessionLocal() as db:
            video = await db.get(Video, video_id)
            song = await db.get(Song, song_id)
            if not video or not video.filepath:
                raise RuntimeError("Video row gone or has no file")
            if not song or not song.filepath:
                raise RuntimeError("Song row gone or has no file")
            video_path = settings.storage_dir / video.filepath
            song_path = settings.storage_dir / song.filepath

        out_name = f"{video_id}_muxed.mp4"
        out_path = settings.videos_dir / out_name

        _progress[video_key] = {
            "phase": "muxing",
            "message": "Adding soundtrack…",
            "pct": 50,
        }
        await mux_soundtrack(
            video_path, song_path, out_path,
            ffmpeg_path=settings.ffmpeg_path,
            fade_out_seconds=1.0,
        )

        async with AsyncSessionLocal() as db:
            video = await db.get(Video, video_id)
            if video:
                video.soundtrack_song_id = song_id
                video.muxed_filename = out_name
                video.error = None
                await db.commit()
        _progress.pop(video_key, None)
        logger.info("Soundtrack attached: video=%s song=%s → %s", video_id, song_id, out_name)

    except Exception as exc:
        logger.exception("Soundtrack mux failed for video=%s song=%s", video_id, song_id)
        _progress.pop(video_key, None)
        msg = str(exc).strip()
        err = f"{type(exc).__name__}: {msg}" if msg else type(exc).__name__
        async with AsyncSessionLocal() as db:
            video = await db.get(Video, video_id)
            if video:
                video.error = err[:1000]
                await db.commit()


@router.post("/jobs/{video_id}/soundtrack", status_code=202)
async def attach_soundtrack(
    video_id: uuid.UUID,
    body: SoundtrackAttach,
    db: AsyncSession = Depends(get_db),
):
    video = await db.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if video.status != "done" or not video.filename:
        raise HTTPException(status_code=409, detail="Video is not ready (status must be 'done')")

    song = await db.get(Song, body.song_id)
    if not song:
        raise HTTPException(status_code=404, detail="Song not found")
    if song.status != "done" or not song.filename:
        raise HTTPException(status_code=409, detail="Song is not ready (status must be 'done')")

    # Optimistic UI signal — the actual write happens in the background task.
    _progress[str(video_id)] = {
        "phase": "muxing",
        "message": "Adding soundtrack…",
        "pct": 10,
    }
    safe_create_task(_run_soundtrack_mux(video_id, body.song_id), name=f"soundtrack_mux:{video_id}")
    return _serialize(video)


@router.delete("/jobs/{video_id}/soundtrack")
async def detach_soundtrack(video_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    video = await db.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if video.muxed_filename:
        mp = settings.videos_dir / video.muxed_filename
        mp.unlink(missing_ok=True)
    video.muxed_filename = None
    video.soundtrack_song_id = None
    await db.commit()
    await db.refresh(video)
    return _serialize(video)


@router.get("/file/{filename}")
async def serve_video(filename: str):
    safe = Path(filename).name
    p = settings.videos_dir / safe
    if p.exists():
        return FileResponse(p, media_type="video/mp4")
    raise HTTPException(status_code=404, detail="Video not found")


# ── Serializer ────────────────────────────────────────────────────────────────

def _serialize(v: Video) -> dict:
    # When a soundtrack is attached, the muxed file is the default `url`
    # so players load the audio-bearing variant. The silent original stays
    # available via `original_url`.
    primary_name = v.muxed_filename or v.filename
    return {
        "id":                str(v.id),
        "status":            v.status,
        "workflow":          v.workflow or "flf2v",
        "filename":          v.filename,
        "url":               f"/api/video/file/{primary_name}" if primary_name else None,
        "original_url":      f"/api/video/file/{v.filename}" if v.filename else None,
        "soundtrack_song_id": str(v.soundtrack_song_id) if v.soundtrack_song_id else None,
        "muxed_filename":    v.muxed_filename,
        "has_soundtrack":    bool(v.muxed_filename),
        "thumb_url":         f"/api/video/thumb/{v.id}" if (v.status == "done" and v.filename) else None,
        "image_ids":         [str(i) for i in v.image_ids] if v.image_ids else [],
        "prompt":            v.prompt,
        "title":             v.title,
        "notes":             v.notes,
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
