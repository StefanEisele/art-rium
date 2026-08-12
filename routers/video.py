"""
Key-frame video generation — three workflow types, all producing per-segment
CLIPS that land in a shared clip library (one "stack" per job):

  i2v_multi    Each image is animated independently (WanImageToVideo, silent).
               Supports 1–10 images with per-image prompts and frame counts.
               Prompts can be auto-suggested via POST /api/video/suggest-i2v.

  minimax_i2v  Each image is animated independently via MiniMax H3, which
               samples video and native stereo audio jointly. Supports 1–6
               images with per-image prompts and frame counts. Runs at a
               fixed 24 fps on the model's own canvas — see MINIMAX_FPS and
               adapt_minimax_canvas. Same suggest-i2v endpoint, different
               prompt writer.

  flf2v        Each adjacent pair of key frames becomes its own independent
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

The SEEDVR2 upscale exists at both levels, and which one to use is a real
choice: per CLIP (before merging) keeps the restorer and RIFE inside one
continuous shot, per VIDEO runs them across the cuts a merge introduced,
which shows up as morphing at the edits. Upscale the clips, merge, then
grain the result.

POST /api/video/generate            → enqueue job, return {video_id}
POST /api/video/suggest-transitions → VLM-suggested per-transition prompts (flf2v)
POST /api/video/suggest-i2v         → VLM-suggested surreal per-image prompts (i2v/minimax)
GET  /api/video/jobs/{id}           → poll status
GET  /api/video/jobs/{id}/progress  → lightweight progress (ComfyUI queue + phase)
GET  /api/video/clips               → all library clips (frontend groups by job)
DELETE /api/video/clips/{clip_id}   → delete one clip (empty source jobs are pruned)
POST /api/video/clips/{id}/upscale  → SEEDVR2 pass on ONE clip, before merging
DELETE /api/video/clips/{id}/upscale→ drop it again
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
from core.models import AUDIO_WORKFLOWS, Image, Song, Video, VideoClip
from core.tasks import safe_create_task
from core.video_thumb import (
    make_video_thumbnail,
    probe_has_audio,
    probe_video_dimensions,
    probe_video_duration,
)
from services.comfy.client import (
    free_memory,
    poll_history,
    post_workflow,
    queue_info,
    upload_image,
)
from services.comfy.ingest import ingest_comfy_image
from services.comfy.zimage import ZIMAGE_SAVE_NODE, build_zimage_workflow
from workers.comfy_listener import get_listener
from services.ollama.analysis import (
    cancel_titler_warmup,
    generate_i2v_motion_prompts,
    generate_minimax_motion_prompts,
    generate_transition_prompts,
)
from services.ollama.chat import unload_model, wait_until_unloaded
from services.ollama.story_frames import (
    describe_image_for_story,
    generate_story_frame_prompts,
)
from services.ollama.zimage_enhance import get_zimage_style_block
from services.video.audio_stretch import stretch_audio_to_video, stretch_native_audio
from services.video.grain import render_grain, render_grain_preview
from services.video.merge import MergeInput, merge_clips
from services.video.soundtrack import mux_soundtrack
from services.video.upscale import (
    RESOLUTION_MAX,
    RESOLUTION_MIN,
    RIFE_MULTIPLIERS,
    build_upscale_workflow,
    clamp_resolution,
    clamp_rife,
    estimate_seconds,
    output_dimensions,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/video", dependencies=[Depends(require_auth)])

# ── Per-job progress (module-level, single-process safe) ──────────────────────
# str(video_id) → {"phase": str, "message": str, "pct": int, plus optional
# private keys "_prompt_id" (the ComfyUI prompt running *right now*, so the
# progress endpoint can attach that prompt's live node/step detail) and "_band"
# (the pct range this ComfyUI submission owns, so its 0-100% maps into the
# job's overall bar). Underscore keys are stripped before the response.
_progress: dict[str, dict] = {}

# Phases that run on an already-finished video, so status alone cannot say
# whether the job is idle. See _is_upscaling / _is_graining, which test the
# same dict one phase at a time.
_POST_PASS_PHASES = frozenset({"upscaling", "graining"})

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

# ── MiniMax H3 i2v (separate model family, native stereo audio) ───────────────
# Omni-modal model: video and audio are denoised jointly in one AV latent, so a
# single sampler pass yields both. Combo strings verified against ComfyUI
# /object_info; the text encoder is a Qwen3-VL-32B (nvfp4) loaded as a CLIP of
# type "minimax". Guidance-free (BasicGuider, no negative prompt) — the model
# takes no cfg, so steering happens through the positive prompt alone.
# int8, after measuring the 11.3 GB int4 community quant against it on the
# 16 GB 4060 Ti and finding it slower, not faster (2026-08-09):
#   int8, dynamic VRAM, 1344x768x56f  45.5 s/step  (16.1 min)
#   int4, dynamic VRAM, 1344x768x56f  53.4 s/step  (18.7 min)
#   int4, static load,  1344x768x56f  OOM at 15.1/16 GB (weights fit, activations don't)
#   int4, static load,   768x768x56f  29.9 s/step  (12.5 min, GPU pinned at 99%)
# Halving the weights does not halve the step: the loader streams either way,
# and int4 adds a dequantise-to-bf16 pass per step that costs more than the
# smaller transfer saves. Making it resident needs static loading, which only
# fits at a reduced canvas — and even there the card is compute-bound, so the
# win is small and paid for in resolution. The int4 file is kept on disk;
# revisit if a future ComfyUI keeps a fitting model resident under dynamic VRAM.
_MMX_UNET      = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
_MMX_CLIP      = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
_MMX_VAE       = "minimax_h3_video_vae_fp16.safetensors"
_MMX_AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"
_MMX_SAMPLER   = "res_multistep"
_MMX_SCHEDULER = "simple"
_MMX_STEPS     = 20
# The model is trained at a fixed 24 fps and its audio latent is locked to that
# rate (comfy_extras/nodes_minimax_h3.py: FPS = 24, AUDIO_LATENT_FPS = 40), so
# unlike the Wan workflows the frame rate is NOT user-selectable — another rate would
# desync the generated audio from the video it was jointly sampled with.
MINIMAX_FPS = 24
# Canvas rules mirrored from comfy_extras/nodes_minimax_h3.py. MiniMaxH3ImageToVideo
# itself does NOT clamp — it renders whatever it is given — so we apply the
# model's own documented canvas here instead of letting an off-canvas request
# through at a quality and speed penalty.
_MMX_CANVAS_MULTIPLE = 32
_MMX_MAX_PIXELS      = 768 * 1344   # the model's native canvas is a 768px short edge
# Frame counts live on a 17k+5 grid at 24 fps; the trained range is ~124-362
# frames (≈5-15 s) per the node's own tooltip.
_MMX_FRAME_GRID   = 17
_MMX_FRAME_OFFSET = 5
MINIMAX_FRAMES_MIN = 5
MINIMAX_FRAMES_MAX = 600

POLL_INTERVAL = 15    # seconds between ComfyUI history polls
POLL_TIMEOUT  = 1800  # 30 minutes max

# ── Pydantic ──────────────────────────────────────────────────────────────────

class GenerateVideoRequest(BaseModel):
    image_ids: list[uuid.UUID]
    workflow: str = "i2v_multi"    # "i2v_multi" | "minimax_i2v" | "flf2v"
    width:  int = 1088
    height: int = 1088
    frame_count: int = 49          # fallback frame count when prompts/frame_counts arrays are absent
    fps:    int = 24               # ignored by minimax_i2v (model is fixed at MINIMAX_FPS)
    prompt: str = ""               # fallback prompt when `prompts` is absent/mismatched length
    prompts: list[str] = []        # i2v_multi/minimax_i2v: one per image; flf2v: one per transition (n-1)
    frame_counts: list[int] = []   # i2v_multi/minimax_i2v: one per image; flf2v: one per transition (n-1)
    rife_multiplier: int = 3       # RIFE VFI frame interpolation factor (2/3/4; minimax_i2v also allows 1 = off)
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


# ── MiniMax H3 i2v workflow builder (joint video+audio sampling) ──────────────

def align_minimax_length(frame_count: int) -> int:
    """Snap a frame count up onto MiniMax H3's 17k+5 grid.

    Mirrors `align_frame_count` in comfy_extras/nodes_minimax_h3.py. The node
    snaps internally too, so this is not needed for the graph to validate —
    it is needed because the caller must know the *resulting* length to derive
    the generated audio's true duration when re-syncing it after RIFE.
    """
    n = max(MINIMAX_FRAMES_MIN, frame_count)
    while n % _MMX_FRAME_GRID != _MMX_FRAME_OFFSET:
        n += 1
    return n


def adapt_minimax_canvas(width: int, height: int) -> tuple[int, int]:
    """Round a requested size onto a canvas MiniMax H3 can actually sample.

    Two hard requirements, both taken from comfy_extras/nodes_minimax_h3.py:
    each axis a multiple of 32 (the node's own step, and the video latent is
    width//16), and the total area within the model's 768*1344 budget.
    MiniMaxH3ImageToVideo enforces neither — it renders whatever it is handed —
    so a 1920x1088 request really would be sampled at 1920x1088, twice the
    trained pixel budget for worse output at several times the cost.

    Sizes already inside the budget keep their own scale. An earlier version
    also *enlarged* smaller requests onto the model's native 768px short edge,
    which made a deliberately small canvas impossible to ask for. Going below
    native costs fidelity but is far cheaper per step, and that trade belongs
    to the caller, not here.
    """
    w, h = float(max(1, width)), float(max(1, height))
    if w * h > _MMX_MAX_PIXELS:
        s = (_MMX_MAX_PIXELS / (w * h)) ** 0.5
        w, h = w * s, h * s
    m = _MMX_CANVAS_MULTIPLE
    return (max(m, round(w / m) * m), max(m, round(h / m) * m))


def _build_minimax_single_workflow(
    comfy_filename: str,
    prompt: str,
    frame_count: int,
    width: int, height: int,
    vid_prefix: str,
    rife_multiplier: int = 1,
) -> tuple[dict, str, int]:
    """Single-image MiniMax H3 i2v segment producing an mp4 *with* generated audio.

    API-format translation of the `Image to Video (MiniMax H3)` subgraph from
    workflows/video_minimax_h3_i2v.json: load models → conditioning + empty AV
    latent from the source frame → one guidance-free sampler pass → decode video
    and audio from the same latent → VHS mux.

    H3 denoises video and audio jointly in a single NestedTensor latent, so one
    SamplerCustomAdvanced call produces both streams already in sync — no separate
    audio sampling stage, and no upscale/refine pass.

    Like the Wan builders, one image == one ComfyUI submission so the VRAM
    peak is independent of how many images the batch holds. Returns (workflow,
    save_node_id, snapped_length); the caller needs `snapped_length` (frames at
    MINIMAX_FPS) to compute the native pre-RIFE audio duration when stretching
    audio to match a RIFE'd clip.

    Unlike the Wan builders this takes no `fps`: the model's rate is not a
    caller decision (see MINIMAX_FPS), and accepting a value only to ignore it
    would invite a caller to believe 30 fps was honoured.

    rife_multiplier > 1 inserts a RIFE VFI pass on the decoded video frames,
    which stretches real-time duration rather than adding smoothness at a fixed
    duration (VHS_VideoCombine's frame_rate stays put, frame count multiplies).
    The generated audio is decoded at the *original* length, so it ends up
    shorter than the RIFE'd video; the caller re-syncs it via
    services.video.audio_stretch.stretch_native_audio after harvesting.
    """
    width, height = adapt_minimax_canvas(width, height)
    length = align_minimax_length(frame_count)
    seed = random.randint(0, 2**32 - 1)

    p = "mmx_"
    wf: dict = {
        p+"load":  {"class_type": "LoadImage", "inputs": {"image": comfy_filename, "upload": "image"}},
        # ── Models ──
        p+"unet":  {"class_type": "UNETLoader", "inputs": {"unet_name": _MMX_UNET, "weight_dtype": "default"}},
        p+"clip":  {"class_type": "CLIPLoader", "inputs": {"clip_name": _MMX_CLIP, "type": "minimax", "device": "default"}},
        p+"vae":   {"class_type": "VAELoader",  "inputs": {"vae_name": _MMX_VAE}},
        p+"avae":  {"class_type": "VAELoader",  "inputs": {"vae_name": _MMX_AUDIO_VAE}},
        # ── Image preprocess ──
        # The node stretches first_frame onto the canvas with crop disabled, which
        # distorts anything whose aspect ratio differs from the target. Center-crop
        # to the exact canvas first so that internal resize is a no-op.
        p+"scale": {"class_type": "ImageScale", "inputs": {
            "image": [p+"load", 0], "upscale_method": "lanczos",
            "width": width, "height": height, "crop": "center",
        }},
        # ── Conditioning + empty AV latent (one node does both) ──
        p+"i2v":   {"class_type": "MiniMaxH3ImageToVideo", "inputs": {
            "clip": [p+"clip", 0], "vae": [p+"vae", 0],
            "prompt": prompt,
            "width": width, "height": height, "length": length,
            "first_frame": [p+"scale", 0],
        }},
        # ── Sampling (guidance-free: BasicGuider takes positive only) ──
        p+"guider": {"class_type": "BasicGuider",    "inputs": {"model": [p+"unet", 0], "conditioning": [p+"i2v", 0]}},
        p+"sched":  {"class_type": "BasicScheduler", "inputs": {"model": [p+"unet", 0], "scheduler": _MMX_SCHEDULER, "steps": _MMX_STEPS, "denoise": 1.0}},
        p+"samsel": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": _MMX_SAMPLER}},
        p+"noise":  {"class_type": "RandomNoise",    "inputs": {"noise_seed": seed}},
        p+"ks":     {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": [p+"noise", 0], "guider": [p+"guider", 0],
            "sampler": [p+"samsel", 0], "sigmas": [p+"sched", 0],
            "latent_image": [p+"i2v", 1],
        }},
        # ── Decode both streams from the same AV latent ──
        p+"vdec":  {"class_type": "VAEDecode",      "inputs": {"samples": [p+"ks", 0], "vae": [p+"vae", 0]}},
        p+"adec":  {"class_type": "VAEDecodeAudio", "inputs": {"samples": [p+"ks", 0], "vae": [p+"avae", 0]}},
    }

    # Optional RIFE VFI on the decoded frames (off at rife_multiplier=1 — no node
    # added at all, matching the Wan builders' param shape exactly).
    frames_node = (p+"vdec", 0)
    if rife_multiplier > 1:
        wf[p+"rife"] = {"class_type": "RIFE VFI", "inputs": {
            "ckpt_name":                  _RIFE_CKPT,
            "clear_cache_after_n_frames": 10,
            "multiplier":                 rife_multiplier,
            "fast_mode":                  True,
            "ensemble":                   True,
            "scale_factor":               1,
            "dtype":                      "float32",
            "torch_compile":              False,
            "batch_size":                 1,
            "frames":                     [p+"vdec", 0],
        }}
        frames_node = (p+"rife", 0)

    # h264/yuv420p for broad browser playback of a clip that carries audio
    # (the silent Wan workflows save h265 10-bit instead).
    # When RIFE is on, this audio track is intentionally left at its native
    # (shorter) length — the caller stretches it to match afterward.
    save_id = "mmx_save"
    wf[save_id] = {"class_type": "VHS_VideoCombine", "inputs": {
        "frame_rate":      MINIMAX_FPS,
        "loop_count":      0,
        "filename_prefix": vid_prefix,
        "format":          "video/h264-mp4",
        "pix_fmt":         "yuv420p",
        "crf":             19,
        "save_metadata":   False,
        # A format parameter of h264-mp4.json, like crf/pix_fmt above. Left
        # implicit it defaults to false and only logs a warning, but it must
        # stay false: this is the one workflow where video and audio lengths
        # legitimately differ (RIFE lengthens the video, the audio is stretched
        # afterwards by us), and trimming to the short native audio here would
        # throw away the interpolation without any error.
        "trim_to_audio":   False,
        "pingpong":        False,
        "save_output":     True,
        "images":          list(frames_node),
        "audio":           [p+"adec", 0],
    }}
    return wf, save_id, length


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
            fps=MINIMAX_FPS if req.workflow == "minimax_i2v" else req.fps,
            has_audio=(req.workflow in AUDIO_WORKFLOWS),
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

def _set_progress(
    vid_key: str,
    phase: str,
    message: str,
    pct: int,
    *,
    prompt_id: str | None = None,
    band: tuple[int, int] | None = None,
) -> None:
    """Record the job's coarse phase, plus what ComfyUI submission (if any) owns it.

    `prompt_id` + `band` are what let GET /jobs/{id}/progress replace a static
    "generating frames…" with ComfyUI's own live stage and sampler step,
    scaled into the slice of the bar this submission is responsible for.
    """
    entry = {"phase": phase, "message": message, "pct": pct}
    if prompt_id:
        entry["_prompt_id"] = prompt_id
    if band:
        entry["_band"] = band
    _progress[vid_key] = entry


def _attach_live_stage(prog: dict) -> dict:
    """Fold ComfyUI's current node + sampler step into a progress payload.

    The listener sees every ComfyUI event already (workers/comfy_listener.py);
    this reads the latest one for the prompt the job is waiting on, so a
    multi-minute render reports "Sampling… · step 7/20" and moves its bar
    inside the band, instead of sitting frozen at one percentage until the
    segment finishes. Returns a copy with the private keys removed.
    """
    prompt_id = prog.pop("_prompt_id", None)
    lo, hi = prog.pop("_band", (None, None))
    listener = get_listener()
    step = listener.get_step_progress(prompt_id) if listener else None
    if not step:
        return prog

    label = step.get("label")
    value, maximum = step.get("value"), step.get("max")
    if maximum:
        ratio = max(0.0, min(1.0, float(value or 0) / float(maximum)))
        if lo is not None:
            prog["pct"] = int(lo + ratio * (hi - lo))
        prog["step"] = {"value": int(value or 0), "max": int(maximum)}
        prog["detail"] = f"{label or 'Sampling…'} step {int(value or 0)}/{int(maximum)}"
    elif label:
        # Between samplers — loading a model, decoding, encoding the mp4. No
        # counter to report, but the stage name is the informative part anyway.
        prog["detail"] = label
    return prog


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


def _register_labels(prompt_id: str, workflow: dict) -> None:
    """Teach the listener what this submission's node ids mean (best-effort)."""
    listener = get_listener()
    if listener:
        listener.register_node_labels(prompt_id, workflow)


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
        _register_labels(prompt_id, wf)
        logger.info("Video job %s transition %d → ComfyUI prompt %s", video_id, i + 1, prompt_id)
        if i == 0:
            await _save_comfy_prompt_id(video_id, prompt_id)

        _set_progress(
            vid_key, "running", f"Transition {i + 1}/{n_trans} — generating frames…", pct_seg_mid,
            prompt_id=prompt_id, band=(pct_seg_mid, pct_seg_hi),
        )
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
        native_length = None
        if req.workflow == "minimax_i2v":
            wf, save_node, native_length = _build_minimax_single_workflow(
                fname, p_i, fc_i, req.width, req.height, seg_prefix,
                req.rife_multiplier,
            )
        else:
            wf, save_node = _build_i2v_single_workflow(
                fname, p_i, fc_i, req.width, req.height, req.fps, seg_prefix,
                req.rife_multiplier, req.pingpong,
            )
        prompt_id = await _post_workflow_with_retry(client, wf)
        _register_labels(prompt_id, wf)
        logger.info("Video job %s segment %d → ComfyUI prompt %s", video_id, i + 1, prompt_id)
        if i == 0:
            await _save_comfy_prompt_id(video_id, prompt_id)

        _set_progress(
            vid_key, "running", f"Clip {i + 1}/{n_imgs} — generating frames…", pct_seg_mid,
            prompt_id=prompt_id, band=(pct_seg_mid, pct_seg_hi),
        )
        seg_outputs = await poll_history(client, prompt_id, timeout=POLL_TIMEOUT, interval=POLL_INTERVAL)
        seg_src = _comfy_save_path(seg_outputs.get(save_node, {}), f"Segment {i + 1}")

        # Persist the clip under a stable name and register it in the library
        # immediately — a crash on a later segment loses nothing.
        seg_dest  = seg_dir / f"seg_{i}.mp4"
        seg_thumb = seg_dir / f"seg_{i}_thumb.jpg"
        if native_length is not None and req.rife_multiplier > 1:
            # RIFE stretched the video's real-time duration but not the model's
            # generated audio track — re-sync by time-stretching the audio
            # (pitch preserved) to the RIFE'd video's actual duration.
            await stretch_native_audio(
                seg_src, seg_dest,
                native_length=native_length, fps=MINIMAX_FPS,
                ffmpeg_path=settings.ffmpeg_path,
            )
        else:
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


async def _evict_ollama() -> None:
    """Ask Ollama to give the card back, and wait for it to confirm.

    Three steps, each because the previous one is not enough:

    1. Call off any in-flight startup warm-up. Evicting first does nothing
       against it — during a cold load the model is not resident yet, so the
       eviction finds an empty server and the warm-up then loads 5.3 GB *into*
       the render. See services/ollama/analysis.py::cancel_titler_warmup.
    2. Evict whatever is resident. The "✨ Suggest prompts" step just before a
       generation leaves the titler VLM in VRAM on a 30 min keep-alive, and
       Ollama sizes its KV cache from total VRAM, so even a 3B model holds
       gigabytes of a 16 GB card.
    3. Wait for confirmation — `unload_model` returns once Ollama accepts the
       request (~250 ms measured) while the runner keeps its VRAM longer.

    Still not a guarantee: none of this can stop an Ollama cold load that is
    already under way, which is why the caller checks free VRAM afterwards
    rather than trusting this to have worked.
    """
    await cancel_titler_warmup()
    for model in (
        settings.ollama_titler_model,
        settings.ollama_vlm_model,
        settings.ollama_prompt_model,
    ):
        if model:
            await unload_model(model)
    await wait_until_unloaded()


async def _free_ollama_vram(workflow: str | None = None) -> None:
    """Make the GPU ready for a ComfyUI render, or fail loudly trying.

    Video diffusion gets whatever Ollama leaves behind: MiniMax H3 pairs a 32B
    text encoder with the DiT and needs essentially the whole card, and
    ComfyUI's dynamic-VRAM loader does not fall back to a slower path — it
    dies with a CUDA OOM that takes its prompt worker thread with it. So evict
    first, then verify the card is actually free before submitting.
    """
    await _evict_ollama()
    await _preflight_comfyui(workflow)


# How much of the card each workflow needs free before it is safe to start.
# MiniMax stages ~15 GB for its text encoder alone ("14956MB Staged" in the
# ComfyUI log), so it needs essentially the whole 16 GB card; the Wan
# workflows are smaller. Falling short is not a slow path — ComfyUI's dynamic
# loader dies with a CUDA OOM that takes its prompt worker thread with it.
_MIN_FREE_VRAM = {
    "minimax_i2v": 13.5 * 1024**3,
}
_MIN_FREE_VRAM_DEFAULT = 9.0 * 1024**3

# How long to wait for someone else to let go of the card. Sized for the
# thing that actually holds it: an Ollama cold load of the titler VLM, which
# takes ~150 s and cannot be aborted once it has started.
_VRAM_WAIT_TIMEOUT = 210.0
_VRAM_WAIT_POLL = 5.0

# ComfyUI's /free returns as soon as it has dropped its references; CUDA hands
# the memory back a moment later. Re-measuring immediately reads the old
# number — the same reason the per-segment loop sleeps after free_memory.
_COMFY_FREE_SETTLE = 3.0


async def _release_comfy_models() -> None:
    """Ask ComfyUI to unload its resident models. Best-effort, never raises."""
    async with httpx.AsyncClient(timeout=15) as client:
        await free_memory(client)


async def _comfy_devices() -> list[dict]:
    """ComfyUI's device list, or a clear error explaining why there isn't one."""
    try:
        async with httpx.AsyncClient(timeout=8) as client:
            r = await client.get(f"http://{settings.comfyui_host}/system_stats")
    except Exception as exc:
        raise RuntimeError(f"ComfyUI is unreachable at {settings.comfyui_host} ({exc})") from exc

    # A CUDA OOM kills ComfyUI's `prompt_worker` *thread*, not its process: the
    # HTTP server keeps answering and /prompt keeps handing out prompt_ids while
    # nothing ever executes, so a job sits "generating" until the poll times out
    # half an hour later. /system_stats returning 500 is the reliable tell.
    if r.status_code != 200:
        raise RuntimeError(
            f"ComfyUI is up but not working (/system_stats → {r.status_code}). Its worker "
            "thread most likely died on an earlier out-of-memory error; restart ComfyUI."
        )
    try:
        return r.json().get("devices", []) or []
    except Exception:
        return []


async def _preflight_comfyui(workflow: str | None = None) -> None:
    """Refuse to submit until the GPU actually has room.

    Evicting Ollama is not sufficient, and neither is cancelling its warm-up.
    An Ollama *cold* load takes ~150 s, allocates VRAM progressively as it
    goes, and cannot be stopped from the client side — `/api/ps` does not even
    list the model until the load finishes, so both `wait_until_unloaded` and
    `cancel_titler_warmup` see an idle server and wave the job through while
    several GB are quietly being taken. Measured on a post-reboot start:
    ComfyUI up at 14:51:18, job submitted 14:52:05, dead at 14:52:12.

    So stop reasoning about *who* holds the card and check the only thing that
    matters — how much is free — retrying the eviction while we wait. Failing
    here with a number is strictly better than the alternative: a CUDA OOM
    that kills ComfyUI's worker thread and needs a restart.

    Ollama is not the only holder, though, and for a post-pass it is usually
    not the holder at all: ComfyUI keeps the models from the render that just
    finished, so an upscale queued straight after a MiniMax job finds ~3 GB
    free and no amount of evicting Ollama changes that. On the first shortfall
    we therefore ask ComfyUI to let go too, before starting to wait.
    """
    required = _MIN_FREE_VRAM.get(workflow or "", _MIN_FREE_VRAM_DEFAULT)
    deadline = asyncio.get_event_loop().time() + _VRAM_WAIT_TIMEOUT
    warned = False
    released = False

    while True:
        devices = await _comfy_devices()
        if not devices:
            logger.warning("ComfyUI reported no devices — skipping the VRAM pre-flight")
            return
        dev = devices[0]
        free, total = dev.get("vram_free", 0), dev.get("vram_total", 0)

        if free >= required:
            logger.info(
                "Pre-flight VRAM on %s: %.1f GB free of %.1f GB (need %.1f)",
                dev.get("name"), free / 1e9, total / 1e9, required / 1e9,
            )
            return

        # Before waiting on anyone, make ComfyUI drop what the previous render
        # left resident. Waiting cannot fix that on its own — nothing else
        # frees it — so a missing release here is a guaranteed timeout, not a
        # slow start.
        if not released:
            released = True
            logger.info(
                "Only %.1f GB free on %s — asking ComfyUI to unload before waiting",
                free / 1e9, dev.get("name"),
            )
            await _release_comfy_models()
            await asyncio.sleep(_COMFY_FREE_SETTLE)
            continue

        if not warned:
            logger.warning(
                "Only %.1f GB free on %s, %s needs %.1f GB — waiting for the card",
                free / 1e9, dev.get("name"), workflow or "this workflow", required / 1e9,
            )
            warned = True

        if asyncio.get_event_loop().time() >= deadline:
            raise RuntimeError(
                f"GPU still busy after {_VRAM_WAIT_TIMEOUT:.0f}s: only "
                f"{free / 1e9:.1f} GB free of {total / 1e9:.1f} GB, {workflow or 'this workflow'} "
                f"needs {required / 1e9:.1f} GB. Something else is holding the card "
                "(an Ollama model — check `ollama ps` — or a ComfyUI render that "
                "will not unload; restarting ComfyUI clears the latter)."
            )

        # Whoever it is may only just have finished loading; try both holders
        # again before the next check.
        await _evict_ollama()
        await _release_comfy_models()
        await asyncio.sleep(_VRAM_WAIT_POLL)


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

        await _free_ollama_vram(req.workflow)

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
            if c.upscale_filename:
                (seg_dir / c.upscale_filename).unlink(missing_ok=True)
            _progress.pop(_clip_key(c.id), None)
            await db.delete(c)
        await db.commit()
        for jid in job_ids:
            await _prune_empty_clip_job(db, jid)


def _merge_canvas(clips: list[VideoClip]) -> tuple[int, int]:
    """Normalisation target for a merge: the largest effective canvas selected.

    Effective, not stored — an upscaled clip's real size is its upscale's, and
    the merge is the whole reason that pass exists. Largest rather than the
    first clip's: with a uniform selection (the ordinary case) the two agree,
    and where they disagree it is because only some clips were upscaled, where
    "first wins" would scale the restored ones back down and undo the work.
    """
    sizes = [_clip_dimensions(c) for c in clips] or [(None, None)]
    return max(((w or 960, h or 960) for w, h in sizes), key=lambda wh: wh[0] * wh[1])


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
            # The upscaled rendition when the clip has one: upscaling happens
            # per clip precisely so the merge can consume it, and reading
            # `filename` here would throw that work away.
            f = _clip_primary_path(c)
            if not f.exists():
                raise FileNotFoundError(f"Clip file missing on disk: {f}")
            inputs.append(MergeInput(path=f, has_audio=c.has_audio))

        width, height = _merge_canvas(clips)
        fps = clips[0].fps or 24
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
    workflow: str = "i2v_multi"  # suggest-i2v only: "i2v_multi" (Wan, silent) | "minimax_i2v" (MiniMax H3, native audio)


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


_SUGGEST_I2V_JOBS_KEEP = 20     # in-memory job entries retained (newest first)

# job_id → {status, message, pct, prompts, error, created_at}
_suggest_i2v_jobs: dict[str, dict] = {}


def _prune_suggest_i2v_jobs() -> None:
    if len(_suggest_i2v_jobs) <= _SUGGEST_I2V_JOBS_KEEP:
        return
    for job_id, _ in sorted(
        _suggest_i2v_jobs.items(), key=lambda kv: kv[1].get("created_at", "")
    )[: len(_suggest_i2v_jobs) - _SUGGEST_I2V_JOBS_KEEP]:
        _suggest_i2v_jobs.pop(job_id, None)


async def _run_suggest_i2v(job_id: str, jpgs: list[bytes], workflow: str, context: str) -> None:
    """Background task behind /suggest-i2v: N serialized per-image VLM calls
    (MiniMax H3's audio-aware prompts in particular) reliably clear a minute for
    more than a couple of images, well past the Cloudflare tunnel's ~100s
    upstream timeout — running this inline in the request handler was the
    524. Polled like /story-frames instead."""
    n = len(jpgs)
    generate = (
        generate_minimax_motion_prompts if workflow == "minimax_i2v"
        else generate_i2v_motion_prompts
    )

    def on_progress(done: int, total: int) -> None:
        job = _suggest_i2v_jobs.get(job_id)
        if job is not None:
            job.update(message=f"Prompt {done}/{total}…", pct=max(1, int(100 * done / total)))

    try:
        prompts = await generate(jpgs, context=context, on_progress=on_progress)
        job = _suggest_i2v_jobs.get(job_id)
        if job is not None:
            job.update(status="done", message=f"{n} prompt(s) ready", pct=100, prompts=prompts)
    except Exception as exc:
        logger.exception("Suggest-i2v job %s failed (%d images)", job_id, n)
        job = _suggest_i2v_jobs.get(job_id)
        if job is not None:
            msg = str(exc).strip()
            job.update(
                status="failed",
                message="Suggestion failed",
                error=(f"{type(exc).__name__}: {msg}" if msg else type(exc).__name__)[:1000],
            )


@router.post("/suggest-i2v", status_code=202)
async def suggest_i2v(
    body: SuggestTransitionsRequest, db: AsyncSession = Depends(get_db),
):
    """Kick off VLM-suggested per-image surreal animation prompts for
    i2v_multi/minimax_i2v — one vision call PER image, run as a background job
    polled via GET /suggest-i2v/{job_id} (see _run_suggest_i2v for why: this
    used to run inline and 524'd behind the Cloudflare tunnel). `body.workflow`
    picks the model-specific prompt writer: minimax_i2v gets MiniMax H3's
    audio-aware variant, everything else gets the Wan2.2 (silent) variant."""
    n = len(body.image_ids)
    if n < 1:
        raise HTTPException(status_code=400, detail="Need at least 1 image to suggest prompts")
    if n > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 images")

    jpgs = await _load_suggest_jpgs(body.image_ids, db)
    logger.info(
        "Suggest-i2v: %d images, workflow=%s, model=%s, payload=%dKB",
        n, body.workflow, settings.ollama_titler_model, sum(len(j) for j in jpgs) // 1024,
    )

    job_id = str(uuid.uuid4())
    _suggest_i2v_jobs[job_id] = {
        "job_id":     job_id,
        "status":     "running",
        "message":    f"Prompt 0/{n}…",
        "pct":        1,
        "prompts":    [],
        "error":      None,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _prune_suggest_i2v_jobs()

    safe_create_task(
        _run_suggest_i2v(job_id, jpgs, body.workflow, body.context),
        name=f"suggest_i2v:{job_id}",
    )
    logger.info("Queued suggest-i2v job %s (%d images, workflow=%s)", job_id, n, body.workflow)
    return {"job_id": job_id, "status": "running"}


@router.get("/suggest-i2v/{job_id}")
async def get_suggest_i2v_job(job_id: str):
    job = _suggest_i2v_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Suggest-i2v job not found (server restarted?)")
    return job


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
    # None = inherit the source image's own LoRA(s) (high consistency across
    # the story sequence); an explicit value still overrides it.
    loras: list[dict] | None = None


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
        job.pop("_prompt_id", None)
        job.pop("_band", None)


def _story_stage(job_id: str, prompt_id: str, lo: int, hi: int) -> None:
    """Point the job at the ComfyUI prompt it is now waiting on — see
    _attach_live_stage, which turns this into a live node/step readout."""
    job = _story_jobs.get(job_id)
    if job is not None:
        job["_prompt_id"] = prompt_id
        job["_band"] = (lo, hi)


def _seed_for_frame(frame_prompt: str, base_seed: int, seen_prompts: dict[str, int]) -> int:
    """Bump the seed when *frame_prompt* repeats verbatim earlier in the same
    job (e.g. generate_story_frame_prompts padding a short response by
    duplicating the last frame). With the seed otherwise shared across every
    frame, a repeated prompt is a byte-identical ComfyUI submission — the
    node-level cache then skips execution and returns no fresh SaveImage
    output. Mutates `seen_prompts` to track repeat counts across calls."""
    dup_count = seen_prompts.get(frame_prompt, 0)
    seen_prompts[frame_prompt] = dup_count + 1
    return base_seed + dup_count


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
        lora_by_filename = {lora["filename"]: lora for lora in LORAS}
        triggers = [
            lora_by_filename[sel["name"]]["trigger"]
            for sel in req.loras
            if sel["name"] in lora_by_filename and lora_by_filename[sel["name"]]["trigger"]
        ]
        trigger = ", ".join(triggers) or None
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

        seen_prompts: dict[str, int] = {}

        async with httpx.AsyncClient(timeout=60) as client:
            for i, frame_prompt in enumerate(prompts):
                _story_update(
                    job_id, "generating",
                    f"Frame {i + 1}/{n} — generating…",
                    18 + int(78 * i / n),
                )
                frame_seed = _seed_for_frame(frame_prompt, seed, seen_prompts)

                wf = build_zimage_workflow(
                    frame_prompt, frame_seed, req.width, req.height, req.loras,
                )
                prompt_id = await _post_workflow_with_retry(client, wf)
                _register_labels(prompt_id, wf)
                # Same live-stage treatment as a video segment — this loop is
                # otherwise silent for the whole of each frame's render.
                _story_stage(job_id, prompt_id, 18 + int(78 * i / n), 18 + int(78 * (i + 1) / n))
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
                    seed=frame_seed,
                    width=req.width,
                    height=req.height,
                    loras=req.loras,
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

    # Default LoRA(s) to the source image's own generation params — keeps the
    # story sequence visually consistent unless explicitly overridden.
    if body.loras is None:
        body.loras = image.loras or [{"name": DEFAULT_LORA, "strength": 0.5}]
    for sel in body.loras:
        if sel["name"] not in ALLOWED_LORAS:
            raise HTTPException(status_code=400, detail=f"Unknown LoRA: {sel['name']}")

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
    return _attach_live_stage(dict(job))


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
    elif body.workflow == "minimax_i2v":
        if not (1 <= n <= 6):
            raise HTTPException(status_code=400, detail="minimax_i2v requires 1–6 image IDs")
        # Wider than the Wan/flf2v 5–81 band: MiniMax H3's trained range is
        # ~124–362 frames (≈5–15 s at its fixed 24 fps). Values are snapped up
        # onto the model's 17k+5 grid by the builder.
        for fc in body.frame_counts:
            if not (MINIMAX_FRAMES_MIN <= fc <= MINIMAX_FRAMES_MAX):
                raise HTTPException(
                    status_code=400,
                    detail=f"frame_counts values must be {MINIMAX_FRAMES_MIN}–{MINIMAX_FRAMES_MAX}",
                )
        if body.rife_multiplier not in (1, 2, 3, 4):
            raise HTTPException(status_code=400, detail="rife_multiplier must be 1 (off), 2, 3 or 4")
        # Normalise the canvas here rather than only inside the builder, so the
        # dimensions persisted on the Video/VideoClip rows are the ones actually
        # rendered — the merge path normalises against them.
        body.width, body.height = adapt_minimax_canvas(body.width, body.height)
    else:
        if not (1 <= n <= 10):
            raise HTTPException(status_code=400, detail="i2v_multi requires 1–10 image IDs")
        for fc in body.frame_counts:
            if not (5 <= fc <= 81):
                raise HTTPException(status_code=400, detail="frame_counts values must be 5–81")
        if body.rife_multiplier not in (2, 3, 4):
            raise HTTPException(status_code=400, detail="rife_multiplier must be 2, 3 or 4")

    # Summarise per-clip prompts for display
    if body.workflow in ("i2v_multi", "minimax_i2v", "flf2v") and body.prompts:
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
        fps=MINIMAX_FPS if body.workflow == "minimax_i2v" else body.fps,
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

    live = dict(_progress.get(str(video_id), {}))

    if video.status == "done":
        done = {"phase": "done", "message": "Complete", "pct": 100}
        # Post-passes (upscale, grain) run on a video whose status is already
        # 'done'. `phase` must stay "done" — the generation poller stops on it,
        # and starting an upscale seconds after a render finishes would
        # otherwise leave that poller running forever. The pass reports
        # alongside instead, for the chip that is watching it.
        if live.get("phase") in _POST_PASS_PHASES:
            staged = _attach_live_stage(live)
            done["pass"] = {
                "phase":   staged["phase"],
                "message": staged.get("message", ""),
                "pct":     staged.get("pct", 0),
            }
            if staged.get("detail"):
                done["detail"] = staged["detail"]
        return done
    if video.status == "failed":
        return {"phase": "failed", "message": video.error or "Generation failed", "pct": 0}

    prog = live or {"phase": "processing", "message": "Processing…", "pct": 30}

    # Enrich with live ComfyUI queue info when the prompt is submitted
    live_prompt_id = prog.get("_prompt_id") or video.comfy_prompt_id
    if live_prompt_id and prog["phase"] in ("queued", "submitting", "processing"):
        qi = await queue_info(live_prompt_id)
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

    # …then with the node/step ComfyUI is actually on right now.
    return _attach_live_stage(prog)


def _serialize_clip(c: VideoClip) -> dict:
    # `url` serves the upscaled rendition when there is one, so previewing a
    # clip shows what the merge will actually consume. `width`/`height` stay
    # the generation canvas (that is what the row means); `out_width`/
    # `out_height` are what the merge will see.
    primary = c.upscale_filename or c.filename
    out_w, out_h = _clip_dimensions(c)
    return {
        "id":          str(c.id),
        "video_id":    str(c.video_id),
        "idx":         c.idx,
        "url":         f"/api/video/segments/{c.video_id}/{primary}",
        "thumb_url":   f"/api/video/segments/{c.video_id}/{c.thumb}",
        "prompt":      c.prompt,
        "frame_count": c.frame_count,
        "workflow":    c.workflow,
        "width":       c.width,
        "height":      c.height,
        "out_width":   out_w,
        "out_height":  out_h,
        "fps":         c.fps,
        "has_audio":   c.has_audio,
        "upscale_resolution": c.upscale_resolution,
        "upscale_rife":       c.upscale_rife,
        "has_upscale":        bool(c.upscale_filename),
        "upscale_rendering":  _is_clip_upscaling(c),
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
    if clip.upscale_filename:
        (seg_dir / clip.upscale_filename).unlink(missing_ok=True)
    _progress.pop(_clip_key(clip_id), None)
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
    if video.upscale_filename:
        (settings.videos_dir / video.upscale_filename).unlink(missing_ok=True)
    if video.grain_filename:
        (settings.videos_dir / video.grain_filename).unlink(missing_ok=True)
    (settings.videos_dir / _grain_preview_name(video_id)).unlink(missing_ok=True)
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

        # Both post-passes read the muxed file, so a new soundtrack invalidates
        # whichever of them already exists — re-run them rather than leave a
        # stale rendition (which _serialize would still prefer) carrying the
        # old audio, or none at all.
        await _refresh_derived_renders(video_id, video_key)

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
    # The mirror of the guard in _validate_upscale_target, and it has to exist
    # here too: attaching a song rewrites the upscale's source, so the
    # re-render would run the interpolated path over the music and
    # time-stretch it. A song and an interpolated upscale are mutually
    # exclusive, in both directions.
    if (video.upscale_rife or 1) > 1:
        raise HTTPException(
            status_code=409,
            detail=f"This video's upscale interpolates {video.upscale_rife}×, and re-rendering "
                   "it would time-stretch the song. Re-run the upscale without "
                   "interpolation first.",
        )

    # Clear any stale error from a previous failed attempt — otherwise the
    # frontend poller can misread it as this attempt failing before the new
    # mux job has even finished.
    if video.error is not None:
        video.error = None
        await db.commit()

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

    # Dropping the soundtrack swaps the post-passes' source file back to the
    # silent original, so any existing rendition still carries audio that was
    # just removed. Re-render in the background — an upscale in particular is
    # far too slow to hold this request open.
    if video.upscale_resolution or video.grain_strength:
        _progress[str(video_id)] = {
            "phase": "upscaling" if video.upscale_resolution else "graining",
            "message": "Re-rendering…",
            "pct": 10,
        }
        safe_create_task(_run_derived_refresh(video_id), name=f"rerender:{video_id}")
    return _serialize(video)


# ── SEEDVR2 upscale (post-hoc pass over a finished video) ─────────────────────
# MiniMax H3 costs close to linearly in pixels, so the generation canvas is kept
# small (864×480) and the resolution is bought back here, on demand, from the
# video's own card. Another derived sibling file — see services/video/upscale.py
# for the model, the measured cost, and why it cannot run on the second GPU.

class UpscaleApply(BaseModel):
    resolution: int = 1080     # target SHORT edge in px; validated in the endpoint
    rife_multiplier: int = 1   # 1 = no interpolation; 2/3/4 run RIFE after the restore


def _upscale_source(video: Video) -> Path:
    """The un-upscaled file an upscale pass should read.

    Never `upscale_filename` (a second pass would restore its own
    reconstruction) and never `grain_filename` (grain belongs at the delivery
    resolution — handing a restorer a grained picture has it reconstruct the
    noise as detail). Prefers the muxed variant so an attached soundtrack
    survives.
    """
    if video.muxed_filename:
        return settings.videos_dir / video.muxed_filename
    return settings.storage_dir / video.filepath


def _upscale_name(video_id: uuid.UUID) -> str:
    return f"{video_id}_upscale.mp4"


async def _upscale_plan_from_file(
    src: Path,
    resolution: int,
    fallback_w: int | None = None,
    fallback_h: int | None = None,
    rife_multiplier: int = 1,
) -> dict:
    """Source facts the upscale needs: output size and a wall-clock estimate.

    Dimensions come from the file rather than the row because the row records
    the *generation* canvas, which a merge or an attached soundtrack may have
    moved away from; the row's values are only the fallback for an unreadable
    file.
    """
    duration = await probe_video_duration(src)
    w, h = await probe_video_dimensions(src)
    w = w or fallback_w or resolution
    h = h or fallback_h or resolution
    out_w, out_h = output_dimensions(w, h, resolution)
    return {
        "width": out_w,
        "height": out_h,
        "source_width": w,
        "source_height": h,
        # Unrounded: the audio re-sync after an interpolated run divides by
        # this, so display precision is not good enough.
        "duration": duration,
        "rife_multiplier": clamp_rife(rife_multiplier),
        "seconds": estimate_seconds(duration, out_w, out_h, rife_multiplier),
    }


# One SEEDVR2 render at a time, process-wide. ComfyUI would queue concurrent
# prompts happily enough, but each run brackets itself with `free_memory()` to
# get a clean card — and a second run calling that while the first is sampling
# pulls the models out from under it. Bulk-upscaling a stack of clips is the
# normal case now, so the serialisation has to be here rather than in the UI.
_upscale_gate = asyncio.Lock()


async def _seedvr2_render(
    src: Path,
    dest: Path,
    *,
    resolution: int,
    rife: int,
    plan: dict,
    progress_key: str,
    prefix: str,
    log_subject: str,
) -> None:
    """Restore `src` into `dest` with SEEDVR2, honouring the one-at-a-time gate.

    Shared by the video-level pass and the per-clip pass — they differ only in
    which row they read their source from and which row they write the result
    onto, so everything between those two ends lives here.
    """
    # A silent source must leave the muxer's audio slot unconnected — VHS
    # raises rather than returning an empty track when it finds no stream.
    has_audio = await probe_has_audio(src)
    wf, save_node = build_upscale_workflow(
        src,
        resolution=resolution,
        filename_prefix=prefix,
        has_audio=has_audio,
        rife_multiplier=rife,
    )

    # Three times the estimate, floored well above it: the estimate assumes
    # ~24 fps and the shared 30-minute POLL_TIMEOUT is far too short for
    # anything longer than a few seconds of footage.
    timeout = max(1800, plan["seconds"] * 3)

    if _upscale_gate.locked():
        prior = _progress.get(progress_key, {})
        _set_progress(
            progress_key, "upscaling",
            "Waiting for the GPU (another upscale is running)…",
            prior.get("pct", 10),
        )
    async with _upscale_gate:
        # "upscale" is not a generation workflow, so it takes the default budget
        # — the 3B SEEDVR2 DiT is far smaller than MiniMax's text encoder.
        await _free_ollama_vram("upscale")
        async with httpx.AsyncClient(timeout=60) as client:
            # The DiT, its VAE and whatever the generation pass left resident do
            # not fit together; force a clean card before loading.
            await free_memory(client)
            prompt_id = await post_workflow(client, wf)
            _register_labels(prompt_id, wf)
            # This pass runs for minutes with nothing else to report; hand the
            # progress endpoint the live prompt so it can show SEEDVR2's own
            # per-batch step counter instead of a frozen "Upscaling…". The
            # caller already set the wording (a re-render says so), so keep it.
            prior = _progress.get(progress_key, {})
            _set_progress(
                progress_key, "upscaling",
                prior.get("message", "Upscaling…"), prior.get("pct", 30),
                prompt_id=prompt_id, band=(prior.get("pct", 30), 88),
            )
            logger.info(
                "Upscale submitted: %s → %dx%d, ~%ds (prompt %s)",
                log_subject, plan["width"], plan["height"], plan["seconds"], prompt_id,
            )
            outputs = await poll_history(
                client, prompt_id, timeout=timeout, interval=POLL_INTERVAL,
            )
            await free_memory(client)

    comfy_src = _comfy_save_path(outputs.get(save_node, {}), "Upscale")
    dest.parent.mkdir(parents=True, exist_ok=True)

    if rife > 1 and has_audio:
        # RIFE lengthened the picture while frame_rate stayed put, so the
        # track VHS carried through is now short (padded out with silence).
        # Same shape the generation path produces — same fix.
        await stretch_audio_to_video(
            comfy_src, dest,
            native_audio_duration=plan["duration"],
            ffmpeg_path=settings.ffmpeg_path,
        )
    else:
        await asyncio.to_thread(shutil.copy2, comfy_src, dest)


async def _apply_upscale(
    video_id: uuid.UUID, resolution: int, rife_multiplier: int = 1,
) -> None:
    """Run the SEEDVR2 pass from the un-upscaled source and persist it.

    Raises on failure; each caller decides how to report. Also used to
    re-render after a soundtrack change, which swaps the source file
    underneath an existing upscale.
    """
    rife = clamp_rife(rife_multiplier)
    async with AsyncSessionLocal() as db:
        video = await db.get(Video, video_id)
        if not video or not video.filepath:
            raise RuntimeError("Video row gone or has no file")
        src = _upscale_source(video)
        fallback_w, fallback_h = video.width, video.height

    if not src.exists():
        raise RuntimeError(f"Source file missing: {src.name}")
    plan = await _upscale_plan_from_file(src, resolution, fallback_w, fallback_h, rife)

    out_name = _upscale_name(video_id)
    await _seedvr2_render(
        src, settings.videos_dir / out_name,
        resolution=resolution, rife=rife, plan=plan,
        progress_key=str(video_id),
        prefix=f"artrium_up_{video_id.hex[:10]}",
        log_subject=f"video={video_id}",
    )

    async with AsyncSessionLocal() as db:
        video = await db.get(Video, video_id)
        if video:
            video.upscale_resolution = resolution
            video.upscale_rife = rife
            video.upscale_filename = out_name
            video.error = None
            await db.commit()
    logger.info(
        "Upscale applied: video=%s resolution=%d rife=%dx → %s",
        video_id, resolution, rife, out_name,
    )


async def _run_upscale(
    video_id: uuid.UUID, resolution: int, rife_multiplier: int = 1,
) -> None:
    """Background task behind POST /jobs/{id}/upscale.

    Minutes per second of footage, so it is polled like the grain render
    rather than awaited inline. An existing grain is re-rendered afterwards:
    it was graded against the small picture and would otherwise be the older,
    lower-resolution file that _serialize keeps preferring.
    """
    video_key = str(video_id)
    _progress[video_key] = {"phase": "upscaling", "message": "Upscaling…", "pct": 30}
    try:
        await _apply_upscale(video_id, resolution, rife_multiplier)
        await _reapply_grain_if_any(video_id, video_key)
        _progress.pop(video_key, None)
    except Exception as exc:
        logger.exception("Upscale failed for video=%s resolution=%s", video_id, resolution)
        _progress.pop(video_key, None)
        await _persist_post_pass_error(video_id, exc)


async def _reapply_grain_if_any(video_id: uuid.UUID, video_key: str) -> None:
    """Re-render the grain pass when the file underneath it has changed.

    Grain is always the last pass, so anything that rewrites its source
    invalidates it — an upscale most of all, since the grained file would
    otherwise stay the older, smaller rendition that _serialize keeps
    preferring.
    """
    async with AsyncSessionLocal() as db:
        video = await db.get(Video, video_id)
        strength = video.grain_strength if video else None
    if strength:
        _progress[video_key] = {
            "phase": "graining", "message": "Re-applying grain…", "pct": 85,
        }
        await _apply_grain(video_id, strength)


async def _refresh_derived_renders(video_id: uuid.UUID, video_key: str) -> None:
    """Re-run every post-pass a video already carries, in production order.

    Called when the *source* of those passes changes — attaching or dropping a
    soundtrack rewrites the file both the upscale and the grain read from, so
    leaving them alone would keep serving a rendition built on the old audio.
    Order matters: upscale first, grain on top of it.
    """
    async with AsyncSessionLocal() as db:
        video = await db.get(Video, video_id)
        resolution = video.upscale_resolution if video else None
        rife = (video.upscale_rife if video else None) or 1

    if resolution:
        _progress[video_key] = {
            "phase": "upscaling", "message": "Re-running upscale…", "pct": 40,
        }
        await _apply_upscale(video_id, resolution, rife)
    await _reapply_grain_if_any(video_id, video_key)


async def _run_derived_refresh(video_id: uuid.UUID) -> None:
    """Background wrapper around _refresh_derived_renders for request paths
    that cannot hold a multi-minute re-render open."""
    video_key = str(video_id)
    try:
        await _refresh_derived_renders(video_id, video_key)
        _progress.pop(video_key, None)
    except Exception as exc:
        logger.exception("Derived re-render failed for video=%s", video_id)
        _progress.pop(video_key, None)
        await _persist_post_pass_error(video_id, exc)


async def _persist_post_pass_error(video_id: uuid.UUID, exc: Exception) -> None:
    """Write a failed post-pass onto the row so the frontend poller can see it."""
    msg = str(exc).strip()
    err = f"{type(exc).__name__}: {msg}" if msg else type(exc).__name__
    async with AsyncSessionLocal() as db:
        video = await db.get(Video, video_id)
        if video:
            video.error = err[:1000]
            await db.commit()


def _validate_upscale_target(
    video: Video | None, resolution: int, rife_multiplier: int = 1,
) -> None:
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if video.status != "done" or not video.filename:
        raise HTTPException(status_code=409, detail="Video is not ready (status must be 'done')")
    if not RESOLUTION_MIN <= resolution <= RESOLUTION_MAX:
        raise HTTPException(
            status_code=422,
            detail=f"resolution must be between {RESOLUTION_MIN} and {RESOLUTION_MAX}",
        )
    if rife_multiplier not in RIFE_MULTIPLIERS:
        raise HTTPException(
            status_code=422,
            detail=f"rife_multiplier must be one of {list(RIFE_MULTIPLIERS)}",
        )
    # Interpolation stretches whatever audio the file carries. That is exactly
    # right for a model's own generated track, which was sampled against the
    # pre-RIFE frame count — and exactly wrong for an attached song, where a
    # 3x time-stretch destroys the music rather than re-syncing it.
    if rife_multiplier > 1 and video.soundtrack_song_id:
        raise HTTPException(
            status_code=409,
            detail="Interpolation would time-stretch the attached soundtrack — "
                   "remove the song first, or upscale without RIFE",
        )


@router.get("/jobs/{video_id}/upscale/estimate")
async def estimate_upscale(
    video_id: uuid.UUID,
    resolution: int = 1080,
    rife_multiplier: int = 1,
    db: AsyncSession = Depends(get_db),
):
    """Target size and expected wall-clock for an upscale, before committing.

    This pass runs for minutes per second of footage — long enough that
    starting it blind is a real cost, and the number is cheap to produce
    (two ffprobe calls).
    """
    video = await db.get(Video, video_id)
    _validate_upscale_target(video, resolution, rife_multiplier)
    src = _upscale_source(video)
    if not src.exists():
        raise HTTPException(status_code=409, detail="Source video file is missing on disk")
    return await _upscale_plan_from_file(
        src, resolution, video.width, video.height, rife_multiplier,
    )


@router.post("/jobs/{video_id}/upscale", status_code=202)
async def apply_upscale(
    video_id: uuid.UUID, body: UpscaleApply, db: AsyncSession = Depends(get_db),
):
    video = await db.get(Video, video_id)
    _validate_upscale_target(video, body.resolution, body.rife_multiplier)
    resolution = clamp_resolution(body.resolution)
    rife = clamp_rife(body.rife_multiplier)

    # Clear a stale error from an earlier attempt so the frontend poller can't
    # read it as this attempt failing before the render has even started.
    if video.error is not None:
        video.error = None
        await db.commit()

    _progress[str(video_id)] = {"phase": "upscaling", "message": "Upscaling…", "pct": 5}
    safe_create_task(_run_upscale(video_id, resolution, rife), name=f"upscale:{video_id}")
    return _serialize(video)


@router.delete("/jobs/{video_id}/upscale")
async def remove_upscale(video_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    video = await db.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if video.upscale_filename:
        (settings.videos_dir / video.upscale_filename).unlink(missing_ok=True)
    video.upscale_filename = None
    video.upscale_resolution = None
    video.upscale_rife = None
    await db.commit()
    await db.refresh(video)

    # An existing grain was rendered from the upscaled file, which just went
    # away — re-render it from the small source rather than keep serving a
    # 1080p grained file the row no longer claims to have.
    if video.grain_strength:
        _progress[str(video_id)] = {
            "phase": "graining", "message": "Re-applying grain…", "pct": 10,
        }
        safe_create_task(_run_grain(video_id, video.grain_strength), name=f"grain:{video_id}")
    return _serialize(video)


# ── Per-clip SEEDVR2 upscale (run BEFORE the merge) ───────────────────────────
# Upscaling the merged video makes both halves of this pass work across the
# hard cuts between segments: SEEDVR2 restores several frames jointly, and RIFE
# invents in-between frames from each pair — so at a cut it morphs one shot into
# the next instead of leaving a clean edit. A clip has no cut inside it, so
# upscaling per clip and merging afterwards produces the same resolution with
# the edits intact. Grain still belongs last, on the merged result.


def _clip_key(clip_id: uuid.UUID) -> str:
    """Progress key for a clip pass — namespaced so it cannot collide with the
    job ids that share `_progress`."""
    return f"clip:{clip_id}"


def _clip_upscale_name(clip: VideoClip) -> str:
    return f"{Path(clip.filename).stem}_up.mp4"


def _clip_file(clip: VideoClip) -> Path:
    """The clip as generated — always the upscale's source, never its output."""
    return _segments_dir(clip.video_id) / clip.filename


def _clip_primary_path(clip: VideoClip) -> Path:
    """The rendition of this clip that playback and the merge should use."""
    if clip.upscale_filename:
        return _segments_dir(clip.video_id) / clip.upscale_filename
    return _clip_file(clip)


def _clip_dimensions(clip: VideoClip) -> tuple[int | None, int | None]:
    """Effective size of the rendition `_clip_primary_path` returns."""
    if clip.upscale_filename and clip.upscale_width and clip.upscale_height:
        return clip.upscale_width, clip.upscale_height
    return clip.width, clip.height


def _is_clip_upscaling(clip: VideoClip) -> bool:
    return _progress.get(_clip_key(clip.id), {}).get("phase") == "upscaling"


async def _apply_clip_upscale(
    clip_id: uuid.UUID, resolution: int, rife_multiplier: int = 1,
) -> None:
    """Render the SEEDVR2 pass for one clip and persist it. Raises on failure."""
    rife = clamp_rife(rife_multiplier)
    async with AsyncSessionLocal() as db:
        clip = await db.get(VideoClip, clip_id)
        if not clip:
            raise RuntimeError("Clip row gone")
        src = _clip_file(clip)
        fallback_w, fallback_h = clip.width, clip.height
        out_name = _clip_upscale_name(clip)
        dest = _segments_dir(clip.video_id) / out_name

    if not src.exists():
        raise RuntimeError(f"Clip file missing: {src.name}")
    plan = await _upscale_plan_from_file(src, resolution, fallback_w, fallback_h, rife)

    await _seedvr2_render(
        src, dest,
        resolution=resolution, rife=rife, plan=plan,
        progress_key=_clip_key(clip_id),
        prefix=f"artrium_clipup_{clip_id.hex[:10]}",
        log_subject=f"clip={clip_id}",
    )

    async with AsyncSessionLocal() as db:
        clip = await db.get(VideoClip, clip_id)
        if clip:
            clip.upscale_resolution = resolution
            clip.upscale_rife = rife
            clip.upscale_filename = out_name
            clip.upscale_width = plan["width"]
            clip.upscale_height = plan["height"]
            await db.commit()
    logger.info(
        "Clip upscale applied: clip=%s → %dx%d rife=%dx",
        clip_id, plan["width"], plan["height"], rife,
    )


async def _run_clip_upscale(
    clip_id: uuid.UUID, resolution: int, rife_multiplier: int = 1,
) -> None:
    """Background task behind POST /clips/{id}/upscale."""
    key = _clip_key(clip_id)
    _progress[key] = {"phase": "upscaling", "message": "Upscaling…", "pct": 10}
    try:
        await _apply_clip_upscale(clip_id, resolution, rife_multiplier)
        _progress.pop(key, None)
    except Exception as exc:
        logger.exception("Clip upscale failed for clip=%s", clip_id)
        # Clips have no error column — the phase carries the failure until the
        # client picks it up, which is all a per-clip chip needs.
        _progress[key] = {
            "phase": "failed",
            "message": f"{type(exc).__name__}: {exc}"[:300],
            "pct": 0,
        }


def _validate_clip_upscale(
    clip: VideoClip | None, resolution: int, rife_multiplier: int,
) -> None:
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")
    if not RESOLUTION_MIN <= resolution <= RESOLUTION_MAX:
        raise HTTPException(
            status_code=422,
            detail=f"resolution must be between {RESOLUTION_MIN} and {RESOLUTION_MAX}",
        )
    if rife_multiplier not in RIFE_MULTIPLIERS:
        raise HTTPException(
            status_code=422,
            detail=f"rife_multiplier must be one of {list(RIFE_MULTIPLIERS)}",
        )


@router.get("/clips/{clip_id}/upscale/estimate")
async def estimate_clip_upscale(
    clip_id: uuid.UUID,
    resolution: int = 1080,
    rife_multiplier: int = 1,
    db: AsyncSession = Depends(get_db),
):
    """Target size and expected wall-clock for one clip's upscale."""
    clip = await db.get(VideoClip, clip_id)
    _validate_clip_upscale(clip, resolution, rife_multiplier)
    src = _clip_file(clip)
    if not src.exists():
        raise HTTPException(status_code=409, detail="Clip file is missing on disk")
    return await _upscale_plan_from_file(
        src, resolution, clip.width, clip.height, rife_multiplier,
    )


@router.post("/clips/{clip_id}/upscale", status_code=202)
async def apply_clip_upscale(
    clip_id: uuid.UUID, body: UpscaleApply, db: AsyncSession = Depends(get_db),
):
    """Queue a SEEDVR2 pass for one clip.

    Returns immediately; renders run one at a time behind `_upscale_gate`, so
    queueing a whole stack at once is safe and is the expected way to use this.
    """
    clip = await db.get(VideoClip, clip_id)
    _validate_clip_upscale(clip, body.resolution, body.rife_multiplier)
    resolution = clamp_resolution(body.resolution)
    rife = clamp_rife(body.rife_multiplier)

    if _is_clip_upscaling(clip):
        raise HTTPException(status_code=409, detail="This clip is already being upscaled")

    _progress[_clip_key(clip_id)] = {
        "phase": "upscaling", "message": "Queued…", "pct": 5,
    }
    safe_create_task(
        _run_clip_upscale(clip_id, resolution, rife), name=f"clip_upscale:{clip_id}",
    )
    return _serialize_clip(clip)


@router.get("/clips/{clip_id}/upscale/progress")
async def clip_upscale_progress(clip_id: uuid.UUID):
    """Live phase/step for a clip's upscale. Cheap enough to poll."""
    prog = dict(_progress.get(_clip_key(clip_id), {}))
    if not prog:
        return {"phase": "idle", "message": "", "pct": 0}
    return _attach_live_stage(prog)


@router.delete("/clips/{clip_id}/upscale")
async def remove_clip_upscale(clip_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """Drop a clip's upscale and go back to the generated rendition."""
    clip = await db.get(VideoClip, clip_id)
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")
    if clip.upscale_filename:
        (_segments_dir(clip.video_id) / clip.upscale_filename).unlink(missing_ok=True)
    clip.upscale_filename = None
    clip.upscale_resolution = None
    clip.upscale_rife = None
    clip.upscale_width = None
    clip.upscale_height = None
    await db.commit()
    _progress.pop(_clip_key(clip_id), None)
    await db.refresh(clip)
    return _serialize_clip(clip)


# ── Film grain (post-hoc pass over a finished video) ──────────────────────────
# Wan2.2 output is clean to the point of looking plastic. The grain pass is a
# derived sibling file like the soundtrack mux, never an overwrite: `filename`
# stays pristine so any strength can be tried, undone, or re-tried. It runs
# last, on top of any upscale, so grain sits at the delivery resolution.

class GrainApply(BaseModel):
    strength: int  # 1–100 UI scale; validated in the endpoint


def _grain_source(video: Video) -> Path:
    """The ungrained file a grain pass should read.

    Deliberately never `grain_filename` itself: re-grading always starts from
    a clean source, so moving the slider replaces the grain instead of baking
    a second pass on top of the first. Otherwise the most complete rendition
    wins — the upscale when there is one, else the muxed (audio-bearing)
    variant so an attached soundtrack survives the re-encode.
    """
    if video.upscale_filename:
        return settings.videos_dir / video.upscale_filename
    if video.muxed_filename:
        return settings.videos_dir / video.muxed_filename
    return settings.storage_dir / video.filepath


def _grain_name(video_id: uuid.UUID) -> str:
    return f"{video_id}_grain.mp4"


def _grain_preview_name(video_id: uuid.UUID) -> str:
    return f"{video_id}_grainprev.mp4"


async def _apply_grain(video_id: uuid.UUID, strength: int) -> None:
    """Render the grain pass from the ungrained source and persist it.

    Raises on failure; each caller decides how to report. Also used to
    re-render after a soundtrack change, since that swaps the source file
    underneath an existing grain.
    """
    async with AsyncSessionLocal() as db:
        video = await db.get(Video, video_id)
        if not video or not video.filepath:
            raise RuntimeError("Video row gone or has no file")
        src = _grain_source(video)

    if not src.exists():
        raise RuntimeError(f"Source file missing: {src.name}")

    out_name = _grain_name(video_id)
    await render_grain(
        src, settings.videos_dir / out_name, strength,
        ffmpeg_path=settings.ffmpeg_path,
    )

    async with AsyncSessionLocal() as db:
        video = await db.get(Video, video_id)
        if video:
            video.grain_strength = strength
            video.grain_filename = out_name
            video.error = None
            await db.commit()
    logger.info("Grain applied: video=%s strength=%d → %s", video_id, strength, out_name)


async def _run_grain(video_id: uuid.UUID, strength: int) -> None:
    """Background task behind POST /jobs/{id}/grain — a full re-encode of a
    16-30s clip runs well past a request's patience, so it is polled like the
    soundtrack mux rather than awaited inline."""
    video_key = str(video_id)
    _progress[video_key] = {"phase": "graining", "message": "Adding grain…", "pct": 50}
    try:
        await _apply_grain(video_id, strength)
        _progress.pop(video_key, None)
    except Exception as exc:
        logger.exception("Grain render failed for video=%s strength=%s", video_id, strength)
        _progress.pop(video_key, None)
        await _persist_post_pass_error(video_id, exc)


def _validate_grain_target(video: Video | None, strength: int) -> None:
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if video.status != "done" or not video.filename:
        raise HTTPException(status_code=409, detail="Video is not ready (status must be 'done')")
    if not 1 <= strength <= 100:
        raise HTTPException(status_code=422, detail="strength must be between 1 and 100")


@router.post("/jobs/{video_id}/grain/preview")
async def preview_grain(
    video_id: uuid.UUID, body: GrainApply, db: AsyncSession = Depends(get_db),
):
    """Grade a few seconds out of the middle of the clip and return its URL.

    Runs inline: the point is a tight look-adjust-look loop, and a 4s excerpt
    encodes in a second or two. The file is overwritten on every call, so the
    caller must cache-bust the URL it gets back.
    """
    video = await db.get(Video, video_id)
    _validate_grain_target(video, body.strength)

    src = _grain_source(video)
    if not src.exists():
        raise HTTPException(status_code=409, detail="Source video file is missing on disk")

    out_name = _grain_preview_name(video_id)
    try:
        seconds = await render_grain_preview(
            src, settings.videos_dir / out_name, body.strength,
            ffmpeg_path=settings.ffmpeg_path,
        )
    except Exception as exc:
        logger.exception("Grain preview failed for video=%s", video_id)
        raise HTTPException(status_code=502, detail=f"Preview failed: {exc}")

    return {
        "url": f"/api/video/file/{out_name}",
        "strength": body.strength,
        "seconds": round(seconds, 2),
    }


@router.post("/jobs/{video_id}/grain", status_code=202)
async def apply_grain(
    video_id: uuid.UUID, body: GrainApply, db: AsyncSession = Depends(get_db),
):
    video = await db.get(Video, video_id)
    _validate_grain_target(video, body.strength)

    # Clear a stale error from an earlier attempt so the frontend poller can't
    # read it as this attempt failing before the render has even started.
    if video.error is not None:
        video.error = None
        await db.commit()

    _progress[str(video_id)] = {"phase": "graining", "message": "Adding grain…", "pct": 10}
    safe_create_task(_run_grain(video_id, body.strength), name=f"grain:{video_id}")
    return _serialize(video)


@router.delete("/jobs/{video_id}/grain")
async def remove_grain(video_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    video = await db.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if video.grain_filename:
        (settings.videos_dir / video.grain_filename).unlink(missing_ok=True)
    (settings.videos_dir / _grain_preview_name(video_id)).unlink(missing_ok=True)
    video.grain_filename = None
    video.grain_strength = None
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

def _is_graining(v: Video) -> bool:
    """Whether a grain render is in flight for this video right now.

    Re-applying the *same* strength leaves `grain_strength` unchanged, so a
    client watching that field alone would call the job finished on its first
    poll — while ffmpeg is still writing the file. This is the signal that
    actually flips.
    """
    return _progress.get(str(v.id), {}).get("phase") == "graining"


def _is_upscaling(v: Video) -> bool:
    """Whether an upscale render is in flight for this video right now.

    Same reasoning as _is_graining: re-running at the *same* resolution leaves
    `upscale_resolution` unchanged, so a client watching that field alone would
    call the job finished on its first poll — while ComfyUI is still sampling.
    """
    return _progress.get(str(v.id), {}).get("phase") == "upscaling"


def _render_version(v: Video, primary_name: str | None) -> str | None:
    """Cache-busting token for the derived file currently being served.

    Post-processing reuses one filename per video (`{id}_grain.mp4`,
    `{id}_muxed.mp4`), so re-rendering replaces the bytes behind a URL without
    changing the URL itself. /api/video/file sends an ETag but no
    Cache-Control, which leaves browsers (and Cloudflare's edge cache for
    static extensions) caching heuristically — roughly 10% of the file's age,
    so days for anything not brand new — and replaying the *first* render
    without ever revalidating. A re-applied grain then looked like it had
    never been applied.

    Keying the URL on the file's mtime gives every distinct render its own
    URL, while letting an unchanged one stay cached. The original is written
    once and never rewritten, so it needs no token.
    """
    if not primary_name or primary_name == v.filename:
        return None
    try:
        return str((settings.videos_dir / primary_name).stat().st_mtime_ns)
    except OSError:
        return None


def _serialize(v: Video) -> dict:
    # Derived variants take precedence in the order they are produced —
    # soundtrack, then upscale, then grain — each pass reading the one before
    # it, so the last one present is always the most complete rendition. The
    # clean original stays available via `original_url`.
    # services/instagram/media.py::resolve_video_path mirrors this precedence
    # — keep the two in step.
    primary_name = v.grain_filename or v.upscale_filename or v.muxed_filename or v.filename
    version = _render_version(v, primary_name)
    primary_url = f"/api/video/file/{primary_name}" if primary_name else None
    if primary_url and version:
        primary_url = f"{primary_url}?v={version}"
    return {
        "id":                str(v.id),
        "status":            v.status,
        "workflow":          v.workflow or "flf2v",
        "filename":          v.filename,
        "url":               primary_url,
        "original_url":      f"/api/video/file/{v.filename}" if v.filename else None,
        "soundtrack_song_id": str(v.soundtrack_song_id) if v.soundtrack_song_id else None,
        "muxed_filename":    v.muxed_filename,
        "has_soundtrack":    bool(v.muxed_filename),
        "upscale_resolution": v.upscale_resolution,
        "upscale_rife":      v.upscale_rife,
        "has_upscale":       bool(v.upscale_filename),
        "upscale_rendering": _is_upscaling(v),
        "grain_strength":    v.grain_strength,
        "has_grain":         bool(v.grain_filename),
        "grain_rendering":   _is_graining(v),
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
