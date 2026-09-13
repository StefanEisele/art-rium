"""
Key-frame video generation — four workflow types, all producing per-segment
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

               Wan 2.2 has no first-last-frame checkpoint of its own, so this
               wants the Wan2.2-Fun-InP expert pair; on the plain i2v pair the
               end frame is an out-of-distribution constraint and the clip
               cross-fades instead of moving. See _UNET_FUN_HIGH.

  minimax_flf  The same per-PAIR shape on MiniMax H3, whose checkpoint IS a
               first-last model (fl2va) and which therefore needs no extra
               weights. Native audio, 2–7 images, same suggest-transitions
               endpoint.

Both Wan workflows share one sampler, and it has two dials the request carries:
`steps` (4-16) buys resolved detail, and `lora_high` (0.0-1.0) is the distill
strength on the high-noise expert — the expert that plans motion, which at full
strength plans almost none. The high/low handover is derived from the sigma
schedule rather than fixed at half the steps; services/comfy/wan_moe.py has the
arithmetic. MiniMax H3 ignores both and runs its own recipe.

A job is "done" when all of its clips are rendered — there is no per-job
final file anymore. Final videos are created by merging clips (from any
number of jobs, in any order, mixed workflows allowed) via POST
/api/video/merge, which normalizes resolution/fps/audio and re-encodes into
a new Video row with workflow="merge". A merge source may also be a finished
VIDEO rather than a library clip — that is how two finished pieces (each with
its own soundtrack, upscale or grain) are joined into a longer one, and how a
merge gets extended without re-picking all of its clips. Sources can
optionally be deleted after a successful merge.

The SEEDVR2 upscale exists at both levels, and which one to use is a real
choice: per CLIP (before merging) keeps the restorer and RIFE inside one
continuous shot, per VIDEO runs them across the cuts a merge introduced,
which shows up as morphing at the edits. Upscale the clips, merge, then
grain the result.

POST /api/video/generate            → enqueue job, return {video_id}
POST /api/video/suggest-transitions → VLM-suggested per-transition prompts (flf2v/minimax_flf)
POST /api/video/suggest-i2v         → VLM-suggested surreal per-image prompts (i2v/minimax)
GET  /api/video/jobs/{id}           → poll status
GET  /api/video/jobs/{id}/progress  → lightweight progress (ComfyUI queue + phase)
POST /api/video/jobs/{id}/cancel    → stop the render, keep the clips it has
GET  /api/video/clips               → all library clips (frontend groups by job)
DELETE /api/video/clips/{clip_id}   → delete one clip (empty source jobs are pruned)
POST /api/video/clips/{id}/upscale  → SEEDVR2 pass on ONE clip, before merging
DELETE /api/video/clips/{id}/upscale→ drop it again
POST /api/video/merge               → concat chosen clips + finished videos into a new video
GET  /api/video/thumb/{id}          → first-frame JPEG thumbnail
GET  /api/video/file/{fname}        → serve MP4
GET  /api/videos                    → list all videos
DELETE /api/video/{id}              → delete a video/job (cascades to its clips)
"""
import asyncio
import logging
import random
import re
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import httpx
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.auth import require_auth
from core.comfy import WORKFLOW_NAME as ZIMAGE_WORKFLOW_NAME
from core.config import settings
from core.db import AsyncSessionLocal, get_db
from core.imaging import prepare_jpg_for_web
from core.job_control import CANCELLED_MSG, cancel_job
from core.loras import ALLOWED_LORAS, DEFAULT_LORA, LORAS
from core.models import AUDIO_WORKFLOWS, Image, Song, Video, VideoClip
from core.subproc import communicate
from core.tasks import safe_create_task
from core.video_thumb import (
    make_video_thumbnail,
    probe_has_audio,
    probe_video_dimensions,
    probe_video_duration,
    probe_video_frames,
)
from services.comfy.client import (
    loader_choices,
    free_memory,
    poll_history,
    post_workflow,
    queue_info,
    upload_image,
)
from services.comfy.ingest import ingest_comfy_image
from services.comfy.progress import attach_live_stage as _attach_live_stage
from services.comfy.wan_moe import I2V_BOUNDARY, moe_split_step
from services.comfy.wan_moe import SCHEDULER as WAN_SCHEDULER
from services.comfy.wan_transition_loras import offered as offered_transition_loras
from services.comfy.wan_transition_loras import with_lora_trigger
from services.comfy.vram import free_vram_for
from services.comfy.zimage import ZIMAGE_SAVE_NODE, build_zimage_workflow
from workers.comfy_listener import get_listener
from services.ollama.analysis import (
    generate_i2v_motion_prompts,
    generate_minimax_motion_prompts,
    generate_minimax_transition_prompts,
    generate_transition_prompts,
)
from services.ollama.story_frames import (
    describe_image_for_story,
    generate_story_frame_prompts,
)
from services.ollama.zimage_enhance import get_zimage_style_block
from services.video.audio_stretch import stretch_audio_to_video, stretch_native_audio
from services.video.look import (
    DEFAULT_PRESET,
    PRESET_BY_KEY,
    Look,
    clamp_strength as clamp_look_strength,
    preset_options,
    render_look,
    render_look_preview,
)
from services.video.merge import MergeInput, merge_clips
from services.video.audio_bed import BED_VOLUME_DEFAULT, clamp_bed_volume
from services.video.soundtrack import mux_soundtrack
from services.video.upscale import (
    FPS_PRESETS,
    clamp_fps,
    plan_frame_rate,
    RESOLUTION_DEFAULT,
    RESOLUTION_KEEP,
    RESOLUTION_MAX,
    RESOLUTION_MIN,
    RIFE_MULTIPLIERS,
    build_upscale_workflow,
    clamp_resolution,
    clamp_rife,
    estimate_seconds,
    needs_comfy,
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

# Wan's own stock negative, plus 慢动作 ("slow motion"). Worth stating plainly:
# on the distilled path this string is never scored. cfg is 1, and at cfg 1
# ComfyUI skips the unconditional pass entirely — that is exactly what makes a
# 4-to-16-step render affordable. The slow-motion cure therefore lives in
# `lora_high` below, not here; this line only starts working if someone raises
# cfg above 1, and it is written so that it would be right when they do.
_NEG_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走，慢动作"
)
_CLIP_NAME  = "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
_UNET_HIGH  = "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"
_UNET_LOW   = "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"
_LORA_HIGH  = "wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors"
_LORA_LOW   = "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors"

# ── The right weights for a transition ───────────────────────────────────────
# Wan 2.2 shipped three checkpoints: T2V-A14B, I2V-A14B and TI2V-5B. None of
# them is a first-last-frame model — unlike Wan 2.1, which had a dedicated
# FLF2V-14B-720P. What `WanFirstLastFrameToVideo` does is pin the last pixel
# frame in the concat latent and unmask everything between, and I2V-A14B was
# trained with that mask covering only the *first* frame. The end constraint is
# therefore out of distribution: the model has no learned prior for how to
# arrive there, so it satisfies the boundary the cheapest way available to it
# and blends. That is the fade, and no number of steps fixes a missing prior.
#
# Alibaba-PAI's Wan2.2-Fun-A14B-InP is the model that *was* trained on start
# and end frames, repackaged by Comfy-Org as these two files. Same MoE pair,
# same architecture, same UMT5 encoder, same VAE, and the same lightx2v 4-step
# distill LoRAs apply — so it is a weight swap and nothing else, which is why
# the graph below only chooses a file name.
#
#   https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged
#     split_files/diffusion_models/wan2.2_fun_inpaint_high_noise_14B_fp8_scaled.safetensors
#     split_files/diffusion_models/wan2.2_fun_inpaint_low_noise_14B_fp8_scaled.safetensors
#
# 14.3 GB each. If they are not in ComfyUI's models directory the transition
# builder falls back to the i2v pair and says so in the log — a slower, fading
# render is still better than refusing to render.
_UNET_FUN_HIGH = "wan2.2_fun_inpaint_high_noise_14B_fp8_scaled.safetensors"
_UNET_FUN_LOW  = "wan2.2_fun_inpaint_low_noise_14B_fp8_scaled.safetensors"
_VAE_NAME   = "wan_2.1_vae.safetensors"
_RIFE_CKPT  = "rife49.pth"

# ── Wan 2.2 sampler tuning (shared by i2v_multi and flf2v) ───────────────────
# Both Wan builders run the distilled Lightning path: cfg=1, so every step is a
# single model evaluation and the negative prompt is never scored. That part is
# unchanged — it is what makes 14B affordable on a 16 GB 4060 Ti at all.
#
# Two things around it were wrong, and both cost visible quality:
#
# 1. FOUR STEPS IS NOT ENOUGH. It is the floor the distill LoRA makes *possible*,
#    not the point where it looks good. Detail — grass, fabric, the ridges in a
#    paint pour — resolves between 6 and 10 steps and keeps improving to ~16.
#    Steps are not cheap, and it is worth being exact about the price rather
#    than hoping the cold model loads dominate. Measured 2026-08-26 on the
#    16 GB 4060 Ti — 960x960 x 49 frames, RIFE 3, one clip per submission with
#    a `free_memory` and a cold reload before each:
#
#         4 steps   356 s   1.00x
#         6 steps   496 s   1.39x
#         8 steps   647 s   1.82x
#        10 steps   797 s   2.24x
#
#    Dead straight: ~74 s per sampler step against ~58 s for everything else
#    put together — the two cold 14 GB UNET loads, both VAE passes, RIFE and
#    the h265 encode. Sampling is ~84% of a 6-step render, so the step count
#    really does multiply the wait, and the 4 → 6 default move costs ~39%.
#    See `_WAN_STEPS_DEFAULT` for why it is worth paying, and
#    `estimate_wan_seconds` for what the numbers are used for.
#
# 2. THE HIGH/LOW SPLIT WAS HARD-CODED AT HALF. Wan 2.2 hands over between its
#    two experts at a fixed diffusion *timestep*, not at a fixed step index —
#    services/comfy/wan_moe.py has the arithmetic and the reference. Splitting
#    at steps//2 gives the high-noise expert more of the schedule than the model
#    was trained to, and the high-noise expert is the one that decides motion.
#
# cfg stays at 1 on purpose, and that decision is already tested: an 8-step
# asymmetric variant (weakened distill + cfg 3 on the high-noise expert, per
# https://huggingface.co/lightx2v/Wan2.2-Lightning/discussions/5) improved
# flf2v's end-frame adherence but doubled the render time per transition, and
# was reverted 2026-07-09. Raising `steps` costs one model evaluation per extra
# step; raising cfg costs two per step across the whole schedule. When more
# quality is wanted, buy steps.
_WAN_SHIFT = 5.0     # ModelSamplingSD3; the recommended i2v range is 5-8
_WAN_CFG   = 1       # distilled path: guidance-free, and it has to be

# ── Guidance for an expert that no longer carries the distill ────────────────
# cfg=1 is not a preference, it is what a distill LoRA requires: the LoRA bakes
# the guidance in, and scoring a negative on top of it double-counts. But the
# motion dial can take the distill *off* the high-noise expert entirely
# (lora_high = 0), and at that point cfg=1 stops being right and starts being a
# bug — a plain Wan expert running guidance-free. The block above _WAN_LORA_HIGH
# _DEFAULT said so from the day it was written: "those first steps run an
# undistilled expert guidance-free, which is not what the base model expects".
#
# What it costs the picture is precisely what was missing here. With no
# guidance the prompt steers only through the conditional path, so an
# instruction to *transform* barely registers and the model takes the cheapest
# route between two pinned frames — it cross-fades. And the negative prompt is
# never scored at all, which matters more than it sounds: _NEG_PROMPT ends in
# 慢动作 (slow motion) and 静止不动的画面 (static image), the two things a
# transition must not be. At cfg=1 those words are decoration.
#
# Applied to the HIGH-noise pass only. The low-noise expert keeps its distill at
# full strength (see _WAN_LORA_LOW), so it keeps cfg=1 for the same reason the
# distilled path always did.
#
# Cost is bounded and small, because it lands only on the steps the high-noise
# expert actually runs: 2 model evaluations per step instead of 1, over
# `split` of `steps`. At 10 steps and shift 5 the split is 3, so 13 evaluations
# instead of 10 — about +30%, not the +100% that raising cfg across the whole
# schedule would cost. That is why the 2026-07-09 experiment (weakened distill
# + cfg on high) was recorded as doubling the time and reverted: it raised cfg
# while the distill was still partly on, so it paid for both.
#
# 3.0 is the value that experiment used and the one it found improved end-frame
# adherence. estimate_wan_seconds and wan_poll_timeout both know about the
# extra evaluations, so the ETA next to the chips stays honest.
_WAN_CFG_HIGH_UNDISTILLED = 3.0


def wan_cfg_high(lora_high: float) -> float:
    """Guidance for the high-noise pass, given its distill strength.

    Two regimes, not a slider: with any distill on the expert the model wants
    cfg 1, and with none it wants real guidance. There is no useful middle,
    which is why this reads the motion dial instead of adding a second one.
    """
    return _WAN_CFG if lora_high > 0 else _WAN_CFG_HIGH_UNDISTILLED

# Total sampler steps across both experts. 6 is the smallest count that is
# honestly watchable; the UI offers 4-16 and this is only the fallback for a
# caller that does not say.
_WAN_STEPS_DEFAULT = 6
_WAN_STEPS_MIN, _WAN_STEPS_MAX = 4, 16

# ── The motion dial ──────────────────────────────────────────────────────────
# Strength of the 4-step lightx2v distill LoRA on the HIGH-noise expert, and the
# single setting that decides whether a clip moves. At 1.0 the distill dominates
# the expert that plans motion, and Wan 2.2 comes back in slow motion — smoke
# that hangs, paint that barely creeps, a camera drifting through treacle. It is
# the most-reported complaint about this model and it is not a prompt problem.
#
# Lowering it restores travel. Zero removes the distill from the high-noise pass
# entirely — the graph then drops the LoRA node rather than loading a file to
# multiply it by nothing — and gives the most motion. It is not free: cfg stays
# at 1, so those first steps run an *undistilled* expert guidance-free, which is
# not what the base model expects, and composition wants more steps to settle.
# The LOW-noise expert keeps the distill at full strength in every case — that
# is where the detail comes from, and weakening it costs quality without buying
# any movement.
#
# 0.5 is the default because it is the configuration the reference comparison
# was shot at (high 0.5 / low 1.0), not because it was tuned here.
#
#   0.0   most motion, needs 8+ steps to stay clean
#   0.4   lively; the video's own preferred setting for cars and smoke
#   0.5   default here — clearly moving, still stable at 6 steps
#   1.0   the old behaviour: stable, detailed, and barely moving
_WAN_LORA_HIGH_DEFAULT = 0.5
_WAN_LORA_LOW          = 1.0   # never lowered; see above

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
POLL_TIMEOUT  = 1800  # 30 minutes — the floor, and what a non-Wan job still gets

# ── What a Wan clip costs ────────────────────────────────────────────────────
# A flat 30-minute budget was safe while every Wan render was four steps. Steps
# are a setting now, up to 16, and they multiply against a canvas that already
# reaches 1920x1088 and 81 frames — three factors that used to be one. A 16-step
# 1920x1088 clip would sail past 1800 s and be killed mid-render with the file
# already written, which is the worst way to lose a job.
#
# Least-squares fit over the four measurements in the sampler-tuning block
# above (960x960, 49 frames, RIFE 3): 73.7 s per step, 58.0 s fixed, and every
# point within 1% of the line.
#
# SageAttention changes the per-step term and nothing else — it touches only
# the attention inside a sampler step, not the model loads, the VAE, RIFE or
# the encode. So there is one sampling coefficient per attention backend and
# one shared coefficient for the rest, and the Sage slope was fitted holding
# that shared 58 s fixed rather than given its own intercept, which two points
# cannot honestly separate:
#
#     Sage   6 steps  measured 326 s   model 331 s  (+1.6%)
#     Sage  10 steps  measured 517 s   model 513 s  (-0.6%)
#
# 45.5 s per step against SDPA's 73.7 — 1.62x on the sampling itself, from a
# 3.0x attention kernel, which puts attention at roughly half of a Wan step.
#
# All three coefficients are per pixel-frame (width x height x frames), and
# THAT scaling is an assumption rather than a measurement — only the step count
# was varied, at one canvas and one RIFE factor. It is good enough for what it
# is used for: a deadline with 3x headroom, and a rough "about N minutes" next
# to the chips so an hour-long job announces itself before it is queued. It is
# deliberately not sold as more than that.
_WAN_SEC_PER_STEP_PF      = 1.632e-6   # sampling per step — PyTorch SDPA
_WAN_SEC_PER_STEP_PF_SAGE = 1.008e-6   # sampling per step — SageAttention 2.2
_WAN_SEC_POST_PF          = 1.284e-6   # model loads, VAE, RIFE 3, h265 encode


def wan_model_evals(steps: int, lora_high: float | None = None) -> int:
    """Model evaluations one clip costs — the thing the clock actually tracks.

    Usually one per step. But an undistilled high-noise expert runs with real
    guidance (see wan_cfg_high), and a guided step is two evaluations: the
    conditional and the unconditional. Only the high-noise portion is guided,
    so the surcharge is the split, not the step count.
    """
    steps = max(1, int(steps))
    lh = _WAN_LORA_HIGH_DEFAULT if lora_high is None else lora_high
    if wan_cfg_high(lh) <= 1:
        return steps
    return steps + moe_split_step(steps, _WAN_SHIFT, I2V_BOUNDARY)


def estimate_wan_seconds(
    width: int, height: int, frames: int, steps: int, *,
    sage: bool | None = None, lora_high: float | None = None,
) -> int:
    """Roughly how long one Wan clip takes on this machine.

    `sage` defaults to whatever the graph builder will actually do, so callers
    that just want a number do not have to know the setting exists. Pass it
    explicitly only to price the other backend. `lora_high` matters for the
    same reason it matters to the picture: at 0 the high-noise pass is guided
    and each of its steps costs two model evaluations.
    """
    if sage is None:
        sage = settings.wan_sage_attention
    per_step = _WAN_SEC_PER_STEP_PF_SAGE if sage else _WAN_SEC_PER_STEP_PF
    pixel_frames = max(1, width) * max(1, height) * max(1, frames)
    evals = wan_model_evals(steps, lora_high)
    return max(1, round(pixel_frames * (per_step * evals + _WAN_SEC_POST_PF)))


def wan_poll_timeout(
    width: int, height: int, frames: int, steps: int, lora_high: float | None = None,
) -> int:
    """Deadline for one Wan submission — three times the estimate, never under
    the old flat budget. Mirrors what the upscale path already does for the same
    reason: a pass whose length is a user setting cannot share one constant."""
    return max(POLL_TIMEOUT,
               estimate_wan_seconds(width, height, frames, steps, lora_high=lora_high) * 3)

# ── Pydantic ──────────────────────────────────────────────────────────────────

class GenerateVideoRequest(BaseModel):
    image_ids: list[uuid.UUID]
    workflow: str = "i2v_multi"    # see GENERATE_WORKFLOWS
    width:  int = 1088
    height: int = 1088
    frame_count: int = 49          # fallback frame count when prompts/frame_counts arrays are absent
    fps:    int = 24               # legacy: no builder reads it any more. MiniMax is
                                   # fixed at MINIMAX_FPS and the Wan builders derive
                                   # their rate from WAN_NATIVE_FPS x rife_multiplier.
    prompt: str = ""               # fallback prompt when `prompts` is absent/mismatched length
    prompts: list[str] = []        # per-image mode: one per image; transition mode: one per pair (n-1)
    frame_counts: list[int] = []   # per-image mode: one per image; transition mode: one per pair (n-1)
    rife_multiplier: int = 3       # RIFE VFI frame interpolation factor (2/3/4; MiniMax also allows 1 = off)
    # An optional transition/metamorphosis LoRA for the high-noise expert —
    # the slot the community's morph LoRAs are trained for. Name as ComfyUI
    # lists it; None means the graph has no such node at all.
    style_lora: str | None = None
    style_lora_strength: float = 1.0
    # Wan-only sampler controls (WAN_WORKFLOWS; MiniMax runs its own
    # fixed recipe). Both are clamped by clamp_wan_steps / clamp_lora_high
    # rather than validated here, so an older cached frontend that omits them
    # simply gets the new defaults instead of a 422.
    steps: int = _WAN_STEPS_DEFAULT          # total sampler steps across both experts (4–16)
    lora_high: float = _WAN_LORA_HIGH_DEFAULT  # high-noise distill strength; LOWER = more motion
    pingpong: bool = False         # i2v_multi: VHS_VideoCombine pingpong (boomerang) flag; unused by flf2v
    end_on_keyframe: bool = False  # flf2v: append the raw end key frame after the diffused clip (pixel-exact landing, but reads as a cut when diffusion undershoots)


class MergeItem(BaseModel):
    """One entry of a merge selection: a library clip or a finished video."""
    kind: Literal["clip", "video"] = "clip"
    id: uuid.UUID


class MergeRequest(BaseModel):
    # The ordered selection to concatenate, in playback order. `items` is the
    # current shape and may mix library clips with finished videos; `clip_ids`
    # is the clip-only shape a service-worker-cached older frontend still
    # sends, and is read only when `items` is absent.
    items: list[MergeItem] = []
    clip_ids: list[uuid.UUID] = []
    delete_sources: bool = False   # delete the sources (and empty source jobs) after a successful merge

    def sources(self) -> list[MergeItem]:
        return self.items or [MergeItem(kind="clip", id=cid) for cid in self.clip_ids]


# ── Wan 2.2 expert pair (shared by both Wan builders) ────────────────────────

def clamp_wan_steps(steps: int | None) -> int:
    """Sampler steps, held inside the range this card can actually finish."""
    if steps is None:
        return _WAN_STEPS_DEFAULT
    return max(_WAN_STEPS_MIN, min(_WAN_STEPS_MAX, int(steps)))


def clamp_lora_high(strength: float | None) -> float:
    """High-noise distill strength — the motion dial, 0.0 (most) to 1.0 (least)."""
    if strength is None:
        return _WAN_LORA_HIGH_DEFAULT
    return max(0.0, min(1.0, float(strength)))


def clamp_style_strength(value: float | None) -> float:
    """Strength for a transition/metamorphosis LoRA on the high-noise expert.

    Allowed above 1.0 because these LoRAs are routinely pushed there — the
    published guidance for the ones this slot exists for says 1.0 is the
    *strongest documented* setting, not a ceiling the format imposes — but
    stopped at 2.0, past which Wan reliably falls apart rather than trying
    harder.
    """
    if value is None:
        return 1.0
    return max(0.0, min(2.0, float(value)))


# ── The rate Wan actually animates at ────────────────────────────────────────
# Wan 2.2's A14B experts are trained at 16 fps. That is not a preference, it is
# what one generated frame *means*: 49 frames is 3.06 seconds of motion, and no
# sampler setting changes it. ComfyUI's own bundled template says the same in
# one number — video_wan2_2_14B_i2v.json writes `CreateVideo [16]`, against the
# 5B TI2V template's 24.
#
# This is where the transitions were being lost. RIFE multiplies the frame
# count; if the container rate does not rise by the same factor, the extra
# frames stretch the clip instead of smoothing it. Writing 3x the frames at a
# fixed 24 fps played 16 fps content at 8 — **every Wan clip this tool has
# rendered ran at half speed**, and a first-to-last-frame transition at half
# speed is indistinguishable from a cross-fade. It is also why more steps never
# helped: steps buy detail, and this was never a detail problem.
#
# services/video/upscale.py::plan_frame_rate already states the rule for the
# upscale path — "the clip keeps its duration; RIFE multiplies the frames, so
# the rate has to rise by the same factor or the picture turns into slow
# motion". The generation path simply never applied it. It does now, and the
# fps selector is no longer offered for the Wan builders: 16 x RIFE is the only
# rate at which the model's own motion plays at the speed it was sampled for.
WAN_NATIVE_FPS = 16

async def _fun_inp_available() -> bool:
    """Whether ComfyUI is offering both Fun-InP expert files.

    Asked through ComfyUI's own loader enum rather than by looking on disk:
    this process does not know where ComfyUI keeps its models (extra_model_paths
    can put them anywhere), and the enum is exactly what the submit-time
    validator will check the workflow against.
    """
    names = await loader_choices("UNETLoader", "unet_name")
    return _UNET_FUN_HIGH in names and _UNET_FUN_LOW in names


# ── Workflow families ────────────────────────────────────────────────────────
# Four generation modes over two model families and two shapes. The shape
# decides what a "clip" is: a per-image mode animates each picture on its own,
# a transition mode animates the gap between adjacent pictures and so yields
# one clip fewer. The family decides the sampler, the frame grid and the rate.
#
#   i2v_multi     Wan 2.2      per image        silent
#   flf2v         Wan 2.2      per transition   silent
#   minimax_i2v   MiniMax H3   per image        native audio
#   minimax_flf   MiniMax H3   per transition   native audio
#
# minimax_flf costs nothing to have: the checkpoint on disk is already the
# fl2va ("first-last to video+audio") one, and MiniMaxH3ImageToVideo takes an
# optional `last_frame` this builder simply never filled in.
WAN_WORKFLOWS       = frozenset({"i2v_multi", "flf2v"})
MINIMAX_WORKFLOWS   = frozenset({"minimax_i2v", "minimax_flf"})
TRANSITION_WORKFLOWS = frozenset({"flf2v", "minimax_flf"})
GENERATE_WORKFLOWS  = WAN_WORKFLOWS | MINIMAX_WORKFLOWS


def align_wan_length(frame_count: int) -> int:
    """Snap a frame count up onto Wan's 4n+1 temporal grid.

    The Wan VAE compresses time by 4 with the first frame standing alone, so a
    clip is 1 + 4n frames and every Wan node in ComfyUI declares
    `Int.Input("length", default=81, step=4)` — its own UI cannot express an
    off-grid length. This tool's number field used step 1 and could, which is
    how a job went out at 48: legal enough to render, but not the shape the
    model was trained on, and on a transition it is the *end* frame that sits
    on the ragged edge.

    Snapped up rather than down, matching align_minimax_length: a caller asking
    for a length gets at least that much clip.
    """
    n = max(5, int(frame_count or 0))
    return n + (-(n - 1) % 4)


def wan_output_fps(rife_multiplier: int | None) -> int:
    """Container rate for a Wan clip that has been RIFE-interpolated by N.

    16 -> 32 -> 48 -> 64 for N = 1..4. All of them are legal mp4 rates, and
    everything downstream that cares (the merge, the beat cut, the Instagram
    reel concat) already conforms mixed rates to one.
    """
    return WAN_NATIVE_FPS * max(1, min(4, int(rife_multiplier or 1)))


# ── Attention backend ────────────────────────────────────────────────────────
# Wan's DiT spends most of a step in self-attention over a very long sequence —
# 960x960 x 49 frames is ~46k tokens — which is exactly where an approximate
# attention kernel pays. SageAttention 2.2 quantises Q/K to INT8 and the PV
# accumulation to FP8, on kernels compiled for this card's sm_89.
#
# Measured 2026-08-26 in ComfyUI's own venv, one Wan-shaped attention call
# (1 x 40 heads x 4096 x 128, fp16): PyTorch SDPA 12.21 ms, Sage 4.07 ms —
# 3.0x — at a relative L2 error of 0.038 against SDPA's output. End to end on
# the real graph: 6 steps 496 s -> 326 s (1.52x), 10 steps 797 s -> 517 s.
#
# WHAT THAT ERROR DOES IS WORTH BEING PRECISE ABOUT, because "3.8%" invites the
# wrong conclusion. It does not make the clip 3.8% worse. It makes it a
# DIFFERENT CLIP. Same image, same seed, same prompt, 6 steps, compared frame
# by frame: SSIM 0.99 at frame 1, decaying smoothly to 0.68 by frame 145. The
# first frame is the source photo either way; from there the tiny per-step
# perturbation compounds and the two samples walk apart. Looking at frame 120
# of each, neither is degraded — same detail, same palette, no artefacts — the
# paint has simply gone somewhere else. services/comfy/vace.py records the same
# lesson from the other direction: a seed only means something relative to the
# exact sampler that consumed it.
#
# That is also the whole reason this is scoped to the two Wan builders rather
# than switched on globally with ComfyUI's --use-sage-attention flag. Every
# other model in this project (MiniMax, Z-Image, SEEDVR2, ACE-Step, VACE) was
# calibrated against exact attention, and a global flag would silently re-roll
# all of them. `settings.wan_sage_attention` turns it off without touching code
# — but note that turning it off does not restore an earlier clip either; it
# just picks the other trajectory.
def _attention_backend(nodes: dict, p: str, tag: str, model_node: str) -> str:
    """Route one expert through SageAttention. Returns the node to sample from.

    A no-op that returns `model_node` unchanged when the setting is off, so the
    graph is exactly the pre-SageAttention one and nothing needs installing.
    """
    if not settings.wan_sage_attention:
        return model_node
    node_id = f"{p}sage_{tag}"
    nodes[node_id] = {"class_type": "PathchSageAttentionKJ", "inputs": {
        "model": [model_node, 0],
        # "auto" lets sageattention pick the best kernel for the card rather
        # than pinning one this file would have to keep correct.
        "sage_attention": "auto",
        "allow_compile": False,
    }}
    return node_id


def _wan_expert_nodes(
    p: str, cond_node: str, seed: int, steps: int, lora_high: float,
    unet_high: str = _UNET_HIGH, unet_low: str = _UNET_LOW,
    style_lora: str | None = None, style_strength: float = 1.0,
) -> dict:
    """Both Wan 2.2 experts and the two sampler passes that share one latent.

    `cond_node` is whatever produced the conditioning and the empty latent for
    this clip — a WanImageToVideo for an i2v segment, a WanFirstLastFrameToVideo
    for a transition. Everything downstream of it is identical between the two
    workflows, which is why they now share this instead of keeping two copies
    that drifted apart on shift and LoRA strength.

    The high-noise pass runs steps 0..split and hands the *unfinished* latent to
    the low-noise pass (`return_with_leftover_noise`), which finishes it. `split`
    comes from the sigma schedule, not from steps//2 — see
    services/comfy/wan_moe.py for why that distinction is the whole point.

    At lora_high == 0 the LoRA node is left out rather than loaded at zero
    strength: ComfyUI would otherwise read a distill LoRA off disk to multiply
    it by nothing, on a cold load, once per clip.

    `style_lora` chains a second LoRA onto the high-noise expert, after the
    distill. That is the slot the community's transition and metamorphosis
    LoRAs are trained for — they are all "Wan 2.2 I2V, high noise" — and it is
    the high-noise expert because that is the one that decides what the clip
    *becomes*, not merely how sharp it ends up. It is deliberately additive
    rather than a replacement for the distill: the two dials are independent,
    the distill governs how far things travel and this governs what happens on
    the way. Nothing is added to the low-noise expert, which is doing texture.
    """
    split = moe_split_step(steps, _WAN_SHIFT, I2V_BOUNDARY)
    nodes: dict = {
        p+"unet_h": {"class_type": "UNETLoader",          "inputs": {"unet_name": unet_high, "weight_dtype": "default"}},
        p+"unet_l": {"class_type": "UNETLoader",          "inputs": {"unet_name": unet_low,  "weight_dtype": "default"}},
        p+"lora_l": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": [p+"unet_l", 0], "lora_name": _LORA_LOW, "strength_model": _WAN_LORA_LOW}},
    }
    nodes[p+"samp_l"] = {"class_type": "ModelSamplingSD3", "inputs": {
        "model": [_attention_backend(nodes, p, "l", p + "lora_l"), 0], "shift": _WAN_SHIFT,
    }}
    high_model = p + "unet_h"
    if lora_high > 0:
        nodes[p+"lora_h"] = {"class_type": "LoraLoaderModelOnly", "inputs": {
            "model": [high_model, 0], "lora_name": _LORA_HIGH, "strength_model": lora_high,
        }}
        high_model = p + "lora_h"
    if style_lora:
        nodes[p+"lora_s"] = {"class_type": "LoraLoaderModelOnly", "inputs": {
            "model": [high_model, 0], "lora_name": style_lora,
            "strength_model": clamp_style_strength(style_strength),
        }}
        high_model = p + "lora_s"
    nodes[p+"samp_h"] = {"class_type": "ModelSamplingSD3", "inputs": {
        "model": [_attention_backend(nodes, p, "h", high_model), 0], "shift": _WAN_SHIFT,
    }}
    nodes[p+"ks_h"] = {"class_type": "KSamplerAdvanced", "inputs": {
        "model":                     [p+"samp_h", 0],
        "add_noise":                 "enable",
        "noise_seed":                seed,
        "steps": steps, "cfg": wan_cfg_high(lora_high),
        "sampler_name": "euler", "scheduler": WAN_SCHEDULER,
        "start_at_step": 0, "end_at_step": split,
        "return_with_leftover_noise": "enable",
        "positive":     [cond_node, 0],
        "negative":     [cond_node, 1],
        "latent_image": [cond_node, 2],
    }}
    nodes[p+"ks_l"] = {"class_type": "KSamplerAdvanced", "inputs": {
        "model":                     [p+"samp_l", 0],
        "add_noise":                 "disable",
        "noise_seed":                0,
        "steps": steps, "cfg": _WAN_CFG,
        "sampler_name": "euler", "scheduler": WAN_SCHEDULER,
        "start_at_step": split, "end_at_step": 10000,
        "return_with_leftover_noise": "disable",
        "positive":     [cond_node, 0],
        "negative":     [cond_node, 1],
        "latent_image": [p+"ks_h",  0],
    }}
    return nodes


# ── FLF2V workflow builder (key-frame transitions) ────────────────────────────

def _transition_nodes(
    t: int,
    start_img_node: str,
    end_img_node: str,
    width: int, height: int, length: int,
    prompt: str, seed: int,
    steps: int, lora_high: float,
    fun_inp: bool = False,
    style_lora: str | None = None, style_strength: float = 1.0,
) -> tuple[dict, str]:
    """Build one Wan 2.2 FLF2V transition subgraph. Returns (nodes, decode_node_id).

    `fun_inp` picks the Fun-InP expert pair over the i2v pair — the only
    difference between a transition that moves and one that dissolves. See the
    _UNET_FUN_HIGH block for why.
    """
    p = f"t{t}_"
    nodes = {
        p+"clip":   {"class_type": "CLIPLoader",     "inputs": {"clip_name": _CLIP_NAME, "type": "wan", "device": "default"}},
        p+"pos":    {"class_type": "CLIPTextEncode", "inputs": {"clip": [p+"clip", 0], "text": with_lora_trigger(prompt, style_lora)}},
        p+"neg":    {"class_type": "CLIPTextEncode", "inputs": {"clip": [p+"clip", 0], "text": _NEG_PROMPT}},
        p+"vae":    {"class_type": "VAELoader",      "inputs": {"vae_name": _VAE_NAME}},
        p+"flf2v":  {"class_type": "WanFirstLastFrameToVideo", "inputs": {
            "positive":    [p+"pos",  0],
            "negative":    [p+"neg",  0],
            "vae":         [p+"vae",  0],
            "start_image": [start_img_node, 0],
            "end_image":   [end_img_node,   0],
            "width": width, "height": height, "length": length, "batch_size": 1,
        }},
    }
    nodes.update(_wan_expert_nodes(
        p, p + "flf2v", seed, steps, lora_high,
        *( (_UNET_FUN_HIGH, _UNET_FUN_LOW) if fun_inp else (_UNET_HIGH, _UNET_LOW) ),
        style_lora=style_lora, style_strength=style_strength,
    ))
    nodes[p+"decode"] = {"class_type": "VAEDecode", "inputs": {
        "samples": [p+"ks_l", 0],
        "vae":     [p+"vae",  0],
    }}
    return nodes, p + "decode"


def _build_flf2v_single_workflow(
    start_fname: str,
    end_fname: str,
    prompt: str,
    frame_count: int,
    width: int, height: int,
    vid_prefix: str,
    rife_multiplier: int,
    append_end_frame: bool = False,
    steps: int | None = None,
    lora_high: float | None = None,
    fun_inp: bool = False,
    style_lora: str | None = None,
    style_lora_strength: float = 1.0,
) -> tuple[dict, str]:
    """Single key-frame transition with its own VHS save.

    One ComfyUI submission per transition keeps the VRAM peak independent of
    how many key frames the user picked — mirrors _build_i2v_single_workflow.

    There is no `fps` argument: a Wan clip's rate is not a caller decision.
    The model animates at 16 fps and RIFE multiplies the frames, so the only
    rate that plays the motion at the speed it was sampled for is
    16 x rife_multiplier — see WAN_NATIVE_FPS.

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
        0, "img_start", "img_end", width, height, align_wan_length(frame_count), prompt,
        random.randint(0, 2**32 - 1),
        clamp_wan_steps(steps), clamp_lora_high(lora_high),
        fun_inp=fun_inp, style_lora=style_lora, style_strength=style_lora_strength,
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
        "frame_rate":      wan_output_fps(rife_multiplier),
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
    steps: int, lora_high: float,
    style_lora: str | None = None, style_strength: float = 1.0,
) -> dict:
    """One WanImageToVideo segment — Lightning path, two-pass KSampler, RIFE ×N.

    Descended from the `enable_turbo=true` branch of
    video_wan2_2_14B_i2v_reworked_API.json, minus that file's ComfySwitchNode
    multiplexers (turbo is the only path here) and minus its two fixed numbers:
    the step count and the 50/50 expert split are now the caller's, because
    those two are what decide whether the clip moves and how much of it
    resolves. See _wan_expert_nodes and services/comfy/wan_moe.py.
    """
    p = f"s{seg}_"
    nodes = {
        p+"clip":   {"class_type": "CLIPLoader",      "inputs": {"clip_name": _CLIP_NAME, "type": "wan", "device": "default"}},
        p+"vae":    {"class_type": "VAELoader",       "inputs": {"vae_name": _VAE_NAME}},
        p+"pos":    {"class_type": "CLIPTextEncode",  "inputs": {"clip": [p+"clip", 0], "text": with_lora_trigger(prompt, style_lora)}},
        p+"neg":    {"class_type": "CLIPTextEncode",  "inputs": {"clip": [p+"clip", 0], "text": _NEG_PROMPT}},
        p+"i2v":    {"class_type": "WanImageToVideo", "inputs": {
            "width": width, "height": height, "length": frame_count, "batch_size": 1,
            "positive": [p+"pos", 0], "negative": [p+"neg", 0],
            "vae": [p+"vae", 0], "start_image": [img_node_id, 0],
        }},
    }
    nodes.update(_wan_expert_nodes(
        p, p + "i2v", seed, steps, lora_high,
        style_lora=style_lora, style_strength=style_strength,
    ))
    nodes[p+"decode"] = {"class_type": "VAEDecode", "inputs": {"samples": [p+"ks_l", 0], "vae": [p+"vae", 0]}}
    nodes[p+"rife"] = {"class_type": "RIFE VFI", "inputs": {
        "ckpt_name": _RIFE_CKPT, "clear_cache_after_n_frames": 10, "multiplier": rife_multiplier,
        "fast_mode": True, "ensemble": True, "scale_factor": 1,
        "dtype": "float32", "torch_compile": False, "batch_size": 1,
        "frames": [p+"decode", 0],
    }}
    return nodes


def _build_i2v_single_workflow(
    comfy_filename: str,
    prompt: str,
    frame_count: int,
    width: int, height: int,
    vid_prefix: str,
    rife_multiplier: int,
    pingpong: bool,
    steps: int | None = None,
    lora_high: float | None = None,
    style_lora: str | None = None,
    style_lora_strength: float = 1.0,
) -> tuple[dict, str]:
    """Single-image i2v segment with its own VHS save.

    One ComfyUI submission per segment keeps the VRAM peak independent of how
    many images the user picked: each prompt starts with a clean GPU state.
    Segments are stitched together server-side via ffmpeg concat.

    Takes no `fps` for the same reason the transition builder does not: Wan
    animates at 16 fps, and writing RIFE'd frames at anything other than
    16 x rife_multiplier changes the speed of the motion rather than its
    smoothness. See WAN_NATIVE_FPS.
    """
    wf: dict = {"img0": {"class_type": "LoadImage", "inputs": {"image": comfy_filename, "upload": "image"}}}
    wf.update(_i2v_segment(
        0, "img0", prompt, align_wan_length(frame_count), width, height,
        random.randint(0, 2**32 - 1), rife_multiplier,
        clamp_wan_steps(steps), clamp_lora_high(lora_high),
        style_lora=style_lora, style_strength=style_lora_strength,
    ))

    save_id = "i2v_save"
    wf[save_id] = {"class_type": "VHS_VideoCombine", "inputs": {
        "frame_rate":      wan_output_fps(rife_multiplier),
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


# MiniMax samples the audio jointly with the picture, from the same prompt
# text, and the graph is guidance-free — there is no negative prompt to push
# back with. Left to decide for itself on this library's still, moody frames it
# reliably reaches for score, so "sound, not music" has to be stated in the
# positive prompt. prompts/video-minimax-motion.md says it too, but that only
# reaches prompts the suggester wrote; this reaches every submission, including
# hand-typed ones.
# The constraint is against a *score* — music laid over the scene — not against
# every musical sound: an instrument visible in the frame may be played, because
# that is the room making the noise. prompts/video-minimax-motion.md draws the
# same line for the suggester.
_MMX_NO_MUSIC = "No background music or score — only sound that exists in the scene itself."
_MMX_AUDIO_FALLBACK = (
    "Audio: only the sound the moving thing itself makes, and the acoustic space around it."
)
_AUDIO_LINE_RE = re.compile(r"(?mi)^\s*audio\s*:")
_NO_MUSIC_RE = re.compile(r"(?i)\bno (background )?(music|score|soundtrack)\b")


def ensure_sound_only_audio(prompt: str) -> str:
    """Make the prompt ask for noise rather than music.

    Two independent gaps to close: a prompt with no audio line at all leaves
    the choice to the model, and a prompt that describes sound without ruling
    music out leaves it ambiguous. Adds only what is missing, and never
    rewrites or drops what the caller wrote — a hand-typed audio description
    survives verbatim, it just stops being an open invitation for a score.

    Applied when building the workflow, not when persisting the clip, so the
    stored prompt stays the one the user actually wrote.
    """
    out = prompt.strip()
    if not _AUDIO_LINE_RE.search(out):
        out = f"{out}\n{_MMX_AUDIO_FALLBACK}" if out else _MMX_AUDIO_FALLBACK
    if not _NO_MUSIC_RE.search(out):
        out = f"{out} {_MMX_NO_MUSIC}"
    return out


def _build_minimax_single_workflow(
    comfy_filename: str,
    prompt: str,
    frame_count: int,
    width: int, height: int,
    vid_prefix: str,
    rife_multiplier: int = 1,
    end_comfy_filename: str | None = None,
) -> tuple[dict, str, int]:
    """Single MiniMax H3 segment producing an mp4 *with* generated audio.

    With `end_comfy_filename` the same graph becomes a transition instead of an
    animation: the second picture is pinned at the last frame and H3 invents
    the way there. This is the model's fl2va task, not a bolt-on — the
    checkpoint is named for it, and `MiniMaxH3ImageToVideo` carries the
    `last_frame` input for exactly this. Unlike the Wan transition path there
    is no out-of-distribution end constraint to work around, which is why it
    morphs where Wan-on-i2v-weights fades.

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
            "prompt": ensure_sound_only_audio(prompt),
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

    # ── Optional end key frame (fl2va) ──
    # The node center-crops `last_frame` itself but plain-stretches
    # `first_frame`, so the end picture gets the same explicit pre-scale the
    # start one does: two images that arrive on the canvas by different rules
    # would not line up, and a transition is entirely about them lining up.
    if end_comfy_filename:
        wf[p+"load_end"]  = {"class_type": "LoadImage", "inputs": {
            "image": end_comfy_filename, "upload": "image",
        }}
        wf[p+"scale_end"] = {"class_type": "ImageScale", "inputs": {
            "image": [p+"load_end", 0], "upscale_method": "lanczos",
            "width": width, "height": height, "crop": "center",
        }}
        wf[p+"i2v"]["inputs"]["last_frame"] = [p+"scale_end", 0]

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
            fps=(MINIMAX_FPS if req.workflow in MINIMAX_WORKFLOWS
                 else wan_output_fps(req.rife_multiplier)),
            has_audio=(req.workflow in AUDIO_WORKFLOWS),
            # The clamped values, not the requested ones — what the graph was
            # actually built with is the only version worth comparing against.
            wan_steps=(None if req.workflow in MINIMAX_WORKFLOWS else clamp_wan_steps(req.steps)),
            wan_lora_high=(None if req.workflow in MINIMAX_WORKFLOWS else clamp_lora_high(req.lora_high)),
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
    band_node: str | None = None,
) -> None:
    """Record the job's coarse phase, plus what ComfyUI submission (if any) owns it.

    `prompt_id` + `band` are what let GET /jobs/{id}/progress replace a static
    "generating frames…" with ComfyUI's own live stage and sampler step,
    scaled into the slice of the bar this submission is responsible for.

    `band_node` narrows which node's counter is allowed to move the bar. Every
    node reports one, and most are not a fraction of the job — a VHS loader
    announces "93/93" a second in. Naming the sampler keeps the bar monotonic;
    omitting it keeps the original behaviour.
    """
    entry = {"phase": phase, "message": message, "pct": pct}
    if prompt_id:
        entry["_prompt_id"] = prompt_id
    if band:
        entry["_band"] = band
    if band_node:
        entry["_band_node"] = band_node
    _progress[vid_key] = entry


# ── Stopping a job ────────────────────────────────────────────────────────────

def forget_progress(key: str | None = None) -> int:
    """Drop the cached progress entry for one job, or for every job.

    The all-at-once form is what routers/system.py's cancel-all uses: after
    every job task has been cancelled, an entry left behind would keep a
    finished-looking bar on screen for a render that is not running.
    """
    if key is not None:
        return 1 if _progress.pop(key, None) is not None else 0
    n = len(_progress)
    _progress.clear()
    return n


def _live_prompt_ids(key: str, video: Video | None = None) -> list[str | None]:
    """Every ComfyUI prompt this job may still have in flight.

    Two sources, because they answer at different resolutions: the row carries
    the last prompt this job *submitted*, and the progress entry carries the
    one it is *waiting on right now*. In a per-segment render they are usually
    the same id; when they are not, the newer one is the one still burning GPU.
    """
    return [_progress.get(key, {}).get("_prompt_id"), video.comfy_prompt_id if video else None]


async def _stop_video_job(video: Video) -> dict:
    """Cancel a video job's task and drop its ComfyUI prompt. Idempotent."""
    key = str(video.id)
    return await cancel_job(video.id, prompt_ids=_live_prompt_ids(key, video))


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
    transitions) instead of single images.

    Serves both transition workflows: `flf2v` on the Wan expert pair, and
    `minimax_flf` on MiniMax H3's fl2va task. They differ only in the builder
    and in what has to happen to the audio afterwards."""
    n_imgs = len(comfy_names)
    if n_imgs < 2:
        raise ValueError("A transition workflow requires at least 2 images")
    n_trans = n_imgs - 1
    minimax = req.workflow in MINIMAX_WORKFLOWS

    # Ask ComfyUI once, not once per transition: whether the Fun-InP pair is
    # installed decides between a transition that moves and one that fades, and
    # it is worth saying out loud in the log which one this job got.
    fun_inp = False
    if not minimax:
        fun_inp = await _fun_inp_available()
        logger.info(
            "Video job %s: flf2v experts = %s",
            video_id, "Wan2.2-Fun-InP" if fun_inp else "Wan2.2-i2v (Fun-InP not installed)",
        )

    prompts_list = req.prompts if len(req.prompts) == n_trans else [req.prompt] * n_trans
    fc_list = req.frame_counts if len(req.frame_counts) == n_trans else [req.frame_count] * n_trans
    if not minimax:
        # What the graph will really render, so the clip row and the poll
        # deadline are about the same clip the builder builds.
        fc_list = [align_wan_length(fc) for fc in fc_list]
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
        native_length = None
        if minimax:
            wf, save_node, native_length = _build_minimax_single_workflow(
                start_fname, p_i, fc_i, req.width, req.height, seg_prefix,
                req.rife_multiplier, end_comfy_filename=end_fname,
            )
        else:
            wf, save_node = _build_flf2v_single_workflow(
                start_fname, end_fname, p_i, fc_i, req.width, req.height, seg_prefix,
                req.rife_multiplier, append_end_frame=req.end_on_keyframe,
                steps=req.steps, lora_high=req.lora_high, fun_inp=fun_inp,
                style_lora=req.style_lora, style_lora_strength=req.style_lora_strength,
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
        # MiniMax keeps the flat budget: its cost model is a different one and
        # was never measured against the Wan coefficients.
        seg_outputs = await poll_history(
            client, prompt_id, interval=POLL_INTERVAL,
            timeout=POLL_TIMEOUT if minimax else wan_poll_timeout(
                req.width, req.height, fc_i, clamp_wan_steps(req.steps),
                clamp_lora_high(req.lora_high),
            ),
        )
        seg_src = _comfy_save_path(seg_outputs.get(save_node, {}), f"Transition {i + 1}")

        # Persist the clip under a stable name and register it in the library
        # immediately — a crash on a later transition loses nothing.
        seg_dest  = seg_dir / f"seg_{i}.mp4"
        seg_thumb = seg_dir / f"seg_{i}_thumb.jpg"
        if native_length is not None and req.rife_multiplier > 1:
            # RIFE stretched the picture but not the jointly-sampled audio —
            # same re-sync _run_i2v_multi does for minimax_i2v.
            await stretch_native_audio(
                seg_src, seg_dest,
                native_length=native_length, fps=MINIMAX_FPS,
                ffmpeg_path=settings.ffmpeg_path,
            )
        else:
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
    if req.workflow not in MINIMAX_WORKFLOWS:
        fc_list = [align_wan_length(fc) for fc in fc_list]
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
        if req.workflow in MINIMAX_WORKFLOWS:
            wf, save_node, native_length = _build_minimax_single_workflow(
                fname, p_i, fc_i, req.width, req.height, seg_prefix,
                req.rife_multiplier,
            )
        else:
            wf, save_node = _build_i2v_single_workflow(
                fname, p_i, fc_i, req.width, req.height, seg_prefix,
                req.rife_multiplier, req.pingpong,
                steps=req.steps, lora_high=req.lora_high,
                style_lora=req.style_lora, style_lora_strength=req.style_lora_strength,
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
        # MiniMax keeps the flat budget: its cost model is a different one and
        # was never measured against these coefficients.
        seg_timeout = POLL_TIMEOUT if req.workflow in MINIMAX_WORKFLOWS else wan_poll_timeout(
            req.width, req.height, fc_i, clamp_wan_steps(req.steps),
            clamp_lora_high(req.lora_high),
        )
        seg_outputs = await poll_history(
            client, prompt_id, timeout=seg_timeout, interval=POLL_INTERVAL,
        )
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


async def _finalize_video_done(
    video_id: uuid.UUID, dest: Path, vid_key: str, *, status: str = "done",
) -> None:
    """Common success path: thumbnail + persist filename/filepath + status.

    `status` exists for a job whose picture is finished but whose *piece* is
    not. A beat cut is the case: its song is muxed on afterwards, and flipping
    the row to 'done' before that would let the poller stop, the card latch the
    silent rendition, and an upscale start against the file the mux is about to
    replace. Such a caller lands the file as 'assembling' and finishes the row
    itself once the last step is really done.
    """
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
            video.status   = status
            video.error    = None
            await db.commit()
    if status == "done":
        _progress.pop(vid_key, None)


# How much of the card each workflow needs free before it is safe to start.
# MiniMax stages ~15 GB for its text encoder alone ("14956MB Staged" in the
# ComfyUI log), so it needs essentially the whole 16 GB card; the Wan
# workflows are smaller. Falling short is not a slow path — ComfyUI's dynamic
# loader dies with a CUDA OOM that takes its prompt worker thread with it.
_MIN_FREE_VRAM = {
    "minimax_i2v": 13.5 * 1024**3,
    "minimax_flf": 13.5 * 1024**3,   # same model, one extra VAE-encoded frame
}
_MIN_FREE_VRAM_DEFAULT = 9.0 * 1024**3


async def _free_ollama_vram(workflow: str | None = None) -> None:
    """Resolve how much VRAM `workflow` needs, then hand off to the shared
    pre-flight (services/comfy/vram.py) — evict Ollama, then verify the card
    is actually free before submitting."""
    required = _MIN_FREE_VRAM.get(workflow or "", _MIN_FREE_VRAM_DEFAULT)
    await free_vram_for(required, workflow or "this workflow")


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
            if req.workflow in TRANSITION_WORKFLOWS:
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


def _video_owned_paths(video: Video) -> list[Path]:
    """Every file a Video row owns: the original, its derived renditions, the
    grain preview and the thumbnail. Its segments directory is separate — see
    `_segments_dir` — because removing a directory is not an unlink."""
    paths: list[Path] = []
    if video.filepath:
        paths.append(settings.storage_dir / video.filepath)
    for name in (video.muxed_filename, video.upscale_filename, video.grain_filename):
        if name:
            paths.append(settings.videos_dir / name)
    paths.append(settings.videos_dir / _look_preview_name(video.id))
    paths.append(settings.videos_dir / f"{video.id}_thumb.jpg")
    return paths


async def _delete_source_videos(video_ids: list[uuid.UUID]) -> None:
    """Delete finished videos that a merge consumed.

    Row first, files after: a video can be referenced elsewhere (a scheduled
    Instagram reel, an improv session), and a FK that refuses the delete has to
    leave the video intact and playable rather than a row pointing at files
    that are already gone.
    """
    async with AsyncSessionLocal() as db:
        for vid in video_ids:
            video = await db.get(Video, vid)
            if not video:
                continue
            paths, seg_dir = _video_owned_paths(video), _segments_dir(vid)
            try:
                await db.delete(video)
                await db.commit()
            except Exception:
                await db.rollback()
                logger.warning("Could not delete merged source video %s (still referenced?)", vid)
                continue
            _progress.pop(str(vid), None)
            for p in paths:
                p.unlink(missing_ok=True)
            shutil.rmtree(seg_dir, ignore_errors=True)


def _video_primary_name(v: Video) -> str | None:
    """The rendition of a finished video that everything should read.

    Post-processing writes siblings rather than overwriting the original, each
    pass consuming the one before it — soundtrack, then upscale, then grain —
    so the last one present is the most complete file. Mirrored by
    services/instagram/media.py::resolve_video_path; keep the two in step.
    """
    return v.grain_filename or v.upscale_filename or v.muxed_filename or v.filename


def _video_ungrained_name(v: Video) -> str | None:
    """The most finished rendition that has NO film grain in it.

    For a beat cut, grain has to come last. Feeding an already-grained video
    into an edit that will be grained again stacks two passes on that one shot
    and leaves its neighbours with one, which is visible immediately — and the
    grain of the piece should belong to the piece, not to whichever clip
    happened to be grained on its own beforehand.

    Everything below the grain is kept: an upscale is a better picture and a
    muxed file's video stream is a copy of the one under it.
    """
    return v.upscale_filename or v.muxed_filename or v.filename


def _video_primary_path(v: Video, *, ungrained: bool = False) -> Path | None:
    name = _video_ungrained_name(v) if ungrained else _video_primary_name(v)
    if not name:
        return None
    if name == v.filename and v.filepath:
        return settings.storage_dir / v.filepath
    return settings.videos_dir / name


def _merge_canvas(sizes: list[tuple[int | None, int | None]]) -> tuple[int, int]:
    """Normalisation target for a merge: the largest effective canvas selected.

    Effective, not stored — an upscaled source's real size is its upscale's,
    and the merge is the whole reason that pass exists. Largest rather than the
    first source's: with a uniform selection (the ordinary case) the two agree,
    and where they disagree it is because only some sources were upscaled,
    where "first wins" would scale the restored ones back down and undo the
    work. A source of unknown size falls back to a square default rather than
    dropping out of the vote.
    """
    sizes = sizes or [(None, None)]
    return max(((w or 960, h or 960) for w, h in sizes), key=lambda wh: wh[0] * wh[1])


@dataclass(frozen=True)
class _MergeSource:
    """One resolved merge input: the file to read plus what it really is."""
    inp: MergeInput
    width: int | None
    height: int | None
    fps: int


async def _resolve_merge_sources(
    sources: list[tuple[str, uuid.UUID]],
    *,
    ungrained: bool = False,
) -> list[_MergeSource]:
    """Resolve each selected clip/video to its playable file, audio flag and
    effective canvas — the three facts the concat needs.

    A clip answers all of them from its row: `_clip_primary_path` and
    `_clip_dimensions` already account for a per-clip upscale, and `has_audio`
    is persisted at render time. A finished video does not — the row's
    `width`/`height` record the *generation* canvas and are left untouched by
    an upscale, and nothing on it says whether the current rendition carries
    audio (a muxed soundtrack adds one). So the file itself is probed, the
    same rule `_upscale_plan_from_file` follows.

    `ungrained` picks the rendition below any film-grain pass — see
    `_video_ungrained_name`. A plain merge takes what the tool plays; a beat cut
    takes the clean picture and grains the finished edit instead.
    """
    clip_ids  = [sid for kind, sid in sources if kind == "clip"]
    video_ids = [sid for kind, sid in sources if kind == "video"]
    async with AsyncSessionLocal() as db:
        clips: dict[uuid.UUID, VideoClip] = {}
        if clip_ids:
            r = await db.execute(select(VideoClip).where(VideoClip.id.in_(clip_ids)))
            clips = {c.id: c for c in r.scalars().all()}
        videos: dict[uuid.UUID, Video] = {}
        if video_ids:
            r = await db.execute(select(Video).where(Video.id.in_(video_ids)))
            videos = {v.id: v for v in r.scalars().all()}

    resolved: list[_MergeSource] = []
    for kind, sid in sources:
        if kind == "clip":
            c = clips.get(sid)
            if not c:
                raise ValueError("One or more selected clips no longer exist")
            # The upscaled rendition when the clip has one: upscaling happens
            # per clip precisely so the merge can consume it, and reading
            # `filename` here would throw that work away.
            f = _clip_primary_path(c)
            if not f.exists():
                raise FileNotFoundError(f"Clip file missing on disk: {f}")
            w, h = _clip_dimensions(c)
            resolved.append(_MergeSource(
                MergeInput(path=f, has_audio=c.has_audio), w, h, c.fps or 24,
            ))
        else:
            v = videos.get(sid)
            if not v:
                raise ValueError("One or more selected videos no longer exist")
            f = _video_primary_path(v, ungrained=ungrained)
            if not f or not f.exists():
                raise FileNotFoundError(f"Video file missing on disk: {v.filename or sid}")
            pw, ph = await probe_video_dimensions(f)
            resolved.append(_MergeSource(
                MergeInput(path=f, has_audio=await probe_has_audio(f)),
                pw or v.width, ph or v.height, v.fps or 24,
            ))
    return resolved


async def _run_merge(
    video_id: uuid.UUID, sources: list[tuple[str, uuid.UUID]], delete_sources: bool,
) -> None:
    """Concatenate the chosen sources into the merge Video's final file.

    Runs as a background task. Sources may be library clips from different
    jobs/workflows, finished videos, or a mix of both; services.video.merge
    normalizes resolution/fps/audio in one ffmpeg pass. Reports progress
    through the same _progress dict as _run_generation so the existing polling
    endpoint keeps working without any client-side branching.
    """
    vid_key = str(video_id)

    try:
        resolved = await _resolve_merge_sources(sources)
        inputs = [s.inp for s in resolved]
        width, height = _merge_canvas([(s.width, s.height) for s in resolved])
        fps = resolved[0].fps
        dest = settings.videos_dir / f"{video_id}_artrium.mp4"

        _set_progress(vid_key, "finalizing", f"Merging {len(inputs)} source(s)…", 40)
        if len(inputs) == 1:
            await asyncio.to_thread(shutil.copy2, inputs[0].path, dest)
        else:
            await merge_clips(
                inputs, dest, width, height, fps, ffmpeg_path=settings.ffmpeg_path,
            )

        # The row was created from the first source's numbers, which are only a
        # guess at what comes out: `_merge_canvas` may have picked a bigger one,
        # and a probed video source can differ from what its row claims. Persist
        # what was actually rendered before the card goes to "done".
        async with AsyncSessionLocal() as db:
            row = await db.get(Video, video_id)
            if row:
                row.width, row.height, row.fps = width, height, fps
                await db.commit()

        await _finalize_video_done(video_id, dest, vid_key)
        logger.info("Video %s merged from %d source(s)", video_id, len(inputs))

    except Exception as exc:
        logger.exception("Video merge %s failed", video_id)
        await _finalize_video_failure(video_id, exc, vid_key)
        return

    if delete_sources:
        # The merge itself succeeded — a cleanup hiccup must not flip the
        # finished video back to failed, so this runs outside the main try.
        clip_ids  = [sid for kind, sid in sources if kind == "clip"]
        video_ids = [sid for kind, sid in sources if kind == "video"]
        try:
            if clip_ids:
                await _delete_clips(clip_ids)
            if video_ids:
                await _delete_source_videos(video_ids)
            logger.info(
                "Merge %s: deleted %d source clip(s) and %d source video(s)",
                video_id, len(clip_ids), len(video_ids),
            )
        except Exception:
            logger.exception("Merge %s: source cleanup failed (video is fine)", video_id)


# ── Endpoints ─────────────────────────────────────────────────────────────────

_TRANSITION_MAX_EDGE = 512
_TRANSITION_JPG_QUALITY = 80
_TRANSITION_TIMEOUT_FLOOR = 180.0
_TRANSITION_TIMEOUT_PER_IMAGE = 20.0


class SuggestTransitionsRequest(BaseModel):
    image_ids: list[uuid.UUID]   # in the user's selected playback order
    context: str = ""            # optional story/narrative context (story-frames flow)
    # Which family the prompts are for. Both suggest endpoints read it: a
    # MiniMax prompt carries an Audio: line and names its arrival, a Wan one
    # is a short silent motion line. Defaulted to the Wan side so an older
    # client that does not send it keeps its old behaviour.
    workflow: str = "i2v_multi"


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
    """VLM-suggested per-transition prompts — one vision call, N-1 prompts
    back. Purely advisory: nothing is persisted here; the client fills its own
    per-transition textareas and the user can edit before calling /generate.

    Serves both transition workflows, with a different writer for each.
    MiniMax H3 samples its audio jointly with the picture, so its prompts carry
    an `Audio:` line the Wan ones must not have — and it is the family that
    rewards naming the destination, which is what its writer is built around.
    """
    n = len(body.image_ids)
    if n < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 images to suggest transitions")
    if n > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 images")

    jpgs = await _load_suggest_jpgs(body.image_ids, db)
    logger.info(
        "Suggest-transitions: %d images, workflow=%s, model=%s, payload=%dKB",
        n, body.workflow, settings.ollama_titler_model,
        sum(len(j) for j in jpgs) // 1024,
    )

    timeout = max(_TRANSITION_TIMEOUT_FLOOR, _TRANSITION_TIMEOUT_PER_IMAGE * n)
    minimax = body.workflow in MINIMAX_WORKFLOWS
    writer = generate_minimax_transition_prompts if minimax else generate_transition_prompts
    try:
        prompts = await writer(jpgs, context=body.context, timeout=timeout)
    except Exception as exc:
        logger.exception("Transition prompt suggestion failed for %d images", n)
        raise HTTPException(status_code=502, detail=f"Suggestion failed: {exc}")

    if minimax:
        # The writer is asked for an Audio: line and a small VLM does not
        # always give one. The builder would add a fallback at submit time
        # anyway, but then the user never sees it and cannot edit it — so it
        # is completed here, where the answer is still on its way to a
        # textarea. A line the model did write survives verbatim.
        prompts = [ensure_sound_only_audio(p) if p.strip() else p for p in prompts]

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
    elif body.workflow in MINIMAX_WORKFLOWS:
        # A transition mode renders n-1 clips, so its image bound is one wider
        # at the bottom (two pictures make one transition) and one wider at the
        # top for the same GPU-time budget.
        lo, hi = (2, 7) if body.workflow == "minimax_flf" else (1, 6)
        if not (lo <= n <= hi):
            raise HTTPException(
                status_code=400,
                detail=f"{body.workflow} requires {lo}–{hi} image IDs",
            )
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
    if body.workflow in GENERATE_WORKFLOWS and body.prompts:
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
        fps=(MINIMAX_FPS if body.workflow in MINIMAX_WORKFLOWS
             else wan_output_fps(body.rife_multiplier)),
        status="generating",
        created_at=datetime.now(timezone.utc),
    )
    db.add(video)
    await db.commit()
    await db.refresh(video)

    safe_create_task(_run_generation(video.id, body), name=f"video_generation:{video.id}")
    logger.info("Queued video generation job %s (%s, %d images)", video.id, body.workflow, n)

    return {"video_id": str(video.id), "status": "generating"}


def _expected_clip_count(video: Video) -> int | None:
    """How many clips this job will produce when it finishes.

    A transition workflow animates the gaps *between* key frames, so it yields
    one clip fewer than it was given images; the others animate each image on
    its own. Only meaningful for generation jobs — a merge has no stack of its
    own.
    """
    if not video.n_images or video.workflow == "merge":
        return None
    if video.workflow in TRANSITION_WORKFLOWS:
        return max(1, video.n_images - 1)
    return video.n_images


@router.get("/jobs/{video_id}/progress")
async def get_job_progress(video_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """Lightweight progress endpoint — reads module-level dict + optional ComfyUI queue check."""
    video = await db.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video job not found")

    live = dict(_progress.get(str(video_id), {}))

    # Segments land one at a time and each is browsable the moment it exists,
    # so the poller is told how many are ready rather than being made to wait
    # for the whole job. It refetches the stack only when this number moves.
    clips_done = await db.scalar(
        select(func.count()).select_from(VideoClip).where(VideoClip.video_id == video_id)
    ) or 0
    clip_counts = {"clips_done": clips_done, "clips_expected": _expected_clip_count(video)}

    if video.status == "done":
        done = {"phase": "done", "message": "Complete", "pct": 100, **clip_counts}
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
        # Partial stacks are usable, so a failed job still reports what it got
        # far enough to save.
        return {
            "phase": "failed", "message": video.error or "Generation failed", "pct": 0,
            **clip_counts,
        }
    if video.status == "review":
        # A two-stage AnimateLCM render, parked between its base pass and its
        # hires pass (routers/vace.py). Read off the row rather than out of
        # `_progress`, because the decision can outlive this process by days
        # and a restart must not turn "waiting for you" back into "processing".
        return {
            "phase": "review", "message": "Preview ready", "pct": 100, **clip_counts,
        }

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
    return {**_attach_live_stage(prog), **clip_counts}


@router.post("/jobs/{video_id}/cancel")
async def cancel_video_job(video_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """Stop this job without deleting it — the clips it already rendered stay.

    A post-pass (upscale, look, soundtrack) is stopped the same way, but the
    row keeps its status: the *video* is fine, it is the pass on top of it that
    was called off, and marking it failed would hide a finished piece behind an
    error it does not have.
    """
    video = await db.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    stopped = await _stop_video_job(video)
    forget_progress(str(video_id))
    if video.status in ("queued", "generating", "assembling"):
        video.status = "failed"
        video.error = CANCELLED_MSG
        await db.commit()
    await db.refresh(video)
    logger.info("Cancelled video job %s (%s)", video_id, video.status)
    return {**_serialize(video), "cancelled": stopped}


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
        # Null on MiniMax clips and on anything rendered before these became
        # settings — the card omits the chip rather than inventing a default.
        "wan_steps":     c.wan_steps,
        "wan_lora_high": c.wan_lora_high,
        "upscale_resolution": c.upscale_resolution,
        "upscale_rife":       c.upscale_rife,
        "upscale_fps":        c.upscale_fps,
        "has_upscale":        bool(c.upscale_filename),
        "upscale_rendering":  _is_clip_upscaling(c),
        "created_at":  c.created_at.isoformat(),
    }


@router.get("/sampler-info")
async def sampler_info():
    """What the Steps/Motion row needs to describe a job the way it will run.

    The tool page carries fallback copies of all three so the hint renders
    before this lands (and if it never does), but they are only fallbacks: the
    expert split moves with the sigma schedule and the cost moves with the
    attention backend, and both of those live here. A page guessing either one
    is worse than a page that waits a moment for the real answer.
    """
    return {
        "sage_attention": settings.wan_sage_attention,
        "sec_per_step_pf": (_WAN_SEC_PER_STEP_PF_SAGE if settings.wan_sage_attention
                            else _WAN_SEC_PER_STEP_PF),
        "sec_post_pf": _WAN_SEC_POST_PF,
        "splits": {
            str(s): moe_split_step(s, _WAN_SHIFT, I2V_BOUNDARY)
            for s in range(_WAN_STEPS_MIN, _WAN_STEPS_MAX + 1)
        },
    }


@router.get("/wan-loras")
async def wan_loras():
    """The transition-LoRA shelf, restricted to what ComfyUI actually has.

    What is installed is read from ComfyUI, because the point of the slot is
    that the user adds files to it; *which* of those files belong in a Wan
    transition is decided by services/comfy/wan_transition_loras.py, because
    the folder is shared with every other model family in the project and a
    foreign LoRA fails by doing nothing at all.

    The two lightx2v distills are dropped before matching: they are what the
    Motion dial already drives, and offering them here would let someone stack
    a second copy of the LoRA the graph is loading anyway.
    """
    names = await loader_choices("LoraLoaderModelOnly", "lora_name")
    return {"loras": offered_transition_loras(names - {_LORA_HIGH, _LORA_LOW})}


@router.get("/clips")
async def list_clips(
    video_id: uuid.UUID | None = None, db: AsyncSession = Depends(get_db),
):
    """Library clips — all of them, or one job's stack with `?video_id=`.

    The filter exists for the running-job poller: a generation job's stack
    grows one clip at a time and is refetched each time it does, which should
    not mean pulling the whole library down on every segment.
    """
    stmt = select(VideoClip).order_by(VideoClip.video_id, VideoClip.idx)
    if video_id is not None:
        stmt = stmt.where(VideoClip.video_id == video_id)
    result = await db.execute(stmt)
    return [_serialize_clip(c) for c in result.scalars().all()]


@router.delete("/clips/{clip_id}", status_code=204)
async def delete_clip(clip_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """Delete one clip (file + thumbnail + row). A generation job that loses
    its last clip is pruned entirely — an empty stack has nothing to show."""
    clip = await db.get(VideoClip, clip_id)
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")
    job_id = clip.video_id
    # A clip carries its own task (an upscale runs as `clip_upscale:<clip id>`),
    # so it is stopped by clip id, not by the job's.
    await cancel_job(clip_id, prompt_ids=_live_prompt_ids(_clip_key(clip_id)))
    seg_dir = _segments_dir(job_id)
    (seg_dir / clip.filename).unlink(missing_ok=True)
    (seg_dir / clip.thumb).unlink(missing_ok=True)
    if clip.upscale_filename:
        (seg_dir / clip.upscale_filename).unlink(missing_ok=True)
    _progress.pop(_clip_key(clip_id), None)
    await db.delete(clip)
    await db.commit()
    await _prune_empty_clip_job(db, job_id)


MERGE_MAX_SOURCES = 50


@router.post("/merge", status_code=202)
async def merge_videos(body: MergeRequest, db: AsyncSession = Depends(get_db)):
    """Concatenate the chosen sources — library clips from any job or workflow,
    finished videos, or a mix — in the given order into a new final video
    (workflow='merge'). Resolution, fps and audio are normalized to make mixed
    selections always mergeable. With delete_sources=true the sources are
    removed after success.

    A finished video enters the merge as the rendition the tool plays: its
    soundtrack, upscale and grain are all baked in already, so joining two
    finished pieces keeps what was done to each of them."""
    sources = body.sources()
    if not sources:
        raise HTTPException(status_code=400, detail="Nothing selected to merge")
    if len(sources) > MERGE_MAX_SOURCES:
        raise HTTPException(
            status_code=400, detail=f"Maximum {MERGE_MAX_SOURCES} sources per merge",
        )
    keys = [(s.kind, s.id) for s in sources]
    if len(set(keys)) != len(keys):
        raise HTTPException(status_code=400, detail="Duplicate source in selection")

    clip_ids  = [s.id for s in sources if s.kind == "clip"]
    video_ids = [s.id for s in sources if s.kind == "video"]

    clips: dict[uuid.UUID, VideoClip] = {}
    if clip_ids:
        r = await db.execute(select(VideoClip).where(VideoClip.id.in_(clip_ids)))
        clips = {c.id: c for c in r.scalars().all()}
        missing = [str(cid) for cid in clip_ids if cid not in clips]
        if missing:
            raise HTTPException(status_code=404, detail=f"Clip(s) not found: {', '.join(missing)}")

    videos: dict[uuid.UUID, Video] = {}
    if video_ids:
        r = await db.execute(select(Video).where(Video.id.in_(video_ids)))
        videos = {v.id: v for v in r.scalars().all()}
        missing = [str(vid) for vid in video_ids if vid not in videos]
        if missing:
            raise HTTPException(status_code=404, detail=f"Video(s) not found: {', '.join(missing)}")
        for v in videos.values():
            if v.status != "done" or not v.filename:
                raise HTTPException(
                    status_code=400,
                    detail=f"Video {v.id} is not a finished video (status: {v.status})",
                )
            # A post-pass rewrites the very file this merge would read, and it
            # is a sibling under a fixed name, so a half-written rendition
            # would go straight into the concat.
            if _is_upscaling(v) or _is_graining(v):
                raise HTTPException(
                    status_code=409,
                    detail=f"Video {v.id} is still rendering a post-pass — wait for it to finish",
                )

    # Placeholder numbers for the card while the merge runs; _run_merge
    # overwrites them with what actually came out.
    first = sources[0]
    src_w, src_h, src_fps = (
        (clips[first.id].width, clips[first.id].height, clips[first.id].fps)
        if first.kind == "clip"
        else (videos[first.id].width, videos[first.id].height, videos[first.id].fps)
    )

    video = Video(
        id=uuid.uuid4(),
        workflow="merge",
        status="assembling",
        width=src_w,
        height=src_h,
        fps=src_fps,
        n_images=len(sources),   # for merges: number of sources concatenated
        created_at=datetime.now(timezone.utc),
    )
    db.add(video)
    await db.commit()
    await db.refresh(video)

    safe_create_task(
        _run_merge(video.id, keys, body.delete_sources),
        name=f"video_merge:{video.id}",
    )
    logger.info(
        "Queued merge %s from %d clip(s) + %d video(s), delete_sources=%s",
        video.id, len(clip_ids), len(video_ids), body.delete_sources,
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
async def video_thumbnail(video_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """The card image for a video — made on the spot if it is missing.

    Every path that writes a video also writes one of these, but "every path"
    has not always been true and will not stay true on its own: rows rendered
    before that convention existed have a perfectly good video and no
    thumbnail, and a workflow added later can forget again. The old behaviour
    was a 404, which the grid drew as an empty tile — the picture was right
    there in the file the whole time.

    So a miss falls back to making it. It is written to the usual path, which
    means each video pays for this exactly once, and a workflow that forgets
    the thumbnail is now a slow first paint rather than a permanent hole.
    """
    p = settings.videos_dir / f"{video_id}_thumb.jpg"
    if p.exists():
        return FileResponse(p, media_type="image/jpeg")

    video = await db.get(Video, video_id)
    name = _video_primary_name(video) if video else None
    source = settings.videos_dir / name if name else None
    if source and source.is_file():
        try:
            await make_video_thumbnail(source, p)
        except Exception as exc:
            logger.warning("On-demand thumbnail failed for %s: %s", video_id, exc)
        if p.exists():
            logger.info("Generated missing thumbnail for video %s", video_id)
            return FileResponse(p, media_type="image/jpeg")

    raise HTTPException(status_code=404, detail="Thumbnail not found")


@router.get("")
async def list_videos(db: AsyncSession = Depends(get_db)):
    videos = (await db.execute(
        select(Video).order_by(desc(Video.created_at))
    )).scalars().all()

    # One pass over the clips rather than a query per video: a job that was
    # never assembled has no file of its own, and its stack is the only thing
    # that can describe it in a grid. Ordered by idx so the first clip — the
    # opening shot — is the one that becomes the thumbnail.
    clips = (await db.execute(
        select(VideoClip).order_by(VideoClip.video_id, VideoClip.idx)
    )).scalars().all()
    counts: dict[uuid.UUID, int] = {}
    thumbs: dict[uuid.UUID, str] = {}
    for c in clips:
        counts[c.video_id] = counts.get(c.video_id, 0) + 1
        if c.video_id not in thumbs and c.thumb:
            thumbs[c.video_id] = f"/api/video/segments/{c.video_id}/{c.thumb}"

    return [
        _serialize(v, counts.get(v.id, 0), thumbs.get(v.id))
        for v in videos
    ]


class BulkDeleteVideosRequest(BaseModel):
    ids: list[uuid.UUID]


@router.delete("")
async def bulk_delete_videos(
    body: BulkDeleteVideosRequest, db: AsyncSession = Depends(get_db),
):
    """Delete several videos in one request.

    **Deliberately not one transaction.** A video can be referenced elsewhere —
    a scheduled Instagram post, an improv session — and such a row refuses to
    go. In a single transaction one refusal takes the whole batch down with it,
    and since the files would already have been unlinked by then, the survivors
    would come back as rows pointing at nothing. So each video is committed on
    its own and a refusal is reported rather than raised.

    **Row first, files after**, for the same reason `_delete_source_videos`
    does it that way: if the delete is refused, the video has to stay intact
    and playable instead of becoming a row whose files are gone.

    A video that is already missing counts as deleted. The caller asked for it
    not to exist, it does not exist, and the grid should drop the tile — an
    error there would only make a stale gallery look broken.
    """
    deleted: list[str] = []
    failed: list[dict] = []
    # dict.fromkeys rather than set(): a duplicated id should be collapsed, but
    # the order the caller sent still decides what the log reads like.
    for vid in dict.fromkeys(body.ids):
        video = await db.get(Video, vid)
        if not video:
            deleted.append(str(vid))
            continue
        await _stop_video_job(video)      # see delete_video — stop, then delete
        paths, seg_dir = _video_owned_paths(video), _segments_dir(vid)
        try:
            await db.delete(video)
            await db.commit()
        except Exception:
            await db.rollback()
            logger.warning("Bulk delete: video %s refused (still referenced?)", vid)
            failed.append({"id": str(vid),
                           "reason": "Wird noch von einem Beitrag verwendet"})
            continue
        _progress.pop(str(vid), None)
        for path in paths:
            path.unlink(missing_ok=True)
        shutil.rmtree(seg_dir, ignore_errors=True)
        deleted.append(str(vid))

    logger.info("Bulk deleted %d video(s), %d refused", len(deleted), len(failed))
    return {"deleted": deleted, "failed": failed}


@router.delete("/{video_id}", status_code=204)
async def delete_video(video_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    video = await db.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    # Stop the render before touching anything. Deleting the row used to leave
    # the job running: ComfyUI finished the clip nobody would ever see, and the
    # task then wrote its segments back into the directory removed below.
    await _stop_video_job(video)
    paths, seg_dir = _video_owned_paths(video), _segments_dir(video_id)
    # Row first, files after — see bulk_delete_videos above. This used to
    # unlink first, which turned a refused delete into a playable-looking row
    # with no files behind it.
    await db.delete(video)
    await db.commit()
    _progress.pop(str(video_id), None)
    for p in paths:
        p.unlink(missing_ok=True)
    shutil.rmtree(seg_dir, ignore_errors=True)


# ── Soundtrack (mux a generated Song onto a generated Video) ──────────────────

class SoundtrackAttach(BaseModel):
    song_id: uuid.UUID
    # Keep the clip's own generated audio under the song instead of replacing
    # it. Off by default: the historical behaviour, and the right one for a
    # silent Wan clip where there is nothing to keep.
    include_bed: bool = False
    bed_volume: float = BED_VOLUME_DEFAULT


async def _run_soundtrack_mux(
    video_id: uuid.UUID, song_id: uuid.UUID, bed_volume: float | None = None,
) -> None:
    """Background task: probe the video, mux the song's audio with fade-out,
    persist the muxed filename + FK. On failure write `error` on the Video row.

    `bed_volume` not None keeps the video's own generated audio under the song
    at that level; None replaces it, as attaching a song always used to."""
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
            # A beat cut that starts at a later bar carries the offset on the
            # row, not in this call: the mux is re-run after every upscale and
            # grain pass, and a caller-supplied offset would be lost there.
            song_start = video.soundtrack_start_seconds or 0.0

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
            include_bed=bed_volume is not None,
            bed_volume=bed_volume if bed_volume is not None else BED_VOLUME_DEFAULT,
            song_start=song_start,
        )

        async with AsyncSessionLocal() as db:
            video = await db.get(Video, video_id)
            if video:
                video.soundtrack_song_id = song_id
                video.soundtrack_bed_volume = bed_volume
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
    # time-stretch it. Only the slow-motion variant, though — a pass with a
    # target rate kept the clip's length, so its re-render will too.
    if (video.upscale_rife or 1) > 1 and not video.upscale_fps:
        raise HTTPException(
            status_code=409,
            detail=f"This video's upscale interpolates {video.upscale_rife}×, and re-rendering "
                   "it would time-stretch the song. Re-run the upscale without "
                   "interpolation first.",
        )

    # A beat cut's song offset belongs to the cut that was planned against that
    # track. Swapping in a different song throws the sync away regardless, so
    # the offset goes with it rather than silently skipping into the new one.
    # Re-attaching the *same* song keeps it, which is the case it exists for.
    if body.song_id != video.soundtrack_song_id:
        video.soundtrack_start_seconds = None

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
    bed = clamp_bed_volume(body.bed_volume) if body.include_bed else None
    safe_create_task(
        _run_soundtrack_mux(video_id, body.song_id, bed), name=f"soundtrack_mux:{video_id}",
    )
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
    video.soundtrack_start_seconds = None
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
    # Target SHORT edge in px, validated in the endpoint. 0 is RESOLUTION_KEEP:
    # run the pass for its timing stages alone and leave the picture's size
    # untouched, which is what makes interpolation and a frame rate reachable
    # without paying for the restoration.
    resolution: int = 1080
    # 1 = no interpolation; 2/3/4 run RIFE after the restore. None (the
    # default, and what an omitted field gives) means "work it out from `fps`" —
    # nobody should have to divide 24 by 8 themselves. With no `fps` either,
    # there is nothing to derive and it reads as off.
    rife_multiplier: int | None = None
    # Target playback rate. None keeps the source's rate, which with RIFE means
    # slow motion — the clip gets `rife_multiplier` times longer. A number keeps
    # the duration and raises the rate instead.
    fps: int | None = None


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


def _pass_verb(resolution: int) -> str:
    """What to call this pass in front of the user.

    The same endpoint now runs two visibly different jobs — a restoration that
    takes minutes per second of footage, and a retime that takes seconds — and
    a progress line saying "Upscaling" through the second one is a small lie
    the user has no way to check.
    """
    return "Upscaling" if clamp_resolution(resolution) > RESOLUTION_KEEP else "Retiming"


async def _upscale_plan_from_file(
    src: Path,
    resolution: int,
    fallback_w: int | None = None,
    fallback_h: int | None = None,
    rife_multiplier: int | None = None,
    target_fps: int | None = None,
) -> dict:
    """Source facts the upscale needs: output size and a wall-clock estimate.

    Dimensions come from the file rather than the row because the row records
    the *generation* canvas, which a merge or an attached soundtrack may have
    moved away from; the row's values are only the fallback for an unreadable
    file.

    `resolution` = RESOLUTION_KEEP plans a pass with no restoration in it: the
    output size is the source size and the estimate drops the term that
    dominates it, leaving what interpolation and the encode actually cost.
    """
    duration = await probe_video_duration(src)
    _, source_fps = await probe_video_frames(src)
    w, h = await probe_video_dimensions(src)
    w = w or fallback_w or resolution or RESOLUTION_DEFAULT
    h = h or fallback_h or resolution or RESOLUTION_DEFAULT
    out_w, out_h = output_dimensions(w, h, resolution)
    # A multiplier of None asks for it to be derived from the target rate —
    # 24 fps out of an 8 fps clip means 3x, and nobody should have to work that
    # out. An explicit number is honoured instead, so "2x, and write 24" stays
    # sayable.
    rate = plan_frame_rate(source_fps, target_fps, rife_multiplier)
    restore = clamp_resolution(resolution) > RESOLUTION_KEEP
    return {
        "width": out_w,
        "height": out_h,
        "source_width": w,
        "source_height": h,
        # Unrounded: the audio re-sync after an interpolated run divides by
        # this, so display precision is not good enough.
        "duration": duration,
        "rife_multiplier": rate["rife_multiplier"],
        # What the pass will actually do, so the UI can label it and
        # `_refine_render` can pick its route without re-deriving either.
        "resolution": clamp_resolution(resolution),
        "restore": restore,
        "needs_comfy": needs_comfy(resolution, rate["rife_multiplier"]),
        "seconds": estimate_seconds(
            duration, out_w, out_h, rate["rife_multiplier"], restore,
        ),
        # What the frame rate will actually do, so the UI can say it before the
        # user commits rather than after: the interpolation factor is derived
        # from the target, and an unreachable target is conformed afterwards.
        "source_fps": source_fps,
        "target_fps": rate["target_fps"],
        "render_fps": rate["render_fps"],
        "needs_conform": rate["needs_conform"],
        "duration_factor": rate["duration_factor"],
        "fps_exact": rate["exact"],
        "fps_presets": list(FPS_PRESETS),
    }


# One SEEDVR2 render at a time, process-wide. ComfyUI would queue concurrent
# prompts happily enough, but each run brackets itself with `free_memory()` to
# get a clean card — and a second run calling that while the first is sampling
# pulls the models out from under it. Bulk-upscaling a stack of clips is the
# normal case now, so the serialisation has to be here rather than in the UI.
_upscale_gate = asyncio.Lock()


async def _conform_frame_rate(
    src: Path, dest: Path, fps: int, *, has_audio: bool,
) -> None:
    """Resample `src` to exactly `fps`, duration untouched.

    `-vf fps=` drops or duplicates whole frames against the output clock, which
    is what keeps the timeline in place. It is only ever asked to close the gap
    between a reachable RIFE multiple and the requested rate, so it is dropping
    out of a denser sequence rather than inventing anything.
    """
    args = [settings.ffmpeg_path, "-y", "-v", "error", "-i", str(src),
            "-vf", f"fps={fps}", "-c:v", "libx264", "-preset", "medium",
            "-crf", "17", "-pix_fmt", "yuv420p"]
    args += ["-c:a", "copy"] if has_audio else ["-an"]
    args.append(str(dest))
    proc = await asyncio.create_subprocess_exec(
        *args, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
    )
    _, err = await communicate(proc)
    if proc.returncode != 0 or not dest.is_file():
        raise RuntimeError(
            f"frame-rate conform to {fps} failed: "
            f"{err.decode(errors='replace')[:300]}"
        )


async def _refine_render(
    src: Path,
    dest: Path,
    *,
    resolution: int,
    plan: dict,
    progress_key: str,
    prefix: str,
    log_subject: str,
) -> None:
    """Render `src` into `dest`: restore it, retime it, or both.

    Shared by the video-level pass and the per-clip pass — they differ only in
    which row they read their source from and which row they write the result
    onto, so everything between those two ends lives here.

    Three routes out, in order of what they cost:

      * nothing for the GPU (keep the resolution, no interpolation) — the whole
        pass is an ffmpeg frame-rate conform, seconds rather than minutes, and
        it never queues behind the upscale gate or evicts anything from VRAM.
      * interpolation only — a ComfyUI graph with no diffusion model in it.
      * the full restoration, with or without interpolation on top.

    The last two are the same submission; `build_upscale_workflow` decides
    which nodes are in it.
    """
    # A silent source must leave the muxer's audio slot unconnected — VHS
    # raises rather than returning an empty track when it finds no stream.
    has_audio = await probe_has_audio(src)

    if not plan.get("needs_comfy", True):
        # A pure retime. `needs_conform` is false only when the target rate is
        # the one the file already has, and then there is genuinely nothing to
        # do but put the rendition where the row expects it.
        dest.parent.mkdir(parents=True, exist_ok=True)
        if plan.get("needs_conform"):
            _set_progress(progress_key, "upscaling",
                          f"Conforming to {plan['target_fps']} fps…", 60)
            await _conform_frame_rate(
                src, dest, plan["target_fps"], has_audio=has_audio,
            )
        else:
            await asyncio.to_thread(shutil.copy2, src, dest)
        logger.info("Retime applied: %s → %s fps", log_subject, plan.get("target_fps"))
        return

    wf, save_node = build_upscale_workflow(
        src,
        resolution=resolution,
        filename_prefix=prefix,
        has_audio=has_audio,
        rife_multiplier=plan["rife_multiplier"],
        render_fps=plan.get("render_fps"),
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
                "%s submitted: %s → %dx%d, ~%ds (prompt %s)",
                "Upscale" if plan.get("restore", True) else "Interpolation",
                log_subject, plan["width"], plan["height"], plan["seconds"], prompt_id,
            )
            outputs = await poll_history(
                client, prompt_id, timeout=timeout, interval=POLL_INTERVAL,
            )
            await free_memory(client)

    comfy_src = _comfy_save_path(outputs.get(save_node, {}), "Upscale")
    dest.parent.mkdir(parents=True, exist_ok=True)

    stretched = plan["duration_factor"] != 1.0

    if plan.get("needs_conform"):
        # RIFE reaches 2/3/4 only, so an exact 24 or 60 is met by interpolating
        # past it and dropping frames here. Duration is untouched either way —
        # that is the point of rendering at the interpolated rate rather than
        # writing the target onto a denser sequence.
        await _conform_frame_rate(
            comfy_src, dest, plan["target_fps"], has_audio=has_audio,
        )
    elif stretched and has_audio:
        # No target rate: the picture got longer while frame_rate stayed put,
        # so the track VHS carried through is short (padded out with silence).
        # Same shape the generation path produces — same fix.
        await stretch_audio_to_video(
            comfy_src, dest,
            native_audio_duration=plan["duration"],
            ffmpeg_path=settings.ffmpeg_path,
        )
    else:
        await asyncio.to_thread(shutil.copy2, comfy_src, dest)


async def _apply_upscale(
    video_id: uuid.UUID, resolution: int, rife_multiplier: int | None = None,
    target_fps: int | None = None,
) -> None:
    """Run the pass from the un-upscaled source and persist it.

    Raises on failure; each caller decides how to report. Also used to
    re-render after a soundtrack change, which swaps the source file
    underneath an existing upscale.

    `rife_multiplier` None asks for the factor to be derived from `target_fps`,
    so what lands on the row is `plan["rife_multiplier"]` — what actually ran —
    rather than what was requested.
    """
    async with AsyncSessionLocal() as db:
        video = await db.get(Video, video_id)
        if not video or not video.filepath:
            raise RuntimeError("Video row gone or has no file")
        src = _upscale_source(video)
        fallback_w, fallback_h = video.width, video.height

    if not src.exists():
        raise RuntimeError(f"Source file missing: {src.name}")
    plan = await _upscale_plan_from_file(
        src, resolution, fallback_w, fallback_h, rife_multiplier, target_fps,
    )

    out_name = _upscale_name(video_id)
    await _refine_render(
        src, settings.videos_dir / out_name,
        resolution=resolution, plan=plan,
        progress_key=str(video_id),
        prefix=f"artrium_up_{video_id.hex[:10]}",
        log_subject=f"video={video_id}",
    )

    async with AsyncSessionLocal() as db:
        video = await db.get(Video, video_id)
        if video:
            video.upscale_resolution = plan["resolution"]
            video.upscale_rife = plan["rife_multiplier"]
            # Persisted so a re-render is faithful: without it a same-length
            # 3x pass would come back as slow motion the next time its source
            # changes, which is the opposite of what was asked for.
            video.upscale_fps = plan["target_fps"]
            video.upscale_filename = out_name
            video.error = None
            await db.commit()
    logger.info(
        "Upscale applied: video=%s resolution=%s rife=%dx fps=%s → %s",
        video_id, plan["resolution"] or "keep", plan["rife_multiplier"],
        plan["target_fps"] or "source", out_name,
    )


async def _run_upscale(
    video_id: uuid.UUID, resolution: int, rife_multiplier: int | None = None,
    target_fps: int | None = None,
) -> None:
    """Background task behind POST /jobs/{id}/upscale.

    Minutes per second of footage, so it is polled like the grain render
    rather than awaited inline. An existing grain is re-rendered afterwards:
    it was graded against the small picture and would otherwise be the older,
    lower-resolution file that _serialize keeps preferring.
    """
    video_key = str(video_id)
    _progress[video_key] = {
        "phase": "upscaling",
        "message": _pass_verb(resolution) + "…",
        "pct": 30,
    }
    try:
        await _apply_upscale(video_id, resolution, rife_multiplier, target_fps)
        await _reapply_grain_if_any(video_id, video_key)
        _progress.pop(video_key, None)
    except Exception as exc:
        logger.exception("Upscale failed for video=%s resolution=%s", video_id, resolution)
        _progress.pop(video_key, None)
        await _persist_post_pass_error(video_id, exc)


async def _reapply_grain_if_any(video_id: uuid.UUID, video_key: str) -> None:
    """Re-render the look pass when the file underneath it has changed.

    The look is always the last pass, so anything that rewrites its source
    invalidates it — an upscale most of all, since the graded file would
    otherwise stay the older, smaller rendition that _serialize keeps
    preferring.
    """
    async with AsyncSessionLocal() as db:
        video = await db.get(Video, video_id)
        look = _stored_look(video)
        has_file = bool(video and video.grain_filename)
    # Keyed on the file as well as the dials: a look can legitimately be all
    # zeros on a row that never had one, and re-rendering that would write a
    # graded file nobody asked for.
    if has_file and not look.is_empty:
        _progress[video_key] = {
            "phase": "graining", "message": "Re-applying look…", "pct": 85,
        }
        await _apply_look(video_id, look)


async def _refresh_derived_renders(video_id: uuid.UUID, video_key: str) -> None:
    """Re-run every post-pass a video already carries, in production order.

    Called when the *source* of those passes changes — attaching or dropping a
    soundtrack rewrites the file both the upscale and the grain read from, so
    leaving them alone would keep serving a rendition built on the old audio.
    Order matters: upscale first, grain on top of it.
    """
    async with AsyncSessionLocal() as db:
        video = await db.get(Video, video_id)
        has_pass = bool(video and video.upscale_filename)
        resolution = (video.upscale_resolution if video else None) or RESOLUTION_KEEP
        rife = (video.upscale_rife if video else None) or 1
        target_fps = video.upscale_fps if video else None

    # Keyed on the file rather than on the resolution: a retime-only pass has
    # RESOLUTION_KEEP on the row, and testing the number would skip exactly the
    # renditions that most need rebuilding.
    if has_pass:
        _progress[video_key] = {
            "phase": "upscaling", "message": "Re-running upscale…", "pct": 40,
        }
        # The stored target rate comes along, so a same-length interpolation
        # stays same-length. Re-running is for putting grain back on top, not
        # for redeciding the timing.
        await _apply_upscale(video_id, resolution, rife, target_fps)
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


def _validate_pass_settings(
    resolution: int, rife_multiplier: int | None, target_fps: int | None = None,
) -> None:
    """The three dials, checked together — shared by the video and clip paths.

    Each is optional on its own now, which makes one new way to be wrong: a
    request that asks for no restoration, no interpolation and no rate is a
    render with nothing in it. Saying so is better than producing a byte-copy
    of the source and calling it an upscale.
    """
    if resolution != RESOLUTION_KEEP and not RESOLUTION_MIN <= resolution <= RESOLUTION_MAX:
        raise HTTPException(
            status_code=422,
            detail=f"resolution must be {RESOLUTION_KEEP} (keep) or between "
                   f"{RESOLUTION_MIN} and {RESOLUTION_MAX}",
        )
    if rife_multiplier is not None and rife_multiplier not in RIFE_MULTIPLIERS:
        raise HTTPException(
            status_code=422,
            detail=f"rife_multiplier must be null (derive from fps) or one of "
                   f"{list(RIFE_MULTIPLIERS)}",
        )
    if (resolution == RESOLUTION_KEEP
            and (rife_multiplier or 1) <= 1
            and target_fps is None):
        raise HTTPException(
            status_code=422,
            detail="Nothing to do — pick a resolution, an interpolation factor "
                   "or a frame rate",
        )


def _validate_upscale_target(
    video: Video | None, resolution: int, rife_multiplier: int | None = None,
    target_fps: int | None = None,
) -> None:
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if video.status != "done" or not video.filename:
        raise HTTPException(status_code=409, detail="Video is not ready (status must be 'done')")
    _validate_pass_settings(resolution, rife_multiplier, target_fps)
    # Interpolation *without a target rate* stretches whatever audio the file
    # carries. That is exactly right for a model's own generated track, which
    # was sampled against the pre-RIFE frame count — and exactly wrong for an
    # attached song, where a 3x time-stretch destroys the music rather than
    # re-syncing it. With a target rate the duration holds and the song is
    # untouched, so that combination is allowed: the incompatibility is with
    # slow motion, not with interpolation.
    if (rife_multiplier or 1) > 1 and target_fps is None and video.soundtrack_song_id:
        raise HTTPException(
            status_code=409,
            detail="Interpolation without a target frame rate would time-stretch "
                   "the attached soundtrack — pick a frame rate so the length "
                   "holds, or remove the song first",
        )


@router.get("/jobs/{video_id}/upscale/estimate")
async def estimate_upscale(
    video_id: uuid.UUID,
    resolution: int = 1080,
    rife_multiplier: int | None = None,
    fps: int | None = None,
    db: AsyncSession = Depends(get_db),
):
    """Target size and expected wall-clock for a pass, before committing.

    With the restoration in it this runs for minutes per second of footage —
    long enough that starting it blind is a real cost — and the number is cheap
    to produce (two ffprobe calls). Without it, the estimate is what tells the
    user a retime is a matter of seconds rather than minutes.
    """
    video = await db.get(Video, video_id)
    _validate_upscale_target(video, resolution, rife_multiplier, clamp_fps(fps))
    src = _upscale_source(video)
    if not src.exists():
        raise HTTPException(status_code=409, detail="Source video file is missing on disk")
    return await _upscale_plan_from_file(
        src, resolution, video.width, video.height, rife_multiplier, clamp_fps(fps),
    )


@router.post("/jobs/{video_id}/upscale", status_code=202)
async def apply_upscale(
    video_id: uuid.UUID, body: UpscaleApply, db: AsyncSession = Depends(get_db),
):
    video = await db.get(Video, video_id)
    target_fps = clamp_fps(body.fps)
    _validate_upscale_target(video, body.resolution, body.rife_multiplier, target_fps)
    resolution = clamp_resolution(body.resolution)
    # None survives clamping here on purpose: it is the request to derive the
    # factor from the target rate, and clamp_rife would flatten it to 1.
    rife = None if body.rife_multiplier is None else clamp_rife(body.rife_multiplier)

    # Clear a stale error from an earlier attempt so the frontend poller can't
    # read it as this attempt failing before the render has even started.
    if video.error is not None:
        video.error = None
        await db.commit()

    _progress[str(video_id)] = {
        "phase": "upscaling", "message": _pass_verb(resolution) + "…", "pct": 5,
    }
    safe_create_task(_run_upscale(video_id, resolution, rife, target_fps),
                     name=f"upscale:{video_id}")
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
    video.upscale_fps = None
    await db.commit()
    await db.refresh(video)

    # An existing look was rendered from the upscaled file, which just went
    # away — re-render it from the small source rather than keep serving a
    # 1080p graded file the row no longer claims to have.
    look = _stored_look(video)
    if video.grain_filename and not look.is_empty:
        _progress[str(video_id)] = {
            "phase": "graining", "message": "Re-applying look…", "pct": 10,
        }
        safe_create_task(_run_look(video_id, look), name=f"look:{video_id}")
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
    clip_id: uuid.UUID, resolution: int, rife_multiplier: int | None = None,
    target_fps: int | None = None,
) -> None:
    """Render the pass for one clip and persist it. Raises on failure."""
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
    plan = await _upscale_plan_from_file(
        src, resolution, fallback_w, fallback_h, rife_multiplier, target_fps,
    )

    await _refine_render(
        src, dest,
        resolution=resolution, plan=plan,
        progress_key=_clip_key(clip_id),
        prefix=f"artrium_clipup_{clip_id.hex[:10]}",
        log_subject=f"clip={clip_id}",
    )

    async with AsyncSessionLocal() as db:
        clip = await db.get(VideoClip, clip_id)
        if clip:
            clip.upscale_resolution = plan["resolution"]
            clip.upscale_rife = plan["rife_multiplier"]
            clip.upscale_fps = plan["target_fps"]
            clip.upscale_filename = out_name
            clip.upscale_width = plan["width"]
            clip.upscale_height = plan["height"]
            await db.commit()
    logger.info(
        "Clip upscale applied: clip=%s → %dx%d rife=%dx fps=%s",
        clip_id, plan["width"], plan["height"], plan["rife_multiplier"],
        plan["target_fps"] or "source",
    )


async def _run_clip_upscale(
    clip_id: uuid.UUID, resolution: int, rife_multiplier: int | None = None,
    target_fps: int | None = None,
) -> None:
    """Background task behind POST /clips/{id}/upscale."""
    key = _clip_key(clip_id)
    _progress[key] = {"phase": "upscaling", "message": "Upscaling…", "pct": 10}
    try:
        await _apply_clip_upscale(clip_id, resolution, rife_multiplier, target_fps)
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
    clip: VideoClip | None, resolution: int, rife_multiplier: int | None,
    target_fps: int | None = None,
) -> None:
    if not clip:
        raise HTTPException(status_code=404, detail="Clip not found")
    _validate_pass_settings(resolution, rife_multiplier, target_fps)


@router.get("/clips/{clip_id}/upscale/estimate")
async def estimate_clip_upscale(
    clip_id: uuid.UUID,
    resolution: int = 1080,
    rife_multiplier: int | None = None,
    fps: int | None = None,
    db: AsyncSession = Depends(get_db),
):
    """Target size and expected wall-clock for one clip's pass."""
    clip = await db.get(VideoClip, clip_id)
    _validate_clip_upscale(clip, resolution, rife_multiplier, clamp_fps(fps))
    src = _clip_file(clip)
    if not src.exists():
        raise HTTPException(status_code=409, detail="Clip file is missing on disk")
    return await _upscale_plan_from_file(
        src, resolution, clip.width, clip.height, rife_multiplier, clamp_fps(fps),
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
    target_fps = clamp_fps(body.fps)
    _validate_clip_upscale(clip, body.resolution, body.rife_multiplier, target_fps)
    resolution = clamp_resolution(body.resolution)
    rife = None if body.rife_multiplier is None else clamp_rife(body.rife_multiplier)

    if _is_clip_upscaling(clip):
        raise HTTPException(status_code=409, detail="This clip is already being upscaled")

    _progress[_clip_key(clip_id)] = {
        "phase": "upscaling", "message": "Queued…", "pct": 5,
    }
    safe_create_task(
        _run_clip_upscale(clip_id, resolution, rife, target_fps),
        name=f"clip_upscale:{clip_id}",
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
    clip.upscale_fps = None
    clip.upscale_width = None
    clip.upscale_height = None
    await db.commit()
    _progress.pop(_clip_key(clip_id), None)
    await db.refresh(clip)
    return _serialize_clip(clip)


# ── Look pass (post-hoc grade over a finished video) ──────────────────────────
# Wan2.2 and AnimateLCM come out clean to the point of looking plastic, and
# this pass is what is between a render and delivery — so it carries the whole
# correction, not just the grain it started as. Seven dials, one ffmpeg chain,
# one derived sibling file: `filename` stays pristine, so any look can be
# tried, undone, or re-tried. It runs last, on top of any upscale, so the grade
# sits at the delivery resolution. See services/video/look.py for the order the
# dials are applied in and the measurements behind the encoder settings.

class LookApply(BaseModel):
    # The seven dials, as services/video/look.py::Look reads them. Sent whole
    # rather than as a patch: the frontend always knows the complete look it is
    # showing, and a partial update would make "what will this render" depend
    # on what the row happened to hold.
    look: dict = {}
    # A preset name fills in for `look` when the client just wants a starting
    # point. `look` wins when both are given, so the sliders always beat the
    # chip they were seeded from.
    preset: str | None = None

    def resolved(self) -> Look:
        if self.look:
            return Look.from_dict(self.look)
        if self.preset and self.preset in PRESET_BY_KEY:
            return PRESET_BY_KEY[self.preset]
        return Look()


def _look_source(video: Video) -> Path:
    """The ungraded file the look pass should read.

    Deliberately never `grain_filename` itself: re-grading always starts from
    a clean source, so moving a slider replaces the look instead of baking a
    second pass on top of the first. Otherwise the most complete rendition
    wins — the upscale when there is one, else the muxed (audio-bearing)
    variant so an attached soundtrack survives the re-encode.
    """
    if video.upscale_filename:
        return settings.videos_dir / video.upscale_filename
    if video.muxed_filename:
        return settings.videos_dir / video.muxed_filename
    return settings.storage_dir / video.filepath


# Both names predate the other six dials. They stay: renaming them would
# orphan every file already on disk for the sake of a word.
def _look_name(video_id: uuid.UUID) -> str:
    return f"{video_id}_grain.mp4"


def _look_preview_name(video_id: uuid.UUID) -> str:
    return f"{video_id}_grainprev.mp4"


def _stored_look(video: Video | None) -> Look:
    """The look a row is carrying.

    Rows written before `look_params` existed have only `grain_strength`, which
    is exactly a grain-only look — so they read back as one rather than as
    nothing.
    """
    if not video:
        return Look()
    if video.look_params:
        return Look.from_dict(video.look_params)
    return Look(grain=clamp_look_strength(video.grain_strength))


async def _apply_look(video_id: uuid.UUID, look: Look) -> None:
    """Render the look pass from the ungraded source and persist it.

    Raises on failure; each caller decides how to report. Also used to
    re-render after a soundtrack or upscale change, since those swap the source
    file underneath an existing look.
    """
    async with AsyncSessionLocal() as db:
        video = await db.get(Video, video_id)
        if not video or not video.filepath:
            raise RuntimeError("Video row gone or has no file")
        src = _look_source(video)

    if not src.exists():
        raise RuntimeError(f"Source file missing: {src.name}")

    out_name = _look_name(video_id)
    await render_look(
        src, settings.videos_dir / out_name, look,
        ffmpeg_path=settings.ffmpeg_path,
    )

    async with AsyncSessionLocal() as db:
        video = await db.get(Video, video_id)
        if video:
            video.look_params = look.to_dict()
            # Mirrored, not derived-on-read: services/improv and the gallery
            # ask "is this grained" of this column, and they should not have to
            # learn about look_params to get an answer.
            video.grain_strength = look.grain
            video.grain_filename = out_name
            video.error = None
            await db.commit()
    logger.info("Look applied: video=%s %s → %s", video_id, look.to_dict(), out_name)


async def _run_look(video_id: uuid.UUID, look: Look) -> None:
    """Background task behind POST /jobs/{id}/look — a full re-encode of a
    16-30s clip runs well past a request's patience, so it is polled like the
    soundtrack mux rather than awaited inline."""
    video_key = str(video_id)
    _progress[video_key] = {"phase": "graining", "message": "Grading…", "pct": 50}
    try:
        await _apply_look(video_id, look)
        _progress.pop(video_key, None)
    except Exception as exc:
        logger.exception("Look render failed for video=%s look=%s", video_id, look)
        _progress.pop(video_key, None)
        await _persist_post_pass_error(video_id, exc)


def _validate_look_target(video: Video | None, look: Look) -> None:
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if video.status != "done" or not video.filename:
        raise HTTPException(status_code=409, detail="Video is not ready (status must be 'done')")
    # Every dial clamps rather than rejecting, so the only unusable request is
    # one that asks for nothing at all — which is a removal, not a render.
    if look.is_empty:
        raise HTTPException(
            status_code=422,
            detail="Nothing to apply — set at least one dial, or DELETE to remove the look",
        )


@router.get("/look/presets")
async def look_presets():
    """The presets, values included, so picking a chip fills the sliders
    without a second round trip."""
    return {"presets": preset_options(), "default": DEFAULT_PRESET}


@router.post("/jobs/{video_id}/look/preview")
async def preview_look(
    video_id: uuid.UUID, body: LookApply, db: AsyncSession = Depends(get_db),
):
    """Grade a few seconds out of the middle of the clip and return its URL.

    Runs inline: the point is a tight look-adjust-look loop, and a 4s excerpt
    encodes in a second or two. The file is overwritten on every call, so the
    caller must cache-bust the URL it gets back.
    """
    video = await db.get(Video, video_id)
    look = body.resolved()
    _validate_look_target(video, look)

    src = _look_source(video)
    if not src.exists():
        raise HTTPException(status_code=409, detail="Source video file is missing on disk")

    out_name = _look_preview_name(video_id)
    try:
        seconds = await render_look_preview(
            src, settings.videos_dir / out_name, look,
            ffmpeg_path=settings.ffmpeg_path,
        )
    except Exception as exc:
        logger.exception("Look preview failed for video=%s", video_id)
        raise HTTPException(status_code=502, detail=f"Preview failed: {exc}")

    return {
        "url": f"/api/video/file/{out_name}",
        "look": look.to_dict(),
        "seconds": round(seconds, 2),
    }


@router.post("/jobs/{video_id}/look", status_code=202)
async def apply_look(
    video_id: uuid.UUID, body: LookApply, db: AsyncSession = Depends(get_db),
):
    video = await db.get(Video, video_id)
    look = body.resolved()
    _validate_look_target(video, look)

    # Clear a stale error from an earlier attempt so the frontend poller can't
    # read it as this attempt failing before the render has even started.
    if video.error is not None:
        video.error = None
        await db.commit()

    _progress[str(video_id)] = {"phase": "graining", "message": "Grading…", "pct": 10}
    safe_create_task(_run_look(video_id, look), name=f"look:{video_id}")
    return _serialize(video)


@router.delete("/jobs/{video_id}/look")
async def remove_look(video_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    video = await db.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if video.grain_filename:
        (settings.videos_dir / video.grain_filename).unlink(missing_ok=True)
    (settings.videos_dir / _look_preview_name(video_id)).unlink(missing_ok=True)
    video.grain_filename = None
    video.grain_strength = None
    video.look_params = None
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


def _cut_summary(plan: dict | None) -> dict | None:
    """The headline of a beat cut, for the card.

    The stored plan holds every shot; a card needs four numbers. Sending the
    whole thing down with every video in the list would be the bulk of the
    response for no one's benefit — the planner endpoint serves the full plan
    when something actually wants it.
    """
    if not plan:
        return None
    return {
        "style":  plan.get("style"),
        "seed":   plan.get("seed"),
        "bpm":    plan.get("bpm"),
        # A beat cut counts shots and a layer cut counts slots. One number on
        # the card either way — the field is "how many pieces is this made of".
        "cuts":   len(plan.get("cuts") or plan.get("slots") or []),
        # Only a layer cut carries these, and the card uses their presence to
        # tell the two apart without a second field to keep in step.
        "tracks": plan.get("tracks"),
        "transition": plan.get("transition"),
    }


def _serialize(
    v: Video, clip_count: int = 0, clip_thumb: str | None = None,
) -> dict:
    """One video row for the API.

    `clip_count`/`clip_thumb` describe the job's segment stack, and they exist
    because a job can legitimately be finished without ever producing a single
    file: the review flow renders each segment as its own clip and only
    assembles them into one video when asked to. Such a row has no `filename`,
    so it used to serialise with no `url` and no `thumb_url` at all — and the
    gallery drew it as an empty tile that could not be opened or played, even
    though the clips behind it were real work sitting on disk.
    """
    # Derived variants take precedence in the order they are produced (see
    # _video_primary_name); the clean original stays available via
    # `original_url`.
    primary_name = _video_primary_name(v)
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
        "soundtrack_bed_volume": v.soundtrack_bed_volume,
        "soundtrack_start_seconds": v.soundtrack_start_seconds,
        "cut_plan":          _cut_summary(v.cut_plan),
        "upscale_resolution": v.upscale_resolution,
        "upscale_rife":      v.upscale_rife,
        "upscale_fps":       v.upscale_fps,
        "has_upscale":       bool(v.upscale_filename),
        "upscale_rendering": _is_upscaling(v),
        "grain_strength":    v.grain_strength,
        "has_grain":          bool(v.grain_filename),
        "grain_rendering":    _is_graining(v),
        # The whole look, so the chip can draw seven sliders where the row was
        # written by an older version that only knew about grain.
        "look":              _stored_look(v).to_dict(),
        # Any row with a file of its own can show a picture, and the endpoint
        # makes one on demand if it is missing. The gate used to also require
        # status "done", which hid exactly the jobs a person most needs to
        # see: an AnimateLCM render parked in `review` has a finished,
        # watchable preview on disk and was drawn as an empty tile.
        # Falls back to the stack's first clip, so a job that was never
        # assembled still shows what it actually contains.
        "thumb_url": (
            f"/api/video/thumb/{v.id}" if primary_name else (clip_thumb or None)
        ),
        # >0 with no `url` means "a stack that was never assembled" — the
        # gallery opens it as clips instead of as a video.
        "clip_count":        clip_count,
        "assembled":         bool(primary_name),
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
