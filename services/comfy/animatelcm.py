"""
AnimateLCM structure-video builder — a faithful rebuild of the user's own
`AnimateDiff LCM Workflow.json`, plus the handles it never exposed.

Read the original before changing anything here. Its quality does not come from
the base render, which is a 544x544 nine-step LCM pass, but from what happens
after it:

    base      544x544, AnimateLCM + depth ControlNet + IP-Adapter, 9 steps, cfg 1
    hires     4x-UltraSharp, scaled back to 0.5 (net 2x), VAE-encoded and
              **re-sampled at denoise 0.4**, guided by lineart + depth
              ControlNets derived from the base render
    RIFE      4x interpolation to 30 fps

The hires pass is the whole story. It is a second diffusion pass that keeps 60%
of what is already there and invents the rest at twice the resolution, and it is
guided by two ControlNets taken from the base frames, so it cannot wander.

**It does not flicker, and the reason matters.** A per-frame diffusion upscale
boils, because every frame draws its own noise. This one does not, because the
model handed to the hires sampler is still the AnimateDiff-wrapped model — the
motion module sees the whole batch in the second pass exactly as in the first.
Temporal coherence is not bolted on afterwards; it is a property of the sampler.

This is why the VACE detour underperformed. VACE *repaints* — it starts from
noise and reconstructs the frame from a control signal, so whatever texture the
source had is gone by construction. The pass below *refines*: at denoise 0.4 the
picture survives and gains detail. Those are different operations, and only one
of them was ever going to preserve material.

Faithful defaults, and where they came from
───────────────────────────────────────────
Everything below is set to the original file's value unless marked. Deviations
are the two places where the original's own settings are demonstrably a
workaround rather than a choice.

  beta_schedule    "sqrt_linear (AnimateDiff)" — NOT the `lcm[100_ots]` the
                   AnimateLCM authors suggest. The original used sqrt_linear and
                   is the reference result, so it wins over the documentation.
  sampler          lcm / karras, 9 steps, cfg 1.0
  motion LoRA      shatterAnimatediff_v10 at 0.45, motion scale 1.2
  noise            FreeNoise — a looping noise pattern across context windows,
                   which is what keeps a 94-frame clip from drifting
  IP-Adapter       weight 1.0, "K+V w/ C penalty", end_at 0.8. Not "V only":
                   K+V carries composition as well as colour, and at end_at 0.8
                   it lets go before the last fifth so the sampler can settle.
                   With regions, an unmasked base adapter goes underneath the
                   chain — see BASE_IP_DEFAULT for the measurements that made
                   it necessary.
  depth ControlNet 0.45, end_percent 0.7
  hires            denoise 0.4, 11 steps; lineart CN 0.3/end 0.5, depth CN
                   0.5/end 0.6

Embeddings — kentskooking's method
──────────────────────────────────
Textual-inversion embeddings are the third material channel next to the prompt
and the IP-Adapter picture. How they are joined is taken from kentskooking's
own account (Purz, Creative Exploration EP95), not from documentation:

  concat, never combine   Each embedding is encoded on its own and the chunks
                          are concatenated along the token axis. ConditioningCombine
                          would run a second UNet evaluation per step (roughly
                          double the VRAM and time); concat of thirty embeddings
                          costs what one does.
  the blank sandwich      An *empty* encode between two embeddings makes the mix
                          better. Found by trial, not explained — a blank
                          encode is not zeros, it is CLIP's BOS/EOS padding.
  inline                  All in one prompt, the way A1111 users write it. Kept
                          as the baseline the other two are measured against.

The stack ends in the node the samplers already read (`pos`), so base and hires
pass see the same material. Unmeasured on this graph so far — in particular
whether embeddings survive the LCM LoRA at cfg 1.0, or need Kent's plain
euler/20 steps, which `lcm_lora_strength=0` plus `sampler`/`scheduler` allow.
"""
from __future__ import annotations

import copy
import math
import random
import re
from dataclasses import dataclass, field

# ── Models ───────────────────────────────────────────────────────────────────
# The original names this `sd_lcm\\juggernaut_reborn.safetensors`; on this
# machine the same file sits in `sd_15\\`.
CHECKPOINT = "sd_15\\juggernaut_reborn.safetensors"
MOTION_MODEL = "AnimateLCM_sd15_t2v.ckpt"
MOTION_LORA = "AnimateLCM_sd15_t2v_lora.safetensors"
MOTION_LORA_STRENGTH = 1.0          # strength_clip stayed at 0 in the original
AD_MOTION_LORA = "shatterAnimatediff_v10.safetensors"
AD_MOTION_LORA_STRENGTH = 0.45
MOTION_SCALE = 1.2

DEPTH_CONTROLNET = "control_v11f1p_sd15_depth.pth"
LINEART_CONTROLNET = "control_v11p_sd15_lineart.pth"
UPSCALE_MODEL = "4x-UltraSharp.pth"

BETA_SCHEDULE = "sqrt_linear (AnimateDiff)"
NOISE_TYPE = "FreeNoise"
IPADAPTER_PRESET = "PLUS (high strength)"

SAMPLER = "lcm"
SCHEDULER = "karras"
DEFAULT_STEPS = 9
DEFAULT_CFG = 1.0

# The original writes its base at 8 fps and its interpolated result at 30. The
# slowness is part of the look, not an accident: the motion module produces one
# increment of movement per frame whatever the playback rate.
FPS = 8
DEFAULT_LENGTH = 94

CONTEXT_LENGTH = 16
CONTEXT_OVERLAP = 3
CONTEXT_STRIDE = 1

# The original renders 544x544 and widens afterwards by outpainting 208px per
# side. The outpaint is skipped here, and not replaced by a wider preset either:
# the control track already carries the aspect the user framed the scene in, so
# the canvas is derived from it. Nothing is padded, nothing is cropped, and
# there is no flag to get wrong.
#
# The budget is the original's own pixel count. Holding *area* rather than an
# edge is what keeps every aspect equally expensive and equally well-behaved —
# SD 1.5 duplicates content when the pixel count climbs, not when a particular
# edge does.
PIXEL_BUDGET = 544 * 544

# ...with one guard on top: past roughly 768 on an edge SD 1.5 starts repeating
# motifs even inside the budget, so an extreme ratio gives up area rather than
# stretch further.
LONG_EDGE_MAX = 768

# Named presets remain, for a caller that wants a shape the source does not have.
_CANVASES: dict[str, tuple[int, int]] = {
    "square": (544, 544),
    "wide": (768, 432),
    "tall": (432, 768),
}
# "auto" = follow the control track. Anything else is a deliberate override, and
# then `fit` decides how the mismatch is resolved.
DEFAULT_ASPECT = "auto"

# ── Dials ────────────────────────────────────────────────────────────────────
DEPTH_MIN, DEPTH_MAX = 0.10, 1.50
DEPTH_DEFAULT = 0.45
DEPTH_END_DEFAULT = 0.70

IP_MIN, IP_MAX = 0.10, 1.50
IP_DEFAULT = 1.00
IP_END_DEFAULT = 0.80

# ── The base layer ───────────────────────────────────────────────────────────
# Masked IP-Adapters only condition what their mask covers. Everything else
# runs on the text prompt alone — and at cfg 1.0 there is no unconditional pass
# to temper it, so the checkpoint's own palette takes over completely.
#
# Measured 2026-09-04 on the renders that prompted this: three SAM 3 masks
# covered 17.6-18.5% of the frame, so **82% of every frame had no reference at
# all**. Saturation inside a mask read 89.5 against 175.9 outside it (PIL HSV
# S), and the finished clips came out 1.6-2.8x more saturated than the pictures
# they were supposed to be made of — where the single-reference path, which has
# no masks and therefore covers everything, landed *below* its reference.
#
# So a region render also gets an unmasked adapter underneath the chain. It is
# its own picture rather than a repeat of a region's: what it is for is the
# ground the regions sit in, which is usually not what any of them shows.
# Lower by default than a region's own weight — it sets the palette, it does
# not compete with the material on top of it.
BASE_IP_DEFAULT = 0.55

# The hires denoise, and the one number in this file that was measured rather
# than inherited. The original sat at 0.40; the sweep of 2026-08-19 found more
# detail all the way to 0.60 with the composition untouched, and — against the
# expectation that more invention costs stability — the 0.60 clip is *steadier*
# than the 0.40 one over 94 frames (mean frame diff 2.78 vs 2.95 at 384px, max
# 4.10 vs 4.57, no spikes in either). The likely reason: at a higher denoise the
# second pass resolves each frame more completely against the same two
# ControlNets, so less of the soft, half-formed base survives to wobble.
#
# 0.60 is therefore the default. The ceiling is left above it because nothing
# broke at the top of the sweep, but 0.65-0.75 is untested — and past that the
# pass stops refining and starts repainting, which is the failure this whole
# graph exists to avoid.
HIRES_DENOISE_MIN, HIRES_DENOISE_MAX = 0.20, 0.75
HIRES_DENOISE_DEFAULT = 0.60
HIRES_STEPS = 11
HIRES_SCALE = 0.5                   # after a 4x model: net 2x
HIRES_LINEART_STRENGTH = 0.30
HIRES_LINEART_END = 0.50
HIRES_DEPTH_STRENGTH = 0.50
HIRES_DEPTH_END = 0.60
PREPROCESSOR_RESOLUTION = 1024

RIFE_MODEL = "rife47.pth"
RIFE_MULTIPLIER = 4

DEPTH_SWEEP = (0.25, 0.45, 0.65, 0.85)
IP_SWEEP = (0.60, 0.80, 1.00, 1.20)
HIRES_SWEEP = (0.30, 0.40, 0.50, 0.60)

# The original left both negatives empty, and at cfg 1.0 that costs nothing:
# there is no unconditional pass for a negative prompt to act through.
DEFAULT_NEGATIVE = ""

DEFAULT_INVERT_DEPTH = True
_FIT_METHODS = {"pad": "pad", "crop": "fill / crop", "stretch": "stretch"}
FITS = tuple(_FIT_METHODS)
DEFAULT_FIT = "pad"

# ── Embeddings ───────────────────────────────────────────────────────────────
EMBEDDING_JOINS = ("inline", "concat", "sandwich")
# Measured 2026-09-28 (scripts/embedding_sweep.py, cube track, seed 1234):
# inline keeps the prompt's scene and lays the embedding over it; concat and
# sandwich drown the prompt — it is one chunk among equals there, and the rows
# looked like the no-prompt render. Sandwich against concat moved the picture
# by 10.8 where real variants differ by 30-50. So inline is the default for a
# render that has a prompt, and the other two stay for kentskooking's own case,
# which has none.
DEFAULT_EMBEDDING_JOIN = "inline"
EMBEDDING_WEIGHT_MIN, EMBEDDING_WEIGHT_MAX = 0.0, 2.0
# ComfyUI finds an embedding by the word after `embedding:`. A space ends the
# word, a colon or bracket is read as weight syntax, and a `..` segment leaves
# the embeddings folder, which ComfyUI refuses — all of them silently: it logs
# a warning and renders without the embedding.
_EMBEDDING_NAME = re.compile(r"^(?!.*(^|/)\.+(/|$))[\w.\-]+(/[\w.\-]+)*$")


def is_embedding_name(name: str | None) -> bool:
    return bool(_EMBEDDING_NAME.match(name or ""))


# ── Embeddings over time ─────────────────────────────────────────────────────
# kentskooking's signature: the vibe changes across the clip. He schedules
# conditionings with a keyframe node pack because embeddings do not survive a
# batch prompt schedule (only the first one registers). The same effect is made
# here from core nodes: an embedding with a curve becomes its own conditioning
# layer — prompt plus that embedding — carrying a per-frame mask, and the base
# layer takes whatever the curves leave. ComfyUI averages the layers'
# predictions weighted by their masks, and AnimateDiff-Evolved slices a mask
# whose batch equals the frame count to each context window
# (animatediff/sampling.py::get_resized_cond), so a curve reaches every frame.
#
# The price is a UNet evaluation per layer per step — one curve roughly doubles
# the render — which is the same trade Kent describes for Conditioning Combine.
#
# Curves are computed here, per frame, and handed to KJNodes'
# CreateFadeMaskAdvanced as explicit keyframes, so no interpolation mode on the
# node side can bend them.
CURVES = ("fade_in", "fade_out", "swell", "pulse")
CYCLES_MIN, CYCLES_MAX, CYCLES_DEFAULT = 1, 8, 2
# The base layer never drops to zero: a frame whose masks all read 0 would be
# divided by nothing when ComfyUI normalises the layers.
_BASE_FLOOR = 0.001
_CURVE_MASK_SIZE = 64          # resized to the latent by ComfyUI anyway


def curve_values(curve: str, frames: int, cycles: int = CYCLES_DEFAULT) -> list[float]:
    """How much of each frame an embedding owns, 0..1.

    `swell` and `pulse` start and end at the same value, so a clip rendered
    with closed_loop still loops; `fade_in`/`fade_out` deliberately do not.
    """
    n = max(1, int(frames))
    out = []
    for f in range(n):
        t = f / (n - 1) if n > 1 else 1.0
        if curve == "fade_in":
            v = 0.5 - 0.5 * math.cos(math.pi * t)
        elif curve == "fade_out":
            v = 0.5 + 0.5 * math.cos(math.pi * t)
        elif curve == "swell":
            v = math.sin(math.pi * t) ** 2
        elif curve == "pulse":
            k = min(max(int(cycles), CYCLES_MIN), CYCLES_MAX)
            v = 0.5 - 0.5 * math.cos(2 * math.pi * k * f / n)
        else:
            raise ValueError(f"unknown curve: {curve}")
        out.append(round(v, 3))
    return out


def base_curve(curves: list[list[float]], frames: int) -> list[float]:
    """What the prompt layer owns: everything the moving layers leave."""
    return [round(max(_BASE_FLOOR, 1.0 - sum(c[f] for c in curves)), 3)
            for f in range(frames)]


def fade_points(values: list[float]) -> str:
    """CreateFadeMaskAdvanced's keyframe syntax, one key per frame."""
    return ",\n".join(f"{i}:({v:.3f})" for i, v in enumerate(values))


@dataclass
class Embedding:
    """One textual-inversion file in ComfyUI's embeddings folder, by stem.

    `curve` makes it move through the clip (see CURVES); None holds it for the
    whole clip, which is how every render worked before curves existed.
    """
    name: str
    weight: float = 1.0
    curve: str | None = None
    cycles: int = CYCLES_DEFAULT


@dataclass
class Region:
    """One colour region of the mask video and what it is made of: a picture
    (IP-Adapter, masked), embeddings (a conditioning layer, masked), or both.

    kentskooking's RGB masks carry a stack of embeddings per colour; with
    embeddings a region no longer needs a picture. Region embeddings hold for
    the whole clip — a curve on one is ignored.
    """
    color: tuple[int, int, int]
    reference: str | None = None
    weight: float = IP_DEFAULT
    threshold: int = 20
    invert: bool = False
    embeddings: list[Embedding] = field(default_factory=list)


# ── Image injection ──────────────────────────────────────────────────────────
# kentskooking's "sample settings image injection": a video is composited into
# the sampler's own estimate of the finished frames partway through, and the
# sampler carries on from there. Implemented by AnimateDiff-Evolved
# (sampling.py::perform_image_injection): the current x0 is decoded, the
# injected frames are laid over it (masked, if a mask is given), re-encoded,
# the noise residual is added back, and the result is blended in at `strength`.
# He uses pixelated masks, other generators' clips and TV static; it "gets
# around the context window" because the same frames steer every window.
#
# ADE splits the sampler at the injection's start and skips an injection whose
# start lies before a sampler's own range — which is what keeps it out of the
# hires pass, as long as the start stays below 1 - hires_denoise.
INJECT_STRENGTH_MIN, INJECT_STRENGTH_MAX, INJECT_STRENGTH_DEFAULT = 0.05, 1.0, 0.30
INJECT_START_MIN, INJECT_START_MAX, INJECT_START_DEFAULT = 0.05, 0.60, 0.30
# Margin between the injection and the hires pass's first step.
_INJECT_HIRES_MARGIN = 0.05


@dataclass
class Injection:
    """A video laid into the base pass partway through."""
    video: str                                  # absolute path
    strength: float = INJECT_STRENGTH_DEFAULT
    start: float = INJECT_START_DEFAULT         # fraction of the schedule
    # Only inside this colour of the mask video; None = the whole frame.
    color: tuple[int, int, int] | None = None
    threshold: int = 20
    # Frames the source has; a shorter one is looped to the clip's length.
    source_frames: int | None = None


def injection_start(inj: Injection, hires: bool, hires_denoise: float) -> float:
    """The start actually used: clamped to its band and, with a hires pass,
    kept before that pass's first step so the refine never receives it."""
    start = clamp(inj.start, INJECT_START_MIN, INJECT_START_MAX, INJECT_START_DEFAULT)
    if hires:
        start = min(start, round(1.0 - hires_denoise - _INJECT_HIRES_MARGIN, 2))
    return max(INJECT_START_MIN, start)


@dataclass
class AnimateLcmRequest:
    control_video: str
    prompt: str
    negative: str = DEFAULT_NEGATIVE
    reference_image: str | None = None
    mask_video: str | None = None
    regions: list[Region] = field(default_factory=list)
    aspect: str = DEFAULT_ASPECT
    fit: str = DEFAULT_FIT
    width: int | None = None
    height: int | None = None
    length: int = DEFAULT_LENGTH
    fps: int = FPS

    # Base pass
    depth_strength: float = DEPTH_DEFAULT
    depth_end: float = DEPTH_END_DEFAULT
    ip_weight: float = IP_DEFAULT
    ip_end: float = IP_END_DEFAULT
    # The unmasked layer under a region chain. None means the regions are the
    # only conditioning there is, which is how this used to behave and is
    # almost never what anyone wants — see BASE_IP_DEFAULT.
    base_reference: str | None = None
    base_weight: float = BASE_IP_DEFAULT
    ip_scaling: str = "K+V w/ C penalty"
    # Joined into the positive in the order given; see the module docstring.
    embeddings: list[Embedding] = field(default_factory=list)
    embedding_join: str = DEFAULT_EMBEDDING_JOIN
    injection: Injection | None = None
    steps: int = DEFAULT_STEPS
    cfg: float = DEFAULT_CFG
    sampler: str = SAMPLER
    scheduler: str = SCHEDULER
    # 0 drops the LCM distillation LoRA — Kent's recipe keeps the AnimateLCM
    # motion module but samples plainly (euler/normal, ~20 steps, cfg > 1).
    lcm_lora_strength: float = MOTION_LORA_STRENGTH
    beta_schedule: str = BETA_SCHEDULE
    motion_lora: str | None = AD_MOTION_LORA
    motion_lora_strength: float = AD_MOTION_LORA_STRENGTH
    motion_scale: float = MOTION_SCALE
    seed: int = -1

    # Hires pass — the detail engine. Off only for a cheap look at the base.
    hires: bool = True
    hires_denoise: float = HIRES_DENOISE_DEFAULT
    hires_steps: int = HIRES_STEPS
    hires_model: str = UPSCALE_MODEL
    hires_scale: float = HIRES_SCALE

    # Interpolation. 1 = off.
    rife: int = 1

    invert_depth: bool = DEFAULT_INVERT_DEPTH
    derive_depth: bool = False
    depth_model: str = "depth_anything_v2_vitl.pth"
    force_rate: float = 0.0
    source_frames: int | None = None
    # The control track's own pixel size, when the caller has probed it. With
    # `aspect="auto"` this is what decides the canvas.
    source_width: int | None = None
    source_height: int | None = None
    closed_loop: bool = True
    filename_prefix: str = "artrium_alcm"


def clamp(value: float | None, low: float, high: float, fallback: float) -> float:
    if value is None:
        return fallback
    return round(min(max(float(value), low), high), 2)


def clamp_depth(value: float | None) -> float:
    return clamp(value, DEPTH_MIN, DEPTH_MAX, DEPTH_DEFAULT)


def clamp_ip(value: float | None) -> float:
    return clamp(value, IP_MIN, IP_MAX, IP_DEFAULT)


def clamp_hires(value: float | None) -> float:
    return clamp(value, HIRES_DENOISE_MIN, HIRES_DENOISE_MAX, HIRES_DENOISE_DEFAULT)


def snap_size(value: int) -> int:
    """SD 1.5 latents are 8x downsampled."""
    return max(64, (int(value) // 8) * 8)


def canvas_for_ratio(ratio: float, budget: int = PIXEL_BUDGET) -> tuple[int, int]:
    """A canvas with the given width/height ratio at the standard pixel budget."""
    ratio = max(0.1, min(10.0, float(ratio)))
    height = math.sqrt(budget / ratio)
    width = height * ratio
    longest = max(width, height)
    if longest > LONG_EDGE_MAX:
        scale = LONG_EDGE_MAX / longest
        width, height = width * scale, height * scale
    return snap_size(width), snap_size(height)


def canvas_size(
    aspect: str = DEFAULT_ASPECT,
    source: tuple[int | None, int | None] | None = None,
) -> tuple[int, int]:
    """The render canvas.

    "auto" follows the control track's own shape, which is the case worth
    optimising for: a portrait Blender render should come back portrait without
    anyone selecting anything. It falls back to the original's square only when
    the source dimensions are unknown, because guessing an aspect is worse than
    admitting there is none.
    """
    if aspect == "auto":
        width, height = source or (None, None)
        if width and height:
            return canvas_for_ratio(width / height)
        return _CANVASES["square"]
    width, height = _CANVASES.get(aspect) or _CANVASES["square"]
    return snap_size(width), snap_size(height)


def output_size(
    aspect: str = DEFAULT_ASPECT,
    hires: bool = True,
    hires_scale: float = HIRES_SCALE,
    source: tuple[int | None, int | None] | None = None,
) -> tuple[int, int]:
    """What comes out, so a caller can say so before starting."""
    width, height = canvas_size(aspect, source)
    if not hires:
        return width, height
    factor = 4 * hires_scale
    return snap_size(int(width * factor)), snap_size(int(height * factor))


def _color_mask(p: str, color, threshold: int, invert: bool = False) -> dict:
    return {
        "class_type": "ColorToMask",
        "inputs": {
            "images": [p + "maskfit", 0], "invert": invert,
            "red": color[0], "green": color[1], "blue": color[2],
            "threshold": threshold, "per_batch": 16,
        },
    }


def _injection_nodes(wf: dict, p: str, req: "AnimateLcmRequest",
                     width: int, height: int, length: int) -> list:
    """The injected frames, fitted to the canvas and exactly `length` long."""
    inj = req.injection
    wf[p + "injvid"] = _load_video(inj.video, length, 0.0)
    frames = [p + "injvid", 0]
    if inj.source_frames and inj.source_frames < length:
        # The injection is applied to every frame of the batch at once, so a
        # short source is looped rather than left to run out.
        wf[p + "injrep"] = {
            "class_type": "RepeatImageBatch",
            "inputs": {"image": frames, "amount": math.ceil(length / inj.source_frames)},
        }
        wf[p + "injcut"] = {
            "class_type": "ImageFromBatch",
            "inputs": {"image": [p + "injrep", 0], "batch_index": 0, "length": length},
        }
        frames = [p + "injcut", 0]
    wf[p + "injfit"] = _fit_node(frames, width, height, req.fit, "lanczos")
    wf[p + "injstr"] = {
        "class_type": "ADE_MultivalDynamic",
        "inputs": {"float_val": clamp(inj.strength, INJECT_STRENGTH_MIN,
                                      INJECT_STRENGTH_MAX, INJECT_STRENGTH_DEFAULT)},
    }
    inputs = {
        "image": [p + "injfit", 0], "vae": [p + "ckpt", 2],
        "invert_mask": False, "resize_image": True,
        "start_percent": injection_start(inj, req.hires, clamp_hires(req.hires_denoise)),
        "guarantee_steps": 1, "strength_multival": [p + "injstr", 0],
    }
    if inj.color is not None:
        wf[p + "injmask"] = _color_mask(p, inj.color, inj.threshold)
        inputs["mask_opt"] = [p + "injmask", 0]
    wf[p + "inj"] = {"class_type": "ADE_NoisedImageInjection", "inputs": inputs}
    return [p + "inj", 0]


def _load_video(path: str, length: int, force_rate: float) -> dict:
    return {
        "class_type": "VHS_LoadVideoPath",
        "inputs": {
            "video": str(path),
            "force_rate": float(force_rate),
            "custom_width": 0, "custom_height": 0,
            "frame_load_cap": length,
            "skip_first_frames": 0, "select_every_nth": 1,
        },
    }


def _fit_node(image_ref, width: int, height: int, fit: str, interpolation: str) -> dict:
    return {
        "class_type": "ImageResize+",
        "inputs": {
            "image": image_ref,
            "width": width, "height": height,
            "interpolation": interpolation,
            "method": _FIT_METHODS.get(fit, _FIT_METHODS[DEFAULT_FIT]),
            "condition": "always",
            "multiple_of": 0,
        },
    }


def resolve_seed(seed: int) -> int:
    """Turn a -1 into a real seed.

    Called by the runner *before* building anything, because a two-stage render
    has to sample the second pass with the seed the first one used — and a seed
    invented inside the builder is invisible to the caller who has to hand it
    back later.
    """
    return seed if seed >= 0 else random.randint(0, 2**32 - 1)


def _model_stack(wf: dict, p: str, req: AnimateLcmRequest,
                 width: int, height: int, length: int) -> list:
    """Checkpoint, motion module, context options and every IP-Adapter.

    Shared by all three builders, and it has to be: the hires pass is sampled
    with this exact model, motion module included. That is the reason a
    diffusion upscale of a video does not boil here — see the module docstring
    — and rebuilding a plain checkpoint for the second stage would throw it
    away.
    """
    wf[p + "ckpt"] = {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {"ckpt_name": CHECKPOINT},
    }
    base_model = [p + "ckpt", 0]
    if req.lcm_lora_strength > 0:
        wf[p + "lcmlora"] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "lora_name": MOTION_LORA,
                "strength_model": float(req.lcm_lora_strength),
                "model": base_model,
            },
        }
        base_model = [p + "lcmlora", 0]
    wf[p + "admodel"] = {
        "class_type": "ADE_LoadAnimateDiffModel",
        "inputs": {"model_name": MOTION_MODEL},
    }
    wf[p + "mscale"] = {
        "class_type": "ADE_MultivalDynamic",
        "inputs": {"float_val": float(req.motion_scale)},
    }
    apply_inputs = {
        "motion_model": [p + "admodel", 0],
        "scale_multival": [p + "mscale", 0],
    }
    if req.motion_lora:
        wf[p + "mlora"] = {
            "class_type": "ADE_AnimateDiffLoRALoader",
            # `name`, not `lora_name` — this node does not follow the naming
            # every other loader uses.
            "inputs": {
                "name": req.motion_lora,
                "strength": float(req.motion_lora_strength),
            },
        }
        apply_inputs["motion_lora"] = [p + "mlora", 0]
    wf[p + "adapply"] = {
        "class_type": "ADE_ApplyAnimateDiffModelSimple",
        "inputs": apply_inputs,
    }
    wf[p + "ctx"] = {
        "class_type": "ADE_LoopedUniformContextOptions",
        "inputs": {
            "context_length": CONTEXT_LENGTH,
            "context_stride": CONTEXT_STRIDE,
            "context_overlap": CONTEXT_OVERLAP,
            "closed_loop": bool(req.closed_loop),
            "fuse_method": "pyramid",
            "use_on_equal_length": False,
            "start_percent": 0.0,
            "guarantee_steps": 1,
        },
    }
    # FreeNoise repeats one noise pattern across the context windows instead of
    # drawing fresh noise per window, which is what stops a clip longer than 16
    # frames from drifting at every seam.
    wf[p + "settings"] = {
        "class_type": "ADE_AnimateDiffSamplingSettings",
        "inputs": {
            "batch_offset": 0, "noise_type": NOISE_TYPE, "seed_gen": "comfy",
            "seed_offset": 0, "adapt_denoise_steps": False,
        },
    }
    if req.injection:
        wf[p + "settings"]["inputs"]["image_inject"] = _injection_nodes(
            wf, p, req, width, height, length)
    wf[p + "evolved"] = {
        "class_type": "ADE_UseEvolvedSampling",
        "inputs": {
            "model": base_model,
            "m_models": [p + "adapply", 0],
            "context_options": [p + "ctx", 0],
            "sample_settings": [p + "settings", 0],
            "beta_schedule": req.beta_schedule,
        },
    }

    # ── Material: IP-Adapter, one per region, all in one pass ────────────────
    if req.mask_video:
        wf[p + "maskvid"] = _load_video(req.mask_video, length, req.force_rate)
        wf[p + "maskfit"] = _fit_node(
            [p + "maskvid", 0], width, height, req.fit, "nearest-exact",
        )

    adapters: list[tuple[str | None, str, float]] = []
    if req.regions:
        # First and unmasked, so the regions apply on top of a frame that is
        # already sitting on a reference rather than on the checkpoint's own
        # idea of the colour. Without this, everything outside the masks is
        # unconditioned — the failure this whole layer exists to prevent.
        if req.base_reference:
            adapters.append((None, req.base_reference, clamp_ip(req.base_weight)))
        for i, region in enumerate(req.regions):
            node = f"{p}mask{i}"
            wf[node] = _color_mask(p, region.color, region.threshold, region.invert)
            # A region made only of embeddings still needs its mask — the text
            # layer keys on it — but has no picture to adapt.
            if region.reference:
                adapters.append((node, region.reference, clamp_ip(region.weight)))
    elif req.reference_image:
        adapters.append((None, req.reference_image, clamp_ip(req.ip_weight)))

    # Embeddings alone are a complete material channel. Without a picture the
    # adapter stack is left out entirely rather than loaded and fed nothing.
    if not adapters:
        return [p + "evolved", 0]

    wf[p + "ipaload"] = {
        "class_type": "IPAdapterUnifiedLoader",
        "inputs": {"model": [p + "evolved", 0], "preset": IPADAPTER_PRESET},
    }
    model_ref = [p + "ipaload", 0]
    for i, (mask_node, picture, weight) in enumerate(adapters):
        ref_node = f"{p}ref{i}"
        wf[ref_node] = {"class_type": "LoadImage", "inputs": {"image": picture}}
        node = f"{p}ipa{i}"
        inputs = {
            "model": model_ref, "ipadapter": [p + "ipaload", 1],
            "image": [ref_node, 0], "weight": weight,
            "weight_type": "linear", "combine_embeds": "concat",
            "start_at": 0.0, "end_at": clamp(req.ip_end, 0.0, 1.0, IP_END_DEFAULT),
            "embeds_scaling": req.ip_scaling,
        }
        if mask_node:
            inputs["attn_mask"] = [mask_node, 0]
        wf[node] = {"class_type": "IPAdapterAdvanced", "inputs": inputs}
        model_ref = [node, 0]
    return model_ref


def _embedding_token(embedding: Embedding) -> str:
    weight = clamp(embedding.weight, EMBEDDING_WEIGHT_MIN, EMBEDDING_WEIGHT_MAX, 1.0)
    if weight == 1.0:
        return f"embedding:{embedding.name}"
    return f"(embedding:{embedding.name}:{weight:g})"


def _chunks(prompt: str, embeddings: list[Embedding], join: str) -> list[str]:
    prompt = (prompt or "").strip()
    tokens = [_embedding_token(e) for e in embeddings]
    if not tokens or join == "inline":
        return [", ".join(t for t in (prompt, *tokens) if t)]
    chunks = ([prompt] if prompt else []) + tokens
    if join == "sandwich":
        chunks = [part for chunk in chunks for part in ("", chunk)][1:]
    return chunks


def moving_embeddings(req: AnimateLcmRequest) -> list[Embedding]:
    """The embeddings that carry a curve — one extra conditioning layer each."""
    return [e for e in req.embeddings if e.curve]


def positive_chunks(req: AnimateLcmRequest) -> list[str]:
    """The texts the (base) positive is encoded from, in order; "" is a
    sandwich blank. Only the held embeddings — a moving one gets a layer of its
    own, see `_text_nodes`.

    One chunk means one plain CLIPTextEncode — which is what a request without
    embeddings has always produced, node for node.
    """
    held = [e for e in req.embeddings if not e.curve]
    return _chunks(req.prompt, held, req.embedding_join)


def _encode(text: str, p: str) -> dict:
    return {"class_type": "CLIPTextEncode", "inputs": {"text": text, "clip": [p + "ckpt", 1]}}


def _conditioning(wf: dict, p: str, chunks: list[str], out: str) -> None:
    """Encode `chunks` and join them into one conditioning at node `out`."""
    if len(chunks) == 1:
        wf[out] = _encode(chunks[0], p)
        return
    refs = []
    for i, text in enumerate(chunks):
        if text:
            wf[f"{out}_c{i}"] = _encode(text, p)
            refs.append([f"{out}_c{i}", 0])
        else:
            # Every blank in a sandwich is the same conditioning.
            wf.setdefault(p + "pos_blank", _encode("", p))
            refs.append([p + "pos_blank", 0])
    joined = refs[0]
    for i, ref in enumerate(refs[1:], start=1):
        node = out if i == len(refs) - 1 else f"{out}_j{i}"
        wf[node] = {
            "class_type": "ConditioningConcat",
            # `to` comes first in the token sequence, `from` is appended.
            "inputs": {"conditioning_to": joined, "conditioning_from": ref},
        }
        joined = [node, 0]


def _set_mask(wf: dict, node: str, cond: str, mask: list) -> list:
    wf[node] = {
        "class_type": "ConditioningSetMask",
        "inputs": {"conditioning": [cond, 0], "mask": mask,
                   "strength": 1.0, "set_cond_area": "default"},
    }
    return [node, 0]


def _curve_mask(wf: dict, node: str, values: list[float], width: int, height: int) -> list:
    if len(values) < 2:
        # The node refuses a one-frame batch; ComfyUI trims the mask to the
        # latent's batch anyway, so the second frame is never read.
        values = values * 2
    wf[node] = {
        "class_type": "CreateFadeMaskAdvanced",
        "inputs": {
            "points_string": fade_points(values), "invert": False,
            "frames": len(values), "width": width, "height": height,
            "interpolation": "linear",
        },
    }
    return [node, 0]


def _outside_regions(wf: dict, p: str, indices: list[int]) -> list:
    """1 - the union of the given regions' masks: where the global layers
    still speak once regions carry embeddings of their own."""
    union = [f"{p}mask{indices[0]}", 0]
    for n, i in enumerate(indices[1:], start=1):
        wf[f"{p}pos_union{n}"] = {
            "class_type": "MaskComposite",
            "inputs": {"destination": union, "source": [f"{p}mask{i}", 0],
                       "x": 0, "y": 0, "operation": "add"},
        }
        union = [f"{p}pos_union{n}", 0]
    wf[p + "pos_outside"] = {"class_type": "InvertMask", "inputs": {"mask": union}}
    return [p + "pos_outside", 0]


def _text_nodes(wf: dict, p: str, req: AnimateLcmRequest, length: int,
                width: int = 0, height: int = 0) -> None:
    """Positive and negative. The positive always ends in `pos`, whatever it
    is built from, because every sampler and ControlNet downstream reads it.

    Layers, each a conditioning with a mask, combined:
      - the base: prompt + held embeddings;
      - one per embedding with a curve (prompt + held + itself), owning the
        frames its curve gives it, the base taking the rest;
      - one per region with embeddings (prompt + the region's own), owning its
        mask — the global layers are masked out of it, so inside a region only
        that region's embeddings speak.
    With none of the last two it is a single conditioning, exactly as before.
    """
    moving = moving_embeddings(req)
    region_idx = [i for i, r in enumerate(req.regions) if r.embeddings]
    if not moving and not region_idx:
        _conditioning(wf, p, positive_chunks(req), p + "pos")
    else:
        held = [e for e in req.embeddings if not e.curve]
        curves = [curve_values(e.curve, length, e.cycles) for e in moving]
        # Curve masks are multiplied with the region masks, which come off the
        # fitted mask video at canvas size; the two have to be the same size.
        mw, mh = (width, height) if region_idx else (_CURVE_MASK_SIZE, _CURVE_MASK_SIZE)
        outside = _outside_regions(wf, p, region_idx) if region_idx else None

        def global_mask(node: str, values: list[float] | None) -> list:
            mask = _curve_mask(wf, node + "_mask", values, mw, mh) if values else None
            if outside is None:
                return mask
            if mask is None:
                return outside
            wf[node + "_out"] = {
                "class_type": "MaskComposite",
                "inputs": {"destination": mask, "source": outside,
                           "x": 0, "y": 0, "operation": "multiply"},
            }
            return [node + "_out", 0]

        _conditioning(wf, p, positive_chunks(req), p + "pos_base")
        layers = [_set_mask(wf, p + "pos_base_m", p + "pos_base",
                            global_mask(p + "pos_base_m",
                                        base_curve(curves, length) if moving else None))]
        for i, (emb, values) in enumerate(zip(moving, curves)):
            _conditioning(wf, p, _chunks(req.prompt, held + [emb], req.embedding_join),
                          f"{p}pos_l{i}")
            layers.append(_set_mask(wf, f"{p}pos_l{i}_m", f"{p}pos_l{i}",
                                    global_mask(f"{p}pos_l{i}_m", values)))
        for i in region_idx:
            region_embs = [Embedding(e.name, e.weight) for e in req.regions[i].embeddings]
            _conditioning(wf, p, _chunks(req.prompt, region_embs, req.embedding_join),
                          f"{p}pos_r{i}")
            layers.append(_set_mask(wf, f"{p}pos_r{i}_m", f"{p}pos_r{i}",
                                    [f"{p}mask{i}", 0]))
        joined = layers[0]
        for i, layer in enumerate(layers[1:], start=1):
            node = p + "pos" if i == len(layers) - 1 else f"{p}pos_k{i}"
            wf[node] = {
                "class_type": "ConditioningCombine",
                "inputs": {"conditioning_1": joined, "conditioning_2": layer},
            }
            joined = [node, 0]
    wf[p + "neg"] = {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": req.negative, "clip": [p + "ckpt", 1]},
    }


def _hires_nodes(wf: dict, p: str, req: AnimateLcmRequest,
                 images_ref: list, model_ref: list, seed: int) -> list:
    """The detail engine, given base frames from anywhere.

    An upscale model for edges, scaled back to a net 2x, then re-sampled at the
    hires denoise — so most of the base picture survives and the rest is
    redrawn at the higher resolution. The two ControlNets are taken from the
    *base* frames, which is what keeps the redraw honest.

    `images_ref` is the only thing that differs between the one-shot graph
    (where it is the base sampler's own decode, still in VRAM) and the
    finishing graph (where it is a video file loaded back off disk). Nothing
    downstream can tell the difference, which is what makes the split safe.
    """
    wf[p + "lineart"] = {
        "class_type": "LineArtPreprocessor",
        "inputs": {"image": images_ref, "coarse": "disable",
                   "resolution": PREPROCESSOR_RESOLUTION},
    }
    wf[p + "midas"] = {
        "class_type": "MiDaS-DepthMapPreprocessor",
        "inputs": {"image": images_ref, "a": 6.283185307179586,
                   "bg_threshold": 0.1, "resolution": PREPROCESSOR_RESOLUTION},
    }
    wf[p + "cn2load"] = {
        "class_type": "ControlNetLoader",
        "inputs": {"control_net_name": LINEART_CONTROLNET},
    }
    wf[p + "cn2"] = {
        "class_type": "ControlNetApplyAdvanced",
        "inputs": {
            "positive": [p + "pos", 0], "negative": [p + "neg", 0],
            "control_net": [p + "cn2load", 0], "image": [p + "lineart", 0],
            "strength": HIRES_LINEART_STRENGTH,
            "start_percent": 0.0, "end_percent": HIRES_LINEART_END,
        },
    }
    wf[p + "cn3load"] = {
        "class_type": "ControlNetLoader",
        "inputs": {"control_net_name": DEPTH_CONTROLNET},
    }
    wf[p + "cn3"] = {
        "class_type": "ControlNetApplyAdvanced",
        "inputs": {
            "positive": [p + "cn2", 0], "negative": [p + "cn2", 1],
            "control_net": [p + "cn3load", 0], "image": [p + "midas", 0],
            "strength": HIRES_DEPTH_STRENGTH,
            "start_percent": 0.0, "end_percent": HIRES_DEPTH_END,
        },
    }
    wf[p + "upmodel"] = {
        "class_type": "UpscaleModelLoader",
        "inputs": {"model_name": req.hires_model},
    }
    wf[p + "up"] = {
        "class_type": "ImageUpscaleWithModel",
        "inputs": {"upscale_model": [p + "upmodel", 0], "image": images_ref},
    }
    wf[p + "down"] = {
        "class_type": "ImageScaleBy",
        "inputs": {"image": [p + "up", 0], "upscale_method": "bicubic",
                   "scale_by": float(req.hires_scale)},
    }
    wf[p + "enc"] = {
        "class_type": "VAEEncode",
        "inputs": {"pixels": [p + "down", 0], "vae": [p + "ckpt", 2]},
    }
    wf[p + "ks2"] = {
        "class_type": "KSampler",
        "inputs": {
            "model": model_ref,
            "positive": [p + "cn3", 0], "negative": [p + "cn3", 1],
            "latent_image": [p + "enc", 0],
            "seed": seed, "steps": req.hires_steps, "cfg": req.cfg,
            "sampler_name": req.sampler, "scheduler": req.scheduler,
            "denoise": clamp_hires(req.hires_denoise),
        },
    }
    wf[p + "dec2"] = {
        "class_type": "VAEDecode",
        "inputs": {"samples": [p + "ks2", 0], "vae": [p + "ckpt", 2]},
    }
    return [p + "dec2", 0]


def _rife_nodes(wf: dict, node_id: str, images_ref: list, factor: int) -> list:
    wf[node_id] = {
        "class_type": "RIFE VFI",
        "inputs": {
            "frames": images_ref, "ckpt_name": RIFE_MODEL,
            "clear_cache_after_n_frames": 100, "multiplier": int(factor),
            "fast_mode": True, "ensemble": True, "scale_factor": 1.0,
            # All three carry defaults in the UI but are declared required,
            # so the graph has to name them.
            "dtype": "float32", "torch_compile": False, "batch_size": 1,
        },
    }
    return [node_id, 0]


def _combine_node(wf: dict, node_id: str, images_ref: list,
                  fps: int, prefix: str) -> str:
    wf[node_id] = {
        "class_type": "VHS_VideoCombine",
        "inputs": {
            "images": images_ref,
            "frame_rate": fps, "loop_count": 0,
            "filename_prefix": prefix,
            "format": "video/h264-mp4", "pix_fmt": "yuv420p", "crf": 15,
            "save_metadata": False, "trim_to_audio": False,
            "pingpong": False, "save_output": True,
        },
    }
    return node_id


def _plan_dimensions(req: AnimateLcmRequest) -> tuple[int, int, int]:
    canvas_w, canvas_h = canvas_size(req.aspect, (req.source_width, req.source_height))
    width = snap_size(req.width or canvas_w)
    height = snap_size(req.height or canvas_h)
    length = max(1, min(req.length, req.source_frames or req.length))
    return width, height, length


def _validate(req: AnimateLcmRequest) -> None:
    if req.regions and not req.mask_video:
        raise ValueError("regions need a mask_video to key them out of")
    # The tool insists on a picture itself (routers/vace.py); the builder only
    # refuses a render with nothing at all to condition on, so that a
    # prompt-only baseline stays possible for a sweep.
    if (not req.regions and not req.reference_image and not req.embeddings
            and not (req.prompt or "").strip()):
        raise ValueError("a prompt, an embedding or a reference image is required")
    for region in req.regions:
        if not region.reference and not region.embeddings:
            raise ValueError("a region needs a reference image or an embedding")
        for embedding in region.embeddings:
            if not is_embedding_name(embedding.name):
                raise ValueError(f"not a usable embedding name: {embedding.name!r}")
    if req.injection:
        if not req.injection.video:
            raise ValueError("an injection needs a video")
        if req.injection.color is not None and not req.mask_video:
            raise ValueError("a region-limited injection needs a mask_video")
    if req.embedding_join not in EMBEDDING_JOINS:
        raise ValueError(f"embedding_join must be one of {EMBEDDING_JOINS}")
    for embedding in req.embeddings:
        if not is_embedding_name(embedding.name):
            raise ValueError(f"not a usable embedding name: {embedding.name!r}")
        if embedding.curve is not None and embedding.curve not in CURVES:
            raise ValueError(f"curve must be one of {CURVES}")


def _geometry_nodes(wf: dict, p: str, req: AnimateLcmRequest,
                    width: int, height: int, length: int) -> None:
    """Control track in, depth ControlNet out."""
    wf[p + "ctrl"] = _load_video(req.control_video, length, req.force_rate)
    ctrl_ref = [p + "ctrl", 0]
    if req.derive_depth:
        wf[p + "depth"] = {
            "class_type": "DepthAnythingV2Preprocessor",
            "inputs": {
                "image": ctrl_ref, "ckpt_name": req.depth_model,
                "resolution": max(width, height),
            },
        }
        ctrl_ref = [p + "depth", 0]
    elif req.invert_depth:
        wf[p + "inv"] = {"class_type": "ImageInvert", "inputs": {"image": ctrl_ref}}
        ctrl_ref = [p + "inv", 0]
    wf[p + "fit"] = _fit_node(ctrl_ref, width, height, req.fit, "lanczos")

    wf[p + "cnload"] = {
        "class_type": "ControlNetLoader",
        "inputs": {"control_net_name": DEPTH_CONTROLNET},
    }
    wf[p + "cn"] = {
        "class_type": "ControlNetApplyAdvanced",
        "inputs": {
            "positive": [p + "pos", 0], "negative": [p + "neg", 0],
            "control_net": [p + "cnload", 0], "image": [p + "fit", 0],
            "strength": clamp_depth(req.depth_strength),
            "start_percent": 0.0,
            "end_percent": clamp(req.depth_end, 0.0, 1.0, DEPTH_END_DEFAULT),
        },
    }


def build_animatelcm_workflow(req: AnimateLcmRequest) -> tuple[dict, str]:
    """Build the whole graph in one submission. Returns (workflow, output node)."""
    _validate(req)
    width, height, length = _plan_dimensions(req)
    seed = resolve_seed(req.seed)

    p = "al_"
    wf: dict = {}
    model_ref = _model_stack(wf, p, req, width, height, length)
    _text_nodes(wf, p, req, length, width, height)
    _geometry_nodes(wf, p, req, width, height, length)

    # ── Base pass ────────────────────────────────────────────────────────────
    wf[p + "latent"] = {
        "class_type": "EmptyLatentImage",
        "inputs": {"width": width, "height": height, "batch_size": length},
    }
    wf[p + "ks"] = {
        "class_type": "KSampler",
        "inputs": {
            "model": model_ref,
            "positive": [p + "cn", 0], "negative": [p + "cn", 1],
            "latent_image": [p + "latent", 0],
            "seed": seed, "steps": req.steps, "cfg": req.cfg,
            "sampler_name": req.sampler, "scheduler": req.scheduler, "denoise": 1.0,
        },
    }
    wf[p + "dec"] = {
        "class_type": "VAEDecode",
        "inputs": {"samples": [p + "ks", 0], "vae": [p + "ckpt", 2]},
    }
    images_ref = [p + "dec", 0]

    if req.hires:
        images_ref = _hires_nodes(wf, p, req, images_ref, model_ref, seed)

    # ── Interpolation ────────────────────────────────────────────────────────
    # After the hires pass, never before: RIFE would otherwise multiply the
    # number of frames the expensive sampler has to redraw.
    fps = int(req.fps)
    if req.rife and req.rife > 1:
        images_ref = _rife_nodes(wf, p + "rife", images_ref, req.rife)
        fps = fps * int(req.rife)

    return wf, _combine_node(wf, p + "out", images_ref, fps, req.filename_prefix)


def build_animatelcm_base_workflow(req: AnimateLcmRequest) -> tuple[dict, str, str]:
    """Stage one of a two-stage render: the base pass only.

    Returns (workflow, preview node, base node). The base pass is roughly a
    third of the cost of the whole graph, which is what makes stopping here
    worth doing — the composition, the motion and whether the reference's
    material came through at all are all decided by it, and none of them are
    worth waiting out the hires pass to find out.

    **Two outputs, and they are not the same file.** The base node writes the
    sampler's own frames at their own rate, and that file is the input the
    finishing stage reads; interpolating it first would hand the expensive
    second pass four times as many frames to redraw. The preview node writes
    what the user actually watches — interpolated, because at 8 fps a clip
    stutters in a way the finished piece never will, and a preview that
    misrepresents the result is worse than no preview.

    With `rife == 1` there is nothing to interpolate and both names refer to
    the same node.
    """
    _validate(req)
    width, height, length = _plan_dimensions(req)
    seed = resolve_seed(req.seed)

    p = "al_"
    wf: dict = {}
    model_ref = _model_stack(wf, p, req, width, height, length)
    _text_nodes(wf, p, req, length, width, height)
    _geometry_nodes(wf, p, req, width, height, length)

    wf[p + "latent"] = {
        "class_type": "EmptyLatentImage",
        "inputs": {"width": width, "height": height, "batch_size": length},
    }
    wf[p + "ks"] = {
        "class_type": "KSampler",
        "inputs": {
            "model": model_ref,
            "positive": [p + "cn", 0], "negative": [p + "cn", 1],
            "latent_image": [p + "latent", 0],
            "seed": seed, "steps": req.steps, "cfg": req.cfg,
            "sampler_name": req.sampler, "scheduler": req.scheduler, "denoise": 1.0,
        },
    }
    wf[p + "dec"] = {
        "class_type": "VAEDecode",
        "inputs": {"samples": [p + "ks", 0], "vae": [p + "ckpt", 2]},
    }
    base_images = [p + "dec", 0]
    fps = int(req.fps)

    base_node = _combine_node(
        wf, p + "outbase", base_images, fps, req.filename_prefix + "_base",
    )
    if not req.rife or req.rife <= 1:
        return wf, base_node, base_node

    preview_images = _rife_nodes(wf, p + "rifeprev", base_images, req.rife)
    preview_node = _combine_node(
        wf, p + "outprev", preview_images, fps * int(req.rife),
        req.filename_prefix + "_preview",
    )
    return wf, preview_node, base_node


def build_animatelcm_hires_workflow(
    req: AnimateLcmRequest, base_video: str,
) -> tuple[dict, str]:
    """Stage two: refine an already-rendered base pass into the final clip.

    `base_video` is the file stage one wrote. Everything else is rebuilt from
    the same request, and it has to be *the same* request: the hires sampler
    runs on the IP-Adapter-loaded, AnimateDiff-wrapped model, so a graph that
    skipped the reference pictures here would refine the base into something
    made of different material.

    Reading the base back as h264 rather than keeping it in VRAM costs a little
    quantisation on the input to a pass that redraws most of it anyway, and
    buys the thing this exists for: the user gets to look at it first.
    """
    _validate(req)
    width, height, length = _plan_dimensions(req)
    seed = resolve_seed(req.seed)

    p = "al_"
    wf: dict = {}
    model_ref = _model_stack(wf, p, req, width, height, length)
    _text_nodes(wf, p, req, length, width, height)

    # No geometry chain: the base render already carries the control track's
    # shape, and the hires pass is steered by the ControlNets derived from it.
    wf[p + "basevid"] = _load_video(base_video, length, 0.0)
    images_ref = _hires_nodes(wf, p, req, [p + "basevid", 0], model_ref, seed)

    fps = int(req.fps)
    if req.rife and req.rife > 1:
        images_ref = _rife_nodes(wf, p + "rife", images_ref, req.rife)
        fps = fps * int(req.rife)

    return wf, _combine_node(wf, p + "out", images_ref, fps, req.filename_prefix)


_DIALS = ("depth", "ip", "hires")


def sweep_workflows(
    req: AnimateLcmRequest, dial: str = "hires", values=None,
) -> list[tuple[float, dict, str]]:
    """The same request across one dial at a fixed seed.

    One dial at a time: two moving together produce a sheet nobody can read
    backwards into a cause.
    """
    if dial not in _DIALS:
        raise ValueError(f"dial must be one of {_DIALS}")
    values = values or {
        "depth": DEPTH_SWEEP, "ip": IP_SWEEP, "hires": HIRES_SWEEP,
    }[dial]
    seed = req.seed if req.seed >= 0 else random.randint(0, 2**32 - 1)

    out = []
    for value in values:
        variant = copy.deepcopy(req)
        variant.seed = seed
        if dial == "depth":
            variant.depth_strength = value
        elif dial == "ip":
            variant.ip_weight = value
            for region in variant.regions:
                region.weight = value
        else:
            variant.hires_denoise = value
        variant.filename_prefix = f"{req.filename_prefix}_{dial}{int(value * 100):03d}"
        wf, node = build_animatelcm_workflow(variant)
        out.append((value, wf, node))
    return out
