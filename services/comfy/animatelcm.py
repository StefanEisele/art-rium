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
  depth ControlNet 0.45, end_percent 0.7
  hires            denoise 0.4, 11 steps; lineart CN 0.3/end 0.5, depth CN
                   0.5/end 0.6
"""
from __future__ import annotations

import copy
import math
import random
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


@dataclass
class Region:
    """One colour region of the mask video and the picture it is made of."""
    color: tuple[int, int, int]
    reference: str
    weight: float = IP_DEFAULT
    threshold: int = 20
    invert: bool = False


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
    ip_scaling: str = "K+V w/ C penalty"
    steps: int = DEFAULT_STEPS
    cfg: float = DEFAULT_CFG
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
    wf[p + "lcmlora"] = {
        "class_type": "LoraLoaderModelOnly",
        "inputs": {
            "lora_name": MOTION_LORA,
            "strength_model": MOTION_LORA_STRENGTH,
            "model": [p + "ckpt", 0],
        },
    }
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
    wf[p + "evolved"] = {
        "class_type": "ADE_UseEvolvedSampling",
        "inputs": {
            "model": [p + "lcmlora", 0],
            "m_models": [p + "adapply", 0],
            "context_options": [p + "ctx", 0],
            "sample_settings": [p + "settings", 0],
            "beta_schedule": req.beta_schedule,
        },
    }

    # ── Material: IP-Adapter, one per region, all in one pass ────────────────
    wf[p + "ipaload"] = {
        "class_type": "IPAdapterUnifiedLoader",
        "inputs": {"model": [p + "evolved", 0], "preset": IPADAPTER_PRESET},
    }
    model_ref = [p + "ipaload", 0]

    if req.mask_video:
        wf[p + "maskvid"] = _load_video(req.mask_video, length, req.force_rate)
        wf[p + "maskfit"] = _fit_node(
            [p + "maskvid", 0], width, height, req.fit, "nearest-exact",
        )

    adapters: list[tuple[str | None, str, float]] = []
    if req.regions:
        for i, region in enumerate(req.regions):
            node = f"{p}mask{i}"
            wf[node] = {
                "class_type": "ColorToMask",
                "inputs": {
                    "images": [p + "maskfit", 0], "invert": region.invert,
                    "red": region.color[0], "green": region.color[1],
                    "blue": region.color[2], "threshold": region.threshold,
                    "per_batch": 16,
                },
            }
            adapters.append((node, region.reference, clamp_ip(region.weight)))
    else:
        adapters.append((None, req.reference_image, clamp_ip(req.ip_weight)))

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


def _text_nodes(wf: dict, p: str, req: AnimateLcmRequest) -> None:
    wf[p + "pos"] = {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": req.prompt, "clip": [p + "ckpt", 1]},
    }
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
            "sampler_name": SAMPLER, "scheduler": SCHEDULER,
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
    if not req.regions and not req.reference_image:
        raise ValueError("a reference image is required")
    for region in req.regions:
        if not region.reference:
            raise ValueError("a region needs a reference image")


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
    _text_nodes(wf, p, req)
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
            "sampler_name": SAMPLER, "scheduler": SCHEDULER, "denoise": 1.0,
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
    _text_nodes(wf, p, req)
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
            "sampler_name": SAMPLER, "scheduler": SCHEDULER, "denoise": 1.0,
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
    _text_nodes(wf, p, req)

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
