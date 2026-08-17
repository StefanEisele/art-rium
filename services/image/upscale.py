"""
Diffusion upscale for stills — Ultimate SD Upscale driven by Z-Image Turbo.

An ESRGAN pass alone enlarges what is already there; it cannot put detail back
that was never rendered. Ultimate SD Upscale runs the ESRGAN enlargement first
and then re-diffuses the result **in tiles**, so the generator paints real
detail at the new size while each tile stays inside the resolution the model
was trained at. Z-Image Turbo does the painting because it is the model that
made these pictures — the same texture prior, so an upscale looks like more of
the same image rather than a different model's idea of it.

Settings, and why they are what they are
────────────────────────────────────────
`denoise` is the whole control surface, and the published guidance is
consistent: 0.15–0.20 changes almost nothing, ~0.35 is the enhancement sweet
spot, and past ~0.5 each tile starts reimagining its own contents, which is how
you get the classic "collage of unrelated tiles". The node's own default is
0.2. So the slider runs 0.10–0.45 and defaults to 0.25 — enough to paint
detail, short of the range where tiles stop agreeing with each other.

`steps` does NOT scale with denoise here. ComfyUI (comfy/samplers.py,
`KSampler.set_steps`) computes `int(steps/denoise)` sigmas and keeps the last
`steps + 1` of them, so a tile always samples the full `steps` — the denoise
only decides how far down the schedule it starts. That is the opposite of
A1111, where effective steps are `steps × denoise` and low denoise starves the
render. Nothing needs compensating: 9 steps is 9 steps per tile, which is what
Z-Image Turbo is distilled for. UltimateSDUpscale reaches the sampler through
`nodes.common_ksampler`, so those semantics apply unchanged.

The rest mirrors the generation workflow exactly (cfg 1, res_multistep,
simple, AuraFlow shift 3) — a distilled model steered by anything else is a
different model. Tiles are 1024², the canvas Z-Image renders natively.
"""
from __future__ import annotations

import random
from pathlib import Path

# ── The two upscale models, per the user's choice ────────────────────────────
# RealESRGAN_x4plus is the general-purpose default; 4xNomos8kSC is a photo
# upscaler (RRDBNet, trained to cope with JPEG compression and blur) and reads
# sharper, which suits a photographic frame and can over-crisp a painterly one.
UPSCALE_MODELS: dict[str, str] = {
    "realesrgan": "RealESRGAN_x4plus.pth",
    "nomos8ksc":  "4xNomos8kSC.pth",
}
DEFAULT_UPSCALE_MODEL = "realesrgan"

# ── Z-Image Turbo, identical to workflows/z-image_turbo.json ─────────────────
_UNET        = "z_image_turbo_bf16.safetensors"
_UNET_DTYPE  = "fp8_e4m3fn"
_CLIP        = "qwen_3_4b.safetensors"
_CLIP_TYPE   = "lumina2"
_VAE         = "ae.safetensors"
_SHIFT       = 3
_STEPS       = 9      # real sampled steps per tile — see the module docstring
_CFG         = 1.0    # distilled, guidance-free; the negative is a zeroed copy
_SAMPLER     = "res_multistep"
_SCHEDULER   = "simple"

# ── Tiling ───────────────────────────────────────────────────────────────────
# Tile at the model's native canvas: fewer seams than 512² and no tile is
# outside what the model knows how to compose. Padding gives each tile context
# past its own edge, mask blur hides the joins.
TILE_SIZE     = 1024
TILE_PADDING  = 32
MASK_BLUR     = 8
MODE_TYPE     = "Linear"   # switch to "Chess" if seams ever show
# Left off deliberately: the node's seam_fix_denoise defaults to 1.0, which
# would fully reimagine the seam bands — far past the range the tiles
# themselves are held to. Padding + mask blur handle the joins at these
# denoise levels.
SEAM_FIX_MODE = "None"

# ── User-facing ranges ───────────────────────────────────────────────────────
DENOISE_MIN     = 0.10   # "more consistency" — essentially a clean enlargement
DENOISE_MAX     = 0.45   # "more creativity" — short of per-tile hallucination
DENOISE_DEFAULT = 0.25

SCALE_CHOICES  = (1.5, 2.0, 3.0, 4.0)
SCALE_DEFAULT  = 2.0

# A 1080x1920 source at 4x is 33 MP and ~48 tiles; the cap keeps a stray click
# from queueing a half-hour render and a 60 MB PNG.
MAX_OUTPUT_PIXELS = 40_000_000

# Wall-clock model, calibrated against a measured run on the 16 GB 4060 Ti:
# 1024² at 1.5x (1536², 4 tiles) took 101 s end to end. Every tile is one small
# generation, so the marginal cost scales with output *area* rather than with
# the scale factor — but a large part of a short run is the cold model load,
# which happens once. Modelling both keeps a 4-tile estimate from reading half
# the real wait.
_STARTUP_SECONDS  = 35.0
_SECONDS_PER_TILE = 17.0


def clamp_denoise(value: float | None) -> float:
    """Pull a denoise request into the range the tiles stay coherent in."""
    if value is None:
        return DENOISE_DEFAULT
    return round(min(max(float(value), DENOISE_MIN), DENOISE_MAX), 3)


def clamp_scale(value: float | None, src_w: int, src_h: int) -> float:
    """Clamp a scale factor to the offered choices and the output-size budget.

    Reducing rather than refusing: asking for 4x on a large source is a
    reasonable thing to want, and the honest answer is the biggest scale that
    still fits, not an error.
    """
    scale = float(value) if value else SCALE_DEFAULT
    scale = min(max(scale, min(SCALE_CHOICES)), max(SCALE_CHOICES))
    if src_w > 0 and src_h > 0:
        budget = (MAX_OUTPUT_PIXELS / (src_w * src_h)) ** 0.5
        scale = min(scale, budget)
    return round(scale, 2)


def output_size(src_w: int, src_h: int, scale: float) -> tuple[int, int]:
    return max(1, round(src_w * scale)), max(1, round(src_h * scale))


def tile_count(out_w: int, out_h: int) -> int:
    """How many tiles the redraw will sample — the unit of both time and VRAM."""
    cols = max(1, -(-out_w // TILE_SIZE))    # ceil division
    rows = max(1, -(-out_h // TILE_SIZE))
    return cols * rows


def estimate_seconds(src_w: int, src_h: int, scale: float) -> int:
    out_w, out_h = output_size(src_w, src_h, scale)
    return int(_STARTUP_SECONDS + tile_count(out_w, out_h) * _SECONDS_PER_TILE)


def build_image_upscale_workflow(
    image_name: str,
    *,
    prompt: str,
    denoise: float,
    scale: float,
    upscale_model: str = DEFAULT_UPSCALE_MODEL,
    seed: int | None = None,
    filename_prefix: str = "artrium_imgup",
) -> tuple[dict, str]:
    """Build the ComfyUI workflow. Returns (workflow, save_node_id).

    `image_name` is the file as ComfyUI's /upload/image named it. `prompt`
    should be the image's own generation prompt when there is one — the tiles
    are conditioned on it, so it steers the invented detail towards the
    picture's own subject instead of a generic texture.
    """
    model_file = UPSCALE_MODELS.get(upscale_model, UPSCALE_MODELS[DEFAULT_UPSCALE_MODEL])
    p = "iu_"
    wf: dict = {
        p+"load":  {"class_type": "LoadImage", "inputs": {"image": image_name}},
        p+"unet":  {"class_type": "UNETLoader", "inputs": {
            "unet_name": _UNET, "weight_dtype": _UNET_DTYPE,
        }},
        p+"shift": {"class_type": "ModelSamplingAuraFlow", "inputs": {
            "shift": _SHIFT, "model": [p+"unet", 0],
        }},
        p+"clip":  {"class_type": "CLIPLoader", "inputs": {
            "clip_name": _CLIP, "type": _CLIP_TYPE, "device": "default",
        }},
        p+"pos":   {"class_type": "CLIPTextEncode", "inputs": {
            "text": prompt or "", "clip": [p+"clip", 0],
        }},
        # cfg is 1, so the negative is never actually contrasted against — a
        # zeroed copy of the positive is what the generation workflow uses and
        # what the model expects.
        p+"neg":   {"class_type": "ConditioningZeroOut", "inputs": {
            "conditioning": [p+"pos", 0],
        }},
        p+"vae":   {"class_type": "VAELoader", "inputs": {"vae_name": _VAE}},
        p+"upmod": {"class_type": "UpscaleModelLoader", "inputs": {"model_name": model_file}},
        p+"usdu":  {"class_type": "UltimateSDUpscale", "inputs": {
            "image":        [p+"load", 0],
            "model":        [p+"shift", 0],
            "positive":     [p+"pos", 0],
            "negative":     [p+"neg", 0],
            "vae":          [p+"vae", 0],
            "upscale_model": [p+"upmod", 0],
            "upscale_by":   round(float(scale), 2),
            "seed":         seed if seed is not None and seed >= 0 else random.randint(0, 2**32 - 1),
            "steps":        _STEPS,
            "cfg":          _CFG,
            "sampler_name": _SAMPLER,
            "scheduler":    _SCHEDULER,
            "denoise":      clamp_denoise(denoise),
            "mode_type":    MODE_TYPE,
            "tile_width":   TILE_SIZE,
            "tile_height":  TILE_SIZE,
            "mask_blur":    MASK_BLUR,
            "tile_padding": TILE_PADDING,
            "seam_fix_mode":      SEAM_FIX_MODE,
            "seam_fix_denoise":   1.0,   # inert while seam_fix_mode is "None"
            "seam_fix_width":     64,
            "seam_fix_mask_blur": 8,
            "seam_fix_padding":   16,
            "force_uniform_tiles": True,
            "tiled_decode":       False,
            "batch_size":         1,
        }},
        p+"save":  {"class_type": "SaveImage", "inputs": {
            "filename_prefix": filename_prefix, "images": [p+"usdu", 0],
        }},
    }
    return wf, p + "save"


def upscaled_rel_path(image_filepath: str) -> tuple[str, str]:
    """(storage-relative path, bare filename) for the upscaled rendition.

    Named off the original like the enhance and grain siblings, so an image has
    exactly one upscale no matter which rendition it was rendered from, and
    re-running replaces it instead of littering.
    """
    original = Path(image_filepath)
    name = f"{original.stem}_upscaled.png"
    return str(original.with_name(name)).replace("\\", "/"), name
