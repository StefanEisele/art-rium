"""
Diffusion upscale for stills — an enlargement stage, then a tiled Z-Image redraw.

Two stages, and each owns a different artefact
───────────────────────────────────────────────
1. **Enlargement** makes the picture bigger. SeedVR2 (a one-step diffusion
   restoration model) or an ESRGAN file. It decides what the *calm* areas look
   like, because the redraw on top of it keeps whatever fine texture it was
   handed.
2. **Redraw** — Ultimate SD Upscale re-diffuses the enlarged picture in 1024²
   tiles with Z-Image Turbo, the model that made these pictures, so real
   detail is painted at the new size. The Kreativität slider is how much.

MEASURED, 2026-09-26 — why the settings are what they are
─────────────────────────────────────────────────────────
Two 512² windows from real library pictures (VHS-filtered, grainy, one with a
fog wall and cloth folds, one with soft bokeh figures), 2x, fixed seed, each
factor changed on its own. Measured against a Lanczos enlargement of the
source, which by construction invents nothing:

  noise   fine grain added in CALM areas     (1.0 = as much as the source)
  mottle  cloudy blotches added in CALM areas
  detail  structure in BUSY areas            (1.0 = as sharp as the source)
  drift   how far large-scale content moved  (the creativity axis)

What the old pipeline did (4xNomos8kSC, res_multistep, AuraFlow shift 3,
denoise 0.25 — the combination nearly every upscale in the library used):
noise 2.0 / 1.6, mottle 1.34 / 1.10, detail 0.69 / 0.84. Twice the grain in
the calm areas, a woven cross-hatch pattern in them, and *less* structure than
plain Lanczos where there was structure. That is the complaint, measured.

The three causes, each with its own fix:

  • Grain → the enlargement model. 4xNomos8kSC was trained on JPEG
    compression and blur only (its model card), never on noise, so it takes
    the VHS grain for signal and sharpens it: 2.4 / 1.95 *before* any diffusion
    runs. RealESRGAN's training degradations include Gaussian, Poisson, colour
    and grey noise (Real-ESRGAN paper, §3.1), so it removes grain: 0.66 / 0.29.
    SeedVR2 keeps it at the source's own level: 1.00 / 0.70. The redraw passes
    through what it is given — a clean enlargement stays clean.
  • Blotches → the sampler and the start noise. res_multistep, a
    second-order multistep solver, invents low-frequency variation in flat
    areas when started part-way down the schedule (mottle 1.34); euler does
    not (0.99), and neither does a lower start noise (1.00).
  • Lost detail → too much of the tile repainted. See the shift note below.

Tested and found to make no difference, so deliberately absent: stripping
"VHS filter" and the LoRA triggers from the prompt, or an empty prompt
(noise 2.02 vs 2.04); loading the picture's own LoRAs; Detail Daemon at
−0.5 … +2 (detail 0.99 vs 1.00 — nine low-sigma steps leave it nothing to
work with). The beta scheduler was ~1–2 % better on detail and within noise
of simple, so the redraw keeps the scheduler the generator uses.

Result, same windows, Kreativität 0.25: RealESRGAN noise 0.65 / 0.30, mottle
0.69 / 0.87, detail 1.00 / 0.89; SeedVR2 noise 0.90 / 0.61, mottle 0.82 / 0.87,
detail 0.98 / 0.94. Both: sharp, faithful folds where the old pass re-invented
them, and no pattern in the fog.

The creativity slider is the start noise, not a denoise fraction
────────────────────────────────────────────────────────────────
With `denoise` fed to the KSampler under AuraFlow shift 3, the schedule tail
starts at σ = 3d / (1 + 2d): "0.25" was 50 % noise, "0.35" 61 %, "0.45" 71 %
— most of every tile repainted, which is how cloth folds came back as a
riveted padlock. So the redraw now gets explicit sigmas from a *shift-1*
schedule: σ₀ ≈ the slider value, in steps spaced evenly below it. The
sampling model keeps shift 3 — flow models read σ directly as the timestep
(σ·1000), so the shift only ever redistributes the schedule, and the
decomposed sampler at the old settings was verified identical to the node's
built-in one before anything was changed.

Measured drift along the new axis (RealESRGAN, window a): 0.25 → 1.5,
0.40 → 2.2, 0.55 → 3.2 (≈ the old default's 3.5). Blotches and softening grow
with it — that is what repainting more of a tile costs — so the default sits
low and the top end is for pictures that should be reimagined.

0 skips the redraw entirely: enlargement only, nothing invented.
"""
from __future__ import annotations

import random
from pathlib import Path

# ── Enlargement stages ───────────────────────────────────────────────────────
# `file` is an ESRGAN checkpoint in models/upscale_models; `None` is SeedVR2.
# Label and hint are what the gallery shows — the measured behaviour, stated
# plainly, because the old hint ("schärfer") steered people to the model that
# was the source of the grain.
#
# `default_denoise` differs on purpose. SeedVR2 on its own measured best of
# everything at 2x (detail 1.02, 40 s), and a 0.25 redraw on top was all but
# invisible for six times the wait — so it defaults to no redraw. An ESRGAN
# file on its own is either plastic (RealESRGAN) or grainy (Nomos) and has
# painted nothing, so those default to a light redraw.
ENLARGERS: dict[str, dict] = {
    "seedvr2": {
        "file": None,
        "label": "SeedVR2",
        "hint": "Restauriert: scharfe Kanten, ruhige Flächen, Korn wie im Original. Empfohlen.",
        "default_denoise": 0.0,
    },
    "realesrgan": {
        "file": "RealESRGAN_x4plus.pth",
        "label": "RealESRGAN",
        "hint": "Entfernt Korn und Rauschen — Flächen werden glatt, fast glasig.",
        "default_denoise": 0.25,
    },
    "nomos8ksc": {
        "file": "4xNomos8kSC.pth",
        "label": "Nomos8kSC",
        "hint": "Schärft auch Korn zu Rauschen — nur für saubere, fotografische Bilder.",
        "default_denoise": 0.25,
    },
}
DEFAULT_UPSCALE_MODEL = "seedvr2"

# Kept for callers that only need the ESRGAN file names.
UPSCALE_MODELS: dict[str, str] = {
    k: v["file"] for k, v in ENLARGERS.items() if v["file"]
}

# ── SeedVR2, as the video restoration pass runs it ───────────────────────────
# (services/video/upscale.py). 3B fp8 is the one that fits beside everything
# else on 16 GB; the tiled VAE is what keeps an 8 MP still from spiking.
_SVR_DIT = "seedvr2_ema_3b_fp8_e4m3fn.safetensors"
_SVR_VAE = "ema_vae_fp16.safetensors"
_SVR_VAE_TILE = 512
_SVR_VAE_OVERLAP = 64
# Restoration shifts colour a little; `lab` grades it back onto the source.
_SVR_COLOR = "lab"
# Fixed: restoring the same picture twice should give the same picture.
_SVR_SEED = 42
# MEASURED on the 16 GB 4060 Ti: one pass over a whole 1080x1920 → 2160x3840
# still (8.3 MP out) restores cleanly in 40 s; the same picture at 4x (33 MP
# out) dies in the DiT with 17.4 GiB allocated. Above this budget the picture
# goes through the tiling node instead — one pass is kept below it because it
# is the case that cannot possibly show a seam.
_SVR_SINGLE_PASS_MAX_PIXELS = 10_000_000
# Tiled path: every tile comes out at most this big (the size a single pass
# is proven at, with room to spare), so the input tile is this divided by the
# factor. Multiband blending (mask_blur 0) is the node's best-detail seam mode.
_SVR_TILE_OUT = 2048
_SVR_TILE_PADDING = 32     # source pixels of overlap between tiles

# ── Z-Image Turbo, identical to workflows/z-image_turbo.json ─────────────────
_UNET        = "z_image_turbo_bf16.safetensors"
_UNET_DTYPE  = "fp8_e4m3fn"
_CLIP        = "qwen_3_4b.safetensors"
_CLIP_TYPE   = "lumina2"
_VAE         = "ae.safetensors"
_SHIFT       = 3      # the sampling model, exactly as generated
_STEPS       = 9      # real sampled steps per tile
_CFG         = 1.0    # distilled, guidance-free; the negative is a zeroed copy
_SCHEDULER   = "simple"
# The redraw's own two settings — see the module docstring for the numbers.
_SAMPLER     = "euler"      # res_multistep blotches flat areas when started mid-schedule
_SCHED_SHIFT = 1.0          # σ₀ ≈ Kreativität, steps spaced evenly below it

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
# start-noise levels.
SEAM_FIX_MODE = "None"

# ── User-facing ranges ───────────────────────────────────────────────────────
# The slider is the redraw's start noise σ₀. 0 = enlargement only.
DENOISE_MIN     = 0.0
DENOISE_MAX     = 0.60    # ≈ the old default's drift; objects start being reread
DENOISE_DEFAULT = ENLARGERS[DEFAULT_UPSCALE_MODEL]["default_denoise"]

# (upper bound, label, what it does) — measured on the full picture at 2x,
# SeedVR2 underneath: 0.25 is indistinguishable from none, 0.45 paints
# material into surfaces (a flat orange wall gets rust grain), 0.60 starts
# rereading shapes (a dark fold became a hook and a bracket).
_DENOISE_BANDS: tuple[tuple[float, str, str], ...] = (
    (0.001, "Nur vergrößern",
     "Nichts wird neu gezeichnet — schnell, originaltreu."),
    (0.35, "Nachzeichnen",
     "Z-Image zeichnet die Kacheln leicht nach. Motiv und Korn bleiben."),
    (0.50, "Material",
     "Flächen bekommen gemalte Struktur dazu; die Formen bleiben."),
    (0.61, "Neu deuten",
     "Formen werden umgedeutet — kreativ, aber nicht mehr dasselbe Bild."),
)

SCALE_CHOICES  = (1.5, 2.0, 3.0, 4.0)
SCALE_DEFAULT  = 2.0

# A 1080x1920 source at 4x is 33 MP and ~48 tiles; the cap keeps a stray click
# from queueing a half-hour render and a 60 MB PNG.
MAX_OUTPUT_PIXELS = 40_000_000

# Wall-clock model, calibrated against measured runs on the 16 GB 4060 Ti:
#   redraw     1024² → 1536² (4 tiles) 101 s; 2160x3840 (12 tiles) 214 s
#              → a model load plus ~17 s per tile
#   SeedVR2    one pass 8.3 MP out: 40 s end to end
#              tiled    33 MP out:  312 s end to end
#   ESRGAN     a few seconds at any size — folded into the redraw's startup
_STARTUP_SECONDS  = 35.0
_SECONDS_PER_TILE = 17.0
_SVR_STARTUP_SECONDS    = 20.0
_SVR_SECONDS_PER_MP     = 2.5     # one pass
_SVR_TILED_SECONDS_PER_MP = 9.0   # tiled: overlap and a load per tile
_ESR_SECONDS = 8.0


def clamp_denoise(value: float | None, model: str | None = None) -> float:
    """Pull a Kreativität request into the measured range.

    `None` means "whatever suits this enlarger" — see `default_denoise`.
    """
    if value is None:
        return ENLARGERS[resolve_enlarger(model)]["default_denoise"]
    return round(min(max(float(value), DENOISE_MIN), DENOISE_MAX), 3)


def resolve_enlarger(key: str | None) -> str:
    return key if key in ENLARGERS else DEFAULT_UPSCALE_MODEL


def denoise_bands() -> list[dict]:
    lower = DENOISE_MIN
    out = []
    for upper, name, effect in _DENOISE_BANDS:
        out.append({"from": round(lower, 3), "to": round(min(upper, DENOISE_MAX), 3),
                    "band": name, "effect": effect})
        lower = upper
    return out


def describe_denoise(value: float) -> dict:
    v = clamp_denoise(value)
    for upper, name, effect in _DENOISE_BANDS:
        if v < upper:
            break
    return {"denoise": v, "band": name, "effect": effect}


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


def timing(src_w: int, src_h: int, scale: float) -> dict:
    """The two stages' costs apart, so a client can add up whatever the user
    picks without a round trip: SeedVR2 at no redraw is ~40 s, the same with a
    redraw is ~4 min, and the slider is what moves between them."""
    out_w, out_h = output_size(src_w, src_h, scale)
    mp = out_w * out_h / 1e6
    per_mp = (_SVR_SECONDS_PER_MP if out_w * out_h <= _SVR_SINGLE_PASS_MAX_PIXELS
              else _SVR_TILED_SECONDS_PER_MP)
    return {
        "enlarge": {
            key: int(_SVR_STARTUP_SECONDS + per_mp * mp) if spec["file"] is None
            else int(_ESR_SECONDS)
            for key, spec in ENLARGERS.items()
        },
        "redraw": int(_STARTUP_SECONDS + tile_count(out_w, out_h) * _SECONDS_PER_TILE),
    }


def estimate_seconds(
    src_w: int, src_h: int, scale: float,
    *, model: str = DEFAULT_UPSCALE_MODEL, denoise: float | None = None,
) -> int:
    t = timing(src_w, src_h, scale)
    model = resolve_enlarger(model)
    secs = t["enlarge"][model]
    if clamp_denoise(denoise, model) > 0:
        # The ESRGAN pass runs inside the redraw node, so it is not paid twice.
        secs = (secs if model == "seedvr2" else 0) + t["redraw"]
    return int(max(secs, 10))


def _seedvr2_nodes(
    p: str, image_ref: list, *, src_w: int, src_h: int, scale: float,
) -> dict:
    """SeedVR2 loaders plus the node that does the enlargement.

    One pass while the output fits the measured budget; the tiling node above
    it. Both size by the SHORT side, so the long side follows the source's
    aspect rather than being rounded separately.
    """
    short_side = round(min(src_w, src_h) * scale)
    out_w, out_h = output_size(src_w, src_h, scale)
    nodes = _seedvr2_loaders(p)
    if out_w * out_h <= _SVR_SINGLE_PASS_MAX_PIXELS:
        nodes[p+"svr"] = {"class_type": "SeedVR2VideoUpscaler", "inputs": {
            "image": image_ref, "dit": [p+"svr_dit", 0], "vae": [p+"svr_vae", 0],
            "seed": _SVR_SEED, "resolution": short_side, "max_resolution": 0,
            "batch_size": 1, "uniform_batch_size": False,
            "color_correction": _SVR_COLOR, "offload_device": "cpu",
        }}
    else:
        # Input tiles sized so each comes out at _SVR_TILE_OUT: the node
        # scales every tile by the overall factor and only *caps* at
        # tile_upscale_resolution — a cap below tile×factor would upscale the
        # tile short and then stretch it, which is exactly the softness this
        # whole pass exists to avoid.
        tile = max(256, int(_SVR_TILE_OUT / scale) // 8 * 8)
        nodes[p+"svr"] = {"class_type": "SeedVR2TilingUpscaler", "inputs": {
            "image": image_ref, "dit": [p+"svr_dit", 0], "vae": [p+"svr_vae", 0],
            "seed": _SVR_SEED, "new_resolution": short_side,
            "resolution_target": "shortest",
            "tile_width": tile, "tile_height": tile,
            "tile_padding": _SVR_TILE_PADDING,
            "tile_upscale_resolution": _SVR_TILE_OUT,
            "mask_blur": 0, "blending_method": "auto",
            "tiling_strategy": "Chess", "anti_aliasing_strength": 0.0,
            "color_correction": _SVR_COLOR, "tile_batch_size": 1,
        }}
    return nodes


def _seedvr2_loaders(p: str) -> dict:
    return {
        p+"svr_dit": {"class_type": "SeedVR2LoadDiTModel", "inputs": {
            "model": _SVR_DIT, "device": "cuda:0", "blocks_to_swap": 0,
            "swap_io_components": False, "offload_device": "cpu",
            # Not cached: the Z-Image redraw needs the card straight after.
            "cache_model": False, "attention_mode": "sdpa",
        }},
        p+"svr_vae": {"class_type": "SeedVR2LoadVAEModel", "inputs": {
            "model": _SVR_VAE, "device": "cuda:0",
            "encode_tiled": True, "encode_tile_size": _SVR_VAE_TILE,
            "encode_tile_overlap": _SVR_VAE_OVERLAP,
            "decode_tiled": True, "decode_tile_size": _SVR_VAE_TILE,
            "decode_tile_overlap": _SVR_VAE_OVERLAP,
            "offload_device": "cpu", "cache_model": False,
        }},
    }


def build_image_upscale_workflow(
    image_name: str,
    *,
    prompt: str,
    denoise: float,
    scale: float,
    upscale_model: str = DEFAULT_UPSCALE_MODEL,
    seed: int | None = None,
    filename_prefix: str = "artrium_imgup",
    src_width: int = 0,
    src_height: int = 0,
) -> tuple[dict, str]:
    """Build the ComfyUI workflow. Returns (workflow, save_node_id).

    `image_name` is the file as ComfyUI's /upload/image named it. `prompt`
    should be the image's own generation prompt when there is one — it steers
    what a high-Kreativität redraw paints toward the picture's own subject.
    `src_width`/`src_height` are only needed for SeedVR2, which sizes by the
    short side rather than by a factor.
    """
    enlarger = resolve_enlarger(upscale_model)
    spec = ENLARGERS[enlarger]
    sigma0 = clamp_denoise(denoise, enlarger)
    scale = round(float(scale), 2)
    p = "iu_"

    wf: dict = {p+"load": {"class_type": "LoadImage", "inputs": {"image": image_name}}}

    # ── Stage 1: enlargement ─────────────────────────────────────────────────
    if enlarger == "seedvr2":
        if not (src_width and src_height):
            raise ValueError("SeedVR2 needs the source size to pick its resolution")
        wf.update(_seedvr2_nodes(
            p, [p+"load", 0], src_w=src_width, src_h=src_height, scale=scale,
        ))
        enlarged = [p+"svr", 0]
    else:
        wf[p+"upmod"] = {"class_type": "UpscaleModelLoader",
                         "inputs": {"model_name": spec["file"]}}
        enlarged = None     # USDU runs the ESRGAN model itself, below

    # ── Kreativität 0: enlargement only, nothing redrawn ─────────────────────
    if sigma0 <= 0:
        if enlarged is None:
            wf[p+"esr"] = {"class_type": "ImageUpscaleWithModel", "inputs": {
                "upscale_model": [p+"upmod", 0], "image": [p+"load", 0]}}
            # The ESRGAN files are all 4x; land on the asked-for factor.
            wf[p+"fit"] = {"class_type": "ImageScaleBy", "inputs": {
                "image": [p+"esr", 0], "upscale_method": "lanczos",
                "scale_by": round(scale / 4.0, 4)}}
            enlarged = [p+"fit", 0]
        wf[p+"save"] = {"class_type": "SaveImage", "inputs": {
            "filename_prefix": filename_prefix, "images": enlarged}}
        return wf, p + "save"

    # ── Stage 2: the tiled Z-Image redraw ────────────────────────────────────
    wf.update({
        p+"unet":  {"class_type": "UNETLoader", "inputs": {
            "unet_name": _UNET, "weight_dtype": _UNET_DTYPE,
        }},
        p+"shift": {"class_type": "ModelSamplingAuraFlow", "inputs": {
            "shift": _SHIFT, "model": [p+"unet", 0],
        }},
        # Seen only by the scheduler. Flow models take σ as the timestep
        # directly, so this redistributes the steps without changing what the
        # sampling model is.
        p+"sshift": {"class_type": "ModelSamplingAuraFlow", "inputs": {
            "shift": _SCHED_SHIFT, "model": [p+"unet", 0],
        }},
        p+"sigmas": {"class_type": "BasicScheduler", "inputs": {
            "model": [p+"sshift", 0], "scheduler": _SCHEDULER,
            "steps": _STEPS, "denoise": sigma0,
        }},
        p+"sampler": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": _SAMPLER}},
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
    })
    usdu_inputs = {
        "model":        [p+"shift", 0],
        "positive":     [p+"pos", 0],
        "negative":     [p+"neg", 0],
        "vae":          [p+"vae", 0],
        "seed":         seed if seed is not None and seed >= 0 else random.randint(0, 2**32 - 1),
        # The custom-sample node ignores these four once custom_sampler and
        # custom_sigmas are both wired (it calls SamplerCustom with them), but
        # they are required inputs. Set to what they would mean, not junk.
        "steps":        _STEPS,
        "cfg":          _CFG,
        "sampler_name": _SAMPLER,
        "scheduler":    _SCHEDULER,
        "denoise":      sigma0,
        "custom_sampler": [p+"sampler", 0],
        "custom_sigmas":  [p+"sigmas", 0],
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
    }
    if enlarged is None:
        # ESRGAN: USDU enlarges with the model itself, then tiles.
        usdu_inputs |= {"image": [p+"load", 0], "upscale_model": [p+"upmod", 0],
                        "upscale_by": scale}
    else:
        # SeedVR2 already produced the target size; USDU only redraws at 1x.
        usdu_inputs |= {"image": enlarged, "upscale_by": 1.0}
    wf[p+"usdu"] = {"class_type": "UltimateSDUpscaleCustomSample", "inputs": usdu_inputs}
    wf[p+"save"] = {"class_type": "SaveImage", "inputs": {
        "filename_prefix": filename_prefix, "images": [p+"usdu", 0],
    }}
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
