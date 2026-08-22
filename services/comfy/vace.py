"""
Wan VACE structure-video builder — geometry decides the form, reference images
decide the material.

This is the successor to the user's AnimateDiff/LCM/ControlNet/IP-Adapter
graph. That graph kept three concerns apart, and the separation is what made it
work: a depth video from Blender set space and motion, a colour-coded mask video
said *where* things were, and one IP-Adapter reference per region said what each
region was made of. VACE takes all three as inputs of a single node, so the
three parallel ~15-node channel apparatuses collapse into a chain.

How VACE actually reads its inputs, from comfy_extras/nodes_wan.py
─────────────────────────────────────────────────────────────────
Worth writing down, because the node's names invite a wrong mental model.

`control_masks` is **not** a "which reference applies here" selector. It is an
inpainting mask, and the node splits the control video by it:

    inactive = control_video * (1 - mask)    # preserved as context
    reactive = control_video * mask          # the part being generated

Both halves are VAE-encoded and handed to the model together. With no mask the
whole frame is reactive, which is the plain "drive the whole picture from this
depth video" case.

`reference_image` takes **one** image (`reference_image[:1]`), encodes it and
prepends it to the control latent as extra frames — which is why the node also
returns `trim_latent`, the number of frames `TrimVideoLatent` must cut off the
sampler's output afterwards.

Which makes one workflow one region, and multi-region work a *sequence* of
passes rather than one clever graph. Chaining several masked blocks into a
single pass looks like it should work — the node appends to `vace_frames` /
`vace_mask` / `vace_strength` (`append=True`), model_base.py stacks the list,
and the model sums the blocks (`x += c_skip * vace_strength[iii]`) — but it
renders a grey box, and the reason is in the two lines above. Three narrow
masks say "preserve almost everything" three times over, and the thing being
preserved is the depth pass, which is grey.

Preserving only means something once the control video is a *picture*. So
regions work the way the old graph did them, one pass each:

    pass 0   depth video, no mask, reference A      -> a full render
    pass 1   control = pass 0's render, mask = B, reference B
    pass 2   control = pass 1's render, mask = C, reference C

`plan_region_passes` lays that out. Measured on the reference cube: the change
from a region pass is ~3x larger inside its mask than outside (16-20 vs 5.5-6
mean absolute difference), so the base render really does survive and the
region really is repainted.

"""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from pathlib import Path

# ── Models ───────────────────────────────────────────────────────────────────
# Two sizes, one interface. The sketch tier exists to find the right `strength`
# cheaply; the final tier runs the identical graph at the same seed.
SKETCH_UNET = "wan2.1_vace_1.3B_fp16.safetensors"          # models/diffusion_models
FINAL_UNET = "Wan2.1_14B_VACE-Q5_K_M.gguf"                 # models/unet, needs ComfyUI-GGUF
SPEED_LORA = "Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors"

_CLIP = "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
_CLIP_TYPE = "wan"
_VAE = "wan_2.1_vae.safetensors"

# Wan's own cadence — 81 frames at 16 fps is what the model was trained on, and
# it is the rate the *output* is written at. The control track is deliberately
# NOT resampled to it; see `_load_video` for why that costs guidance.
FPS = 16
DEFAULT_LENGTH = 81          # must be 4n+1
DEFAULT_SHIFT = 8.0          # ModelSamplingSD3, per the official VACE template

# Both tiers run the SAME 14B model, and the 1.3B is not used at all. Measured
# 2026-08-19, and it settled the question the two-tier design was built on:
#
#   1.3B  480x480, 81 frames, 20 steps, no LoRA   226 s
#   14B   480x480, 81 frames,  6 steps + lightx2v 229 s
#   14B   720x720, 81 frames,  6 steps + lightx2v 575 s   (fits 16 GB)
#
# The distilled LoRA buys back everything the larger model costs, so at 480 the
# 1.3B is the same price for a visibly poorer picture — dark, flat, and without
# the environment the 14B invents. There is no reason to load it.
#
# What the same measurement also showed, and it is the uncomfortable half: at a
# fixed seed, strength and prompt, the 1.3B and the 14B produce **completely
# different scenes**, and so do the 14B at 480 and at 720. Changing the latent's
# shape changes what a seed means; that is diffusion, not a bug. So a cheap tier
# can preview the *direction* — palette, material, how literally the geometry is
# taken — but never the actual clip. Nothing here should be sold as a preview of
# the final frame, because it is not one.
# ── Canvas and recipe: two axes that used to be one ──────────────────────────
# `tier` conflated the render size with the sampler settings, and the conflation
# made the one experiment that matters impossible to run. A seed means something
# different at every latent shape — measured above — so "is the distilled
# sampler what costs the look?" can only be answered by changing the sampler and
# nothing else. Size and recipe are therefore chosen separately.

# Wan 2.1's own trained shapes. 832x480 is what the model calls 480p: 1.73:1,
# close enough to frame 16:9 with, and named by its pixels rather than by a
# ratio the model was never given. Square is kept because the Blender turntables
# are square, but it is off-distribution for Wan and it shows.
_CANVASES: dict[str, dict[str, tuple[int, int]]] = {
    "sketch": {"wide": (832, 480), "tall": (480, 832), "square": (480, 480)},
    "final": {"wide": (1280, 720), "tall": (720, 1280), "square": (720, 720)},
}
DEFAULT_CANVAS = "sketch"
DEFAULT_ASPECT = "wide"

# The sampler recipes. `fast` is what every render so far has used, and it is
# the prime suspect for why VACE comes back smoother and less hand-made than the
# old AnimateDiff/LCM graph: a cfg-step-distill LoRA at 6 steps buys its speed
# with exactly the micro-texture and prompt nuance this tool exists for.
#
# Two consequences of cfg, both easy to get wrong:
#   - At cfg 1.0 the negative prompt is never evaluated. Under `fast` the
#     negative below is inert, whatever it says.
#   - Above 1.0 every step runs the model twice. `full` is 20 steps at cfg, so
#     40 model evaluations against `fast`'s 6 — about seven times the sampling,
#     not three.
_RECIPES: dict[str, dict] = {
    "fast": {
        "unet": FINAL_UNET, "gguf": True,
        "lora": SPEED_LORA, "lora_strength": 1.0,
        "steps": 6, "cfg": 1.0,
    },
    # The middle path: the distill LoRA weakened rather than removed, paid for
    # in steps, and still at cfg 1.0 so it stays one model evaluation per step.
    # Untested — it exists to be measured against the other two.
    "rich": {
        "unet": FINAL_UNET, "gguf": True,
        "lora": SPEED_LORA, "lora_strength": 0.5,
        "steps": 12, "cfg": 1.0,
    },
    # No distillation at all: the model as trained, with a real classifier-free
    # guidance pass. This is the A/B that decides whether the missing look is a
    # settings problem or a model problem.
    "full": {
        "unet": FINAL_UNET, "gguf": True,
        "lora": None, "lora_strength": 0.0,
        "steps": 20, "cfg": 4.5,
    },
    # The 1.3B, kept as a look and not as a draft — darker, flatter, with far
    # less invented environment, and closer to the user's own gallery than the
    # 14B's brighter picture. Measured at the same price as the 14B, so it is
    # never the cheap option.
    "small": {
        "unet": SKETCH_UNET, "gguf": False,
        "lora": None, "lora_strength": 0.0,
        "steps": 20, "cfg": 4.0,
    },
}
DEFAULT_RECIPE = "fast"

# How a control track that is not the canvas's shape gets there.
#   pad      fit inside, fill the rest with black. On an inverted depth pass
#            black is "infinitely far", so the bars become background the model
#            invents into — which is the point of a wide canvas over a square
#            turntable. On a colour-ID mask black matches no region.
#   crop     fill the canvas and cut the overflow. Keeps proportions, loses the
#            top and bottom of a square source pushed wide.
#   stretch  what this builder did before: distorts the geometry to fit.
FITS = ("pad", "crop", "stretch")
DEFAULT_FIT = "pad"
_FIT_METHODS = {"pad": "pad", "crop": "fill / crop", "stretch": "stretch"}

SAMPLER = "uni_pc"
SCHEDULER = "simple"

# ── Strength: the one dial that matters ──────────────────────────────────────
# The node accepts 0.0–1000.0. The useful window is a narrow slice near the
# bottom of that, and it is not where intuition puts it.
#
# Measured 2026-08-19 on the sketch tier: one rotating Blender cube as the depth
# track, one of the user's own pictures as the reference, fixed seed, five
# strengths. What the contact sheet shows, and it is not subtle:
#
#   0.2  The cube is gone. A tall spire on a pooled base in a lit hall — the
#        prompt and the reference own the frame; the depth only says "a vertical
#        mass, here".
#   0.4  A molten, glossy sculpture over a liquid floor. Beautiful, and the
#        closest thing to the user's own paint-pour work in the whole sheet —
#        but the source geometry is unreadable.
#   0.6  Both at once: the cube is legible and rotating correctly, its faces are
#        painterly rust and iridescent blue-green, and it stands in a reflective
#        pool under coloured light. This is "the cube becomes a world".
#   0.8  A cube. Clean, lit, literal. The world has mostly drained away.
#   1.0  A cube on black. The most faithful and the least interesting frame in
#        the sheet.
#
# So the world-building lives *below* full strength, and it switches off fast
# between 0.6 and 0.8 — the environment is far more sensitive than the geometry.
# Anything at or above 0.9 is a rendering of the control video, not a variation
# on it.
STRENGTH_MIN = 0.10
STRENGTH_MAX = 1.00
STRENGTH_DEFAULT = 0.60

# The band to point the user at. Narrow on purpose: outside it you get either a
# grey box or an unrecognisable blob, and both are one slider-nudge away.
SWEET_MIN, SWEET_MAX = 0.50, 0.70

# Below the band the source stops being readable but the results are often the
# most striking — worth naming rather than hiding, since it is a legitimate
# place to work.
FREE_MIN, FREE_MAX = 0.30, 0.50

SWEEP_VALUES = (0.2, 0.4, 0.6, 0.8, 1.0)
REFINE_VALUES = (0.50, 0.55, 0.60, 0.65, 0.70)

_STRENGTH_BANDS: tuple[tuple[float, str, str], ...] = (
    (0.30, "Frei",
     "Die Geometrie verschwindet. Prompt und Referenzbild bestimmen fast alles."),
    (0.50, "Skulptural",
     "Aus der Vorlage wird eine verwandte Form — Masse und Ort stimmen, die Gestalt nicht."),
    (0.75, "Welt",
     "Die Vorlage ist lesbar und steht zugleich in einer erfundenen Umgebung. Der Zielbereich."),
    (1.01, "Wörtlich",
     "Das Steuervideo wird ausgemalt, mehr nicht. Die Umgebung fällt weg."),
)


def describe_strength(value: float) -> dict:
    """Band name + effect for a strength value, for the slider readout."""
    strength = clamp_strength(value)
    for upper, name, effect in _STRENGTH_BANDS:
        if strength < upper:
            break
    return {
        "strength": strength,
        "band": name,
        "effect": effect,
        "recommended": SWEET_MIN <= strength <= SWEET_MAX,
    }


def strength_bands() -> list[dict]:
    """The band table, for a client that labels its own slider."""
    lower, out = STRENGTH_MIN, []
    for upper, name, effect in _STRENGTH_BANDS:
        out.append({"from": round(lower, 2), "to": round(min(upper, STRENGTH_MAX), 2),
                    "band": name, "effect": effect})
        lower = upper
        if lower >= STRENGTH_MAX:
            break
    return out

# A depth pass rendered out of Blender usually arrives with the background
# white and the subject dark — the opposite of what the models read as depth.
# The old graph fixed this with an ImageInvert before its ControlNet; this one
# keeps the same switch rather than silently assuming an orientation.
DEFAULT_INVERT_DEPTH = True

# Wan's stock negative prompt, shipped with the official templates. Note what
# it forbids: 风格化 (stylised), 作品 / 画作 / 画面 (artwork / painting /
# picture), 整体发灰 (greyish overall). For a tool whose whole purpose is a
# painted, stylised, deliberately desaturated picture, that is the wrong list.
STOCK_NEGATIVE = (
    "静态, 过曝, 模糊, 字幕, 风格化, 作品, 画作, 画面, 静止, 整体发灰, 最差质量, "
    "低质量, JPEG压缩残留, 丑陋的, 残缺的, 多余的手指, 画得不好的手部, 画得不好的脸部, "
    "畸形的, 毁容的, 形态畸形的肢体, 手指融合, 静止不动的画面, 杂乱的背景, 三条腿, "
    "背景人很多, 倒着走"
)

# The same list minus the five terms above. The anatomy and compression terms
# stay — they cost nothing and catch real failures — and so do the "static"
# terms, which keep the motion alive.
#
# Worth knowing before reading anything into a render: this only takes effect
# under a recipe with cfg > 1. At cfg 1.0 there is no unconditional pass and the
# negative is never evaluated at all.
ARTISTIC_NEGATIVE = (
    "静态, 过曝, 模糊, 字幕, 静止, 最差质量, 低质量, JPEG压缩残留, 丑陋的, 残缺的, "
    "多余的手指, 画得不好的手部, 画得不好的脸部, 畸形的, 毁容的, 形态畸形的肢体, "
    "手指融合, 静止不动的画面, 三条腿, 背景人很多, 倒着走"
)

_DEFAULT_NEGATIVE = ARTISTIC_NEGATIVE


@dataclass
class Region:
    """One colour region of the mask video, with the material it is made of.

    `color` is the RGB triple Blender wrote for this object — pure red, green
    or blue in the reference scene. `reference` is the ComfyUI input filename
    of the picture that region should look like.
    """
    color: tuple[int, int, int]
    reference: str
    strength: float = STRENGTH_DEFAULT
    threshold: int = 20      # the old graph's ColorToMask tolerance
    invert: bool = False


@dataclass
class VaceRequest:
    control_video: str                       # absolute path to the depth video
    prompt: str
    canvas: str = DEFAULT_CANVAS             # render size:   sketch | final
    aspect: str = DEFAULT_ASPECT             # shape:         wide | tall | square
    recipe: str = DEFAULT_RECIPE             # sampler:       fast | rich | full | small
    fit: str = DEFAULT_FIT                   # how the track meets the canvas
    # Grade the render back onto the control track's palette, 0 = off. Only
    # meaningful when the control track is a picture; against a depth pass it
    # would grade the result towards grey.
    color_match: float = 0.0
    negative: str = _DEFAULT_NEGATIVE
    reference_image: str | None = None       # ComfyUI input filename; no-mask case
    mask_video: str | None = None            # absolute path; required by `regions`
    regions: list[Region] = field(default_factory=list)
    width: int | None = None                 # None = the canvas's own size
    height: int | None = None
    length: int = DEFAULT_LENGTH
    strength: float = STRENGTH_DEFAULT
    seed: int = -1
    steps: int | None = None
    cfg: float | None = None
    shift: float = DEFAULT_SHIFT
    invert_depth: bool = DEFAULT_INVERT_DEPTH
    # True when the control track is ordinary footage rather than a depth pass:
    # DepthAnythingV2 derives the depth in-graph. A Blender Z/Mist render is
    # already depth and must NOT go through it.
    derive_depth: bool = False
    depth_model: str = "depth_anything_v2_vitl.pth"
    # 0 = take the control track's own frames. Anything else resamples and
    # risks running out of guidance before `length` — see `_load_video`.
    force_rate: float = 0.0
    # What the control track actually holds, when the caller has probed it.
    # `length` is clamped to it so VACE is never asked for frames that would
    # come back as flat grey.
    source_frames: int | None = None
    filename_prefix: str = "artrium_vace"


def clamp_strength(value: float | None) -> float:
    if value is None:
        return STRENGTH_DEFAULT
    return round(min(max(float(value), STRENGTH_MIN), STRENGTH_MAX), 2)


def snap_length(frames: int) -> int:
    """Wan samples 4n+1 frames; anything else is silently wrong."""
    return max(5, ((max(1, frames) - 1) // 4) * 4 + 1)


def snap_size(value: int) -> int:
    """VACE takes multiples of 16 (the node declares step=16)."""
    return max(16, (int(value) // 16) * 16)


def loop_length(source_frames: int) -> int:
    """The longest valid Wan length that fits inside a looping control track.

    A 3D turntable render already loops: the last frame is one rotation step
    short of the first, so playing the whole range and wrapping is seamless.
    Nothing needs to be added to make that work — the only thing standing in
    the way is Wan's 4n+1 requirement, and the fix is to take the largest
    4n+1 length the track can fill rather than padding it out.

    Measured on the reference cube: 94 rendered frames, frame 0 against frame
    93 differs only by a few pixels of edge, i.e. exactly one step. `93` is
    4·23+1, so the whole turn is usable and the wrap costs a single frame of
    rotation — below the threshold of noticing at 16 fps.

    Which makes this `snap_length` under another name. It earns its own name
    anyway: callers reach for it to say "as much of this loop as Wan will
    take", and that intent is invisible at the call site otherwise.
    """
    return snap_length(source_frames)


def _load_video(path: str, length: int, force_rate: float) -> dict:
    """Load a control track.

    `force_rate` defaults to 0 — the source's own frames, untouched — and that
    default is load-bearing. Resampling a 30 fps, 94-frame Blender render to
    Wan's 16 fps yields 50 frames; ask VACE for 81 and it pads the missing 31
    with `value=0.5`, i.e. flat mid-grey (nodes_wan.py). The tail of the clip
    then renders with no structure guidance at all and quietly drifts, which
    is not an error anyone sees until they watch the end of the video.

    Taking the frames as they are instead means the move plays back over
    `length / FPS` seconds rather than its original duration — a 3-second
    rotation becomes a 5-second one. That is a look, not a fault, and it is
    the honest trade against losing guidance entirely.
    """
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


# ── What the UI shows for the two axes ───────────────────────────────────────
# Here rather than in the router for the same reason the strength bands are:
# every one of these sentences is a claim about this graph, and it goes stale
# the moment the graph is retuned.
_CANVAS_LABELS: dict[str, tuple[str, str]] = {
    "sketch": ("Skizze · 480",
               "Zum Finden von Stärke, Prompt und Look."),
    "final": ("Final · 720",
              "Rund 2,5x so teuer. Die Szene fällt anders aus als in der Skizze — "
              "ein Seed bedeutet bei jeder Latent-Größe etwas anderes."),
}
_ASPECT_LABELS: dict[str, tuple[str, str]] = {
    "wide": ("Breit · 16:9",
             "Wans trainiertes 480p/720p-Format. Eine quadratische Vorlage steht "
             "mittig, die Seiten erfindet das Modell."),
    "tall": ("Hoch · 9:16", "Für Reels und Stories."),
    "square": ("Quadratisch",
               "Passt zur Blender-Vorlage, ist für Wan aber ein untrainiertes "
               "Format — Komposition und Umgebung fallen schwächer aus."),
}
_RECIPE_LABELS: dict[str, tuple[str, str]] = {
    "fast": ("Schnell · 6 Steps",
             "Distill-LoRA bei cfg 1.0. Bisher der einzige Modus — und der "
             "Verdächtige für die fehlende Textur. Der Negativ-Prompt wirkt hier nicht."),
    "rich": ("Reich · 12 Steps",
             "Distill-LoRA halbiert, doppelte Schrittzahl, weiterhin cfg 1.0. "
             "Der Mittelweg, noch ungemessen."),
    "full": ("Voll · 20 Steps · cfg 4.5",
             "Ohne Distillation, mit echtem Guidance-Durchlauf. Der A/B-Test: "
             "kommt der Look zurück, war es eine Einstellung. Rund 7x so teuer wie Schnell."),
    "small": ("Klein · 1.3B",
              "Dunkler, flacher, weniger erfundene Umgebung. Ein Look, kein Entwurf."),
}


def canvas_options() -> list[dict]:
    return [
        {"key": key, "label": label, "hint": hint,
         "sizes": {a: list(_CANVASES[key][a]) for a in _CANVASES[key]}}
        for key, (label, hint) in _CANVAS_LABELS.items()
    ]


def aspect_options() -> list[dict]:
    return [{"key": key, "label": label, "hint": hint}
            for key, (label, hint) in _ASPECT_LABELS.items()]


def recipe_options() -> list[dict]:
    return [
        {"key": key, "label": label, "hint": hint,
         "steps": _RECIPES[key]["steps"], "cfg": _RECIPES[key]["cfg"],
         # The one fact a caller needs to reason about the negative prompt.
         "negative_active": _RECIPES[key]["cfg"] > 1.0}
        for key, (label, hint) in _RECIPE_LABELS.items()
    ]


def canvas_size(canvas: str, aspect: str) -> tuple[int, int]:
    """Pixel size for a canvas and an aspect, snapped to VACE's step of 16."""
    shapes = _CANVASES.get(canvas) or _CANVASES[DEFAULT_CANVAS]
    width, height = shapes.get(aspect) or shapes[DEFAULT_ASPECT]
    return snap_size(width), snap_size(height)


def _fit_node(image_ref, width: int, height: int, fit: str, interpolation: str) -> dict:
    """Bring a control track onto the canvas.

    `ImageResize+` (comfyui_essentials) rather than core `ImageScale`, because
    only it can pad: core scaling either stretches or centre-crops, and a square
    turntable on a wide canvas needs neither. Its padding is black, which on an
    inverted depth pass reads as "infinitely far" and gives the model empty
    background to invent into rather than a wall at mid-distance.
    """
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


# Sampling cost, for an ETA shown before someone starts a 50-minute render by
# accident. Anchored on the two timings measured 2026-08-19 (14B, 81 frames,
# 6 distilled steps): 229 s at 480x480 and 575 s at 720x720.
#
#   (575 - 229) / ((720*720 - 480*480) * 81 * 6) = 2.5e-6 s per pixel-frame
#   per model evaluation
#
# Fitted proportionally rather than affinely — the two points imply a negative
# fixed cost, i.e. the curve is slightly superlinear — so 2.3e-6 splits the
# difference and lands within ~20% at both ends. An ETA, not a promise.
_SECONDS_PER_PIXEL_FRAME_EVAL = 2.3e-6


def estimate_seconds(canvas: str, aspect: str, recipe: str, length: int) -> int:
    """Rough wall-clock for one pass. Above cfg 1.0 each step costs two model
    evaluations, which is the part people underestimate."""
    spec = _RECIPES.get(recipe) or _RECIPES[DEFAULT_RECIPE]
    width, height = canvas_size(canvas, aspect)
    evals = spec["steps"] * (2 if spec["cfg"] > 1.0 else 1)
    return int(width * height * length * evals * _SECONDS_PER_PIXEL_FRAME_EVAL)


def build_vace_workflow(req: VaceRequest) -> tuple[dict, str]:
    """Build the ComfyUI graph. Returns (workflow, video_output_node_id).

    Three shapes come out of the same builder, which is the point:

      - depth video + one reference image        → one VACE block, no mask
      - the same, mask restricting the redraw    → one block with a mask
      - depth + 1..3 colour regions              → one block per region, chained
    """
    recipe = _RECIPES[req.recipe]
    canvas_w, canvas_h = canvas_size(req.canvas, req.aspect)
    width = snap_size(req.width or canvas_w)
    height = snap_size(req.height or canvas_h)
    # Never ask for more frames than the control track can guide: the surplus
    # would come back as flat grey and the clip would drift at the end.
    length = snap_length(min(req.length, req.source_frames or req.length))
    seed = req.seed if req.seed >= 0 else random.randint(0, 2**32 - 1)
    steps = req.steps if req.steps is not None else recipe["steps"]
    cfg = req.cfg if req.cfg is not None else recipe["cfg"]

    if req.regions and not req.mask_video:
        raise ValueError("regions need a mask_video to key them out of")
    # One workflow is one sampling pass, and one pass repaints one region. Use
    # plan_region_passes() to lay out a multi-region job; chaining several
    # masked blocks into a single pass renders the control video instead (see
    # the module docstring).
    if len(req.regions) > 1:
        raise ValueError(
            "one region per pass — use plan_region_passes() for several"
        )
    if req.regions and not req.regions[0].reference:
        raise ValueError("a region needs a reference image")

    p = "vc_"
    wf: dict = {}

    # ── Model chain ──────────────────────────────────────────────────────────
    if recipe["gguf"]:
        wf[p + "unet"] = {
            "class_type": "UnetLoaderGGUF",
            "inputs": {"unet_name": recipe["unet"]},
        }
    else:
        wf[p + "unet"] = {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": recipe["unet"], "weight_dtype": "default"},
        }
    model_ref = [p + "unet", 0]

    if recipe["lora"]:
        wf[p + "lora"] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "lora_name": recipe["lora"],
                "strength_model": float(recipe["lora_strength"]),
                "model": model_ref,
            },
        }
        model_ref = [p + "lora", 0]

    wf[p + "shift"] = {
        "class_type": "ModelSamplingSD3",
        "inputs": {"shift": float(req.shift), "model": model_ref},
    }

    # ── Text + VAE ───────────────────────────────────────────────────────────
    wf[p + "clip"] = {
        "class_type": "CLIPLoader",
        "inputs": {"clip_name": _CLIP, "type": _CLIP_TYPE, "device": "default"},
    }
    wf[p + "pos"] = {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": req.prompt, "clip": [p + "clip", 0]},
    }
    wf[p + "neg"] = {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": req.negative, "clip": [p + "clip", 0]},
    }
    wf[p + "vae"] = {"class_type": "VAELoader", "inputs": {"vae_name": _VAE}}

    # ── Control video ────────────────────────────────────────────────────────
    wf[p + "ctrl"] = _load_video(req.control_video, length, req.force_rate)
    ctrl_ref = [p + "ctrl", 0]

    # Ordinary footage becomes a depth pass here. DepthAnything already outputs
    # the orientation the models expect — near bright, far dark — so a derived
    # track must not also be inverted, whatever the flag says.
    if req.derive_depth:
        wf[p + "depth"] = {
            "class_type": "DepthAnythingV2Preprocessor",
            "inputs": {
                "image": ctrl_ref,
                "ckpt_name": req.depth_model,
                # Match the render canvas: the preprocessor's own default of 512
                # would resample the track twice for no gain.
                "resolution": max(width, height),
            },
        }
        ctrl_ref = [p + "depth", 0]
    elif req.invert_depth:
        wf[p + "inv"] = {"class_type": "ImageInvert", "inputs": {"image": ctrl_ref}}
        ctrl_ref = [p + "inv", 0]

    # Fit explicitly. VACE's own fallback is a *centre crop*, which would eat
    # the edges of a square Blender render pushed into a wide canvas.
    wf[p + "fit"] = _fit_node(ctrl_ref, width, height, req.fit, "lanczos")
    ctrl_ref = [p + "fit", 0]

    # ── Masks ────────────────────────────────────────────────────────────────
    if req.mask_video:
        wf[p + "maskvid"] = _load_video(req.mask_video, length, req.force_rate)
        # The same fit as the control track, or the regions land somewhere
        # other than the thing they were keyed off. Nearest so the pure R/G/B
        # of a colour-ID mask survives the resample as pure R/G/B.
        wf[p + "maskfit"] = _fit_node(
            [p + "maskvid", 0], width, height, req.fit, "nearest-exact",
        )

    # ── VACE chain: one block per region, or a single block without ─────────
    blocks: list[tuple[str | None, str | None, float]] = []   # (mask_node, reference, strength)
    if req.regions:
        region = req.regions[0]
        node = p + "mask0"
        wf[node] = {
            "class_type": "ColorToMask",
            "inputs": {
                "images": [p + "maskfit", 0],
                "invert": region.invert,
                "red": region.color[0], "green": region.color[1], "blue": region.color[2],
                "threshold": region.threshold,
                "per_batch": 16,
            },
        }
        blocks.append((node, region.reference, clamp_strength(region.strength)))
    else:
        blocks.append((None, req.reference_image, clamp_strength(req.strength)))

    pos_ref, neg_ref = [p + "pos", 0], [p + "neg", 0]
    vace_id = ""
    for i, (mask_node, reference, strength) in enumerate(blocks):
        vace_id = f"{p}vace{i}"
        inputs: dict = {
            "positive": pos_ref, "negative": neg_ref, "vae": [p + "vae", 0],
            "width": width, "height": height, "length": length, "batch_size": 1,
            "strength": strength,
            "control_video": ctrl_ref,
        }
        if mask_node:
            inputs["control_masks"] = [mask_node, 0]
        if reference:
            ref_node = f"{p}ref{i}"
            wf[ref_node] = {"class_type": "LoadImage", "inputs": {"image": reference}}
            inputs["reference_image"] = [ref_node, 0]
        wf[vace_id] = {"class_type": "WanVaceToVideo", "inputs": inputs}
        # Chaining threads the conditioning through, which is what makes the
        # blocks stack instead of replacing one another.
        pos_ref, neg_ref = [vace_id, 0], [vace_id, 1]

    # ── Sample ───────────────────────────────────────────────────────────────
    wf[p + "ks"] = {
        "class_type": "KSampler",
        "inputs": {
            "model": [p + "shift", 0],
            "positive": pos_ref, "negative": neg_ref,
            "latent_image": [vace_id, 2],
            "seed": seed, "steps": steps, "cfg": cfg,
            "sampler_name": SAMPLER, "scheduler": SCHEDULER, "denoise": 1.0,
        },
    }
    # The reference image rode along as extra latent frames; trim_latent says
    # how many, and it must come off before the decode or the clip opens on a
    # frame of the reference picture.
    wf[p + "trim"] = {
        "class_type": "TrimVideoLatent",
        "inputs": {"samples": [p + "ks", 0], "trim_amount": [vace_id, 3]},
    }
    wf[p + "dec"] = {
        "class_type": "VAEDecode",
        "inputs": {"samples": [p + "trim", 0], "vae": [p + "vae", 0]},
    }
    images_ref = [p + "dec", 0]

    # Grade back onto the control track. Measured 2026-08-19: repainting a warm,
    # muted stage-A render at strength 0.75 comes back in saturated cyan and
    # orange — the model's own prior, not anything in the prompt. Rewording the
    # prompt to ask for desaturation changed nothing at all, and it cannot: the
    # distilled recipes run at cfg 1.0, where there is no unconditional pass for
    # a negative to act through. So the palette is fixed where it is cheap to
    # fix, after the decode, against the picture that already had the right one.
    #
    # LAB because it separates lightness from chroma: the repaint's new light
    # and contrast survive, only the colour is pulled back.
    if req.color_match > 0:
        wf[p + "match"] = {
            "class_type": "ImageColorMatch+",
            "inputs": {
                "image": images_ref,
                "reference": [p + "fit", 0],
                "color_space": "LAB",
                "factor": round(float(req.color_match), 2),
                # CPU, and the whole clip in one go. Both are forced by how the
                # node works: it computes the reference statistics **per frame**
                # over the entire reference batch, then splits only the image
                # into `batch_size` chunks — so any chunk smaller than the clip
                # compares N frames of statistics against the reference's full
                # count and raises. 0 means "the whole batch" (image.shape[0]).
                # Chunking is therefore not available, and doing the arithmetic
                # on the GPU would spike VRAM right after sampling for what is
                # a per-pixel affine transform. It is cheap on the CPU.
                "device": "cpu",
                "batch_size": 0,
            },
        }
        images_ref = [p + "match", 0]

    wf[p + "out"] = {
        "class_type": "VHS_VideoCombine",
        "inputs": {
            "images": images_ref,
            "frame_rate": FPS, "loop_count": 0,
            "filename_prefix": req.filename_prefix,
            "format": "video/h264-mp4", "pix_fmt": "yuv420p", "crf": 18,
            "save_metadata": False, "trim_to_audio": False,
            "pingpong": False, "save_output": True,
        },
    }
    return wf, p + "out"


def sweep_workflows(req: VaceRequest, values=SWEEP_VALUES) -> list[tuple[float, dict, str]]:
    """The same request at several strengths, fixed seed — a contact sheet, not
    a table of numbers. Which band actually turns a cube into a world is the
    one thing here that cannot be reasoned out in advance."""
    seed = req.seed if req.seed >= 0 else random.randint(0, 2**32 - 1)
    out = []
    for value in values:
        variant = copy.deepcopy(req)
        variant.seed = seed
        variant.strength = value
        for region in variant.regions:
            region.strength = value
        variant.filename_prefix = f"{req.filename_prefix}_s{int(value * 100):03d}"
        wf, node = build_vace_workflow(variant)
        out.append((value, wf, node))
    return out


def quality_sweep_workflows(
    req: VaceRequest, recipes=("fast", "rich", "full"),
) -> list[tuple[str, dict, str]]:
    """The same request under several sampler recipes — one seed, one canvas.

    The canvas is what makes this comparable at all. Strength, prompt, control
    track and seed are already fixed by copying the request; changing the render
    size on top of that would change the scene outright and the sheet would
    compare two different pictures rather than two samplers.
    """
    seed = req.seed if req.seed >= 0 else random.randint(0, 2**32 - 1)
    out = []
    for recipe in recipes:
        variant = copy.deepcopy(req)
        variant.seed = seed
        variant.recipe = recipe
        variant.filename_prefix = f"{req.filename_prefix}_{recipe}"
        wf, node = build_vace_workflow(variant)
        out.append((recipe, wf, node))
    return out


def plan_region_passes(req: VaceRequest) -> list[VaceRequest]:
    """Lay out a multi-region job as one request per pass.

    Pass 0 takes the depth track with no mask and the first region's reference,
    which paints the whole frame. Every later pass repaints one region on top
    of the previous pass's render, so the caller must fill in
    `control_video` with the file the previous pass produced before building
    it — that path does not exist yet at planning time.

    Ordering matters and is the caller's to choose: the last region painted is
    the one that sees the most finished picture around it.
    """
    if not req.regions:
        return [copy.deepcopy(req)]
    if not req.mask_video:
        raise ValueError("regions need a mask_video to key them out of")

    passes: list[VaceRequest] = []
    first = copy.deepcopy(req)
    first.regions = []
    first.reference_image = req.regions[0].reference
    first.strength = clamp_strength(req.regions[0].strength)
    first.filename_prefix = f"{req.filename_prefix}_p0"
    passes.append(first)

    for i, region in enumerate(req.regions[1:], start=1):
        later = copy.deepcopy(req)
        later.regions = [region]
        later.reference_image = None
        # The control track is a finished render from here on: it is neither a
        # depth pass to invert nor footage to derive depth from.
        later.invert_depth = False
        later.derive_depth = False
        later.control_video = ""      # caller fills in the previous render
        later.filename_prefix = f"{req.filename_prefix}_p{i}"
        passes.append(later)
    return passes


def resolve_input(name_or_path: str) -> str:
    """Absolute path for a file that already lives in ComfyUI's input folder."""
    from core.config import settings

    candidate = Path(name_or_path)
    if candidate.is_absolute():
        return str(candidate)
    return str((settings.comfyui_output_dir.parent / "input" / name_or_path).resolve())
