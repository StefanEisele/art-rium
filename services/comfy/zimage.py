"""
Z-Image Turbo workflow builders — shared by the interactive generate router
(WebSocket ingestion path) and the video story-frames pipeline (poll_history
path). The JSON templates are loaded once at import time; builders deep-copy
them per submission.

Two graphs live here, and they are the same graph apart from where the latent
comes from:

  - `build_zimage_workflow`         EmptySD3LatentImage → KSampler(denoise 1)
  - `build_zimage_variant_workflow` LoadImage → VAEEncode → KSampler(denoise<1)

Everything else — model, text encoder, VAE, AuraFlow shift, sampler, step
count, the zeroed negative and the LoRA chain — is deliberately identical. A
distilled model steered differently is a different model, and a variant that
was sampled unlike its source would not read as a variant of it.
"""
import copy
import json
import random
from pathlib import Path

# SaveImage node id inside both workflow templates — poll_history callers
# harvest the output PNG from outputs[ZIMAGE_SAVE_NODE]["images"][0].
ZIMAGE_SAVE_NODE = "9"

# The node the model chain has to end at, in both templates.
_MODEL_SINK = "47"        # ModelSamplingAuraFlow
_UNET_NODE = "46"         # UNETLoader — the head of the chain

_WORKFLOW_DIR = Path(__file__).resolve().parent.parent.parent / "workflows"

_TEMPLATE = json.loads((_WORKFLOW_DIR / "z-image_turbo.json").read_text())
_VARIANT_TEMPLATE = json.loads((_WORKFLOW_DIR / "z-image_turbo_variant.json").read_text())


# ── Variant denoise: the whole control surface ───────────────────────────────
#
# `steps` does not need touching. ComfyUI's `KSampler.set_steps`
# (comfy/samplers.py) does `new_steps = int(steps/denoise); sigmas[-(steps+1):]`,
# so a variant always samples the full 9 steps no matter how low the denoise
# is — denoise only picks how far down the schedule sampling starts. Raising
# steps to "make up for" a low denoise would only take the distilled model
# outside what it was distilled for.
#
# What denoise costs is signal, and `ModelSamplingAuraFlow` with shift 3 makes
# it cost more than the number suggests. The shift maps the schedule position
# through `time_snr_shift` (comfy/model_sampling.py) as `sigma = 3d / (1 + 2d)`,
# and in flow matching `x_t = (1 - sigma)·x_0 + sigma·noise`, so `1 - sigma` is
# the fraction of the source still in the latent when sampling starts:
# denoise 0.40 leaves 33%, denoise 0.70 leaves 12%.
#
# That maths alone would put the useful band around 0.30–0.50. A measured sweep
# says otherwise, and the sweep wins.
#
# ── What the sweep actually showed ───────────────────────────────────────────
# Six renders off one 1080x1920 source at a fixed seed (0.15 / 0.25 / 0.35 /
# 0.45 / 0.55 / 0.70), plus the same seed with no source at all as a control.
# The control came back a completely different picture, so the source was doing
# real work at every level below it. But the fall-off is far gentler than the
# sigma numbers imply: up to ~0.55 what moves is texture, skin, material and
# light, while composition, framing and subject hold almost exactly. Pose and
# proportion only start moving around 0.70.
#
# The reason is that at cfg 1 the prompt is the only steering there is, and a
# variant is normally run with the *source's own prompt* — conditioning and
# latent pull towards the same picture, so they reinforce instead of fighting.
#
# Which is why the recommendation depends on what the user did with the prompt,
# and a second sweep measured that directly. Same source, one deliberate edit
# (flat grey studio wall → "tall sunlit window wall throwing hard golden
# light"):
#
#     denoise 0.35 → the edit does not appear at all. Grey wall, no light.
#     denoise 0.55 → a hint of brightening, no window, no golden light.
#     denoise 0.75 → the edit lands: golden light across column and floor,
#                    and the composition still holds.
#
# So below ~0.6 the latent simply overrules a changed prompt and hands back a
# re-textured copy of the source. Two bands, not one.
DENOISE_MIN = 0.10        # below this the render comes back as its own source
DENOISE_MAX = 0.90        # past this, use Direct mode — the source stops mattering

# Prompt kept: variation of this picture. Prompt edited: enough noise for the
# edit to survive the source. Both measured, see above.
KEPT_SWEET_MIN, KEPT_SWEET_MAX = 0.30, 0.55
EDITED_SWEET_MIN, EDITED_SWEET_MAX = 0.65, 0.80
DENOISE_DEFAULT = 0.40
DENOISE_DEFAULT_EDITED = 0.72

# (upper bound, label, what it does) — the slider reads the matching entry out
# under the handle, so the number never stands on its own.
_DENOISE_BANDS: tuple[tuple[float, str, str], ...] = (
    (0.30, "Feinschliff",
     "Dasselbe Bild, neu ausgemalt — Textur und Detail ändern sich, sonst nichts."),
    (0.55, "Variante",
     "Haut, Material, Ausdruck und Licht verschieben sich; Komposition und Motiv bleiben. "
     "Ein geänderter Prompt schlägt hier noch nicht durch."),
    (0.80, "Neuinterpretation",
     "Pose, Proportionen und Hintergrund bewegen sich mit — und ein geänderter Prompt "
     "greift ab hier wirklich."),
    (1.01, "Frei",
     "Die Quelle wirkt nur noch als grobes Gerüst. Für ein wirklich neues Bild ist "
     "Direct der ehrlichere Weg."),
)


def clamp_denoise(value: float | None, *, prompt_changed: bool = False) -> float:
    """Pull a denoise request into the range where the source still counts.

    `None` falls back to the default for what the user did with the prompt —
    an edited prompt needs more noise than a kept one before it shows up.
    """
    if value is None:
        return DENOISE_DEFAULT_EDITED if prompt_changed else DENOISE_DEFAULT
    return round(min(max(float(value), DENOISE_MIN), DENOISE_MAX), 2)


def recommended_range(*, prompt_changed: bool = False) -> tuple[float, float]:
    """The band to point the user at, given what they did with the prompt."""
    if prompt_changed:
        return EDITED_SWEET_MIN, EDITED_SWEET_MAX
    return KEPT_SWEET_MIN, KEPT_SWEET_MAX


def describe_denoise(value: float, *, prompt_changed: bool = False) -> dict:
    """Band name + one-line effect for a denoise value, plus the surviving
    fraction of the source after the AuraFlow shift — the honest number behind
    the slider position."""
    denoise = clamp_denoise(value, prompt_changed=prompt_changed)
    for upper, name, effect in _DENOISE_BANDS:
        if denoise < upper:
            break
    low, high = recommended_range(prompt_changed=prompt_changed)
    sigma = 3 * denoise / (1 + 2 * denoise)
    return {
        "denoise": denoise,
        "band": name,
        "effect": effect,
        "source_retained": round(1 - sigma, 2),
        "recommended": low <= denoise <= high,
    }


def denoise_bands() -> list[dict]:
    """The full band table, for a client that wants to label its own slider."""
    lower = DENOISE_MIN
    bands = []
    for upper, name, effect in _DENOISE_BANDS:
        bands.append({
            "from": round(lower, 2),
            "to": round(min(upper, DENOISE_MAX), 2),
            "band": name,
            "effect": effect,
        })
        lower = upper
        if lower >= DENOISE_MAX:
            break
    return bands


def latent_size(width: int, height: int) -> tuple[int, int]:
    """The size a source image actually comes back at.

    `VAEEncode` center-crops the picture to a multiple of the VAE's spatial
    compression (8) before encoding — `VAE.vae_encode_crop_pixels` in
    comfy/sd.py. A 1620x2880 upscale therefore returns 1616x2880, and the DB
    row has to say so rather than repeating the source's own dimensions.
    """
    return max(8, width - width % 8), max(8, height - height % 8)


def _apply_lora_chain(wf: dict, loras: list[dict]) -> None:
    """Wire *loras* between the UNETLoader and ModelSamplingAuraFlow, in list
    order — each LoraLoaderModelOnly feeds the next. An empty list leaves the
    UNETLoader wired straight through with no LoRA at all."""
    model_ref = [_UNET_NODE, 0]
    for i, lora in enumerate(loras):
        node_id = f"lora_{i}"
        wf[node_id] = {
            "inputs": {
                "lora_name": lora["name"],
                "strength_model": round(max(0.0, min(1.0, float(lora["strength"]))), 3),
                "model": model_ref,
            },
            "class_type": "LoraLoaderModelOnly",
            "_meta": {"title": f"Load LoRA {i + 1}"},
        }
        model_ref = [node_id, 0]
    wf[_MODEL_SINK]["inputs"]["model"] = model_ref


def build_zimage_workflow(
    prompt: str, seed: int, width: int, height: int,
    loras: list[dict],
) -> dict:
    """
    *loras*: [{"name": "<filename>.safetensors", "strength": 0.0-1.0}, ...],
    applied as a chain in list order (each LoraLoaderModelOnly feeds the
    next). An empty list wires the UNETLoader straight into
    ModelSamplingAuraFlow with no LoRA at all.
    """
    wf = copy.deepcopy(_TEMPLATE)
    wf["45"]["inputs"]["text"] = prompt
    wf["44"]["inputs"]["seed"] = seed if seed >= 0 else random.randint(0, 2**32 - 1)
    wf["41"]["inputs"]["width"] = width
    wf["41"]["inputs"]["height"] = height
    _apply_lora_chain(wf, loras)
    return wf


def build_zimage_variant_workflow(
    image_name: str, prompt: str, seed: int, denoise: float,
    loras: list[dict],
) -> dict:
    """Re-diffuse an existing picture into a variant of itself.

    *image_name* is the file as ComfyUI's /upload/image named it; it is encoded
    to a latent rather than starting from noise, so the output keeps the
    source's size (see `latent_size`) and there is no width/height to pass.

    *prompt* is the conditioning the variant is painted towards — the source
    image's own prompt when the caller wants a faithful variant, an edited one
    when they want the picture pushed somewhere. *denoise* decides how much of
    the source survives to be painted over; see the module constants.

    *loras* behaves exactly as in `build_zimage_workflow`.
    """
    wf = copy.deepcopy(_VARIANT_TEMPLATE)
    wf["49"]["inputs"]["image"] = image_name
    wf["45"]["inputs"]["text"] = prompt
    wf["44"]["inputs"]["seed"] = seed if seed >= 0 else random.randint(0, 2**32 - 1)
    wf["44"]["inputs"]["denoise"] = clamp_denoise(denoise)
    _apply_lora_chain(wf, loras)
    return wf
