"""
Z-Image Turbo workflow builders — shared by the interactive generate router
(WebSocket ingestion path) and the video story-frames pipeline (poll_history
path). The JSON templates are loaded once at import time; builders deep-copy
them per submission.

Two graphs live here, and they are the same graph apart from where the latent
comes from:

  - `build_zimage_workflow`         EmptySD3LatentImage → sampler(denoise 1)
  - `build_zimage_variant_workflow` LoadImage → VAEEncode → sampler(denoise<1)

Everything else — model, text encoder, VAE, AuraFlow shift, sampler, step
count, the zeroed negative and the LoRA chain — is deliberately identical. A
distilled model steered differently is a different model, and a variant that
was sampled unlike its source would not read as a variant of it.

Both templates still carry a `KSampler`, and both builders swap it for the
Detail Daemon sampler chain on the way out — see `_apply_detail_sampler`. The
templates keep the simple node because it is the readable statement of what
the sampling *is*; the chain is the same thing decomposed so a wrapper can be
inserted, and at detail 0 it was measured to be pixel-identical.
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


# ── Detail Daemon: how much material the model invents ──────────────────────
#
# ComfyUI-Detail-Daemon (Jonseed) wraps the sampler and lowers the sigma handed
# to the model over a schedule, which makes it resolve more structure than the
# step count alone would. It is *not* a sharpen filter and it does not run
# after the fact — it changes what gets sampled.
#
# It needs a SAMPLER object, which a plain `KSampler` never produces, so the
# graph is submitted as the decomposition `KSampler` performs internally:
# RandomNoise + CFGGuider + BasicScheduler + KSamplerSelect →
# DetailDaemonSamplerNode → SamplerCustomAdvanced. That is the wiring the
# node's own Z-Image Turbo example uses.
#
# MEASURED, 2026-09-19, 1024x1024, 9 steps, res_multistep/simple, cfg 1, one
# fixed seed: the decomposition at detail 0 is **pixel-identical** to the
# KSampler it replaces (same SHA over the decoded pixels). That is why there is
# no second code path — 0 really is "as before", not "close to before".
#
# The sweep at ±: what the dial actually does on this model.
#   -1.0  flat. Paint texture and rust grain smoothed away; clean but lifeless.
#    0.0  the render this tool has always produced.
#   +0.5  real material detail — roller grain in paint, granular rust, chips.
#   +1.0  much more flaking, chip edges with undercoat showing. Still physical.
#   +2.0  the surface turns *liquid*: pour-like swirls, speckles, curls. A
#         different picture rather than a more detailed one — which happens to
#         suit this library's paint-pour look, so it stays reachable.
#   +3.0  clutter: invented foliage and spikes crowd the subject out.
#   +4.0  breakdown into blue/white shards; 2% of the frame clipped.
#
# Hence the range below. The cap is where the sweep stopped producing a picture
# of the thing that was asked for.
DETAIL_MIN = -1.0
DETAIL_MAX = 2.5
DETAIL_DEFAULT = 0.0
DETAIL_SWEET_MIN, DETAIL_SWEET_MAX = 0.4, 1.2

# The schedule shape around `detail_amount`, taken from the node author's
# Z-Image Turbo example workflow rather than from the node defaults — the
# defaults (start 0.2, end 0.8, exponent 1) are tuned for many-step SDXL/Flux
# runs, and this model gets nine. Exposed as one dial on purpose: the other
# eight inputs are schedule shaping, and a second row of sliders would not
# earn its space next to a nine-step sampler.
_DETAIL_SCHEDULE: dict = {
    "start": 0.0,
    "end": 1.0,
    "bias": 0.5,
    "exponent": 3.0,
    "start_offset": 0.0,
    "end_offset": 0.0,
    "fade": 0.0,
    "smooth": True,
    # 0 = let the wrapper read the CFG off the guider. At cfg 1 the scale
    # factor is 1.0, so the adjustment applies undiminished — the dial is not
    # inert on a distilled model, which is the first thing worth checking.
    "cfg_scale_override": 0.0,
}

# Node ids for the swapped-in sampler chain. Deliberately above everything in
# both templates (9, 39-49) so they cannot collide with a template node.
_DD_NOISE, _DD_GUIDER, _DD_SIGMAS = "50", "51", "52"
_DD_SELECT, _DD_DAEMON, _DD_SAMPLER = "53", "54", "55"

_KSAMPLER_NODE = "44"     # the node the chain replaces, in both templates
_DECODE_NODE = "43"       # VAEDecode — has to be re-pointed at the new sampler

_DETAIL_BANDS: tuple[tuple[float, str, str], ...] = (
    (-0.25, "Geglättet",
     "Textur wird zurückgenommen — Flächen werden ruhiger, Korn und Rost verschwinden."),
    (0.25, "Wie gehabt",
     "Der Sampler läuft unverändert. Bei 0 exakt das Bild, das dieses Tool immer geliefert hat."),
    (1.30, "Material",
     "Mehr Oberfläche: Farbkorn, Rostkörnung, Abplatzer mit sichtbaren Kanten. "
     "Motiv und Komposition bleiben, wie sie sind."),
    (2.51, "Erfunden",
     "Das Modell setzt eigene Struktur dazu — Farbe wird flüssig, Schlieren und "
     "Sprenkel kommen hinzu. Ein anderes Bild, nicht nur ein detaillierteres."),
)


def clamp_detail(value: float | None) -> float:
    """Pull a detail request into the range that still returns the picture
    that was asked for. `None` means the dial was not touched."""
    if value is None:
        return DETAIL_DEFAULT
    return round(min(max(float(value), DETAIL_MIN), DETAIL_MAX), 2)


def describe_detail(value: float | None) -> dict:
    """Band name + one-line effect for a detail value."""
    detail = clamp_detail(value)
    for upper, name, effect in _DETAIL_BANDS:
        if detail < upper:
            break
    return {
        "detail": detail,
        "band": name,
        "effect": effect,
        "recommended": DETAIL_SWEET_MIN <= detail <= DETAIL_SWEET_MAX,
    }


def detail_bands() -> list[dict]:
    """The full band table, for a client that wants to label its own slider."""
    lower = DETAIL_MIN
    bands = []
    for upper, name, effect in _DETAIL_BANDS:
        bands.append({
            "from": round(lower, 2),
            "to": round(min(upper, DETAIL_MAX), 2),
            "band": name,
            "effect": effect,
        })
        lower = upper
        if lower >= DETAIL_MAX:
            break
    return bands


def _apply_detail_sampler(wf: dict, detail: float) -> None:
    """Replace the template's KSampler with the Detail Daemon sampler chain.

    Unconditional, including at detail 0, because the decomposition was
    measured to be pixel-identical there — one graph is easier to reason about
    than two, and it keeps "same seed, dial moved" an honest comparison
    instead of one that also swapped samplers.

    Everything the KSampler was configured with is carried across rather than
    restated, so the sampler, scheduler, cfg, steps and denoise stay the
    template's business and this function only ever adds the wrapper.
    """
    ks = wf.pop(_KSAMPLER_NODE)["inputs"]

    wf[_DD_NOISE] = {
        "inputs": {"noise_seed": ks["seed"]},
        "class_type": "RandomNoise",
        "_meta": {"title": "Noise"},
    }
    wf[_DD_GUIDER] = {
        "inputs": {
            "model": ks["model"],
            "positive": ks["positive"],
            "negative": ks["negative"],
            "cfg": ks["cfg"],
        },
        "class_type": "CFGGuider",
        "_meta": {"title": "Guider"},
    }
    wf[_DD_SIGMAS] = {
        "inputs": {
            "model": ks["model"],
            "scheduler": ks["scheduler"],
            "steps": ks["steps"],
            # BasicScheduler does the same `int(steps/denoise)` then
            # `sigmas[-(steps+1):]` that KSampler.set_steps does, so a variant
            # still samples its full step count from a later starting point —
            # the long note at the top of this module still holds.
            "denoise": ks["denoise"],
        },
        "class_type": "BasicScheduler",
        "_meta": {"title": "Sigmas"},
    }
    wf[_DD_SELECT] = {
        "inputs": {"sampler_name": ks["sampler_name"]},
        "class_type": "KSamplerSelect",
        "_meta": {"title": "Sampler"},
    }
    wf[_DD_DAEMON] = {
        "inputs": {
            "sampler": [_DD_SELECT, 0],
            "detail_amount": detail,
            **_DETAIL_SCHEDULE,
        },
        "class_type": "DetailDaemonSamplerNode",
        "_meta": {"title": "Detail Daemon"},
    }
    wf[_DD_SAMPLER] = {
        "inputs": {
            "noise": [_DD_NOISE, 0],
            "guider": [_DD_GUIDER, 0],
            "sampler": [_DD_DAEMON, 0],
            "sigmas": [_DD_SIGMAS, 0],
            "latent_image": ks["latent_image"],
        },
        "class_type": "SamplerCustomAdvanced",
        "_meta": {"title": "Sampler (Detail Daemon)"},
    }
    # Slot 0 is `output`, which is what KSampler returned; slot 1 is
    # `denoised_output` and would be a different picture.
    wf[_DECODE_NODE]["inputs"]["samples"] = [_DD_SAMPLER, 0]


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
    loras: list[dict], detail: float | None = None,
) -> dict:
    """
    *loras*: [{"name": "<filename>.safetensors", "strength": 0.0-1.0}, ...],
    applied as a chain in list order (each LoraLoaderModelOnly feeds the
    next). An empty list wires the UNETLoader straight into
    ModelSamplingAuraFlow with no LoRA at all.

    *detail*: the Detail Daemon dial, `None` or 0 for the sampler as it was.
    """
    wf = copy.deepcopy(_TEMPLATE)
    wf["45"]["inputs"]["text"] = prompt
    wf["44"]["inputs"]["seed"] = seed if seed >= 0 else random.randint(0, 2**32 - 1)
    wf["41"]["inputs"]["width"] = width
    wf["41"]["inputs"]["height"] = height
    _apply_lora_chain(wf, loras)
    _apply_detail_sampler(wf, clamp_detail(detail))
    return wf


def build_zimage_variant_workflow(
    image_name: str, prompt: str, seed: int, denoise: float,
    loras: list[dict], detail: float | None = None,
) -> dict:
    """Re-diffuse an existing picture into a variant of itself.

    *image_name* is the file as ComfyUI's /upload/image named it; it is encoded
    to a latent rather than starting from noise, so the output keeps the
    source's size (see `latent_size`) and there is no width/height to pass.

    *prompt* is the conditioning the variant is painted towards — the source
    image's own prompt when the caller wants a faithful variant, an edited one
    when they want the picture pushed somewhere. *denoise* decides how much of
    the source survives to be painted over; see the module constants.

    *loras* and *detail* behave exactly as in `build_zimage_workflow`.
    """
    wf = copy.deepcopy(_VARIANT_TEMPLATE)
    wf["49"]["inputs"]["image"] = image_name
    wf["45"]["inputs"]["text"] = prompt
    wf["44"]["inputs"]["seed"] = seed if seed >= 0 else random.randint(0, 2**32 - 1)
    wf["44"]["inputs"]["denoise"] = clamp_denoise(denoise)
    _apply_lora_chain(wf, loras)
    _apply_detail_sampler(wf, clamp_detail(detail))
    return wf
