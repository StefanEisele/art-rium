"""
Ernie Image Turbo workflow builder — the third image-generator model
alongside Z-Image Turbo and ArtiVision XL (SDXL). Structurally a clone of
Z-Image Turbo's own graph (services/comfy/zimage.py): one UNETLoader, one
CLIPLoader, one CLIPTextEncode whose conditioning is zeroed out for
"negative" (Ernie Image Turbo ignores negative prompts, same as Z-Image
Turbo), no LoRA node — there is no Ernie-compatible LoRA in the library yet.
The JSON template is loaded once at import time; builders deep-copy it per
submission.

Per the user: reuse the Z-Image Turbo prompt enhancer unmodified for this
model (services/ollama/zimage_enhance.py) rather than building a
model-specific one — already tested to work well, and Ernie Image Turbo's
own official ComfyUI template even ships a built-in LLM prompt-expansion
subgraph that we deliberately don't use here for the same reason.
"""
import copy
import json
import random
from pathlib import Path

_TEMPLATE = json.loads(
    (Path(__file__).resolve().parent.parent.parent / "workflows" / "ernie_image_turbo.json")
    .read_text()
)

# EmptyFlux2LatentImage stores latents at 1/16 resolution (vs. 1/8 for
# SD/SDXL), and requires width/height to be exact multiples of 16 — anything
# else gets floor-divided internally, silently shrinking the output. Round
# to the nearest valid size instead of failing or truncating.
_LATENT_DOWNSCALE = 16


def _round_to_16(x: int) -> int:
    return max(_LATENT_DOWNSCALE, round(x / _LATENT_DOWNSCALE) * _LATENT_DOWNSCALE)


def build_ernie_workflow(prompt: str, seed: int, width: int, height: int) -> dict:
    wf = copy.deepcopy(_TEMPLATE)
    wf["4"]["inputs"]["text"] = prompt
    wf["6"]["inputs"]["width"] = _round_to_16(width)
    wf["6"]["inputs"]["height"] = _round_to_16(height)
    wf["7"]["inputs"]["seed"] = seed if seed >= 0 else random.randint(0, 2**32 - 1)
    return wf
