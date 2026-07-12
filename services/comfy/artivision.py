"""
ArtiVision XL (SDXL) workflow builder — the second image-generator model
alongside Z-Image Turbo (services/comfy/zimage.py). The JSON template is
loaded once at import time; builders deep-copy it per submission.

SDXL degrades (duplicated subjects, broken composition) when sampled
directly at much more than its ~1024x1024 training area, so requests for a
larger final size are rendered at a native-area bucket first, then finished
with a hires-fix pass: ESRGAN pixel upscale (4x-UltraSharp) -> exact-size
resize -> VAE re-encode -> a second, low-denoise KSampler pass. Requests
already at/near native resolution skip that second pass entirely (nodes
8-13 are pruned and SaveImage reads straight off the base VAEDecode), since
there is nothing for a refinement pass to add at native res.
"""
import copy
import json
import math
import random
from pathlib import Path

_TEMPLATE = json.loads(
    (Path(__file__).resolve().parent.parent.parent / "workflows" / "artivision_xl.json")
    .read_text()
)

# SDXL's trained-native area (1024x1024). Targets within _HIRES_TRIGGER_RATIO
# of this render in a single pass; anything larger renders at a scaled-down
# native-area bucket and gets upscaled+refined back up to the exact target.
_NATIVE_PIXELS = 1024 * 1024
_HIRES_TRIGGER_RATIO = 1.15
_MIN_BASE_SIDE = 512

_DEFAULT_NEGATIVE = (
    "worst quality, low quality, blurry, jpeg artifacts, watermark, text, "
    "signature, deformed, disfigured, bad anatomy, extra limbs, extra fingers, "
    "mutated hands, cropped, out of frame, duplicate, cloned, multiple views, "
    "tiled, mosaic, collage, split frame"
)


def _round_to_64(x: float) -> int:
    return max(_MIN_BASE_SIDE, round(x / 64) * 64)


def _sdxl_base_size(width: int, height: int) -> tuple[int, int]:
    """Native-resolution render size for this target, before hires-fix."""
    if width * height <= _NATIVE_PIXELS * _HIRES_TRIGGER_RATIO:
        return width, height
    scale = math.sqrt(_NATIVE_PIXELS / (width * height))
    return _round_to_64(width * scale), _round_to_64(height * scale)


def build_artivision_workflow(
    prompt: str, negative_prompt: str, seed: int, width: int, height: int,
    lora_name: str, lora_strength: float, lora_strength_clip: float = 0.0,
) -> dict:
    wf = copy.deepcopy(_TEMPLATE)
    seed = seed if seed >= 0 else random.randint(0, 2**32 - 1)
    base_w, base_h = _sdxl_base_size(width, height)

    wf["3"]["inputs"]["text"] = prompt
    wf["4"]["inputs"]["text"] = negative_prompt.strip() or _DEFAULT_NEGATIVE
    wf["2"]["inputs"]["lora_name"] = lora_name
    wf["2"]["inputs"]["strength_model"] = round(max(0.0, min(1.0, lora_strength)), 3)
    wf["2"]["inputs"]["strength_clip"] = lora_strength_clip
    wf["5"]["inputs"]["width"] = base_w
    wf["5"]["inputs"]["height"] = base_h
    wf["6"]["inputs"]["seed"] = seed

    if (base_w, base_h) == (width, height):
        # Already native resolution — the hires-fix pass has nothing to add.
        for node_id in ("8", "9", "10", "11", "12", "13"):
            del wf[node_id]
        wf["14"]["inputs"]["images"] = ["7", 0]
    else:
        wf["10"]["inputs"]["width"] = width
        wf["10"]["inputs"]["height"] = height
        wf["12"]["inputs"]["seed"] = seed

    return wf
