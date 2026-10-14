"""
LoRA registry — Single Source of Truth for the image-generator LoRA pickers.

Backend uses ALLOWED_LORAS/SDXL_ALLOWED_LORAS (the sets of filenames) for
validation. Frontend fetches the matching list via GET /api/loras?model=...
and renders the buttons from the result, so adding a LoRA only requires
editing this file.

The registries are separate because a LoRA's weights are tied to the
base model architecture it was trained against — a Z-Image Turbo LoRA
cannot be loaded onto the SDXL checkpoint (ArtiVision XL) or vice versa.
The SD 1.5 one serves the Struktur tool's AnimateLCM graph (routers/vace.py).
"""
from typing import TypedDict


class LoraSpec(TypedDict):
    filename: str   # actual .safetensors filename inside ComfyUI's loras directory
    label: str      # short button label
    trigger: str    # trigger word inserted at the start of an empty prompt


LORAS: list[LoraSpec] = [
    {
        "filename": "rustorangeanddimblue_lora_copy.safetensors",
        "label":    "Rust & Blue",
        "trigger":  "rustorangeanddimblue",
    },
    {
        "filename": "art_vision_ZIG.safetensors",
        "label":    "Art Vision",
        "trigger":  "art_vision",
    },
    {
        "filename": "zImageT_zidiusArt_melancholy.safetensors",
        "label":    "Melancholy",
        "trigger":  "zidiusArt",
    },
    {
        "filename": "art_vision_v2_epoch_10.safetensors",
        "label":    "Art Vision v2",
        "trigger":  "art_vision",
    },
    {
        # Trained 2026-10-06 on Civitai from 46 curated art-rium outputs
        # (storage/lora_datasets/artvis3_dataset.zip). Checkpoint 7 of 10: the
        # strongest look that still follows the prompt — 8–10 collapse every
        # subject into "tiny figure in empty space".
        "filename": "artvision_v3_epoch7.safetensors",
        "label":    "Art Vision v3",
        "trigger":  "artvis3",
    },
    {
        # The same run one checkpoint earlier: keeps the prompted subject
        # intact (the lighthouse stays a lighthouse) with a lighter touch.
        "filename": "artvision_v3_epoch6.safetensors",
        "label":    "Art Vision v3 · treu",
        "trigger":  "artvis3",
    },
]

ALLOWED_LORAS: set[str] = {lora["filename"] for lora in LORAS}
DEFAULT_LORA: str = LORAS[0]["filename"]


class SdxlLoraSpec(TypedDict):
    filename: str        # actual .safetensors filename inside ComfyUI's loras directory
    label: str            # short button label
    trigger: str          # trigger word inserted at the start of an empty prompt (may be empty)
    strength_clip: float  # this LoRA's tuned CLIP strength (SDXL LoRAs often want CLIP untouched)


SDXL_LORAS: list[SdxlLoraSpec] = [
    {
        "filename":      "xl_more_art-full_v1.safetensors",
        "label":         "More Art (XL)",
        "trigger":       "",
        "strength_clip": 0.0,
    },
]

SDXL_ALLOWED_LORAS: set[str] = {lora["filename"] for lora in SDXL_LORAS}
SDXL_DEFAULT_LORA: str = SDXL_LORAS[0]["filename"]


class Sd15StyleLoraSpec(TypedDict):
    filename: str     # actual .safetensors filename inside ComfyUI's loras directory
    label: str        # short label for the picker
    strength: float   # measured default; applied to model and text encoder alike


# Picture-style LoRAs for the AnimateLCM graph (SD 1.5, Juggernaut Reborn).
# Not the distillation LoRAs in the same folder (AnimateLCM, LCM, Hyper-SD):
# those change how the sampler converges, not what the picture is made of.
SD15_STYLE_LORAS: list[Sd15StyleLoraSpec] = [
    {
        # Trained on the 46-picture artvision v3 set. 0.8 measured 2026-10-10
        # (scripts/style_channel_sweep.py): 1.0 starts to break the background
        # into a marbled mosaic late in the clip; the trigger word changes
        # nothing measurable, so it is not prepended.
        "filename": "artvis3_sd15.safetensors",
        "label":    "Art Vision v3",
        "strength": 0.8,
    },
]

SD15_ALLOWED_STYLE_LORAS: set[str] = {lora["filename"] for lora in SD15_STYLE_LORAS}
SD15_DEFAULT_STYLE_LORA: str = SD15_STYLE_LORAS[0]["filename"]
