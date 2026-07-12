"""
LoRA registry — Single Source of Truth for the image-generator LoRA pickers.

Backend uses ALLOWED_LORAS/SDXL_ALLOWED_LORAS (the sets of filenames) for
validation. Frontend fetches the matching list via GET /api/loras?model=...
and renders the buttons from the result, so adding a LoRA only requires
editing this file.

The two registries are separate because a LoRA's weights are tied to the
base model architecture it was trained against — a Z-Image Turbo LoRA
cannot be loaded onto the SDXL checkpoint (ArtiVision XL) or vice versa.
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
