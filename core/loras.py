"""
LoRA registry — Single Source of Truth for the Z-Image Turbo LoRA picker.

Backend uses ALLOWED_LORAS (the set of filenames) for validation.
Frontend fetches the same list via GET /api/loras and renders the buttons
from the result, so adding a LoRA only requires editing this file.
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

ALLOWED_LORAS: set[str] = {l["filename"] for l in LORAS}
DEFAULT_LORA: str = LORAS[0]["filename"]
