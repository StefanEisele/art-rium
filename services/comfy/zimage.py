"""
Z-Image Turbo workflow builder — shared by the interactive generate router
(WebSocket ingestion path) and the video story-frames pipeline (poll_history
path). The JSON template is loaded once at import time; builders deep-copy it
per submission.
"""
import copy
import json
import random
from pathlib import Path

# SaveImage node id inside workflows/z-image_turbo.json — poll_history callers
# harvest the output PNG from outputs[ZIMAGE_SAVE_NODE]["images"][0].
ZIMAGE_SAVE_NODE = "9"

_TEMPLATE = json.loads(
    (Path(__file__).resolve().parent.parent.parent / "workflows" / "z-image_turbo.json")
    .read_text()
)


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

    model_ref = ["46", 0]  # UNETLoader output
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

    wf["47"]["inputs"]["model"] = model_ref
    return wf
