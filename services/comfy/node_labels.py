"""ComfyUI node → human-readable stage label.

ComfyUI's `executing` / `progress` WebSocket events identify the current node
by *id* only. Ids are workflow-local (Z-Image's "44" is a KSampler, MiniMax's
"mmx_sampler" is a SamplerCustomAdvanced), so a frontend cannot name a stage
from the id alone — which is why the old hard-coded id→label map in shared.js
only ever read correctly for the Z-Image workflow.

The submitting side always knows the workflow dict, so the mapping is built
there instead: `build_label_map(workflow)` turns it into id → label once at
submit time, and workers/comfy_listener.py hands the label out with every
progress event. Every workflow in the project is covered automatically,
including ones added later — an unknown class_type degrades to a humanised
version of its own name rather than to "Node 47".
"""
from __future__ import annotations

import re

# Curated labels, keyed by ComfyUI class_type. Phrased as the stage a user is
# waiting through ("Loading model…"), not as the node's name, because this is
# what the progress line reads out loud while a render is running.
_CLASS_LABELS: dict[str, str] = {
    # ── Model loading ────────────────────────────────────────────────────────
    "CheckpointLoaderSimple":  "Loading model…",
    "UNETLoader":              "Loading model…",
    "CLIPLoader":              "Loading text encoder…",
    "DualCLIPLoader":          "Loading text encoders…",
    "VAELoader":               "Loading VAE…",
    "LoraLoader":              "Loading LoRA…",
    "LoraLoaderModelOnly":     "Loading LoRA…",
    "UpscaleModelLoader":      "Loading upscale model…",
    "SeedVR2LoadDiTModel":     "Loading upscaler…",
    "SeedVR2LoadVAEModel":     "Loading upscaler VAE…",
    # ── Conditioning ─────────────────────────────────────────────────────────
    "CLIPTextEncode":            "Encoding prompt…",
    "TextEncodeAceStepAudio1.5": "Encoding lyrics…",
    "ConditioningZeroOut":       "Zeroing conditioning…",
    "BasicGuider":               "Preparing guidance…",
    "BasicScheduler":            "Building sigma schedule…",
    "KSamplerSelect":            "Selecting sampler…",
    "RandomNoise":               "Seeding noise…",
    "ModelSamplingSD3":          "Configuring sampler…",
    "ModelSamplingAuraFlow":     "Configuring sampler…",
    "PathchSageAttentionKJ":     "Switching to fast attention…",
    # ── Latents ──────────────────────────────────────────────────────────────
    "EmptyLatentImage":          "Preparing latent…",
    "EmptySD3LatentImage":       "Preparing latent…",
    "EmptyFlux2LatentImage":     "Preparing latent…",
    "EmptyAceStep1.5LatentAudio": "Preparing audio latent…",
    "WanImageToVideo":           "Preparing video latent…",
    "WanFirstLastFrameToVideo":  "Preparing transition latent…",
    "MiniMaxH3ImageToVideo":     "Preparing video + audio latent…",
    "VAEEncode":                 "Encoding image to latent…",
    # ── Sampling (the long one) ──────────────────────────────────────────────
    "KSampler":               "Sampling…",
    "KSamplerAdvanced":       "Sampling…",
    "SamplerCustomAdvanced":  "Sampling…",
    "SeedVR2VideoUpscaler":   "Restoring detail…",
    # ── Decode / post ────────────────────────────────────────────────────────
    "VAEDecode":            "Decoding picture…",
    "VAEDecodeAudio":       "Decoding audio…",
    "ImageUpscaleWithModel": "Upscaling…",
    "ImageScale":           "Resizing…",
    "ImageBatch":           "Batching frames…",
    "RIFE VFI":             "Interpolating frames…",
    # ── Input / output ───────────────────────────────────────────────────────
    "LoadImage":           "Loading image…",
    "VHS_LoadVideoPath":   "Reading source video…",
    "VHS_VideoInfoSource": "Reading video info…",
    "SaveImage":           "Saving image…",
    "SaveAudioMP3":        "Saving audio…",
    "VHS_VideoCombine":    "Encoding video…",
}

# "KSamplerAdvanced" → "K Sampler Advanced"; "VHS_LoadVideoPath" → "Load Video Path".
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def label_for_class(class_type: str) -> str:
    """Stage label for a ComfyUI class_type, humanised when uncurated."""
    known = _CLASS_LABELS.get(class_type)
    if known:
        return known
    name = class_type.split("_", 1)[-1] if "_" in class_type else class_type
    return _CAMEL_BOUNDARY.sub(" ", name).replace("_", " ").strip() + "…"


def build_label_map(workflow: dict) -> dict[str, str]:
    """node_id → stage label for one API-format workflow, ready to hand to a client."""
    return {
        node_id: label_for_class(node.get("class_type", ""))
        for node_id, node in workflow.items()
        if isinstance(node, dict)
    }
