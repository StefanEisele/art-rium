"""SEEDVR2 upscale pass — buys resolution back after generating small.

MiniMax H3 costs close to linearly in pixels: the same clip renders in 6.2 min
at 864×480 where 1344×768 needs 16.1 (measured, same seed and length). So the
video workflow generates on a small canvas and this pass restores the pixels
afterwards, on a finished clip, on demand — the same post-hoc shape as
services/video/grain.py, except the work happens in ComfyUI rather than ffmpeg.

SEEDVR2 is a diffusion *restorer*, not a resampler: it reconstructs detail
rather than interpolating it, and it sees several frames at once so the
reconstruction is consistent over time. Measured against Lanczos at the same
target size, fine structure comes back reconstructed instead of smeared.

This module only *builds* the workflow. Submitting and polling stays in
routers/video.py with the rest of the ComfyUI orchestration, matching
services/comfy/zimage.py's split.

Measured on the 16 GB RTX 4060 Ti, 56 frames of 864×480 → 1944×1080:
    batch 1, no VAE tiling ................ 7.2 min
    batch 5 + VAE tiling + overlap 2 ...... 7.3 min, 6.9 GB peak
    batch 5, no VAE tiling ................ OOM in the VAE
Equal speed, so the tiled batch-5 path is the one worth having: batch 1 is
per-frame restoration with no temporal context at all.

The RTX 2070 cannot run this at all — the VAE's InflatedCausalConv3d runs
bfloat16 internally whatever the checkpoint dtype, and sm_75 has no bf16
conv3d kernel (`GET was unable to find an engine to execute this
computation`). Neither loader exposes a dtype switch, so there is no
second-GPU variant of this pass to fall back on.
"""
from __future__ import annotations

from pathlib import Path

# 3B rather than 7B: the 7B fp16 is 15.35 GB and leaves nothing for
# activations on a 16 GB card. fp8_e4m3fn is the smallest 3B variant that
# needs no extra runtime (the Q4/Q8 GGUFs would need the `gguf` module,
# which is absent from the ComfyUI venv).
_DIT_MODEL = "seedvr2_ema_3b_fp8_e4m3fn.safetensors"
_VAE_MODEL = "ema_vae_fp16.safetensors"

# sdpa is the only backend installed — ComfyUI logs
# `SageAttention x | Flash Attention x | Triton x` at startup, so the
# measured times above are the slow fallback path, not this pass's best case.
_ATTENTION_MODE = "sdpa"

# Restoration shifts colour slightly; `lab` grades the output back onto the
# input's palette and is the node author's default.
_COLOR_CORRECTION = "lab"

# Frames restored together. The node requires 4n+1, and higher means more
# temporal context. 5 is the largest that fits alongside a 1944×1080 decode.
_BATCH_SIZE = 5

# Frames shared between consecutive batches, blended across the seam. Without
# it every 5th frame boundary is a potential consistency break.
_TEMPORAL_OVERLAP = 2

# VAE tiling is what makes _BATCH_SIZE = 5 fit: 10.68 GiB were already
# allocated when the untiled decode asked for another 2.53 GiB. 512px tiles
# with 64px overlap bring the peak to 6.9 GB at no measurable time cost.
_TILE_SIZE = 512
_TILE_OVERLAP = 64

# Fixed rather than random: this pass restores an existing picture, it does
# not invent a new one, so a caller re-running it wants the same result.
_SEED = 42

# RIFE frame interpolation, optionally run *after* the restoration rather than
# during generation. Ordering it here is what decouples the upscale's cost from
# the interpolation factor: SEEDVR2 then only ever sees the model's real
# frames, so a 3x clip costs the same to restore as a 1x one.
#
# Measured on the 4060 Ti, 124 frames at 1088x1920 (2.09 MPx, the same pixel
# count as a 1944x1080 target), RIFE 3x end to end including model load and the
# h264 encode of 372 frames: 50 s, i.e. 202 ms per interpolated frame against
# SEEDVR2's 7.8 s per restored frame. Two orders of magnitude apart, which is
# why this ordering is a near-free win rather than a trade.
_RIFE_CKPT = "rife49.pth"
RIFE_MULTIPLIERS = (1, 2, 3, 4)  # 1 = off

SAVE_NODE = "sv_save"

# Target short edge. 1080 turns MiniMax's 864×480 into 1944×1080 and its
# 480×864 into 1080×1944; 720 is the cheaper half-step.
RESOLUTION_MIN = 480
RESOLUTION_MAX = 1440
RESOLUTION_DEFAULT = 1080

# Cost model for the ETA shown before a user commits to the render, anchored
# on the measurement above: 438 s for 56 frames of 1944×1080 output.
#   438 / (56 * 1944 * 1080) = 3.73e-6 s per output pixel-frame
# Expressed per *source second* (all art-rium videos play at ~24 fps, and a
# RIFE pass lengthens duration at a fixed frame rate, so duration already
# carries the interpolated frame count):
#   3.73e-6 * 24 = 8.94e-5
_SECONDS_PER_OUTPUT_PIXEL_SECOND = 8.94e-5

# Same shape for the RIFE stage, anchored on 50 s for 248 interpolated frames
# at 2.09 MPx: 0.202 / 2.09e6 = 9.66e-8 per interpolated pixel-frame, times
# 24 fps = 2.32e-6 per source second, per *extra* multiplier step. About 2.6%
# of the restoration's rate — visible in an estimate, never dominant.
_RIFE_SECONDS_PER_OUTPUT_PIXEL_SECOND = 2.32e-6


def clamp_rife(value: int | float | None) -> int:
    """Interpolation factor onto the supported set. None/garbage reads as off."""
    try:
        v = int(round(float(value)))
    except (TypeError, ValueError):
        return 1
    return v if v in RIFE_MULTIPLIERS else 1


def clamp_resolution(value: int | float | None) -> int:
    """Target short edge onto the supported range. None/garbage reads as the default."""
    try:
        return max(RESOLUTION_MIN, min(RESOLUTION_MAX, int(round(float(value)))))
    except (TypeError, ValueError):
        return RESOLUTION_DEFAULT


def output_dimensions(width: int, height: int, resolution: int) -> tuple[int, int]:
    """What the pass will produce for a `width`×`height` source.

    Mirrors the node: the shorter edge becomes `resolution`, the longer one
    follows the source aspect ratio, both rounded to even values. Sources
    already at or above the target are left alone — SEEDVR2 would happily
    *downscale*, which is never what this pass is for.
    """
    w, h = max(1, int(width)), max(1, int(height))
    short = min(w, h)
    if short >= resolution:
        return w, h
    scale = resolution / short
    return (_even(w * scale), _even(h * scale))


def _even(value: float) -> int:
    return max(2, int(round(value / 2)) * 2)


def estimate_seconds(
    duration: float,
    out_width: int,
    out_height: int,
    rife_multiplier: int = 1,
) -> int:
    """Rough wall-clock estimate for the render, in seconds.

    Deliberately surfaced to the user before they start: this pass runs for
    minutes per second of footage, which is worth knowing in advance rather
    than discovering from a progress bar. Accurate to roughly ±25% — the
    frame rate is assumed, not probed.

    `duration` is the *source* length. Interpolation does not change what the
    restoration has to do, only what RIFE adds afterwards, so raising the
    multiplier barely moves this number — which is the whole point of running
    RIFE after the upscale instead of before it.
    """
    if duration <= 0:
        return 0
    pixel_seconds = duration * out_width * out_height
    cost = pixel_seconds * _SECONDS_PER_OUTPUT_PIXEL_SECOND
    extra_steps = max(0, clamp_rife(rife_multiplier) - 1)
    cost += pixel_seconds * extra_steps * _RIFE_SECONDS_PER_OUTPUT_PIXEL_SECOND
    return max(30, int(round(cost)))


def build_upscale_workflow(
    src: Path,
    *,
    resolution: int,
    filename_prefix: str,
    has_audio: bool,
    rife_multiplier: int = 1,
) -> tuple[dict, str]:
    """SEEDVR2 restoration of a finished mp4. Returns (workflow, save_node_id).

    Reads the source straight off disk by absolute path rather than uploading
    it — ComfyUI runs on this machine and the rest of the video pipeline
    already reads its output directory directly.

    `has_audio` must reflect the file, not the workflow that made it: VHS
    raises when asked to extract audio from a silent input, so a silent clip
    has to leave the muxer's audio slot unconnected instead of passing an
    empty track. The audio itself never passes through the model — it goes
    from the loader to the muxer directly.

    `rife_multiplier` > 1 interpolates *after* the restoration, so SEEDVR2
    only ever sees the source's real frames. Like the generation path, RIFE
    lengthens the picture while VHS_VideoCombine's frame_rate stays put, so
    the audio comes out short and the caller must re-sync it afterwards with
    services.video.audio_stretch.stretch_audio_to_video.
    """
    p = "sv_"
    wf: dict = {
        p+"load": {"class_type": "VHS_LoadVideoPath", "inputs": {
            "video":             str(src),
            "force_rate":        0,
            "custom_width":      0,
            "custom_height":     0,
            "frame_load_cap":    0,
            "skip_first_frames": 0,
            "select_every_nth":  1,
        }},
        # The source's own frame rate, read off the file and wired into the
        # muxer. Taking it from the DB instead would desync every RIFE'd clip,
        # whose stored fps describes the model's rate rather than the file's.
        p+"info": {"class_type": "VHS_VideoInfoSource", "inputs": {
            "video_info": [p+"load", 3],
        }},
        p+"dit": {"class_type": "SeedVR2LoadDiTModel", "inputs": {
            "model":              _DIT_MODEL,
            "device":             "cuda:0",
            "blocks_to_swap":     0,
            "swap_io_components": False,
            "offload_device":     "cpu",
            "cache_model":        False,
            "attention_mode":     _ATTENTION_MODE,
        }},
        p+"vae": {"class_type": "SeedVR2LoadVAEModel", "inputs": {
            "model":               _VAE_MODEL,
            "device":              "cuda:0",
            "encode_tiled":        True,
            "encode_tile_size":    _TILE_SIZE,
            "encode_tile_overlap": _TILE_OVERLAP,
            "decode_tiled":        True,
            "decode_tile_size":    _TILE_SIZE,
            "decode_tile_overlap": _TILE_OVERLAP,
            "offload_device":      "cpu",
            "cache_model":         False,
        }},
        p+"up": {"class_type": "SeedVR2VideoUpscaler", "inputs": {
            "image":              [p+"load", 0],
            "dit":                [p+"dit", 0],
            "vae":                [p+"vae", 0],
            "seed":               _SEED,
            "resolution":         clamp_resolution(resolution),
            "max_resolution":     0,
            "batch_size":         _BATCH_SIZE,
            "uniform_batch_size": False,
            "color_correction":   _COLOR_CORRECTION,
            "temporal_overlap":   _TEMPORAL_OVERLAP,
            "offload_device":     "cpu",
        }},
    }

    # Interpolate the *restored* frames. Same node settings as the generation
    # path so a clip interpolated here is indistinguishable from one
    # interpolated there — only the order and therefore the cost differ.
    frames_node = [p+"up", 0]
    if clamp_rife(rife_multiplier) > 1:
        wf[p+"rife"] = {"class_type": "RIFE VFI", "inputs": {
            "ckpt_name":                  _RIFE_CKPT,
            "clear_cache_after_n_frames": 10,
            "multiplier":                 clamp_rife(rife_multiplier),
            "fast_mode":                  True,
            "ensemble":                   True,
            "scale_factor":               1,
            "dtype":                      "float32",
            "torch_compile":              False,
            "batch_size":                 1,
            "frames":                     [p+"up", 0],
        }}
        frames_node = [p+"rife", 0]

    wf[SAVE_NODE] = {"class_type": "VHS_VideoCombine", "inputs": {
        "frame_rate":      [p+"info", 0],
        "loop_count":      0,
        "filename_prefix": filename_prefix,
        "format":          "video/h264-mp4",
        "pix_fmt":         "yuv420p",
        # A little richer than the generation pass's 19: this is the delivery
        # rendition, and restored fine detail is exactly what a coarse
        # quantiser throws away first.
        "crf":             17,
        "save_metadata":   False,
        # Same reasoning as the MiniMax builder — a format parameter of
        # h264-mp4.json. Left true it would trim the picture back to the audio
        # track, and a RIFE'd source is legitimately longer than its audio.
        "trim_to_audio":   False,
        "pingpong":        False,
        "save_output":     True,
        "images":          frames_node,
    }}
    if has_audio:
        wf[SAVE_NODE]["inputs"]["audio"] = [p+"load", 2]
    return wf, SAVE_NODE
