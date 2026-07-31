"""ffmpeg grain pass — roughens up an over-clean generated clip.

Wan2.2 output is smooth to the point of looking plastic. A temporal luma-noise
pass puts texture back without touching colour, sharpness or geometry.

Preview and full render share one filter definition (`grain_filter`) and one
strength mapping, so the excerpt a user judges cannot drift away from the
render they then apply.

Mirrors the shape of services/video/soundtrack.py — async entry points +
private cmd builders + a thin _run_ffmpeg wrapper.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from core.video_thumb import probe_video_duration

logger = logging.getLogger(__name__)

# The UI slider runs 0-100 so its whole travel is useful. ffmpeg's own noise
# scale goes to 100 as well, but anything past ~60 on the luma plane is pure
# static rather than grain, so the slider maps onto 0-60 instead of 1:1.
NOISE_CEILING = 60

PREVIEW_SECONDS = 4.0

# Grain is close to incompressible by design, so the encoder settings decide
# whether this feature costs 2x the source size or 36x it. Measured on a real
# 16s clip at strength 30, against a losslessly-grained reference:
#
#   libx264 crf 18 (tune grain)   121 MB   36.6x source   grain kept 37.7 dB
#   libx265 crf 28 (tune grain)    17 MB    5.2x source   grain kept 29.3 dB
#   libx265 crf 30 (tune grain)     6.7MB    2.0x source   grain kept 28.5 dB
#
# Retention flattens out below ~29 dB because random noise is not reproducible
# bit-for-bit at any bitrate — past that point extra bits buy a closer match to
# one particular noise field, not more visible texture. crf 30 sits just past
# the knee: visually the grain is unmistakably there (checked on magnified
# stills), at 2x the source size instead of 36x. `-tune grain` still matters —
# without it the encoder spends its bits smoothing the noise back out.
_CODEC = "libx265"
_CRF = "30"
_PRESET = "medium"
# hvc1 rather than the default hev1 tag: QuickTime and Safari refuse to play
# HEVC in MP4 without it, and these files are watched in a browser.
_HEVC_TAG = "hvc1"


def clamp_strength(value: int | float | None) -> int:
    """UI strength onto 0-100. None/garbage reads as 0 (grain off)."""
    try:
        return max(0, min(100, int(round(float(value)))))
    except (TypeError, ValueError):
        return 0


def grain_filter(strength: int) -> str:
    """ffmpeg -vf value for `strength` on the 0-100 UI scale.

    Luma plane only (`c0`): film grain lives in luminance, and noising the
    chroma planes produces coloured speckle that reads as a compression fault
    rather than as texture. `c0f=t` regenerates the pattern every frame —
    without the temporal flag the grain freezes into a static dirt overlay
    that looks stuck to the lens instead of moving with the picture.
    """
    return f"noise=c0s={round(clamp_strength(strength) * NOISE_CEILING / 100)}:c0f=t"


def preview_window(duration: float, seconds: float = PREVIEW_SECONDS) -> tuple[float, float]:
    """(start, length) of the excerpt to grade for a preview.

    Taken from the middle: the opening frames of an i2v clip are the source
    still barely moving, which is the least representative place to judge how
    grain sits on actual motion. Clips shorter than the window are graded
    whole rather than producing a zero-length preview.

    A duration of 0 means the probe failed (probe_video_duration swallows its
    own errors); grade from the start for the full window and let ffmpeg stop
    at whatever the real end turns out to be.
    """
    if duration <= 0:
        return 0.0, seconds
    if duration <= seconds:
        return 0.0, duration
    return (duration - seconds) / 2.0, seconds


async def render_grain(
    src: Path,
    out: Path,
    strength: int,
    *,
    ffmpeg_path: str = "ffmpeg",
) -> None:
    """Grade the whole of `src` into `out`. Raises RuntimeError on failure."""
    await _run_ffmpeg(_grain_cmd(ffmpeg_path, src, out, strength), label="grain")


async def render_grain_preview(
    src: Path,
    out: Path,
    strength: int,
    *,
    ffmpeg_path: str = "ffmpeg",
    seconds: float = PREVIEW_SECONDS,
) -> float:
    """Grade a short excerpt out of the middle of `src` into `out`.

    Returns the excerpt's length in seconds. Raises RuntimeError on failure.
    """
    duration = await probe_video_duration(src)
    start, length = preview_window(duration, seconds)
    await _run_ffmpeg(
        _grain_preview_cmd(ffmpeg_path, src, out, strength, start=start, length=length),
        label="grain_preview",
    )
    return length


def _encode_args(strength: int) -> list[str]:
    """Filter + encoder settings shared by the preview and the full render.

    Shared deliberately: if the preview encoded differently from the render,
    the strength a user dials in on the preview would not be the strength they
    get, which defeats the point of previewing at all.
    """
    return [
        "-vf", grain_filter(strength),
        "-c:v", _CODEC,
        "-tag:v", _HEVC_TAG,
        "-tune", "grain",
        "-crf", _CRF,
        "-preset", _PRESET,
        "-pix_fmt", "yuv420p",
    ]


def _grain_cmd(ffmpeg: str, src: Path, out: Path, strength: int) -> list[str]:
    return [
        ffmpeg, "-y",
        "-i", str(src),
        *_encode_args(strength),
        # Audio (an attached soundtrack, or LTX's native track) is passed
        # through untouched; this pass only ever re-encodes the picture.
        "-c:a", "copy",
        "-movflags", "+faststart",
        str(out),
    ]


def _grain_preview_cmd(
    ffmpeg: str,
    src: Path,
    out: Path,
    strength: int,
    *,
    start: float,
    length: float,
) -> list[str]:
    return [
        ffmpeg, "-y",
        # -ss ahead of -i seeks by index rather than decoding up to the mark,
        # which is what keeps a preview to a couple of seconds.
        "-ss", f"{start:.3f}",
        "-t", f"{length:.3f}",
        "-i", str(src),
        *_encode_args(strength),
        # No audio: input seeking can land the audio a few ms off the video,
        # and a grain preview is a picture judgement anyway.
        "-an",
        "-movflags", "+faststart",
        str(out),
    ]


async def _run_ffmpeg(cmd: list[str], *, label: str) -> None:
    logger.info("ffmpeg %s: %s", label, " ".join(str(c) for c in cmd))
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        tail = stderr.decode(errors="replace")[-600:]
        raise RuntimeError(f"ffmpeg {label} failed (rc={proc.returncode}): {tail}")
