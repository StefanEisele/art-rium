"""Time-stretch LTX-2.3's native audio to match a RIFE-interpolated clip.

RIFE VFI multiplies frame count while VHS_VideoCombine's frame_rate stays
constant, so a RIFE'd LTX segment's video track ends up `rife_multiplier`x
longer in real time than its native audio track. But the *source* file we
receive isn't just the short native audio — VHS_VideoCombine itself already
pads it with trailing silence out to (roughly) the video's length before we
ever see it (`apad=whole_dur=...` in VideoHelperSuite's combine_video). So
the real generated audio only occupies the first `native_length/fps` seconds
of the source's audio stream; the rest is dead air.

This module first trims that padding away, then time-stretches the *real*
audio via ffmpeg's `rubberband` filter (Rubber Band Library — a phase-vocoder
with transient detection, tuned here for percussive/transient content like
footsteps and clicks, which held up far better by ear than plain `atempo`
resampling in an A/B listening test), and finally pads/trims the result to
match the video exactly.

Mirrors the shape of services/video/soundtrack.py and services/improv/mux.py:
one async entry point + pure cmd builders + a thin _run_ffmpeg wrapper.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from core.video_thumb import probe_video_duration

logger = logging.getLogger(__name__)


def build_rubberband_filter(ratio: float, *, detector: str = "percussive") -> str:
    """Build a `rubberband=tempo=...` filter string reaching `ratio`
    (output_duration / input_duration is 1/ratio — tempo is a *speed*
    multiplier, so ratio < 1 slows audio down / stretches it). Unlike
    atempo, a single stage covers any ratio in [0.01, 100] — no chaining
    needed. `detector=percussive` preserves transient attacks (footsteps,
    clicks) noticeably better than the default compound detector for
    LTX's generated sound effects.

    Raises ValueError for a non-positive ratio (nothing sane to build).
    """
    if ratio <= 0:
        raise ValueError(f"rubberband tempo ratio must be positive, got {ratio!r}")
    return f"rubberband=tempo={ratio:.6f}:detector={detector}"


def build_stretch_cmd(
    ffmpeg: str,
    src: Path,
    dest: Path,
    *,
    ratio: float,
    native_audio_duration: float,
    target_duration: float,
) -> list[str]:
    """Video stream copied unchanged. Audio: trim away VHS_VideoCombine's
    trailing silence pad down to the `native_audio_duration` seconds of
    *real* generated audio, time-stretch that via Rubber Band (pitch
    preserved), then pad/trim to exactly `target_duration` seconds so any
    rounding drift can't leave a trailing gap or a truncated tail."""
    af = (
        f"atrim=end={native_audio_duration:.6f},asetpts=PTS-STARTPTS,"
        f"{build_rubberband_filter(ratio)},apad"
    )
    return [
        ffmpeg, "-y",
        "-i", str(src),
        "-map", "0:v:0",
        "-map", "0:a:0",
        "-c:v", "copy",
        "-af", af,
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
        "-t", f"{target_duration:.3f}",
        "-movflags", "+faststart",
        str(dest),
    ]


async def stretch_ltx_audio(
    src: Path,
    dest: Path,
    *,
    native_length: int,
    fps: int,
    ffmpeg_path: str = "ffmpeg",
) -> None:
    """Re-sync `src` (a RIFE'd LTX segment: correct-length video, native
    short-length audio) into `dest`, with audio stretched to match.

    `native_length`/`fps` give the *exact* pre-RIFE audio duration analytically
    (that's what LTXVEmptyLatentAudio was built at — no need to probe it). The
    video's actual RIFE'd duration is probed rather than computed from the
    RIFE multiplier, since RIFE VFI's real output frame count isn't a clean
    multiply — probing self-corrects for that. Raises RuntimeError if ffmpeg
    fails or the probe comes back empty.
    """
    video_duration = await probe_video_duration(src)
    if video_duration <= 0:
        raise RuntimeError(f"Could not probe video duration of {src}")
    native_audio_duration = native_length / fps
    ratio = native_audio_duration / video_duration
    cmd = build_stretch_cmd(
        ffmpeg_path, src, dest,
        ratio=ratio, native_audio_duration=native_audio_duration, target_duration=video_duration,
    )
    await _run_ffmpeg(cmd, label="ltx_audio_stretch")


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
