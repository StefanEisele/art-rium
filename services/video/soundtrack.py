"""ffmpeg pipeline for attaching a generated song to a generated video.

Audio is trimmed to the video's length (-shortest) with a configurable
fade-out at the end. Video stream is copied (no re-encode), so a typical
mux finishes in well under a second.

Mirrors the shape of services/improv/mux.py — single async function +
private cmd builder + a thin _run_ffmpeg wrapper.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


async def mux_soundtrack(
    video_path: Path,
    song_path: Path,
    out_path: Path,
    *,
    ffmpeg_path: str = "ffmpeg",
    fade_out_seconds: float = 1.0,
) -> None:
    """Mux video stream from `video_path` with audio from `song_path` into
    `out_path`. Audio is trimmed to the video's duration with a fade-out of
    `fade_out_seconds` seconds at the end. Raises RuntimeError on failure."""
    duration = await _probe_duration(video_path, ffmpeg_path=ffmpeg_path)
    fade_start = max(0.0, duration - fade_out_seconds)
    cmd = _mux_cmd(
        ffmpeg_path, video_path, song_path, out_path,
        fade_start=fade_start, fade_duration=fade_out_seconds,
    )
    await _run_ffmpeg(cmd, label="soundtrack_mux")


def _mux_cmd(
    ffmpeg: str,
    video: Path,
    song: Path,
    out: Path,
    *,
    fade_start: float,
    fade_duration: float,
) -> list[str]:
    afade = f"afade=t=out:st={fade_start:.3f}:d={fade_duration:.3f}"
    return [
        ffmpeg, "-y",
        "-i", str(video),
        "-i", str(song),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
        "-af", afade,
        "-shortest",
        "-movflags", "+faststart",
        str(out),
    ]


async def _probe_duration(path: Path, *, ffmpeg_path: str) -> float:
    """Read container duration in seconds via ffprobe. ffprobe is assumed
    to live next to ffmpeg (standard distribution shape)."""
    ffprobe = _ffprobe_for(ffmpeg_path)
    cmd = [
        ffprobe,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        tail = stderr.decode(errors="replace")[-400:]
        raise RuntimeError(f"ffprobe failed (rc={proc.returncode}): {tail}")
    raw = stdout.decode(errors="replace").strip()
    try:
        return float(raw)
    except ValueError as exc:
        raise RuntimeError(f"ffprobe returned non-numeric duration: {raw!r}") from exc


def _ffprobe_for(ffmpeg_path: str) -> str:
    """Derive the ffprobe binary path from the configured ffmpeg path."""
    # Bare "ffmpeg" on PATH → assume "ffprobe" on PATH.
    if ffmpeg_path in ("ffmpeg", "ffmpeg.exe"):
        return "ffprobe"
    p = Path(ffmpeg_path)
    candidate = p.with_name("ffprobe" + p.suffix)
    return str(candidate)


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
