"""ffmpeg pipeline for attaching a generated song to a generated video.

Audio is trimmed to the video's length (-shortest) with a configurable
fade-out at the end. Video stream is copied (no re-encode), so a typical
mux finishes in well under a second.

`include_bed` keeps the video's *own* generated audio quietly underneath the
song rather than discarding it — a MiniMax H3 clip's room and material stay
audible under the music. Same idea, same levels and same filter shape as the
improv tool's ambient bed; see services/video/audio_bed.py.

`song_start` skips into the track before muxing, for a beat cut that was told
to begin at a later bar. It is an input-level `-ss`, so the seek happens before
decoding and the fade-out is still measured from the *video's* length.

Mirrors the shape of services/improv/mux.py — single async function +
private cmd builder + a thin _run_ffmpeg wrapper.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from core.subproc import communicate
from core.video_thumb import probe_has_audio
from services.video.audio_bed import (
    BED_VOLUME_DEFAULT,
    bed_mix_filter,
    clamp_bed_volume,
)

logger = logging.getLogger(__name__)


async def mux_soundtrack(
    video_path: Path,
    song_path: Path,
    out_path: Path,
    *,
    ffmpeg_path: str = "ffmpeg",
    fade_out_seconds: float = 1.0,
    include_bed: bool = False,
    bed_volume: float = BED_VOLUME_DEFAULT,
    song_start: float = 0.0,
) -> None:
    """Mux video stream from `video_path` with audio from `song_path` into
    `out_path`. Audio is trimmed to the video's duration with a fade-out of
    `fade_out_seconds` seconds at the end. Raises RuntimeError on failure.

    With `include_bed`, the video's own audio is mixed in under the song at
    `bed_volume` instead of being dropped. Silently a no-op on a silent
    source — probing beats trusting the caller, since the same video can be
    silent or not depending on which workflow rendered it.

    `song_start` seconds are skipped at the head of the song.
    """
    duration = await _probe_duration(video_path, ffmpeg_path=ffmpeg_path)
    fade_start = max(0.0, duration - fade_out_seconds)
    use_bed = include_bed and await probe_has_audio(video_path)
    builder = _mux_cmd_with_bed if use_bed else _mux_cmd
    extra = {"bed_volume": clamp_bed_volume(bed_volume)} if use_bed else {}
    cmd = builder(
        ffmpeg_path, video_path, song_path, out_path,
        fade_start=fade_start, fade_duration=fade_out_seconds,
        song_start=max(0.0, float(song_start)), **extra,
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
    song_start: float = 0.0,
) -> list[str]:
    afade = f"afade=t=out:st={fade_start:.3f}:d={fade_duration:.3f}"
    return [
        ffmpeg, "-y",
        "-i", str(video),
        *(("-ss", f"{song_start:.3f}") if song_start > 0 else ()),
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


def _mux_cmd_with_bed(
    ffmpeg: str,
    video: Path,
    song: Path,
    out: Path,
    *,
    fade_start: float,
    fade_duration: float,
    bed_volume: float,
    song_start: float = 0.0,
) -> list[str]:
    """Like `_mux_cmd`, but keeps the video's own audio quietly under the song.

    The fade moves onto the *mix* rather than onto the song alone: the bed is
    part of the track now, and fading only the music out would leave the clip's
    own noise running on alone after it.

    The limiter is what pays for `normalize=0` in bed_mix_filter — the song
    keeps its own level, so the bed adds on top of it and the sum can pass
    full scale. Unlike the improv path there is no loudness pass here to catch
    that, and a clipped soundtrack is a worse outcome than a slightly tamed peak.
    """
    afade = f"afade=t=out:st={fade_start:.3f}:d={fade_duration:.3f}"
    return [
        ffmpeg, "-y",
        "-i", str(video),
        *(("-ss", f"{song_start:.3f}") if song_start > 0 else ()),
        "-i", str(song),
        "-filter_complex", bed_mix_filter(
            "0:a", "1:a", bed_volume=bed_volume, tail=f"alimiter=limit=0.95,{afade}",
        ),
        "-map", "0:v:0",
        "-map", "[aout]",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
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
    stdout, stderr = await communicate(proc)
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
    _, stderr = await communicate(proc)
    if proc.returncode != 0:
        tail = stderr.decode(errors="replace")[-600:]
        raise RuntimeError(f"ffmpeg {label} failed (rc={proc.returncode}): {tail}")
