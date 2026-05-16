"""
ffmpeg pipeline for piano-improvisation mixes.

Input:
  - source video: a generated MP4 from the videos library (no usable audio).
  - recording:    an iPhone MP4 from Blackmagic Camera with the user's hands +
                  Scarlett 2i4 piano audio embedded as a stereo track.

Output:
  - mix_synth:    source video + the recording's audio, loudness-normalised
                  to -14 LUFS (Instagram target). Cut to whichever stream is
                  shorter (-shortest).
  - mix_hands:    the recording itself, audio loudness-normalised, re-encoded
                  to h264/aac for consistency with the gallery.

The two ffmpeg jobs are run sequentially (each is ~real-time-ish on Wan-2.2-
sized snippets, so total wall-time stays well under a minute for typical
5-15s improv clips).
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

# Loudness target — Instagram recommends -14 LUFS integrated, true-peak <= -1 dBTP.
_LOUDNORM_FILTER = "loudnorm=I=-14:TP=-1.5:LRA=11"


async def mux_session(
    source_video: Path,
    recording: Path,
    output_dir: Path,
    ffmpeg_path: str = "ffmpeg",
) -> tuple[Path, Path]:
    """
    Produce the two mix outputs for one improv session.

    Returns (mix_synth_path, mix_hands_path), both inside `output_dir`.
    Raises RuntimeError if any ffmpeg invocation exits non-zero.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    session_uuid = uuid.uuid4().hex
    mix_synth = output_dir / f"improv_synth_{session_uuid}.mp4"
    mix_hands = output_dir / f"improv_hands_{session_uuid}.mp4"

    await _run_ffmpeg(
        _synth_cmd(ffmpeg_path, source_video, recording, mix_synth),
        label="mix_synth",
    )
    await _run_ffmpeg(
        _hands_cmd(ffmpeg_path, recording, mix_hands),
        label="mix_hands",
    )
    return mix_synth, mix_hands


# ── Private helpers ─────────────────────────────────────────────────────────


def _synth_cmd(ffmpeg: str, source: Path, recording: Path, out: Path) -> list[str]:
    """
    source video stream + recording audio stream → MP4, loudness-normalised.
    `-shortest` clips whichever ends first so the result is gap-free.
    """
    return [
        ffmpeg, "-y",
        "-i", str(source),
        "-i", str(recording),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
        "-af", _LOUDNORM_FILTER,
        "-shortest",
        "-movflags", "+faststart",
        str(out),
    ]


def _hands_cmd(ffmpeg: str, recording: Path, out: Path) -> list[str]:
    """
    Re-encode the iPhone recording with normalised audio. Video is copied
    if the container allows it; we re-encode audio only for the loudness pass.
    """
    return [
        ffmpeg, "-y",
        "-i", str(recording),
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
        "-af", _LOUDNORM_FILTER,
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
