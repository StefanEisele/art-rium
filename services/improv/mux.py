"""
ffmpeg pipeline for piano-improvisation mixes.

Inputs:
  - source video: a generated MP4 from the videos library (no usable audio).
  - recording:    an iPhone MP4 from Blackmagic Camera with the user's hands +
                  Scarlett 2i4 piano audio embedded as a stereo track.

Outputs (three sequential ffmpeg jobs):
  - mix_synth: source video + the recording's audio, loudness-normalised to
               -14 LUFS (Instagram target). Cut to whichever stream is
               shorter (-shortest).
  - mix_hands: the recording itself, audio loudness-normalised, video copied
               unchanged.
  - mix_pip:   source video as background with the recording inset top-right
               (24% width, 24px margin, 12px rounded corners) and the
               recording's audio normalised.

Total wall-time stays well under a minute for typical 5–15 s improv clips.
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

# Loudness target — Instagram recommends -14 LUFS integrated, true-peak <= -1 dBTP.
_LOUDNORM_FILTER = "loudnorm=I=-14:TP=-1.5:LRA=11"

# PiP inset geometry — confirmed with user 2026-05-16.
_PIP_WIDTH_PCT = 0.24    # 24% of background width
_PIP_MARGIN_PX = 24      # 24px margin from top + right
_PIP_RADIUS_PX = 12      # 12px corner radius


async def mux_session(
    source_video: Path,
    recording: Path,
    output_dir: Path,
    ffmpeg_path: str = "ffmpeg",
) -> tuple[Path, Path, Path]:
    """
    Produce the three mix outputs for one improv session.

    Returns (mix_synth_path, mix_hands_path, mix_pip_path), all inside `output_dir`.
    Raises RuntimeError if any ffmpeg invocation exits non-zero.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    session_uuid = uuid.uuid4().hex
    mix_synth = output_dir / f"improv_synth_{session_uuid}.mp4"
    mix_hands = output_dir / f"improv_hands_{session_uuid}.mp4"
    mix_pip   = output_dir / f"improv_pip_{session_uuid}.mp4"

    await _run_ffmpeg(
        _synth_cmd(ffmpeg_path, source_video, recording, mix_synth),
        label="mix_synth",
    )
    await _run_ffmpeg(
        _hands_cmd(ffmpeg_path, recording, mix_hands),
        label="mix_hands",
    )
    await _run_ffmpeg(
        _pip_cmd(ffmpeg_path, source_video, recording, mix_pip),
        label="mix_pip",
    )
    return mix_synth, mix_hands, mix_pip


# ── Private helpers ─────────────────────────────────────────────────────────


def _synth_cmd(ffmpeg: str, source: Path, recording: Path, out: Path) -> list[str]:
    """
    Source video stream + recording audio stream → MP4, loudness-normalised.
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


def _pip_cmd(ffmpeg: str, source: Path, recording: Path, out: Path) -> list[str]:
    """
    Picture-in-picture: source video as background, recording inset in the
    top-right corner with rounded corners and a small margin. Audio comes
    from the recording (loudness-normalised).

    Filter pipeline:
      [0:v]            → background, untouched dimensions
      [1:v]scale       → inset width = round(background_w * 0.24) to even px
      geq alpha mask   → rounded-corner alpha channel (12 px radius)
      overlay          → top-right with 24 px margin
      libx264 yuv420p  → IG-friendly H.264, audio AAC, faststart

    `geq` is per-pixel on the inset only, so the cost is proportional to the
    inset area (a few % of total frame). For short improv clips this is
    negligible.
    """
    w_expr = f"trunc(iw*{_PIP_WIDTH_PCT}/2)*2"
    r = _PIP_RADIUS_PX
    m = _PIP_MARGIN_PX
    # ffmpeg filter_complex: commas inside an expression must be escaped (\,).
    # The alpha mask: opaque pixel iff it's at least `r` px from every edge
    # OR it's inside the corner-arc circle of radius `r`.
    geq_alpha = (
        f"if(gt(min(min(X\\,W-X)\\,min(Y\\,H-Y))\\,{r})\\,255\\,"
        f"if(lte(hypot(max(0\\,{r}-min(X\\,W-X))\\,max(0\\,{r}-min(Y\\,H-Y)))\\,{r})\\,255\\,0))"
    )
    filter_complex = (
        f"[1:v]scale={w_expr}:-2,format=yuva420p,"
        f"geq=lum='p(X\\,Y)':cb='p(X\\,Y)':cr='p(X\\,Y)':a='{geq_alpha}'[ovr];"
        f"[0:v][ovr]overlay=W-w-{m}:{m}[v]"
    )
    return [
        ffmpeg, "-y",
        "-i", str(source),
        "-i", str(recording),
        "-filter_complex", filter_complex,
        "-map", "[v]",
        "-map", "1:a:0",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "medium", "-crf", "20",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
        "-af", _LOUDNORM_FILTER,
        "-shortest",
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
