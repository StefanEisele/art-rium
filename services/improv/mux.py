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
  - mix_pip:   source video as background with the recording inset in one of
               the four corners (24% width, 24px margin, 12px rounded corners)
               and the recording's audio normalised.

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
PIP_WIDTH_PCT_DEFAULT = 0.24    # 24% of background width
PIP_WIDTH_PCT_MIN     = 0.10    # 10% floor (any smaller and hands aren't readable)
PIP_WIDTH_PCT_MAX     = 0.50    # 50% ceiling (any larger covers half the source)
_PIP_MARGIN_PX = 24      # margin from the chosen corner
_PIP_RADIUS_PX = 12      # 12px corner radius

# Valid corner keys → ffmpeg overlay X:Y expressions. The background size is
# W,H and the inset size is w,h in overlay's coordinate space.
_PIP_CORNERS: dict[str, tuple[str, str]] = {
    "tr": ("W-w-{m}", "{m}"),
    "br": ("W-w-{m}", "H-h-{m}"),
    "tl": ("{m}",     "{m}"),
    "bl": ("{m}",     "H-h-{m}"),
}
PIP_CORNER_DEFAULT = "tr"


def clamp_pip_width(pct: float) -> float:
    """Clamp PiP width percent to the supported [MIN, MAX] window."""
    return max(PIP_WIDTH_PCT_MIN, min(PIP_WIDTH_PCT_MAX, pct))


async def mux_session(
    source_video: Path,
    recording: Path,
    output_dir: Path,
    ffmpeg_path: str = "ffmpeg",
    pip_corner: str = PIP_CORNER_DEFAULT,
    pip_width_pct: float = PIP_WIDTH_PCT_DEFAULT,
) -> tuple[Path, Path, Path]:
    """
    Produce the three mix outputs for one improv session.

    `pip_corner` is one of "tr" / "br" / "tl" / "bl" — picks the PiP inset
    location for the third output. Falls back to "tr" on unknown values
    rather than raising, since this is a UX preference, not a hard contract.

    `pip_width_pct` is the inset's width as a fraction of the background
    width (e.g. 0.24 = 24%). Clamped to [0.10, 0.50].

    Returns (mix_synth_path, mix_hands_path, mix_pip_path), all inside `output_dir`.
    Raises RuntimeError if any ffmpeg invocation exits non-zero.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    session_uuid = uuid.uuid4().hex
    mix_synth = output_dir / f"improv_synth_{session_uuid}.mp4"
    mix_hands = output_dir / f"improv_hands_{session_uuid}.mp4"
    mix_pip   = output_dir / f"improv_pip_{session_uuid}.mp4"

    corner = pip_corner if pip_corner in _PIP_CORNERS else PIP_CORNER_DEFAULT
    width_pct = clamp_pip_width(pip_width_pct)

    await _run_ffmpeg(
        _synth_cmd(ffmpeg_path, source_video, recording, mix_synth),
        label="mix_synth",
    )
    await _run_ffmpeg(
        _hands_cmd(ffmpeg_path, recording, mix_hands),
        label="mix_hands",
    )
    await _run_ffmpeg(
        _pip_cmd(ffmpeg_path, source_video, recording, mix_pip, corner=corner, width_pct=width_pct),
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


def _pip_cmd(
    ffmpeg: str,
    source: Path,
    recording: Path,
    out: Path,
    *,
    corner: str = PIP_CORNER_DEFAULT,
    width_pct: float = PIP_WIDTH_PCT_DEFAULT,
) -> list[str]:
    """
    Picture-in-picture: source video as background, recording inset in one of
    the four corners with rounded corners and a small margin. Audio comes
    from the recording (loudness-normalised).

    Filter pipeline:
      [0:v]            → background, untouched dimensions
      [1:v]scale       → inset width = round(background_w * width_pct) to even px
      geq alpha mask   → rounded-corner alpha channel (12 px radius)
      overlay          → corner-of-choice with 24 px margin
      libx264 yuv420p  → IG-friendly H.264, audio AAC, faststart

    `geq` is per-pixel on the inset only, so the cost is proportional to the
    inset area (a few % of total frame). For short improv clips this is
    negligible.
    """
    w_expr = f"trunc(iw*{width_pct:.4f}/2)*2"
    r = _PIP_RADIUS_PX
    m = _PIP_MARGIN_PX
    x_tpl, y_tpl = _PIP_CORNERS.get(corner, _PIP_CORNERS[PIP_CORNER_DEFAULT])
    overlay_xy = f"{x_tpl.format(m=m)}:{y_tpl.format(m=m)}"
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
        f"[0:v][ovr]overlay={overlay_xy}[v]"
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
