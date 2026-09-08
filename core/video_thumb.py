"""Shared first-frame thumbnail + dimension helpers for generated videos.

Convention: `{settings.videos_dir}/{video_id}_thumb.jpg`, served by the
`/api/video/thumb/{video_id}` endpoint. Best-effort — failures are logged
and swallowed so a missing thumb never breaks a pipeline.
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from core.config import settings

logger = logging.getLogger(__name__)


def _ffprobe_path() -> str:
    p = Path(settings.ffmpeg_path)
    sibling = p.parent / ("ffprobe" + p.suffix)
    return str(sibling) if sibling.exists() else "ffprobe"


async def make_video_thumbnail(src: Path, dst: Path) -> None:
    """Write a representative JPEG thumbnail next to a video file.

    Not the first frame, which is what this used to take and why finished
    videos showed up as a black card with a play button on it: a diffusion clip
    routinely opens on a dark or half-formed frame, and a fade-in guarantees it.

    `thumbnail=100` scores a hundred frames on how far each one differs from the
    average of the batch and keeps the most representative — so it skips the
    dark opening without needing to know the duration, and without picking a
    frame at some arbitrary fixed offset that a short clip may not have.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            settings.ffmpeg_path,
            "-y", "-i", str(src),
            "-vf", "thumbnail=100",
            "-frames:v", "1", "-q:v", "3",
            str(dst),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.wait()
    except Exception as e:
        logger.warning("Thumbnail generation failed for %s: %s", src, e)


async def probe_video_duration(src: Path) -> float:
    """Return the video duration in seconds, or 0.0 if ffprobe fails."""
    try:
        proc = await asyncio.create_subprocess_exec(
            _ffprobe_path(),
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            str(src),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await proc.communicate()
        if proc.returncode != 0:
            return 0.0
        return float(out.decode().strip() or 0.0)
    except Exception as e:
        logger.warning("ffprobe duration probe failed for %s: %s", src, e)
        return 0.0


async def extract_video_frames(
    src: Path,
    *,
    count: int = 3,
    max_edge: int = 384,
) -> list[bytes]:
    """Extract *count* JPG sample frames from the video, evenly spaced.

    Frames are scaled so the longer edge is at most *max_edge* pixels, then
    returned as raw JPG bytes (in playback order). The art-rium use-case is
    feeding video context to the article LLM — small frames keep the VL
    token budget reasonable.

    Returns an empty list if ffprobe can't read the file or ffmpeg fails on
    every extraction. Partial success is fine — the caller will just have
    fewer samples to work with.
    """
    if count < 1:
        return []

    duration = await probe_video_duration(src)
    if duration <= 0:
        return []

    # Evenly-spaced internal points (avoids the first/last frame which are
    # often static or compressed differently): for count=3 → 0.25, 0.5, 0.75.
    fracs = [(i + 1) / (count + 1) for i in range(count)]

    # ffmpeg scale filter that constrains the longer edge — keeps aspect.
    vf = (
        f"scale='if(gt(iw,ih),{max_edge},-2)':'if(gt(iw,ih),-2,{max_edge})'"
    )

    frames: list[bytes] = []
    for frac in fracs:
        ts = duration * frac
        try:
            proc = await asyncio.create_subprocess_exec(
                settings.ffmpeg_path,
                "-y",
                "-ss", f"{ts:.3f}",
                "-i", str(src),
                "-frames:v", "1",
                "-vf", vf,
                "-q:v", "4",            # JPEG quality (lower = better; 2–5 is good)
                "-f", "image2pipe",
                "-vcodec", "mjpeg",
                "pipe:1",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            out, _ = await proc.communicate()
            if proc.returncode == 0 and out:
                frames.append(out)
        except Exception as e:
            logger.warning("Frame extraction failed for %s at t=%.2fs: %s", src, ts, e)

    return frames


async def probe_has_audio(src: Path) -> bool:
    """Return True if `src` has at least one audio stream. Used to decide
    whether an "include background audio" opt-in has anything to mix — a
    silent source makes it a harmless no-op rather than an error."""
    try:
        proc = await asyncio.create_subprocess_exec(
            _ffprobe_path(),
            "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=index",
            "-of", "csv=p=0",
            str(src),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await proc.communicate()
        return proc.returncode == 0 and bool(out.decode().strip())
    except Exception as e:
        logger.warning("ffprobe audio-stream probe failed for %s: %s", src, e)
        return False


async def probe_video_dimensions(src: Path) -> tuple[int | None, int | None]:
    """Return (width, height) of the first video stream, or (None, None)
    when ffprobe fails. Used to populate Video.width/height for sources
    where the producer doesn't already know the dimensions (e.g. Improv
    mixes muxed from iPhone recordings).
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            _ffprobe_path(),
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "json",
            str(src),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, _ = await proc.communicate()
        if proc.returncode != 0:
            return None, None
        info = json.loads(out.decode("utf-8", errors="replace"))
        stream = (info.get("streams") or [{}])[0]
        w = stream.get("width")
        h = stream.get("height")
        return (int(w) if w else None, int(h) if h else None)
    except Exception as e:
        logger.warning("ffprobe dimension probe failed for %s: %s", src, e)
        return None, None


def _rate(value: str | None) -> float | None:
    """An ffprobe rational ("30000/1001") as a float, or None.

    None for 0/0 — which is what ffprobe writes when it has no answer — and
    for anything past 240, since no source in this library is a real high-speed
    camera and a number that large means the field was never a frame rate.
    """
    if not value or "/" not in value:
        return None
    num, den = value.split("/", 1)
    try:
        fps = float(num) / float(den)
    except (ValueError, ZeroDivisionError):
        return None
    return fps if 0 < fps <= 240 else None


async def probe_video_frames(src: Path) -> tuple[int | None, float | None]:
    """Return (frame_count, fps) of the first video stream, or (None, None).

    `nb_frames` is read first because it is exact when the container carries
    it; when it does not, duration × rate is the honest fallback and is
    rounded down — overstating the length would have VACE pad the tail of a
    render with flat grey (comfy_extras/nodes_wan.py) and the clip would drift
    unguided at the end.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            _ffprobe_path(),
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=nb_frames,avg_frame_rate,r_frame_rate,duration",
            "-of", "json",
            str(src),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, _ = await proc.communicate()
        if proc.returncode != 0:
            return None, None
        stream = (json.loads(out.decode("utf-8", errors="replace")).get("streams") or [{}])[0]

        # `avg_frame_rate` first. `r_frame_rate` is the *highest* rate the
        # stream could contain, and on variable-rate phone footage that is a
        # tick resolution rather than a frame rate — real tracks in this
        # library came back at 1000 and 2000 fps from it, which then silently
        # mispaired a mask with the footage it was keyed from. The average is
        # what the file actually plays at.
        fps = _rate(stream.get("avg_frame_rate")) or _rate(stream.get("r_frame_rate"))

        frames = stream.get("nb_frames")
        if frames:
            return int(frames), fps
        duration = stream.get("duration")
        if duration and fps:
            return int(float(duration) * fps), fps
        return None, fps
    except Exception as e:
        logger.warning("ffprobe frame probe failed for %s: %s", src, e)
        return None, None
