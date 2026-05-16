"""Shared first-frame thumbnail helper for generated videos.

Convention: `{settings.videos_dir}/{video_id}_thumb.jpg`, served by the
`/api/video/thumb/{video_id}` endpoint. Best-effort — failures are logged
and swallowed so a missing thumb never breaks a pipeline.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from core.config import settings

logger = logging.getLogger(__name__)


async def make_video_thumbnail(src: Path, dst: Path) -> None:
    """Write a first-frame JPEG thumbnail next to a video file."""
    try:
        proc = await asyncio.create_subprocess_exec(
            settings.ffmpeg_path,
            "-y", "-i", str(src),
            "-frames:v", "1", "-q:v", "3",
            str(dst),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.wait()
    except Exception as e:
        logger.warning("Thumbnail generation failed for %s: %s", src, e)
