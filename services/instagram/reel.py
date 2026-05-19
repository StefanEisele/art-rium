"""
Remote-scheduled reel publisher.

Generates the slideshow MP4 (or reuses an existing Video row) and creates a
Graph API REELS container with `scheduled_publish_time` set, so Instagram
publishes it autonomously at the configured companion time. Once the
container reaches FINISHED, the local server is no longer required for the
reel to go live.

Companion publish time = companion_at(post.scheduled_at, reel_delay_minutes,
companion_time). The same offset semantics are used by the immediate-publish
fallback in workers/instagram_companion.py::publish_reel.
"""
from __future__ import annotations

import logging
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import httpx
from sqlalchemy import select

from core.config import settings
from core.db import AsyncSessionLocal
from core.models import Image, InstagramPost, Video
from core.scheduling import companion_at
from services.instagram.graph import (
    REEL_POLL_INTERVAL,
    REEL_POLL_TIMEOUT,
    create_media_container,
    missing_config,
    share_url,
    wait_container_ready,
)
from services.instagram.publisher import MIN_REMOTE_LEAD_SECONDS
from workers.video_generator import generate_slideshow

logger = logging.getLogger(__name__)

ScheduleStatus = Literal["remote_scheduled", "failed", "skipped"]


def reel_publish_time(post: InstagramPost) -> datetime | None:
    """When the reel companion should publish, based on the feed schedule + delay."""
    if post.reel_delay_minutes is None:
        return None
    return companion_at(post.scheduled_at, post.reel_delay_minutes, post.companion_time)


async def schedule_reel(post_id: uuid.UUID) -> tuple[ScheduleStatus, str | None]:
    """
    Generate the reel video (if needed) and create a REELS container with
    scheduled_publish_time. Returns (status, creation_id).
    """
    missing = missing_config()
    if missing:
        logger.warning("schedule_reel %s skipped — missing config: %s", post_id, ", ".join(missing))
        return "skipped", None

    async with AsyncSessionLocal() as db:
        post = await db.get(InstagramPost, post_id)
        if not post or post.status != "scheduled":
            return "skipped", None
        if post.reel_delay_minutes is None:
            return "skipped", None
        if post.reel_creation_id:
            return "skipped", post.reel_creation_id

        publish_at = reel_publish_time(post)
        lead = (publish_at - datetime.now(timezone.utc)).total_seconds()
        if lead < MIN_REMOTE_LEAD_SECONDS:
            logger.info("schedule_reel %s skipped — only %ds lead, need ≥%ds",
                        post_id, lead, MIN_REMOTE_LEAD_SECONDS)
            return "skipped", None

        all_ids = [post.image_id] + list(post.carousel_image_ids or [])
        img_result = await db.execute(select(Image).where(Image.id.in_(all_ids)))
        images = {img.id: img for img in img_result.scalars().all()}
        caption = post.caption or ""
        reel_video_id = post.reel_video_id
        existing_filename = post.reel_video_filename

    video_path, video_filename, kind = await _resolve_video(
        post_id, all_ids, images, reel_video_id, existing_filename,
    )

    creation_id, api_error = None, None
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            creation_id = await create_media_container(
                client,
                {
                    "media_type":             "REELS",
                    "video_url":              share_url(video_filename, kind=kind),
                    "caption":                caption,
                    "share_to_feed":          "true",
                    "scheduled_publish_time": str(int(publish_at.timestamp())),
                },
                "reel container",
            )
            logger.info("schedule_reel %s — container %s, polling for FINISHED…", post_id, creation_id)

            await wait_container_ready(
                client, creation_id,
                max_wait=REEL_POLL_TIMEOUT,
                poll_interval=REEL_POLL_INTERVAL,
            )
    except Exception as exc:
        api_error = f"{type(exc).__name__}: {exc}"
        logger.error("schedule_reel %s failed: %s\n%s", post_id, exc, traceback.format_exc())

    async with AsyncSessionLocal() as db:
        post = await db.get(InstagramPost, post_id)
        if not post:
            return "skipped", None
        now = datetime.now(timezone.utc)
        if creation_id and not api_error:
            post.reel_creation_id    = creation_id
            post.reel_status         = "remote_scheduled"
            post.reel_scheduled_at   = publish_at
            post.reel_video_filename = video_filename if kind == "reel" else None
            post.error               = None
            post.updated_at          = now
            await db.commit()
            logger.info("schedule_reel %s → remote_scheduled (creation_id=%s, publish_at=%s)",
                        post_id, creation_id, publish_at.isoformat())
            return "remote_scheduled", creation_id

        post.reel_status = "failed"
        post.error       = api_error or "Unknown error"
        post.updated_at  = now
        await db.commit()
        return "failed", None


async def _resolve_video(
    post_id: uuid.UUID,
    all_ids: list[uuid.UUID],
    images: dict[uuid.UUID, Image],
    reel_video_id: uuid.UUID | None,
    existing_filename: str | None,
) -> tuple[Path, str, str]:
    """
    Return (path, filename, kind) where kind is "video" (curated Video row) or
    "reel" (temp slideshow). Reuses an existing slideshow if filename already
    set on the post.
    """
    if reel_video_id:
        async with AsyncSessionLocal() as db:
            vid = await db.get(Video, reel_video_id)
        if not vid or vid.status != "done" or not vid.filepath:
            raise ValueError(f"Referenced video {reel_video_id} is not ready")
        path = settings.storage_dir / vid.filepath
        return path, path.name, "video"

    if existing_filename:
        path = settings.reels_dir / existing_filename
        if path.exists():
            logger.info("schedule_reel %s — reusing existing slideshow %s", post_id, existing_filename)
            return path, existing_filename, "reel"

    image_paths: list[Path] = []
    for img_id in all_ids:
        img = images.get(img_id)
        if not img:
            raise ValueError(f"Reel: image {img_id} not found in DB")
        image_paths.append(settings.storage_dir / img.filepath)

    settings.reels_dir.mkdir(parents=True, exist_ok=True)
    path = await generate_slideshow(
        image_paths,
        settings.reels_dir,
        ffmpeg_path=settings.ffmpeg_path,
    )
    return path, path.name, "reel"
