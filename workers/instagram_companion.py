"""
Instagram companion posts — Stories and Reel.

Called automatically after a feed post (single or carousel) is published.

Stories: one image-story per image in the post (up to 10).
Reel:    a crossfade slideshow MP4 generated from the same images.

Both run fire-and-update: failures are logged and written to the DB but
do NOT raise — the feed post is already live and must not be rolled back.
"""
import asyncio
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx

from core.config import settings
from core.db import AsyncSessionLocal
from core.models import InstagramPost, Video
from services.instagram.companions import find_companion, get_or_create_companion
from services.instagram.graph import (
    REEL_POLL_INTERVAL,
    REEL_POLL_TIMEOUT,
    create_media_container,
    missing_config,
    publish_container,
    share_url,
    wait_container_ready,
)
from services.instagram.ig_video import ensure_ig_compatible
from services.instagram.media import load_media_refs
from workers.video_generator import generate_slideshow

logger = logging.getLogger(__name__)


# ── Stories ───────────────────────────────────────────────────────────────────

async def publish_stories(post_id: uuid.UUID) -> None:
    missing = missing_config()
    if missing:
        logger.warning("Stories skipped for %s — missing config: %s", post_id, ", ".join(missing))
        return

    async with AsyncSessionLocal() as db:
        post = await db.get(InstagramPost, post_id)
        if not post:
            return
        # Stories only support images — skip any video children on a mixed post.
        image_refs = [r for r in await load_media_refs(post, db) if r.kind == "image"]

    media_ids: list[str] = []
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            for ref in image_refs:
                container_id = await create_media_container(
                    client,
                    {"image_url": share_url(ref.filename), "media_type": "STORIES"},
                    f"story container for {ref.filename}",
                )
                story_media_id = await publish_container(
                    client, container_id, f"story publish for {ref.filename}",
                )
                media_ids.append(story_media_id)
                logger.info("Story published: %s (%s)", story_media_id, ref.filename)
                await asyncio.sleep(3)  # avoid "Media ID not available" on rapid successive stories

        status = "posted"
    except Exception as exc:
        logger.error("Stories failed for post %s: %s", post_id, exc)
        status = "posted" if media_ids else "failed"

    async with AsyncSessionLocal() as db:
        post = await db.get(InstagramPost, post_id)
        if post:
            story = get_or_create_companion(post, "story")
            story.status     = status
            story.media_ids  = media_ids or None
            post.updated_at  = datetime.now(timezone.utc)
            await db.commit()


# ── Reel ──────────────────────────────────────────────────────────────────────

async def publish_reel(post_id: uuid.UUID) -> None:
    missing = missing_config()
    if missing:
        logger.warning("Reel skipped for %s — missing config: %s", post_id, ", ".join(missing))
        return

    async with AsyncSessionLocal() as db:
        post = await db.get(InstagramPost, post_id)
        if not post:
            return
        # Companion-reel slideshow currently only ingests images. Videos in a
        # mixed feed post are skipped here — they're already in the carousel.
        image_refs = [r for r in await load_media_refs(post, db) if r.kind == "image"]
        caption = post.caption or ""
        reel_companion = find_companion(post, "reel")
        reel_video_id = reel_companion.video_id if reel_companion else None

    video_path: Path | None = None
    video_is_temp = True  # False when reusing an existing generated video
    reel_media_id: str | None = None
    status = "failed"

    try:
        if reel_video_id:
            # ── Use an existing generated video ───────────────────────────────
            async with AsyncSessionLocal() as db2:
                vid = await db2.get(Video, reel_video_id)
            if not vid or vid.status != "done" or not vid.filepath:
                raise ValueError(f"Referenced video {reel_video_id} is not ready")
            video_path = await ensure_ig_compatible(settings.storage_dir / vid.filepath)
            video_is_temp = False
            logger.info("Reel: using existing video %s for post %s", vid.filename, post_id)
        else:
            # ── 1. Build ordered image file paths ─────────────────────────────
            if not image_refs:
                raise ValueError("Reel companion needs images on the feed post (or an explicit reel_video_id)")
            image_paths: list[Path] = [
                settings.storage_dir / r.filepath for r in image_refs
            ]

            # ── 2. Generate slideshow video ────────────────────────────────────
            settings.reels_dir.mkdir(parents=True, exist_ok=True)
            video_path = await generate_slideshow(
                image_paths,
                settings.reels_dir,
                ffmpeg_path=settings.ffmpeg_path,
            )

        # ── 3. Create Reel container ──────────────────────────────────────────
        kind = "video" if not video_is_temp else "reel"
        async with httpx.AsyncClient(timeout=60) as client:
            container_id = await create_media_container(
                client,
                {
                    "media_type":    "REELS",
                    "video_url":     share_url(video_path.name, kind=kind),
                    "caption":       caption,
                    "share_to_feed": "true",
                },
                "reel container",
            )
            logger.info("Reel container: %s", container_id)

            # ── 4. Poll until FINISHED ────────────────────────────────────────
            await wait_container_ready(
                client, container_id,
                max_wait=REEL_POLL_TIMEOUT,
                poll_interval=REEL_POLL_INTERVAL,
            )

            # ── 5. Publish ────────────────────────────────────────────────────
            reel_media_id = await publish_container(client, container_id, "reel publish")
            logger.info("Reel published: %s", reel_media_id)

        status = "posted"

    except Exception as exc:
        logger.error("Reel failed for post %s: %s", post_id, exc)
        status = "failed"

    finally:
        # Only delete temp (slideshow-generated) files, never the user's stored videos
        if video_is_temp and video_path and video_path.exists():
            try:
                video_path.unlink()
            except Exception as e:
                logger.warning("Could not delete reel temp file %s: %s", video_path, e)

    async with AsyncSessionLocal() as db:
        post = await db.get(InstagramPost, post_id)
        if post:
            reel = get_or_create_companion(post, "reel")
            reel.status      = status
            reel.media_id    = reel_media_id
            post.updated_at  = datetime.now(timezone.utc)
            await db.commit()


