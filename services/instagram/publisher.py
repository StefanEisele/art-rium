"""
Instagram feed publisher — Single Source of Truth for putting a feed post on
Instagram via the Graph API.

Two paths:

  schedule_feed(post_id) -> (status, creation_id)
      Creates a container with `scheduled_publish_time` set; Instagram queues
      the publish itself, so the local server may be offline at the actual
      publish moment. Requires scheduled_at to be at least
      MIN_REMOTE_LEAD_SECONDS in the future.

  publish_feed(post_id) -> (status, media_id)
      Synchronously creates a container, polls until ready, then calls
      media_publish — used for /post-now and as a fallback when scheduled_at
      is too close to now to use remote scheduling.

The scheduling-vs-publishing flow is dictated by the Graph API: setting
`scheduled_publish_time` on container creation is mutually exclusive with
calling /media_publish — Instagram does the publish itself at the scheduled
moment. See https://developers.facebook.com/docs/instagram-api/guides/content-publishing
"""
from __future__ import annotations

import logging
import traceback
import uuid
from datetime import datetime, timezone
from typing import Literal

import httpx
from sqlalchemy import select

from core.config import settings
from core.db import AsyncSessionLocal
from core.models import Image, InstagramPost
from core.scheduling import companion_at
from services.instagram.graph import (
    create_media_container,
    missing_config,
    publish_container,
    share_url,
    wait_container_ready,
)

logger = logging.getLogger(__name__)

PublishStatus = Literal["posted", "failed", "skipped"]
ScheduleStatus = Literal["remote_scheduled", "failed", "skipped"]

# Instagram requires scheduled_publish_time to be ≥10 minutes in the future.
# We add a small buffer so the request itself doesn't slip past the boundary.
MIN_REMOTE_LEAD_SECONDS = 11 * 60


# ── Remote scheduling (server-independent) ───────────────────────────────────


async def schedule_feed(post_id: uuid.UUID) -> tuple[ScheduleStatus, str | None]:
    """
    Create the feed container with scheduled_publish_time set so Instagram
    publishes it autonomously. Returns (status, creation_id).

      remote_scheduled — container created, Instagram will publish at scheduled_at
      failed           — Graph API rejected (post.error populated)
      skipped          — config missing, lead time too short, or already scheduled
    """
    missing = missing_config()
    if missing:
        logger.warning("schedule_feed %s skipped — missing config: %s", post_id, ", ".join(missing))
        return "skipped", None

    snapshot = await _load_post_snapshot(post_id)
    if snapshot is None:
        return "skipped", None
    if snapshot.feed_creation_id:
        return "skipped", snapshot.feed_creation_id  # already scheduled remotely

    lead = (snapshot.scheduled_at - datetime.now(timezone.utc)).total_seconds()
    if lead < MIN_REMOTE_LEAD_SECONDS:
        logger.info("schedule_feed %s skipped — only %ds lead, need ≥%ds", post_id, lead, MIN_REMOTE_LEAD_SECONDS)
        return "skipped", None

    creation_id, api_error = await _call_graph_api(
        post_id, snapshot,
        scheduled_publish_time=int(snapshot.scheduled_at.timestamp()),
    )

    async with AsyncSessionLocal() as db:
        post = await db.get(InstagramPost, post_id)
        if not post:
            return "skipped", None
        if creation_id:
            post.feed_creation_id = creation_id
            post.error = None
            post.updated_at = datetime.now(timezone.utc)
            await db.commit()
            logger.info("schedule_feed %s → remote_scheduled (creation_id=%s)", post_id, creation_id)
            return "remote_scheduled", creation_id
        post.error = api_error or "Unknown error"
        post.updated_at = datetime.now(timezone.utc)
        await db.commit()
        return "failed", None


# ── Synchronous publish (post-now / short-lead fallback) ─────────────────────


async def publish_feed(post_id: uuid.UUID) -> tuple[PublishStatus, str | None]:
    """Publish a scheduled feed post immediately and return (status, media_id)."""
    missing = missing_config()
    if missing:
        logger.warning("publish_feed %s skipped — missing config: %s", post_id, ", ".join(missing))
        return "skipped", None

    snapshot = await _load_post_snapshot(post_id)
    if snapshot is None:
        return "skipped", None

    creation_id, api_error = await _call_graph_api(post_id, snapshot)

    media_id: str | None = None
    if creation_id:
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                logger.info("publish_feed %s — waiting for container %s…", post_id, creation_id)
                await wait_container_ready(client, creation_id)
                media_id = await publish_container(client, creation_id, "publish container")
        except Exception as exc:
            api_error = f"{type(exc).__name__}: {exc}"
            logger.error("publish_feed %s publish step failed: %s\n%s", post_id, exc, traceback.format_exc())
            media_id = None

    return await _finalize_post(post_id, snapshot, media_id, api_error)


# ── Internal ─────────────────────────────────────────────────────────────────


class _Snapshot:
    """Plain-Python snapshot of post + image filenames, decoupled from any DB session."""
    __slots__ = (
        "is_carousel", "caption", "primary_id", "all_ids",
        "filenames_by_id", "story_delay", "reel_delay", "companion_time",
        "scheduled_at", "feed_creation_id",
    )

    def __init__(self, post: InstagramPost, filenames_by_id: dict[uuid.UUID, str]):
        self.is_carousel    = bool(post.carousel_image_ids)
        self.caption        = post.caption or ""
        self.primary_id     = post.image_id
        self.all_ids        = [post.image_id] + list(post.carousel_image_ids or [])
        self.filenames_by_id = filenames_by_id
        self.story_delay    = post.story_delay_minutes
        self.reel_delay     = post.reel_delay_minutes
        self.companion_time = post.companion_time
        self.scheduled_at   = post.scheduled_at
        self.feed_creation_id = post.feed_creation_id


async def _load_post_snapshot(post_id: uuid.UUID) -> _Snapshot | None:
    async with AsyncSessionLocal() as db:
        post = await db.get(InstagramPost, post_id)
        if not post or post.status != "scheduled":
            return None
        all_ids = [post.image_id] + list(post.carousel_image_ids or [])
        img_result = await db.execute(select(Image).where(Image.id.in_(all_ids)))
        filenames = {img.id: img.filename for img in img_result.scalars().all()}
        return _Snapshot(post, filenames)


async def _call_graph_api(
    post_id: uuid.UUID,
    snap: _Snapshot,
    *,
    scheduled_publish_time: int | None = None,
) -> tuple[str | None, str | None]:
    """
    Create a media container and return (creation_id, api_error).

    When `scheduled_publish_time` is provided, the container is created with
    that field set and the caller MUST NOT call /media_publish — Instagram
    schedules the publish itself.
    """
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            builder = _create_carousel_container if snap.is_carousel else _create_single_container
            container_id = await builder(
                client, post_id, snap,
                scheduled_publish_time=scheduled_publish_time,
            )
            return container_id, None
    except Exception as exc:
        api_error = f"{type(exc).__name__}: {exc}"
        logger.error("Graph API container creation %s failed: %s\n%s", post_id, exc, traceback.format_exc())
        return None, api_error


async def _create_single_container(
    client: httpx.AsyncClient, post_id: uuid.UUID, snap: _Snapshot,
    *, scheduled_publish_time: int | None = None,
) -> str:
    fname = snap.filenames_by_id.get(snap.primary_id)
    if not fname:
        raise ValueError(f"Primary image {snap.primary_id} not found in DB")
    data: dict[str, str] = {"image_url": share_url(fname), "caption": snap.caption}
    if scheduled_publish_time is not None:
        data["scheduled_publish_time"] = str(scheduled_publish_time)
    return await create_media_container(client, data, "create single container")


async def _create_carousel_container(
    client: httpx.AsyncClient, post_id: uuid.UUID, snap: _Snapshot,
    *, scheduled_publish_time: int | None = None,
) -> str:
    child_ids: list[str] = []
    for img_id in snap.all_ids:
        fname = snap.filenames_by_id.get(img_id)
        if not fname:
            raise ValueError(f"Carousel image {img_id} not found in DB")
        item_id = await create_media_container(
            client,
            {"image_url": share_url(fname), "is_carousel_item": "true"},
            f"create carousel item {fname}",
        )
        child_ids.append(item_id)
        logger.info("publish_feed %s — waiting for item %s (%s)…", post_id, item_id, fname)
        await wait_container_ready(client, item_id)

    data: dict[str, str] = {
        "media_type": "CAROUSEL",
        "children":   ",".join(child_ids),
        "caption":    snap.caption,
    }
    if scheduled_publish_time is not None:
        data["scheduled_publish_time"] = str(scheduled_publish_time)
    return await create_media_container(client, data, "create carousel container")


async def _finalize_post(
    post_id: uuid.UUID, snap: _Snapshot,
    media_id: str | None, api_error: str | None,
) -> tuple[PublishStatus, str | None]:
    async with AsyncSessionLocal() as db:
        post = await db.get(InstagramPost, post_id)
        if not post:
            return "skipped", None

        now = datetime.now(timezone.utc)
        if media_id:
            post.status             = "posted"
            post.instagram_media_id = media_id
            post.error              = None
            post.updated_at         = now

            if snap.reel_delay is not None:
                post.reel_status       = "pending"
                post.reel_scheduled_at = companion_at(now, snap.reel_delay, snap.companion_time)

            await db.commit()
            logger.info("publish_feed %s → posted (media_id=%s)", post_id, media_id)
            return "posted", media_id

        post.status     = "failed"
        post.error      = api_error or "Unknown error"
        post.updated_at = now
        await db.commit()
        return "failed", None
