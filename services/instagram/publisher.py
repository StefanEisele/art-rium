"""
Instagram feed publisher — Single Source of Truth for publishing a scheduled
post to Instagram via the Graph API.

Used by:
  - routers/instagram.py::post_now      (manual "post now" button)
  - workers/instagram_scheduler.py      (auto-poster background loop)
  - cli/publish_due.py                   (Phase 3 — Windows Task Scheduler entry)

Contract:
  publish_feed(post_id) -> (status, media_id)

  status is one of:
    "posted"   — feed published, companion Stories/Reels scheduled per delay
    "failed"   — Graph API rejected; post.error populated, post.status='failed'
    "skipped"  — post missing / already posted / cancelled / config missing

The function records the outcome on the InstagramPost row in-place. It raises
only on unrecoverable infrastructure problems (e.g. DB unreachable) — Graph API
errors are caught and recorded.
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
    check_response,
    missing_config,
    share_url,
    wait_container_ready,
)

logger = logging.getLogger(__name__)

PublishStatus = Literal["posted", "failed", "skipped"]


async def publish_feed(post_id: uuid.UUID) -> tuple[PublishStatus, str | None]:
    """Publish a scheduled feed post and return (status, media_id)."""
    missing = missing_config()
    if missing:
        logger.warning("publish_feed %s skipped — missing config: %s", post_id, ", ".join(missing))
        return "skipped", None

    snapshot = await _load_post_snapshot(post_id)
    if snapshot is None:
        return "skipped", None

    media_id, api_error = await _call_graph_api(post_id, snapshot)

    return await _finalize_post(post_id, snapshot, media_id, api_error)


# ── Internal ─────────────────────────────────────────────────────────────────


class _Snapshot:
    """Plain-Python snapshot of post + image filenames, decoupled from any DB session."""
    __slots__ = (
        "is_carousel", "caption", "primary_id", "all_ids",
        "filenames_by_id", "story_delay", "reel_delay", "companion_time",
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


async def _load_post_snapshot(post_id: uuid.UUID) -> _Snapshot | None:
    async with AsyncSessionLocal() as db:
        post = await db.get(InstagramPost, post_id)
        if not post or post.status != "scheduled":
            return None
        all_ids = [post.image_id] + list(post.carousel_image_ids or [])
        img_result = await db.execute(select(Image).where(Image.id.in_(all_ids)))
        filenames = {img.id: img.filename for img in img_result.scalars().all()}
        return _Snapshot(post, filenames)


async def _call_graph_api(post_id: uuid.UUID, snap: _Snapshot) -> tuple[str | None, str | None]:
    """Returns (media_id, api_error) — exactly one of the two is None."""
    graph = settings.instagram_graph_api_base
    uid   = settings.instagram_user_id
    token = settings.instagram_access_token

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            container_id = (
                await _create_carousel_container(client, graph, uid, token, post_id, snap)
                if snap.is_carousel
                else await _create_single_container(client, graph, uid, token, snap)
            )

            logger.info("publish_feed %s — waiting for container %s…", post_id, container_id)
            await wait_container_ready(client, container_id)

            r = await client.post(f"{graph}/{uid}/media_publish", data={
                "creation_id":  container_id,
                "access_token": token,
            })
            body = r.json()
            check_response(body, "publish container")
            return body["id"], None
    except Exception as exc:
        api_error = f"{type(exc).__name__}: {exc}"
        logger.error("publish_feed %s failed: %s\n%s", post_id, exc, traceback.format_exc())
        return None, api_error


async def _create_single_container(
    client: httpx.AsyncClient, graph: str, uid: str, token: str, snap: _Snapshot,
) -> str:
    fname = snap.filenames_by_id.get(snap.primary_id)
    if not fname:
        raise ValueError(f"Primary image {snap.primary_id} not found in DB")
    r = await client.post(f"{graph}/{uid}/media", data={
        "image_url":    share_url(fname),
        "caption":      snap.caption,
        "access_token": token,
    })
    body = r.json()
    check_response(body, "create single container")
    return body["id"]


async def _create_carousel_container(
    client: httpx.AsyncClient, graph: str, uid: str, token: str,
    post_id: uuid.UUID, snap: _Snapshot,
) -> str:
    child_ids: list[str] = []
    for img_id in snap.all_ids:
        fname = snap.filenames_by_id.get(img_id)
        if not fname:
            raise ValueError(f"Carousel image {img_id} not found in DB")
        r = await client.post(f"{graph}/{uid}/media", data={
            "image_url":        share_url(fname),
            "is_carousel_item": "true",
            "access_token":     token,
        })
        body = r.json()
        check_response(body, f"create carousel item {fname}")
        item_id = body["id"]
        child_ids.append(item_id)
        logger.info("publish_feed %s — waiting for item %s (%s)…", post_id, item_id, fname)
        await wait_container_ready(client, item_id)

    r2 = await client.post(f"{graph}/{uid}/media", data={
        "media_type":   "CAROUSEL",
        "children":     ",".join(child_ids),
        "caption":      snap.caption,
        "access_token": token,
    })
    body2 = r2.json()
    check_response(body2, "create carousel container")
    return body2["id"]


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

            if snap.story_delay is not None:
                post.story_status       = "pending"
                post.story_scheduled_at = companion_at(now, snap.story_delay, snap.companion_time)
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
