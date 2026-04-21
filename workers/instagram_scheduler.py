"""
Instagram auto-poster — background worker that publishes scheduled posts.

Runs every CHECK_INTERVAL seconds, finds posts with status='scheduled' and
scheduled_at <= now(), and publishes them via the Instagram Graph API.

On success: status → 'posted', instagram_media_id saved.
On failure:  status → 'failed' after MAX_ATTEMPTS retries; error logged.
"""
import asyncio
import logging
from datetime import datetime, timedelta, timezone

import httpx
from sqlalchemy import select

from core.config import settings
from core.db import AsyncSessionLocal
from core.models import Image, InstagramPost
from workers.instagram_companion import publish_stories, publish_reel

logger = logging.getLogger(__name__)

CHECK_INTERVAL = 60   # seconds between scans
MAX_ATTEMPTS   = 3    # how many times to try before marking 'failed'


def _companion_at(published_at: datetime, delay_minutes: int, companion_time: str | None) -> datetime:
    """Compute companion scheduled_at. Day+ delays snap to the HH:MM target time."""
    dt = published_at + timedelta(minutes=delay_minutes)
    if delay_minutes >= 1440 and companion_time:
        try:
            h, m = map(int, companion_time.split(':'))
            dt = dt.replace(hour=h, minute=m, second=0, microsecond=0)
        except (ValueError, AttributeError):
            pass
    return dt


class InstagramScheduler:
    async def run(self) -> None:
        logger.info("InstagramScheduler started — checking every %ds", CHECK_INTERVAL)
        while True:
            try:
                await self._tick()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("InstagramScheduler tick error: %s", exc)
            await asyncio.sleep(CHECK_INTERVAL)

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _tick(self) -> None:
        now = datetime.now(timezone.utc)

        # ── Feed posts ────────────────────────────────────────────────────────
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(InstagramPost)
                .where(InstagramPost.status == "scheduled")
                .where(InstagramPost.scheduled_at <= now)
            )
            due_feed = result.scalars().all()

        if due_feed:
            logger.info("InstagramScheduler: %d feed post(s) due", len(due_feed))
            for post in due_feed:
                try:
                    await self._publish(post.id)
                except Exception as exc:
                    logger.error("Failed to publish post %s: %s", post.id, exc)

        # ── Story companions ──────────────────────────────────────────────────
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(InstagramPost)
                .where(InstagramPost.story_status == "pending")
                .where(InstagramPost.story_scheduled_at <= now)
            )
            due_stories = result.scalars().all()
            for post in due_stories:
                post.story_status = "processing"
            if due_stories:
                await db.commit()

        for post in due_stories:
            logger.info("InstagramScheduler: launching story for post %s", post.id)
            asyncio.create_task(publish_stories(post.id))

        # ── Reel companions ───────────────────────────────────────────────────
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(InstagramPost)
                .where(InstagramPost.reel_status == "pending")
                .where(InstagramPost.reel_scheduled_at <= now)
            )
            due_reels = result.scalars().all()
            for post in due_reels:
                post.reel_status = "processing"
            if due_reels:
                await db.commit()

        for post in due_reels:
            logger.info("InstagramScheduler: launching reel for post %s", post.id)
            asyncio.create_task(publish_reel(post.id))

    async def _publish(self, post_id) -> None:
        missing = [k for k, v in {
            "INSTAGRAM_USER_ID":    settings.instagram_user_id,
            "INSTAGRAM_ACCESS_TOKEN": settings.instagram_access_token,
            "PUBLIC_BASE_URL":      settings.public_base_url,
        }.items() if not v]
        if missing:
            logger.warning(
                "InstagramScheduler: skipping post %s — missing config: %s",
                post_id, ", ".join(missing),
            )
            return

        graph = settings.instagram_graph_api_base
        uid   = settings.instagram_user_id
        token = settings.instagram_access_token

        def _image_url(filename: str) -> str:
            base = settings.public_base_url.rstrip("/")
            url  = f"{base}/share/image/{filename}"
            if settings.image_share_token:
                url += f"?token={settings.image_share_token}"
            return url

        async with AsyncSessionLocal() as db:
            post = await db.get(InstagramPost, post_id)
            if not post or post.status != "scheduled":
                return   # already handled or cancelled

            # Collect all image IDs (primary first, then carousel extras)
            all_ids = [post.image_id]
            if post.carousel_image_ids:
                all_ids.extend(post.carousel_image_ids)

            img_result = await db.execute(select(Image).where(Image.id.in_(all_ids)))
            images = {img.id: img for img in img_result.scalars().all()}

            is_carousel = bool(post.carousel_image_ids)
            caption     = post.caption or ""

            try:
                async with httpx.AsyncClient(timeout=30) as client:
                    if is_carousel:
                        container_id = await _create_carousel(
                            client, graph, uid, token, all_ids, images, _image_url, caption
                        )
                    else:
                        primary = images.get(post.image_id)
                        if not primary:
                            raise ValueError(f"Primary image {post.image_id} not found in DB")
                        container_id = await _create_single(
                            client, graph, uid, token, _image_url(primary.filename), caption
                        )

                    media_id = await _publish_container(client, graph, uid, token, container_id)

                published_at            = datetime.now(timezone.utc)
                post.status             = "posted"
                post.instagram_media_id = media_id
                post.updated_at         = published_at

                # Schedule companion posts according to per-post delay settings
                if post.story_delay_minutes is not None:
                    post.story_status       = "pending"
                    post.story_scheduled_at = _companion_at(published_at, post.story_delay_minutes, post.companion_time)
                if post.reel_delay_minutes is not None:
                    post.reel_status       = "pending"
                    post.reel_scheduled_at = _companion_at(published_at, post.reel_delay_minutes, post.companion_time)

                await db.commit()
                logger.info("Auto-posted %s → media_id=%s", post_id, media_id)

            except Exception as exc:
                # Mark as failed so we don't retry forever
                post.status     = "failed"
                post.updated_at = datetime.now(timezone.utc)
                await db.commit()
                logger.error("Auto-post %s failed: %s", post_id, exc)
                raise


# ── Graph API helpers (module-level, reusable) ────────────────────────────────

def _check(body: dict, context: str) -> None:
    if "error" in body:
        err = body["error"]
        raise RuntimeError(f"{context}: {err.get('message', body)}")


async def _create_single(client, graph, uid, token, image_url, caption) -> str:
    r = await client.post(f"{graph}/{uid}/media", data={
        "image_url":    image_url,
        "caption":      caption,
        "access_token": token,
    })
    body = r.json()
    _check(body, "create single container")
    return body["id"]


async def _create_carousel(client, graph, uid, token, all_ids, images, image_url_fn, caption) -> str:
    # Step 1 — per-image carousel item containers
    child_ids = []
    for img_id in all_ids:
        img = images.get(img_id)
        if not img:
            raise ValueError(f"Carousel image {img_id} not found in DB")
        r = await client.post(f"{graph}/{uid}/media", data={
            "image_url":        image_url_fn(img.filename),
            "is_carousel_item": "true",
            "access_token":     token,
        })
        body = r.json()
        _check(body, f"create carousel item {img.filename}")
        child_ids.append(body["id"])
        logger.debug("Carousel item container: %s (%s)", body["id"], img.filename)

    # Step 2 — carousel container
    r2 = await client.post(f"{graph}/{uid}/media", data={
        "media_type":   "CAROUSEL",
        "children":     ",".join(child_ids),
        "caption":      caption,
        "access_token": token,
    })
    body2 = r2.json()
    _check(body2, "create carousel container")
    return body2["id"]


async def _publish_container(client, graph, uid, token, container_id) -> str:
    r = await client.post(f"{graph}/{uid}/media_publish", data={
        "creation_id":  container_id,
        "access_token": token,
    })
    body = r.json()
    _check(body, "publish container")
    return body["id"]
