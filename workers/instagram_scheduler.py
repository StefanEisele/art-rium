"""
Instagram auto-poster — background worker that publishes scheduled posts.

Runs every CHECK_INTERVAL seconds, finds posts with status='scheduled' and
scheduled_at <= now(), and publishes them via the Instagram Graph API.

On success: status → 'posted', instagram_media_id saved.
On failure:  status → 'failed' after MAX_ATTEMPTS retries; error logged.
"""
import asyncio
import logging
from datetime import datetime, timezone

import httpx
from sqlalchemy import select

from core.config import settings
from core.db import AsyncSessionLocal
from core.models import Image, InstagramPost
from workers.instagram_companion import publish_companion_posts

logger = logging.getLogger(__name__)

CHECK_INTERVAL = 60   # seconds between scans
MAX_ATTEMPTS   = 3    # how many times to try before marking 'failed'


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

        async with AsyncSessionLocal() as db:
            stmt = (
                select(InstagramPost)
                .where(InstagramPost.status == "scheduled")
                .where(InstagramPost.scheduled_at <= now)
            )
            result = await db.execute(stmt)
            due = result.scalars().all()

        if not due:
            return

        logger.info("InstagramScheduler: %d post(s) due", len(due))

        for post in due:
            try:
                await self._publish(post.id)
            except Exception as exc:
                logger.error("Failed to publish post %s: %s", post.id, exc)

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

                post.status             = "posted"
                post.instagram_media_id = media_id
                post.updated_at         = datetime.now(timezone.utc)
                await db.commit()
                logger.info("Auto-posted %s → media_id=%s", post_id, media_id)

                # Companion posts run concurrently; errors are caught inside
                asyncio.create_task(publish_companion_posts(post_id))

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
