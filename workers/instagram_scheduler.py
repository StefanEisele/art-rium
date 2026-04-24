"""
Instagram auto-poster — background worker that publishes scheduled posts.

Runs every CHECK_INTERVAL seconds, finds posts with status='scheduled' and
scheduled_at <= now(), and publishes them via the Instagram Graph API.

On success: status → 'posted', instagram_media_id saved.
On failure:  status → 'failed', error message stored in post.error.
"""
import asyncio
import logging
import traceback
from datetime import datetime, timedelta, timezone

import httpx
from sqlalchemy import select

from core.config import settings
from core.db import AsyncSessionLocal
from core.models import Image, InstagramPost
from workers.instagram_companion import publish_stories, publish_reel

logger = logging.getLogger(__name__)

CHECK_INTERVAL = 60   # seconds between scans


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
            due_feed = [p.id for p in result.scalars().all()]  # only IDs, avoid detached ORM issues

        if due_feed:
            logger.info("InstagramScheduler: %d feed post(s) due", len(due_feed))
            for post_id in due_feed:
                try:
                    await self._publish(post_id)
                except Exception as exc:
                    logger.error("Failed to publish post %s: %s", post_id, exc)

        # ── Story companions ──────────────────────────────────────────────────
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(InstagramPost)
                .where(InstagramPost.story_status == "pending")
                .where(InstagramPost.story_scheduled_at <= now)
            )
            due_stories = result.scalars().all()
            due_story_ids = [p.id for p in due_stories]
            for post in due_stories:
                post.story_status = "processing"
            if due_stories:
                await db.commit()

        for post_id in due_story_ids:
            logger.info("InstagramScheduler: launching story for post %s", post_id)
            asyncio.create_task(publish_stories(post_id))

        # ── Reel companions ───────────────────────────────────────────────────
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(InstagramPost)
                .where(InstagramPost.reel_status == "pending")
                .where(InstagramPost.reel_scheduled_at <= now)
            )
            due_reels = result.scalars().all()
            due_reel_ids = [p.id for p in due_reels]
            for post in due_reels:
                post.reel_status = "processing"
            if due_reels:
                await db.commit()

        for post_id in due_reel_ids:
            logger.info("InstagramScheduler: launching reel for post %s", post_id)
            asyncio.create_task(publish_reel(post_id))

    async def _publish(self, post_id) -> None:
        missing = [k for k, v in {
            "INSTAGRAM_USER_ID":      settings.instagram_user_id,
            "INSTAGRAM_ACCESS_TOKEN": settings.instagram_access_token,
            "PUBLIC_BASE_URL":        settings.public_base_url,
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

        # ── Step 1: Load all needed data from DB into plain Python values ─────
        async with AsyncSessionLocal() as db:
            post = await db.get(InstagramPost, post_id)
            if not post or post.status != "scheduled":
                return   # already handled or cancelled

            all_ids = [post.image_id]
            if post.carousel_image_ids:
                all_ids.extend(post.carousel_image_ids)

            img_result = await db.execute(select(Image).where(Image.id.in_(all_ids)))
            filenames_by_id = {img.id: img.filename for img in img_result.scalars().all()}

            is_carousel    = bool(post.carousel_image_ids)
            caption        = post.caption or ""
            primary_id     = post.image_id
            story_delay    = post.story_delay_minutes
            reel_delay     = post.reel_delay_minutes
            companion_time = post.companion_time

        # ── Step 2: Call Instagram Graph API (no DB session open) ─────────────
        media_id: str | None = None
        api_error: str | None = None
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                if is_carousel:
                    child_ids: list[str] = []
                    for img_id in all_ids:
                        fname = filenames_by_id.get(img_id)
                        if not fname:
                            raise ValueError(f"Carousel image {img_id} not found in DB")
                        r = await client.post(f"{graph}/{uid}/media", data={
                            "image_url":        _image_url(fname),
                            "is_carousel_item": "true",
                            "access_token":     token,
                        })
                        body = r.json()
                        _check(body, f"create carousel item {fname}")
                        item_id = body["id"]
                        child_ids.append(item_id)
                        logger.info("Auto-post %s — waiting for item container %s (%s)…", post_id, item_id, fname)
                        await _wait_container_ready(client, graph, token, item_id)

                    r2 = await client.post(f"{graph}/{uid}/media", data={
                        "media_type":   "CAROUSEL",
                        "children":     ",".join(child_ids),
                        "caption":      caption,
                        "access_token": token,
                    })
                    body2 = r2.json()
                    _check(body2, "create carousel container")
                    container_id = body2["id"]
                else:
                    fname = filenames_by_id.get(primary_id)
                    if not fname:
                        raise ValueError(f"Primary image {primary_id} not found in DB")
                    r = await client.post(f"{graph}/{uid}/media", data={
                        "image_url":    _image_url(fname),
                        "caption":      caption,
                        "access_token": token,
                    })
                    body = r.json()
                    _check(body, "create single container")
                    container_id = body["id"]

                logger.info("Auto-post %s — waiting for container %s to finish processing…", post_id, container_id)
                await _wait_container_ready(client, graph, token, container_id)

                r_pub = await client.post(f"{graph}/{uid}/media_publish", data={
                    "creation_id":  container_id,
                    "access_token": token,
                })
                body_pub = r_pub.json()
                _check(body_pub, "publish container")
                media_id = body_pub["id"]

        except Exception as exc:
            api_error = f"{type(exc).__name__}: {exc}"
            logger.error("Auto-post %s failed: %s\n%s", post_id, exc, traceback.format_exc())

        # ── Step 3: Persist result in a fresh DB session ──────────────────────
        async with AsyncSessionLocal() as db:
            post = await db.get(InstagramPost, post_id)
            if not post:
                return

            if media_id:
                published_at            = datetime.now(timezone.utc)
                post.status             = "posted"
                post.instagram_media_id = media_id
                post.error              = None
                post.updated_at         = published_at

                if story_delay is not None:
                    post.story_status       = "pending"
                    post.story_scheduled_at = _companion_at(published_at, story_delay, companion_time)
                if reel_delay is not None:
                    post.reel_status       = "pending"
                    post.reel_scheduled_at = _companion_at(published_at, reel_delay, companion_time)

                await db.commit()
                logger.info("Auto-posted %s → media_id=%s", post_id, media_id)
            else:
                post.status     = "failed"
                post.error      = api_error or "Unknown error"
                post.updated_at = datetime.now(timezone.utc)
                await db.commit()
                raise RuntimeError(api_error or "No media_id returned from Instagram")


# ── Graph API helpers ─────────────────────────────────────────────────────────

def _check(body: dict, context: str) -> None:
    if "error" in body:
        err = body["error"]
        raise RuntimeError(f"{context}: {err.get('message', body)}")


async def _wait_container_ready(
    client: httpx.AsyncClient,
    graph: str,
    token: str,
    container_id: str,
    max_wait: int = 60,
    poll_interval: int = 3,
) -> None:
    """Poll until Instagram has finished processing a media container."""
    import asyncio as _asyncio
    deadline = _asyncio.get_event_loop().time() + max_wait
    while True:
        r = await client.get(
            f"{graph}/{container_id}",
            params={"fields": "status_code", "access_token": token},
        )
        body = r.json()
        status = body.get("status_code", "")
        logger.debug("Container %s status: %s", container_id, status)
        if status == "FINISHED":
            return
        if status == "ERROR":
            raise RuntimeError(f"Instagram container {container_id} failed processing: {body}")
        if _asyncio.get_event_loop().time() >= deadline:
            raise TimeoutError(
                f"Instagram container {container_id} not ready after {max_wait}s (status={status!r})"
            )
        await _asyncio.sleep(poll_interval)
