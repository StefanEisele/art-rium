"""
Instagram auto-poster — background worker.

When `scheduled_publish_time` is available (Meta whitelist), the router fires
schedule_feed/schedule_reel once at post-creation time. This loop does NOT
retry remote scheduling — repeated retries on a permanent failure (e.g.
"User must be on whitelist") would create dozens of orphan child containers.

This worker only:

  1. Falls back to immediate publish for posts whose scheduled_at has come
     and that never got a remote schedule (server-dependent path — this is
     also the path used by every account that isn't on the whitelist).
  2. Detects Instagram-side publication and flips status `scheduled` →
     `posted` so the UI reflects what Instagram actually did.
"""
import asyncio
import logging
from datetime import datetime, timedelta, timezone

import httpx
from sqlalchemy import select

from core.config import settings
from core.db import AsyncSessionLocal
from core.models import InstagramPost
from core.tasks import safe_create_task
from services.instagram.graph import missing_config
from services.instagram.publisher import publish_feed
from services.instagram.reel import reel_publish_time
from services.instagram import outpost as outpost_svc
from workers.instagram_companion import publish_reel

logger = logging.getLogger(__name__)

CHECK_INTERVAL = 60   # seconds between scans
PUBLISHED_GRACE = 300  # seconds after scheduled_at before we believe IG missed it


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

    async def _tick(self) -> None:
        now = datetime.now(timezone.utc)
        await self._fallback_immediate_feed(now)
        await self._fallback_immediate_reel(now)
        await self._sync_remote_publication_status(now)
        await outpost_svc.sync_outpost_status()

    # ── 1. Fallback immediate publish ────────────────────────────────────────

    async def _fallback_immediate_feed(self, now: datetime) -> None:
        """For local-dispatch posts that are due now and never got a remote schedule."""
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(InstagramPost.id)
                .where(InstagramPost.status == "scheduled")
                .where(InstagramPost.dispatch_target == "local")
                .where(InstagramPost.feed_creation_id.is_(None))
                .where(InstagramPost.scheduled_at <= now)
            )
            due_ids = [row[0] for row in result.all()]

        if not due_ids:
            return

        logger.info("InstagramScheduler: %d feed post(s) due (immediate fallback)", len(due_ids))
        for post_id in due_ids:
            try:
                status, media_id = await publish_feed(post_id)
                logger.info("Feed %s → %s (media_id=%s)", post_id, status, media_id)
            except Exception as exc:
                logger.error("Failed to publish post %s: %s", post_id, exc)

    async def _fallback_immediate_reel(self, now: datetime) -> None:
        """Local-dispatch reel companion is due, never remote-scheduled, feed has posted."""
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(InstagramPost)
                .where(InstagramPost.status == "posted")
                .where(InstagramPost.dispatch_target == "local")
                .where(InstagramPost.reel_delay_minutes.isnot(None))
                .where(InstagramPost.reel_creation_id.is_(None))
                .where(InstagramPost.reel_status.in_(("pending", None, "failed")))
            )
            candidates = result.scalars().all()
            due_ids: list = []
            for post in candidates:
                publish_at = post.reel_scheduled_at or reel_publish_time(post)
                if publish_at and publish_at <= now and post.reel_status != "processing":
                    post.reel_status = "processing"
                    due_ids.append(post.id)
            if due_ids:
                await db.commit()

        for post_id in due_ids:
            logger.info("InstagramScheduler: launching reel for post %s (immediate fallback)", post_id)
            safe_create_task(publish_reel(post_id), name=f"publish_reel:{post_id}")

    # ── 2. Detect Instagram-side publication ─────────────────────────────────

    async def _sync_remote_publication_status(self, now: datetime) -> None:
        """
        Flip status `scheduled` → `posted` once Instagram has actually
        published a remote-scheduled container. Best-effort: failures here
        don't matter — IG already did its job, we're just catching up the UI.
        """
        if missing_config():
            return
        cutoff = now - timedelta(seconds=PUBLISHED_GRACE)
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(InstagramPost)
                .where(InstagramPost.status == "scheduled")
                .where(InstagramPost.dispatch_target == "local")
                .where(InstagramPost.feed_creation_id.isnot(None))
                .where(InstagramPost.scheduled_at <= cutoff)
            )
            posts = result.scalars().all()

        if not posts:
            return

        token = settings.instagram_access_token
        graph = settings.instagram_graph_api_base

        async with httpx.AsyncClient(timeout=15) as client:
            for post in posts:
                try:
                    r = await client.get(
                        f"{graph}/{post.feed_creation_id}",
                        params={"fields": "status_code", "access_token": token},
                    )
                    body = r.json()
                    code = body.get("status_code", "")
                except Exception as exc:
                    logger.debug("Container %s status query failed: %s", post.feed_creation_id, exc)
                    continue

                if code != "PUBLISHED":
                    continue

                async with AsyncSessionLocal() as db:
                    fresh = await db.get(InstagramPost, post.id)
                    if fresh and fresh.status == "scheduled":
                        fresh.status              = "posted"
                        fresh.instagram_media_id  = fresh.instagram_media_id or fresh.feed_creation_id
                        fresh.error               = None
                        fresh.updated_at          = now
                        await db.commit()
                        logger.info("Synced post %s → posted (IG container %s)", post.id, post.feed_creation_id)
