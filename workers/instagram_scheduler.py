"""
Instagram auto-poster — background worker that triggers due publishes.

Runs every CHECK_INTERVAL seconds and dispatches three kinds of work:
  - Feed posts ready to publish        → services.instagram.publisher.publish_feed
  - Story companions ready             → workers.instagram_companion.publish_stories
  - Reel companions ready              → workers.instagram_companion.publish_reel

This worker contains no Graph API logic itself — it only finds due rows and
delegates. The same publishing functions are used by the manual /post-now
endpoint and the CLI Task-Scheduler entry (Phase 3), so behaviour stays
consistent across all three trigger sources.
"""
import asyncio
import logging
from datetime import datetime, timezone

from sqlalchemy import select

from core.db import AsyncSessionLocal
from core.models import InstagramPost
from services.instagram.publisher import publish_feed
from workers.instagram_companion import publish_stories, publish_reel

logger = logging.getLogger(__name__)

CHECK_INTERVAL = 60   # seconds between scans


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
        await self._dispatch_due_feeds(now)
        await self._dispatch_due_companions(
            now,
            status_field=InstagramPost.story_status,
            scheduled_field=InstagramPost.story_scheduled_at,
            attr="story_status",
            kind="story",
            runner=publish_stories,
        )
        await self._dispatch_due_companions(
            now,
            status_field=InstagramPost.reel_status,
            scheduled_field=InstagramPost.reel_scheduled_at,
            attr="reel_status",
            kind="reel",
            runner=publish_reel,
        )

    async def _dispatch_due_feeds(self, now: datetime) -> None:
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(InstagramPost.id)
                .where(InstagramPost.status == "scheduled")
                .where(InstagramPost.scheduled_at <= now)
            )
            due_ids = [row[0] for row in result.all()]

        if not due_ids:
            return

        logger.info("InstagramScheduler: %d feed post(s) due", len(due_ids))
        for post_id in due_ids:
            try:
                status, media_id = await publish_feed(post_id)
                logger.info("Feed %s → %s (media_id=%s)", post_id, status, media_id)
            except Exception as exc:
                logger.error("Failed to publish post %s: %s", post_id, exc)

    async def _dispatch_due_companions(
        self,
        now: datetime,
        *,
        status_field,
        scheduled_field,
        attr: str,
        kind: str,
        runner,
    ) -> None:
        """Reserve due 'pending' companions by flipping them to 'processing', then launch."""
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(InstagramPost)
                .where(status_field == "pending")
                .where(scheduled_field <= now)
            )
            due = result.scalars().all()
            due_ids = [p.id for p in due]
            for post in due:
                setattr(post, attr, "processing")
            if due:
                await db.commit()

        for post_id in due_ids:
            logger.info("InstagramScheduler: launching %s for post %s", kind, post_id)
            asyncio.create_task(runner(post_id))
