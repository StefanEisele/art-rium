"""
Startup sweep for jobs orphaned by a server restart.

Generation/publish jobs run as fire-and-forget background tasks
(see core.tasks.safe_create_task) with no persistence: if the process
dies mid-job, the DB row is left sitting in whatever in-flight status the
task set right before spawning (e.g. Video.status="generating"). Nothing
else ever moves it out of that status, so — from the UI's point of view —
the job just hangs forever.

At startup, any row still showing an in-flight status unambiguously
belongs to the *previous* process — a fresh process cannot have anything
genuinely in progress yet. So no age/threshold check is needed here; every
match is real. Sweep runs once, before the scheduler/listener start, and
marks each orphan "failed" with an explanatory error message.

InstagramPost.reel_status is the one exception worth calling out: marking
it "failed" isn't just cosmetic — the scheduler's fallback query selects
`reel_status IN (pending, NULL, failed)`, so this also re-queues the row
for its next 60s tick instead of leaving it stuck forever.

outpost_status / outpost_reel_status are deliberately NOT swept here —
`services.instagram.outpost.sync_outpost_status()` already reconciles
those against the Pi's own /status/{id} on a recurring poll, so sweeping
them here would just fight that reconciliation.
"""
import logging

from sqlalchemy import select

from core.db import AsyncSessionLocal
from core.models import ImprovSession, InstagramPost, Song, Video

logger = logging.getLogger(__name__)

_INTERRUPTED_MSG = "Interrupted by server restart"


async def sweep_stuck_jobs() -> None:
    async with AsyncSessionLocal() as db:
        n = 0

        result = await db.execute(
            select(Video).where(Video.status.in_(("generating", "assembling")))
        )
        for video in result.scalars():
            video.status = "failed"
            video.error = _INTERRUPTED_MSG
            n += 1

        result = await db.execute(select(Song).where(Song.status == "generating"))
        for song in result.scalars():
            song.status = "failed"
            song.error = _INTERRUPTED_MSG
            n += 1

        result = await db.execute(
            select(ImprovSession).where(ImprovSession.status.in_(("queued", "processing")))
        )
        for session in result.scalars():
            session.status = "failed"
            session.error = _INTERRUPTED_MSG
            n += 1

        result = await db.execute(
            select(InstagramPost).where(InstagramPost.reel_status == "processing")
        )
        for post in result.scalars():
            post.reel_status = "failed"
            n += 1

        if n:
            await db.commit()
            logger.warning(f"Startup sweep: marked {n} orphaned job(s) as failed")
        else:
            logger.info("Startup sweep: no orphaned jobs found")
