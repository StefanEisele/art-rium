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

PostCompanion(kind='reel').status is the one exception worth calling out:
marking it "failed" isn't just cosmetic — the scheduler's fallback query
selects rows with reel status IN (pending, NULL, failed), so this also
re-queues the row for its next 60s tick instead of leaving it stuck forever.

outpost_status / outpost_reel_status are deliberately NOT swept here —
`services.instagram.outpost.sync_outpost_status()` already reconciles
those against the Pi's own /status/{id} on a recurring poll, so sweeping
them here would just fight that reconciliation.
"""
import json
import logging

from sqlalchemy import select

from core.config import settings
from core.db import AsyncSessionLocal
from core.models import ImprovSession, PostCompanion, Song, Video, VideoClip

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
            select(PostCompanion).where(
                PostCompanion.kind == "reel", PostCompanion.status == "processing"
            )
        )
        for companion in result.scalars():
            companion.status = "failed"
            n += 1

        if n:
            await db.commit()
            logger.warning(f"Startup sweep: marked {n} orphaned job(s) as failed")
        else:
            logger.info("Startup sweep: no orphaned jobs found")


async def backfill_review_clips() -> None:
    """One-time adoption of legacy status='review' video jobs into the clip
    library.

    Before the clip library existed, multi-segment jobs parked in status
    'review' with their segments described by a sidecar meta.json. The review/
    assemble flow is gone; nothing would ever move those rows again. Import
    each job's segments as VideoClip rows (idempotent — skipped if the job
    already has clips) and mark the job 'done'; if the sidecar or files are
    gone, mark it failed instead of leaving it stuck.
    """
    async with AsyncSessionLocal() as db:
        result = await db.execute(select(Video).where(Video.status == "review"))
        videos = list(result.scalars().all())
        if not videos:
            return

        n_jobs = n_clips = 0
        for video in videos:
            seg_dir = settings.videos_dir / "segments" / str(video.id)
            meta_path = seg_dir / "meta.json"
            if not meta_path.exists():
                video.status = "failed"
                video.error = "Legacy review job: segment metadata missing"
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception as exc:
                video.status = "failed"
                video.error = f"Legacy review job: unreadable meta.json ({exc})"[:1000]
                continue

            existing = await db.execute(
                select(VideoClip.id).where(VideoClip.video_id == video.id).limit(1)
            )
            if existing.first() is None:
                workflow = meta.get("workflow") or video.workflow or "i2v_multi"
                for s in meta.get("segments", []):
                    if not (seg_dir / s["filename"]).exists():
                        continue
                    db.add(VideoClip(
                        video_id=video.id,
                        idx=s["index"],
                        filename=s["filename"],
                        thumb=s["thumb"],
                        prompt=s.get("prompt") or None,
                        frame_count=s.get("frame_count"),
                        workflow=workflow,
                        width=meta.get("width"),
                        height=meta.get("height"),
                        fps=meta.get("fps"),
                        has_audio=(workflow == "ltx_i2v"),
                    ))
                    n_clips += 1
            video.status = "done"
            video.error = None
            n_jobs += 1

        await db.commit()
        logger.warning(
            "Startup backfill: adopted %d legacy review job(s) into the clip library (%d clip(s))",
            n_jobs, n_clips,
        )
