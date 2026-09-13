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

Cloud renders (Video.workflow == API_WORKFLOW) are excluded for the same
reason, and it matters more there: a MiniMax task keeps running — and keeps
being billed — while this process is down. Marking it failed here would strand
its budget reservation and throw away a clip that was paid for.
`services.video_api.queue.CloudVideoQueue.reconcile()` owns those rows.

`sweep_stuck_jobs` has a second caller now: core/job_control.py::cancel_all_jobs
runs it right after cancelling every job task, where the same "an in-flight row
must be an orphan" argument holds for the same reason — nothing is left running
that could still claim one. Hence the `reason` parameter.
"""
import json
import logging

from sqlalchemy import select

from core.config import settings
from core.db import AsyncSessionLocal
from core.models import (
    API_WORKFLOW,
    AUDIO_WORKFLOWS,
    ImprovSession,
    PostCompanion,
    Song,
    Video,
    VideoClip,
)

logger = logging.getLogger(__name__)

_INTERRUPTED_MSG = "Interrupted by server restart"


async def sweep_stuck_jobs(reason: str = _INTERRUPTED_MSG) -> int:
    """Mark every row still claiming to be in flight as failed; return the count.

    `reason` is what the row will say happened. The default is the startup
    case; core/job_control.py::cancel_all_jobs passes its own, because after it
    has cancelled every job task the same query finds the same kind of orphan
    for an entirely different reason, and a job the user stopped on purpose
    should not claim the server restarted.
    """
    async with AsyncSessionLocal() as db:
        n = 0

        result = await db.execute(
            select(Video).where(
                Video.status.in_(("generating", "assembling")),
                Video.workflow.is_distinct_from(API_WORKFLOW),
            )
        )
        for video in result.scalars():
            video.status = "failed"
            video.error = reason
            n += 1

        result = await db.execute(select(Song).where(Song.status == "generating"))
        for song in result.scalars():
            song.status = "failed"
            song.error = reason
            n += 1

        result = await db.execute(
            select(ImprovSession).where(ImprovSession.status.in_(("queued", "processing")))
        )
        for session in result.scalars():
            session.status = "failed"
            session.error = reason
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
            logger.warning(f"Job sweep ({reason}): marked {n} orphaned job(s) as failed")
        else:
            logger.info("Job sweep: no orphaned jobs found")
        return n


# routers/vace.py parks its two-stage AnimateLCM renders in status 'review'
# between the base pass and the hires pass. Kept here as a literal rather than
# imported, because a startup sweep importing a router would drag the whole
# request layer into process start.
LCM_REVIEW_WORKFLOW = "vace_control"


async def backfill_review_clips() -> None:
    """One-time adoption of legacy status='review' video jobs into the clip
    library.

    Before the clip library existed, multi-segment jobs parked in status
    'review' with their segments described by a sidecar meta.json. The review/
    assemble flow is gone; nothing would ever move those rows again. Import
    each job's segments as VideoClip rows (idempotent — skipped if the job
    already has clips) and mark the job 'done'.

    **'review' means something else now.** routers/vace.py parks an AnimateLCM
    job there on purpose, between its base pass and the hires pass, waiting for
    the user to look at the preview and decide. Those rows must survive a
    restart — being swept would turn "your preview is ready" into "your job
    failed" every time the server came back. Hence the workflow filter, and
    hence a missing sidecar now means "not mine, leave it alone" rather than
    "failed": this backfill is a one-time adoption that has already run, so
    there is nothing left for it to legitimately fail.
    """
    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(Video).where(
                Video.status == "review",
                Video.workflow.is_distinct_from(LCM_REVIEW_WORKFLOW),
            )
        )
        videos = list(result.scalars().all())
        if not videos:
            return

        n_jobs = n_clips = 0
        for video in videos:
            seg_dir = settings.videos_dir / "segments" / str(video.id)
            meta_path = seg_dir / "meta.json"
            if not meta_path.exists():
                continue        # not a legacy clip job — see the docstring
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
                        has_audio=(workflow in AUDIO_WORKFLOWS),
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
