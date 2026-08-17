"""The cloud render queue — submit, poll, download, settle.

**All queue state lives in the database, not in this process.** A job is
"waiting" because its Video row says `status='queued'`, and "in flight" because
it has an `api_task_id`. That is not an aesthetic choice: a MiniMax task keeps
running when art-rium restarts, so any in-memory queue would lose track of work
that is still being billed. Driving off the rows means a restart picks the same
jobs back up, and `reconcile()` is a short function rather than a rebuild.

The failure this design exists to prevent, spelled out: app restarts mid-job,
the polling loops are gone, the reservations sit at 'reserved' forever. After a
few restarts the budget reads "full" without a cent having been spent.

Concurrency is capped at `video_api_max_concurrent` (2) because that is what
pay-as-you-go allows; beyond it submissions simply fail.
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
from sqlalchemy import select

from core.config import settings
from core.db import AsyncSessionLocal
from core.models import API_WORKFLOW, LedgerEntry, Video
from core.video_thumb import (
    make_video_thumbnail,
    probe_video_dimensions,
    probe_video_duration,
)
from services.video_api import budget
from services.video_api.backend import (
    TASK_TIMEOUT_SECONDS,
    MiniMaxBackend,
    VideoBackend,
    VideoBackendError,
    VideoRequest,
    poll_delay,
)

logger = logging.getLogger(__name__)

TICK_SECONDS = 3
DOWNLOAD_TIMEOUT = 600.0

# Per-video polling bookkeeping, rebuilt from the DB on restart.
_poll_state: dict[uuid.UUID, dict] = {}
# What each queued job should be submitted with. Requests carry file paths and
# a prompt, which the Video row cannot fully express; a row whose payload is
# missing after a restart is submitted from the row's own fields instead.
_pending_requests: dict[uuid.UUID, VideoRequest] = {}


def remember_request(video_id: uuid.UUID, request: VideoRequest) -> None:
    _pending_requests[video_id] = request


def _request_from_row(video: Video) -> VideoRequest:
    """Rebuild a submittable request from the row alone.

    Used when a restart lost the in-memory payload. Attachments cannot be
    recovered this way, so a job that had them is failed rather than silently
    re-submitted as a plain text-to-video — which would bill for the wrong
    thing and quietly produce a different clip.
    """
    return VideoRequest(
        prompt=video.prompt or "",
        duration_s=video.duration_s or 6,
        resolution=video.api_resolution or "2K",
        ratio=video.api_ratio or "adaptive",
    )


class CloudVideoQueue:
    """Single background loop driving every cloud render."""

    def __init__(self, backend: VideoBackend | None = None):
        self.backend = backend or MiniMaxBackend()
        self._stop = asyncio.Event()

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def run(self) -> None:
        await self.reconcile()
        while not self._stop.is_set():
            try:
                await self.tick()
            except Exception:
                logger.exception("Cloud video queue tick failed")
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=TICK_SECONDS)
            except asyncio.TimeoutError:
                pass

    def stop(self) -> None:
        self._stop.set()

    # ── Reconciliation ───────────────────────────────────────────────────────

    async def reconcile(self) -> dict:
        """Bring the ledger back in step with reality after a restart.

        Three cases, in the order they matter:
          1. reserved with no task id, older than five minutes → the submit
             never landed; release it.
          2. a job still marked in-flight → poll it once; settle, release, or
             resume polling depending on what the provider says.
          3. a job left 'queued' → it was never submitted, so it is simply
             picked up again by the normal tick.
        """
        released = 0
        async with AsyncSessionLocal() as db:
            released = await budget.release_stranded(db)

            in_flight = (await db.execute(
                select(Video).where(
                    Video.workflow == API_WORKFLOW,
                    Video.status == "generating",
                    Video.api_task_id.is_not(None),
                )
            )).scalars().all()
            adopted = [v.id for v in in_flight]
            queued = len((await db.execute(
                select(Video.id).where(
                    Video.workflow == API_WORKFLOW, Video.status == "queued",
                )
            )).scalars().all())

        for video_id in adopted:
            _poll_state[video_id] = {"attempt": 0, "next_at": datetime.now(timezone.utc)}

        if adopted or released or queued:
            logger.warning(
                "Cloud queue reconciled: %d in flight adopted, %d stranded reservation(s) "
                "released, %d still queued",
                len(adopted), released, queued,
            )
        return {"adopted": len(adopted), "released": released, "queued": queued}

    # ── One pass ─────────────────────────────────────────────────────────────

    async def tick(self) -> None:
        await self._submit_ready()
        await self._poll_in_flight()

    async def _in_flight_count(self, db) -> int:
        rows = (await db.execute(
            select(Video.id).where(
                Video.workflow == API_WORKFLOW,
                Video.status == "generating",
                Video.api_task_id.is_not(None),
            )
        )).scalars().all()
        return len(rows)

    async def _submit_ready(self) -> None:
        """Submit as many queued jobs as the concurrency cap allows."""
        async with AsyncSessionLocal() as db:
            free = settings.video_api_max_concurrent - await self._in_flight_count(db)
            if free <= 0:
                return
            waiting = (await db.execute(
                select(Video)
                .where(Video.workflow == API_WORKFLOW, Video.status == "queued")
                .order_by(Video.created_at)
                .limit(free)
            )).scalars().all()
            ids = [v.id for v in waiting]

        for video_id in ids:
            await self._submit_one(video_id)

    async def _submit_one(self, video_id: uuid.UUID) -> None:
        async with AsyncSessionLocal() as db:
            video = await db.get(Video, video_id)
            if not video or video.status != "queued":
                return
            entry = (await db.execute(
                select(LedgerEntry).where(LedgerEntry.video_id == video_id)
                .order_by(LedgerEntry.created_at.desc()).limit(1)
            )).scalar_one_or_none()
            request = _pending_requests.get(video_id)
            source_task_id = video.api_task_id  # unset here; regeneration sets it below
            is_regeneration = bool(video.source_video_id)
            if is_regeneration:
                source = await db.get(Video, video.source_video_id)
                source_task_id = source.api_task_id if source else None
            had_attachments = bool(video.image_ids)

        if entry is None:
            await self._fail(video_id, "No budget reservation found for this job", None)
            return

        try:
            if is_regeneration:
                if not source_task_id:
                    raise VideoBackendError(
                        "The source take has no MiniMax task id — only clips generated "
                        "through this tool can be pulled up to 2K.",
                        kind="invalid_request",
                    )
                task_id = await self.backend.submit_regeneration(source_task_id=source_task_id)
            else:
                if request is None:
                    if had_attachments:
                        raise VideoBackendError(
                            "Job was interrupted before it was submitted and its "
                            "attachments are no longer in memory — start it again.",
                            kind="lost_payload",
                        )
                    request = _request_from_row(video)
                task_id = await self.backend.submit(request)
        except VideoBackendError as exc:
            # Nothing was submitted, so nothing will be billed: give the money
            # straight back rather than waiting for a timeout to do it.
            await self._fail(video_id, str(exc), entry.id)
            return
        except (httpx.HTTPError, OSError) as exc:
            await self._fail(video_id, f"Could not reach MiniMax: {exc}", entry.id)
            return

        async with AsyncSessionLocal() as db:
            video = await db.get(Video, video_id)
            if video:
                video.api_task_id = task_id
                video.status = "generating"
                await db.commit()
            await budget.mark_submitted(db, entry.id, task_id)

        _pending_requests.pop(video_id, None)
        _poll_state[video_id] = {"attempt": 0, "next_at": datetime.now(timezone.utc)}
        logger.info("Cloud job %s submitted as task %s", video_id, task_id)

    async def _poll_in_flight(self) -> None:
        now = datetime.now(timezone.utc)
        async with AsyncSessionLocal() as db:
            videos = (await db.execute(
                select(Video).where(
                    Video.workflow == API_WORKFLOW,
                    Video.status == "generating",
                    Video.api_task_id.is_not(None),
                )
            )).scalars().all()
            jobs = [(v.id, v.api_task_id, v.created_at) for v in videos]

        for video_id, task_id, created_at in jobs:
            state = _poll_state.setdefault(video_id, {"attempt": 0, "next_at": now})
            if state["next_at"] > now:
                continue
            await self._poll_one(video_id, task_id, created_at, now)

    async def _poll_one(
        self, video_id: uuid.UUID, task_id: str, created_at: datetime, now: datetime,
    ) -> None:
        state = _poll_state.setdefault(video_id, {"attempt": 0, "next_at": now})
        state["attempt"] += 1
        state["next_at"] = now + timedelta(seconds=poll_delay(state["attempt"]))

        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        if now - created_at > timedelta(seconds=TASK_TIMEOUT_SECONDS):
            await self._finish_failed(
                video_id, f"MiniMax task {task_id} did not finish within 20 minutes",
            )
            return

        try:
            task = await self.backend.poll(task_id)
        except VideoBackendError as exc:
            if exc.retryable:
                return          # transient — the next tick tries again
            await self._finish_failed(video_id, str(exc))
            return
        except (httpx.HTTPError, OSError) as exc:
            logger.warning("Poll of task %s failed transiently: %s", task_id, exc)
            return

        if not task.done:
            return
        if task.status == "succeeded" and task.output_url:
            await self._finish_succeeded(video_id, task)
        elif task.status == "succeeded":
            await self._finish_failed(video_id, "MiniMax reported success without a video URL")
        elif task.status == "cancelled":
            await self._finish_failed(video_id, "MiniMax cancelled the task", billed=False)
        else:
            await self._finish_failed(video_id, task.error or "MiniMax task failed", billed=False)

    # ── Terminal transitions ─────────────────────────────────────────────────

    async def _finish_succeeded(self, video_id: uuid.UUID, task) -> None:
        dest = settings.videos_dir / f"{video_id}_h3.mp4"
        try:
            await self._download(task.output_url, dest)
        except Exception as exc:
            # The task itself succeeded and *is* billed, so the reservation is
            # settled even though the download failed. Recording it as free
            # would be the wrong kind of optimism.
            logger.exception("Download of finished task for %s failed", video_id)
            await self._settle_entry(video_id, task)
            await self._mark_failed_row(
                video_id, f"MiniMax finished but the download failed: {exc}",
            )
            return

        width, height = await probe_video_dimensions(dest)
        duration = await probe_video_duration(dest)
        await make_video_thumbnail(dest, settings.videos_dir / f"{video_id}_thumb.jpg")

        async with AsyncSessionLocal() as db:
            video = await db.get(Video, video_id)
            if video:
                video.filename = dest.name
                video.filepath = str(dest.relative_to(settings.storage_dir))
                video.width = width
                video.height = height
                video.fps = 24
                video.frame_count = int(duration * 24) if duration else None
                video.status = "done"
                video.error = None
                await db.commit()

        await self._settle_entry(video_id, task)
        _poll_state.pop(video_id, None)
        logger.info("Cloud job %s finished: %s (%sx%s)", video_id, dest.name, width, height)

    async def _settle_entry(self, video_id: uuid.UUID, task) -> None:
        async with AsyncSessionLocal() as db:
            entry = (await db.execute(
                select(LedgerEntry).where(
                    LedgerEntry.video_id == video_id, LedgerEntry.state == "reserved",
                ).order_by(LedgerEntry.created_at.desc()).limit(1)
            )).scalar_one_or_none()
            if entry:
                await budget.settle(
                    db, entry.id,
                    billed_seconds=task.billed_seconds,
                    billed_image_count=task.billed_image_count,
                    billed_input_video_seconds=task.billed_input_video_seconds,
                )

    async def _finish_failed(self, video_id: uuid.UUID, message: str, *, billed: bool = False) -> None:
        """A task that ended badly. `billed=False` means the money comes back."""
        await self._mark_failed_row(video_id, message)
        if not billed:
            async with AsyncSessionLocal() as db:
                entry = (await db.execute(
                    select(LedgerEntry).where(
                        LedgerEntry.video_id == video_id, LedgerEntry.state == "reserved",
                    ).order_by(LedgerEntry.created_at.desc()).limit(1)
                )).scalar_one_or_none()
                if entry:
                    await budget.release(db, entry.id, note=message)
        _poll_state.pop(video_id, None)
        logger.warning("Cloud job %s failed: %s", video_id, message)

    async def _mark_failed_row(self, video_id: uuid.UUID, message: str) -> None:
        async with AsyncSessionLocal() as db:
            video = await db.get(Video, video_id)
            if video:
                video.status = "failed"
                video.error = message[:1000]
                await db.commit()

    async def _fail(self, video_id: uuid.UUID, message: str, entry_id: uuid.UUID | None) -> None:
        """Submit-time failure: nothing was sent, so nothing is owed."""
        await self._mark_failed_row(video_id, message)
        if entry_id:
            async with AsyncSessionLocal() as db:
                await budget.release(db, entry_id, note=message)
        _pending_requests.pop(video_id, None)
        logger.warning("Cloud job %s could not be submitted: %s", video_id, message)

    async def _download(self, url: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        async with httpx.AsyncClient(timeout=DOWNLOAD_TIMEOUT, follow_redirects=True) as client:
            async with client.stream("GET", url) as response:
                response.raise_for_status()
                with tmp.open("wb") as fh:
                    async for chunk in response.aiter_bytes(1024 * 256):
                        fh.write(chunk)
        tmp.replace(dest)
