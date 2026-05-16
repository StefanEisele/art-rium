"""
Background runner that drives one ImprovSession from `processing` to `done`.

Steps:
  1. Read the session row + the source Video row.
  2. Call mux_session() → three output MP4s in storage/videos/.
  3. Create three Video rows (synth, hands, pip) and link them on the session.
  4. Generate a first-frame thumbnail for each so the gallery shows previews.
  5. Flip status to `done` (or `failed` on any exception, with error string).

Spawned via asyncio.create_task() from the POST /api/improv/sessions handler.
"""
from __future__ import annotations

import logging
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

from core.config import settings
from core.db import AsyncSessionLocal
from core.models import ImprovSession, Video
from core.video_thumb import make_video_thumbnail
from services.improv.mux import mux_session

logger = logging.getLogger(__name__)


async def run_improv_session(session_id: uuid.UUID) -> None:
    try:
        await _run(session_id)
    except Exception as exc:
        logger.exception("Improv session %s crashed", session_id)
        await _mark_failed(session_id, f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-1500:]}")


async def _run(session_id: uuid.UUID) -> None:
    async with AsyncSessionLocal() as db:
        session = await db.get(ImprovSession, session_id)
        if not session:
            logger.error("Improv session %s vanished before runner picked it up", session_id)
            return
        source = await db.get(Video, session.source_video_id)
        if not source or not source.filepath:
            await _mark_failed_inline(db, session, "Source video not available")
            return

        recording_path = settings.improv_dir / session.recording_filename
        if not recording_path.exists():
            await _mark_failed_inline(db, session, f"Recording file missing: {session.recording_filename}")
            return

        session.status = "processing"
        session.error = None
        await db.commit()

        source_path = settings.storage_dir / source.filepath
        source_video_id = source.id
        source_filename = source.filename or "video"

    # ── ffmpeg pass (off the DB session) ────────────────────────────────────
    mix_synth_path, mix_hands_path, mix_pip_path = await mux_session(
        source_path,
        recording_path,
        settings.videos_dir,
        ffmpeg_path=settings.ffmpeg_path,
    )

    # ── Persist output Video rows + session refs ────────────────────────────
    async with AsyncSessionLocal() as db:
        synth = _make_video_row(mix_synth_path, source_filename, kind="synth")
        hands = _make_video_row(mix_hands_path, source_filename, kind="hands")
        pip   = _make_video_row(mix_pip_path,   source_filename, kind="pip")
        db.add_all([synth, hands, pip])
        await db.flush()

        # First-frame thumbnails for the gallery — generated after the rows exist
        # so the filenames are stable. Best-effort: failures only log.
        for v, src in ((synth, mix_synth_path), (hands, mix_hands_path), (pip, mix_pip_path)):
            await make_video_thumbnail(src, settings.videos_dir / f"{v.id}_thumb.jpg")

        session = await db.get(ImprovSession, session_id)
        if not session:
            logger.error("Improv session %s disappeared mid-run", session_id)
            return
        session.mix_synth_video_id = synth.id
        session.mix_hands_video_id = hands.id
        session.mix_pip_video_id   = pip.id
        session.status = "done"
        session.error = None
        session.completed_at = datetime.now(timezone.utc)
        await db.commit()
        logger.info(
            "Improv session %s → done (source=%s, synth=%s, hands=%s, pip=%s)",
            session_id, source_video_id, synth.id, hands.id, pip.id,
        )


_KIND_LABELS = {
    "synth": ("Improv mix (synth)", "improv_synth"),
    "hands": ("Improv mix (hands)", "improv_hands"),
    "pip":   ("Improv mix (PiP)",   "improv_pip"),
}


def _make_video_row(path: Path, source_filename: str, *, kind: str) -> Video:
    label, workflow = _KIND_LABELS[kind]
    return Video(
        id=uuid.uuid4(),
        filename=path.name,
        filepath=str(path.relative_to(settings.storage_dir)).replace("\\", "/"),
        prompt=f"{label} of {source_filename}",
        status="done",
        workflow=workflow,
    )


async def _mark_failed(session_id: uuid.UUID, message: str) -> None:
    async with AsyncSessionLocal() as db:
        session = await db.get(ImprovSession, session_id)
        if not session:
            return
        session.status = "failed"
        session.error = message[:2000]
        session.completed_at = datetime.now(timezone.utc)
        await db.commit()


async def _mark_failed_inline(db, session: ImprovSession, message: str) -> None:
    session.status = "failed"
    session.error = message[:2000]
    session.completed_at = datetime.now(timezone.utc)
    await db.commit()
    logger.error("Improv session %s failed: %s", session.id, message)
