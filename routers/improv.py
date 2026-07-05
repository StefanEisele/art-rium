"""
Piano-improvisation tool.

Workflow:
  1. User picks a generated source video on /tools/improv/.
  2. Frontend shows a public share URL (with image_share_token) so the user
     can open the source video on a second device next to the piano.
  3. User records on iPhone (Blackmagic Camera + Focusrite Scarlett 2i4),
     uploads the resulting MP4 here.
  4. ffmpeg mux produces two outputs (synth + hands) — see services/improv/mux.

POST   /api/improv/sessions            multipart (source_video_id, recording)
GET    /api/improv/sessions/{id}       poll
GET    /api/improv/sessions            list (newest first)
DELETE /api/improv/sessions/{id}       remove session + output files (best-effort)
"""
from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.auth import require_auth
from core.config import settings
from core.db import get_db
from core.models import ImprovSession, Video
from core.tasks import safe_create_task
from services.improv.runner import run_improv_session
from services.instagram.graph import share_url

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/improv", dependencies=[Depends(require_auth)])

# Hard cap on uploaded recordings to prevent runaway uploads. iPhone 4K30 ~2 min
# is on the order of 500 MB; 1 GB is comfortable headroom for short snippets.
_MAX_UPLOAD_BYTES = 1024 * 1024 * 1024


# ── Endpoints ────────────────────────────────────────────────────────────────


_VALID_PIP_CORNERS = {"tr", "br", "tl", "bl"}
_PIP_WIDTH_MIN = 0.10
_PIP_WIDTH_MAX = 0.50


@router.post("/sessions", status_code=202)
async def create_session(
    source_video_id: uuid.UUID = Form(...),
    recording: UploadFile = File(...),
    pip_corner: str = Form("tr"),
    pip_width_pct: float = Form(0.24),
    db: AsyncSession = Depends(get_db),
):
    source = await db.get(Video, source_video_id)
    if not source or source.status != "done" or not source.filepath:
        raise HTTPException(404, f"Source video {source_video_id} not found or not ready")

    if not (recording.content_type or "").startswith("video/"):
        raise HTTPException(400, f"Expected video/* upload, got {recording.content_type!r}")

    corner = pip_corner if pip_corner in _VALID_PIP_CORNERS else "tr"
    width_pct = max(_PIP_WIDTH_MIN, min(_PIP_WIDTH_MAX, pip_width_pct))

    settings.improv_dir.mkdir(parents=True, exist_ok=True)
    session_id = uuid.uuid4()
    suffix = _safe_suffix(recording.filename)
    rec_name = f"recording_{session_id.hex}{suffix}"
    rec_path = settings.improv_dir / rec_name

    written = 0
    try:
        with rec_path.open("wb") as f:
            while True:
                chunk = await recording.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > _MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "Recording exceeds 1 GB upload cap")
                f.write(chunk)
    except HTTPException:
        rec_path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        rec_path.unlink(missing_ok=True)
        raise HTTPException(500, f"Upload failed: {exc}") from exc

    session = ImprovSession(
        id=session_id,
        source_video_id=source_video_id,
        recording_filename=rec_name,
        status="queued",
    )
    db.add(session)
    await db.commit()

    safe_create_task(
        run_improv_session(session_id, pip_corner=corner, pip_width_pct=width_pct),
        name=f"improv_session:{session_id}",
    )
    logger.info(
        "Improv session %s queued — source=%s, recording=%s (%.1f MB), pip_corner=%s, pip_width=%.2f",
        session_id, source_video_id, rec_name, written / 1_048_576, corner, width_pct,
    )
    return {"id": str(session_id), "status": "queued"}


@router.get("/share-url/{video_id}")
async def get_share_url(
    video_id: uuid.UUID,
    countdown: int = 4,
    tick_every: int = 24,
    accent_every: int = 4,
    loop_bars: int = 0,
    db: AsyncSession = Depends(get_db),
):
    """Public URL for the source video — points at the loop player so the
    second device (iPad next to the piano) replays it indefinitely.

    Loop-player parameters are appended as query params so the QR scan
    boots the iPad straight into the desired countdown/metronome/bar config.
    The iPad can still tap the cog to live-edit; those edits persist in
    its localStorage and override the URL on subsequent scans.
    """
    video = await db.get(Video, video_id)
    if not video or not video.filename:
        raise HTTPException(404, "Video not found")

    fps = video.fps or 24
    base = share_url(video.filename, kind="video-loop")
    extra = {
        "fps": fps,
        "countdown": max(0, min(10, countdown)),
        "tick_every": max(1, tick_every),
        "accent_every": max(0, accent_every),
        "loop_bars": max(0, loop_bars),
    }
    sep = "&" if "?" in base else "?"
    qs = "&".join(f"{k}={v}" for k, v in extra.items())
    return {"url": f"{base}{sep}{qs}", "fps": fps, "duration": video.frame_count}


@router.get("/sessions/{session_id}")
async def get_session(session_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    session = await db.get(ImprovSession, session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    return await _serialize(session, db)


@router.get("/sessions")
async def list_sessions(limit: int = 20, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(ImprovSession).order_by(desc(ImprovSession.created_at)).limit(limit)
    )
    sessions = result.scalars().all()
    return [await _serialize(s, db) for s in sessions]


@router.delete("/sessions/{session_id}", status_code=204)
async def delete_session(session_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    session = await db.get(ImprovSession, session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    for video_id in (
        session.mix_synth_video_id,
        session.mix_hands_video_id,
        session.mix_pip_video_id,
    ):
        if not video_id:
            continue
        video = await db.get(Video, video_id)
        if video and video.filepath:
            (settings.storage_dir / video.filepath).unlink(missing_ok=True)
            (settings.videos_dir / f"{video.id}_thumb.jpg").unlink(missing_ok=True)
            await db.delete(video)

    (settings.improv_dir / session.recording_filename).unlink(missing_ok=True)
    await db.delete(session)
    await db.commit()


# ── Serialisation ────────────────────────────────────────────────────────────


async def _serialize(session: ImprovSession, db: AsyncSession) -> dict:
    source = await db.get(Video, session.source_video_id)
    synth = await db.get(Video, session.mix_synth_video_id) if session.mix_synth_video_id else None
    hands = await db.get(Video, session.mix_hands_video_id) if session.mix_hands_video_id else None
    pip   = await db.get(Video, session.mix_pip_video_id)   if session.mix_pip_video_id   else None
    return {
        "id":            str(session.id),
        "status":        session.status,
        "error":         session.error,
        "created_at":    session.created_at.isoformat() if session.created_at else None,
        "completed_at":  session.completed_at.isoformat() if session.completed_at else None,
        "source_video":  _video_summary(source),
        "mix_synth":     _video_summary(synth),
        "mix_hands":     _video_summary(hands),
        "mix_pip":       _video_summary(pip),
    }


def _video_summary(video: Video | None) -> dict | None:
    if not video:
        return None
    return {
        "id":        str(video.id),
        "filename":  video.filename,
        "title":     video.title,
        "prompt":    video.prompt,
        "status":    video.status,
        "thumb_url": f"/api/video/thumb/{video.id}" if video.status == "done" else None,
    }


def _safe_suffix(name: str | None) -> str:
    if not name:
        return ".mp4"
    suffix = ""
    for ext in (".mp4", ".mov", ".m4v"):
        if name.lower().endswith(ext):
            suffix = ext
            break
    return suffix or ".mp4"
