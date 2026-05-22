"""
Image / Video Titler — generate title suggestions directly via Ollama.

Synchronous POST: client sends an id, server prepares the VLM payload
(small JPG for image; N evenly-spaced frame JPGs for video), calls the
local VLM, and returns the parsed title list in the response.
"""
import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from core.auth import require_auth
from core.config import settings
from core.db import get_db
from core.imaging import prepare_jpg_for_web
from core.models import Image, Video
from core.video_thumb import extract_video_frames
from services.ollama.client import generate_titles, generate_video_titles

logger = logging.getLogger(__name__)
router = APIRouter(dependencies=[Depends(require_auth)])

_TITLER_MAX_EDGE = 512
_TITLER_JPG_QUALITY = 80
_TITLER_N = 5
# Frames per video — 3 samples (25%/50%/75%) give the VLM enough motion
# context without blowing up the VL token budget on qwen2.5vl:3b.
_VIDEO_FRAMES = 3
_VIDEO_FRAME_MAX_EDGE = 512


class TitlerRequest(BaseModel):
    image_id: str


class VideoTitlerRequest(BaseModel):
    video_id: str
    n_frames: int | None = None   # optional override (1..6); default _VIDEO_FRAMES


@router.post("/api/titler/run")
async def run_titler(
    req: TitlerRequest,
    db: AsyncSession = Depends(get_db),
):
    try:
        image_uuid = uuid.UUID(req.image_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid image_id")

    img = await db.get(Image, image_uuid)
    if not img:
        raise HTTPException(status_code=404, detail="Image not found")

    src = settings.storage_dir / img.filepath
    if not src.exists():
        raise HTTPException(status_code=404, detail="Image file not found on disk")

    jpg_bytes, _ = await prepare_jpg_for_web(
        src, max_edge=_TITLER_MAX_EDGE, quality=_TITLER_JPG_QUALITY,
    )
    logger.info(
        "Titler: image=%s, model=%s, payload=%dKB",
        img.id, settings.ollama_titler_model, len(jpg_bytes) // 1024,
    )

    try:
        titles = await generate_titles(jpg_bytes, n=_TITLER_N)
    except Exception as exc:
        logger.exception("Titler failed for image %s", img.id)
        raise HTTPException(status_code=502, detail=f"Titler failed: {exc}")

    if not titles:
        raise HTTPException(status_code=502, detail="Titler returned no titles")

    return {"titles": titles}


@router.post("/api/titler/run-video")
async def run_video_titler(
    req: VideoTitlerRequest,
    db: AsyncSession = Depends(get_db),
):
    try:
        video_uuid = uuid.UUID(req.video_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid video_id")

    video = await db.get(Video, video_uuid)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    if video.status != "done" or not video.filepath:
        raise HTTPException(status_code=400, detail="Video is not ready (status != done)")

    src = settings.storage_dir / video.filepath
    if not src.exists():
        raise HTTPException(status_code=404, detail="Video file not found on disk")

    count = req.n_frames if req.n_frames is not None else _VIDEO_FRAMES
    count = max(1, min(6, count))

    frames = await extract_video_frames(src, count=count, max_edge=_VIDEO_FRAME_MAX_EDGE)
    if not frames:
        raise HTTPException(status_code=502, detail="Could not extract sample frames")

    logger.info(
        "Video titler: video=%s, model=%s, frames=%d, total=%dKB",
        video.id, settings.ollama_titler_model, len(frames),
        sum(len(f) for f in frames) // 1024,
    )

    try:
        titles = await generate_video_titles(frames, n=_TITLER_N)
    except Exception as exc:
        logger.exception("Video titler failed for video %s", video.id)
        raise HTTPException(status_code=502, detail=f"Titler failed: {exc}")

    if not titles:
        raise HTTPException(status_code=502, detail="Titler returned no titles")

    return {"titles": titles, "frames_used": len(frames)}
