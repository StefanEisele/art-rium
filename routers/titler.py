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
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.auth import require_auth
from core.config import settings
from core.db import get_db
from core.imaging import prepare_jpg_for_web
from core.models import Image, ImageSeries, ImageSeriesItem, Video
from core.video_thumb import extract_video_frames
from services.ollama.analysis import (
    generate_series_titles,
    generate_titles,
    generate_video_titles,
)

logger = logging.getLogger(__name__)
router = APIRouter(dependencies=[Depends(require_auth)])

_TITLER_MAX_EDGE = 512
_TITLER_JPG_QUALITY = 80
_TITLER_N = 5
# Frames per video — 3 samples (25%/50%/75%) give the VLM enough motion
# context without blowing up the VL token budget on qwen2.5vl:3b.
_VIDEO_FRAMES = 3
_VIDEO_FRAME_MAX_EDGE = 512
# Members sampled from a series, for the same reason the video titler caps
# its frames at 3: qwen2.5vl:3b runs against a 16k context, and a 20-picture
# series at 512 px does not fit in it. Four is enough to show what the set
# has in common — first, last, and two from the middle.
_SERIES_IMAGES = 4


async def _titles_with_retry(generate, subject: str) -> list[str]:
    """Call the VLM, and once more if it answered with an empty list.

    Observed on qwen2.5vl:3b at temperature 0.8: a valid `{"titles": []}` comes
    back occasionally, on an image that titles perfectly well a second later.
    It is a sampling roll, not a property of the picture, and a warm call costs
    ~1.5 s — so retrying here beats handing the user an error to click through.
    """
    for attempt in (1, 2):
        titles = await generate()
        if titles:
            return titles
        logger.warning("Titler returned no titles for %s (attempt %d)", subject, attempt)
    return []


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
        titles = await _titles_with_retry(
            lambda: generate_titles(jpg_bytes, n=_TITLER_N), f"image {img.id}",
        )
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
        titles = await _titles_with_retry(
            lambda: generate_video_titles(frames, n=_TITLER_N), f"video {video.id}",
        )
    except Exception as exc:
        logger.exception("Video titler failed for video %s", video.id)
        raise HTTPException(status_code=502, detail=f"Titler failed: {exc}")

    if not titles:
        raise HTTPException(status_code=502, detail="Titler returned no titles")

    return {"titles": titles, "frames_used": len(frames)}


class SeriesTitlerRequest(BaseModel):
    series_id: str
    n: int | None = None          # optional override; default _TITLER_N


def _sample_evenly(items: list, count: int) -> list:
    """Pick `count` items spread across the list, keeping first, last and order.

    Sending the first four of a twenty-picture series would title whatever
    happens to open it rather than the series.
    """
    if len(items) <= count:
        return items
    step = (len(items) - 1) / (count - 1)
    return [items[round(i * step)] for i in range(count)]


@router.post("/api/titler/run-series")
async def run_series_titler(
    req: SeriesTitlerRequest,
    db: AsyncSession = Depends(get_db),
):
    """Titles for a series as a whole — several works, one name.

    The title this produces becomes the Instagram caption and the article's
    subject, so the model is shown members rather than one picture and asked
    what they share.
    """
    try:
        series_uuid = uuid.UUID(req.series_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid series_id")

    series = await db.get(ImageSeries, series_uuid)
    if not series:
        raise HTTPException(status_code=404, detail="Series not found")

    rows = (await db.execute(
        select(ImageSeriesItem)
        .where(ImageSeriesItem.series_id == series_uuid)
        .order_by(ImageSeriesItem.position)
    )).scalars().all()
    if not rows:
        raise HTTPException(status_code=400, detail="Series has no images")

    picked_ids = _sample_evenly([r.image_id for r in rows], _SERIES_IMAGES)
    found = {
        img.id: img
        for img in (await db.execute(
            select(Image).where(Image.id.in_(picked_ids))
        )).scalars().all()
    }

    jpgs: list[bytes] = []
    for image_id in picked_ids:                 # sample order, not query order
        img = found.get(image_id)
        if not img:
            continue
        # img.filepath, matching the single-image titler above: the model is
        # being asked about the work, and the renditions on top of it are
        # delivery, not subject.
        src = settings.storage_dir / img.filepath
        if not src.exists():
            continue
        jpg_bytes, _ = await prepare_jpg_for_web(
            src, max_edge=_TITLER_MAX_EDGE, quality=_TITLER_JPG_QUALITY,
        )
        jpgs.append(jpg_bytes)
    if not jpgs:
        raise HTTPException(status_code=404, detail="No readable images in this series")

    n = max(1, min(10, req.n if req.n is not None else _TITLER_N))
    logger.info(
        "Series titler: series=%s, model=%s, images=%d of %d, total=%dKB",
        series.id, settings.ollama_titler_model, len(jpgs), len(rows),
        sum(len(j) for j in jpgs) // 1024,
    )

    try:
        titles = await _titles_with_retry(
            lambda: generate_series_titles(jpgs, n=n), f"series {series.id}",
        )
    except Exception as exc:
        logger.exception("Series titler failed for series %s", series.id)
        raise HTTPException(status_code=502, detail=f"Titler failed: {exc}")

    if not titles:
        raise HTTPException(status_code=502, detail="Titler returned no titles")

    return {"titles": titles, "images_used": len(jpgs), "item_count": len(rows)}
