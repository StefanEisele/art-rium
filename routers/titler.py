"""
Image Titler — generate title suggestions for an image directly via Ollama.

Synchronous POST: client sends image_id, server re-encodes a small JPG,
calls the local VLM, and returns the parsed title list in the response.
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
from core.models import Image
from services.ollama.client import generate_titles

logger = logging.getLogger(__name__)
router = APIRouter(dependencies=[Depends(require_auth)])

_TITLER_MAX_EDGE = 512
_TITLER_JPG_QUALITY = 80
_TITLER_N = 5


class TitlerRequest(BaseModel):
    image_id: str


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
