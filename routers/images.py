"""
Image gallery API — list, search, tag, rate, and delete ingested images.
"""
import uuid
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from core.auth import auth_ok
from core.config import settings
from core.db import get_db
from core.models import Image

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/images")


class ImageUpdate(BaseModel):
    title: Optional[str] = None
    tags: Optional[list[str]] = None
    rating: Optional[int] = None
    notes: Optional[str] = None


@router.get("")
async def list_images(
    request: Request,
    limit: int = Query(50, le=200),
    offset: int = Query(0, ge=0),
    tag: Optional[str] = None,
    workflow: Optional[str] = None,
    search: Optional[str] = None,
    rating_min: Optional[int] = Query(None, ge=1, le=5),
    db: AsyncSession = Depends(get_db),
):
    if not auth_ok(request):
        raise HTTPException(status_code=401, detail="Invalid API key")

    stmt = select(Image).order_by(desc(Image.created_at)).offset(offset).limit(limit)
    if tag:
        stmt = stmt.where(Image.tags.contains([tag]))
    if workflow:
        stmt = stmt.where(Image.workflow_name == workflow)
    if search:
        stmt = stmt.where(Image.prompt.ilike(f"%{search}%"))
    if rating_min is not None:
        stmt = stmt.where(Image.rating >= rating_min)

    result = await db.execute(stmt)
    images = result.scalars().all()
    return [_serialize(img) for img in images]


@router.get("/{image_id}")
async def get_image_meta(image_id: uuid.UUID, request: Request, db: AsyncSession = Depends(get_db)):
    if not auth_ok(request):
        raise HTTPException(status_code=401, detail="Invalid API key")
    img = await db.get(Image, image_id)
    if not img:
        raise HTTPException(status_code=404, detail="Image not found")
    return _serialize(img)


@router.patch("/{image_id}")
async def update_image(
    image_id: uuid.UUID,
    body: ImageUpdate,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    if not auth_ok(request):
        raise HTTPException(status_code=401, detail="Invalid API key")
    img = await db.get(Image, image_id)
    if not img:
        raise HTTPException(status_code=404, detail="Image not found")
    if body.title is not None:
        img.title = body.title or None
    if body.tags is not None:
        img.tags = body.tags
    if body.rating is not None:
        if not (1 <= body.rating <= 5):
            raise HTTPException(status_code=400, detail="Rating must be 1–5")
        img.rating = body.rating
    if body.notes is not None:
        img.notes = body.notes
    await db.commit()
    return _serialize(img)


@router.delete("/{image_id}", status_code=204)
async def delete_image(
    image_id: uuid.UUID,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    if not auth_ok(request):
        raise HTTPException(status_code=401, detail="Invalid API key")
    img = await db.get(Image, image_id)
    if not img:
        raise HTTPException(status_code=404, detail="Image not found")

    # Remove file from managed storage
    filepath = settings.storage_dir / img.filepath
    if filepath.exists():
        try:
            filepath.unlink()
        except Exception as e:
            logger.warning(f"Could not delete file {filepath}: {e}")

    await db.delete(img)
    await db.commit()
    logger.info(f"Deleted image {image_id}")


def _serialize(img: Image) -> dict:
    return {
        "id": str(img.id),
        "filename": img.filename,
        "url": f"/api/image/{img.filename}",
        "title": img.title,
        "prompt": img.prompt,
        "seed": img.seed,
        "width": img.width,
        "height": img.height,
        "workflow_name": img.workflow_name,
        "batch_id": str(img.batch_id) if img.batch_id else None,
        "tags": img.tags or [],
        "rating": img.rating,
        "notes": img.notes,
        "created_at": img.created_at.isoformat(),
    }
