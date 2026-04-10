"""
Image gallery API — list, tag, rate, delete ingested images.
Phase 1: read + update. Delete in Phase 2.
"""
import uuid
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from core.auth import auth_ok
from core.db import get_db
from core.models import Image

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/images")


class ImageUpdate(BaseModel):
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
    db: AsyncSession = Depends(get_db),
):
    if not auth_ok(request):
        raise HTTPException(status_code=401, detail="Invalid API key")

    stmt = select(Image).order_by(desc(Image.created_at)).offset(offset).limit(limit)
    if tag:
        stmt = stmt.where(Image.tags.contains([tag]))
    if workflow:
        stmt = stmt.where(Image.workflow_name == workflow)

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


def _serialize(img: Image) -> dict:
    return {
        "id": str(img.id),
        "filename": img.filename,
        "url": f"/api/image/{img.filename}",
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
