"""
Image gallery API — list, search, tag, rate, delete and auto-enhance
ingested images.

The auto-enhance ("Zauberstab") endpoints follow the same shape as the video
tool's grain pass: a sibling file rendered from the original, a live preview
so the strength can be judged before committing, and a delete that puts the
original back. See services/image/enhance.py for what it does to the pixels
and services/image/rendition.py for who gets which rendition afterwards.
"""
import uuid
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from core.auth import require_auth
from core.config import settings
from core.db import get_db
from core.models import Image
from core.thumbnail import make_thumbnail, thumb_rel_path
from services.image.enhance import (
    STRENGTH_DEFAULT,
    STRENGTH_MAX,
    clamp_strength,
    enhance_file,
    preview_bytes,
)
from services.image.rendition import (
    enhanced_path,
    enhanced_rel_path,
    is_enhanced,
    original_path,
    primary_filename,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/images", dependencies=[Depends(require_auth)])


class ImageUpdate(BaseModel):
    title: Optional[str] = None
    tags: Optional[list[str]] = None
    rating: Optional[int] = None
    notes: Optional[str] = None


class EnhanceRequest(BaseModel):
    strength: int = STRENGTH_DEFAULT


class BulkDeleteRequest(BaseModel):
    ids: list[uuid.UUID]


@router.get("")
async def list_images(
    limit: int = Query(50, le=200),
    offset: int = Query(0, ge=0),
    tag: Optional[str] = None,
    workflow: Optional[str] = None,
    search: Optional[str] = None,
    rating_min: Optional[int] = Query(None, ge=1, le=5),
    wp_uploaded: Optional[bool] = Query(None, description="True: only WP-uploaded images. False: only not-yet-uploaded. None: all."),
    db: AsyncSession = Depends(get_db),
):
    stmt = select(Image).order_by(desc(Image.created_at)).offset(offset).limit(limit)
    if tag:
        stmt = stmt.where(Image.tags.contains([tag]))
    if workflow:
        stmt = stmt.where(Image.workflow_name == workflow)
    if search:
        stmt = stmt.where(Image.prompt.ilike(f"%{search}%"))
    if rating_min is not None:
        stmt = stmt.where(Image.rating >= rating_min)
    if wp_uploaded is True:
        stmt = stmt.where(Image.wp_media_id.is_not(None))
    elif wp_uploaded is False:
        stmt = stmt.where(Image.wp_media_id.is_(None))

    result = await db.execute(stmt)
    images = result.scalars().all()
    return [_serialize(img) for img in images]


@router.get("/{image_id}")
async def get_image_meta(image_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    img = await db.get(Image, image_id)
    if not img:
        raise HTTPException(status_code=404, detail="Image not found")
    return _serialize(img)


@router.patch("/{image_id}")
async def update_image(
    image_id: uuid.UUID,
    body: ImageUpdate,
    db: AsyncSession = Depends(get_db),
):
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


# ── Auto-enhance ("Zauberstab") ──────────────────────────────────────────────


async def _refresh_thumbnail(img: Image) -> None:
    """Re-cut the thumbnail from whichever rendition is now current.

    Cheap, and it is what makes the enhancement visible everywhere at once:
    the gallery grid, the Instagram picker and the articles picker all render
    `/api/image/{filename}/thumb`, so refreshing this one file updates every
    surface without any of them knowing renditions exist.
    """
    src = enhanced_path(img) if is_enhanced(img) else original_path(img)
    if src and src.exists():
        await make_thumbnail(src, settings.storage_dir / thumb_rel_path(img.filename))


@router.get("/{image_id}/enhance/preview")
async def preview_enhance(
    image_id: uuid.UUID,
    strength: int = Query(STRENGTH_DEFAULT, ge=0, le=STRENGTH_MAX),
    db: AsyncSession = Depends(get_db),
):
    """A downscaled JPEG of what `strength` would produce. Writes nothing.

    Backs the live slider, so it must stay fast enough to feel immediate —
    hence a bounded edge rather than a full-resolution render (measured
    ~0.1-0.4 s full size, most of which is the pixels the user cannot see at
    modal size anyway).
    """
    img = await db.get(Image, image_id)
    if not img:
        raise HTTPException(status_code=404, detail="Image not found")
    src = original_path(img)
    if not src.exists():
        raise HTTPException(status_code=404, detail="Source image missing on disk")

    jpg = await preview_bytes(src, strength)
    return Response(
        content=jpg,
        media_type="image/jpeg",
        # Same strength always yields the same bytes, but the file behind it
        # can be re-enhanced; a short cache keeps slider drags cheap without
        # outliving the edit session.
        headers={"Cache-Control": "private, max-age=60"},
    )


@router.post("/{image_id}/enhance")
async def enhance_image_endpoint(
    image_id: uuid.UUID,
    body: EnhanceRequest,
    db: AsyncSession = Depends(get_db),
):
    """Render (or re-render) the enhanced rendition of this image.

    Always reads the original, so re-running at a new strength replaces the
    correction rather than stacking it. Strength 0 means "off" and is routed
    to the same teardown the DELETE performs, so there is one way to end up
    unenhanced instead of two.
    """
    img = await db.get(Image, image_id)
    if not img:
        raise HTTPException(status_code=404, detail="Image not found")

    strength = clamp_strength(body.strength)
    if strength <= 0:
        return await _clear_enhancement(img, db)

    src = original_path(img)
    if not src.exists():
        raise HTTPException(status_code=404, detail="Source image missing on disk")

    rel, name = enhanced_rel_path(img)
    params = await enhance_file(src, settings.storage_dir / rel, strength)

    img.enhance_strength = strength
    img.enhanced_filename = name
    img.enhanced_filepath = rel
    img.enhance_params = params
    await _refresh_thumbnail(img)
    await db.commit()
    logger.info("Enhanced image %s at strength %d", image_id, strength)
    return _serialize(img)


@router.delete("/{image_id}/enhance")
async def remove_enhancement(image_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """Drop the enhanced rendition and put the original back in front."""
    img = await db.get(Image, image_id)
    if not img:
        raise HTTPException(status_code=404, detail="Image not found")
    return await _clear_enhancement(img, db)


async def _clear_enhancement(img: Image, db: AsyncSession) -> dict:
    """Delete the rendition file, null the columns, restore the thumbnail."""
    path = enhanced_path(img)
    if path and path.exists():
        try:
            path.unlink()
        except OSError as exc:
            # Not fatal: the row is what decides which rendition is served, so
            # a stranded file is wasted disk, not a wrong picture.
            logger.warning("Could not delete enhanced rendition %s: %s", path, exc)

    img.enhance_strength = None
    img.enhanced_filename = None
    img.enhanced_filepath = None
    img.enhance_params = None
    await _refresh_thumbnail(img)
    await db.commit()
    return _serialize(img)


@router.delete("", status_code=204)
async def bulk_delete_images(
    body: BulkDeleteRequest,
    db: AsyncSession = Depends(get_db),
):
    """Delete multiple images by ID in one request."""
    if not body.ids:
        return

    result = await db.execute(select(Image).where(Image.id.in_(body.ids)))
    images = result.scalars().all()

    for img in images:
        _delete_files(img)
        await db.delete(img)

    await db.commit()
    logger.info(f"Bulk deleted {len(images)} images")


@router.delete("/{image_id}", status_code=204)
async def delete_image(
    image_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    img = await db.get(Image, image_id)
    if not img:
        raise HTTPException(status_code=404, detail="Image not found")

    _delete_files(img)
    await db.delete(img)
    await db.commit()
    logger.info(f"Deleted image {image_id}")


def _delete_files(img: Image) -> None:
    """Remove the full image, its enhanced rendition and its thumbnail from
    disk (best-effort)."""
    for rel in filter(None, [img.filepath, img.enhanced_filepath, img.thumbnail_path]):
        path = settings.storage_dir / rel
        if path.exists():
            try:
                path.unlink()
            except Exception as e:
                logger.warning(f"Could not delete file {path}: {e}")


def _serialize(img: Image) -> dict:
    # The thumbnail is re-cut in place whenever the enhancement changes, so its
    # URL has to carry a version or every cache between here and the browser
    # keeps showing the previous rendition. `url` stays the original on
    # purpose: the gallery modal needs both to offer a before/after compare,
    # and `primary_url` is the one that means "what viewers get".
    version = f"?v=e{img.enhance_strength}" if is_enhanced(img) else ""
    return {
        "id": str(img.id),
        "filename": img.filename,
        "url": f"/api/image/{img.filename}",
        "thumb_url": f"/api/image/{img.filename}/thumb{version}",
        "primary_url": f"/api/image/{primary_filename(img)}{version}",
        "enhanced": is_enhanced(img),
        "enhance_strength": img.enhance_strength,
        "enhance_params": img.enhance_params,
        "enhanced_url": (
            f"/api/image/{img.enhanced_filename}{version}" if is_enhanced(img) else None
        ),
        "title": img.title,
        "prompt": img.prompt,
        "seed": img.seed,
        "width": img.width,
        "height": img.height,
        "loras": img.loras or [],
        "workflow_name": img.workflow_name,
        "batch_id": str(img.batch_id) if img.batch_id else None,
        "tags": img.tags or [],
        "rating": img.rating,
        "notes": img.notes,
        "wp_media_id": img.wp_media_id,
        "wp_uploaded": img.wp_media_id is not None,
        "created_at": img.created_at.isoformat(),
    }
