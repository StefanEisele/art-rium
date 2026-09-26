"""Image series — the package a set of pictures is posted as.

A series is named, ordered and kept, so the same set can be handed to the
titler, the video tool, an article and an Instagram carousel without being
reassembled by hand each time. Position 0 is the cover; the order here is the
order the carousel goes out in.

Two things in here are load-bearing and easy to get wrong:

  * **Membership changes always go through `_set_members`**, which clears,
    flushes, and re-appends. Not only when replacing: a plain reorder issued
    as UPDATEs can violate UNIQUE(series_id, position) *transiently* — swap
    two positions and both rows briefly want the same number, and the order
    of UPDATEs inside one flush is not something callers control. One path
    for both, borrowed from services/instagram/media.py::replace_media_items.

  * **Appending uses MAX(position)+1, never len(items).** Deleting a picture
    cascades its item row away and leaves gaps (0, 1, 3), so the count and
    the next free position are different numbers.

Deleting a series never touches the pictures, and an emptied series is kept
rather than swept up: a named series with notes is something the user wrote.
"""
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from core.auth import require_auth
from core.db import get_db
from core.models import Image, ImageSeries, ImageSeriesItem
from routers.images import serialize_image

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/series", dependencies=[Depends(require_auth)])

# A series of one is not a sequence, and every consumer downstream is one: the
# transition video modes need two frames to have a transition at all, a
# carousel of one is just a post, and "walk the series" in the titler has
# nothing to walk.
MIN_IMAGES = 2
# The highest any consumer can take (flf2v). The real ceilings below it —
# articles 6, carousel 10, video 6/7/10/20 by mode — are enforced where the
# handoff happens, and shown in the gallery before the button is pressed.
MAX_IMAGES = 20

# How many member thumbnails the list endpoint sends per series, for the
# fanned cards behind the stack tile.
_PREVIEW_THUMBS = 3


class SeriesCreate(BaseModel):
    title: Optional[str] = None
    notes: Optional[str] = None
    image_ids: list[uuid.UUID]


class SeriesUpdate(BaseModel):
    title: Optional[str] = None
    notes: Optional[str] = None
    # Replaces membership *and* order in one go — the gallery's drag-to-sort
    # sends the whole list rather than a diff.
    image_ids: Optional[list[uuid.UUID]] = None


class SeriesImagesAdd(BaseModel):
    image_ids: list[uuid.UUID]


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _dedupe(ids: list[uuid.UUID]) -> list[uuid.UUID]:
    """Drop repeats, keep first-seen order.

    UNIQUE(series_id, image_id) would catch these, but as a 500 IntegrityError
    halfway through an insert rather than as an answer.
    """
    seen: set[uuid.UUID] = set()
    out: list[uuid.UUID] = []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


async def _images_in_order(ids: list[uuid.UUID], db: AsyncSession) -> list[Image]:
    """Load every picture in `ids`, 404 if one is missing, keep the given order.

    `WHERE id IN (...)` answers in whatever order it likes, and for a series
    the order *is* the content — the same reason routers/video.py re-orders
    its key frames after resolving them.

    `populate_existing` because this also runs *after* a membership change:
    the session already holds these Image rows with their `series_items`
    eagerly loaded, and without it they would be handed back from the
    identity map still listing the membership we just rewrote.
    """
    if not ids:
        return []
    result = await db.execute(
        select(Image)
        .where(Image.id.in_(ids))
        .execution_options(populate_existing=True)
    )
    by_id = {img.id: img for img in result.scalars().all()}
    missing = [str(i) for i in ids if i not in by_id]
    if missing:
        raise HTTPException(
            status_code=404,
            detail=f"Image(s) not found: {', '.join(missing)}",
        )
    return [by_id[i] for i in ids]


def _validate_count(n: int) -> None:
    if n < MIN_IMAGES:
        raise HTTPException(
            status_code=400,
            detail=f"A series needs at least {MIN_IMAGES} images",
        )
    if n > MAX_IMAGES:
        raise HTTPException(
            status_code=400,
            detail=f"A series holds at most {MAX_IMAGES} images",
        )


async def _set_members(
    series: ImageSeries,
    ids: list[uuid.UUID],
    db: AsyncSession,
) -> None:
    """Replace the members with this ordered list, renumbered 0..n-1.

    The flush between clearing and appending is what keeps
    UNIQUE(series_id, position) from seeing the same position held by both a
    row on its way out and a row on its way in.
    """
    if series.items:
        series.items.clear()
        await db.flush()
    for position, image_id in enumerate(ids):
        series.items.append(ImageSeriesItem(position=position, image_id=image_id))


async def _get_series(series_id: uuid.UUID, db: AsyncSession) -> ImageSeries:
    result = await db.execute(
        select(ImageSeries)
        .where(ImageSeries.id == series_id)
        .options(selectinload(ImageSeries.items))
        .execution_options(populate_existing=True)
    )
    series = result.scalar_one_or_none()
    if not series:
        raise HTTPException(status_code=404, detail="Series not found")
    return series


async def _serialize_full(series_id: uuid.UUID, db: AsyncSession) -> dict:
    """Re-read, then serialize.

    Takes an id rather than the row on purpose: every caller reaches here
    just after committing a change, and the in-memory row's members are the
    ones from before it.
    """
    series = await _get_series(series_id, db)
    ids = [it.image_id for it in series.items]
    images = await _images_in_order(ids, db) if ids else []
    return {
        "id": str(series.id),
        "title": series.title,
        "notes": series.notes,
        "item_count": len(images),
        "images": [serialize_image(img) for img in images],
        "created_at": series.created_at.isoformat(),
        "updated_at": series.updated_at.isoformat(),
    }


@router.get("")
async def list_series(
    limit: int = Query(50, le=200),
    offset: int = Query(0, ge=0),
    search: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    """The stack grid's feed: no members, just enough to draw a tile.

    Three queries for a whole page, not three per series — the members of
    every series on the page come back in one go and are grouped here.
    """
    stmt = (
        select(ImageSeries)
        .order_by(desc(ImageSeries.updated_at))
        .offset(offset)
        .limit(limit)
    )
    if search:
        stmt = stmt.where(ImageSeries.title.ilike(f"%{search}%"))
    series_rows = (await db.execute(stmt)).scalars().all()
    if not series_rows:
        return []

    series_ids = [s.id for s in series_rows]
    items = (await db.execute(
        select(ImageSeriesItem)
        .where(ImageSeriesItem.series_id.in_(series_ids))
        .order_by(ImageSeriesItem.series_id, ImageSeriesItem.position)
    )).scalars().all()

    by_series: dict[uuid.UUID, list[uuid.UUID]] = {sid: [] for sid in series_ids}
    for it in items:
        by_series[it.series_id].append(it.image_id)

    # Only the pictures the tiles actually show get loaded.
    wanted = {img_id for ids in by_series.values() for img_id in ids[:_PREVIEW_THUMBS]}
    thumbs: dict[uuid.UUID, str] = {}
    if wanted:
        preview_rows = (await db.execute(
            select(Image).where(Image.id.in_(wanted))
        )).scalars().all()
        # Through serialize_image so the `?v=` rendition marker is the same
        # one the gallery grid uses; a hand-built /thumb URL would go stale
        # the first time a member is enhanced.
        thumbs = {img.id: serialize_image(img)["thumb_url"] for img in preview_rows}

    return [
        {
            "id": str(s.id),
            "title": s.title,
            "notes": s.notes,
            "item_count": len(by_series[s.id]),
            "preview_thumbs": [
                thumbs[i] for i in by_series[s.id][:_PREVIEW_THUMBS] if i in thumbs
            ],
            "created_at": s.created_at.isoformat(),
            "updated_at": s.updated_at.isoformat(),
        }
        for s in series_rows
    ]


@router.post("", status_code=201)
async def create_series(body: SeriesCreate, db: AsyncSession = Depends(get_db)):
    ids = _dedupe(body.image_ids)
    _validate_count(len(ids))
    await _images_in_order(ids, db)          # 404s before anything is inserted

    series = ImageSeries(
        title=(body.title or "").strip() or None,
        notes=(body.notes or "").strip() or None,
    )
    db.add(series)
    await _set_members(series, ids, db)
    await db.commit()
    logger.info("Created series %s with %d images", series.id, len(ids))
    return await _serialize_full(series.id, db)


@router.get("/{series_id}")
async def get_series(series_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    return await _serialize_full(series_id, db)


@router.patch("/{series_id}")
async def update_series(
    series_id: uuid.UUID,
    body: SeriesUpdate,
    db: AsyncSession = Depends(get_db),
):
    series = await _get_series(series_id, db)
    if body.title is not None:
        series.title = body.title.strip() or None
    if body.notes is not None:
        series.notes = body.notes.strip() or None
    if body.image_ids is not None:
        ids = _dedupe(body.image_ids)
        _validate_count(len(ids))
        await _images_in_order(ids, db)
        await _set_members(series, ids, db)
    series.updated_at = _now()
    await db.commit()
    return await _serialize_full(series.id, db)


@router.post("/{series_id}/images")
async def add_series_images(
    series_id: uuid.UUID,
    body: SeriesImagesAdd,
    db: AsyncSession = Depends(get_db),
):
    """Append pictures that aren't in the series yet.

    Already-present ids are skipped rather than rejected — the picker hands
    over whatever was ticked, and re-ticking something already in the series
    is not an error worth stopping on.
    """
    series = await _get_series(series_id, db)
    present = {it.image_id for it in series.items}
    new_ids = [i for i in _dedupe(body.image_ids) if i not in present]
    if not new_ids:
        return await _serialize_full(series.id, db)

    if len(present) + len(new_ids) > MAX_IMAGES:
        raise HTTPException(
            status_code=400,
            detail=f"A series holds at most {MAX_IMAGES} images",
        )
    await _images_in_order(new_ids, db)

    # Not len(items): a series that lost a picture has gaps, and the count
    # would point at an occupied position.
    next_pos = (await db.execute(
        select(func.coalesce(func.max(ImageSeriesItem.position), -1))
        .where(ImageSeriesItem.series_id == series.id)
    )).scalar_one() + 1
    for offset, image_id in enumerate(new_ids):
        series.items.append(
            ImageSeriesItem(position=next_pos + offset, image_id=image_id)
        )
    series.updated_at = _now()
    await db.commit()
    return await _serialize_full(series.id, db)


@router.delete("/{series_id}/images/{image_id}")
async def remove_series_image(
    series_id: uuid.UUID,
    image_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    """Take one picture out of the series. The picture itself is untouched."""
    series = await _get_series(series_id, db)
    item = next((it for it in series.items if it.image_id == image_id), None)
    if not item:
        raise HTTPException(status_code=404, detail="Image is not in this series")
    series.items.remove(item)
    series.updated_at = _now()
    await db.commit()
    return await _serialize_full(series.id, db)


@router.delete("/{series_id}", status_code=204)
async def delete_series(series_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """Delete the series. The pictures stay in the gallery, always."""
    series = await _get_series(series_id, db)
    await db.delete(series)
    await db.commit()
    logger.info("Deleted series %s (images untouched)", series_id)
