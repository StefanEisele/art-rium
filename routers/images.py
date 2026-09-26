"""
Image gallery API — list, search, tag, rate, delete, crop, auto-enhance and
grain ingested images.

The crop ("Zuschnitt"), auto-enhance ("Zauberstab") and film-grain endpoints
follow the same shape as the video tool's post-passes: a sibling file rendered
from the rendition below it, and a delete that puts the previous state back.
The wand and the grain add a live preview so a strength can be judged before
committing; the crop needs none, its editor draws the box over the picture.
See services/image/crop.py, enhance.py and grain.py for what they do to the
pixels, and services/image/rendition.py for the order they stack in and who
gets which rendition afterwards.
"""
import asyncio
import logging
import shutil
import uuid
from pathlib import Path
from typing import Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from core.auth import require_auth
from core.config import settings
from core.db import AsyncSessionLocal, get_db
from core.job_control import cancel_job
from core.models import Image, ImageSeriesItem
from core.tasks import safe_create_task
from core.thumbnail import make_thumbnail, thumb_rel_path
from services.comfy.client import poll_history, post_workflow, upload_image
from services.comfy.progress import attach_live_stage
from workers.comfy_listener import get_listener
from services.image import crop as crop_service
from services.image import grain as grain_service
from services.image import preview as image_preview
from services.image.enhance import (
    STRENGTH_DEFAULT,
    STRENGTH_MAX,
    clamp_strength,
    enhance_file,
    preview_bytes,
)
from services.image import upscale as upscale_service
from services.image.rendition import (
    crop_source_path,
    cropped_path,
    cropped_rel_path,
    delivered_size,
    enhance_source_path,
    enhanced_path,
    enhanced_rel_path,
    grain_source_path,
    grained_path,
    grained_rel_path,
    is_cropped,
    is_enhanced,
    is_grained,
    is_upscaled,
    primary_filename,
    primary_filepath,
    upscale_source_path,
    upscaled_path,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/images", dependencies=[Depends(require_auth)])

# A tiled redraw is one small generation per tile; a 4x portrait is ~48 of
# them. Generous enough that a big job finishes rather than being abandoned
# halfway through with a half-written file.
_UPSCALE_TIMEOUT = 3600


class ImageUpdate(BaseModel):
    title: Optional[str] = None
    tags: Optional[list[str]] = None
    rating: Optional[int] = None
    notes: Optional[str] = None


class EnhanceRequest(BaseModel):
    strength: int = STRENGTH_DEFAULT


class GrainRequest(BaseModel):
    strength: int = grain_service.STRENGTH_DEFAULT


class CropRequest(BaseModel):
    # Fractions of the rendition the crop is cut from (the upscale when there
    # is one) — the editor draws on exactly that picture.
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    w: float = Field(gt=0, le=1)
    h: float = Field(gt=0, le=1)
    aspect: str = crop_service.ASPECT_FREE


class UpscaleRequest(BaseModel):
    # Kreativität: the redraw's start noise. 0 = enlargement only; up to 0.6
    # the tiles are repainted more and more. None = whatever suits the chosen
    # enlarger (services/image/upscale.py::ENLARGERS).
    denoise: Optional[float] = None
    scale: float = upscale_service.SCALE_DEFAULT
    model: str = upscale_service.DEFAULT_UPSCALE_MODEL
    seed: Optional[int] = None


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
    series_id: Optional[uuid.UUID] = Query(None, description="Only this series' members, in series order."),
    in_series: Optional[bool] = Query(None, description="True: only images that belong to some series. False: only loose ones. None: all."),
    db: AsyncSession = Depends(get_db),
):
    stmt = select(Image)

    if series_id is not None:
        # Here the order IS the content, so it comes from the series rather
        # than the clock. Anything handing a whole series to another tool
        # should still read /api/series/{id} — this filter is for pickers
        # that want to browse one.
        stmt = (
            stmt.join(ImageSeriesItem, ImageSeriesItem.image_id == Image.id)
            .where(ImageSeriesItem.series_id == series_id)
            .order_by(ImageSeriesItem.position)
        )
    else:
        stmt = stmt.order_by(desc(Image.created_at))

    # A subquery rather than a join: a picture may sit in several series, and
    # a join would hand it back once per membership.
    if in_series is True:
        stmt = stmt.where(Image.id.in_(select(ImageSeriesItem.image_id)))
    elif in_series is False:
        stmt = stmt.where(Image.id.not_in(select(ImageSeriesItem.image_id)))

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

    stmt = stmt.offset(offset).limit(limit)
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

    Cheap, and it is what makes an edit visible everywhere at once: the gallery
    grid, the Instagram picker and the articles picker all render
    `/api/image/{filename}/thumb`, so refreshing this one file updates every
    surface without any of them knowing renditions exist.
    """
    src = settings.storage_dir / primary_filepath(img)
    if src.exists():
        await make_thumbnail(src, settings.storage_dir / thumb_rel_path(img.filename))


async def _rerender_crop(img: Image) -> None:
    """Re-cut the crop after the rendition underneath it changed.

    The crop is cut from the upscale when there is one, so finishing an
    upscale — or removing it again — re-cuts the same framing from the new
    source: the box is fractions, so it lands in the same place at any size.
    Always call it *before* `_rerender_enhancement`, which reads its output.
    """
    if not img.crop_box:
        return
    src = crop_source_path(img)
    if not src.exists():
        return
    rel, name = cropped_rel_path(img)
    box = crop_service.CropBox.from_dict(img.crop_box)
    exact, w, h = await crop_service.crop_file(src, settings.storage_dir / rel, box)
    img.crop_box = exact.as_dict()
    img.cropped_filename = name
    img.cropped_filepath = rel
    img.crop_width, img.crop_height = w, h


async def _rerender_enhancement(img: Image) -> None:
    """Re-run the wand after the rendition underneath it changed.

    The enhancement reads the upscale when there is one, so finishing an
    upscale — or removing it again — invalidates it. Re-rendering rather than
    dropping is what keeps an enhanced image from shrinking back to its
    generated size the moment it is upscaled: the same correction, at the new
    resolution. Costs a Pillow pass, which is nothing next to the GPU minutes
    that produced the source.

    The analysis runs again on the new source, so `enhance_params` follows the
    picture that is actually being corrected; the user's chosen strength is
    what carries over.
    """
    if not img.enhance_strength:
        return
    src = enhance_source_path(img)
    if not src.exists():
        return
    rel, name = enhanced_rel_path(img)
    img.enhance_params = await enhance_file(
        src, settings.storage_dir / rel, img.enhance_strength,
    )
    img.enhanced_filename = name
    img.enhanced_filepath = rel


async def _rerender_grain(img: Image) -> None:
    """Re-run the grain pass after the rendition underneath it changed.

    Grain reads the enhanced file when there is one and the upscale otherwise,
    so enhancing, re-enhancing at a new strength, clearing the enhancement and
    upscaling all invalidate it. Same rule as `_reapply_grain_if_any` in
    routers/video.py, and the reason grain can be left switched on while the
    wand is played with. Always call it *after* `_rerender_enhancement`, since
    it reads that pass's output.
    """
    if not img.grain_strength:
        return
    src = grain_source_path(img)
    if not src.exists():
        return
    rel, name = grained_rel_path(img)
    await grain_service.grain_file(src, settings.storage_dir / rel, img.grain_strength)
    img.grained_filename = name
    img.grained_filepath = rel


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
    src = enhance_source_path(img)
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

    Always reads the rendition *below* it — the upscale when there is one, the
    original otherwise — so re-running at a new strength replaces the
    correction rather than stacking it, and an upscaled image stays upscaled
    while the wand is played with. Strength 0 means "off" and is routed to the
    same teardown the DELETE performs, so there is one way to end up unenhanced
    instead of two.
    """
    img = await db.get(Image, image_id)
    if not img:
        raise HTTPException(status_code=404, detail="Image not found")

    strength = clamp_strength(body.strength)
    if strength <= 0:
        return await _clear_enhancement(img, db)

    src = enhance_source_path(img)
    if not src.exists():
        raise HTTPException(status_code=404, detail="Source image missing on disk")

    rel, name = enhanced_rel_path(img)
    params = await enhance_file(src, settings.storage_dir / rel, strength)

    img.enhance_strength = strength
    img.enhanced_filename = name
    img.enhanced_filepath = rel
    img.enhance_params = params
    await _rerender_grain(img)
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
    """Delete the rendition file, null the columns, restore the thumbnail.

    The upscale is deliberately untouched: it sits *under* the enhancement, so
    removing the correction puts the upscaled picture back, not the small one.
    """
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
    await _rerender_grain(img)
    await _refresh_thumbnail(img)
    await db.commit()
    return _serialize(img)


# ── Film grain ───────────────────────────────────────────────────────────────
# Same three endpoints as the enhancement, one layer up the rendition stack.


@router.get("/{image_id}/grain/preview")
async def preview_grain(
    image_id: uuid.UUID,
    strength: int = Query(
        grain_service.STRENGTH_DEFAULT, ge=0, le=grain_service.STRENGTH_MAX,
    ),
    db: AsyncSession = Depends(get_db),
):
    """A downscaled JPEG of what `strength` would produce. Writes nothing.

    Rendered off the same source the real pass uses, so the preview shows the
    grain sitting on the enhanced picture when the wand is on.
    """
    img = await db.get(Image, image_id)
    if not img:
        raise HTTPException(status_code=404, detail="Image not found")
    src = grain_source_path(img)
    if not src.exists():
        raise HTTPException(status_code=404, detail="Source image missing on disk")

    jpg = await grain_service.preview_bytes(src, strength)
    return Response(
        content=jpg,
        media_type="image/jpeg",
        # Short-lived: the noise field is regenerated per render, so a longer
        # cache would pin one arbitrary field for the rest of the session.
        headers={"Cache-Control": "private, max-age=30"},
    )


@router.post("/{image_id}/grain")
async def grain_image_endpoint(
    image_id: uuid.UUID,
    body: GrainRequest,
    db: AsyncSession = Depends(get_db),
):
    """Render (or re-render) the grained rendition of this image.

    Always reads the rendition *below* the grain, so re-running at a new
    strength replaces the grain instead of piling a second field onto the
    first. Strength 0 means "off" and is routed to the same teardown the
    DELETE performs.
    """
    img = await db.get(Image, image_id)
    if not img:
        raise HTTPException(status_code=404, detail="Image not found")

    strength = grain_service.clamp_strength(body.strength)
    if strength <= 0:
        return await _clear_grain(img, db)

    src = grain_source_path(img)
    if not src.exists():
        raise HTTPException(status_code=404, detail="Source image missing on disk")

    rel, name = grained_rel_path(img)
    await grain_service.grain_file(src, settings.storage_dir / rel, strength)

    img.grain_strength = strength
    img.grained_filename = name
    img.grained_filepath = rel
    await _refresh_thumbnail(img)
    await db.commit()
    logger.info("Grained image %s at strength %d", image_id, strength)
    return _serialize(img)


@router.delete("/{image_id}/grain")
async def remove_grain(image_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """Drop the grained rendition and put the rendition under it back in front."""
    img = await db.get(Image, image_id)
    if not img:
        raise HTTPException(status_code=404, detail="Image not found")
    return await _clear_grain(img, db)


async def _clear_grain(img: Image, db: AsyncSession) -> dict:
    """Delete the rendition file, null the columns, restore the thumbnail."""
    path = grained_path(img)
    if path and path.exists():
        try:
            path.unlink()
        except OSError as exc:
            # Not fatal: the row decides which rendition is served, so a
            # stranded file is wasted disk, not a wrong picture.
            logger.warning("Could not delete grained rendition %s: %s", path, exc)

    img.grain_strength = None
    img.grained_filename = None
    img.grained_filepath = None
    await _refresh_thumbnail(img)
    await db.commit()
    return _serialize(img)


# ── Zuschnitt (crop) ─────────────────────────────────────────────────────────
# One layer above the upscale and below the wand: setting or clearing it
# re-renders the two Pillow passes on top, the same way an upscale landing
# does, so the delivered picture is always the framing with the tone on it.


@router.post("/{image_id}/crop")
async def crop_image_endpoint(
    image_id: uuid.UUID,
    body: CropRequest,
    db: AsyncSession = Depends(get_db),
):
    """Cut (or re-cut) the cropped rendition of this image.

    Always from the rendition *below* it, so re-framing replaces the previous
    crop instead of cropping the crop. A box that keeps the whole frame is no
    crop at all and is routed to the same teardown the DELETE performs — one
    way to end up uncropped, not two.
    """
    img = await db.get(Image, image_id)
    if not img:
        raise HTTPException(status_code=404, detail="Image not found")

    src = crop_source_path(img)
    if not src.exists():
        raise HTTPException(status_code=404, detail="Source image missing on disk")

    box = crop_service.normalize(body.x, body.y, body.w, body.h, body.aspect)
    src_w, src_h = await asyncio.to_thread(_png_size, src)
    if crop_service.is_full_frame(src_w, src_h, box):
        return await _clear_crop(img, db) if img.crop_box else _serialize(img)
    if crop_service.is_too_small(src_w, src_h, box):
        raise HTTPException(status_code=400, detail="Ausschnitt ist zu klein")

    rel, name = cropped_rel_path(img)
    exact, w, h = await crop_service.crop_file(src, settings.storage_dir / rel, box)

    img.crop_box = exact.as_dict()
    img.cropped_filename = name
    img.cropped_filepath = rel
    img.crop_width, img.crop_height = w, h
    await _rerender_enhancement(img)
    await _rerender_grain(img)
    await _refresh_thumbnail(img)
    await db.commit()
    logger.info("Cropped image %s to %dx%d (%s)", image_id, w, h, box.aspect)
    return _serialize(img)


@router.delete("/{image_id}/crop")
async def remove_crop(image_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """Drop the cropped rendition and put the full frame back."""
    img = await db.get(Image, image_id)
    if not img:
        raise HTTPException(status_code=404, detail="Image not found")
    if not img.crop_box:
        # Nothing to take off — and no reason to re-render the wand and the
        # grain, which on an upscale is seconds each.
        return _serialize(img)
    return await _clear_crop(img, db)


async def _clear_crop(img: Image, db: AsyncSession) -> dict:
    """Delete the rendition file, null the columns, re-render what sat on it."""
    path = cropped_path(img)
    if path and path.exists():
        try:
            path.unlink()
        except OSError as exc:
            # Not fatal: the row decides which rendition is served.
            logger.warning("Could not delete cropped rendition %s: %s", path, exc)

    img.crop_box = None
    img.cropped_filename = None
    img.cropped_filepath = None
    img.crop_width = None
    img.crop_height = None
    # The wand and the grain were rendered from the crop; without it they
    # have to come from the full frame again, or the gallery would go on
    # serving the old framing under a row that says it is gone.
    await _rerender_enhancement(img)
    await _rerender_grain(img)
    await _refresh_thumbnail(img)
    await db.commit()
    return _serialize(img)


# ── Diffusion upscale (Z-Image Turbo + Ultimate SD Upscale) ──────────────────
# Unlike the wand and the grain, this one is GPU minutes rather than
# milliseconds, so it runs as a background task with a polled progress entry —
# the same shape routers/video.py uses for its own upscale pass. One at a time,
# process-wide: two tiled diffusion runs on one card would just thrash.
#
# It reads the *original* regardless of what the wand and the grain are set to,
# and both of those are re-rendered on top of the result when it lands — so an
# upscale never has to be redone because something cheap above it changed, and
# never has to be dropped to honour one.

_upscale_progress: dict[str, dict] = {}
_image_upscale_gate = asyncio.Lock()


def _set_upscale_progress(
    key: str, phase: str, message: str, pct: int, **extra,
) -> None:
    _upscale_progress[key] = {"phase": phase, "message": message, "pct": pct, **extra}


@router.get("/{image_id}/upscale/options")
async def upscale_options(image_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """What this image can be upscaled to, and what it would cost in time.

    Server-side because the clamps are: the scale a request actually gets is
    bounded by the output-size budget, and the client should show the honest
    number before the user commits, not after.
    """
    img = await db.get(Image, image_id)
    if not img:
        raise HTTPException(status_code=404, detail="Image not found")

    src_w, src_h = img.width or 0, img.height or 0
    scales = []
    for choice in upscale_service.SCALE_CHOICES:
        allowed = upscale_service.clamp_scale(choice, src_w, src_h)
        w, h = upscale_service.output_size(src_w, src_h, allowed)
        scales.append({
            "scale": choice,
            "effective_scale": allowed,
            "capped": allowed < choice - 0.01,
            "width": w, "height": h,
            "seconds": upscale_service.estimate_seconds(src_w, src_h, allowed),
            # The stages apart, so the ETA can follow the model and the
            # Kreativität slider without asking again.
            "timing": upscale_service.timing(src_w, src_h, allowed),
            "tiles": upscale_service.tile_count(w, h),
        })
    return {
        "denoise_min": upscale_service.DENOISE_MIN,
        "denoise_max": upscale_service.DENOISE_MAX,
        "denoise_default": upscale_service.DENOISE_DEFAULT,
        "denoise_bands": upscale_service.denoise_bands(),
        "scale_default": upscale_service.SCALE_DEFAULT,
        # Labels and hints are the measured behaviour and live next to the
        # measurements, not here.
        "models": [
            {"key": key, "label": spec["label"], "hint": spec["hint"],
             "default_denoise": spec["default_denoise"]}
            for key, spec in upscale_service.ENLARGERS.items()
        ],
        "default_model": upscale_service.DEFAULT_UPSCALE_MODEL,
        "scales": scales,
        "source": {"width": src_w, "height": src_h},
    }


@router.get("/{image_id}/upscale/progress")
async def upscale_progress(image_id: uuid.UUID):
    """Live progress for a running pass. `detail` carries ComfyUI's own
    per-node/step readout when the listener has one — a tiled run is minutes
    of silence otherwise."""
    entry = dict(_upscale_progress.get(str(image_id)) or {})
    if not entry:
        return {"phase": "idle", "message": "", "pct": 0}
    return attach_live_stage(entry)


@router.post("/{image_id}/upscale", status_code=202)
async def upscale_image_endpoint(
    image_id: uuid.UUID,
    body: UpscaleRequest,
    db: AsyncSession = Depends(get_db),
):
    """Queue the Z-Image + Ultimate SD Upscale pass for this image.

    Returns immediately; poll `/upscale/progress` and re-fetch the image when
    the phase reads `done`. The pass reads the untouched original, so re-running
    it replaces the previous upscale instead of enlarging it a second time —
    and the wand and grain settings on the image are simply re-applied on top
    of the new result.
    """
    img = await db.get(Image, image_id)
    if not img:
        raise HTTPException(status_code=404, detail="Image not found")
    if _upscale_progress.get(str(image_id), {}).get("phase") not in (None, "done", "failed"):
        raise HTTPException(status_code=409, detail="An upscale is already running for this image")

    src = upscale_source_path(img)
    if not src.exists():
        raise HTTPException(status_code=404, detail="Source image missing on disk")

    # Not UPSCALE_MODELS: that is the ESRGAN files only, and SeedVR2 — the
    # default — is not a file. Checking against it would quietly swap every
    # SeedVR2 request for the fallback.
    model = upscale_service.resolve_enlarger(body.model)
    denoise = upscale_service.clamp_denoise(body.denoise, model)
    scale = upscale_service.clamp_scale(body.scale, img.width or 0, img.height or 0)

    _set_upscale_progress(str(image_id), "queued", "Warte auf die GPU…", 3)
    safe_create_task(
        _run_image_upscale(image_id, denoise=denoise, scale=scale, model=model, seed=body.seed),
        name=f"image_upscale:{image_id}",
    )
    logger.info(
        "Queued image upscale %s — %.2fx, denoise %.2f, %s",
        image_id, scale, denoise, model,
    )
    return {
        "status": "queued",
        "scale": scale,
        "denoise": denoise,
        "model": model,
        "seconds": upscale_service.estimate_seconds(
            img.width or 0, img.height or 0, scale, model=model, denoise=denoise,
        ),
    }


async def _run_image_upscale(
    image_id: uuid.UUID, *, denoise: float, scale: float, model: str, seed: int | None,
) -> None:
    """Background task: submit to ComfyUI, wait, persist the rendition."""
    key = str(image_id)
    try:
        async with AsyncSessionLocal() as db:
            img = await db.get(Image, image_id)
            if not img:
                raise RuntimeError("Image row gone")
            src = upscale_source_path(img)
            prompt = img.prompt or ""
            src_w, src_h = img.width or 0, img.height or 0
            rel, name = upscale_service.upscaled_rel_path(img.filepath)

        if _image_upscale_gate.locked():
            _set_upscale_progress(key, "queued", "Warte auf die GPU (ein anderer Upscale läuft)…", 5)

        async with _image_upscale_gate:
            _set_upscale_progress(key, "uploading", "Bild wird an ComfyUI übergeben…", 10)
            async with httpx.AsyncClient(timeout=120) as client:
                uploaded = await upload_image(client, src, f"artrium_up_{image_id.hex[:10]}.png")
                wf, save_node = upscale_service.build_image_upscale_workflow(
                    uploaded, prompt=prompt, denoise=denoise, scale=scale,
                    upscale_model=model, seed=seed,
                    filename_prefix=f"artrium_imgup_{image_id.hex[:8]}",
                    src_width=src_w, src_height=src_h,
                )
                _set_upscale_progress(key, "submitting", "Workflow wird gestartet…", 15)
                prompt_id = await post_workflow(client, wf)
                listener = get_listener()
                if listener:
                    listener.register_node_labels(prompt_id, wf)
                _set_upscale_progress(
                    key, "running",
                    "Wird vergrößert und nachgezeichnet…" if denoise > 0 else "Wird vergrößert…",
                    25,
                    # The band the tiled redraw owns: ComfyUI's own step counter
                    # maps into it, so the bar moves through the tiles instead
                    # of sitting at 25% for several minutes.
                    _prompt_id=prompt_id, _band=(25, 90),
                )
                logger.info("Image upscale %s submitted (prompt %s)", image_id, prompt_id)
                outputs = await poll_history(
                    client, prompt_id, timeout=_UPSCALE_TIMEOUT, interval=5,
                )

        entry = (outputs.get(save_node) or {}).get("images") or []
        if not entry:
            raise RuntimeError(f"ComfyUI returned no image: {outputs.get(save_node)}")
        comfy_src = settings.comfyui_output_dir / entry[0].get("subfolder", "") / entry[0]["filename"]
        if not comfy_src.exists():
            raise FileNotFoundError(f"Upscaled file not found at {comfy_src}")

        _set_upscale_progress(key, "finalizing", "Wird gespeichert…", 92)
        dest = settings.storage_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(shutil.copy2, comfy_src, dest)
        out_w, out_h = await asyncio.to_thread(_png_size, dest)

        async with AsyncSessionLocal() as db:
            img = await db.get(Image, image_id)
            if img:
                img.upscaled_filename = name
                img.upscaled_filepath = rel
                img.upscale_scale = scale
                img.upscale_denoise = denoise
                img.upscale_model = model
                img.upscale_width, img.upscale_height = out_w, out_h
                # The Pillow passes live on top of this and were rendered from
                # the smaller source; re-run them in stack order so the delivery
                # file carries them at the new size instead of dropping back to
                # a stale small one.
                await _rerender_crop(img)
                await _rerender_enhancement(img)
                await _rerender_grain(img)
                await _refresh_thumbnail(img)
                await db.commit()

        _set_upscale_progress(key, "done", f"Fertig — {out_w}×{out_h}", 100)
        logger.info("Image %s upscaled to %dx%d", image_id, out_w, out_h)
    except Exception as exc:
        logger.exception("Image upscale %s failed", image_id)
        _set_upscale_progress(key, "failed", f"{type(exc).__name__}: {exc}", 0)


def _png_size(path) -> tuple[int, int]:
    from PIL import Image as PILImage
    with PILImage.open(path) as im:
        return im.width, im.height


async def _stop_upscale(image_id: uuid.UUID) -> None:
    """Cancel an upscale still running for this image, if there is one.

    The tiled redraw is minutes of GPU, and it writes its result onto a row
    that may be about to disappear — so dropping the rendition or deleting the
    image has to call this off first, or the pass finishes and re-populates the
    columns that were just cleared.
    """
    key = str(image_id)
    await cancel_job(image_id, prompt_ids=[_upscale_progress.get(key, {}).get("_prompt_id")])
    _upscale_progress.pop(key, None)


@router.delete("/{image_id}/upscale")
async def remove_upscale(image_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """Drop the upscaled rendition and put the one under it back in front."""
    img = await db.get(Image, image_id)
    if not img:
        raise HTTPException(status_code=404, detail="Image not found")
    await _stop_upscale(image_id)
    return await _clear_upscale(img, db)


async def _clear_upscale(img: Image, db: AsyncSession) -> dict:
    path = upscaled_path(img)
    if path and path.exists():
        try:
            path.unlink()
        except OSError as exc:
            logger.warning("Could not delete upscaled rendition %s: %s", path, exc)

    img.upscaled_filename = None
    img.upscaled_filepath = None
    img.upscale_scale = None
    img.upscale_denoise = None
    img.upscale_model = None
    img.upscale_width = None
    img.upscale_height = None
    _upscale_progress.pop(str(img.id), None)
    # The Pillow passes were rendered on top of the upscale; without it they
    # have to come from the original again, or the gallery would serve a 4K
    # rendition of a picture that is back to 1K. This is the one path that is
    # *meant* to take the image back to its original size.
    await _rerender_crop(img)
    await _rerender_enhancement(img)
    await _rerender_grain(img)
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
        await _stop_upscale(img.id)
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

    await _stop_upscale(image_id)
    _delete_files(img)
    await db.delete(img)
    await db.commit()
    logger.info(f"Deleted image {image_id}")


def _delete_files(img: Image) -> None:
    """Remove the full image, its derived renditions and its thumbnail from
    disk (best-effort)."""
    rels = [
        img.filepath, img.enhanced_filepath, img.upscaled_filepath,
        img.cropped_filepath, img.grained_filepath, img.thumbnail_path,
    ]
    for rel in filter(None, rels):
        path = settings.storage_dir / rel
        if path.exists():
            try:
                path.unlink()
            except Exception as e:
                logger.warning(f"Could not delete file {path}: {e}")

    # Cached AVIF previews are keyed by source stem, one directory per
    # rendition. Stale entries left behind by a *re-render* are harmless —
    # nothing points at them — but the ones belonging to a picture that no
    # longer exists should go with it.
    for rel in filter(None, [
        img.filepath, img.enhanced_filepath, img.upscaled_filepath,
        img.cropped_filepath, img.grained_filepath,
    ]):
        image_preview.purge(settings.previews_dir, Path(rel).stem)


def _serialize(img: Image) -> dict:
    # The thumbnail is re-cut in place whenever a rendition changes, so its URL
    # has to carry a version or every cache between here and the browser keeps
    # showing the previous one. Both passes go into the marker: switching grain
    # on and off at a fixed enhance strength has to bust it too. `url` stays the
    # original on purpose — the gallery modal needs both to offer a before/after
    # compare, and `primary_url` is the one that means "what viewers get".
    # The upscale goes into the marker too: it replaces the primary file and
    # re-cuts the thumbnail, so a cached copy of either would otherwise survive
    # the change. So does the crop — by its box, since re-framing rewrites
    # every file above it under unchanged names.
    version = (
        f"?v=e{img.enhance_strength or 0}g{img.grain_strength or 0}u{img.upscale_scale or 0}"
        f"c{crop_service.token(img.crop_box)}"
        if (is_enhanced(img) or is_grained(img) or is_upscaled(img) or is_cropped(img))
        else ""
    )
    shown_w, shown_h = delivered_size(img)
    return {
        "id": str(img.id),
        "filename": img.filename,
        "url": f"/api/image/{img.filename}",
        "thumb_url": f"/api/image/{img.filename}/thumb{version}",
        "primary_url": f"/api/image/{primary_filename(img)}{version}",
        # What a viewer should actually be shown: the same rendition as
        # `primary_url`, as a small AVIF instead of a multi-megabyte PNG. The
        # caller appends `&w=` for the size it needs.
        #
        # It carries the same `?v=` marker as the thumbnail, and for the same
        # reason. The server side cannot go stale — the cache path behind this
        # is fingerprinted on the file's contents — but the *browser* can:
        # these responses are sent `immutable`, and re-running the wand
        # rewrites `..._enhanced.png` under an unchanged URL. The marker is
        # what makes that a different URL.
        "preview_url": f"/api/image/{primary_filename(img)}/preview{version}",
        "enhanced": is_enhanced(img),
        "enhance_strength": img.enhance_strength,
        "enhance_params": img.enhance_params,
        "enhanced_url": (
            f"/api/image/{img.enhanced_filename}{version}" if is_enhanced(img) else None
        ),
        "grained": is_grained(img),
        "grain_strength": img.grain_strength,
        "upscaled": is_upscaled(img),
        "upscale_scale": img.upscale_scale,
        "upscale_denoise": img.upscale_denoise,
        "upscale_model": img.upscale_model,
        "upscale_width": img.upscale_width,
        "upscale_height": img.upscale_height,
        "upscaled_url": (
            f"/api/image/{img.upscaled_filename}{version}" if is_upscaled(img) else None
        ),
        "cropped": is_cropped(img),
        # Fractions of the rendition it was cut from (the upscale when there
        # is one) plus the aspect it was drawn at — what the editor reopens on.
        "crop": img.crop_box if is_cropped(img) else None,
        "crop_width": img.crop_width,
        "crop_height": img.crop_height,
        "cropped_url": (
            f"/api/image/{img.cropped_filename}{version}" if is_cropped(img) else None
        ),
        # The shape viewers get — `width`/`height` below stay the generated
        # size, which the recipe and the upscale need. Anything laying the
        # picture out (a frame, a preview request) wants these.
        "delivered_width": shown_w,
        "delivered_height": shown_h,
        "title": img.title,
        "prompt": img.prompt,
        "seed": img.seed,
        "width": img.width,
        "height": img.height,
        "loras": img.loras or [],
        # The Detail Daemon dial this was sampled at — part of the recipe
        # "Weiterarbeiten" rebuilds, like the LoRA chain above it.
        "detail_amount": img.detail_amount,
        "workflow_name": img.workflow_name,
        "batch_id": str(img.batch_id) if img.batch_id else None,
        # Which curated series this picture belongs to — it may be several.
        # Ids only: the gallery already holds the titles from /api/series,
        # and nesting them here would mean a second eager relationship on
        # every image query in the app to save one lookup in one grid.
        "series_ids": [str(it.series_id) for it in img.series_items],
        "tags": img.tags or [],
        "rating": img.rating,
        "notes": img.notes,
        "wp_media_id": img.wp_media_id,
        "wp_uploaded": img.wp_media_id is not None,
        "created_at": img.created_at.isoformat(),
    }


# The series router serializes its members with this too, so a picture reads
# the same whichever endpoint handed it over — including the `?v=` rendition
# marker, which is the part nobody should ever rebuild by hand.
serialize_image = _serialize
