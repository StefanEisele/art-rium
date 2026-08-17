"""Cloud video generation (MiniMax H3) — API surface.

Every write path here follows the same order, and the order is the feature:

    validate → create the job row → RESERVE the money → hand to the queue

Nothing reaches MiniMax before the reservation is committed. If the budget
says no, the caller gets a 402 with the numbers in it and no job exists.

Attachments come from art-rium's own libraries — gallery images as first/last
frame or reference, songs as reference audio — rather than from file uploads,
so the cloud path draws on the same material as everything else in the app.
Reference *videos* are supported by the API and priced in pricing.py, but not
exposed here: base64-inlining a 50 MB clip runs into the 64 MB request cap for
a case this library has no use for yet.
"""
from __future__ import annotations

import logging
import uuid
from decimal import Decimal, InvalidOperation

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.auth import require_auth
from core.config import settings
from core.db import get_db
from core.models import API_WORKFLOW, Image, LedgerEntry, Song, Video
from services.image.rendition import resolve_image_path
from services.video_api import budget, pricing
from services.video_api.backend import (
    MiniMaxBackend,
    VideoBackendError,
    VideoRequest,
    validate,
)
from services.video_api.queue import remember_request

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/video-api", dependencies=[Depends(require_auth)])


# ── Schemas ──────────────────────────────────────────────────────────────────


class GenerateRequest(BaseModel):
    prompt: str
    duration_s: int = Field(default=6, ge=pricing.DURATION_MIN, le=pricing.DURATION_MAX)
    resolution: str = "2K"
    ratio: str = "adaptive"
    title: str | None = None
    first_frame_image_id: uuid.UUID | None = None
    last_frame_image_id: uuid.UUID | None = None
    reference_image_ids: list[uuid.UUID] = Field(default_factory=list)
    reference_song_ids: list[uuid.UUID] = Field(default_factory=list)
    # Lets a double-clicked Generate button land on the same reservation.
    idempotency_key: str | None = None


class EstimateRequest(BaseModel):
    duration_s: int = Field(default=6, ge=pricing.DURATION_MIN, le=pricing.DURATION_MAX)
    resolution: str = "2K"
    image_count: int = 0
    kind: str | None = None          # defaults to the generation kind for `resolution`


class BudgetUpdate(BaseModel):
    limit_eur: str | None = None
    warn_threshold_pct: int | None = Field(default=None, ge=1, le=100)
    usd_eur_rate: str | None = None


# ── Helpers ──────────────────────────────────────────────────────────────────


def _decimal(value: str, field: str) -> Decimal:
    try:
        parsed = Decimal(value.replace(",", "."))
    except (InvalidOperation, AttributeError):
        raise HTTPException(status_code=422, detail=f"{field} is not a number")
    if parsed < 0:
        raise HTTPException(status_code=422, detail=f"{field} must not be negative")
    return parsed


async def _image_path(db: AsyncSession, image_id: uuid.UUID) -> tuple[Image, "object"]:
    image = await db.get(Image, image_id)
    if not image:
        raise HTTPException(status_code=404, detail=f"Image {image_id} not found")
    # The published rendition, so an enhanced or grained picture is what the
    # model actually sees — the same rule the Instagram and WordPress paths use.
    path = resolve_image_path(image)
    if not path.exists():
        raise HTTPException(status_code=409, detail=f"Image file missing: {image.filename}")
    return image, path


async def _build_request(db: AsyncSession, body: GenerateRequest) -> tuple[VideoRequest, list[uuid.UUID]]:
    first = last = None
    used_ids: list[uuid.UUID] = []

    if body.first_frame_image_id:
        image, path = await _image_path(db, body.first_frame_image_id)
        first = path
        used_ids.append(image.id)
    if body.last_frame_image_id:
        image, path = await _image_path(db, body.last_frame_image_id)
        last = path
        used_ids.append(image.id)

    refs = []
    for image_id in body.reference_image_ids:
        image, path = await _image_path(db, image_id)
        refs.append(path)
        used_ids.append(image.id)

    audio = []
    for song_id in body.reference_song_ids:
        song = await db.get(Song, song_id)
        if not song or not song.filepath:
            raise HTTPException(status_code=404, detail=f"Song {song_id} not found")
        path = settings.storage_dir / song.filepath
        if not path.exists():
            raise HTTPException(status_code=409, detail=f"Song file missing: {song.filename}")
        audio.append(path)

    return VideoRequest(
        prompt=body.prompt.strip(),
        duration_s=body.duration_s,
        resolution=body.resolution,
        ratio=body.ratio,
        first_frame=first,
        last_frame=last,
        reference_images=refs,
        reference_audio=audio,
    ), used_ids


def _serialize_job(v: Video, entry: LedgerEntry | None = None) -> dict:
    return {
        "id": str(v.id),
        "title": v.title,
        "prompt": v.prompt,
        "status": v.status,
        "error": v.error,
        "resolution": v.api_resolution,
        "ratio": v.api_ratio,
        "duration_s": v.duration_s,
        "task_id": v.api_task_id,
        "width": v.width,
        "height": v.height,
        "source_video_id": str(v.source_video_id) if v.source_video_id else None,
        "url": f"/api/video/file/{v.filename}" if v.filename else None,
        "thumb_url": f"/api/video/thumb/{v.id}" if v.status == "done" else None,
        "created_at": v.created_at.isoformat(),
        "cost": budget.serialize_entry(entry) if entry else None,
    }


async def _entry_for(db: AsyncSession, video_id: uuid.UUID) -> LedgerEntry | None:
    return (await db.execute(
        select(LedgerEntry).where(LedgerEntry.video_id == video_id)
        .order_by(LedgerEntry.created_at.desc()).limit(1)
    )).scalar_one_or_none()


# ── Status & budget ──────────────────────────────────────────────────────────


@router.get("/status")
async def api_status():
    """Whether the cloud path is usable at all, plus the rate card in force."""
    backend = MiniMaxBackend()
    return {
        "configured": backend.configured,
        "model": "MiniMax-H3",
        "base_url": backend.base_url,
        "max_concurrent": settings.video_api_max_concurrent,
        "rate_checked_on": pricing.RATE_CHECKED_ON,
        "rates_usd": {
            "768P_per_second": str(pricing.OUTPUT_RATE_USD["768P"]),
            "2K_per_second": str(pricing.OUTPUT_RATE_USD["2K"]),
            "regeneration_per_second": str(pricing.REGENERATION_RATE_USD),
            "extra_image": str(pricing.IMAGE_RATE_USD),
            "free_images": pricing.FREE_IMAGES,
        },
        "duration_range": [pricing.DURATION_MIN, pricing.DURATION_MAX],
        "ratios": list(pricing.RATIOS),
        "resolutions": list(pricing.RESOLUTIONS),
    }


@router.get("/budget")
async def get_budget(db: AsyncSession = Depends(get_db)):
    return await budget.summary(db)


@router.patch("/budget")
async def patch_budget(body: BudgetUpdate, db: AsyncSession = Depends(get_db)):
    """Edit this month's limit, warning threshold or exchange rate.

    Lowering the limit below what is already committed is deliberately allowed:
    it stops new jobs without cancelling paid-for work in flight.
    """
    rate = _decimal(body.usd_eur_rate, "usd_eur_rate") if body.usd_eur_rate is not None else None
    if rate is not None and rate <= 0:
        raise HTTPException(status_code=422, detail="usd_eur_rate must be greater than zero")
    await budget.update_period(
        db,
        limit_eur=_decimal(body.limit_eur, "limit_eur") if body.limit_eur is not None else None,
        warn_threshold_pct=body.warn_threshold_pct,
        usd_eur_rate=rate,
    )
    return await budget.summary(db)


@router.get("/ledger")
async def get_ledger(
    limit: int = Query(100, ge=1, le=500),
    month: str | None = None,
    db: AsyncSession = Depends(get_db),
):
    return await budget.ledger(db, limit=limit, month=month)


@router.post("/estimate")
async def post_estimate(body: EstimateRequest, db: AsyncSession = Depends(get_db)):
    """Price a call without booking it. Backs the live cost preview."""
    if body.resolution not in pricing.RESOLUTIONS:
        raise HTTPException(status_code=422, detail=f"resolution must be one of {pricing.RESOLUTIONS}")
    kind = body.kind or pricing.kind_for(body.resolution)
    if kind not in ("generate_768p", "generate_2k", "regenerate_2k"):
        raise HTTPException(status_code=422, detail=f"Unknown kind {kind!r}")
    return await budget.estimate(
        db, kind=kind, duration_s=body.duration_s, image_count=max(0, body.image_count),
    )


# ── Generation ───────────────────────────────────────────────────────────────


@router.post("/generate", status_code=202)
async def generate(body: GenerateRequest, db: AsyncSession = Depends(get_db)):
    """Reserve the cost and queue one cloud render.

    Returns 402 with the shortfall when the monthly limit would be broken —
    and in that case no job row is left behind.
    """
    backend = MiniMaxBackend()
    if not backend.configured:
        raise HTTPException(
            status_code=503,
            detail="MINIMAX_API_KEY is not set — add it to .env to generate over the API.",
        )

    idem = f"gen:{body.idempotency_key}" if body.idempotency_key else None
    if idem:
        existing = (await db.execute(
            select(LedgerEntry).where(LedgerEntry.idempotency_key == idem)
        )).scalar_one_or_none()
        if existing and existing.video_id:
            # A repeated click on Generate. Hand back the job it already made.
            video = await db.get(Video, existing.video_id)
            if video:
                return _serialize_job(video, existing)

    request, image_ids = await _build_request(db, body)
    try:
        validate(request)
    except VideoBackendError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    video = Video(
        workflow=API_WORKFLOW,
        status="queued",
        prompt=request.prompt,
        title=body.title or None,
        api_resolution=request.resolution,
        api_ratio=request.ratio,
        duration_s=request.duration_s,
        image_ids=image_ids or None,
        n_images=len(image_ids) or None,
        fps=24,
    )
    db.add(video)
    await db.commit()
    await db.refresh(video)

    try:
        entry = await budget.reserve(
            db,
            kind=pricing.kind_for(request.resolution),
            duration_s=request.duration_s,
            image_count=request.image_count,
            idempotency_key=idem or f"gen:{video.id}",
            video_id=video.id,
        )
    except budget.BudgetExceeded as exc:
        # Nothing was submitted and nothing should linger in the library.
        await db.delete(video)
        await db.commit()
        raise HTTPException(status_code=402, detail=exc.as_detail())

    # The queue picks the row up on its next tick; the payload (file paths)
    # only lives in memory, which is why an interrupted job with attachments is
    # failed rather than re-submitted blind.
    remember_request(video.id, request)
    logger.info(
        "Queued cloud render %s: %ss %s, %s € reserved",
        video.id, request.duration_s, request.resolution, entry.amount_eur,
    )
    return _serialize_job(video, entry)


@router.post("/jobs/{video_id}/regenerate", status_code=202)
async def regenerate_2k(video_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """Pull a finished 768P take up to 2K — the cheap half of the cost model.

    Iterating at 768P and only promoting the keepers is what makes the monthly
    limit go anywhere, so this is a first-class action rather than a re-run.
    """
    source = await db.get(Video, video_id)
    if not source or source.workflow != API_WORKFLOW:
        raise HTTPException(status_code=404, detail="Cloud job not found")
    if source.status != "done":
        raise HTTPException(status_code=409, detail="Only a finished take can be pulled up to 2K")
    if source.api_resolution != "768P":
        raise HTTPException(status_code=409, detail="This take is already 2K")
    if not source.api_task_id:
        raise HTTPException(
            status_code=409,
            detail="This take has no MiniMax task id, so it cannot be regenerated.",
        )

    already = (await db.execute(
        select(Video).where(
            Video.source_video_id == source.id, Video.status.in_(("queued", "generating", "done")),
        )
    )).scalars().first()
    if already:
        raise HTTPException(status_code=409, detail="A 2K version already exists or is running")

    # The regeneration re-bills the original's input material, so the original
    # entry's image count is part of the price.
    source_entry = await _entry_for(db, source.id)
    image_count = source_entry.ref_image_count if source_entry else 0

    target = Video(
        workflow=API_WORKFLOW,
        status="queued",
        prompt=source.prompt,
        title=(f"{source.title} (2K)" if source.title else None),
        api_resolution="2K",
        api_ratio=source.api_ratio,
        duration_s=source.duration_s,
        image_ids=source.image_ids,
        n_images=source.n_images,
        fps=24,
        source_video_id=source.id,
    )
    db.add(target)
    await db.commit()
    await db.refresh(target)

    try:
        entry = await budget.reserve(
            db,
            kind="regenerate_2k",
            duration_s=source.duration_s or 6,
            image_count=image_count,
            idempotency_key=f"regen:{target.id}",
            video_id=target.id,
        )
    except budget.BudgetExceeded as exc:
        await db.delete(target)
        await db.commit()
        raise HTTPException(status_code=402, detail=exc.as_detail())

    logger.info("Queued 2K regeneration %s of %s (%s €)", target.id, source.id, entry.amount_eur)
    return _serialize_job(target, entry)


# ── Jobs ─────────────────────────────────────────────────────────────────────


@router.get("/jobs")
async def list_jobs(
    limit: int = Query(50, ge=1, le=200),
    db: AsyncSession = Depends(get_db),
):
    videos = (await db.execute(
        select(Video).where(Video.workflow == API_WORKFLOW)
        .order_by(Video.created_at.desc()).limit(limit)
    )).scalars().all()
    entries = {}
    if videos:
        rows = (await db.execute(
            select(LedgerEntry).where(LedgerEntry.video_id.in_([v.id for v in videos]))
        )).scalars().all()
        for row in rows:
            entries.setdefault(row.video_id, row)
    return [_serialize_job(v, entries.get(v.id)) for v in videos]


@router.get("/jobs/{video_id}")
async def get_job(video_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    video = await db.get(Video, video_id)
    if not video or video.workflow != API_WORKFLOW:
        raise HTTPException(status_code=404, detail="Cloud job not found")
    return _serialize_job(video, await _entry_for(db, video_id))


@router.delete("/jobs/{video_id}", status_code=204)
async def cancel_or_delete(video_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """Cancel a job that is still queued, or delete a finished/failed one.

    A job MiniMax has already accepted cannot be cancelled and will be billed
    whatever happens — so this refuses rather than pretending, and says why.
    """
    video = await db.get(Video, video_id)
    if not video or video.workflow != API_WORKFLOW:
        raise HTTPException(status_code=404, detail="Cloud job not found")

    if video.status == "generating" and video.api_task_id:
        raise HTTPException(
            status_code=409,
            detail=(
                "MiniMax has accepted this task and it will be billed either way — "
                "it cannot be cancelled. Wait for it to finish, then delete it."
            ),
        )

    entry = await _entry_for(db, video_id)
    if entry and entry.state == "reserved":
        await budget.release(db, entry.id, note="Cancelled before submission")

    for name in filter(None, [video.filename, f"{video.id}_thumb.jpg"]):
        path = settings.videos_dir / name
        if path.exists():
            try:
                path.unlink()
            except OSError as exc:
                logger.warning("Could not delete %s: %s", path, exc)

    await db.delete(video)
    await db.commit()
