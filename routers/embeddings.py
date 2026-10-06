"""
Embeddings: the library a render picks from, and the trainer that fills it.

kentskooking's look comes from embeddings he trained himself on ~20 of his own
pictures; the Civitai ones were only ever the test that the mechanism works on
our graph (scripts/embedding_sweep.py). This router turns a gallery series into
one: POST a training, poll it, look at the preview each snapshot drew, and pick
the snapshot that renders use. See services/embedding/ for the trainer and the
measured defaults.

One training at a time. It takes the 4060 Ti for ~10 minutes, and a second one
would not run beside it so much as fight it for the same 16 GB.
"""
from __future__ import annotations

import asyncio
import logging
import shutil
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.auth import require_auth
from core.config import settings
from core.db import AsyncSessionLocal, get_db
from core.job_control import CANCELLED_MSG, cancel_job
from core.models import EmbeddingTraining, Image, ImageSeriesItem
from core.tasks import safe_create_task
from routers.video import _progress, _set_progress, forget_progress
from services.comfy.client import embedding_names
from services.embedding import (
    LR_DEFAULT,
    LR_MAX,
    LR_MIN,
    MAX_IMAGES,
    MIN_IMAGES,
    SAMPLE_PROMPTS,
    SAVE_EVERY,
    STEPS_DEFAULT,
    STEPS_MAX,
    STEPS_MIN,
    TEMPLATES,
    VECTORS_DEFAULT,
    VECTORS_MAX,
    VECTORS_MIN,
    PlanError,
    TrainingError,
    TrainingPlan,
    active_file,
    active_name,
    estimate_seconds,
    is_sample_filename,
    is_workshop_name,
    list_snapshots,
    plan_training,
    prepare_dataset,
    run_training,
    snapshot_name,
    train_dir,
)
from services.image.rendition import grain_source_path

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/embeddings", dependencies=[Depends(require_auth)])

_RUNNING = ("queued", "training")


class TrainRequest(BaseModel):
    name: str
    series_id: Optional[uuid.UUID] = None
    image_ids: list[uuid.UUID] = []
    template: str = "style"
    init_word: Optional[str] = None
    vectors: Optional[int] = None
    steps: Optional[int] = None
    learning_rate: Optional[float] = None


class ChooseRequest(BaseModel):
    # None = the final save; otherwise one of the snapshot steps.
    step: Optional[int] = None


# ── Helpers ──────────────────────────────────────────────────────────────────
def _options() -> dict:
    return {
        "vectors": {"min": VECTORS_MIN, "max": VECTORS_MAX, "default": VECTORS_DEFAULT},
        "steps": {"min": STEPS_MIN, "max": STEPS_MAX, "default": STEPS_DEFAULT},
        "learning_rate": {"min": LR_MIN, "max": LR_MAX, "default": LR_DEFAULT},
        "images": {"min": MIN_IMAGES, "max": MAX_IMAGES},
        "templates": list(TEMPLATES),
        "save_every": SAVE_EVERY,
        "sample_prompts": [text for text, _ in SAMPLE_PROMPTS],
        # 3.5 steps/s measured at batch 2 on the 4060 Ti; the UI quotes it.
        "steps_per_second": 3.5,
    }


def _sample_url(training_id: uuid.UUID, filename: str) -> str:
    return f"/api/embeddings/trainings/{training_id}/samples/{filename}"


def _snapshots(row: EmbeddingTraining) -> list[dict]:
    """The snapshots on disk, each with the name a render uses for it."""
    folder = train_dir(settings.embeddings_dir, row.name)
    out = []
    for snap in list_snapshots(folder, row.name):
        out.append({
            "step": snap["step"],
            "embedding": snapshot_name(row.name, snap["step"]) if snap["file"] else None,
            "samples": [_sample_url(row.id, f) for f in snap["samples"]],
        })
    return out


def _thumb(row: EmbeddingTraining, snapshots: list[dict]) -> str | None:
    """The first preview of the snapshot renders use — the one a picker tile
    should show, because it is what picking it will look like."""
    with_samples = [s for s in snapshots if s["samples"] and s["step"] > 0]
    if not with_samples:
        return None
    if row.chosen_step is not None:
        for snap in with_samples:
            if snap["step"] == row.chosen_step:
                return snap["samples"][0]
    return with_samples[-1]["samples"][0]


def _serialize(row: EmbeddingTraining, *, with_snapshots: bool = True) -> dict:
    snapshots = _snapshots(row) if with_snapshots else []
    live = _progress.get(str(row.id)) if row.status in _RUNNING else None
    return {
        "id": str(row.id),
        "name": row.name,
        "embedding": active_name(row.name),
        "status": row.status,
        "error": row.error,
        "series_id": str(row.series_id) if row.series_id else None,
        "image_count": len(row.image_ids or []),
        "template": row.template,
        "init_word": row.init_word,
        "vectors": row.vectors,
        "steps": row.steps,
        "learning_rate": row.learning_rate,
        "chosen_step": row.chosen_step,
        "seconds": row.seconds,
        "estimate_seconds": estimate_seconds(row.steps),
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "finished_at": row.finished_at.isoformat() if row.finished_at else None,
        "live": {k: v for k, v in (live or {}).items() if not k.startswith("_")} or None,
        "snapshots": snapshots,
        "thumb": _thumb(row, snapshots),
    }


async def _comfy_embeddings() -> list[str] | None:
    """What ComfyUI can load right now, or None when it is not reachable."""
    return await embedding_names()


async def _get_training(training_id: uuid.UUID, db: AsyncSession) -> EmbeddingTraining:
    row = await db.get(EmbeddingTraining, training_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Training nicht gefunden")
    return row


async def _resolve_images(body: TrainRequest, db: AsyncSession) -> list[uuid.UUID]:
    """The picture ids to train on, in order: the series' order when a series
    is given, the request's otherwise. Duplicates dropped — the same picture
    twice only weights it double."""
    ids: list[uuid.UUID] = []
    if body.series_id:
        items = (await db.execute(
            select(ImageSeriesItem.image_id)
            .where(ImageSeriesItem.series_id == body.series_id)
            .order_by(ImageSeriesItem.position)
        )).scalars().all()
        if not items:
            raise HTTPException(status_code=404, detail="Serie ist leer oder existiert nicht")
        ids.extend(items)
    ids.extend(body.image_ids)
    seen: set[uuid.UUID] = set()
    return [i for i in ids if not (i in seen or seen.add(i))]


# ── Library ──────────────────────────────────────────────────────────────────
@router.get("")
async def library(db: AsyncSession = Depends(get_db)):
    """Everything a render can name, trained ones first.

    A trained embedding appears once, by its active name — its snapshots are
    reachable from the training, not from the picker. Everything else ComfyUI
    finds (the Civitai files) is listed as it is.
    """
    rows = (await db.execute(
        select(EmbeddingTraining).order_by(desc(EmbeddingTraining.created_at))
    )).scalars().all()
    comfy = await _comfy_embeddings()

    items: list[dict] = []
    trained: set[str] = set()
    for row in rows:
        if row.status != "done" or not active_file(settings.embeddings_dir, row.name).is_file():
            continue
        serial = _serialize(row)
        trained.add(serial["embedding"])
        items.append({
            "name": serial["embedding"], "label": row.name, "trained": True,
            "training_id": serial["id"], "thumb": serial["thumb"],
        })
    for name in sorted(comfy or []):
        if name in trained or is_workshop_name(name):
            continue
        items.append({"name": name, "label": name.rsplit("/", 1)[-1],
                      "trained": False, "training_id": None, "thumb": None})

    return {
        "embeddings": items,
        "trainings": [_serialize(r) for r in rows],
        "comfy_online": comfy is not None,
        "options": _options(),
    }


# ── Training ─────────────────────────────────────────────────────────────────
@router.post("/trainings", status_code=202)
async def start_training(body: TrainRequest, db: AsyncSession = Depends(get_db)):
    busy = (await db.execute(
        select(EmbeddingTraining.name).where(EmbeddingTraining.status.in_(_RUNNING))
    )).scalars().first()
    if busy:
        raise HTTPException(status_code=409,
                            detail=f"Es läuft schon ein Training ({busy}) — eins nach dem anderen")

    image_ids = await _resolve_images(body, db)
    training_id = uuid.uuid4()
    try:
        plan = plan_training(
            training_id, body.name, len(image_ids),
            template=body.template, init_word=body.init_word, vectors=body.vectors,
            steps=body.steps, learning_rate=body.learning_rate,
        )
    except PlanError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    found = set((await db.execute(select(Image.id).where(Image.id.in_(image_ids)))).scalars())
    missing = [str(i) for i in image_ids if i not in found]
    if missing:
        raise HTTPException(status_code=404, detail=f"{len(missing)} Bild(er) nicht gefunden")

    taken = (await db.execute(
        select(EmbeddingTraining.id).where(EmbeddingTraining.name == plan.name)
    )).scalars().first()
    if taken or active_file(settings.embeddings_dir, plan.name).exists():
        raise HTTPException(status_code=409,
                            detail=f"Den Namen „{plan.name}“ gibt es schon")

    row = EmbeddingTraining(
        id=training_id, name=plan.name, token=plan.token, series_id=body.series_id,
        image_ids=[str(i) for i in image_ids], template=plan.template,
        init_word=plan.init_word, vectors=plan.vectors, steps=plan.steps,
        learning_rate=plan.learning_rate, status="queued",
    )
    db.add(row)
    await db.commit()

    _set_progress(str(training_id), "generating", "Training wird eingereiht…", 1)
    safe_create_task(_run_training(training_id, plan), name=f"embedding_training:{training_id}")
    logger.info("Queued embedding training %s (%s): %d pictures",
                training_id, plan.name, len(image_ids))
    return _serialize(row, with_snapshots=False)


async def _run_training(training_id: uuid.UUID, plan: TrainingPlan) -> None:
    key = str(training_id)
    started = time.monotonic()
    work_dir = settings.embedding_train_dir / key
    out_dir = train_dir(settings.embeddings_dir, plan.name)
    try:
        async with AsyncSessionLocal() as db:
            row = await db.get(EmbeddingTraining, training_id)
            if row is None:
                return
            ids = [uuid.UUID(i) for i in row.image_ids]
            images = {img.id: img for img in (await db.execute(
                select(Image).where(Image.id.in_(ids))
            )).scalars()}
            # The rendition under the grain: the crop, the upscale and the wand
            # are the user's decisions about the picture and belong in the
            # style; film grain at 512² is only noise, and would be learnt as
            # texture.
            sources = [grain_source_path(images[i]) for i in ids if i in images]
            row.status = "training"
            await db.commit()

        _set_progress(key, "generating", f"{len(sources)} Bilder werden vorbereitet…", 2)
        await asyncio.to_thread(prepare_dataset, sources, work_dir / "dataset")
        if out_dir.exists():
            # A leftover from a deleted training of the same name; snapshots
            # from it would be listed as this one's.
            shutil.rmtree(out_dir)

        _set_progress(key, "generating", "Modell wird geladen…", 4)

        def on_progress(step: int, total: int, loss: float | None) -> None:
            note = f" · loss {loss:.3f}" if loss is not None else ""
            _set_progress(key, "generating", f"Schritt {step}/{total}{note}",
                          5 + int(92 * step / max(1, total)))

        await run_training(plan, work_dir, out_dir, on_progress)

        final = out_dir / f"{plan.name}.safetensors"
        if not final.is_file():
            raise TrainingError("Das Training hat kein Embedding geschrieben")
        active = active_file(settings.embeddings_dir, plan.name)
        active.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(final, active)

        async with AsyncSessionLocal() as db:
            row = await db.get(EmbeddingTraining, training_id)
            if row is not None:
                row.status = "done"
                row.chosen_step = None
                row.seconds = round(time.monotonic() - started, 1)
                row.finished_at = datetime.now(timezone.utc)
                await db.commit()
        logger.info("Embedding %s trained in %.0f s", plan.name, time.monotonic() - started)

    except asyncio.CancelledError:
        # The cancel/delete endpoint owns the row from here: it waits for this
        # task to unwind and then records what the user asked for.
        raise
    except Exception as exc:
        logger.exception("Embedding training %s failed", training_id)
        async with AsyncSessionLocal() as db:
            row = await db.get(EmbeddingTraining, training_id)
            if row is not None:
                row.status = "failed"
                row.error = str(exc)[:2000]
                row.finished_at = datetime.now(timezone.utc)
                await db.commit()
    finally:
        # The row carries the outcome; a live entry would only keep a bar on
        # screen for a job that is over.
        forget_progress(key)


@router.get("/trainings/{training_id}")
async def get_training(training_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    return _serialize(await _get_training(training_id, db))


@router.get("/trainings/{training_id}/samples/{filename}")
async def get_sample(training_id: uuid.UUID, filename: str,
                     db: AsyncSession = Depends(get_db)):
    row = await _get_training(training_id, db)
    if not is_sample_filename(filename):
        raise HTTPException(status_code=400, detail="Kein Vorschaubild")
    path = train_dir(settings.embeddings_dir, row.name) / "sample" / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Vorschaubild fehlt")
    return FileResponse(path, media_type="image/png",
                        headers={"Cache-Control": "public, max-age=31536000, immutable"})


@router.post("/trainings/{training_id}/choose")
async def choose_snapshot(training_id: uuid.UUID, body: ChooseRequest,
                          db: AsyncSession = Depends(get_db)):
    """Make one snapshot the embedding renders use.

    A copy, not a rename: the snapshot stays in the workshop, so choosing
    another later is just another copy. Renders already queued read the file
    when ComfyUI encodes the prompt, so switching mid-queue is harmless.
    """
    row = await _get_training(training_id, db)
    if row.status != "done":
        raise HTTPException(status_code=409, detail="Erst wenn das Training fertig ist")
    folder = train_dir(settings.embeddings_dir, row.name)
    if body.step is None:
        source = folder / f"{row.name}.safetensors"
    else:
        source = folder / f"{row.name}-step{body.step:08d}.safetensors"
    if not source.is_file():
        raise HTTPException(status_code=404, detail="Diesen Snapshot gibt es nicht")
    shutil.copy2(source, active_file(settings.embeddings_dir, row.name))
    row.chosen_step = body.step
    await db.commit()
    return _serialize(row)


@router.post("/trainings/{training_id}/cancel")
async def cancel_training(training_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    row = await _get_training(training_id, db)
    if row.status not in _RUNNING:
        return _serialize(row)
    await cancel_job(training_id)
    forget_progress(str(training_id))
    await db.refresh(row)
    row.status = "cancelled"
    row.error = CANCELLED_MSG
    row.finished_at = datetime.now(timezone.utc)
    await db.commit()
    return _serialize(row)


@router.delete("/trainings/{training_id}", status_code=204)
async def delete_training(training_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """Stop it if it runs, then remove the row, the workshop and the active
    file. A render still naming the embedding afterwards gets ComfyUI's silent
    skip — which is why the picker reloads its list after a delete."""
    row = await _get_training(training_id, db)
    if row.status in _RUNNING:
        await cancel_job(training_id)
        forget_progress(str(training_id))
    shutil.rmtree(train_dir(settings.embeddings_dir, row.name), ignore_errors=True)
    active_file(settings.embeddings_dir, row.name).unlink(missing_ok=True)
    shutil.rmtree(settings.embedding_train_dir / str(training_id), ignore_errors=True)
    await db.delete(row)
    await db.commit()
