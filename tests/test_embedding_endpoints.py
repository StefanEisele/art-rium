"""The embedding trainer's endpoints (routers/embeddings.py) against Postgres.

The trainer itself is replaced by a fake that writes exactly the files the real
smoke run wrote — snapshots, a final save, previews with the "before" row — so
what is proven here is everything around it: the row's life, the active file,
choosing a snapshot, deleting, the one-at-a-time rule, and a failure landing on
the row rather than vanishing into a log.

Throwaway schema, tmp storage and a tmp ComfyUI folder; skips without a
reachable database.
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone

import pytest
import pytest_asyncio
from fastapi import HTTPException
from PIL import Image as PILImage
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from core import tasks
from core.config import settings
from core.db import Base
from core.models import EmbeddingTraining, Image, ImageSeries, ImageSeriesItem
from routers import embeddings as router
from services.embedding import TrainingError

SCHEMA = "artrium_test_embeddings"

pytestmark = pytest.mark.asyncio(loop_scope="module")


@pytest_asyncio.fixture(loop_scope="module", scope="module")
async def sessionmaker_():
    try:
        engine = create_async_engine(
            settings.database_url,
            connect_args={"server_settings": {"search_path": SCHEMA}},
        )
        async with engine.begin() as conn:
            await conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}"))
            await conn.run_sync(Base.metadata.create_all)
    except Exception as exc:                        # no database on this machine
        pytest.skip(f"No Postgres reachable: {exc}")
    yield async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as conn:
        await conn.execute(text(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE"))
    await engine.dispose()


@pytest.fixture
def dirs(tmp_path, monkeypatch, sessionmaker_):
    monkeypatch.setattr(settings, "storage_dir", tmp_path / "storage")
    monkeypatch.setattr(settings, "comfyui_dir", tmp_path / "comfy")
    # The background task opens its own sessions; they must see the test schema.
    monkeypatch.setattr(router, "AsyncSessionLocal", sessionmaker_)
    return tmp_path


@pytest_asyncio.fixture(loop_scope="module")
async def db(sessionmaker_):
    async with sessionmaker_() as session:
        yield session
        await session.rollback()
    async with sessionmaker_() as cleanup:
        for table in ("embedding_trainings", "image_series_items", "image_series", "images"):
            await cleanup.execute(text(f"DELETE FROM {table}"))
        await cleanup.commit()


async def _series(db, n: int) -> ImageSeries:
    series = ImageSeries(id=uuid.uuid4(), title="Echoes")
    db.add(series)
    for i in range(n):
        ident = uuid.uuid4()
        rel = f"images/2026/09/{ident.hex}.png"
        path = settings.storage_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        PILImage.new("RGB", (216, 384), (180, 90, 40)).save(path)
        db.add(Image(id=ident, filename=f"{ident.hex}.png", filepath=rel, width=216,
                     height=384, created_at=datetime.now(timezone.utc)))
        await db.flush()
        db.add(ImageSeriesItem(series_id=series.id, image_id=ident, position=i))
    await db.commit()
    return series


def fake_trainer(calls: list, *, fail: str | None = None):
    """Writes what sd-scripts wrote in the smoke run, at the plan's cadence."""
    async def run(plan, work_dir, output_dir, on_progress=None, timeout=0):
        calls.append({"plan": plan, "dataset": sorted((work_dir / "dataset").glob("*.png"))})
        if fail:
            raise TrainingError(fail)
        (output_dir / "sample").mkdir(parents=True, exist_ok=True)
        for step in (0, 250, 500):
            prefix = "e" if step == 0 else ""
            for idx in range(3):
                (output_dir / "sample" /
                 f"{plan.name}_{prefix}{step:06d}_{idx:02d}_20260928203350_1234.png").write_bytes(b"png")
            if step:
                (output_dir / f"{plan.name}-step{step:08d}.safetensors").write_bytes(f"s{step}".encode())
            if on_progress:
                on_progress(step, 500, 0.1)
        (output_dir / f"{plan.name}.safetensors").write_bytes(b"final")
    return run


async def _finish(training_id: str) -> None:
    for task in tasks.tasks_for(training_id):
        await task


async def _train(db, series, monkeypatch, calls, **kw) -> dict:
    monkeypatch.setattr(router, "run_training", fake_trainer(calls, **kw))
    body = router.TrainRequest(name="Echos", series_id=series.id, steps=500)
    out = await router.start_training(body, db=db)
    await _finish(out["id"])
    return out


async def test_a_series_trains_into_an_active_embedding(db, dirs, monkeypatch):
    series = await _series(db, 4)
    calls: list = []
    out = await _train(db, series, monkeypatch, calls)

    assert out["status"] == "queued" and out["embedding"] == "artrium/echos"
    assert len(calls[0]["dataset"]) == 4, "every series picture is prepared"
    row = await router.get_training(uuid.UUID(out["id"]), db=db)
    assert row["status"] == "done" and row["seconds"] is not None
    assert [s["step"] for s in row["snapshots"]] == [0, 250, 500]
    assert row["snapshots"][1]["embedding"] == "artrium/_train/echos/echos-step00000250"
    assert row["snapshots"][0]["embedding"] is None, "step 0 is the untrained init word"
    assert row["thumb"].endswith("echos_000500_00_20260928203350_1234.png")
    active = settings.embeddings_dir / "artrium" / "echos.safetensors"
    assert active.read_bytes() == b"final"


async def test_choosing_a_snapshot_copies_it_to_the_active_file(db, dirs, monkeypatch):
    series = await _series(db, 3)
    out = await _train(db, series, monkeypatch, [])
    tid = uuid.UUID(out["id"])
    active = settings.embeddings_dir / "artrium" / "echos.safetensors"

    chosen = await router.choose_snapshot(tid, router.ChooseRequest(step=250), db=db)
    assert active.read_bytes() == b"s250"
    assert chosen["chosen_step"] == 250
    assert chosen["thumb"].endswith("echos_000250_00_20260928203350_1234.png")

    await router.choose_snapshot(tid, router.ChooseRequest(step=None), db=db)
    assert active.read_bytes() == b"final"

    with pytest.raises(HTTPException) as err:
        await router.choose_snapshot(tid, router.ChooseRequest(step=999), db=db)
    assert err.value.status_code == 404


async def test_the_library_offers_a_trained_embedding_once(db, dirs, monkeypatch):
    series = await _series(db, 3)
    await _train(db, series, monkeypatch, [])

    async def comfy():
        return ["artrium/echos", "artrium/_train/echos/echos",
                "artrium/_train/echos/echos-step00000250", "style-rustmagic"]
    monkeypatch.setattr(router, "_comfy_embeddings", comfy)
    lib = await router.library(db=db)
    names = [e["name"] for e in lib["embeddings"]]
    assert names == ["artrium/echos", "style-rustmagic"]
    assert lib["embeddings"][0]["trained"] and lib["embeddings"][0]["thumb"]
    assert lib["options"]["vectors"]["max"] == 75


async def test_a_failure_lands_on_the_row(db, dirs, monkeypatch):
    series = await _series(db, 3)
    out = await _train(db, series, monkeypatch, [], fail="CUDA out of memory")
    row = await router.get_training(uuid.UUID(out["id"]), db=db)
    assert row["status"] == "failed" and "out of memory" in row["error"]
    assert not (settings.embeddings_dir / "artrium" / "echos.safetensors").exists()


async def test_one_training_at_a_time(db, dirs, monkeypatch):
    series = await _series(db, 3)
    db.add(EmbeddingTraining(name="busy", token="arx", image_ids=[], template="style",
                             init_word="painting", vectors=8, steps=500,
                             learning_rate=5e-3, status="training"))
    await db.commit()
    with pytest.raises(HTTPException) as err:
        await router.start_training(router.TrainRequest(name="next", series_id=series.id), db=db)
    assert err.value.status_code == 409 and "busy" in err.value.detail


async def test_too_few_pictures_and_a_taken_name(db, dirs, monkeypatch):
    small = await _series(db, 2)
    with pytest.raises(HTTPException) as err:
        await router.start_training(router.TrainRequest(name="x2", series_id=small.id), db=db)
    assert err.value.status_code == 400

    series = await _series(db, 3)
    await _train(db, series, monkeypatch, [])
    with pytest.raises(HTTPException) as err:
        await router.start_training(router.TrainRequest(name="ECHOS", series_id=series.id), db=db)
    assert err.value.status_code == 409


async def test_delete_removes_the_row_and_every_file(db, dirs, monkeypatch):
    series = await _series(db, 3)
    out = await _train(db, series, monkeypatch, [])
    tid = uuid.UUID(out["id"])
    await router.delete_training(tid, db=db)
    assert not (settings.embeddings_dir / "artrium" / "echos.safetensors").exists()
    assert not (settings.embeddings_dir / "artrium" / "_train" / "echos").exists()
    assert not (settings.embedding_train_dir / out["id"]).exists()
    with pytest.raises(HTTPException):
        await router.get_training(tid, db=db)


async def test_cancel_stops_a_running_training(db, dirs, monkeypatch):
    series = await _series(db, 3)
    started = asyncio.Event()

    async def hangs(plan, work_dir, output_dir, on_progress=None, timeout=0):
        started.set()
        await asyncio.sleep(3600)
    monkeypatch.setattr(router, "run_training", hangs)
    out = await router.start_training(router.TrainRequest(name="slow", series_id=series.id), db=db)
    await asyncio.wait_for(started.wait(), 5)

    row = await router.cancel_training(uuid.UUID(out["id"]), db=db)
    assert row["status"] == "cancelled"
    assert not tasks.tasks_for(out["id"])
