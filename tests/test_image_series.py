"""Image series, against Postgres.

A real database earns its keep here because almost every interesting property
is one the database enforces, and no mock reproduces a constraint saying no.

The one that would otherwise reach the browser: `Image.series_items` is
eagerly loaded, and both delete paths in routers/images.py hand the row to the
ORM's `db.delete()`. Get the cascade wrong and SQLAlchemy de-associates the
children the default way — UPDATE ... SET image_id = NULL against a NOT NULL
column — so *deleting any picture that is in a series* fails, taking single
delete, bulk delete, the undo bar and the cull deck with it. Two tests below
pin exactly that, one per delete shape, because the bulk path is the one that
has already loaded the collection when it deletes.

The rest: gaps in `position` are legal and appending must survive them,
reordering must not trip UNIQUE(series_id, position) mid-flush, and an
emptied series is kept rather than swept up.

Everything runs in a throwaway schema created and dropped per session, so the
live tables are never touched. Without a reachable database the module skips.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
import pytest_asyncio
from sqlalchemy import func, inspect, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.orm import selectinload

from core.config import settings
from core.db import Base
from core.models import Image, ImageSeries, ImageSeriesItem

SCHEMA = "artrium_test_image_series"

pytestmark = pytest.mark.asyncio(loop_scope="module")


def _engine():
    return create_async_engine(
        settings.database_url,
        connect_args={"server_settings": {"search_path": SCHEMA}},
    )


@pytest_asyncio.fixture(loop_scope="module", scope="module")
async def sessionmaker_():
    try:
        engine = _engine()
        async with engine.begin() as conn:
            await conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}"))
            await conn.run_sync(Base.metadata.create_all)
    except Exception as exc:                        # no database on this machine
        pytest.skip(f"No Postgres reachable: {exc}")
    yield async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as conn:
        await conn.execute(text(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE"))
    await engine.dispose()


@pytest_asyncio.fixture(loop_scope="module")
async def db(sessionmaker_):
    async with sessionmaker_() as session:
        yield session
        await session.rollback()
    async with sessionmaker_() as cleanup:
        await cleanup.execute(text("DELETE FROM image_series_items"))
        await cleanup.execute(text("DELETE FROM image_series"))
        await cleanup.execute(text("DELETE FROM images"))
        await cleanup.commit()


@pytest_asyncio.fixture(loop_scope="module")
async def fresh_db(sessionmaker_):
    """A second session, for the half of a test that must not see the first.

    This is not ceremony. `expire_on_commit` is False here as it is in the
    app, so a session that *built* a series holds those Image rows with
    `series_items` still as the empty list they were constructed with — a
    re-select hands back the identity-mapped object without refreshing it.
    Deleting through that session therefore finds no children and proves
    nothing. One request, one session; the delete gets its own.
    """
    async with sessionmaker_() as session:
        yield session
        await session.rollback()


def _image(n: int) -> Image:
    ident = uuid.uuid4()
    return Image(
        id=ident,
        filename=f"test_{n}_{ident.hex[:8]}.png",
        filepath=f"images/2026/09/{ident.hex}.png",
        created_at=datetime.now(timezone.utc),
        series_items=[],
    )


async def _make_series(db, count: int, title: str = "Rostküste"):
    images = [_image(i) for i in range(count)]
    for img in images:
        db.add(img)
    await db.flush()
    series = ImageSeries(title=title)
    db.add(series)
    for position, img in enumerate(images):
        series.items.append(ImageSeriesItem(position=position, image_id=img.id))
    await db.commit()
    return series, images


async def _positions(db, series_id) -> list[int]:
    rows = await db.execute(
        select(ImageSeriesItem.position)
        .where(ImageSeriesItem.series_id == series_id)
        .order_by(ImageSeriesItem.position)
    )
    return [r[0] for r in rows]


# ── The regression that would otherwise 500 in the gallery ───────────────────


async def test_deleting_a_member_leaves_the_series_standing(db, fresh_db):
    """Single delete, the path behind the detail modal and the undo bar."""
    series, images = await _make_series(db, 3)

    victim = (await fresh_db.execute(
        select(Image).where(Image.id == images[1].id)
    )).scalar_one()
    assert victim.series_items, "the membership is the premise of this test"
    await fresh_db.delete(victim)
    await fresh_db.commit()                 # the null-out would raise here

    assert await fresh_db.get(Image, images[1].id) is None
    assert await fresh_db.get(ImageSeries, series.id) is not None
    assert await _positions(fresh_db, series.id) == [0, 2]


async def test_bulk_deleting_members_leaves_the_series_standing(db, fresh_db):
    """Bulk delete — the shape routers/images.py::bulk_delete_images uses.

    Distinct from the test above because the `select(...).in_()` resolves
    several rows at once, each arriving with `series_items` eagerly loaded —
    the state in which a wrong cascade tries to null the children's FK.
    """
    series, images = await _make_series(db, 4)

    rows = (await fresh_db.execute(
        select(Image).where(Image.id.in_([images[0].id, images[2].id]))
    )).scalars().all()
    assert len(rows) == 2
    assert all("series_items" not in inspect(r).unloaded for r in rows), \
        "the eager load is the premise of this test"
    assert all(r.series_items for r in rows)
    for row in rows:
        await fresh_db.delete(row)
    await fresh_db.commit()

    assert await _positions(fresh_db, series.id) == [1, 3]
    assert await fresh_db.get(ImageSeries, series.id) is not None


async def test_deleting_the_series_leaves_every_picture(db, fresh_db):
    series, images = await _make_series(db, 3)

    await fresh_db.delete(await fresh_db.get(ImageSeries, series.id))
    await fresh_db.commit()

    for img in images:
        assert await fresh_db.get(Image, img.id) is not None
    assert await _positions(fresh_db, series.id) == []


# ── Positions ────────────────────────────────────────────────────────────────


async def test_appending_after_a_gap_uses_max_position(db, fresh_db):
    """`len(items)` would pick an occupied position once a picture is gone."""
    series, images = await _make_series(db, 3)
    victim = (await fresh_db.execute(
        select(Image).where(Image.id == images[1].id)
    )).scalar_one()
    await fresh_db.delete(victim)
    await fresh_db.commit()
    assert await _positions(db, series.id) == [0, 2]

    next_pos = (await db.execute(
        select(func.coalesce(func.max(ImageSeriesItem.position), -1))
        .where(ImageSeriesItem.series_id == series.id)
    )).scalar_one() + 1
    assert next_pos == 3, "three members minus one is not the next free slot"

    extra = _image(99)
    db.add(extra)
    await db.flush()
    fresh = (await db.execute(
        select(ImageSeries)
        .where(ImageSeries.id == series.id)
        .options(selectinload(ImageSeries.items))
    )).scalar_one()
    fresh.items.append(ImageSeriesItem(position=next_pos, image_id=extra.id))
    await db.commit()

    assert await _positions(db, series.id) == [0, 2, 3]


async def test_reordering_through_clear_and_flush_survives_the_constraint(db):
    """A swap issued as UPDATEs can collide mid-flush; this path cannot."""
    series, images = await _make_series(db, 4)
    reversed_ids = [img.id for img in reversed(images)]

    fresh = (await db.execute(
        select(ImageSeries)
        .where(ImageSeries.id == series.id)
        .options(selectinload(ImageSeries.items))
    )).scalar_one()
    fresh.items.clear()
    await db.flush()                        # without this, UNIQUE trips
    for position, image_id in enumerate(reversed_ids):
        fresh.items.append(ImageSeriesItem(position=position, image_id=image_id))
    await db.commit()

    rows = (await db.execute(
        select(ImageSeriesItem)
        .where(ImageSeriesItem.series_id == series.id)
        .order_by(ImageSeriesItem.position)
    )).scalars().all()
    assert [r.image_id for r in rows] == reversed_ids


# ── Uniqueness ───────────────────────────────────────────────────────────────


async def test_the_same_picture_twice_in_one_series_is_refused(db):
    series, images = await _make_series(db, 2)

    db.add(ImageSeriesItem(series_id=series.id, image_id=images[0].id, position=9))
    with pytest.raises(IntegrityError):
        await db.commit()
    await db.rollback()


async def test_one_picture_may_sit_in_two_series(db):
    series, images = await _make_series(db, 2)

    other = ImageSeries(title="Blaue Stunde")
    db.add(other)
    for position, img in enumerate(images):
        other.items.append(ImageSeriesItem(position=position, image_id=img.id))
    await db.commit()

    memberships = (await db.execute(
        select(func.count()).select_from(ImageSeriesItem)
        .where(ImageSeriesItem.image_id == images[0].id)
    )).scalar_one()
    assert memberships == 2


# ── The eager-load contract _serialize depends on ────────────────────────────


async def test_a_flushed_image_has_its_memberships_settled(db):
    """`db.add(Image(...)); flush()` does NOT settle an eager collection.

    The same lesson as InstagramPost.media: a flush leaves it unloaded, and
    reading it afterwards in async code raises MissingGreenlet rather than
    querying. Passing `series_items=[]` at construction — which
    services/comfy/ingest.py does — is what makes a freshly ingested picture
    safe to serialize.
    """
    bare = Image(
        id=uuid.uuid4(),
        filename=f"bare_{uuid.uuid4().hex[:8]}.png",
        filepath="images/2026/09/bare.png",
        created_at=datetime.now(timezone.utc),
    )
    db.add(bare)
    await db.flush()
    assert "series_items" in inspect(bare).unloaded

    settled = _image(1)                     # built with series_items=[]
    db.add(settled)
    await db.flush()
    assert "series_items" not in inspect(settled).unloaded
    assert settled.series_items == []
    await db.rollback()
