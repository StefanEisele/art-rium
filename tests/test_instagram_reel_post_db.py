"""Why scheduling a reel used to 500, against Postgres.

`InstagramPost.media` is `lazy="selectin"`, which loads eagerly on a *query*
and does nothing for an object this process just created. The feed branch of
create_post fills the collection before the flush, so it is loaded by the time
anything reads it. The reel branch had nothing to fill — a reel has no carousel
children — so `post.media` stayed unloaded, and the first read after the flush
(crop baking, then serialization) made SQLAlchemy emit a SELECT from inside
async code: MissingGreenlet, 500, no post scheduled.

Mocks cannot show this. The whole bug is the difference between "empty" and
"loaded and empty" on a row that has just become persistent, which only a real
session against a real database has. Runs in a throwaway schema; skips when no
Postgres is reachable.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio
from sqlalchemy import inspect, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from core.config import settings
from core.db import Base
from core.models import InstagramPost
from services.instagram.crops import ensure_post_crops

SCHEMA = "artrium_test_ig_reel"

pytestmark = pytest.mark.asyncio(loop_scope="module")


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def sessionmaker_():
    engine = create_async_engine(
        settings.database_url,
        connect_args={"server_settings": {"search_path": SCHEMA}},
    )
    try:
        async with engine.begin() as conn:
            await conn.execute(text(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE"))
            await conn.execute(text(f"CREATE SCHEMA {SCHEMA}"))
    except Exception as exc:                              # pragma: no cover
        await engine.dispose()
        pytest.skip(f"No Postgres for the reel-post integration tests: {exc}")

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield async_sessionmaker(engine, expire_on_commit=False)

    async with engine.begin() as conn:
        await conn.execute(text(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE"))
    await engine.dispose()


@pytest_asyncio.fixture(loop_scope="module")
async def db(sessionmaker_):
    async with sessionmaker_() as session:
        await session.execute(text(f"TRUNCATE {SCHEMA}.instagram_posts CASCADE"))
        await session.commit()
        yield session


def _reel_row(**kw) -> InstagramPost:
    """A kind='reel' row built exactly the way routers/instagram.py builds it."""
    now = datetime.now(timezone.utc)
    return InstagramPost(
        kind="reel",
        reel_video_ids=[uuid.uuid4()],
        caption="test",
        scheduled_at=now + timedelta(days=1),
        status="scheduled",
        dispatch_target="outpost",
        frame_ratio="auto",
        ai_label=True,
        created_at=now,
        updated_at=now,
        **kw,
    )


async def test_reel_row_without_the_assignment_leaves_media_unloaded(db):
    """The trap itself: after the flush the collection is not there yet."""
    post = _reel_row()
    db.add(post)
    await db.flush()
    assert "media" in inspect(post).unloaded


async def test_assigning_an_empty_list_settles_the_collection(db):
    post = _reel_row()
    post.media = []
    db.add(post)
    await db.flush()
    assert "media" not in inspect(post).unloaded


async def test_crop_baking_survives_the_flush_on_a_reel(db):
    """What actually 500'd. ensure_post_crops sorts post.media first thing."""
    post = _reel_row()
    post.media = []
    db.add(post)
    await db.flush()
    # 'auto' with no children falls back to the frame default rather than
    # asking a child for its shape — a reel has no children to ask.
    assert await ensure_post_crops(post, db) > 0


async def test_an_unassigned_reel_still_reproduces_the_original_failure(db):
    """Pins the diagnosis: without the assignment this is a MissingGreenlet,
    not some unrelated error. If a future SQLAlchemy stops emitting IO here,
    this test says so out loud instead of the fix quietly becoming cargo."""
    from sqlalchemy.exc import MissingGreenlet

    post = _reel_row()
    db.add(post)
    await db.flush()
    with pytest.raises(MissingGreenlet):
        await ensure_post_crops(post, db)
