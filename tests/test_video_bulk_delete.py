"""Bulk video delete, against Postgres.

This needs a real database for exactly one reason: the interesting behaviour is
what happens when a delete is *refused*. `improv_sessions.source_video_id` is
`ON DELETE RESTRICT`, so a video that an improv session was recorded against
cannot go — and no mock reproduces that, because it is the database saying no.

Two properties are worth this much machinery:

  **One refusal must not take the batch down.** In a single transaction it
  would, and since the files are unlinked as part of the same operation, the
  survivors would come back as rows pointing at nothing.

  **Row first, files after.** A refused delete has to leave the video intact
  and playable. Deleting the files first turns "this video is still in use"
  into "this video is still listed and no longer plays", which is worse than
  either outcome.

Everything runs in a throwaway schema that is created and dropped per session,
so the live tables are never touched. If no database is reachable — CI, or a
laptop without Postgres — the whole module skips.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from core.config import settings
from core.db import Base
from core.models import ImprovSession, Video

SCHEMA = "artrium_test_bulk_delete"

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
        await cleanup.execute(text("DELETE FROM improv_sessions"))
        await cleanup.execute(text("DELETE FROM videos"))
        await cleanup.commit()


@pytest.fixture
def videos_dir(tmp_path, monkeypatch):
    """Point the router's file helpers at a temp directory.

    The delete unlinks whatever `_video_owned_paths` names, so this has to be
    redirected or the test would reach into real storage.
    """
    # Only `storage_dir` is a field — `videos_dir` is derived from it, so
    # patching the root is what moves the whole tree.
    monkeypatch.setattr(settings, "storage_dir", tmp_path)
    d = tmp_path / "videos"
    d.mkdir(parents=True, exist_ok=True)
    return d


class Made:
    """A created video, as plain values.

    Deliberately not the ORM object. A refused delete ends in `rollback()`,
    which expires every instance in the session — and reading `.filename` off
    an expired instance is a lazy load, which outside an await is a
    `MissingGreenlet` rather than the assertion the test meant to make.
    """

    def __init__(self, vid, name, paths):
        self.id, self.filename, self.paths = vid, name, paths


async def _make_video(db, videos_dir, **kw) -> Made:
    """A video row with real files behind it, so unlinking can be observed."""
    vid = uuid.uuid4()
    name = f"{vid}_artrium.mp4"
    (videos_dir / name).write_bytes(b"video")
    (videos_dir / f"{vid}_thumb.jpg").write_bytes(b"thumb")
    seg = videos_dir / "segments" / str(vid)
    seg.mkdir(parents=True, exist_ok=True)
    (seg / "seg_0.mp4").write_bytes(b"clip")
    db.add(Video(
        id=vid, workflow="i2v", status="done", filename=name,
        # Relative to storage_dir, which is how _video_owned_paths reads it.
        filepath=f"videos/{name}", created_at=datetime.now(timezone.utc), **kw,
    ))
    await db.commit()
    return Made(vid, name, [
        videos_dir / name,
        videos_dir / f"{vid}_thumb.jpg",
        videos_dir / "segments" / str(vid) / "seg_0.mp4",
    ])


class TestBulkDelete:
    async def test_several_videos_go_in_one_request(self, db, videos_dir):
        from routers.video import BulkDeleteVideosRequest, bulk_delete_videos
        made = [await _make_video(db, videos_dir) for _ in range(3)]
        ids = [v.id for v in made]

        res = await bulk_delete_videos(BulkDeleteVideosRequest(ids=ids), db)

        assert sorted(res["deleted"]) == sorted(str(i) for i in ids)
        assert res["failed"] == []
        for v in made:
            assert await db.get(Video, v.id) is None
            for f in v.paths:
                assert not f.exists(), f

    async def test_a_video_that_is_already_gone_counts_as_deleted(self, db, videos_dir):
        # The gallery can be stale. Asking to delete something that is not
        # there is a satisfied request, not an error — and the tile should go.
        from routers.video import BulkDeleteVideosRequest, bulk_delete_videos
        missing = uuid.uuid4()
        res = await bulk_delete_videos(BulkDeleteVideosRequest(ids=[missing]), db)
        assert res["deleted"] == [str(missing)]
        assert res["failed"] == []

    async def test_a_duplicated_id_is_collapsed(self, db, videos_dir):
        from routers.video import BulkDeleteVideosRequest, bulk_delete_videos
        v = await _make_video(db, videos_dir)
        res = await bulk_delete_videos(
            BulkDeleteVideosRequest(ids=[v.id, v.id, v.id]), db)
        assert res["deleted"] == [str(v.id)]

    async def test_an_empty_request_does_nothing(self, db, videos_dir):
        from routers.video import BulkDeleteVideosRequest, bulk_delete_videos
        res = await bulk_delete_videos(BulkDeleteVideosRequest(ids=[]), db)
        assert res == {"deleted": [], "failed": []}

    async def test_one_refusal_does_not_take_the_batch_down(self, db, videos_dir):
        # The property the whole design is for. The middle video is referenced
        # by an improv session (ON DELETE RESTRICT); its neighbours must still
        # go, and it must come back as a reported failure rather than a 500.
        from routers.video import BulkDeleteVideosRequest, bulk_delete_videos
        before = await _make_video(db, videos_dir)
        held = await _make_video(db, videos_dir)
        after = await _make_video(db, videos_dir)
        db.add(ImprovSession(
            id=uuid.uuid4(), source_video_id=held.id,
            recording_filename="take.wav", status="done",
            created_at=datetime.now(timezone.utc),
        ))
        await db.commit()

        res = await bulk_delete_videos(
            BulkDeleteVideosRequest(ids=[before.id, held.id, after.id]), db)

        assert sorted(res["deleted"]) == sorted([str(before.id), str(after.id)])
        assert [f["id"] for f in res["failed"]] == [str(held.id)]
        assert await db.get(Video, before.id) is None
        assert await db.get(Video, after.id) is None
        assert await db.get(Video, held.id) is not None

    async def test_a_refused_video_keeps_its_files(self, db, videos_dir):
        # Row first, files after. If this ever regresses, the video stays in
        # the gallery and no longer plays — the worst of both outcomes.
        from routers.video import BulkDeleteVideosRequest, bulk_delete_videos
        held = await _make_video(db, videos_dir)
        db.add(ImprovSession(
            id=uuid.uuid4(), source_video_id=held.id,
            recording_filename="take.wav", status="done",
            created_at=datetime.now(timezone.utc),
        ))
        await db.commit()

        res = await bulk_delete_videos(BulkDeleteVideosRequest(ids=[held.id]), db)

        assert res["deleted"] == []
        assert len(res["failed"]) == 1
        assert res["failed"][0]["reason"]
        for f in held.paths:
            assert f.exists(), f

    async def test_every_rendition_is_removed_not_just_the_original(
        self, db, videos_dir,
    ):
        from routers.video import BulkDeleteVideosRequest, bulk_delete_videos
        made = await _make_video(db, videos_dir)
        v = await db.get(Video, made.id)
        for attr, name in (("muxed_filename", "m.mp4"),
                           ("upscale_filename", "u.mp4"),
                           ("grain_filename", "g.mp4")):
            (videos_dir / name).write_bytes(b"x")
            setattr(v, attr, name)
        await db.commit()

        await bulk_delete_videos(BulkDeleteVideosRequest(ids=[made.id]), db)

        for name in ("m.mp4", "u.mp4", "g.mp4"):
            assert not (videos_dir / name).exists(), name


class TestSingleDeleteOrdering:
    """The same ordering rule, on the endpoint the detail sheet uses."""

    async def test_a_refused_single_delete_leaves_the_video_playable(
        self, db, videos_dir,
    ):
        import sqlalchemy.exc

        from routers.video import delete_video
        held = await _make_video(db, videos_dir)
        db.add(ImprovSession(
            id=uuid.uuid4(), source_video_id=held.id,
            recording_filename="take.wav", status="done",
            created_at=datetime.now(timezone.utc),
        ))
        await db.commit()

        with pytest.raises(sqlalchemy.exc.IntegrityError):
            await delete_video(held.id, db)
        await db.rollback()

        for f in held.paths:
            assert f.exists(), f
        assert await db.get(Video, held.id) is not None

    async def test_a_normal_single_delete_still_removes_everything(
        self, db, videos_dir,
    ):
        from routers.video import delete_video
        v = await _make_video(db, videos_dir)
        await delete_video(v.id, db)
        assert await db.get(Video, v.id) is None
        for f in v.paths:
            assert not f.exists(), f
