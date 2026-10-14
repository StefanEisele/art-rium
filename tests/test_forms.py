"""The Formen tool's server half: loop arithmetic, formats, the mask check and
the store endpoint.

The invariants that matter, because each one fails silently downstream:

  - a loop is 4n+1 frames, so the Struktur request path (which snaps every
    length to 4n+1) plays all of it — a 48-frame loop would lose three frames
    and the seam would jump;
  - it is stored at AnimateLCM's own base rate, so the track's duration is the
    render's;
  - the mask is written lossless and checked for colours that key into no
    region — an anti-aliased edge would become a hole in the render.

The endpoint tests run against Postgres (throwaway schema) and real ffmpeg;
they skip without a database.
"""
from __future__ import annotations

import io
import json
import subprocess
import uuid

import pytest
import pytest_asyncio
from fastapi import HTTPException, UploadFile
from PIL import Image as PILImage
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from core.config import settings
from core.db import Base
from core.video_thumb import _ffprobe_path
from routers import forms as forms_router
from routers import vace as vace_router
from services.comfy.animatelcm import FPS as LCM_FPS
from services.comfy.animatelcm import canvas_size
from services.comfy.wan_frames import loop_length
from services.forms import (
    DURATIONS,
    FORMATS,
    MAX_FRAMES,
    MIN_FRAMES,
    TRACK_LONG_EDGE,
    encode_command,
    format_table,
    loop_frames,
    mask_regions,
    track_size,
)
from services.forms.plan import KEY_THRESHOLD, _key_of


# ── Loop arithmetic ─────────────────────────────────────────────────────────

class TestLoopFrames:
    @pytest.mark.parametrize("seconds,frames", [(4, 33), (6, 49), (8, 65), (10, 81)])
    def test_the_offered_lengths(self, seconds, frames):
        assert loop_frames(seconds) == frames

    @pytest.mark.parametrize("seconds", [1, 2.5, 3, 4, 5, 6, 7.3, 8, 9, 10])
    def test_the_render_path_keeps_every_frame(self, seconds):
        """routers/vace.py clamps a job to loop_length(track frames). A loop
        built here must come through it whole, or the wrap skips frames."""
        n = loop_frames(seconds)
        assert (n - 1) % 4 == 0
        assert loop_length(n) == n

    def test_duration_is_the_render_rate(self):
        # Within half a frame-group of the asked length, at AnimateLCM's rate.
        for s in DURATIONS:
            assert abs(loop_frames(s) / LCM_FPS - s) <= 2 / LCM_FPS

    def test_bounds(self):
        assert MIN_FRAMES == loop_frames(1) == 9
        assert MAX_FRAMES == loop_frames(DURATIONS[-1]) == 81


class TestFormats:
    @pytest.mark.parametrize("key,rw,rh", [(k, w, h) for k, _, w, h in FORMATS])
    def test_track_keeps_the_ratio(self, key, rw, rh):
        w, h = track_size(rw, rh)
        assert max(w, h) == TRACK_LONG_EDGE
        assert w % 8 == 0 and h % 8 == 0
        assert w / h == pytest.approx(rw / rh, rel=0.012)

    def test_the_render_canvas_follows_the_track(self):
        """'auto' canvas keeps the track's orientation and roughly its ratio —
        the format picked here is the format that comes back."""
        for row in format_table():
            tw, th = row["track"]
            cw, ch = row["canvas"]
            if tw == th:
                assert cw == ch
            else:
                assert (cw > ch) == (tw > th)
            assert cw / ch == pytest.approx(tw / th, rel=0.06)
            assert tuple(row["canvas"]) == canvas_size("auto", (tw, th))
            assert row["output"][0] >= cw and row["output"][1] >= ch

    def test_nine_sixteen_is_a_clean_reel(self):
        assert track_size(9, 16) == (576, 1024)


# ── Mask check ──────────────────────────────────────────────────────────────

def _mask_frame(path, size=(40, 20), aa=False):
    """Left half red, right quarter green, a blue strip along the bottom,
    black elsewhere — or with a ramp of in-between colours at one edge."""
    im = PILImage.new("RGB", size, (0, 0, 0))
    w, h = size
    for y in range(h):
        for x in range(w):
            if y >= h - 4:
                im.putpixel((x, y), (0, 0, 255))
            elif x < w // 2:
                im.putpixel((x, y), (255, 0, 0))
            elif x >= 3 * w // 4:
                im.putpixel((x, y), (0, 255, 0))
    if aa:
        for y in range(h - 4):
            im.putpixel((w // 2, y), (128, 0, 0))     # half-covered edge pixel
    im.save(path)
    return path


class TestMaskRegions:
    def test_keys(self):
        assert _key_of((0, 0, 0)) == -1
        assert _key_of((254, 0, 0)) == 0
        assert _key_of((0, 255, 1)) == 1
        assert _key_of((0, 0, 255 - KEY_THRESHOLD)) == 2
        assert _key_of((128, 0, 0)) is None

    def test_coverage_frames_and_labels(self, tmp_path):
        frames = [_mask_frame(tmp_path / f"m_{i}.png") for i in range(3)]
        regions, stray = mask_regions(frames, ["Kugeln", "Ringe"])
        assert stray == 0.0
        assert [r["label"] for r in regions] == ["Kugeln", "Ringe", "Umgebung"]
        assert [r["frames"] for r in regions] == [3, 3, 3]
        red, green, blue = (r["coverage"] for r in regions)
        assert red == pytest.approx(20 * 16 / 800, abs=1e-4)
        assert green == pytest.approx(10 * 16 / 800, abs=1e-4)
        assert blue == pytest.approx(40 * 4 / 800, abs=1e-4)

    def test_only_colours_that_occur(self, tmp_path):
        path = tmp_path / "m.png"
        PILImage.new("RGB", (8, 8), (255, 0, 0)).save(path)
        regions, _ = mask_regions([path])
        assert [r["color"] for r in regions] == [[255, 0, 0]]

    def test_an_antialiased_edge_is_counted(self, tmp_path):
        frames = [_mask_frame(tmp_path / "m.png", aa=True)]
        _, stray = mask_regions(frames)
        assert stray == pytest.approx(16 / 800, abs=1e-4)


class TestEncodeCommand:
    def test_depth(self, tmp_path):
        args = encode_command(tmp_path / "d_%04d.png", tmp_path / "o.mp4", 49, lossless=False)
        assert args[args.index("-framerate") + 1] == str(LCM_FPS)
        assert args[args.index("-frames:v") + 1] == "49"
        assert "-crf" in args and "yuv420p" in args

    def test_mask_is_lossless_444(self, tmp_path):
        args = encode_command(tmp_path / "m_%04d.png", tmp_path / "o.mp4", 49, lossless=True)
        assert args[args.index("-qp") + 1] == "0"
        assert args[args.index("-pix_fmt") + 1] == "yuv444p"


# ── The endpoint, against Postgres and ffmpeg ───────────────────────────────

SCHEMA = "artrium_test_forms"


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


@pytest_asyncio.fixture(loop_scope="module")
async def db(sessionmaker_):
    async with sessionmaker_() as session:
        yield session
        await session.rollback()
    async with sessionmaker_() as cleanup:
        await cleanup.execute(text("DELETE FROM control_tracks"))
        await cleanup.commit()


@pytest.fixture
def storage(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "storage_dir", tmp_path)
    return tmp_path


W, H = 64, 36


def _png(color, size=(W, H)) -> bytes:
    buf = io.BytesIO()
    PILImage.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def _uploads(frames: list[bytes], stem: str) -> list[UploadFile]:
    return [UploadFile(file=io.BytesIO(b), filename=f"{stem}_{i:04d}.png")
            for i, b in enumerate(frames)]


def _depth_frames(n: int) -> list[bytes]:
    return [_png((40 + 2 * i, 40 + 2 * i, 40 + 2 * i)) for i in range(n)]


def _mask_frames(n: int) -> list[bytes]:
    out = []
    for i in range(n):
        im = PILImage.new("RGB", (W, H), (0, 0, 0))
        im.paste((255, 0, 0), (0, 0, W // 2, H))
        im.paste((0, 0, 255), (W // 2, 0, W, H // 3))
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        out.append(buf.getvalue())
    return out


def _probe(path) -> tuple[int, str, str]:
    out = subprocess.run(
        [_ffprobe_path(), "-v", "error", "-select_streams", "v:0",
         "-count_frames", "-show_entries", "stream=nb_read_frames,avg_frame_rate,pix_fmt",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    s = json.loads(out.stdout)["streams"][0]
    return int(s["nb_read_frames"]), s["avg_frame_rate"], s["pix_fmt"]


pytestmark_db = pytest.mark.asyncio(loop_scope="module")


@pytestmark_db
async def test_a_loop_is_stored_with_its_mask_and_recipe(db, storage):
    n = loop_frames(1)
    recipe = {"v": 1, "motif": "balls", "seed": 7}
    out = await forms_router.save_loop(
        depth=_uploads(_depth_frames(n), "d"), mask=_uploads(_mask_frames(n), "m"),
        title="Testloop", recipe=json.dumps(recipe),
        labels=json.dumps(["Kugeln A", "Kugeln B", "Boden"]), db=db,
    )
    depth, mask = out["depth"], out["mask"]
    assert depth["kind"] == "depth" and mask["kind"] == "mask"
    assert (depth["width"], depth["height"], depth["frame_count"]) == (W, H, n)
    assert depth["usable_frames"] == n, "the render path must not cut the loop"
    assert depth["fps"] == pytest.approx(LCM_FPS)
    assert mask["source_track_id"] == depth["id"], "lineage is what the pairing check reads"
    assert [r["label"] for r in mask["regions"]] == ["Kugeln A", "Boden"]
    assert out["struktur_url"] == f"/tools/vace/?track={depth['id']}&mask={mask['id']}"

    d_frames, d_rate, d_fmt = _probe(storage / f"control/{depth['id']}.mp4")
    m_frames, m_rate, m_fmt = _probe(storage / f"control/{mask['id']}.mp4")
    assert (d_frames, d_rate, d_fmt) == (n, f"{LCM_FPS}/1", "yuv420p")
    assert (m_frames, m_rate, m_fmt) == (n, f"{LCM_FPS}/1", "yuv444p")
    assert not list((storage / "control").glob(".forms_*")), "the PNG workdir is cleaned up"

    loops = await forms_router.list_loops(db=db)
    assert len(loops) == 1
    assert loops[0]["recipe"] == recipe
    assert loops[0]["mask"]["id"] == mask["id"]

    # Deleting the track through Struktur takes the recipe with it.
    await vace_router.delete_track(uuid.UUID(depth["id"]), db=db)
    assert not forms_router.recipe_path(uuid.UUID(depth["id"])).exists()
    assert await forms_router.list_loops(db=db) == []


@pytestmark_db
async def test_without_a_mask(db, storage):
    n = loop_frames(1)
    out = await forms_router.save_loop(depth=_uploads(_depth_frames(n), "d"), mask=None,
                                       title="", recipe="", labels="[]", db=db)
    assert out["mask"] is None
    assert out["depth"]["title"] == "Formen-Loop"
    assert out["struktur_url"] == f"/tools/vace/?track={out['depth']['id']}"
    # No recipe → not one of the editor's loops.
    assert await forms_router.list_loops(db=db) == []


@pytestmark_db
@pytest.mark.parametrize("case", ["too_few", "mask_count", "mask_size", "not_png", "mixed_size",
                                  "stray", "bad_recipe"])
async def test_refusals(db, storage, case):
    n = loop_frames(1)
    depth, mask, recipe = _depth_frames(n), _mask_frames(n), ""
    if case == "too_few":
        depth, mask = depth[:4], mask[:4]
    elif case == "mask_count":
        mask = mask[:-1]
    elif case == "mask_size":
        mask = [_png((255, 0, 0), size=(W * 2, H * 2))] * n
    elif case == "not_png":
        depth = [b"JFIF not a png"] + depth[1:]
    elif case == "mixed_size":
        depth = depth[:-1] + [_png((0, 0, 0), size=(W, H * 2))]
    elif case == "stray":
        # A ramp edge: a tenth of every frame keys into nothing.
        mask = [_png((128, 0, 0))] * n
    elif case == "bad_recipe":
        recipe = "[1, 2]"
    with pytest.raises(HTTPException) as err:
        await forms_router.save_loop(depth=_uploads(depth, "d"), mask=_uploads(mask, "m"),
                                     title="x", recipe=recipe, labels="[]", db=db)
    assert err.value.status_code == 400
    # Nothing half-written is left behind.
    control = storage / "control"
    assert not control.exists() or not [p for p in control.iterdir()]


@pytestmark_db
async def test_options(storage):
    opts = await forms_router.forms_options()
    assert opts["fps"] == LCM_FPS
    assert [d["frames"] for d in opts["durations"]] == [33, 49, 65, 81]
    assert opts["default_format"] in {f["key"] for f in opts["formats"]}
    assert opts["colors"] == [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
