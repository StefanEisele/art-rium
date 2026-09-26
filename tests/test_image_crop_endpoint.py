"""The crop endpoints (routers/images.py) against Postgres and real files.

The service tests prove the geometry; these prove the *stack*: that setting a
crop re-renders the wand and the grain from the cut rather than leaving them on
the full frame, that clearing it puts the full frame back under them, and that
an upscale landing re-cuts the same framing at the new size. Each of those, got
wrong, is a gallery that shows one framing and publishes another.

Throwaway schema, tmp storage; skips without a reachable database.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
import pytest_asyncio
from PIL import Image as PILImage
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from core.config import settings
from core.db import Base
from core.models import Image
from core.thumbnail import thumb_rel_path
from routers import images as images_router
from routers.images import CropRequest, crop_image_endpoint, remove_crop
from services.image.crop import token as crop_token

SCHEMA = "artrium_test_image_crop"

pytestmark = pytest.mark.asyncio(loop_scope="module")

W, H = 216, 384          # a 9:16 frame, small enough for the passes to be instant


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
def storage(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "storage_dir", tmp_path)
    return tmp_path


@pytest_asyncio.fixture(loop_scope="module")
async def db(sessionmaker_):
    async with sessionmaker_() as session:
        yield session
        await session.rollback()
    async with sessionmaker_() as cleanup:
        await cleanup.execute(text("DELETE FROM images"))
        await cleanup.commit()


def _frame(path, size=(W, H)):
    """A frame with a black border down the left — the case the crop exists for."""
    path.parent.mkdir(parents=True, exist_ok=True)
    im = PILImage.new("RGB", size, (120, 90, 70))
    for y in range(size[1]):
        for x in range(max(1, size[0] // 20)):
            im.putpixel((x, y), (0, 0, 0))
    im.save(path)


async def _image(db, storage, **passes) -> Image:
    ident = uuid.uuid4()
    rel = f"images/2026/09/{ident.hex}.png"
    _frame(storage / rel)
    img = Image(
        id=ident, filename=f"{ident.hex}.png", filepath=rel, width=W, height=H,
        created_at=datetime.now(timezone.utc), series_items=[], **passes,
    )
    db.add(img)
    await db.commit()
    return img


def _size(storage, rel) -> tuple[int, int]:
    with PILImage.open(storage / rel) as im:
        return im.size


# The left border cut off, the 9:16 shape kept.
TRIM = CropRequest(x=18 / W, y=16 / H, w=198 / W, h=352 / H, aspect="original")


async def test_a_crop_is_cut_and_served(db, storage):
    img = await _image(db, storage)
    out = await crop_image_endpoint(img.id, TRIM, db=db)

    assert out["cropped"] is True
    assert (out["crop_width"], out["crop_height"]) == (198, 352)
    assert (out["delivered_width"], out["delivered_height"]) == (198, 352)
    assert (out["width"], out["height"]) == (W, H), "the generated size is the recipe's, untouched"
    assert out["primary_url"].split("?")[0] == f"/api/image/{img.id.hex}_crop.png"
    assert out["primary_url"].endswith("c" + crop_token(img.crop_box)), \
        "the cache marker must name the box"
    assert _size(storage, img.cropped_filepath) == (198, 352)
    # The thumbnail is re-cut from the crop, so every picker shows the framing.
    tw, th = _size(storage, thumb_rel_path(img.filename))
    assert tw / th == pytest.approx(198 / 352, abs=0.01)


async def test_the_wand_and_the_grain_are_rerendered_on_the_cut(db, storage):
    img = await _image(db, storage)
    await images_router.enhance_image_endpoint(img.id, images_router.EnhanceRequest(strength=100), db=db)
    await images_router.grain_image_endpoint(img.id, images_router.GrainRequest(strength=30), db=db)
    assert _size(storage, img.enhanced_filepath) == (W, H)

    out = await crop_image_endpoint(img.id, TRIM, db=db)

    assert _size(storage, img.enhanced_filepath) == (198, 352)
    assert _size(storage, img.grained_filepath) == (198, 352)
    assert out["primary_url"].split("?")[0].endswith("_grain.png")
    assert (out["delivered_width"], out["delivered_height"]) == (198, 352)


async def test_the_wand_no_longer_sees_the_border(db, storage):
    # The black strip is 5 % of the frame; cut away, the analysis measures a
    # picture with no pure black in it at all.
    img = await _image(db, storage)
    full = await images_router.enhance_image_endpoint(
        img.id, images_router.EnhanceRequest(strength=100), db=db)
    cut = await crop_image_endpoint(img.id, TRIM, db=db)
    assert full["enhance_params"] != cut["enhance_params"]


async def test_a_full_frame_box_takes_the_crop_off(db, storage):
    img = await _image(db, storage)
    await images_router.enhance_image_endpoint(img.id, images_router.EnhanceRequest(strength=100), db=db)
    await crop_image_endpoint(img.id, TRIM, db=db)
    crop_file = storage / img.cropped_filepath

    out = await crop_image_endpoint(
        img.id, CropRequest(x=0, y=0, w=1, h=1, aspect="original"), db=db)

    assert out["cropped"] is False and out["crop"] is None
    assert not crop_file.exists()
    assert _size(storage, img.enhanced_filepath) == (W, H), "the wand is back on the full frame"
    assert (out["delivered_width"], out["delivered_height"]) == (W, H)


async def test_delete_restores_and_is_a_noop_when_there_is_nothing(db, storage):
    img = await _image(db, storage)
    untouched = await remove_crop(img.id, db=db)
    assert untouched["cropped"] is False

    await crop_image_endpoint(img.id, TRIM, db=db)
    out = await remove_crop(img.id, db=db)
    assert out["cropped"] is False
    assert out["primary_url"].split("?")[0] == f"/api/image/{img.filename}"


async def test_an_upscale_landing_recuts_the_same_framing(db, storage):
    img = await _image(db, storage)
    await crop_image_endpoint(img.id, TRIM, db=db)

    # What _run_image_upscale does once ComfyUI has delivered: a 2x file, the
    # columns, then the passes on top in stack order.
    up_rel = img.filepath.replace(".png", "_upscaled.png")
    _frame(storage / up_rel, (2 * W, 2 * H))
    img.upscaled_filename = up_rel.split("/")[-1]
    img.upscaled_filepath = up_rel
    img.upscale_scale = 2.0
    img.upscale_width, img.upscale_height = 2 * W, 2 * H
    await images_router._rerender_crop(img)
    await db.commit()

    assert (img.crop_width, img.crop_height) == (396, 704)
    assert _size(storage, img.cropped_filepath) == (396, 704)
    assert img.crop_width / img.crop_height == pytest.approx(W / H)


async def test_a_sliver_is_refused(db, storage):
    img = await _image(db, storage)
    with pytest.raises(Exception) as exc:
        await crop_image_endpoint(img.id, CropRequest(x=0.5, y=0.5, w=0.01, h=0.5), db=db)
    assert getattr(exc.value, "status_code", None) == 400


async def test_deleting_the_picture_takes_the_crop_file_with_it(db, storage):
    img = await _image(db, storage)
    await crop_image_endpoint(img.id, TRIM, db=db)
    crop_file = storage / img.cropped_filepath
    assert crop_file.exists()
    images_router._delete_files(img)
    assert not crop_file.exists()
