"""AVIF previews — the cache contract, and the two rules that make it safe.

A preview is only ever an optimisation, so the properties worth pinning are
the ones that decide whether it can ever show the *wrong* picture, or grow
without bound:

- a width ladder, so an open `?w=` cannot mint a cache entry per viewport;
- a fingerprint that moves when the source's contents move, so re-running the
  wand can never leave the old preview in place;
- never upscaling, so "preview" always means "this picture, smaller or equal".
"""
import asyncio

import pytest
from PIL import Image as PILImage

from services.image.preview import (
    DEFAULT_WIDTH,
    PREVIEW_WIDTHS,
    QUALITY,
    QUALITY_DETAIL,
    SUFFIX,
    cache_dir_for,
    cache_path,
    clamp_width,
    fingerprint,
    purge,
    render,
)


def write_png(path, size=(1200, 1600), colour=(40, 60, 90)):
    path.parent.mkdir(parents=True, exist_ok=True)
    PILImage.new("RGB", size, colour).save(path)
    return path


# ── The width ladder ─────────────────────────────────────────────────────────


def test_ladder_is_ascending_and_has_a_sane_default():
    assert list(PREVIEW_WIDTHS) == sorted(PREVIEW_WIDTHS)
    assert DEFAULT_WIDTH in PREVIEW_WIDTHS


@pytest.mark.parametrize("asked,expect", [
    (None, DEFAULT_WIDTH),
    (0, DEFAULT_WIDTH),
    (-5, DEFAULT_WIDTH),
    (1, PREVIEW_WIDTHS[0]),
    (640, 640),
    (641, 1024),
    (1500, 1600),
    (99999, PREVIEW_WIDTHS[-1]),
])
def test_clamp_width_snaps_to_a_rung(asked, expect):
    """An unbounded query parameter must land on a bounded set of files."""
    assert clamp_width(asked) == expect


def test_clamp_width_only_ever_returns_a_rung():
    for asked in range(1, 6000, 37):
        assert clamp_width(asked) in PREVIEW_WIDTHS


def test_the_ladder_reaches_an_upscale_native_size():
    """The reason the top rungs exist: a 2x upscale of a 1080x1920 render is
    3840 on the long edge, and a preview capped below that shows less than the
    GPU minutes bought — exactly the detail you zoom in to check."""
    assert clamp_width(3840) >= 3840
    assert PREVIEW_WIDTHS[-1] >= 4600


@pytest.mark.parametrize("native", [3840, 5760, 7680])
def test_every_upscale_the_gallery_can_make_is_reachable_at_native_size(native):
    """2x, 3x and 4x of a 1920 render. Below the top of the ladder a 3x or 4x
    upscale was shrunk to 4600 on the way to the zoom — 80 % and 60 % of the
    resolution the compare exists to show."""
    assert clamp_width(native) >= native


def test_asking_for_a_native_size_never_interpolates_it_up(tmp_path):
    """Reaching for a big rung must not turn a small picture into a soft one."""
    src = write_png(tmp_path / "a.png", size=(1080, 1920))
    out = asyncio.run(render(tmp_path / "cache", src, clamp_width(1920)))
    with PILImage.open(out) as im:
        assert im.size == (1080, 1920)


# ── Cache identity ───────────────────────────────────────────────────────────


def test_fingerprint_moves_when_the_file_is_rewritten(tmp_path):
    """The case this exists for: re-running the wand overwrites
    `..._enhanced.png` in place. Same path, different picture."""
    src = write_png(tmp_path / "a.png")
    before = fingerprint(src)
    write_png(src, colour=(200, 30, 30))
    assert fingerprint(src) != before


def test_fingerprint_is_stable_for_an_untouched_file(tmp_path):
    src = write_png(tmp_path / "a.png")
    assert fingerprint(src) == fingerprint(src)


def test_two_sources_never_share_a_cache_entry(tmp_path):
    """Same pixels, different names — they must not collide, or deleting one
    picture would take another one's preview with it."""
    a = write_png(tmp_path / "a.png")
    b = write_png(tmp_path / "b.png")
    assert cache_path(tmp_path / "c", a, 640) != cache_path(tmp_path / "c", b, 640)


def test_cache_path_is_sharded_by_stem_and_named_by_width(tmp_path):
    src = write_png(tmp_path / "abcdef.png")
    p = cache_path(tmp_path / "cache", src, 1024)
    assert p.suffix == SUFFIX
    assert p.name.startswith("1024_")
    assert p.parent == cache_dir_for(tmp_path / "cache", "abcdef")
    # One directory level above the per-image one, so a library of thousands
    # does not become one flat directory of thousands.
    assert p.parent.parent.name == "ab"


def test_widths_do_not_overwrite_each_other(tmp_path):
    src = write_png(tmp_path / "a.png")
    paths = {cache_path(tmp_path / "c", src, w) for w in PREVIEW_WIDTHS}
    assert len(paths) == len(PREVIEW_WIDTHS)


# ── Rendering ────────────────────────────────────────────────────────────────


def test_render_writes_an_avif_smaller_than_the_png(tmp_path):
    src = write_png(tmp_path / "a.png", size=(1200, 1600))
    out = asyncio.run(render(tmp_path / "cache", src, 1024))
    assert out.exists()
    with PILImage.open(out) as im:
        assert im.format == "AVIF"
        assert max(im.size) == 1024
    assert out.stat().st_size < src.stat().st_size


def test_render_preserves_aspect_ratio(tmp_path):
    src = write_png(tmp_path / "a.png", size=(1200, 1600))
    out = asyncio.run(render(tmp_path / "cache", src, 640))
    with PILImage.open(out) as im:
        assert im.size == (480, 640)


def test_render_never_upscales(tmp_path):
    """Asking for more pixels than exist returns the picture, not a soft
    interpolation of it."""
    src = write_png(tmp_path / "small.png", size=(300, 400))
    out = asyncio.run(render(tmp_path / "cache", src, PREVIEW_WIDTHS[-1]))
    with PILImage.open(out) as im:
        assert im.size == (300, 400)


def test_render_is_cached_not_repeated(tmp_path):
    src = write_png(tmp_path / "a.png")
    cache = tmp_path / "cache"
    first = asyncio.run(render(cache, src, 640))
    stamp = first.stat().st_mtime_ns
    again = asyncio.run(render(cache, src, 640))
    assert again == first
    assert again.stat().st_mtime_ns == stamp, "a cache hit must not re-encode"


def test_a_rewritten_source_gets_a_new_preview(tmp_path):
    """The whole invalidation story, end to end."""
    src = write_png(tmp_path / "a.png", colour=(10, 10, 10))
    cache = tmp_path / "cache"
    dark = asyncio.run(render(cache, src, 640))
    write_png(src, colour=(240, 240, 240))
    light = asyncio.run(render(cache, src, 640))
    assert light != dark
    assert dark.exists(), "the old entry is simply unreferenced, not deleted mid-serve"


def test_concurrent_renders_produce_one_file(tmp_path):
    """A prefetch racing the view it is prefetching for must not have two
    encoders writing the same path."""
    src = write_png(tmp_path / "a.png")
    cache = tmp_path / "cache"

    async def race():
        return await asyncio.gather(*[render(cache, src, 640) for _ in range(6)])

    outs = asyncio.run(race())
    assert len(set(outs)) == 1
    assert len(list(outs[0].parent.iterdir())) == 1


def test_render_leaves_no_partial_file_behind(tmp_path):
    """It is written to `.part` and moved, so a reader never catches a
    half-encoded file — and nothing is left over afterwards."""
    src = write_png(tmp_path / "a.png")
    out = asyncio.run(render(tmp_path / "cache", src, 640))
    assert not list(out.parent.glob("*.part"))


def test_quality_stays_in_the_measured_band():
    """The encoder settings are chosen against measurements in the module
    docstring; drifting them silently would change every cached file's size."""
    assert 40 <= QUALITY <= 75
    # Measured: below 95 a smooth upscale loses 10-25 % of its fine detail,
    # which is the whole thing the detail tier is asked for.
    assert QUALITY_DETAIL >= 95


# ── The detail tier ──────────────────────────────────────────────────────────


def test_detail_tier_has_its_own_cache_entry(tmp_path):
    """The fitted view and the zoomed compare want different encodes of the
    same picture at the same width; one must never be served as the other."""
    src = write_png(tmp_path / "a.png")
    plain = cache_path(tmp_path / "c", src, 1024)
    detail = cache_path(tmp_path / "c", src, 1024, detail=True)
    assert plain != detail
    assert plain.parent == detail.parent, "purge must still take both"
    assert detail.name.startswith("1024hq_")


def noisy_png(path, size=(512, 512)):
    """Fine per-pixel texture — the thing a low-quality AVIF smooths away."""
    path.parent.mkdir(parents=True, exist_ok=True)
    PILImage.effect_noise(size, 40).convert("RGB").save(path)
    return path


def test_detail_tier_keeps_more_of_the_texture(tmp_path):
    """What it is for, end to end: more of the fine structure survives. File
    size stands in for detail — noise is incompressible, so the encode that
    kept more of it is the bigger one."""
    src = noisy_png(tmp_path / "grain.png")
    cache = tmp_path / "cache"
    plain = asyncio.run(render(cache, src, 640))
    detail = asyncio.run(render(cache, src, 640, detail=True))
    assert plain != detail
    assert detail.stat().st_size > plain.stat().st_size * 1.5
    with PILImage.open(detail) as im:
        assert im.format == "AVIF"
        assert im.size == (512, 512), "the detail tier never upscales either"


def test_purge_takes_the_detail_tier_too(tmp_path):
    src = write_png(tmp_path / "a.png")
    cache = tmp_path / "cache"
    asyncio.run(render(cache, src, 640))
    asyncio.run(render(cache, src, 640, detail=True))
    purge(cache, "a")
    assert not cache_dir_for(cache, "a").exists()


# ── Purging ──────────────────────────────────────────────────────────────────


def test_purge_removes_every_width_for_one_source(tmp_path):
    src = write_png(tmp_path / "a.png")
    cache = tmp_path / "cache"
    for w in (640, 1024):
        asyncio.run(render(cache, src, w))
    assert cache_dir_for(cache, "a").exists()
    purge(cache, "a")
    assert not cache_dir_for(cache, "a").exists()


def test_purge_leaves_other_images_alone(tmp_path):
    cache = tmp_path / "cache"
    a = write_png(tmp_path / "aa.png")
    b = write_png(tmp_path / "ab.png")     # same shard, different image
    asyncio.run(render(cache, a, 640))
    asyncio.run(render(cache, b, 640))
    purge(cache, "aa")
    assert not cache_dir_for(cache, "aa").exists()
    assert cache_dir_for(cache, "ab").exists()


def test_purge_of_something_never_cached_is_not_an_error(tmp_path):
    purge(tmp_path / "cache", "nothing-here")
