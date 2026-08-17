"""
The rendition stack for a gallery image (services/image/rendition.py).

Four files at most, bottom to top:

    original  →  _upscaled  →  _enhanced  →  _grain

Which one a pass reads is the entire feature: the upscale costs GPU minutes,
so it sits at the bottom where nothing cheap above it can invalidate it, and
the two Pillow passes are re-rendered on top of whatever is underneath them.
Getting the order wrong is not a subtle bug — it is an image that visibly
snaps back to its generated size the moment the wand is touched, which is
exactly what these tests pin down.
"""
import pytest

from core.config import settings
from core.models import Image
from services.image.rendition import (
    enhance_source_path,
    enhanced_rel_path,
    grain_source_path,
    grained_rel_path,
    primary_filename,
    upscale_source_path,
)

ORIGINAL = "images/2026/08/frame.png"


@pytest.fixture
def storage(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "storage_dir", tmp_path)
    return tmp_path


def make_image(**kw) -> Image:
    """An Image row with only the columns the rendition rules look at."""
    return Image(filename="frame.png", filepath=ORIGINAL, **kw)


ENHANCED = dict(
    enhance_strength=100,
    enhanced_filename="frame_enhanced.png",
    enhanced_filepath="images/2026/08/frame_enhanced.png",
)
UPSCALED = dict(
    upscale_scale=2.0,
    upscaled_filename="frame_upscaled.png",
    upscaled_filepath="images/2026/08/frame_upscaled.png",
)
GRAINED = dict(
    grain_strength=30,
    grained_filename="frame_grain.png",
    grained_filepath="images/2026/08/frame_grain.png",
)


class TestWhatViewersGet:
    """`primary_*` is the top file that exists — publishing and the gallery
    both resolve through it."""

    def test_nothing_applied_is_the_original(self):
        assert primary_filename(make_image()) == "frame.png"

    def test_the_upscale_replaces_the_original(self):
        assert primary_filename(make_image(**UPSCALED)) == "frame_upscaled.png"

    def test_the_enhancement_sits_above_the_upscale(self):
        # It is *rendered from* the upscale, so it is the larger picture too —
        # serving the upscale here would throw the correction away.
        img = make_image(**UPSCALED, **ENHANCED)
        assert primary_filename(img) == "frame_enhanced.png"

    def test_grain_is_always_last(self):
        img = make_image(**UPSCALED, **ENHANCED, **GRAINED)
        assert primary_filename(img) == "frame_grain.png"


class TestUpscaleReadsTheOriginal:
    """The bottom of the stack, unconditionally: a diffusion model handed a
    grained frame repaints the noise as texture, and one handed a
    contrast-stretched frame repaints the correction."""

    def test_a_plain_image(self, storage):
        assert upscale_source_path(make_image()).name == "frame.png"

    def test_an_enhanced_image_is_still_upscaled_from_the_original(self, storage):
        assert upscale_source_path(make_image(**ENHANCED)).name == "frame.png"

    def test_a_grained_image_is_still_upscaled_from_the_original(self, storage):
        img = make_image(**ENHANCED, **GRAINED)
        assert upscale_source_path(img).name == "frame.png"

    def test_rerunning_reads_the_original_not_the_previous_upscale(self, storage):
        # Otherwise a second 2x pass would silently be a 4x one.
        assert upscale_source_path(make_image(**UPSCALED)).name == "frame.png"


class TestEnhanceReadsTheLayerBelow:
    """This is the regression the whole reorder exists for: the wand must
    correct the upscaled picture, not quietly serve the small one again."""

    def test_without_an_upscale_it_reads_the_original(self, storage):
        assert enhance_source_path(make_image()).name == "frame.png"

    def test_with_an_upscale_it_reads_the_upscale(self, storage):
        assert enhance_source_path(make_image(**UPSCALED)).name == "frame_upscaled.png"

    def test_it_never_reads_its_own_output(self, storage):
        # Re-running at a new strength has to replace the correction, not
        # stack a second one onto it.
        img = make_image(**UPSCALED, **ENHANCED)
        assert enhance_source_path(img).name == "frame_upscaled.png"

    def test_it_never_reads_the_grain_above_it(self, storage):
        img = make_image(**UPSCALED, **ENHANCED, **GRAINED)
        assert enhance_source_path(img).name == "frame_upscaled.png"


class TestGrainReadsTheLayerBelow:
    def test_bare_image(self, storage):
        assert grain_source_path(make_image()).name == "frame.png"

    def test_prefers_the_enhancement(self, storage):
        assert grain_source_path(make_image(**ENHANCED)).name == "frame_enhanced.png"

    def test_falls_through_to_the_upscale_when_the_wand_is_off(self, storage):
        # Delivery resolution: graining the small original and serving it over
        # a 4K upscale is the shrink bug in its other form.
        assert grain_source_path(make_image(**UPSCALED)).name == "frame_upscaled.png"

    def test_never_reads_its_own_output(self, storage):
        img = make_image(**UPSCALED, **ENHANCED, **GRAINED)
        assert grain_source_path(img).name == "frame_enhanced.png"


class TestRenditionNames:
    """Every derived file is named off the *original*, whatever it was
    rendered from — so an image has one enhanced file and one grain file
    whether or not it is upscaled, and toggling the upscale cannot strand a
    second copy."""

    def test_enhanced_name_is_a_sibling_of_the_original(self):
        rel, name = enhanced_rel_path(make_image(**UPSCALED))
        assert name == "frame_enhanced.png"
        assert rel == "images/2026/08/frame_enhanced.png"

    def test_grained_name_is_a_sibling_of_the_original(self):
        rel, name = grained_rel_path(make_image(**UPSCALED, **ENHANCED))
        assert name == "frame_grain.png"
        assert rel == "images/2026/08/frame_grain.png"
