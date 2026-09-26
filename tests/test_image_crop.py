"""
The gallery crop ("Zuschnitt", services/image/crop.py).

A crop is stored as fractions of the rendition it is cut from, so the same
framing lands on the original and on a 2x upscale. What these tests pin down is
that this never costs the shape: a box drawn at 4:5 has to come out exactly
4:5 in pixels at any source size, a box dragged onto the edges has to read as
"no crop" rather than as a one-pixel trim, and a stored box has to describe
the file it produced to the pixel.
"""
import pytest
from PIL import Image as PILImage

from services.image.crop import (
    ASPECT_FREE,
    ASPECT_ORIGINAL,
    CropBox,
    aspect_ratio,
    clean_aspect,
    crop_file,
    exact_box,
    is_full_frame,
    is_too_small,
    normalize,
    pixel_box,
    token,
)


def _size(box: tuple[int, int, int, int]) -> tuple[int, int]:
    left, top, right, bottom = box
    return right - left, bottom - top


class TestAspect:
    @pytest.mark.parametrize("value", ["free", "original", "4:5", "16:9", "1:1"])
    def test_known_values_pass(self, value):
        assert clean_aspect(value) == value

    @pytest.mark.parametrize("value", [None, "", "banana", "4/5", "0:5", "4:0", "-4:5", "4:5:6"])
    def test_anything_else_is_free(self, value):
        assert clean_aspect(value) == ASPECT_FREE

    def test_leading_zeros_are_dropped(self):
        assert clean_aspect("04:05") == "4:5"

    def test_original_is_the_sources_own_shape(self):
        assert aspect_ratio(ASPECT_ORIGINAL, 1080, 1920) == pytest.approx(0.5625)

    def test_a_fixed_ratio_ignores_the_source(self):
        assert aspect_ratio("4:5", 1080, 1920) == pytest.approx(0.8)

    def test_free_has_none(self):
        assert aspect_ratio(ASPECT_FREE, 1080, 1920) is None


class TestNormalize:
    def test_a_box_inside_the_frame_is_kept(self):
        b = normalize(0.1, 0.2, 0.5, 0.6, "4:5")
        assert (b.x, b.y, b.w, b.h, b.aspect) == (0.1, 0.2, 0.5, 0.6, "4:5")

    def test_a_box_hanging_over_an_edge_is_trimmed_not_slid(self):
        b = normalize(0.7, 0.9, 0.5, 0.5)
        assert (b.x, b.y) == (0.7, 0.9)
        assert b.w == pytest.approx(0.3)
        assert b.h == pytest.approx(0.1)

    def test_garbage_falls_back_to_the_whole_frame(self):
        b = normalize("x", float("nan"), None, "?")
        assert (b.x, b.y, b.w, b.h) == (0.0, 0.0, 1.0, 1.0)


class TestPixelBox:
    def test_a_free_box_is_rounded_into_pixels(self):
        assert pixel_box(1000, 500, CropBox(0.1, 0.1, 0.5, 0.5)) == (100, 50, 600, 300)

    def test_a_fixed_ratio_comes_out_exact(self):
        # 0.9 x 1080 = 972 wide; the drawn height is ignored in favour of the
        # shape: 972 / 0.8 = 1215.
        w, h = _size(pixel_box(1080, 1920, CropBox(0.05, 0.1, 0.9, 0.6331, "4:5")))
        assert (w, h) == (972, 1215)

    def test_the_same_box_on_a_2x_upscale_is_twice_the_size_and_still_exact(self):
        box = CropBox(0.05, 0.1, 0.9, 0.6331, "4:5")
        w, h = _size(pixel_box(2160, 3840, box))
        assert (w, h) == (1944, 2430)
        assert w / h == pytest.approx(0.8)

    def test_the_competition_case_keeps_9_16(self):
        # The crop this feature was asked for: a filmstrip border cut off the
        # left of a 2160x3840 frame while keeping the series' shape.
        box = CropBox(180 / 2160, 120 / 3840, 1980 / 2160, 3520 / 3840, ASPECT_ORIGINAL)
        assert pixel_box(2160, 3840, box) == (180, 120, 2160, 3640)

    def test_a_ratio_that_overflows_the_height_is_bounded_by_it(self):
        # Full width at 1:1 on a landscape frame cannot be — the height wins.
        w, h = _size(pixel_box(1920, 1080, CropBox(0.0, 0.0, 1.0, 1.0, "1:1")))
        assert (w, h) == (1080, 1080)

    def test_the_box_never_leaves_the_source(self):
        left, top, right, bottom = pixel_box(1080, 1920, CropBox(0.95, 0.95, 0.05, 0.05, "16:9"))
        assert 0 <= left < right <= 1080
        assert 0 <= top < bottom <= 1920


class TestFullFrame:
    def test_the_whole_picture_is_no_crop(self):
        assert is_full_frame(1080, 1920, CropBox(0, 0, 1, 1))

    def test_float_noise_on_the_edges_is_still_no_crop(self):
        assert is_full_frame(1080, 1920, CropBox(0.0004, 0, 0.9995, 1))

    def test_original_aspect_at_full_size_is_no_crop(self):
        assert is_full_frame(1080, 1920, CropBox(0, 0, 1, 1, ASPECT_ORIGINAL))

    def test_a_real_trim_is_a_crop(self):
        assert not is_full_frame(1080, 1920, CropBox(10 / 1080, 0, 1070 / 1080, 1))

    def test_a_sliver_is_too_small(self):
        assert is_too_small(1080, 1920, CropBox(0.5, 0.5, 0.002, 0.3))
        assert not is_too_small(1080, 1920, CropBox(0.5, 0.5, 0.2, 0.3))


class TestExactBox:
    def test_it_describes_what_was_cut(self):
        box = CropBox(0.05, 0.1, 0.9, 0.6331, "4:5")
        exact = exact_box(1080, 1920, box)
        assert pixel_box(1080, 1920, exact) == pixel_box(1080, 1920, box)
        assert exact.h == pytest.approx(1215 / 1920)

    def test_it_is_a_fixed_point(self):
        # Re-cutting a stored box from the same source must not drift.
        once = exact_box(2160, 3840, CropBox(0.123, 0.456, 0.333, 0.3, "2:3"))
        twice = exact_box(2160, 3840, once)
        assert pixel_box(2160, 3840, once) == pixel_box(2160, 3840, twice)

    def test_round_trips_through_the_stored_dict(self):
        box = CropBox(0.1, 0.2, 0.5, 0.4, "4:5")
        again = CropBox.from_dict(exact_box(1080, 1920, box).as_dict())
        assert pixel_box(1080, 1920, again) == pixel_box(1080, 1920, box)
        assert again.aspect == "4:5"


class TestToken:
    def test_stable(self):
        box = {"x": 0.1, "y": 0.2, "w": 0.5, "h": 0.4, "aspect": "free"}
        assert token(box) == token(dict(reversed(list(box.items()))))

    def test_a_different_box_is_a_different_token(self):
        a = {"x": 0.1, "y": 0.2, "w": 0.5, "h": 0.4, "aspect": "free"}
        assert token(a) != token({**a, "x": 0.11})

    def test_no_box(self):
        assert token(None) == "0"


class TestCropFile:
    async def test_it_writes_the_cut_and_reports_it(self, tmp_path):
        src = tmp_path / "frame.png"
        PILImage.new("RGB", (1080, 1920), (40, 60, 80)).save(src)
        dest = tmp_path / "out" / "frame_crop.png"

        exact, w, h = await crop_file(src, dest, CropBox(0.05, 0.1, 0.9, 0.6331, "4:5"))

        assert (w, h) == (972, 1215)
        with PILImage.open(dest) as out:
            assert out.size == (972, 1215)
        assert exact.aspect == "4:5"
        assert not list(tmp_path.rglob("*.part")), "the temp file must be moved into place"

    async def test_it_keeps_the_pixels_it_was_asked_for(self, tmp_path):
        src = tmp_path / "frame.png"
        im = PILImage.new("RGB", (100, 100), (0, 0, 0))
        im.putpixel((60, 70), (255, 0, 0))
        im.save(src)
        dest = tmp_path / "frame_crop.png"

        await crop_file(src, dest, CropBox(0.5, 0.5, 0.5, 0.5))

        with PILImage.open(dest) as out:
            assert out.getpixel((10, 20)) == (255, 0, 0)
