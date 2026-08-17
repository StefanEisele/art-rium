"""
Unit tests for the Instagram frame/crop math (services/instagram/framing.py).

The numbers come from Meta's media reference: a feed image must sit within
4:5 … 1.91:1, and a carousel renders every child in the first child's frame.
The crop exists so a 9:16 render fills the tallest frame Instagram has instead
of being padded into it.
"""
import pytest

from services.instagram.framing import (
    FEED_MAX_RATIO,
    FEED_MIN_RATIO,
    crop_box,
    crop_rendition_name,
    frame_ratio,
    needs_crop,
)

NINE_SIXTEEN = 1080 / 1920   # 0.5625 — what the z-Image portrait renders are
FOUR_FIVE = 0.8


class TestFrameRatio:
    def test_explicit_choice_wins_over_the_children(self):
        assert frame_ratio("4x5", 1.0) == FOUR_FIVE
        assert frame_ratio("1x1", NINE_SIXTEEN) == 1.0

    def test_auto_follows_the_first_child_when_it_is_in_range(self):
        assert frame_ratio("auto", 1.0) == 1.0

    def test_auto_clamps_a_9_16_first_child_to_4_5(self):
        # This is the whole bug the crop feature answers: Instagram cannot
        # render a 0.5625 frame, so it uses 4:5 and pads the sides.
        assert frame_ratio("auto", NINE_SIXTEEN) == FEED_MIN_RATIO

    def test_auto_clamps_an_ultrawide_first_child(self):
        assert frame_ratio("auto", 3.0) == FEED_MAX_RATIO

    @pytest.mark.parametrize("unknown", [None, 0, 0.0])
    def test_auto_falls_back_to_portrait_without_dimensions(self, unknown):
        assert frame_ratio("auto", unknown) == FOUR_FIVE


class TestCropBox:
    def test_tall_source_is_cut_top_and_bottom(self):
        # 1080x1920 into 4:5 → 1080x1350, centred: 285 off each end.
        assert crop_box(1080, 1920, FOUR_FIVE) == (0, 285, 1080, 1635)

    def test_offset_zero_keeps_the_top(self):
        assert crop_box(1080, 1920, FOUR_FIVE, 0.0) == (0, 0, 1080, 1350)

    def test_offset_one_keeps_the_bottom(self):
        assert crop_box(1080, 1920, FOUR_FIVE, 1.0) == (0, 570, 1080, 1920)

    def test_wide_source_is_cut_left_and_right(self):
        # 1920x1080 into 1:1 → a full-height 1080 square, centred.
        assert crop_box(1920, 1080, 1.0) == (420, 0, 1500, 1080)

    def test_wide_source_offset_keeps_the_left_edge(self):
        assert crop_box(1920, 1080, 1.0, 0.0) == (0, 0, 1080, 1080)

    def test_square_into_4_5_loses_width_not_height(self):
        # A square is *wider* than 4:5, so filling a portrait frame with it
        # narrows it — 1088 x 0.8 = 870 — rather than cutting its height.
        left, upper, right, lower = crop_box(1088, 1088, FOUR_FIVE)
        assert (lower - upper) == 1088         # full height kept
        assert (right - left) == 870           # 1088 * 0.8

    def test_matching_source_is_returned_whole(self):
        assert crop_box(1080, 1350, FOUR_FIVE) == (0, 0, 1080, 1350)

    def test_box_never_leaves_the_source(self):
        for off in (0.0, 0.25, 0.5, 0.75, 1.0):
            left, upper, right, lower = crop_box(1080, 1920, FOUR_FIVE, off)
            assert 0 <= left < right <= 1080
            assert 0 <= upper < lower <= 1920

    def test_offset_outside_the_range_is_clamped(self):
        assert crop_box(1080, 1920, FOUR_FIVE, 5.0) == crop_box(1080, 1920, FOUR_FIVE, 1.0)
        assert crop_box(1080, 1920, FOUR_FIVE, -2.0) == crop_box(1080, 1920, FOUR_FIVE, 0.0)

    def test_zero_dimensions_raise(self):
        with pytest.raises(ValueError):
            crop_box(0, 1920, FOUR_FIVE)


class TestNeedsCrop:
    def test_a_9_16_render_needs_one(self):
        assert needs_crop(1080, 1920, FOUR_FIVE)

    def test_an_exact_4_5_render_does_not(self):
        assert not needs_crop(1080, 1350, FOUR_FIVE)

    def test_a_hair_off_is_left_alone(self):
        # 1080x1351 is 0.7994 — re-encoding it would cost a file for nothing.
        assert not needs_crop(1080, 1351, FOUR_FIVE)


class TestRenderCrop:
    """The bake itself — the file Instagram is actually handed."""

    def _src(self, tmp_path, w, h):
        from PIL import Image as PILImage
        p = tmp_path / "src.png"
        PILImage.new("RGB", (w, h), (30, 60, 90)).save(p)
        return p

    def test_a_9_16_render_comes_out_at_4_5(self, tmp_path):
        from PIL import Image as PILImage
        from services.instagram.framing import render_crop_sync
        dest = tmp_path / "out" / "cropped.png"
        assert render_crop_sync(self._src(tmp_path, 1080, 1920), dest, FOUR_FIVE, 0.5) == (1080, 1350)
        with PILImage.open(dest) as out:
            assert out.size == (1080, 1350)
            assert abs(out.width / out.height - FOUR_FIVE) < 0.001

    def test_it_creates_the_destination_directory(self, tmp_path):
        from services.instagram.framing import render_crop_sync
        dest = tmp_path / "nested" / "deeper" / "c.png"
        render_crop_sync(self._src(tmp_path, 1080, 1920), dest, FOUR_FIVE, 0.5)
        assert dest.exists()


class TestVideoChildrenAreNeverCropped:
    """Cropping a video would mean re-encoding it, so the API normalises the
    request away rather than accepting a setting it will not honour."""

    def test_fill_on_a_video_child_is_stored_as_fit(self):
        from routers.instagram import MediaItem, _crop_specs_from_items
        import uuid as _uuid
        items = [
            MediaItem(kind="image", id=_uuid.uuid4(), crop_mode="fill", crop_offset=0.25),
            MediaItem(kind="video", id=_uuid.uuid4(), crop_mode="fill", crop_offset=0.25),
        ]
        assert _crop_specs_from_items(items) == [("fill", 0.25), ("fit", 0.25)]

    def test_default_is_todays_behaviour(self):
        from routers.instagram import MediaItem, _crop_specs_from_items
        import uuid as _uuid
        assert _crop_specs_from_items([MediaItem(kind="image", id=_uuid.uuid4())]) == [("fit", 0.5)]


class TestCropRenditionName:
    def test_name_encodes_the_crop_it_represents(self):
        assert crop_rendition_name("art_0042.png", 0.8, 0.5) == "art_0042_igr800o50.png"

    def test_different_offsets_are_different_files(self):
        a = crop_rendition_name("art_0042.png", 0.8, 0.0)
        b = crop_rendition_name("art_0042.png", 0.8, 1.0)
        assert a != b

    def test_the_same_crop_is_the_same_file(self):
        # Two posts cropping one picture identically must not each bake a copy.
        assert (crop_rendition_name("a.png", 0.8, 0.5)
                == crop_rendition_name("a.png", 0.8, 0.5))

    def test_name_stays_in_the_share_endpoint_charset(self):
        import re
        name = crop_rendition_name("art_0042.png", 1.91, 0.25)
        assert re.match(r"^[A-Za-z0-9._-]+$", name)
