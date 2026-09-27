"""
The hub icon painter (scripts/ui_icons.py), minus the GPU.

What can break without a render to show it: the style file silently losing a
section (or leaking its documentation into a prompt), BRIA cutting out a
different image than the one that is saved, and the trim — every icon has to
come out as the same square with its object centred, or the tiles stop
looking alike, and an empty mask has to be refused rather than installed as
a blank icon.
"""
import io

import pytest
from PIL import Image as PILImage

from scripts import ui_icons


def _png(im: PILImage.Image) -> bytes:
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


class TestStyles:
    def test_every_section_is_a_style(self):
        assert set(ui_icons.load_styles()) == {"objekt", "guss", "keramik"}

    def test_the_documentation_never_reaches_a_prompt(self):
        for text in ui_icons.load_styles().values():
            assert "##" not in text
            assert "kids app" not in text


class TestWorkflow:
    def test_bria_cuts_out_the_saved_picture(self):
        wf = ui_icons.build_workflow("A vinyl record, x", 7, "art-rium/ui_icons/t")
        saved = wf[ui_icons.zimage.ZIMAGE_SAVE_NODE]["inputs"]
        assert wf["rmbg"]["inputs"]["image"] == saved["images"]
        assert saved["filename_prefix"] == "art-rium/ui_icons/t"
        assert wf[ui_icons._MASK_NODE]["inputs"]["filename_prefix"] == "art-rium/ui_icons/t_mask"


class TestCutout:
    def _object_off_centre(self) -> tuple[bytes, bytes]:
        """A 200x100 object in the top-left corner of a 768 render."""
        picture = PILImage.new("RGB", (768, 768), (128, 128, 128))
        mask = PILImage.new("L", (768, 768), 0)
        picture.paste((200, 90, 40), (40, 60, 240, 160))
        mask.paste(255, (40, 60, 240, 160))
        return _png(picture), _png(mask)

    def test_comes_out_as_the_candidate_square(self):
        icon = ui_icons.cutout(*self._object_off_centre())
        assert icon.size == (ui_icons.CANDIDATE_SIZE, ui_icons.CANDIDATE_SIZE)
        assert icon.mode == "RGBA"

    def test_the_object_is_centred_and_fills_its_width(self):
        icon = ui_icons.cutout(*self._object_off_centre())
        left, top, right, bottom = icon.getchannel("A").point(lambda v: 255 if v > 128 else 0).getbbox()
        size = ui_icons.CANDIDATE_SIZE
        assert (left + right) / 2 == pytest.approx(size / 2, abs=2)
        assert (top + bottom) / 2 == pytest.approx(size / 2, abs=2)
        assert (right - left) / size == pytest.approx(1 / (1 + 2 * ui_icons.TRIM_PAD), abs=0.01)

    def test_the_backdrop_is_transparent(self):
        icon = ui_icons.cutout(*self._object_off_centre())
        assert icon.getpixel((2, 2))[3] == 0

    def test_an_empty_mask_is_refused(self):
        picture = _png(PILImage.new("RGB", (768, 768), (128, 128, 128)))
        mask = _png(PILImage.new("L", (768, 768), 8))     # BRIA's faint haze, nothing more
        with pytest.raises(ui_icons.EmptyCutout):
            ui_icons.cutout(picture, mask)


class TestNumbering:
    def test_a_rerun_continues_after_the_last_candidate(self):
        index = {"video__keramik-1": {}, "video__keramik-3": {}, "video__objekt-5": {},
                 "video-api__keramik-9": {}}
        assert ui_icons.next_number(index, "video", "keramik") == 4
        assert ui_icons.next_number(index, "music", "keramik") == 1
