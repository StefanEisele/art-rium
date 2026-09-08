"""
The two failures that made region renders come out garish, and their guards.

Both were silent — the render succeeded, looked wrong, and nothing in the
system said why. Measured 2026-09-04 on real renders: three SAM 3 masks covered
17.6-18.5% of the frame, the finished clips were 1.6-2.8x more saturated than
the pictures they were made of, and the mask being used had been keyed out of a
30-frame 5 fps decimation of the 80-frame 10 fps control track it was paired
with. So the tests below are mostly about things NOT being quietly accepted.
"""
import uuid

import pytest
from fastapi import HTTPException

from core.video_thumb import _rate
from routers.vace import _validate_mask_pairing
from services.comfy.animatelcm import (
    BASE_IP_DEFAULT,
    AnimateLcmRequest,
    Region,
    build_animatelcm_workflow,
)


def _req(**kw):
    kw.setdefault("control_video", "c.mp4")
    kw.setdefault("prompt", "p")
    kw.setdefault("length", 32)
    return AnimateLcmRequest(**kw)


def _regions(n=3, weight=None):
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    return [
        Region(color=colors[i], reference=f"r{i}.png",
               **({"weight": weight} if weight is not None else {}))
        for i in range(n)
    ]


def _adapters(wf):
    """The IP-Adapter chain, in the order the model flows through it."""
    nodes = {k: v for k, v in wf.items() if v["class_type"] == "IPAdapterAdvanced"}
    by_src = {v["inputs"]["model"][0]: k for k, v in nodes.items()}
    chain, cur = [], next(k for k, v in wf.items()
                          if v["class_type"] == "IPAdapterUnifiedLoader")
    while cur in by_src:
        cur = by_src[cur]
        chain.append((cur, nodes[cur]["inputs"]))
    return chain


class TestBaseLayer:
    """Masked adapters only condition what their mask covers. Everything else
    used to get nothing at all — on a real render, 82% of the frame."""

    def test_the_base_adapter_covers_the_whole_frame(self):
        wf, _ = build_animatelcm_workflow(_req(
            mask_video="m.mp4", regions=_regions(), base_reference="base.png"))
        chain = _adapters(wf)
        assert "attn_mask" not in chain[0][1], "the base must not be masked"
        assert all("attn_mask" in i for _, i in chain[1:])

    def test_the_base_comes_first_so_the_regions_sit_on_top_of_it(self):
        wf, _ = build_animatelcm_workflow(_req(
            mask_video="m.mp4", regions=_regions(), base_reference="base.png"))
        first, _ = _adapters(wf)[0]
        assert wf[wf[first]["inputs"]["image"][0]]["inputs"]["image"] == "base.png"

    def test_it_uses_its_own_picture_not_a_region_s(self):
        wf, _ = build_animatelcm_workflow(_req(
            mask_video="m.mp4", regions=_regions(), base_reference="base.png"))
        images = [wf[i["image"][0]]["inputs"]["image"] for _, i in _adapters(wf)]
        assert images == ["base.png", "r0.png", "r1.png", "r2.png"]

    def test_it_sits_below_the_regions_in_weight_by_default(self):
        # It sets the palette; it does not compete with the material on top.
        assert BASE_IP_DEFAULT < 1.0

    def test_without_one_the_old_behaviour_is_unchanged(self):
        # Still reachable on purpose — someone may genuinely want only the
        # masked areas conditioned — but it is now a choice, not the only way.
        wf, _ = build_animatelcm_workflow(_req(mask_video="m.mp4", regions=_regions()))
        chain = _adapters(wf)
        assert len(chain) == 3
        assert all("attn_mask" in i for _, i in chain)

    def test_a_render_without_regions_ignores_it(self):
        # One unmasked reference already covers the frame; a second would just
        # be the same conditioning twice.
        wf, _ = build_animatelcm_workflow(_req(
            reference_image="ref.png", base_reference="base.png"))
        chain = _adapters(wf)
        assert len(chain) == 1
        assert wf[chain[0][1]["image"][0]]["inputs"]["image"] == "ref.png"

    def test_the_base_weight_is_clamped_like_any_other(self):
        wf, _ = build_animatelcm_workflow(_req(
            mask_video="m.mp4", regions=_regions(), base_reference="base.png",
            base_weight=99.0))
        assert _adapters(wf)[0][1]["weight"] <= 1.5


class TestPerRegionWeight:
    def test_each_region_carries_its_own(self):
        wf, _ = build_animatelcm_workflow(_req(
            mask_video="m.mp4",
            regions=[Region(color=(255, 0, 0), reference="a.png", weight=0.4),
                     Region(color=(0, 255, 0), reference="b.png", weight=1.2)]))
        assert [i["weight"] for _, i in _adapters(wf)] == [0.4, 1.2]


class _Track:
    def __init__(self, **kw):
        self.id = kw.get("id") or uuid.uuid4()
        self.kind = kw.get("kind", "mask")
        self.title = kw.get("title", "t")
        self.source_track_id = kw.get("source_track_id")
        self.frame_count = kw.get("frame_count")
        self.fps = kw.get("fps")


class TestMaskPairing:
    """A mask is only valid against the frames it was keyed out of."""

    def test_a_mask_from_another_track_is_refused(self):
        control = _Track(kind="footage", frame_count=80, fps=10)
        other = uuid.uuid4()
        mask = _Track(source_track_id=other, frame_count=30, fps=5)
        with pytest.raises(HTTPException) as exc:
            _validate_mask_pairing(control, mask)
        assert exc.value.status_code == 409
        # The message has to name the track that WOULD work — "wrong mask" on
        # its own is not actionable and the answer is already in the database.
        assert str(other) in exc.value.detail

    def test_the_matching_pair_is_accepted(self):
        control = _Track(kind="footage", frame_count=30, fps=5)
        mask = _Track(source_track_id=control.id, frame_count=30, fps=5)
        _validate_mask_pairing(control, mask)

    def test_lineage_outranks_a_stale_frame_rate(self):
        # Several rows in this library carry 1000 and 2000 fps from the old
        # r_frame_rate probe. A correct pair must not be refused over that.
        control = _Track(kind="footage", frame_count=67, fps=60.8)
        mask = _Track(source_track_id=control.id, frame_count=67, fps=1000.0)
        _validate_mask_pairing(control, mask)

    def test_a_track_that_is_not_a_mask_is_refused(self):
        control = _Track(kind="footage", frame_count=30, fps=5)
        with pytest.raises(HTTPException) as exc:
            _validate_mask_pairing(control, _Track(kind="footage"))
        assert exc.value.status_code == 400

    def test_a_hand_authored_mask_is_checked_on_length(self):
        # No lineage to go on, so the only question left is whether it can
        # physically line up.
        control = _Track(kind="footage", frame_count=94, fps=30)
        with pytest.raises(HTTPException) as exc:
            _validate_mask_pairing(control, _Track(frame_count=30, fps=30))
        assert exc.value.status_code == 409
        assert "30" in exc.value.detail

    def test_a_hand_authored_mask_is_checked_on_rate(self):
        control = _Track(kind="footage", frame_count=94, fps=30)
        with pytest.raises(HTTPException) as exc:
            _validate_mask_pairing(control, _Track(frame_count=94, fps=15))
        assert exc.value.status_code == 409

    def test_a_hand_authored_mask_that_lines_up_is_accepted(self):
        control = _Track(kind="footage", frame_count=94, fps=30)
        _validate_mask_pairing(control, _Track(frame_count=94, fps=30))

    def test_a_longer_mask_is_fine(self):
        # Surplus frames are simply never read.
        control = _Track(kind="footage", frame_count=60, fps=30)
        _validate_mask_pairing(control, _Track(frame_count=94, fps=30))


class TestFrameRateProbe:
    """`r_frame_rate` is the highest rate a stream could contain, not the one it
    plays at. On variable-rate phone footage it produced 1000 and 2000, which is
    what let a 5 fps mask be paired with a 10 fps control track."""

    def test_a_normal_rational_reads_straight(self):
        assert _rate("30000/1001") == pytest.approx(29.97, abs=0.01)
        assert _rate("25/1") == 25.0

    def test_the_absurd_rates_that_caused_this_are_rejected(self):
        assert _rate("1000/1") is None
        assert _rate("2000/1") is None

    def test_ffprobe_s_no_answer_is_not_a_rate(self):
        assert _rate("0/0") is None
        assert _rate(None) is None
        assert _rate("") is None
        assert _rate("nonsense") is None

    def test_the_ceiling_sits_above_anything_real_here(self):
        assert _rate("240/1") == 240.0
        assert _rate("241/1") is None
