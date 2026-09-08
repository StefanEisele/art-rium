"""The frame budget, and the palette invariant the mask path depends on."""
import math

import pytest

from services.segment import (
    FRAME_CEILING,
    MAX_REGIONS,
    REGION_COLORS,
    REGION_LABELS,
    budget_dict,
    plan_frames,
    trim_filters,
)


# ── The palette ──────────────────────────────────────────────────────────────

def test_every_pair_of_keys_is_maximally_far_apart():
    """The whole mask path rests on this.

    ColorToMask keys a region by euclidean RGB distance to an exact triple, so
    the distance between two keys is the error budget a pixel has before it
    keys into the wrong region. Pure primaries are 360 apart, which is the most
    three colours can manage — anything closer would be spending that budget
    for nothing.
    """
    for i, a in enumerate(REGION_COLORS):
        for b in REGION_COLORS[i + 1:]:
            distance = math.dist(a, b)
            assert distance == pytest.approx(360.62, abs=0.1)


def test_black_is_not_a_region():
    """Unassigned pixels are black, so black must not key as a region — and at
    255 away from every primary it cannot."""
    for color in REGION_COLORS:
        assert math.dist(color, (0, 0, 0)) == 255


def test_palette_and_labels_line_up_with_the_render_cap():
    assert len(REGION_COLORS) == len(REGION_LABELS) == MAX_REGIONS


# ── The frame budget ─────────────────────────────────────────────────────────

def test_seconds_and_stride_compose():
    """The example in the request: 6 s of a 30 fps take, every 2nd frame."""
    b = plan_frames(600, 30.0, seconds=6, stride=2)
    assert b.taken == 180
    assert b.kept == 90
    assert b.span == 6.0


def test_stride_keeps_the_first_frame_so_the_count_rounds_up():
    """`select='not(mod(n,S))'` keeps frame 0 and every Sth after it, so 7
    frames at stride 3 is {0,3,6} — three, not two."""
    assert plan_frames(7, 30.0, stride=3).kept == 3
    assert plan_frames(9, 30.0, stride=3).kept == 3
    assert plan_frames(10, 30.0, stride=3).kept == 4


def test_stride_divides_the_playback_rate():
    """The thinned track still covers the same seconds, so it plays back at a
    lower rate — that is what makes the browser preview honest about duration."""
    b = plan_frames(300, 30.0, stride=3)
    assert b.fps == 10.0
    assert b.span == pytest.approx(10.0)


def test_a_trim_never_invents_frames_the_source_does_not_have():
    b = plan_frames(50, 30.0, seconds=10)
    assert b.taken == 50
    assert b.kept == 50


def test_start_offset_comes_out_of_the_available_frames():
    b = plan_frames(300, 30.0, seconds=2, start=5.0)
    assert b.start == 5.0
    assert b.taken == 60          # 2 s at 30 fps, well inside what is left
    b_late = plan_frames(300, 30.0, start=9.0)
    assert b_late.taken == 30     # only the last second survives


def test_the_ceiling_clamps_and_says_so():
    b = plan_frames(10_000, 30.0)
    assert b.kept == FRAME_CEILING
    assert b.capped is True
    assert plan_frames(100, 30.0).capped is False


def test_the_ceiling_reports_the_frames_it_actually_reads():
    """When the cap bites, `taken` has to shrink with it — it is what the
    ingest passes to ffmpeg, and reading 10 000 frames to keep 400 is the waste
    the budget exists to avoid."""
    b = plan_frames(10_000, 30.0, stride=2)
    assert b.kept == FRAME_CEILING
    assert b.taken == FRAME_CEILING * 2


def test_an_empty_source_is_a_zero_budget_not_a_crash():
    b = plan_frames(0, None)
    assert b.kept == 0
    assert b.span == 0.0


def test_a_missing_source_rate_falls_back_rather_than_dividing_by_zero():
    b = plan_frames(120, None, seconds=2)
    assert b.source_fps == 30.0
    assert b.kept == 60


def test_stride_is_floored_at_one():
    assert plan_frames(100, 30.0, stride=0).stride == 1
    assert plan_frames(100, 30.0, stride=-4).stride == 1


# ── The ffmpeg side ──────────────────────────────────────────────────────────

def test_stride_one_only_rebases_the_timestamps():
    """Nothing to thin, so no `select` — but the rebase stays, which is what
    makes a clip trimmed with `-ss` start at zero."""
    assert trim_filters(plan_frames(100, 30.0, stride=1)) == ["setpts=N/30/TB"]


def test_the_select_expression_escapes_its_comma():
    """ffmpeg splits filter arguments on commas, so `mod(n,3)` has to reach it
    as `mod(n\\,3)` or the filter graph fails to parse."""
    select, _ = trim_filters(plan_frames(100, 30.0, stride=3))
    assert select == "select='not(mod(n\\,3))'"


def test_the_rebase_uses_the_thinned_rate_not_the_source_rate():
    """The frames that survive have to be spread over the seconds they actually
    cover. Writing them at the source rate is what made a stride-2 clip play
    back at double speed — measured, not hypothetical."""
    _, setpts = trim_filters(plan_frames(171, 30.0, stride=2))
    assert setpts == "setpts=N/15/TB"


def test_budget_dict_carries_what_the_frontend_shows():
    d = budget_dict(plan_frames(600, 30.0, seconds=6, stride=2))
    assert d["frames"] == 90
    assert d["fps"] == 15.0
    assert d["ceiling"] == FRAME_CEILING
    assert d["capped"] is False
