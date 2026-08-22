"""
Unit tests for colour harmonisation (services/video/grade.py).

The planning half is pure, so the properties that decide whether an automatic
colour pass is trustworthy can be asserted directly: that it moves clips
*towards* each other and never past each other, that a clip already on the
group's numbers is left alone, that every move is capped, and — the one that
was actually wrong first time round — that the filter chain pins its own bit
depth so the chroma shift means what it says.
"""
import pytest

from services.video.grade import (
    DEFAULT_HARMONY,
    GRADE_FORMAT,
    HARMONIES,
    HARMONY_BY_KEY,
    MAX_BRIGHTNESS,
    MAX_CHROMA_SHIFT,
    MAX_CONTRAST,
    Grade,
    Stats,
    clamp_strength,
    grade_for,
    group_target,
    harmonise,
    harmony_options,
    measure_command,
    parse_stats,
    spread,
)


def stats(low=40.0, avg=100.0, high=180.0, u=128.0, v=128.0, sat=20.0):
    return Stats(y_low=low, y_avg=avg, y_high=high, u_avg=u, v_avg=v, sat_avg=sat, frames=10)


# ── Presets ──────────────────────────────────────────────────────────────────

def test_presets_are_ordered_and_stop_short_of_full():
    strengths = [s for _, _, s, _ in HARMONIES]
    assert strengths == sorted(strengths)
    assert strengths[0] == 0.0, "there must be a way to switch it off"
    assert max(strengths) < 1.0, "a full normalisation is the failure this avoids"
    assert DEFAULT_HARMONY in HARMONY_BY_KEY
    assert 0.0 < HARMONY_BY_KEY[DEFAULT_HARMONY] <= 0.5, "the default should be gentle"


def test_harmony_options_matches_the_table():
    opts = harmony_options()
    assert [o["key"] for o in opts] == [k for k, _, _, _ in HARMONIES]
    assert all(o["hint"] and o["label"] for o in opts)


def test_clamp_strength_bounds():
    assert clamp_strength(None) == 0.0
    assert clamp_strength(-3) == 0.0
    assert clamp_strength(2.5) == 1.0
    assert clamp_strength(0.4) == pytest.approx(0.4)


# ── Measurement ──────────────────────────────────────────────────────────────

def test_measure_command_normalises_the_depth_before_measuring():
    """A 10-bit source reports on 0..1023 and an 8-bit one on 0..255; the
    median across a mixed selection would otherwise be meaningless."""
    cmd = measure_command("ffmpeg", "clip.mp4")
    chain = cmd[cmd.index("-vf") + 1]
    assert "format=yuv420p," in chain
    assert chain.index("format=yuv420p") < chain.index("signalstats")


def test_parse_stats_averages_the_frames():
    text = (
        "lavfi.signalstats.YLOW=40\nlavfi.signalstats.YAVG=100\n"
        "lavfi.signalstats.YHIGH=180\nlavfi.signalstats.UAVG=126\n"
        "lavfi.signalstats.VAVG=130\nlavfi.signalstats.SATAVG=20\n"
        "lavfi.signalstats.YLOW=50\nlavfi.signalstats.YAVG=110\n"
        "lavfi.signalstats.YHIGH=190\nlavfi.signalstats.UAVG=128\n"
        "lavfi.signalstats.VAVG=132\nlavfi.signalstats.SATAVG=22\n"
    )
    s = parse_stats(text)
    assert s.y_low == 45 and s.y_avg == 105 and s.y_high == 185
    assert s.u_avg == 127 and s.v_avg == 131
    assert s.frames == 2


def test_parse_stats_on_nothing():
    assert parse_stats("") is None
    assert parse_stats("not stats at all\n") is None


# ── The filter chain ─────────────────────────────────────────────────────────

def test_identity_grade_emits_no_filters():
    assert Grade().filter_chain() == ""
    assert Grade().is_identity
    assert Grade().describe() == "unverändert"


def test_chain_pins_the_depth_before_shifting_chroma():
    """The bug this test exists for: `eq` emits 8-bit whatever it is given, so
    a lut after it works in 8-bit units. Without the leading `format` the shift
    lands four times too far on a 10-bit source — measured 2026-08-22."""
    chain = Grade(contrast=1.1, brightness=0.02, du=3.0, dv=-2.0).filter_chain()
    assert chain.startswith(f"format={GRADE_FORMAT}")
    assert chain.index("format=") < chain.index("lutyuv")
    # 8-bit units in, 8-bit units out: the number in the expression is the
    # number that was measured, with no scale factor to get wrong.
    assert "val+3.0" in chain and "val-2.0" in chain


def test_chain_omits_the_half_that_is_not_needed():
    assert "lutyuv" not in Grade(contrast=1.2).filter_chain()
    assert "eq=" not in Grade(du=4.0).filter_chain()


def test_chroma_shift_is_clipped_in_the_expression():
    chain = Grade(du=9.0).filter_chain()
    assert "clip(" in chain and "minval,maxval" in chain


# ── Targets ──────────────────────────────────────────────────────────────────

def test_group_target_is_the_median_not_the_mean():
    """One heavily stylised clip must not drag the other four towards it."""
    normal = [stats(low=40, high=180) for _ in range(4)]
    outlier = stats(low=200, high=250)
    target = group_target(normal + [outlier])
    assert target.y_low == 40
    assert target.y_high == 180


# ── Grading one clip ─────────────────────────────────────────────────────────

def test_a_clip_already_on_the_target_is_left_alone():
    s = stats()
    assert grade_for(s, s, 0.6).is_identity


def test_zero_strength_changes_nothing():
    assert grade_for(stats(low=10), stats(low=90), 0.0).is_identity


def test_the_move_is_partial_and_in_the_right_direction():
    """Half way at strength 0.5, and never past the target — the whole point
    of harmonising rather than normalising."""
    src = stats(low=20, avg=80, high=140)
    tgt = stats(low=60, avg=110, high=200)
    g = grade_for(src, tgt, 0.5)
    # Where the clip's own black level lands, under eq's measured formula.
    landed_low = ((src.y_low / 255.0 - 0.5) * g.contrast + 0.5 + g.brightness) * 255.0
    assert src.y_low < landed_low < tgt.y_low, landed_low
    assert landed_low == pytest.approx(src.y_low + (tgt.y_low - src.y_low) * 0.5, abs=1.0)


def test_full_strength_lands_on_the_target():
    src = stats(low=20, avg=80, high=140)
    tgt = stats(low=50, avg=105, high=170)
    g = grade_for(src, tgt, 1.0)
    for field in ("y_low", "y_high"):
        got = ((getattr(src, field) / 255.0 - 0.5) * g.contrast + 0.5 + g.brightness) * 255.0
        assert got == pytest.approx(getattr(tgt, field), abs=1.0)


def test_chroma_moves_partway_towards_the_group():
    g = grade_for(stats(u=120, v=140), stats(u=130, v=130), 0.5)
    assert g.du == pytest.approx(5.0)
    assert g.dv == pytest.approx(-5.0)


def test_every_move_is_capped():
    """However far apart two clips are, no single correction may rewrite the
    picture — the caps are what keeps this a harmonisation."""
    g = grade_for(stats(low=5, avg=20, high=40, u=90, v=170),
                  stats(low=90, avg=160, high=250, u=160, v=90), 1.0)
    assert 1.0 / MAX_CONTRAST <= g.contrast <= MAX_CONTRAST
    assert abs(g.brightness) <= MAX_BRIGHTNESS + 1e-9
    assert abs(g.du) <= MAX_CHROMA_SHIFT + 1e-9
    assert abs(g.dv) <= MAX_CHROMA_SHIFT + 1e-9


def test_a_flat_clip_gets_a_level_shift_not_a_contrast_stretch():
    """Stretching a nearly toneless clip to the group's range would amplify
    what little is there into noise."""
    flat = stats(low=118, avg=120, high=122)
    g = grade_for(flat, stats(low=40, avg=100, high=180), 0.6)
    assert g.contrast == 1.0
    assert g.brightness < 0, "it is brighter than the group, so it should come down"


def test_a_flat_clip_does_not_divide_by_zero():
    same = stats(low=100, avg=100, high=100)
    g = grade_for(same, stats(low=40, avg=100, high=180), 0.8)
    assert g.contrast == 1.0


# ── Grading a selection ──────────────────────────────────────────────────────

def test_harmonise_converges_the_group():
    group = [stats(low=20, avg=70, high=140), stats(low=40, avg=100, high=180),
             stats(low=60, avg=130, high=220)]
    grades = harmonise(group, 0.6)

    def landed(s, g):
        return ((s.y_low / 255.0 - 0.5) * g.contrast + 0.5 + g.brightness) * 255.0

    before = max(s.y_low for s in group) - min(s.y_low for s in group)
    after_levels = [landed(s, g) for s, g in zip(group, grades)]
    assert max(after_levels) - min(after_levels) < before


def test_harmonise_needs_a_group_to_harmonise_towards():
    """One clip has nothing to be harmonised WITH, and inventing a target for
    it would just be an unrequested auto-correction."""
    assert harmonise([stats()], 0.8) == [Grade()]
    assert harmonise([], 0.8) == []


def test_an_unmeasurable_clip_passes_through_and_is_left_out_of_the_median():
    group = [stats(low=20), None, stats(low=60), stats(low=40)]
    grades = harmonise(group, 0.6)
    assert len(grades) == 4
    assert grades[1].is_identity, "a clip that could not be read must not be graded"
    # The median of the three that WERE read is 40, so that one is already home.
    assert grades[3].is_identity


def test_harmonise_off_is_all_identity():
    group = [stats(low=20), stats(low=60), stats(low=40)]
    assert harmonise(group, 0.0) == [Grade(), Grade(), Grade()]


# ── Reporting ────────────────────────────────────────────────────────────────

def test_spread_reports_the_gap_before_anything_is_done():
    got = spread([stats(low=20, high=140, u=120), stats(low=60, high=200, u=134)])
    assert got["measured"] == 2
    assert got["black"] == 40
    assert got["white"] == 60
    assert got["colour"] == 14


def test_spread_of_one_clip_is_no_spread():
    assert spread([stats()])["measured"] == 1
    assert spread([None, None])["black"] == 0.0


def test_describe_names_what_changed():
    text = Grade(contrast=1.20, brightness=0.05, du=3.0, dv=-2.0).describe()
    assert "Kontrast" in text and "Helligkeit" in text and "Farbe" in text
