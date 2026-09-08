"""
Unit tests for the cut planner (services/video/cut.py) and its ffmpeg graph
(services/video/cut_render.py).

Both are pure, so the properties that actually make an edit watchable can be
asserted directly: the cuts tile the song without a gap or an overlap, they
land on beats, the same clip never appears on both sides of a cut, long shots
open on a downbeat, a shot is never asked for more film than it has, and the
rendered segment lengths add up to the plan to the frame.
"""
import pytest

from services.video.beats import BeatMap
from services.video.cut import (
    LADDER,
    MAX_STRETCH_DEFAULT,
    MIN_CUT_SECONDS,
    STYLE_BY_KEY,
    STYLES,
    Cut,
    EditPlan,
    Source,
    Style,
    plan_cut,
    rung_for_energy,
    style_options,
)
from services.video.cut_render import (
    RenderSource,
    build_cut_command,
    plan_duration,
    segment_frames,
)

import random
from pathlib import Path


def beatmap(bpm=120.0, bars=16, beats_per_bar=4, phase=0, energy=None, sections=(0,)):
    """A perfectly regular grid — the planner's decisions are what is under
    test here, not the tracker's."""
    period = 60.0 / bpm
    n = bars * beats_per_bar + phase + 1
    beats = [round(i * period, 4) for i in range(n)]
    if energy is None:
        energy = [i / max(1, bars - 1) for i in range(bars)]
    return BeatMap(
        duration=round(beats[-1] + period, 4),
        bpm=bpm,
        beats=beats,
        beats_per_bar=beats_per_bar,
        downbeat_phase=phase,
        beat_strength=[0.5] * n,
        bar_energy=list(energy[:bars]) + [0.5] * max(0, bars - len(energy)),
        sections=list(sections),
        confidence=0.8,
    )


def sources(n=4, duration=4.0):
    return [Source(key=f"clip:{i}", duration=duration, label=f"Clip {i + 1}") for i in range(n)]


# ── Styles ───────────────────────────────────────────────────────────────────

def test_every_style_is_declared_consistently():
    for st in STYLES:
        assert 0 <= st.quiet < len(LADDER)
        assert 0 <= st.loud < len(LADDER)
        assert st.quiet <= st.loud, f"{st.key} cuts faster when quiet"
        assert st.hint and st.label


def test_style_options_reports_the_rungs():
    opts = {o["key"]: o for o in style_options()}
    assert opts.keys() == STYLE_BY_KEY.keys()
    for key, opt in opts.items():
        st = STYLE_BY_KEY[key]
        assert opt["slowest"] == LADDER[st.quiet]
        assert opt["fastest"] == LADDER[st.loud]
        assert opt["fastest"] <= opt["slowest"]


def test_rung_for_energy_moves_with_the_music():
    st = Style("t", "T", "", quiet=0, loud=4)
    steady = random.Random(0)
    assert rung_for_energy(st, 0.0, steady) == 0
    assert rung_for_energy(st, 1.0, steady) == 4
    assert 0 < rung_for_energy(st, 0.5, steady) < 4


def test_rung_for_energy_clamps_out_of_range_input():
    st = Style("t", "T", "", quiet=1, loud=3)
    rng = random.Random(0)
    assert 0 <= rung_for_energy(st, -5.0, rng) < len(LADDER)
    assert 0 <= rung_for_energy(st, 5.0, rng) < len(LADDER)


# ── Structural guarantees ────────────────────────────────────────────────────

@pytest.mark.parametrize("style", [s.key for s in STYLES])
def test_cuts_tile_the_span_without_gap_or_overlap(style):
    plan = plan_cut(beatmap(), sources(), style=style, seed=11)
    assert plan.cuts
    assert plan.cuts[0].start == pytest.approx(plan.song_start)
    assert plan.cuts[-1].end == pytest.approx(plan.song_end)
    for a, b in zip(plan.cuts, plan.cuts[1:]):
        assert a.end == pytest.approx(b.start), "a gap or an overlap between shots"
        assert a.end > a.start


@pytest.mark.parametrize("style", [s.key for s in STYLES])
def test_no_shot_repeats_across_a_cut(style):
    plan = plan_cut(beatmap(bars=24), sources(5), style=style, seed=3)
    for a, b in zip(plan.cuts, plan.cuts[1:]):
        assert a.source != b.source, "the same picture on both sides of a cut"


def test_cut_boundaries_land_on_beats():
    bm = beatmap()
    plan = plan_cut(bm, sources(), style="welle", seed=5)
    grid = set(bm.beats)
    for cut in plan.cuts[1:]:
        assert any(abs(cut.start - b) < 1e-3 for b in grid), cut.start


def test_long_shots_open_on_a_downbeat():
    """A bar-or-longer shot starting mid-bar is shortened instead, which
    re-syncs the grid rather than dragging the offset through the piece."""
    bm = beatmap(bars=24, phase=0)
    plan = plan_cut(bm, sources(), style="dramaturgie", seed=8)
    for cut in plan.cuts[1:]:
        if cut.beats >= bm.beats_per_bar:
            assert (cut.beat - bm.downbeat_phase) % bm.beats_per_bar == 0, cut


def test_the_edit_covers_the_whole_song():
    """Not just the tracked grid: a 4/4 track whose first downbeat is beat 3
    would otherwise open on a second of music with no picture over it."""
    bm = beatmap(bars=12, phase=3)
    plan = plan_cut(bm, sources(), style="welle", seed=2)
    assert plan.song_start == 0.0
    assert plan.song_end == pytest.approx(bm.duration)
    assert plan.cuts[0].start == 0.0


def test_a_later_start_bar_offsets_the_song_instead():
    bm = beatmap(bars=16)
    plan = plan_cut(bm, sources(), style="welle", seed=2, start_bar=4)
    assert plan.song_start > 0.5
    assert plan.cuts[0].start == pytest.approx(plan.song_start)


def test_sections_force_a_cut():
    bm = beatmap(bars=16, sections=(0, 6, 11))
    plan = plan_cut(bm, sources(), style="atem", seed=1)
    starts = {c.beat for c in plan.cuts}
    for bar in (6, 11):
        assert bm.bar_starts()[bar] in starts, f"held a shot across the change at bar {bar}"
    assert any(c.section and c.fade_in > 0 for c in plan.cuts[1:])


def test_planning_is_deterministic_in_the_seed():
    bm, srcs = beatmap(), sources()
    a = plan_cut(bm, srcs, style="dramaturgie", seed=42)
    b = plan_cut(bm, srcs, style="dramaturgie", seed=42)
    c = plan_cut(bm, srcs, style="dramaturgie", seed=43)
    assert a.to_json() == b.to_json()
    assert a.to_json() != c.to_json(), "reseeding produced the same edit"


def test_an_unseeded_plan_records_the_seed_it_rolled():
    """The preview and the render are two separate requests; the second one
    reproduces the first only because the roll came back with it."""
    plan = plan_cut(beatmap(), sources(), style="puls")
    assert isinstance(plan.seed, int)
    again = plan_cut(beatmap(), sources(), style="puls", seed=plan.seed)
    assert again.to_json() == plan.to_json()


def test_plan_round_trips_through_json():
    plan = plan_cut(beatmap(), sources(), style="welle", seed=4)
    again = EditPlan.from_json(plan.to_json())
    assert again.to_json() == plan.to_json()
    assert isinstance(again.cuts[0], Cut)


# ── Pacing ───────────────────────────────────────────────────────────────────

def test_loud_bars_cut_faster_than_quiet_ones():
    """The one property that separates this from a metronome."""
    bm = beatmap(bars=24, energy=[0.0] * 12 + [1.0] * 12, sections=(0,))
    plan = plan_cut(bm, sources(6), style="dramaturgie", seed=6)
    half = bm.bar_starts()[12]
    quiet = [c.beats for c in plan.cuts if c.beat < half]
    loud = [c.beats for c in plan.cuts if c.beat >= half]
    assert quiet and loud
    assert sum(quiet) / len(quiet) > sum(loud) / len(loud)


def test_a_faster_style_makes_more_cuts():
    bm, srcs = beatmap(bars=24), sources(6)
    counts = {
        key: len(plan_cut(bm, srcs, style=key, seed=9).cuts)
        for key in ("atem", "welle", "puls", "stakkato")
    }
    ordered = [counts["atem"], counts["welle"], counts["puls"], counts["stakkato"]]
    assert ordered == sorted(ordered), counts


def test_no_cut_is_shorter_than_the_floor():
    bm = beatmap(bpm=190.0, bars=20)
    plan = plan_cut(bm, sources(6), style="stakkato", seed=7)
    # The final shot is bounded by the song's end, not by the ladder.
    for cut in plan.cuts[:-1]:
        assert cut.end - cut.start >= MIN_CUT_SECONDS - 1e-6


# ── Material ─────────────────────────────────────────────────────────────────

def test_a_shot_is_never_read_past_its_end():
    plan = plan_cut(beatmap(bars=20), sources(4, duration=3.0), style="dramaturgie", seed=12)
    srcs = sources(4, duration=3.0)
    for cut in plan.cuts:
        assert 0 <= cut.src_in <= cut.src_out <= srcs[cut.source].duration + 1e-6


def test_short_clips_are_slowed_rather_than_the_rhythm_broken():
    """A 16-beat hold at 120 BPM is 8 s and the clips are 3 s. The shot gives
    way, the grid does not."""
    bm = beatmap(bpm=120.0, bars=16, energy=[0.0] * 16)
    plan = plan_cut(bm, sources(4, duration=3.0), style="atem", seed=1,
                    max_stretch=MAX_STRETCH_DEFAULT)
    assert any(c.speed < 0.99 for c in plan.cuts)
    for cut in plan.cuts:
        assert cut.speed >= 1.0 / MAX_STRETCH_DEFAULT - 0.35
    assert any("Zeitlupe" in w for w in plan.warnings)


def test_stretch_is_not_used_when_the_material_is_long_enough():
    bm = beatmap(bpm=120.0, bars=16, energy=[1.0] * 16)
    plan = plan_cut(bm, sources(4, duration=30.0), style="puls", seed=1)
    assert all(c.speed == pytest.approx(1.0) for c in plan.cuts)
    assert not any("Zeitlupe" in w for w in plan.warnings)


def test_a_source_that_never_appears_is_reported():
    """Twenty clips and four bars of music: most of them cannot fit, and
    silently dropping them would look like a bug from the outside."""
    plan = plan_cut(beatmap(bars=3), sources(20), style="atem", seed=1)
    assert any("kamen nicht vor" in w for w in plan.warnings)


def test_one_source_is_a_legal_edit():
    plan = plan_cut(beatmap(bars=8), sources(1, duration=6.0), style="welle", seed=1)
    assert plan.cuts
    assert all(c.source == 0 for c in plan.cuts)


def test_no_sources_is_an_error():
    with pytest.raises(ValueError):
        plan_cut(beatmap(), [], style="welle")


def test_unknown_style_falls_back_to_the_default():
    plan = plan_cut(beatmap(), sources(), style="nonsense", seed=1)
    assert plan.style in STYLE_BY_KEY


# ── Anticipation ─────────────────────────────────────────────────────────────

def test_anticipation_pulls_every_cut_ahead_of_its_beat():
    bm, srcs = beatmap(bars=16), sources()
    plain = plan_cut(bm, srcs, style="welle", seed=5)
    early = plan_cut(bm, srcs, style="welle", seed=5, anticipation=0.08)
    assert early.cuts[0].start == plain.cuts[0].start        # nothing before the first
    assert early.cuts[-1].end == pytest.approx(plain.cuts[-1].end)
    for a, b in zip(plain.cuts[1:], early.cuts[1:]):
        assert b.start < a.start
    for a, b in zip(early.cuts, early.cuts[1:]):
        assert a.end == pytest.approx(b.start)


# ── Rendering ────────────────────────────────────────────────────────────────

def test_segment_frames_never_accumulates_drift():
    """Rounding each duration on its own would let thirty cuts drift more than
    a frame off the music; rounding both ends against the same origin cannot."""
    fps = 24
    bounds = [i * 0.4567 for i in range(41)]
    total = sum(segment_frames(a, b, fps) for a, b in zip(bounds, bounds[1:]))
    assert total == round(bounds[-1] * fps) - round(bounds[0] * fps)


def test_segment_frames_is_at_least_one():
    assert segment_frames(1.0, 1.0001, 24) == 1


def test_plan_duration_matches_the_rendered_frame_count():
    plan = plan_cut(beatmap(bars=16), sources(), style="welle", seed=2)
    frames = sum(segment_frames(c.start, c.end, 24) for c in plan.cuts)
    assert plan_duration(plan, 24) == pytest.approx(frames / 24)
    assert abs(plan_duration(plan, 24) - plan.duration) < 1 / 24


def test_build_cut_command_decodes_each_source_once():
    plan = plan_cut(beatmap(bars=16), sources(3), style="puls", seed=2)
    srcs = [RenderSource(Path(f"/tmp/c{i}.mp4"), 4.0) for i in range(3)]
    cmd = build_cut_command("ffmpeg", plan, srcs, Path("/tmp/out.mp4"), 960, 544, 24)

    assert cmd.count("-i") == 3, "a source was opened once per appearance"
    graph = cmd[cmd.index("-filter_complex") + 1]
    for i, used in enumerate(sorted({c.source for c in plan.cuts})):
        assert f"[{i}:v]split=" in graph or f"[{i}:v]null" in graph
    assert f"concat=n={len(plan.cuts)}:v=1:a=0" in graph


def test_build_cut_command_is_frame_exact_and_silent():
    plan = plan_cut(beatmap(bars=12), sources(3), style="welle", seed=1)
    srcs = [RenderSource(Path(f"/tmp/c{i}.mp4"), 4.0) for i in range(3)]
    cmd = build_cut_command("ffmpeg", plan, srcs, Path("/tmp/out.mp4"), 864, 480, 24)
    graph = cmd[cmd.index("-filter_complex") + 1]

    assert "-an" in cmd, "a beat cut carries the song, not the clips' own audio"
    # Every branch is trimmed to an exact frame count, with a cloned tail
    # standing by in case rounding leaves it one frame short.
    for cut in plan.cuts:
        want = segment_frames(cut.start, cut.end, 24)
        assert f"trim=start_frame=0:end_frame={want}" in graph
    assert graph.count("tpad=stop_mode=clone") == len(plan.cuts)
    # The opening fade is opt-in (default off); the closing one always runs —
    # see TestFadeIn below for both states.
    assert "fade=t=in:st=0" not in graph and "fade=t=out" in graph


def test_build_cut_command_carries_the_speed_change():
    bm = beatmap(bpm=120.0, bars=12, energy=[0.0] * 12)
    plan = plan_cut(bm, sources(3, duration=2.0), style="atem", seed=1)
    slowed = [c for c in plan.cuts if c.speed < 0.99]
    assert slowed, "this fixture is supposed to need slow motion"
    srcs = [RenderSource(Path(f"/tmp/c{i}.mp4"), 2.0) for i in range(3)]
    cmd = build_cut_command("ffmpeg", plan, srcs, Path("/tmp/o.mp4"), 640, 640, 24)
    graph = cmd[cmd.index("-filter_complex") + 1]
    for cut in slowed:
        assert f"setpts=(PTS-STARTPTS)/{cut.speed:.5f}" in graph


def test_build_cut_command_rejects_an_empty_plan():
    plan = plan_cut(beatmap(), sources(), style="welle", seed=1)
    plan.cuts = []
    with pytest.raises(ValueError):
        build_cut_command("ffmpeg", plan, [], Path("/tmp/o.mp4"), 640, 640, 24)


class TestFadeIn:
    """A picture edit starting on a black frame is a choice. It used to be
    made unconditionally for every render; now it is off unless asked for."""

    def _cmd(self, **kw):
        plan = plan_cut(beatmap(bars=12), sources(3), style="welle", seed=1)
        srcs = [RenderSource(Path(f"/tmp/c{i}.mp4"), 4.0) for i in range(3)]
        return build_cut_command("ffmpeg", plan, srcs, Path("/tmp/out.mp4"),
                                 864, 480, 24, **kw)

    def test_default_is_no_opening_fade(self):
        cmd = self._cmd()
        graph = cmd[cmd.index("-filter_complex") + 1]
        assert "fade=t=in:st=0" not in graph

    def test_the_closing_fade_runs_regardless(self):
        # Ending on a hard cut to nothing reads as a mistake in a way an
        # unfaded opening does not — this one stays unconditional.
        for kw in ({}, {"fade_in": True}):
            graph = self._cmd(**kw)
            graph = graph[graph.index("-filter_complex") + 1]
            assert "fade=t=out" in graph

    def test_fade_in_true_adds_the_opening_fade(self):
        graph = self._cmd(fade_in=True)
        graph = graph[graph.index("-filter_complex") + 1]
        assert "fade=t=in:st=0" in graph
