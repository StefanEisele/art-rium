"""
The Schichtenschnitt planner and its renderer.

Almost everything here is one invariant looked at from a different angle:
**every track reads the same source position at the same output moment.** That
is not a nicety — it is the entire effect. The tracks came out of one control
video, so at loop position 1.83 s they all show the same form; dissolving them
there swaps the material while the shape holds. Let the positions drift apart
by a frame and it becomes an ordinary cross-fade between two unrelated
pictures, which is a thing the beat cut already does better.
"""
import math
import re
from pathlib import Path

import pytest

from services.video.beats import BeatMap
from services.video.cut import LADDER
from services.video.cut_render import (
    DEFAULT_MOTION,
    MOTION_BLEND,
    MOTION_FLOW,
    MOTION_ORIGINAL,
    clamp_motion,
    motion_filter,
    motion_options,
    spill_filtergraph,
)
from services.video.layer_render import (
    LayerSource,
    build_layer_command,
    plan_duration,
)
from services.video.layers import (
    BLEND_NEUTRAL,
    BLEND_PEAK,
    BLEND_PIX_FMT,
    MAX_BLEND_SHARE,
    MIN_BLEND_SECONDS,
    SPEED_MAX,
    SPEED_MIN,
    STYLE_BY_KEY,
    blend_expr,
    clamp_speed,
    plan_layers,
    speed_for_energy,
    style_options,
    transition_options,
    wrap_times,
)


# ── Evaluating an ffmpeg blend expression in Python ──────────────────────────
# The expressions are small enough to read numerically, and reading them
# numerically is the only way to test what actually went wrong here: "the white
# flash came out magenta" is a claim about a *number* on the chroma plane, and
# no amount of substring matching would have caught it.

def _clip(x, lo, hi):
    return max(lo, min(hi, x))


def evaluate(expr, *, a, b, p, duration=1.0, x=0.5, y=0.5):
    """Value of one ffmpeg blend expression, with `p` as progress 0..1.

    `x`/`y` are normalised positions in the frame; the expressions only ever
    use X/W and Y/H, so a width of 1 makes X the normalised coordinate.
    """
    env = {
        "A": float(a), "B": float(b), "T": p * duration,
        "X": float(x), "W": 1.0, "Y": float(y), "H": 1.0,
        "pow": pow, "sin": math.sin, "clip": _clip,
    }
    return eval(expr, {"__builtins__": {}}, env)   # noqa: S307 — our own strings


BLENDED = [o["key"] for o in transition_options() if o["key"] not in ("hart", "mix")]


def beatmap(seconds=64.0, bpm=120.0, energy=None):
    """A clean 4/4 grid. Synthetic on purpose — these tests are about the
    planner's arithmetic, not about beat tracking."""
    step = 60.0 / bpm
    n = int(seconds / step)
    bars = max(1, n // 4)
    if energy is None:
        energy = [0.2] * (bars // 2) + [0.9] * (bars - bars // 2)
    energy = (energy + [energy[-1]] * bars)[:bars]
    return BeatMap(
        duration=seconds, bpm=bpm, beats=[i * step for i in range(n)],
        beats_per_bar=4, downbeat_phase=0, beat_strength=[0.6] * n,
        bar_energy=energy, sections=[bars // 2], confidence=0.9,
    )


def plan(**kw):
    kw.setdefault("tracks", 5)
    kw.setdefault("loop", 3.2)
    kw.setdefault("seed", 7)
    bm = kw.pop("beatmap", None) or beatmap()
    return plan_layers(bm, **kw)


class TestSharedSourcePosition:
    """The invariant, four ways."""

    def test_the_source_position_never_jumps(self):
        # Continuous everywhere except at the loop seam, where it is supposed
        # to jump: that is the loop coming back around, and the planner put a
        # slot boundary exactly there so the jump lands between two slots
        # rather than inside one.
        p = plan(loop=3.2)
        for a, b in zip(p.slots, p.slots[1:]):
            seam = a.src_out == pytest.approx(3.2, abs=1e-6) and b.src_in == pytest.approx(0.0, abs=1e-6)
            assert seam or b.src_in == pytest.approx(a.src_out, abs=1e-9), (a, b)

    def test_the_source_position_never_runs_backwards(self):
        p = plan()
        for slot in p.slots:
            assert slot.src_out > slot.src_in

    def test_the_output_timeline_has_no_gaps(self):
        p = plan()
        for a, b in zip(p.slots, p.slots[1:]):
            assert b.start == pytest.approx(a.end, abs=1e-9)

    def test_a_transition_reads_one_window_for_both_tracks(self):
        # Both halves of a dissolve are the same trim. If they were not, the
        # form would move underneath the transition and the whole point of the
        # mode would be gone — so the renderer builds them from one slot.
        p = plan(transition="weich")
        blends = [s for s in p.slots if s.is_blend]
        assert blends
        for s in blends:
            assert s.from_track != s.track
            assert s.src_out > s.src_in


class TestSpeedFollowsTheMusic:
    def test_loud_bars_run_faster_than_quiet_ones(self):
        # With the accent off, so this is about the swell alone. The speeds no
        # longer land exactly on the endpoints — the curve is sampled where the
        # knots fall, not at the extremes of every bar — so what is asserted is
        # the relationship, which is what the dial actually promises.
        p = plan(beatmap=beatmap(energy=[0.0] * 16 + [1.0] * 16),
                 speed_low=0.6, speed_high=1.8, pulse=0.0)
        half = (p.song_start + p.song_end) / 2
        quiet = [s.speed for s in p.slots if s.end <= half]
        loud = [s.speed for s in p.slots if s.start >= half]
        assert sum(quiet) / len(quiet) < sum(loud) / len(loud) - 0.3
        assert all(0.6 - 1e-6 <= s.speed <= 1.8 + 1e-6 for s in p.slots)

    def test_the_whole_stack_is_retimed_together(self):
        # One speed per slot, applied to every track in it — there is nowhere
        # in the structure for two tracks to run at different rates.
        for slot in plan().slots:
            assert isinstance(slot.speed, float) and slot.speed > 0

    def test_speed_is_clamped_to_something_watchable(self):
        p = plan(speed_low=0.001, speed_high=99.0)
        assert p.speed_low == SPEED_MIN and p.speed_high == SPEED_MAX

    def test_reversed_bounds_are_swapped_rather_than_refused(self):
        p = plan(speed_low=1.8, speed_high=0.6)
        assert p.speed_low < p.speed_high

    def test_energy_maps_linearly(self):
        assert speed_for_energy(0.0, 0.5, 1.5) == pytest.approx(0.5)
        assert speed_for_energy(1.0, 0.5, 1.5) == pytest.approx(1.5)
        assert speed_for_energy(0.5, 0.5, 1.5) == pytest.approx(1.0)

    def test_clamp_speed_falls_back_on_garbage(self):
        assert clamp_speed(None, 0.9) == 0.9
        assert clamp_speed("nonsense", 0.9) == 0.9


class TestLengthsComeOffTheLadder:
    def test_every_hold_is_a_recognisable_number_of_beats(self):
        # Not exact, because a hold is what is left of a segment after its
        # transition — but the segments themselves are ladder rungs, so no hold
        # should be longer than the longest rung.
        p = plan(bpm=120.0) if False else plan()
        beat = 60.0 / p.bpm
        assert max(s.duration for s in p.slots) <= LADDER[0] * beat + 1e-6

    def test_a_fast_style_produces_more_slots_than_a_slow_one(self):
        slow = plan(style="schweben")
        fast = plan(style="flimmern")
        assert len(fast.slots) > len(slow.slots)

    def test_covers_the_whole_span(self):
        p = plan()
        assert p.slots[0].start == pytest.approx(p.song_start)
        assert p.slots[-1].end == pytest.approx(p.song_end)


class TestTransitions:
    def test_a_hard_style_produces_no_blend_slots(self):
        p = plan(transition="hart")
        assert not any(s.is_blend for s in p.slots)

    def test_mixed_uses_more_than_one_kind(self):
        p = plan(transition="mix", style="puls")
        kinds = {s.blend for s in p.slots if s.is_blend}
        assert len(kinds) > 1

    def test_the_piece_is_not_mostly_transition(self):
        # Otherwise a 1-beat segment is nothing but transition and the form
        # never gets a moment to simply stand there. Checked in aggregate
        # because a single transition can be split across a loop seam.
        p = plan(style="flimmern", transition="weich")
        blended = sum(s.duration for s in p.slots if s.is_blend)
        assert blended <= p.duration * MAX_BLEND_SHARE + 1e-6

    def test_a_transition_shorter_than_the_floor_is_only_ever_a_fragment(self):
        # The floor is enforced on whole transitions. A slot below it therefore
        # has to be one piece of one that was split at a loop seam, and its
        # progress span says so.
        p = plan(style="flimmern", transition="wisch")
        for s in p.slots:
            if s.is_blend and s.duration < MIN_BLEND_SECONDS:
                assert s.blend_p0 > 0.0 or s.blend_p1 < 1.0, s

    def test_every_expression_mentions_both_inputs_and_the_clock(self):
        for key in BLENDED:
            e = blend_expr(key, 0.5)
            for plane in (e.luma, e.chroma):
                assert "A" in plane and "B" in plane and "T" in plane, key

    def test_a_split_transition_carries_its_progress_across_the_seam(self):
        # The second half must continue the dissolve, not restart it — so its
        # expression has to start from where the first half stopped.
        whole = blend_expr("weich", 1.0)
        second = blend_expr("weich", 0.4, 0.6, 1.0)
        assert "0.60000" in second.luma and second.luma != whole.luma

    def test_both_halves_of_a_split_pick_the_same_variant(self):
        # `variant` comes off the slot's beat, which `_reslice` copies onto
        # both halves. If the two halves disagreed, the wipe would change
        # direction mid-transition, at exactly the loop seam.
        first = blend_expr("wisch", 0.4, 0.0, 0.6, variant=3)
        second = blend_expr("wisch", 0.4, 0.6, 1.0, variant=3)
        axis = re.compile(r"\(1?-?X/W\)|\(1?-?Y/H\)")
        assert axis.search(first.luma).group() == axis.search(second.luma).group()

    def test_a_variant_actually_changes_the_shape(self):
        shapes = {blend_expr("wisch", 1.0, variant=v).luma for v in range(4)}
        assert len(shapes) == 4
        clouds = {blend_expr("aufloesen", 1.0, variant=v).luma for v in range(4)}
        assert len(clouds) == 4

    def test_an_unblended_kind_is_refused_rather_than_guessed(self):
        with pytest.raises(ValueError):
            blend_expr("hart", 0.5)


class TestLoopWrapping:
    def test_the_plan_walks_past_the_loop_and_keeps_going(self):
        # The whole reason six three-second loops can carry a four-minute
        # track. `consumed` is the distance travelled, not the final position.
        p = plan(loop=3.2)
        assert p.source_consumed > 3.2

    def test_wraps_are_reported_where_the_picture_restarts(self):
        p = plan(loop=3.2)
        assert p.wraps
        assert p.source_consumed / 3.2 == pytest.approx(len(p.wraps), abs=1.0)

    def test_wraps_land_inside_the_piece(self):
        p = plan(loop=3.2)
        assert all(p.song_start <= w <= p.song_end + 1e-6 for w in p.wraps)

    def test_a_loop_longer_than_the_piece_never_wraps(self):
        p = plan(loop=10_000.0)
        assert p.wraps == []

    def test_the_split_slots_no_longer_cross_anything(self):
        # wrap_times is what found the seams; after the split there are none
        # left to find, which is the whole point of having split.
        p = plan(loop=3.2)
        assert wrap_times(p.slots, 3.2) == []

    def test_splitting_preserves_the_timeline_and_the_distance(self):
        from services.video.layers import Slot, split_at_wraps
        whole = Slot(start=0.0, end=4.0, src_in=1.0, src_out=9.0,
                     speed_in=2.0, speed_out=2.0,
                     track=1, from_track=0, blend="weich", bar=0, beat=0,
                     beats=8, section=False)
        parts = split_at_wraps([whole], 3.2)
        assert len(parts) == 3                      # 1→3.2, 3.2→6.4, 6.4→9
        assert parts[0].start == 0.0 and parts[-1].end == pytest.approx(4.0)
        for a, b in zip(parts, parts[1:]):
            assert b.start == pytest.approx(a.end)
        travelled = sum(s.src_out - s.src_in for s in parts)
        assert travelled == pytest.approx(8.0)

    def test_splitting_hands_each_piece_its_share_of_the_transition(self):
        from services.video.layers import Slot, split_at_wraps
        whole = Slot(start=0.0, end=4.0, src_in=1.0, src_out=9.0,
                     speed_in=2.0, speed_out=2.0,
                     track=1, from_track=0, blend="weich", bar=0, beat=0,
                     beats=8, section=False)
        parts = split_at_wraps([whole], 3.2)
        assert parts[0].blend_p0 == pytest.approx(0.0)
        assert parts[-1].blend_p1 == pytest.approx(1.0)
        for a, b in zip(parts, parts[1:]):
            assert b.blend_p0 == pytest.approx(a.blend_p1)


class TestPlannerGuards:
    def test_one_track_is_refused(self):
        with pytest.raises(ValueError):
            plan(tracks=1)

    def test_a_zero_loop_is_refused(self):
        with pytest.raises(ValueError):
            plan(loop=0.0)

    def test_a_too_short_span_is_refused(self):
        with pytest.raises(ValueError):
            plan(span=(10.0, 10.2))

    def test_the_same_seed_gives_the_same_plan(self):
        a, b = plan(seed=99), plan(seed=99)
        assert a.to_json() == b.to_json()

    def test_a_different_seed_gives_a_different_plan(self):
        a, b = plan(seed=1), plan(seed=2)
        assert a.to_json() != b.to_json()

    def test_a_rolled_seed_is_reported_back(self):
        # The client renders with the seed its preview resolved, so an
        # unreported seed would mean previewing one edit and rendering another.
        p = plan(seed=None)
        assert isinstance(p.seed, int)
        assert plan(seed=p.seed).to_json()["slots"] == p.to_json()["slots"]

    def test_round_trips_through_json(self):
        from services.video.layers import LayerPlan
        p = plan()
        assert LayerPlan.from_json(p.to_json()).to_json() == p.to_json()

    def test_every_style_and_transition_plans(self):
        for style in STYLE_BY_KEY:
            for opt in transition_options():
                p = plan(style=style, transition=opt["key"])
                assert p.slots, (style, opt["key"])

    def test_style_options_expose_the_ladder_they_use(self):
        for opt in style_options():
            assert opt["fastest"] in LADDER and opt["slowest"] in LADDER


class TestLayerCommand:
    def _sources(self, n=5, duration=3.2):
        return [LayerSource(path=Path(f"t{i}.mp4"), duration=duration)
                for i in range(n)]

    def _cmd(self, p=None, sources=None, fps=24, **kw):
        p = p or plan()
        return build_layer_command(
            "ffmpeg", p, sources or self._sources(), Path("out.mp4"),
            816, 1440, fps, **kw,
        )

    def test_the_inputs_are_never_looped(self):
        # The planner already wrapped every position into [0, loop), so there
        # is nothing for -stream_loop to do — and letting it repeat each file
        # at its own length is exactly how unequal tracks drift apart.
        assert "-stream_loop" not in self._cmd(plan(loop=3.2))

    def test_every_trim_lands_inside_the_loop(self):
        p = plan(loop=3.2)
        for slot in p.slots:
            assert 0 <= slot.src_in < 3.2 + 1e-6
            assert slot.src_out <= 3.2 + 1e-6
            assert slot.src_out > slot.src_in

    def test_every_link_points_at_a_node_that_exists(self):
        cmd = self._cmd()
        graph = cmd[cmd.index("-filter_complex") + 1]
        produced = set()
        for part in graph.split(";"):
            for token in part.split("[")[1:]:
                produced.add(token.split("]")[0])
        for part in graph.split(";"):
            head = part.split("]")[0].lstrip("[")
            if head and not head.endswith(":v"):
                assert head in produced, f"dangling link to {head}"

    def test_a_track_is_decoded_once_however_often_it_appears(self):
        # One input per track and a split, not one input per appearance —
        # the difference between a fast render and a very slow one.
        p = plan(style="flimmern")
        cmd = self._cmd(p)
        assert cmd.count("-i") == p.tracks
        assert len(p.slots) > p.tracks

    def test_both_halves_of_a_transition_read_the_same_window(self):
        # The invariant, as it reaches ffmpeg.
        p = plan(transition="weich")
        cmd = self._cmd(p)
        graph = cmd[cmd.index("-filter_complex") + 1]
        for i, slot in enumerate(p.slots):
            if not slot.is_blend:
                continue
            trims = [part for part in graph.split(";")
                     if part.endswith(f"[a{i}]") or part.endswith(f"[b{i}]")]
            assert len(trims) == 2
            windows = {t[t.index("trim=start="):t.index(",setpts")] for t in trims}
            assert len(windows) == 1, (i, windows)

    def test_the_blend_expression_is_quoted(self):
        # It contains commas, which ffmpeg would otherwise read as the end of
        # the filter.
        cmd = self._cmd(plan(transition="wisch"))
        graph = cmd[cmd.index("-filter_complex") + 1]
        assert "blend=c0_expr='" in graph
        assert ":c1_expr='" in graph and ":c2_expr='" in graph

    def test_no_transition_is_rendered_with_one_expression_for_all_planes(self):
        # `all_expr` is what made the white flash magenta: it applies the luma
        # maths to the chroma planes, whose neutral is not zero.
        for key in BLENDED:
            cmd = self._cmd(plan(transition=key))
            graph = cmd[cmd.index("-filter_complex") + 1]
            assert "all_expr" not in graph, key

    def test_the_branches_meet_in_ten_bits(self):
        graph = self._cmd()[self._cmd().index("-filter_complex") + 1]
        assert f"format={BLEND_PIX_FMT}" in graph
        assert "format=yuv420p," not in graph

    def test_a_hard_cut_plan_has_no_blend_filter(self):
        cmd = self._cmd(plan(transition="hart"))
        graph = cmd[cmd.index("-filter_complex") + 1]
        assert "blend=" not in graph

    def test_frames_telescope_instead_of_accumulating(self):
        # Each slot is round(end*fps) - round(start*fps), so fifty of them sum
        # to the whole rather than to the whole plus fifty half-frames.
        p = plan()
        fps = 24
        assert plan_duration(p, fps) == pytest.approx(
            round(p.song_end * fps) / fps - round(p.song_start * fps) / fps,
            abs=1.5 / fps,
        )

    def test_the_picture_is_rendered_silent(self):
        # The song is muxed on afterwards by the ordinary soundtrack path,
        # which is what keeps upscale, look and Instagram working on the
        # result.
        assert "-an" in self._cmd()

    def test_the_grade_runs_once_per_track_not_once_per_slot(self):
        p = plan(style="flimmern")
        sources = [LayerSource(path=Path(f"t{i}.mp4"), duration=3.2, grade="eq=contrast=1.1")
                   for i in range(p.tracks)]
        graph = self._cmd(p, sources)[self._cmd(p, sources).index("-filter_complex") + 1]
        assert graph.count("eq=contrast=1.1") == p.tracks

    def test_fewer_than_two_sources_is_refused(self):
        with pytest.raises(ValueError):
            build_layer_command("ffmpeg", plan(), self._sources(1), Path("o.mp4"),
                                816, 1440, 24)

    def test_both_branches_meet_in_the_same_format(self):
        # blend requires it, and anything that treated the two differently
        # would break the alignment.
        cmd = self._cmd(plan(transition="licht"))
        graph = cmd[cmd.index("-filter_complex") + 1]
        for part in graph.split(";"):
            if part.endswith("]") and ("[a" in part or "[b" in part) and "trim=" in part:
                assert "format=yuv420p" in part


class TestFadeIn:
    """The layer cut's sibling of TestFadeIn in test_beat_cut.py — same
    default, same reasoning: an opening fade-from-black is a choice now, not
    something every render does."""

    def _cmd(self, **kw):
        p = plan()
        sources = [LayerSource(path=Path(f"t{i}.mp4"), duration=3.2)
                   for i in range(p.tracks)]
        return build_layer_command("ffmpeg", p, sources, Path("out.mp4"),
                                   816, 1440, 24, **kw)

    def test_default_is_no_opening_fade(self):
        cmd = self._cmd()
        graph = cmd[cmd.index("-filter_complex") + 1]
        assert "fade=t=in:st=0" not in graph

    def test_the_closing_fade_runs_regardless(self):
        for kw in ({}, {"fade_in": True}):
            cmd = self._cmd(**kw)
            graph = cmd[cmd.index("-filter_complex") + 1]
            assert "fade=t=out" in graph

    def test_fade_in_true_adds_the_opening_fade(self):
        cmd = self._cmd(fade_in=True)
        graph = cmd[cmd.index("-filter_complex") + 1]
        assert "fade=t=in:st=0" in graph


class TestChromaIsNotLuma:
    """The magenta bug, and the guard that would have caught it.

    Measured before the fix: two neutral grey sources through `licht` produced
    RGB (255, 139, 255) at the midpoint — a hot magenta where the transition
    was supposed to bloom white. The cause was `all_expr` screening the chroma
    planes, where 512 is neutral rather than zero, so lifting both U and V
    together makes magenta by construction.
    """

    @pytest.mark.parametrize("key", BLENDED)
    @pytest.mark.parametrize("p", (0.0, 0.15, 0.35, 0.5, 0.65, 0.85, 1.0))
    def test_neutral_in_neutral_out(self, key, p):
        # Two grey pictures cannot become a coloured one, at any moment of any
        # transition. This is the whole bug in one assertion.
        e = blend_expr(key, 1.0)
        for x in (0.0, 0.25, 0.5, 0.75, 1.0):
            got = evaluate(e.chroma, a=BLEND_NEUTRAL, b=BLEND_NEUTRAL, p=p, x=x)
            assert abs(got - BLEND_NEUTRAL) < 1.0, (key, p, x, got)

    @pytest.mark.parametrize("key", BLENDED)
    def test_the_transition_starts_on_A_and_ends_on_B(self, key):
        e = blend_expr(key, 1.0)
        for plane in (e.luma, e.chroma):
            for x in (0.0, 0.3, 0.7, 1.0):
                for y in (0.0, 0.5, 1.0):
                    assert abs(evaluate(plane, a=200, b=800, p=0.0, x=x, y=y)
                               - 200) < 1.0, (key, x, y)
                    assert abs(evaluate(plane, a=200, b=800, p=1.0, x=x, y=y)
                               - 800) < 1.0, (key, x, y)

    @pytest.mark.parametrize("key", BLENDED)
    @pytest.mark.parametrize("p", (0.1, 0.3, 0.5, 0.7, 0.9))
    def test_nothing_leaves_the_representable_range(self, key, p):
        # An expression that overshoots does not error — it wraps or clamps in
        # the filter and shows up as a blown or inverted patch.
        e = blend_expr(key, 1.0)
        for plane in (e.luma, e.chroma):
            for a, b in ((0, 0), (0, BLEND_PEAK), (BLEND_PEAK, 0),
                         (BLEND_PEAK, BLEND_PEAK), (BLEND_NEUTRAL, 700)):
                for x in (0.0, 0.5, 1.0):
                    v = evaluate(plane, a=a, b=b, p=p, x=x, y=x)
                    assert -1.0 <= v <= BLEND_PEAK + 1.0, (key, p, a, b, v)

    def test_the_flash_lifts_luma_and_drains_chroma(self):
        # What "blooms through white" has to mean numerically: at the centre
        # the picture is brighter than either source and its colour is gone.
        e = blend_expr("licht", 1.0)
        luma = evaluate(e.luma, a=400, b=400, p=0.5)
        chroma = evaluate(e.chroma, a=700, b=700, p=0.5)
        assert luma > 400 * 1.2
        assert abs(chroma - BLEND_NEUTRAL) < abs(700 - BLEND_NEUTRAL) * 0.25

    def test_the_flash_is_confined_to_the_middle(self):
        # The old curve washed the whole transition. The shoulders have to be
        # close to an ordinary dissolve or it stops being a flash.
        e = blend_expr("licht", 1.0)
        shoulder = evaluate(e.chroma, a=700, b=700, p=0.1)
        assert abs(shoulder - 700) < abs(700 - BLEND_NEUTRAL) * 0.15

    def test_a_dissolve_in_light_does_not_sag_in_the_middle(self):
        # The point of mixing in linear light: the midpoint of a dissolve
        # between a dark and a bright picture carries the light of both, not
        # the average of their code values.
        e = blend_expr("weich", 1.0)
        mid = evaluate(e.luma, a=256, b=768, p=0.5)
        assert mid > (256 + 768) / 2 + 20


class TestMotion:
    """The judder. Measured 2026-09-06: at speed 0.70 the old chain left 18 of
    60 output frames identical to their predecessor."""

    def test_the_default_interpolates_rather_than_repeating_frames(self):
        assert DEFAULT_MOTION == MOTION_BLEND
        assert "framerate" in motion_filter(DEFAULT_MOTION, 30)

    def test_each_mode_asks_for_the_target_rate(self):
        for opt in motion_options():
            assert "30" in motion_filter(opt["key"], 30), opt["key"]

    def test_the_modes_are_actually_different_filters(self):
        made = {motion_filter(k, 30) for k in
                (MOTION_ORIGINAL, MOTION_BLEND, MOTION_FLOW)}
        assert len(made) == 3

    def test_original_is_the_old_nearest_frame_behaviour(self):
        assert motion_filter(MOTION_ORIGINAL, 24) == "fps=24"

    def test_an_unknown_mode_renders_smoothly_rather_than_failing(self):
        # A stale client must not be able to put the judder back, or worse,
        # take the render down.
        assert clamp_motion("nonsense") == DEFAULT_MOTION
        assert clamp_motion(None) == DEFAULT_MOTION

    def test_the_mode_reaches_every_branch(self):
        p = plan(transition="weich")
        cmd = build_layer_command(
            "ffmpeg", p, [LayerSource(Path(f"t{i}.mp4"), 9.5) for i in range(5)],
            Path("out.mp4"), 816, 1440, 30, motion=MOTION_FLOW,
        )
        graph = cmd[cmd.index("-filter_complex") + 1]
        branches = graph.count("tpad=stop_mode=clone")
        assert graph.count("minterpolate") == branches
        assert "fps=30," not in graph

    def test_both_halves_of_a_transition_are_resampled_the_same_way(self):
        # They have to show the same source frame at the same moment; two
        # different resamplers would break exactly that.
        p = plan(transition="weich")
        cmd = build_layer_command(
            "ffmpeg", p, [LayerSource(Path(f"t{i}.mp4"), 9.5) for i in range(5)],
            Path("out.mp4"), 816, 1440, 30, motion=MOTION_BLEND,
        )
        graph = cmd[cmd.index("-filter_complex") + 1]
        for i, slot in enumerate(p.slots):
            if not slot.is_blend:
                continue
            parts = [q for q in graph.split(";")
                     if q.endswith(f"[a{i}]") or q.endswith(f"[b{i}]")]
            assert len(parts) == 2
            rates = {q[q.index("framerate="):q.index(",tpad")] for q in parts}
            assert len(rates) == 1, (i, rates)


class TestFiltergraphSpill:
    """A song-length graph does not fit on a Windows command line (32767
    characters). Measured on a four-minute plan: 78 KB for a plain dissolve,
    96 KB for the flash."""

    def test_a_short_graph_is_left_on_the_command_line(self, tmp_path):
        cmd = ["ffmpeg", "-filter_complex", "[0:v]null[v]", "out.mp4"]
        out, spilled = spill_filtergraph(cmd, tmp_path / "out.mp4")
        assert out == cmd and spilled is None

    def test_a_long_graph_moves_into_a_file(self, tmp_path):
        graph = "[0:v]" + "null," * 4000 + "null[v]"
        cmd = ["ffmpeg", "-filter_complex", graph, "out.mp4"]
        out, spilled = spill_filtergraph(cmd, tmp_path / "out.mp4")
        assert "-filter_complex" not in out
        assert out[out.index("-filter_complex_script") + 1] == str(spilled)
        assert spilled.read_text(encoding="utf-8") == graph

    def test_a_command_without_a_graph_is_untouched(self, tmp_path):
        cmd = ["ffmpeg", "-i", "a.mp4", "out.mp4"]
        assert spill_filtergraph(cmd, tmp_path / "out.mp4") == (cmd, None)

    def test_a_real_song_length_plan_would_have_overflowed(self):
        # The regression this guards: without the spill the render dies on a
        # piece of ordinary length, and only on a piece of ordinary length.
        long_map = beatmap(seconds=240.0)
        p = plan_layers(long_map, tracks=5, loop=9.5, seed=7,
                        style="atmen", transition="licht")
        cmd = build_layer_command(
            "ffmpeg", p, [LayerSource(Path(f"t{i}.mp4"), 9.5) for i in range(5)],
            Path("out.mp4"), 816, 1440, 30,
        )
        graph = cmd[cmd.index("-filter_complex") + 1]
        assert len(graph) > 32767


class TestSpeedIsACurve:
    """The stopping, and what replaced it.

    Measured on the old planner: 37 slots, 11 instant speed changes, the
    largest 0.63x — the picture went from 1.56x to 0.93x between one frame and
    the next. The source position was continuous, its derivative was not, and
    that is what read as the video halting on the beat instead of moving with
    it.
    """

    def _steps(self, p):
        """The biggest instantaneous change in speed where the picture is
        moving continuously.

        Loop seams are excluded on purpose, and they are the only exclusion.
        At a seam the source restarts — the picture jumps from the loop's last
        frame to its first — so a kink in the rate lands inside a
        discontinuity the viewer is already looking at. Everywhere else a step
        is exactly the defect this curve exists to remove, and there it has to
        be zero, not small.
        """
        worst = 0.0
        for a, b in zip(p.slots, p.slots[1:]):
            if abs(b.src_in - a.src_out) > 1e-6:        # a seam
                continue
            worst = max(worst, abs(b.speed_in - a.speed_out))
        return worst

    def test_the_speed_never_jumps_while_the_picture_is_moving(self):
        # The old planner's worst step was 0.63 — 1.56x to 0.93x between one
        # frame and the next.
        for style in ("schweben", "atmen", "puls", "welle", "flimmern"):
            for bpm in (90, 120, 174):
                p = plan(beatmap=beatmap(seconds=120.0, bpm=bpm),
                         style=style, loop=4.3, fps=30)
                assert self._steps(p) < 1e-9, (style, bpm, self._steps(p))

    def test_a_slot_ramps_rather_than_holding_one_rate(self):
        p = plan(style="atmen", fps=30)
        ramping = [s for s in p.slots if abs(s.speed_out - s.speed_in) > 1e-3]
        assert len(ramping) > len(p.slots) * 0.5

    def test_the_accent_brakes_into_a_change_and_pushes_after_it(self):
        # Read off the curve itself. A plan's segments cannot answer this: a
        # stretch is cut into several of them and every piece reports
        # `is_blend`, so their starts are not the cuts.
        from services.video.layers import ACCENT_SECONDS, SpeedCurve
        cut = 10.0
        curve = SpeedCurve(marks=[(0.0, 0.5)], low=0.8, high=1.6,
                           swell=0.0, pulse=0.4, cuts=[cut])
        a = ACCENT_SECONDS
        running = curve.at(cut - a)
        braked = curve.at(cut)
        pushing = curve.at(cut + a)
        settled = curve.at(cut + 2 * a)
        assert braked < running - 0.1, "no brake into the cut"
        assert pushing > braked + 0.2, "no push through the cut"
        assert settled == pytest.approx(running, abs=1e-6)
        # and the slowest moment is the change itself
        assert braked == min(curve.at(cut + k * a / 10) for k in range(-10, 21))

    def test_the_accent_reaches_no_further_than_it_says(self):
        from services.video.layers import ACCENT_SECONDS, SpeedCurve
        curve = SpeedCurve(marks=[(0.0, 0.5)], low=0.8, high=1.6,
                           swell=0.0, pulse=0.4, cuts=[10.0])
        plain = curve.at(0.0)
        assert curve.at(10.0 - ACCENT_SECONDS * 1.2) == pytest.approx(plain)
        assert curve.at(10.0 + ACCENT_SECONDS * 2.2) == pytest.approx(plain)

    @staticmethod
    def _speed_at(p, t):
        for s in p.slots:
            if s.start - 1e-9 <= t <= s.end + 1e-9:
                return s.speed_at(t)
        return None

    def test_the_accent_can_be_switched_off(self):
        flat = plan(pulse=0.0, swell=0.0, fps=30)
        rates = [s.speed_in for s in flat.slots] + [s.speed_out for s in flat.slots]
        assert max(rates) - min(rates) < 1e-6

    def test_a_deeper_pulse_moves_the_picture_more(self):
        soft = plan(pulse=0.1, swell=0.0, fps=30)
        hard = plan(pulse=0.6, swell=0.0, fps=30)

        def spread(p):
            r = [s.speed_in for s in p.slots] + [s.speed_out for s in p.slots]
            return max(r) - min(r)

        assert spread(hard) > spread(soft) * 2

    def test_the_swell_can_be_switched_off_without_slowing_the_piece(self):
        # swell=0 has to mean "one tempo", not "the slow end of the band".
        p = plan(swell=0.0, pulse=0.0, speed_low=0.6, speed_high=1.8, fps=30)
        assert p.slots[0].speed == pytest.approx(1.2, abs=1e-6)

    def test_the_stack_is_still_retimed_as_one(self):
        # The curve is a function of output time alone — nothing in it can
        # depend on which track a slot happens to be showing.
        p = plan(fps=30)
        by_time = {}
        for s in p.slots:
            by_time.setdefault(round(s.start, 6), set()).add(
                (round(s.speed_in, 9), round(s.speed_out, 9)))
        assert all(len(v) == 1 for v in by_time.values())


class TestSourceStaysContinuous:
    """The ramp must not break what the whole feature rests on."""

    def test_the_position_never_runs_backwards(self):
        for style in ("schweben", "atmen", "flimmern"):
            p = plan(style=style, loop=3.2, fps=30)
            for s in p.slots:
                assert s.src_out >= s.src_in - 1e-9, (style, s)
                assert 0.0 <= s.src_in < 3.2 + 1e-6

    def test_no_speed_is_ever_negative(self):
        for style in ("schweben", "atmen", "puls", "flimmern"):
            for seed in range(5):
                p = plan(style=style, seed=seed, loop=3.2, fps=30)
                for s in p.slots:
                    assert s.speed_in > 0 and s.speed_out > 0, (style, seed, s)

    def test_what_a_slot_consumes_matches_its_ramp(self):
        p = plan(style="atmen", fps=30)
        for s in p.slots:
            expected = s.duration * (s.speed_in + s.speed_out) / 2
            assert s.src_out - s.src_in == pytest.approx(expected, abs=1e-6)

    def test_the_timeline_has_no_gaps(self):
        p = plan(style="puls", fps=30)
        for a, b in zip(p.slots, p.slots[1:]):
            assert b.start == pytest.approx(a.end, abs=1e-9)

    def test_solve_ramp_inverts_the_quadratic(self):
        from services.video.layers import solve_ramp
        for v0, v1, T in ((0.5, 1.5, 4.0), (1.6, 0.7, 2.0), (1.0, 1.0, 3.0)):
            for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
                t = T * frac
                travelled = v0 * t + (v1 - v0) * t * t / (2 * T)
                assert solve_ramp(v0, v1, T, travelled) == pytest.approx(t, abs=1e-6)


class TestNoSegmentIsTooShortToRender:
    """A one-frame branch collapses and silently truncates the render.

    Measured 2026-09-06: 200 one-frame branches produced 0.10 s of a 6.67 s
    piece, ffmpeg exiting 0. The old planner emitted 1936 such slots across a
    sweep of styles, tempi and seeds — every one a blend sliver left where
    `split_at_wraps` cut a transition beside a loop boundary.
    """

    def test_no_plan_emits_a_one_frame_segment(self):
        from services.video.cut_render import segment_frames
        from services.video.layers import MIN_SEGMENT_FRAMES
        offenders = 0
        for bpm in (90, 120, 174):
            for style in ("schweben", "atmen", "flimmern"):
                for seed in range(4):
                    p = plan(beatmap=beatmap(seconds=120.0, bpm=bpm),
                             style=style, loop=4.3, seed=seed, fps=30)
                    for s in p.slots:
                        if segment_frames(s.start, s.end, 30) < MIN_SEGMENT_FRAMES:
                            offenders += 1
        assert offenders == 0

    def test_a_sliver_never_merges_across_a_loop_seam(self):
        # The two segments either side of a seam read opposite ends of the
        # loop. Merging them asks one segment to travel backwards, which came
        # out as a negative speed and a reversed trim.
        for seed in range(6):
            p = plan(style="schweben", loop=3.2, seed=seed, fps=30)
            for s in p.slots:
                assert s.src_out >= s.src_in, s
                assert s.speed_in > 0 and s.speed_out > 0, s

    def test_merging_keeps_the_timeline_whole(self):
        p = plan(style="schweben", loop=3.2, fps=30)
        assert p.slots[0].start == pytest.approx(p.song_start)
        assert p.slots[-1].end == pytest.approx(p.song_end)
        for a, b in zip(p.slots, p.slots[1:]):
            assert b.start == pytest.approx(a.end, abs=1e-9)


class TestRampReachesTheRenderer:
    def test_a_ramping_slot_gets_the_quadratic_setpts(self):
        from services.video.layer_render import _retime
        from services.video.layers import Slot
        s = Slot(start=0.0, end=2.0, src_in=0.0, src_out=2.0,
                 speed_in=0.8, speed_out=1.2, track=0, from_track=None,
                 blend="", bar=0, beat=0, beats=4, section=False)
        expr = _retime(s)
        assert "sqrt" in expr and expr.endswith("/TB")
        assert "T-STARTT" in expr

    def test_a_flat_slot_still_gets_the_plain_one(self):
        from services.video.layer_render import _retime
        from services.video.layers import Slot
        s = Slot(start=0.0, end=2.0, src_in=0.0, src_out=2.0,
                 speed_in=1.0, speed_out=1.0, track=0, from_track=None,
                 blend="", bar=0, beat=0, beats=4, section=False)
        assert _retime(s) == "setpts=(PTS-STARTPTS)/1.00000"

    def test_the_graph_carries_a_ramp_for_every_branch(self):
        p = plan(style="atmen", fps=30)
        cmd = build_layer_command(
            "ffmpeg", p, [LayerSource(Path(f"t{i}.mp4"), 9.5) for i in range(5)],
            Path("out.mp4"), 816, 1440, 30,
        )
        graph = cmd[cmd.index("-filter_complex") + 1]
        branches = graph.count("tpad=stop_mode=clone")
        assert graph.count("setpts=") >= branches      # one retime per branch
        assert "sqrt" in graph
