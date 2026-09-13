"""
The Schichtenschnitt following a recorded performance.

A generated song plans exactly as it always did — that is pinned first, because
the played-music branches live inside the same planner. Then what a recording
adds: changes pulled onto struck accents, a breath forcing a change, touch
shaping the transitions, dynamics driving the speed, and accents between
changes lighting the picture up in the render.
"""
import math
from pathlib import Path

import pytest

from services.video.beats import BeatMap
from services.video.layer_render import (
    GLOW_ATTACK_SECONDS,
    GLOW_DECAY_SECONDS,
    GLOW_DESATURATE,
    GLOW_LIFT,
    LayerSource,
    build_layer_command,
    glow_commands,
    glow_level,
    glow_windows,
)
from services.video.layers import (
    GLOW_CUT_CLEARANCE,
    GLOW_MIN_GAP,
    LayerPlan,
    plan_glows,
    plan_layers,
    played_transition,
    touch_factor,
)

BPM = 90.0
STEP = 60.0 / BPM


def song_beatmap(seconds=40.0):
    n = int(seconds / STEP)
    return BeatMap(
        duration=seconds, bpm=BPM, beats=[i * STEP for i in range(n)],
        beats_per_bar=4, downbeat_phase=0, beat_strength=[0.6] * n,
        bar_energy=[0.3, 0.8] * (n // 8) + [0.5] * (n // 4 - 2 * (n // 8)),
        sections=[0], confidence=0.9,
    )


def played_beatmap(seconds=40.0, dynamics=None, accents=None, phrases=None):
    """A performance whose accents sit 0.18 s after every other beat — off the
    grid, but inside the snap window (0.45 beat = 0.30 s at 90 BPM)."""
    base = song_beatmap(seconds)
    rate = 8.0
    if dynamics is None:
        dynamics = [0.5 + 0.45 * math.sin(2 * math.pi * i / rate / 12.0)
                    for i in range(int(seconds * rate))]
    if accents is None:
        accents = [round(b + 0.18, 4) for b in base.beats[::2] if b + 0.18 < seconds]
    return BeatMap(
        **{**base.to_json(), "profile": "piano", "dynamics": dynamics,
           "dynamics_rate": rate, "accents": accents,
           "accent_strength": [0.9 if i % 3 == 0 else 0.55 for i in range(len(accents))],
           "phrases": phrases or []},
    )


def plan(bm, **kw):
    kw.setdefault("tracks", 4)
    kw.setdefault("loop", 3.0)
    kw.setdefault("seed", 11)
    return plan_layers(bm, **kw)


def changes(p: LayerPlan) -> list[float]:
    """Output-timeline moments where the material changes."""
    out = []
    prev = None
    for s in p.slots:
        if s.is_blend and (prev is None or prev.track != s.track or not prev.is_blend):
            out.append(s.start)
        elif not s.is_blend and prev is not None and prev.track != s.track:
            out.append(s.start)
        prev = s
    return out


# ── a song is untouched ──────────────────────────────────────────────────────

def test_a_song_ignores_the_played_dials():
    bm = song_beatmap()
    a = plan(bm, transition="mix")
    b = plan(bm, transition="mix", touch=1.0, glow=1.0)
    assert a.slots == b.slots
    assert a.profile == "song" and a.glows == []


# ── a performance ────────────────────────────────────────────────────────────

def test_changes_land_on_struck_accents():
    bm = played_beatmap()
    p = plan(bm, transition="hart", style="puls")
    moments = changes(p)
    assert len(moments) >= 8
    on_accent = [c for c in moments if min(abs(c - a) for a in bm.accents) < 1e-3]
    assert len(on_accent) / len(moments) > 0.8, (moments, bm.accents)


def test_a_breath_forces_a_change_and_marks_a_section():
    bm = played_beatmap(phrases=[13.37])
    p = plan(bm, transition="weich", style="schweben")     # long holds, no accident
    assert any(abs(c - 13.37) < 1e-3 for c in changes(p)), changes(p)
    opening = [s for s in p.slots if abs(s.start - 13.37) < 1e-3]
    assert opening and opening[0].section


def test_touch_draws_soft_playing_out_and_pulls_struck_accents_in():
    assert touch_factor(0.1, 0.0, phrase=False, touch=1.0) > 1.5
    assert touch_factor(1.0, 1.0, phrase=False, touch=1.0) == pytest.approx(0.25)
    assert touch_factor(0.9, 0.2, phrase=False, touch=0.0) == 1.0
    assert touch_factor(0.2, 0.2, phrase=True, touch=0.6) == pytest.approx(1.6)


def test_the_music_picks_the_blend():
    assert played_transition(0.8, 0.9) == "licht"
    assert played_transition(0.8, 0.2) == "wisch"
    assert played_transition(0.2, 0.9) == "aufloesen"
    assert played_transition(0.45, 0.3) == "weich"
    assert played_transition(0.2, 0.0, phrase=True) == "aufloesen"


def test_spiel_only_ever_renders_real_blends():
    for bm in (played_beatmap(), song_beatmap()):
        p = plan(bm, transition="spiel")
        assert {s.blend for s in p.slots} <= {"", "weich", "licht", "wisch", "aufloesen"}
        assert any(s.is_blend for s in p.slots)


def test_the_speed_follows_the_dynamics():
    rising = [i / 319 for i in range(320)]                 # 0 → 1 over 40 s
    p = plan(played_beatmap(dynamics=rising), pulse=0.0, swell=1.0)
    quarter = p.song_end / 4
    early = [s.speed for s in p.slots if s.end <= quarter]
    late = [s.speed for s in p.slots if s.start >= 3 * quarter]
    assert sum(early) / len(early) < sum(late) / len(late) - 0.4


def test_glows_stay_clear_of_changes_and_of_each_other():
    bm = played_beatmap()
    p = plan(bm, glow=1.0, transition="weich")
    assert p.glows, "a performance with strong accents produced no glow"
    moments = changes(p)
    times = [t + p.song_start for t, _ in p.glows]
    for t in times:
        assert min(abs(t - c) for c in moments) >= GLOW_CUT_CLEARANCE - 1e-6
    assert all(b - a >= GLOW_MIN_GAP - 1e-6 for a, b in zip(times, times[1:]))
    assert all(0 < amount <= 1 for _, amount in p.glows)
    assert plan(bm, glow=0.0).glows == []


def test_glows_skip_weak_accents():
    got = plan_glows([5.0, 6.0], [0.2, 0.9], cuts=[], start=0.0, end=40.0, amount=1.0)
    assert got == [[6.0, 0.9]]


def test_a_played_plan_survives_json():
    p = plan(played_beatmap(), glow=0.8)
    back = LayerPlan.from_json(p.to_json())
    assert back.glows == p.glows and back.profile == "piano" and back.slots == p.slots


# ── rendering the glow ───────────────────────────────────────────────────────

def test_the_envelope_rises_with_the_attack_and_decays_with_the_note():
    glows = [[2.0, 0.8]]
    assert glow_level(glows, 2.0 - GLOW_ATTACK_SECONDS - 0.01) == 0.0
    assert glow_level(glows, 2.0 - GLOW_ATTACK_SECONDS / 2) == pytest.approx(0.4)
    assert glow_level(glows, 2.0) == pytest.approx(0.8)
    assert glow_level(glows, 2.0 + GLOW_DECAY_SECONDS) == pytest.approx(0.8 / math.e)


def test_a_decay_still_running_is_carried_into_the_next_glow():
    glows = [[1.0, 0.6], [1.5, 0.5]]
    t = 1.6
    expected = 0.6 * math.exp(-0.6 / GLOW_DECAY_SECONDS) + 0.5 * math.exp(-0.1 / GLOW_DECAY_SECONDS)
    assert glow_level(glows, t) == pytest.approx(expected)
    assert glow_level([[1.0, 0.9], [1.1, 0.9]], 1.1) == 1.0          # capped


def test_windows_merge_when_glows_overlap():
    w = glow_windows([[1.0, 1], [1.5, 1], [5.0, 1]], total=10.0)
    assert len(w) == 2 and w[0][0] < 1.0 < 1.5 < w[0][1] and w[1][0] < 5.0


def _parse(line: str) -> tuple[float, dict[str, str]]:
    stamp, rest = line.split(" ", 1)
    parts = {}
    for cmd in rest.rstrip(";").split(", "):
        _, plane, arg = cmd.split(" ", 2)
        parts[plane] = arg.strip("'")
    return float(stamp), parts


def _lut(expr: str, val: float, maxval: float = 1023.0) -> float:
    return eval(expr, {"__builtins__": {}}, {"val": val, "maxval": maxval})   # noqa: S307


def test_the_command_script_follows_the_envelope_frame_by_frame():
    fps, glows = 24, [[2.0, 0.8]]
    lines = glow_commands(glows, total=10.0, fps=fps).strip().splitlines()
    stamps = [_parse(x)[0] for x in lines]
    # One command per frame, each stamped just ahead of its frame.
    frames = [round(s * fps + 0.05) for s in stamps]
    assert frames == list(range(frames[0], frames[-1] + 1))
    assert all(0 < k / fps - s < 1e-3 for s, k in zip(stamps, frames) if k)
    for stamp, k in zip(stamps, frames):
        _, planes = _parse(lines[frames.index(k)])
        level = glow_level(glows, k / fps) if k != frames[-1] else 0.0
        y = _lut(planes["y"], 600.0)
        assert y == pytest.approx(600 + (1023 - 600) * 600 / 1023 * GLOW_LIFT * level, abs=0.05)
        assert _lut(planes["u"], 700.0) == pytest.approx(700 + (512 - 700) * GLOW_DESATURATE * level,
                                                         abs=0.05)
    # The window ends on the identity — nothing after it is touched.
    assert _parse(lines[-1])[1] == {"y": "val", "u": "val", "v": "val"}


def test_shadows_barely_move_and_highlights_bloom():
    _, planes = _parse(glow_commands([[1.0, 1.0]], total=5.0, fps=24).splitlines()[1])   # frame 24 = the accent itself
    # The lift peaks in the midtones and falls to nothing at black.
    dark, bright = _lut(planes["y"], 50.0) - 50.0, _lut(planes["y"], 700.0) - 700.0
    assert bright > 3 * dark > 0


def _graph(cmd: list[str]) -> str:
    return cmd[cmd.index("-filter_complex") + 1]


def test_a_glowing_plan_renders_through_one_lut_on_the_stack():
    p = plan(played_beatmap(), glow=1.0)
    sources = [LayerSource(path=Path(f"t{i}.mp4"), duration=3.0) for i in range(4)]
    graph = _graph(build_layer_command("ffmpeg", p, sources, Path("out.mp4"), 816, 1440, 24))
    assert "[vc]sendcmd=f=out.glow.txt,lutyuv@glow=y=val:u=val:v=val[vg]" in graph
    assert "[vg]fade=" in graph
    assert "nullsrc" not in graph and "geq" not in graph and "all_expr" not in graph


def test_a_plan_without_glows_renders_as_before():
    p = plan(song_beatmap())
    sources = [LayerSource(path=Path(f"t{i}.mp4"), duration=3.0) for i in range(4)]
    graph = _graph(build_layer_command("ffmpeg", p, sources, Path("out.mp4"), 816, 1440, 24))
    assert "sendcmd" not in graph and "[vc]fade=" in graph
