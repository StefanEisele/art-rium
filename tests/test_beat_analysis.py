"""
Unit tests for the beat-map analyser (services/video/beats.py).

No audio files and no ffmpeg: `build_beatmap` takes the three-band envelope
that ffmpeg would have produced, so a synthetic envelope with a known pulse in
it can pin down everything that matters — that the tempo comes back, that the
grid lands on the pulses, that the grid reaches both ends of the track, and
that a track with no pulse is reported as having none rather than being handed
a confident lie.
"""
import math

import pytest

from services.video.beats import (
    ANALYSIS_VERSION,
    FRAME_RATE,
    BeatMap,
    build_beatmap,
    downbeat_phase,
    estimate_period,
    extend_grid,
    find_sections,
    onset_strength,
    parse_envelopes,
    rank_normalize,
    track_beats,
)

FLOOR = -60.0


def pulse_bands(bpm=120.0, seconds=20.0, accent_every=4, quiet_from=None):
    """A three-band envelope with a click on every beat.

    Every fourth beat is louder in the low band, which is what a downbeat is;
    `quiet_from` drops the level from that second on, so a section boundary
    exists to be found.
    """
    n = int(seconds * FRAME_RATE)
    period = 60.0 / bpm * FRAME_RATE
    bands = [[FLOOR] * n for _ in range(3)]
    beat = 0
    while True:
        at = int(round(beat * period))
        if at >= n - 2:
            break
        loud = (beat % accent_every) == 0
        level = 0.0 if loud else -12.0
        if quiet_from is not None and at / FRAME_RATE >= quiet_from:
            level -= 18.0
        bands[0][at] = level if loud else FLOOR + 6.0
        bands[1][at] = level
        bands[2][at] = level - 6.0
        beat += 1
    return bands


# ── Envelope parsing ─────────────────────────────────────────────────────────

def test_parse_envelopes_splits_channels():
    text = (
        "frame:0 pts:0 pts_time:0\n"
        "lavfi.astats.1.RMS_level=-20.5\n"
        "lavfi.astats.2.RMS_level=-30.0\n"
        "lavfi.astats.3.RMS_level=-40.0\n"
        "frame:1 pts:128 pts_time:0.0116\n"
        "lavfi.astats.1.RMS_level=-21.0\n"
        "lavfi.astats.2.RMS_level=-31.0\n"
        "lavfi.astats.3.RMS_level=-41.0\n"
    )
    bands = parse_envelopes(text)
    assert [len(b) for b in bands] == [2, 2, 2]
    assert bands[0] == [-20.5, -21.0]
    assert bands[2] == [-40.0, -41.0]


def test_parse_envelopes_floors_silence():
    """Digital silence prints as -inf; it must not poison the differences."""
    text = (
        "lavfi.astats.1.RMS_level=-inf\n"
        "lavfi.astats.2.RMS_level=nan\n"
        "lavfi.astats.3.RMS_level=-999\n"
    )
    bands = parse_envelopes(text)
    assert bands[0][0] == -120.0
    assert bands[1][0] == -120.0
    assert bands[2][0] == -120.0


def test_parse_envelopes_truncates_to_shortest_band():
    text = (
        "lavfi.astats.1.RMS_level=-10\n"
        "lavfi.astats.2.RMS_level=-10\n"
        "lavfi.astats.3.RMS_level=-10\n"
        "lavfi.astats.1.RMS_level=-11\n"
        "lavfi.astats.2.RMS_level=-11\n"
    )
    bands = parse_envelopes(text)
    assert [len(b) for b in bands] == [1, 1, 1]


# ── Onset strength ───────────────────────────────────────────────────────────

def test_onset_strength_is_positive_only_and_peaks_on_the_clicks():
    bands = pulse_bands(bpm=120.0, seconds=8.0)
    onsets = onset_strength(bands)
    assert all(v >= 0.0 for v in onsets)
    period = 60.0 / 120.0 * FRAME_RATE
    for beat in range(2, 12):
        at = int(round(beat * period))
        assert onsets[at] > 1.0, f"no onset at beat {beat}"


def test_onset_strength_ignores_a_slow_swell():
    """A fade-in is not a series of onsets.

    The running mean catches up with a constant slope within its own window, so
    everything after the first second reads as no event at all — which is the
    property that keeps a swelling pad out of the beat grid. The very start of
    the ramp does register, and should: the music beginning is an onset.
    """
    n = int(10 * FRAME_RATE)
    ramp = [[-60.0 + 50.0 * i / n for i in range(n)] for _ in range(3)]
    onsets = onset_strength(ramp)
    settled = onsets[int(1.5 * FRAME_RATE):]
    assert max(settled) < 1e-6


# ── Tempo ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("bpm", [75.0, 90.0, 120.0, 140.0])
def test_estimate_period_recovers_the_tempo(bpm):
    onsets = onset_strength(pulse_bands(bpm=bpm, seconds=25.0))
    found = 60.0 * FRAME_RATE / estimate_period(onsets, bpm_hint=bpm)
    assert abs(found - bpm) / bpm < 0.05


def test_estimate_period_works_without_a_hint():
    onsets = onset_strength(pulse_bands(bpm=120.0, seconds=25.0))
    found = 60.0 * FRAME_RATE / estimate_period(onsets, bpm_hint=None)
    assert abs(found - 120.0) / 120.0 < 0.05


def test_bpm_hint_picks_the_metrical_level():
    """A click on every beat with every second one accented really does repeat
    at two tempos, and both are defensible answers. The hint says which one was
    meant — that is the whole reason songs.bpm is threaded through here.
    """
    onsets = onset_strength(pulse_bands(bpm=120.0, seconds=30.0, accent_every=2))
    fast = 60.0 * FRAME_RATE / estimate_period(onsets, bpm_hint=120.0)
    slow = 60.0 * FRAME_RATE / estimate_period(onsets, bpm_hint=60.0)
    assert abs(fast - 120.0) / 120.0 < 0.05
    assert abs(slow - 60.0) / 60.0 < 0.05


@pytest.mark.parametrize("hint", [60.0, 95.0, 190.0, None])
def test_a_wrong_hint_still_lands_on_the_real_grid(hint):
    """The bias chooses between metrical levels; it cannot invent one.

    A click train at 140 repeats at 140, 70, 46.7 and so on, and a hint far
    from all of them pulls towards the nearest — which is still a grid the
    music is actually on. What must never happen is a tempo unrelated to the
    pulse, because every cut in the edit would then miss.
    """
    onsets = onset_strength(pulse_bands(bpm=140.0, seconds=30.0, accent_every=1))
    found = 60.0 * FRAME_RATE / estimate_period(onsets, bpm_hint=hint)
    levels = [140.0 / k for k in (1, 2, 3)] + [140.0 * 2]
    assert min(abs(found - lv) / lv for lv in levels) < 0.06, found


def test_estimate_period_survives_a_flat_envelope():
    flat = [0.0] * int(5 * FRAME_RATE)
    assert estimate_period(flat, bpm_hint=None) >= 2


# ── Beat tracking ────────────────────────────────────────────────────────────

def test_track_beats_lands_on_the_pulses():
    onsets = onset_strength(pulse_bands(bpm=120.0, seconds=20.0))
    period = estimate_period(onsets, bpm_hint=120.0)
    beats = track_beats(onsets, period)
    assert len(beats) > 20
    gaps = [b - a for a, b in zip(beats, beats[1:])]
    assert all(abs(g - period) <= 2 for g in gaps), gaps


def test_track_beats_returns_nothing_for_a_degenerate_input():
    assert track_beats([1.0], 8) == []
    assert track_beats([0.0] * 50, 1) == []


def test_extend_grid_reaches_both_ends():
    """The DP stops at the first and last onset it can justify; a cut needs a
    grid over the whole song, intro and decay included."""
    grid = extend_grid([100, 120, 140], 20, 400)
    assert grid[0] < 20
    assert grid[-1] >= 380
    assert all(b - a == 20 for a, b in zip(grid, grid[1:]))


def test_extend_grid_from_nothing_is_a_metronome():
    grid = extend_grid([], 10, 55)
    assert grid == [0, 10, 20, 30, 40, 50]


# ── Metre ────────────────────────────────────────────────────────────────────

def test_downbeat_phase_finds_the_accent():
    """The accented beat is offset by two, so phase 2 is the "1"."""
    n = 400
    onsets = [0.0] * n
    low = [-120.0] * n
    beats = list(range(0, n, 10))
    for i, frame in enumerate(beats):
        accent = (i % 4) == 2
        onsets[frame] = 5.0 if accent else 1.0
        low[frame] = -20.0 if accent else -60.0
    assert downbeat_phase(onsets, beats, low, 4) == 2


def test_downbeat_phase_defaults_to_zero_without_a_metre():
    n = 200
    onsets = [0.0] * n
    low = [-60.0] * n
    beats = list(range(0, n, 10))
    for frame in beats:
        onsets[frame] = 2.0
    assert downbeat_phase(onsets, beats, low, 4) == 0


# ── Ranking ──────────────────────────────────────────────────────────────────

def test_rank_normalize_spans_the_full_range():
    ranks = rank_normalize([5.0, 1.0, 3.0, 9.0])
    assert ranks == pytest.approx([2 / 3, 0.0, 1 / 3, 1.0])


def test_rank_normalize_is_scale_free():
    """A rank cares about order, not units — which is why the planner's pacing
    means the same thing on a loud track and a quiet one."""
    values = [0.001, 0.5, 0.02, 0.9]
    assert rank_normalize(values) == rank_normalize([v * 1000 for v in values])


def test_rank_normalize_handles_short_input():
    assert rank_normalize([]) == []
    assert rank_normalize([7.0]) == [0.5]


# ── Sections ─────────────────────────────────────────────────────────────────

def test_find_sections_marks_the_change():
    loud = [1.0, 0.5, 0.2]
    quiet = [0.05, 0.9, 0.9]
    features = [loud] * 6 + [quiet] * 6
    sections = find_sections(features, kernel=2)
    assert sections[0] == 0
    assert any(abs(s - 6) <= 1 for s in sections), sections


def test_find_sections_on_uniform_music_is_just_the_start():
    features = [[1.0, 0.5, 0.25]] * 12
    assert find_sections(features, kernel=2) == [0]


def test_find_sections_handles_a_track_shorter_than_the_kernel():
    assert find_sections([[1.0, 1.0, 1.0]], kernel=2) == [0]
    assert find_sections([], kernel=2) == []


# ── The whole map ────────────────────────────────────────────────────────────

def test_build_beatmap_end_to_end():
    bands = pulse_bands(bpm=120.0, seconds=24.0, quiet_from=12.0)
    bm = build_beatmap(bands, bpm_hint=120.0, beats_per_bar=4)

    assert abs(bm.bpm - 120.0) / 120.0 < 0.05
    assert bm.version == ANALYSIS_VERSION
    assert bm.beats_per_bar == 4
    assert 22.0 < bm.duration < 25.0
    # The grid covers the whole track: nothing before the first beat but one
    # period, nothing after the last but one.
    period = 60.0 / bm.bpm
    assert bm.beats[0] < period
    assert bm.duration - bm.beats[-1] < period * 1.5
    assert len(bm.beat_strength) == len(bm.beats)
    assert bm.bar_count == len(bm.bar_energy)
    assert bm.sections and bm.sections[0] == 0
    assert bm.confidence > 0.0


def test_build_beatmap_reports_no_pulse_for_a_drone():
    """A featureless wash still gets a grid — a metronome is how anyone cuts an
    ambient track — but it must be labelled as one."""
    n = int(20 * FRAME_RATE)
    drone = [[-25.0 + 0.4 * math.sin(i / 30.0) for i in range(n)] for _ in range(3)]
    bm = build_beatmap(drone, bpm_hint=120.0)
    assert bm.beats, "even a drone gets a grid to cut on"
    assert bm.confidence < 0.35


def test_beatmap_round_trips_through_json():
    bm = build_beatmap(pulse_bands(seconds=12.0), bpm_hint=120.0)
    again = BeatMap.from_json(bm.to_json())
    assert again == bm
    assert again.bar_starts() == bm.bar_starts()


def test_bar_starts_follow_the_downbeat_phase():
    bm = BeatMap(
        duration=8.0, bpm=120.0, beats=[i * 0.5 for i in range(16)],
        beats_per_bar=4, downbeat_phase=2, beat_strength=[0.0] * 16,
        bar_energy=[0.0] * 4, sections=[0], confidence=0.5,
    )
    assert bm.bar_starts() == [2, 6, 10, 14]
    assert bm.bar_count == 4
