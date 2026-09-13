"""
Reading a played piano piece (services/video/piano.py) and taking it in as a
song (services/improv/piano_song.py).

No audio and no ffmpeg, as in test_beat_analysis: the analyser takes the
three-band envelope ffmpeg would have produced, so a synthetic performance with
a known shape — a crescendo, struck chords, a breath before the second phrase —
pins down everything the layer cut is going to follow.
"""
import asyncio
import json
import math

import pytest

from services.improv.piano_song import (
    MAX_GAIN_DB,
    PEAK_CEILING_DBTP,
    TARGET_LUFS,
    extract_command,
    level_gain,
    parse_loudnorm_json,
    song_id_for_session,
)
from services.video import beats, piano
from services.video.beats import FRAME_RATE, BeatMap

FLOOR_DB = -90.0


def performance(notes, seconds=16.0, gain_db=0.0):
    """Three bands of a pianist striking `notes` = [(time, peak_db)].

    Each note is an attack that decays: the body in the low and mid bands over
    half a second, the hammer transient in the high band over 50 ms. `gain_db`
    shifts the whole take, which is all the import ever does to a recording.
    """
    n = int(seconds * FRAME_RATE)
    # The room noise moves with the gain as well — the import scales the whole
    # file, noise included.
    floor = 10 ** ((FLOOR_DB + gain_db) / 20)
    bands = [[0.0] * n for _ in range(3)]
    for i in range(n):
        t = i / FRAME_RATE
        low = mid = high = floor
        for t0, peak in notes:
            if t < t0:
                continue
            a = 10 ** ((peak + gain_db) / 20)
            body = a * math.exp(-(t - t0) / 0.5)
            low += 0.5 * body
            mid += body
            high += 0.3 * a * math.exp(-(t - t0) / 0.05)
        bands[0][i] = 20 * math.log10(low)
        bands[1][i] = 20 * math.log10(mid)
        bands[2][i] = 20 * math.log10(high)
    return bands


# Phrase one: a crescendo from -40 to -14 dB, a note every 0.6 s. A breath from
# 6.6 s to 8.0 s. Phrase two at -22 dB, with one hard chord at 11.0 s.
PHRASE_ONE = [(0.5 + 0.6 * k, -40 + 26 * k / 10) for k in range(11)]
PHRASE_TWO = [(8.0 + 0.6 * k, -22.0) for k in range(12) if abs(8.0 + 0.6 * k - 11.0) > 0.2]
CHORD = (11.0, -6.0)
NOTES = PHRASE_ONE + PHRASE_TWO + [CHORD]


@pytest.fixture(scope="module")
def played():
    return piano.build_piano_beatmap(performance(NOTES))


def test_the_dynamics_follow_the_crescendo(played):
    at = played.dynamics_at
    assert all(0.0 <= v <= 1.0 for v in played.dynamics)
    assert at(1.0) < at(3.5) < at(6.2)
    assert at(6.2) - at(1.0) > 0.4


def test_the_breath_is_a_dip(played):
    assert played.dynamics_at(7.7) < played.dynamics_at(6.2) - 0.2


def test_accents_land_on_struck_notes(played):
    times = [t for t, _ in NOTES]
    assert len(played.accents) >= 10
    for a in played.accents:
        assert min(abs(a - t) for t in times) < 0.05, a


def test_the_hard_chord_is_the_strongest_accent(played):
    strongest = played.accents[played.accent_strength.index(max(played.accent_strength))]
    assert abs(strongest - CHORD[0]) < 0.05


def test_a_breath_starts_a_phrase(played):
    assert played.phrases, "the breath before 8.0 s was not found"
    assert abs(played.phrases[0] - 8.0) < 0.1
    assert not any(p < 7.0 for p in played.phrases)


def test_a_held_gap_without_a_dip_is_not_a_phrase():
    # Nothing struck for a second, but the level never falls: a held chord.
    flat = [0.8] * int(20 * piano.DYNAMICS_RATE)
    assert piano.find_phrases([1.0, 2.2, 3.4, 4.6], flat) == []


def test_a_quiet_take_reads_like_a_loud_one():
    """One gain on the whole file must change nothing the cut follows —
    that is why the import levels with a gain and never with a compressor."""
    loud = piano.build_piano_beatmap(performance(NOTES))
    quiet = piano.build_piano_beatmap(performance(NOTES, gain_db=-20.0))
    assert quiet.accents == loud.accents
    assert quiet.phrases == loud.phrases
    assert max(abs(a - b) for a, b in zip(quiet.dynamics, loud.dynamics)) < 0.02


def test_a_take_without_dynamics_sits_in_the_middle():
    assert set(piano.dynamics_curve([-30.0] * 500)) == {0.5}


def test_the_played_map_says_so_and_survives_json(played):
    assert played.is_played and played.profile == piano.PROFILE
    assert BeatMap.from_json(json.loads(json.dumps(played.to_json()))) == played


def test_a_generated_song_is_not_played():
    song = beats.build_beatmap(performance(NOTES))
    assert not song.is_played
    assert song.profile == "song" and song.dynamics == [] and song.accents == []


def test_the_song_cache_never_reads_a_piano_sidecar(tmp_path, played):
    cache = tmp_path / "x_beats.json"
    cache.write_text(json.dumps(played.to_json()), encoding="utf-8")
    # The piano loader takes it without touching ffmpeg...
    got = asyncio.run(piano.load_or_analyze_piano(
        tmp_path / "missing.flac", cache, ffmpeg_path="no-such-ffmpeg"))
    assert got == played
    # ...the song loader refuses it and goes to analyse, which fails here.
    with pytest.raises((RuntimeError, OSError)):
        asyncio.run(beats.load_or_analyze(
            tmp_path / "missing.flac", cache, ffmpeg_path="no-such-ffmpeg"))


# ── taking a recording in ────────────────────────────────────────────────────

def test_a_quiet_take_is_raised_to_target_but_never_clipped():
    # -30 LUFS wants +16 dB, but a -12 dBTP peak only allows +11.
    assert level_gain(-30.0, -12.0) == pytest.approx(PEAK_CEILING_DBTP + 12.0)
    assert level_gain(-30.0, -25.0) == pytest.approx(TARGET_LUFS + 30.0)


def test_a_loud_take_is_turned_down():
    assert level_gain(-8.0, -0.5) == pytest.approx(TARGET_LUFS + 8.0)


def test_the_gain_is_bounded_and_silence_refused():
    assert level_gain(-70.0, -60.0) == MAX_GAIN_DB
    with pytest.raises(ValueError):
        level_gain(float("-inf"), float("-inf"))


def test_loudnorm_report_is_read_from_the_end_of_stderr():
    stderr = ('[Parsed_loudnorm_0 @ 0x1] \n{\n\t"input_i" : "-27.43",\n'
              '\t"input_tp" : "-9.12",\n\t"input_lra" : "11.20"\n}\n')
    assert parse_loudnorm_json(stderr) == (-27.43, -9.12)


def test_extraction_is_one_gain_and_lossless(tmp_path):
    cmd = extract_command("ffmpeg", tmp_path / "a.mov", tmp_path / "a.flac", 6.5)
    graph = cmd[cmd.index("-af") + 1]
    assert graph == "volume=6.50dB"                 # no loudnorm, no compressor
    assert cmd[cmd.index("-c:a") + 1] == "flac"
    assert "-vn" in cmd


def test_a_session_maps_to_one_song_id():
    import uuid
    s = uuid.uuid4()
    assert song_id_for_session(s) == song_id_for_session(s)
    assert song_id_for_session(s) != song_id_for_session(uuid.uuid4())
