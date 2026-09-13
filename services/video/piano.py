"""
Reading a played piano piece, so a layer cut can follow the performance
instead of a metronome laid over it.

services/video/beats.py was built for generated songs, and it asks them the
right questions: where is the grid, how loud is each bar, where does the
arrangement change. A recorded performance answers those badly and has three
better answers of its own:

  **Tempo moves.** A pianist leans into a phrase and lets the end of it go.
  The song tracker's dynamic program is tuned to resist exactly that
  (DP_ALPHA = 100 — a strict metronome is what a drum track wants).

  **Loudness moves inside the bar.** A crescendo over two beats is invisible
  to a per-bar rank, and it is the most readable thing the picture could
  follow. So the dynamics are kept as a continuous curve.

  **The music has accents and breaths, not just a grid.** A struck chord is a
  moment worth cutting on even when it falls between beats, and the silence
  before a new phrase is where a change of material feels inevitable.

Everything still comes out as a `BeatMap`, so both cutters, the timeline and
the cache keep working. The extra fields — dynamics, accents, phrases — are
what services/video/layers.py reads when `BeatMap.is_played`.

**Measured 2026-09-13 on the improv library** (six distinct iPhone + Scarlett
recordings, 6–21 s): with the song tracker's alpha of 100 the grid was nearly a
metronome — beat-gap variation 4–9% — and landed within ±46 ms of 38–41% of the
strong attacks. At alpha 15 it reached 38–56% at 5–14% variation. At 5 it
started to wander (up to 25%) for little more, so 15 it is.

The loudness side needs no tuning of that kind: it is normalised between
percentiles of the piece itself, so a quiet take and a loud one both span the
full range, and one gain applied to the whole file (which is all the import
does) changes nothing here at all.
"""
from __future__ import annotations

import asyncio
import bisect
import json
import logging
import math
from pathlib import Path

from services.video import beats
from services.video.beats import FRAME_RATE, BeatMap

logger = logging.getLogger(__name__)

PROFILE = "piano"

# Bump when anything below changes the numbers. Separate from the song
# analyser's version: the cache check reads both the profile and this.
PIANO_ANALYSIS_VERSION = 1

# See the module docstring for the measurement.
DP_ALPHA_PIANO = 15.0

# ── Dynamics ─────────────────────────────────────────────────────────────────
# Eight samples a second is finer than any dynamic a listener follows, and
# coarse enough that a four-minute piece is two thousand numbers.
DYNAMICS_RATE = 8.0
# Centred, so a crescendo is not reported late. Long enough to ride over the
# decay of single notes, short enough to keep a swell inside a phrase.
DYNAMICS_SMOOTH_SECONDS = 0.6
# Anything further than this below the loudest moment is room noise, not
# pianissimo, and must not set the floor of the range.
DYNAMICS_RANGE_DB = 60.0
DYNAMICS_LOW_PCT, DYNAMICS_HIGH_PCT = 0.10, 0.95
# A take that moves less than this has no dynamics worth following; it reads
# as a flat middle rather than as noise stretched to full scale.
DYNAMICS_MIN_SPAN_DB = 3.0

# ── Accents ──────────────────────────────────────────────────────────────────
# Onset strength is already in units of its own spread (beats.onset_strength),
# so this is "two deviations above the piece's typical attack".
ACCENT_THRESHOLD = 2.0
# A peak has to be the largest within this many seconds either side — one
# chord is one accent, not three frames of one.
ACCENT_HALF_WINDOW = 0.06
# Two accents closer than this are one gesture; the stronger one stays.
ACCENT_MIN_GAP = 0.12
# A click in a near-silent passage (pedal, bench, breath) is not an accent.
ACCENT_LOUDNESS_FLOOR = 0.2

# ── Phrases ──────────────────────────────────────────────────────────────────
# A phrase begins at an accent that follows a gap with nothing struck in it
# AND a real dip in the dynamics — a held chord has the gap but not the dip.
BREATH_SECONDS = 0.45
BREATH_DROP = 0.15
# Closer than this, two breaths belong to one phrase.
PHRASE_MIN_SECONDS = 2.0


# ── Loudness ─────────────────────────────────────────────────────────────────

def loudness_db(bands: list[list[float]]) -> list[float]:
    """Per-frame loudness over the three bands, as a power sum in dB."""
    n = len(bands[0]) if bands else 0
    out = [0.0] * n
    for i in range(n):
        power = 0.0
        for band in bands:
            power += 10.0 ** (band[i] / 10.0)
        out[i] = 10.0 * math.log10(power) if power > 0 else -120.0
    return out


def _centred_mean(values: list[float], half: int) -> list[float]:
    n = len(values)
    prefix = [0.0]
    for v in values:
        prefix.append(prefix[-1] + v)
    out = [0.0] * n
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        out[i] = (prefix[hi] - prefix[lo]) / (hi - lo)
    return out


def _percentile(sorted_values: list[float], q: float) -> float:
    return sorted_values[int(round(q * (len(sorted_values) - 1)))]


def dynamics_curve(level_db: list[float]) -> list[float]:
    """The performance's loudness, 0..1, sampled at DYNAMICS_RATE.

    Normalised between percentiles of the *sounding* part of the piece, not
    ranked: a pianissimo passage stays near the bottom even when it is long,
    which is exactly what a rank would destroy.
    """
    n = len(level_db)
    if n == 0:
        return []
    half = max(1, int(FRAME_RATE * DYNAMICS_SMOOTH_SECONDS / 2))
    smooth = _centred_mean(level_db, half)
    step = FRAME_RATE / DYNAMICS_RATE
    count = max(1, int(n / step))

    peak = max(smooth)
    sounding = sorted(v for v in smooth if v > peak - DYNAMICS_RANGE_DB)
    if not sounding:
        return [0.0] * count
    lo = _percentile(sounding, DYNAMICS_LOW_PCT)
    hi = _percentile(sounding, DYNAMICS_HIGH_PCT)
    if hi - lo < DYNAMICS_MIN_SPAN_DB:
        return [0.5] * count

    out = []
    for k in range(count):
        v = (smooth[min(n - 1, int(round(k * step)))] - lo) / (hi - lo)
        out.append(round(max(0.0, min(1.0, v)), 4))
    return out


def _sample(curve: list[float], t: float) -> float:
    if not curve:
        return 0.5
    x = max(0.0, t * DYNAMICS_RATE)
    i = int(x)
    if i >= len(curve) - 1:
        return curve[-1]
    return curve[i] + (curve[i + 1] - curve[i]) * (x - i)


# ── Accents ──────────────────────────────────────────────────────────────────

def find_accents(
    onsets: list[float], dynamics: list[float],
) -> tuple[list[float], list[float]]:
    """(times, strengths) of the attacks that stand out, in time order.

    Strength is the attack's height weighted by how loud the playing is
    there, then scaled against the piece's own strong accents, so the
    loudest chord of a quiet piece still counts as a strong one.
    """
    n = len(onsets)
    half = max(1, int(round(FRAME_RATE * ACCENT_HALF_WINDOW)))
    candidates: list[tuple[float, float]] = []
    for i in range(n):
        v = onsets[i]
        if v < ACCENT_THRESHOLD:
            continue
        lo, hi = max(0, i - half), min(n, i + half + 1)
        window = onsets[lo:hi]
        if v < max(window):
            continue
        if any(onsets[j] >= v for j in range(lo, i)):      # a plateau counts once
            continue
        t = i / FRAME_RATE
        loud = _level_after(dynamics, t)
        if loud < ACCENT_LOUDNESS_FLOOR:
            continue
        # Loudness carries the strength; the attack's height only modulates it.
        # Onset strength is a rise in dB, and the first soft note after a
        # silence rises further than any chord in the piece — weighted the
        # other way round, that note came out as the strongest accent.
        candidates.append((t, loud * (0.6 + 0.4 * min(1.0, v / (2 * ACCENT_THRESHOLD)))))

    if not candidates:
        return [], []

    kept: list[tuple[float, float]] = []
    taken: list[float] = []
    for t, s in sorted(candidates, key=lambda c: -c[1]):
        at = bisect.bisect_left(taken, t)
        near = taken[max(0, at - 1):at + 1]
        if any(abs(t - x) < ACCENT_MIN_GAP for x in near):
            continue
        bisect.insort(taken, t)
        kept.append((t, s))
    kept.sort()

    raw = sorted(s for _, s in kept)
    scale = _percentile(raw, 0.95) or 1e-9
    return (
        [round(t, 4) for t, _ in kept],
        [round(min(1.0, s / scale), 4) for _, s in kept],
    )


# ── Phrases ──────────────────────────────────────────────────────────────────

def _level_after(curve: list[float], t: float, window: float = 0.3) -> float:
    """The level an attack at `t` opens onto.

    Not the curve AT the attack: it is smoothed with a centred window, so at
    the first note after a silence half of that window is still the silence,
    and the note reads far quieter than it plays.
    """
    if not curve:
        return 0.5
    i0 = max(0, int(t * DYNAMICS_RATE))
    i1 = min(len(curve), int((t + window) * DYNAMICS_RATE) + 1)
    return max(curve[i0:i1]) if i1 > i0 else _sample(curve, t)


def find_phrases(accents: list[float], dynamics: list[float]) -> list[float]:
    """Seconds at which a phrase begins after a breath.

    A breath is a dip below BOTH sides: below the level the previous accent
    opened and below the one the new accent opens. Measured against the new
    side alone, a loud chord after an ordinary gap reads as a new phrase — the
    dip is only deep relative to the chord.

    The opening of the piece is not listed — nothing changes there, the piece
    merely starts.
    """
    out: list[float] = []
    for prev, t in zip(accents, accents[1:]):
        if t - prev < BREATH_SECONDS:
            continue
        i0 = int(prev * DYNAMICS_RATE) + 1
        i1 = int(t * DYNAMICS_RATE)
        between = dynamics[i0:i1] if i1 > i0 else []
        dip = min(between) if between else _sample(dynamics, (prev + t) / 2)
        sides = min(_level_after(dynamics, prev), _level_after(dynamics, t))
        if sides - dip < BREATH_DROP:
            continue
        if out and t - out[-1] < PHRASE_MIN_SECONDS:
            continue
        out.append(round(t, 3))
    return out


# ── Entry points ─────────────────────────────────────────────────────────────

def build_piano_beatmap(
    bands: list[list[float]], *, beats_per_bar: int = beats.DEFAULT_BEATS_PER_BAR,
) -> BeatMap:
    """The pure half of `analyze_piano`, testable without ffmpeg."""
    n_frames = len(bands[0])
    onsets = beats.onset_strength(bands)
    period = beats.estimate_period(onsets, None)
    tracked = beats.track_beats(onsets, period, alpha=DP_ALPHA_PIANO)
    confidence = beats.beat_confidence(onsets, tracked)
    frames = beats.extend_grid(tracked, period, n_frames)

    phase = beats.downbeat_phase(onsets, frames, bands[0], beats_per_bar)
    features = beats.bar_features(bands, frames, phase, beats_per_bar)
    energies = beats.rank_normalize([sum(f) for f in features])
    strengths = beats.normalize([onsets[f] if f < len(onsets) else 0.0 for f in frames])

    dynamics = dynamics_curve(loudness_db(bands))
    accents, accent_strength = find_accents(onsets, dynamics)
    phrases = find_phrases(accents, dynamics)

    return BeatMap(
        duration=round(n_frames / FRAME_RATE, 3),
        bpm=round(60.0 * FRAME_RATE / period, 2),
        beats=[round(f / FRAME_RATE, 4) for f in frames],
        beats_per_bar=beats_per_bar,
        downbeat_phase=phase,
        beat_strength=[round(v, 4) for v in strengths],
        bar_energy=[round(v, 4) for v in energies],
        sections=beats.find_sections(features),
        confidence=round(confidence, 3),
        version=PIANO_ANALYSIS_VERSION,
        profile=PROFILE,
        dynamics=dynamics,
        dynamics_rate=DYNAMICS_RATE,
        accents=accents,
        accent_strength=accent_strength,
        phrases=phrases,
    )


async def analyze_piano(
    source: Path,
    *,
    beats_per_bar: int = beats.DEFAULT_BEATS_PER_BAR,
    ffmpeg_path: str = "ffmpeg",
) -> BeatMap:
    """Analyse one recording. The arithmetic runs off the event loop: it is
    plain Python over tens of thousands of frames, a few seconds for a long
    piece, and the server has other requests to answer meanwhile."""
    bands = await beats._read_envelopes(source, ffmpeg_path)
    return await asyncio.to_thread(build_piano_beatmap, bands, beats_per_bar=beats_per_bar)


async def load_or_analyze_piano(
    source: Path,
    cache: Path,
    *,
    beats_per_bar: int = beats.DEFAULT_BEATS_PER_BAR,
    ffmpeg_path: str = "ffmpeg",
) -> BeatMap:
    """Cached `analyze_piano`, sharing the song sidecar's path and format."""
    try:
        data = json.loads(cache.read_text(encoding="utf-8"))
        if (data.get("profile") == PROFILE
                and data.get("version") == PIANO_ANALYSIS_VERSION
                and data.get("beats_per_bar") == beats_per_bar):
            return BeatMap.from_json(data)
    except (OSError, ValueError, TypeError):
        pass

    beatmap = await analyze_piano(source, beats_per_bar=beats_per_bar, ffmpeg_path=ffmpeg_path)
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(beatmap.to_json()), encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not cache piano analysis at %s: %s", cache, exc)
    return beatmap
