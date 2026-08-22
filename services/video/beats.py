"""
Beat-map analysis — what a song's rhythm looks like, so a cut can land on it.

No numpy, no librosa. The heavy lifting is ffmpeg's, and the arithmetic left
over is small enough for plain Python:

  1. ffmpeg splits the track into three bands (kick / body / transient), merges
     them into one 3-channel stream and reports the RMS of every 128-sample
     window per channel. That is an 86 Hz, 3-band loudness envelope computed in
     C, and it costs about 0.3 s for a 60-second song.
  2. The positive part of the per-band difference, summed across bands, is the
     onset-strength envelope (Ellis 2007). Slow drift is removed with a
     one-second running mean so a swell does not read as a continuous onset.
  3. Tempo comes from the autocorrelation of that envelope, weighted by a
     log-Gaussian window, and the beat grid from Ellis's dynamic program:
     every beat wants to sit on an onset *and* one tempo period after the last
     one, and the DP finds the sequence that trades those off best.

The one thing this has that a general-purpose beat tracker does not: **the
song's requested BPM**. ACE-Step was told a tempo when the track was generated
and `songs.bpm` still holds it, so it is used to centre the bias window instead
of the usual 120 BPM prior. Measured over the whole library on 2026-08-21 that
moved 26/34 -> 31/34 tracks onto the tempo they were asked for; the three that
stay off are ones ACE did not render at the requested tempo, and two of those
are half or double it, which is a legal metrical level rather than an error.

Beatless material (drones, the "no drums, beatless" tags in this library) has no
grid to find, and the tracker will happily invent a regular one anyway. That is
the right failure: a metronomic grid is exactly how a human cuts an ambient
track. `BeatMap.confidence` says which of the two happened, so the caller can
tell the user rather than pretending.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Bump when anything below changes the numbers, so cached sidecars are dropped
# rather than silently mixed with results from a different algorithm.
ANALYSIS_VERSION = 4

SAMPLE_RATE = 11025
HOP = 128                       # -> 86.13 envelope frames per second
FRAME_RATE = SAMPLE_RATE / HOP

# Kick body / mid body / transient. Three is enough to separate a bass drum
# from a hi-hat, which is all the onset function needs; more bands cost another
# filter chain and change nothing measurable.
BANDS = (
    "lowpass=f=200",
    "bandpass=f=1200:width_type=o:w=2.5",
    "highpass=f=4000",
)

# Tempo search range, in BPM. The lower bound is where a "beat" stops being one
# and becomes a bar; the upper is above anything worth cutting on directly.
BPM_MIN, BPM_MAX = 55.0, 200.0
BPM_FALLBACK = 120.0

# Width of the log-Gaussian bias on the autocorrelation, in octaves. Tight with
# a hint (trust the requested tempo, but let a clearly different real one win),
# wide without.
BIAS_SIGMA_HINTED = 0.45
BIAS_SIGMA_FREE = 1.00

# Ellis's transition weight, against an onset envelope normalised to unit
# spread. Higher = a stricter metronome, lower = follows the onsets even where
# they wander.
DP_ALPHA = 100.0

DEFAULT_BEATS_PER_BAR = 4
DRIFT_WINDOW_SECONDS = 1.0


@dataclass(frozen=True)
class BeatMap:
    """Everything the cut planner needs to know about a piece of music."""
    duration: float
    bpm: float
    beats: list[float]              # beat onsets, seconds from the start
    beats_per_bar: int
    downbeat_phase: int             # beats[phase::beats_per_bar] are the "1"s
    beat_strength: list[float]      # onset strength at each beat, 0..1
    bar_energy: list[float]         # loudness RANK per bar, 0 (quietest) .. 1
    sections: list[int]             # bar indices where the music changes
    confidence: float               # 0 = no audible pulse, 1 = unmistakable
    version: int = ANALYSIS_VERSION

    def bar_starts(self) -> list[int]:
        """Indices into `beats` of every downbeat."""
        return list(range(self.downbeat_phase, len(self.beats), self.beats_per_bar))

    @property
    def bar_count(self) -> int:
        return len(self.bar_starts())

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict) -> "BeatMap":
        return cls(**data)


# ── ffmpeg: the three-band loudness envelope ─────────────────────────────────

def envelope_command(ffmpeg_path: str, source: Path) -> list[str]:
    """ffmpeg argv that prints one RMS line per band per 128-sample window.

    Split into three filtered copies and merged back into a 3-channel stream so
    `astats` reports all three per window in one pass — one decode instead of
    three, and the windows are guaranteed to line up.
    """
    chain = (
        f"[0:a]aresample={SAMPLE_RATE},"
        "aformat=sample_fmts=fltp:channel_layouts=mono,asplit=3[a][b][c];"
    )
    for label, band in zip("abc", BANDS):
        chain += f"[{label}]{band}[x{label}];"
    chain += (
        f"[xa][xb][xc]amerge=inputs=3,asetnsamples=n={HOP}:p=0,"
        "astats=metadata=1:reset=1:measure_perchannel=RMS_level:measure_overall=none,"
        "ametadata=print:file=-"
    )
    return [
        ffmpeg_path, "-v", "error", "-i", str(source),
        "-filter_complex", chain, "-f", "null", "-",
    ]


def parse_envelopes(stdout: str) -> list[list[float]]:
    """Read `ametadata=print` output into one dB series per band.

    Digital silence prints as `-inf` or `nan`; both become the floor rather
    than poisoning the differences downstream.
    """
    bands: list[list[float]] = [[], [], []]
    for line in stdout.splitlines():
        key, sep, raw = line.partition("=")
        if not sep or ".RMS_level" not in key:
            continue
        parts = key.split(".")
        try:
            channel = int(parts[2]) - 1
        except (IndexError, ValueError):
            continue
        if not 0 <= channel < 3:
            continue
        try:
            value = float(raw)
        except ValueError:
            value = -120.0
        if value != value or value < -120.0:      # NaN, or below the floor
            value = -120.0
        bands[channel].append(value)
    n = min((len(b) for b in bands), default=0)
    return [b[:n] for b in bands]


async def _read_envelopes(source: Path, ffmpeg_path: str) -> list[list[float]]:
    proc = await asyncio.create_subprocess_exec(
        *envelope_command(ffmpeg_path, source),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        tail = stderr.decode(errors="replace")[-600:]
        raise RuntimeError(f"ffmpeg envelope pass failed (rc={proc.returncode}): {tail}")
    bands = parse_envelopes(stdout.decode(errors="replace"))
    if not bands or len(bands[0]) < 8:
        raise RuntimeError("Audio too short to analyse")
    return bands


# ── Onset strength ───────────────────────────────────────────────────────────

def onset_strength(bands: list[list[float]]) -> list[float]:
    """Half-wave-rectified spectral flux over the three bands, drift removed.

    Working in dB rather than linear amplitude is deliberate: a hi-hat carries
    a fraction of a kick's energy but the same relative jump, and in dB the two
    contribute comparably.
    """
    n = len(bands[0])
    flux = [0.0] * n
    for band in bands:
        for i in range(1, n):
            rise = band[i] - band[i - 1]
            if rise > 0.0:
                flux[i] += rise

    # A one-second running mean is the high-pass: a slow swell raises the mean
    # with itself and contributes nothing, a transient outruns it.
    window = max(2, int(FRAME_RATE * DRIFT_WINDOW_SECONDS))
    out = [0.0] * n
    running = 0.0
    for i in range(n):
        running += flux[i]
        if i >= window:
            running -= flux[i - window]
        above = flux[i] - running / min(i + 1, window)
        out[i] = above if above > 0.0 else 0.0

    # Dividing by the spread is what makes DP_ALPHA a fixed constant rather than
    # a per-track tuning. The guard matters on material that genuinely has no
    # transients — a slow swell leaves `out` at float dust, and scaling dust by
    # its own tiny deviation would manufacture a full-scale onset function out
    # of rounding error.
    spread = statistics.pstdev(out)
    if spread < 1e-9 or max(out) < 1e-9:
        return [0.0] * n
    return [v / spread for v in out]


# ── Tempo ────────────────────────────────────────────────────────────────────

def estimate_period(onsets: list[float], bpm_hint: float | None = None) -> int:
    """Best beat period, in envelope frames.

    Autocorrelation says which lags the music repeats at; the bias window says
    which of them a listener would call *the* beat. Without a hint that is the
    usual pull towards 120 BPM, with one it is the tempo the song was asked for.
    """
    n = len(onsets)
    lag_min = max(2, int(FRAME_RATE * 60.0 / BPM_MAX))
    lag_max = min(n - 2, int(FRAME_RATE * 60.0 / BPM_MIN))
    if lag_max <= lag_min:
        return max(2, int(FRAME_RATE * 60.0 / BPM_FALLBACK))

    hinted = bool(bpm_hint and BPM_MIN <= bpm_hint <= BPM_MAX)
    centre = 60.0 / bpm_hint if hinted else 60.0 / BPM_FALLBACK
    sigma = BIAS_SIGMA_HINTED if hinted else BIAS_SIGMA_FREE

    best_score, best_lag = -1.0, lag_min
    for lag in range(lag_min, lag_max + 1):
        total = 0.0
        for i in range(lag, n):
            total += onsets[i] * onsets[i - lag]
        total /= (n - lag)
        period = lag / FRAME_RATE
        score = total * math.exp(-0.5 * (math.log(period / centre, 2) / sigma) ** 2)
        if score > best_score:
            best_score, best_lag = score, lag
    return best_lag


# ── Beat tracking ────────────────────────────────────────────────────────────

def track_beats(onsets: list[float], period: int, alpha: float = DP_ALPHA) -> list[int]:
    """Ellis's dynamic program: the beat sequence that best satisfies both
    "sit on an onset" and "stay one period after the last beat".

    The transition cost -(log(gap/period))^2 is symmetric in tempo *ratio*, so
    being 10% early costs the same as being 10% late — which is what makes the
    grid resist being dragged off by a single loud off-beat hit.
    """
    n = len(onsets)
    if n < 2 or period < 2:
        return []
    lo = max(1, int(round(period * 0.5)))
    hi = max(lo + 1, int(round(period * 2.0)))

    score = [0.0] * n
    back = [-1] * n
    for i in range(n):
        best, best_j = 0.0, -1
        start = max(0, i - hi)
        for j in range(start, i - lo + 1):
            candidate = score[j] - alpha * (math.log((i - j) / period)) ** 2
            if best_j < 0 or candidate > best:
                best, best_j = candidate, j
        score[i] = onsets[i] + (best if best_j >= 0 else 0.0)
        back[i] = best_j

    end = max(range(n), key=score.__getitem__)
    beats: list[int] = []
    cursor = end
    while cursor >= 0:
        beats.append(cursor)
        cursor = back[cursor]
    beats.reverse()
    return beats


def extend_grid(beats: list[int], period: int, n_frames: int) -> list[int]:
    """Continue the tracked grid to both ends of the track.

    The DP starts at the first onset it can justify and stops at the last, so a
    quiet intro and a decaying tail fall outside it. A cut needs a grid over the
    *whole* song, and extrapolating at the tracked tempo is exactly what a
    listener does when the drums drop out.
    """
    step = max(2, period)
    if not beats:
        return list(range(0, n_frames, step))
    out = list(beats)
    while out[0] - step >= 0:
        out.insert(0, out[0] - step)
    while out[-1] + step < n_frames:
        out.append(out[-1] + step)
    return out


# ── Metre, energy, sections ──────────────────────────────────────────────────

def downbeat_phase(
    onsets: list[float], beats: list[int], low_band: list[float], beats_per_bar: int,
) -> int:
    """Which beat of the bar is the "1".

    Scored on the low band as well as on the onset function: what separates a
    downbeat from the other three is usually a kick, and the kick lives below
    200 Hz. Ties go to phase 0, which is what an unmetred track should get.
    """
    if not beats or beats_per_bar < 2:
        return 0
    floor_db = -120.0
    best_phase, best_score = 0, None
    for phase in range(beats_per_bar):
        score = 0.0
        for i in range(phase, len(beats), beats_per_bar):
            frame = beats[i]
            if frame < len(onsets):
                score += onsets[frame]
            if frame < len(low_band):
                score += (low_band[frame] - floor_db) / 60.0
        if best_score is None or score > best_score + 1e-9:
            best_phase, best_score = phase, score
    return best_phase


def _linear(db: float) -> float:
    return 10.0 ** (db / 20.0)


def bar_features(
    bands: list[list[float]], beats: list[int], phase: int, beats_per_bar: int,
) -> list[list[float]]:
    """Mean linear amplitude per band over each bar — the section fingerprint."""
    starts = list(range(phase, len(beats), beats_per_bar))
    features: list[list[float]] = []
    for k, start in enumerate(starts):
        a = beats[start]
        b = beats[starts[k + 1]] if k + 1 < len(starts) else len(bands[0])
        if b <= a:
            continue
        features.append([
            sum(_linear(v) for v in band[a:b]) / (b - a) for band in bands
        ])
    return features


def normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [0.5] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def rank_normalize(values: list[float]) -> list[float]:
    """Each value's position in the sorted order, 0 (lowest) to 1 (highest).

    Measured across this library on 2026-08-21, the bar-to-bar level variation
    in these tracks is only a few dB, with one near-silent bar at a fade
    carrying almost the whole absolute range: scale it linearly and every bar
    collapses to the bottom of the chart, scale it in dB and they all pile up
    at the top. Neither is readable, and neither is what the pacing decision
    actually uses — that is the ordering, which is what this returns.
    """
    n = len(values)
    if n < 2:
        return [0.5] * n
    order = sorted(range(n), key=values.__getitem__)
    ranks = [0.0] * n
    for position, index in enumerate(order):
        ranks[index] = position / (n - 1)
    return ranks


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return dot / (na * nb)


def find_sections(features: list[list[float]], kernel: int = 2) -> list[int]:
    """Bar indices where the music changes, by checkerboard novelty.

    The self-similarity matrix of the bar fingerprints has bright blocks along
    the diagonal wherever the music holds still; a checkerboard kernel slid down
    that diagonal peaks exactly at the corners between two blocks. Bar 0 always
    counts as a boundary — the piece starting is the first change there is.
    """
    n = len(features)
    if n < 2 * kernel + 1:
        return [0] if n else []

    novelty = [0.0] * n
    for centre in range(kernel, n - kernel):
        before = range(centre - kernel, centre)
        after = range(centre, centre + kernel)
        same = sum(_cosine(features[i], features[j]) for i in before for j in before)
        same += sum(_cosine(features[i], features[j]) for i in after for j in after)
        across = sum(_cosine(features[i], features[j]) for i in before for j in after)
        novelty[centre] = (same - 2.0 * across) / (2.0 * kernel * kernel)

    peak = max(novelty)
    if peak <= 0:
        return [0]
    threshold = peak * 0.45
    sections = [0]
    for i in range(kernel, n - kernel):
        if novelty[i] < threshold:
            continue
        if novelty[i] < novelty[i - 1] or novelty[i] < novelty[i + 1]:
            continue
        if i - sections[-1] < 2:          # two bars is not a section
            continue
        sections.append(i)
    return sections


# How much louder the onset function is on the beat grid than off it, mapped to
# 0..1. Measured across this library on 2026-08-21: the drift-removed envelope
# is sparse, so even a track with no pulse scores several times its own mean —
# the useful signal is the *spread*, which ran from 6.97 (the one track tagged
# "beatless, drone") to 24.5 (a LoFi track with a hard four-on-the-floor). The
# window below is set just inside that range, which puts the drone near zero
# and the obvious cases at the top.
#
# This is an indicator, not a detector. It answers "is the grid the tool found
# really in the music, or did it lay a metronome over an ambient wash" well
# enough to tell the user which one they are looking at.
CONFIDENCE_FLOOR, CONFIDENCE_CEILING = 6.5, 18.0


def beat_confidence(onsets: list[float], beats: list[int]) -> float:
    """How well the tracked grid lines up with the music's own attacks."""
    if not beats or not onsets:
        return 0.0
    on_grid = [onsets[b] for b in beats if b < len(onsets)]
    if not on_grid:
        return 0.0
    overall = statistics.fmean(onsets) or 1e-9
    ratio = statistics.fmean(on_grid) / overall
    span = CONFIDENCE_CEILING - CONFIDENCE_FLOOR
    return max(0.0, min(1.0, (ratio - CONFIDENCE_FLOOR) / span))


# ── Entry point ──────────────────────────────────────────────────────────────

def build_beatmap(
    bands: list[list[float]],
    *,
    bpm_hint: float | None = None,
    beats_per_bar: int = DEFAULT_BEATS_PER_BAR,
) -> BeatMap:
    """The pure half of `analyze`, so the maths can be tested without ffmpeg."""
    n_frames = len(bands[0])
    onsets = onset_strength(bands)
    period = estimate_period(onsets, bpm_hint)
    tracked = track_beats(onsets, period)
    confidence = beat_confidence(onsets, tracked)
    frames = extend_grid(tracked, period, n_frames)

    phase = downbeat_phase(onsets, frames, bands[0], beats_per_bar)
    features = bar_features(bands, frames, phase, beats_per_bar)
    # A rank, not a level — see `rank_normalize`. The planner reads it to pick
    # a cut length and the timeline draws it underneath the shots, and both
    # want the same thing: where this bar sits among the others.
    energies = rank_normalize([sum(f) for f in features])
    strengths = normalize([onsets[f] if f < len(onsets) else 0.0 for f in frames])

    return BeatMap(
        duration=round(n_frames / FRAME_RATE, 3),
        bpm=round(60.0 * FRAME_RATE / period, 2),
        beats=[round(f / FRAME_RATE, 4) for f in frames],
        beats_per_bar=beats_per_bar,
        downbeat_phase=phase,
        beat_strength=[round(v, 4) for v in strengths],
        bar_energy=[round(v, 4) for v in energies],
        sections=find_sections(features),
        confidence=round(confidence, 3),
    )


async def analyze(
    source: Path,
    *,
    bpm_hint: float | None = None,
    beats_per_bar: int = DEFAULT_BEATS_PER_BAR,
    ffmpeg_path: str = "ffmpeg",
) -> BeatMap:
    """Analyse one audio file into a BeatMap. ~0.5 s for a 60-second track."""
    bands = await _read_envelopes(source, ffmpeg_path)
    return build_beatmap(bands, bpm_hint=bpm_hint, beats_per_bar=beats_per_bar)


# ── Sidecar cache ────────────────────────────────────────────────────────────
# Analysis is cheap but not free, and the planner is re-run on every reseed and
# every style change in the UI. The cache turns that into a file read.

def cache_path(song_dir: Path, song_id) -> Path:
    return song_dir / f"{song_id}_beats.json"


async def load_or_analyze(
    source: Path,
    cache: Path,
    *,
    bpm_hint: float | None = None,
    beats_per_bar: int = DEFAULT_BEATS_PER_BAR,
    ffmpeg_path: str = "ffmpeg",
) -> BeatMap:
    """Cached `analyze`. A sidecar from an older ANALYSIS_VERSION, from a
    different metre, or a corrupt one is simply re-analysed."""
    try:
        data = json.loads(cache.read_text(encoding="utf-8"))
        if (data.get("version") == ANALYSIS_VERSION
                and data.get("beats_per_bar") == beats_per_bar):
            return BeatMap.from_json(data)
    except (OSError, ValueError, TypeError):
        pass

    beatmap = await analyze(
        source, bpm_hint=bpm_hint, beats_per_bar=beats_per_bar, ffmpeg_path=ffmpeg_path,
    )
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(beatmap.to_json()), encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not cache beat map at %s: %s", cache, exc)
    return beatmap
