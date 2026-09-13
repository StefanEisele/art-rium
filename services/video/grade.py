"""
Colour harmonisation for a multi-clip edit — making shots sit together without
making them the same shot.

A beat cut concatenates clips from different models: a Wan clip, a MiniMax
clip and a VACE render put side by side have different black levels, different
contrast and different colour casts, and the montage reads as a montage of
different films rather than as one piece. That is the problem this solves.

**The correction is a partial move towards the group, never a normalisation.**
Every clip is measured, the group's MEDIAN is taken as the target, and each clip
is moved a fraction `strength` of the way there. At strength 1.0 every clip
would end up on identical numbers — which is exactly the failure to avoid, so
the presets stop well below it and every individual move is capped on top. A
clip that is deliberately the dark one stays the dark one; it just stops being
dark in a different *language* from its neighbours.

Median rather than mean, for the same reason: one heavily stylised clip should
not drag the other five towards itself.

Measured behaviour of the two filters this leans on (2026-08-22, ffmpeg 2026-04):

  eq        out = (in - 0.5) * contrast + 0.5 + brightness, on 0..1 — exact to
            within 0.7/255 at both ends, at 8-bit and at 10-bit alike. So the
            contrast/brightness pair that maps one pair of levels onto another
            has a closed form and does not need to be searched for.

  lutyuv    operates in whatever depth reaches it: `val+8` moved an 8-bit
            clip's UAVG by 8 and a 10-bit clip's by 2 (in 8-bit terms). So the
            depth in front of it has to be pinned, not assumed.

  eq        emits yuv420p — 8-bit — for ANY input, 10-bit sources included
            (checked with showinfo on both). That settles the depth question
            for the whole chain: grading at 10 bits is not on offer, because
            the one filter doing the tone work converts down regardless.
            Believing otherwise is how the first version of this shifted chroma
            by four times what it had calculated, converging the clips' levels
            while pulling their colours further apart.

So the chain is pinned to 8 bits at the front, the chroma shift is applied in
the same units everything was measured in, and there is no scale factor to get
wrong. The finished cut is still encoded at 10 bits; what is lost is a little
precision on the two or three sources that had it, which is a smaller price
than a colour pass that silently does the opposite of what it says.
"""
from __future__ import annotations

import asyncio
import logging
import statistics
from dataclasses import dataclass
from pathlib import Path

from core.subproc import communicate

logger = logging.getLogger(__name__)

# Everything is measured and reasoned about in 8-bit units, whatever the source
# is, and converted at the last moment.
ANALYSIS_DEPTH = 8
_LEVELS = 255.0

# The depth the grade runs at, pinned at the head of the chain. Equal to the
# analysis depth on purpose — see the module docstring — so a U shift of 3
# measured units is written as `val+3` and means it.
GRADE_DEPTH = 8
GRADE_FORMAT = "yuv420p"
_CHROMA_SCALE = 1 << (GRADE_DEPTH - ANALYSIS_DEPTH)      # 1, and asserted below

# How much of the clip to look at. Two frames a second at 192px is a few dozen
# samples for a 4-second clip and costs about 0.2 s — the picture's statistics
# do not need more than that, and a per-frame walk would cost more than the
# render it is informing.
SAMPLE_FPS = 2.0
SAMPLE_WIDTH = 192
SAMPLE_FRAMES = 48

# ── Caps: the art-safe half of the calibration ───────────────────────────────
# Harmonising is allowed to close a gap, never to invent a look. These bound
# every individual move regardless of how far apart the clips are, so one
# outlier cannot be dragged into a different picture than the one that was shot.
MAX_CONTRAST = 1.35             # and its reciprocal on the other side
MAX_BRIGHTNESS = 0.12           # in eq's 0..1 units
MAX_CHROMA_SHIFT = 10.0         # 8-bit U/V units; ~4% of the axis
# Below this the clip has almost no tonal range (a near-flat frame), and the
# level-mapping division stops meaning anything.
MIN_LEVEL_SPAN = 8.0


@dataclass(frozen=True)
class Stats:
    """What one clip looks like, averaged over its sampled frames."""
    y_low: float            # 10th-percentile luma — the black level
    y_avg: float
    y_high: float           # 90th-percentile luma — the white level
    u_avg: float
    v_avg: float
    sat_avg: float
    frames: int = 0

    @property
    def span(self) -> float:
        return self.y_high - self.y_low


@dataclass(frozen=True)
class Grade:
    """The correction for one clip. All-default means "leave it alone"."""
    contrast: float = 1.0
    brightness: float = 0.0
    du: float = 0.0             # 8-bit U shift
    dv: float = 0.0             # 8-bit V shift

    @property
    def is_identity(self) -> bool:
        return (abs(self.contrast - 1.0) < 1e-3 and abs(self.brightness) < 1e-4
                and abs(self.du) < 0.05 and abs(self.dv) < 0.05)

    def filter_chain(self) -> str:
        """The ffmpeg filters, or "" when there is nothing to do.

        Ordered luma-then-chroma, and pinned to `GRADE_FORMAT` first, which is
        what makes `du`/`dv` mean the same thing here as they did in the
        measurement. Without that leading `format` the shift lands in whatever
        depth the previous filter happened to output — four times too far on a
        10-bit source.
        """
        if self.is_identity:
            return ""
        parts = [f"format={GRADE_FORMAT}"]
        if abs(self.contrast - 1.0) >= 1e-3 or abs(self.brightness) >= 1e-4:
            parts.append(
                f"eq=contrast={self.contrast:.4f}:brightness={self.brightness:.4f}"
            )
        if abs(self.du) >= 0.05 or abs(self.dv) >= 0.05:
            expr = []
            if abs(self.du) >= 0.05:
                expr.append(f"u='clip(val{self.du * _CHROMA_SCALE:+.1f},minval,maxval)'")
            if abs(self.dv) >= 0.05:
                expr.append(f"v='clip(val{self.dv * _CHROMA_SCALE:+.1f},minval,maxval)'")
            parts.append("lutyuv=" + ":".join(expr))
        return ",".join(parts)

    def describe(self) -> str:
        """One line for the UI — what this clip actually had done to it."""
        bits = []
        if abs(self.contrast - 1.0) >= 0.01:
            bits.append(f"Kontrast ×{self.contrast:.2f}")
        if abs(self.brightness) >= 0.005:
            bits.append(f"Helligkeit {self.brightness * 100:+.0f} %")
        if abs(self.du) >= 0.5 or abs(self.dv) >= 0.5:
            bits.append(f"Farbe {self.du:+.0f}/{self.dv:+.0f}")
        return " · ".join(bits) or "unverändert"


# ── Presets ──────────────────────────────────────────────────────────────────
# Named rather than a raw slider, because the useful range is narrow and the
# top of it is a place nobody wants to be. 1.0 is deliberately not offered.

HARMONIES: tuple[tuple[str, str, float, str], ...] = (
    ("aus", "Aus", 0.0,
     "Jeder Clip behält seine eigene Farbe und seinen eigenen Kontrast."),
    ("sanft", "Sanft", 0.35,
     "Nimmt den gröbsten Bruch zwischen den Clips heraus, lässt die Handschrift "
     "des einzelnen Clips aber deutlich stehen. Der Standard."),
    ("mittel", "Mittel", 0.60,
     "Deutlich eine Fassung — Schwarzwerte und Farbstich ziehen zusammen, die "
     "Stimmung des einzelnen Clips bleibt erkennbar."),
    ("stark", "Stark", 0.85,
     "Fast eine gemeinsame Gradation. Für Material, das aus sehr verschiedenen "
     "Modellen kommt und sonst nicht zusammenfindet."),
)

HARMONY_BY_KEY = {key: strength for key, _, strength, _ in HARMONIES}
DEFAULT_HARMONY = "sanft"


def harmony_options() -> list[dict]:
    return [{"key": k, "label": lbl, "strength": s, "hint": hint}
            for k, lbl, s, hint in HARMONIES]


def clamp_strength(value: float | None) -> float:
    if value is None:
        return 0.0
    return max(0.0, min(1.0, float(value)))


# ── Measurement ──────────────────────────────────────────────────────────────

def measure_command(ffmpeg_path: str, source: Path) -> list[str]:
    """ffmpeg argv that prints signalstats for a sample of the clip's frames.

    `format=yuv420p` before the stats is what makes the numbers comparable: a
    10-bit source would otherwise report on a 0..1023 scale and a 8-bit one on
    0..255, and the median across a mixed selection would be meaningless.
    """
    chain = (
        f"fps={SAMPLE_FPS},scale={SAMPLE_WIDTH}:-2,format=yuv420p,"
        "signalstats,metadata=print:file=-"
    )
    return [
        ffmpeg_path, "-v", "error", "-i", str(source),
        "-vf", chain, "-frames:v", str(SAMPLE_FRAMES), "-f", "null", "-",
    ]


_WANTED = {
    "YLOW": "y_low", "YAVG": "y_avg", "YHIGH": "y_high",
    "UAVG": "u_avg", "VAVG": "v_avg", "SATAVG": "sat_avg",
}


def parse_stats(stdout: str) -> Stats | None:
    """Average the per-frame signalstats lines into one Stats."""
    acc: dict[str, list[float]] = {name: [] for name in _WANTED.values()}
    for line in stdout.splitlines():
        key, sep, raw = line.partition("=")
        if not sep:
            continue
        field = _WANTED.get(key.rsplit(".", 1)[-1])
        if not field:
            continue
        try:
            acc[field].append(float(raw))
        except ValueError:
            continue
    if not acc["y_avg"]:
        return None
    mean = {name: statistics.fmean(values) if values else 0.0
            for name, values in acc.items()}
    return Stats(**mean, frames=len(acc["y_avg"]))


async def measure(source: Path, *, ffmpeg_path: str = "ffmpeg") -> Stats | None:
    """Sample one clip. Returns None if it could not be read — the caller then
    leaves that clip ungraded rather than failing the whole edit."""
    proc = await asyncio.create_subprocess_exec(
        *measure_command(ffmpeg_path, source),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await communicate(proc)
    if proc.returncode != 0:
        logger.warning("Colour measurement failed for %s: %s",
                       source.name, stderr.decode(errors="replace")[-200:])
        return None
    return parse_stats(stdout.decode(errors="replace"))


# ── Planning (pure — this is the calibration) ────────────────────────────────

def group_target(samples: list[Stats]) -> Stats:
    """The look the group is converging on: the median of every statistic.

    Taken per statistic rather than by picking one clip as the reference. No
    single clip is "the right one", and choosing one would make the result
    depend on selection order.
    """
    def med(field: str) -> float:
        return statistics.median(getattr(s, field) for s in samples)

    return Stats(
        y_low=med("y_low"), y_avg=med("y_avg"), y_high=med("y_high"),
        u_avg=med("u_avg"), v_avg=med("v_avg"), sat_avg=med("sat_avg"),
        frames=len(samples),
    )


def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def grade_for(stats: Stats, target: Stats, strength: float) -> Grade:
    """One clip's correction: `strength` of the way from where it is to the
    group, with every move capped."""
    strength = clamp_strength(strength)
    if strength <= 0.0:
        return Grade()

    # Where this clip should land — part of the way, never all of it.
    want_low = stats.y_low + (target.y_low - stats.y_low) * strength
    want_high = stats.y_high + (target.y_high - stats.y_high) * strength

    span = stats.span
    if span < MIN_LEVEL_SPAN or (want_high - want_low) < MIN_LEVEL_SPAN:
        # A nearly flat clip: stretching it to the group's range would amplify
        # whatever little is there into noise. Move its level, not its contrast.
        contrast = 1.0
        shift = (target.y_avg - stats.y_avg) * strength / _LEVELS
        brightness = _clamp(shift, MAX_BRIGHTNESS)
    else:
        contrast = (want_high - want_low) / span
        contrast = max(1.0 / MAX_CONTRAST, min(MAX_CONTRAST, contrast))
        low_n = stats.y_low / _LEVELS
        brightness = _clamp(
            (want_low / _LEVELS) - 0.5 - (low_n - 0.5) * contrast, MAX_BRIGHTNESS,
        )

    return Grade(
        contrast=round(contrast, 4),
        brightness=round(brightness, 4),
        du=round(_clamp((target.u_avg - stats.u_avg) * strength, MAX_CHROMA_SHIFT), 2),
        dv=round(_clamp((target.v_avg - stats.v_avg) * strength, MAX_CHROMA_SHIFT), 2),
    )


def harmonise(samples: list[Stats | None], strength: float) -> list[Grade]:
    """Grades for a whole selection.

    A clip that could not be measured passes through untouched, and — more
    importantly — is left out of the median, so a failed measurement cannot
    quietly pull the target off.
    """
    strength = clamp_strength(strength)
    known = [s for s in samples if s is not None]
    if strength <= 0.0 or len(known) < 2:
        return [Grade() for _ in samples]
    target = group_target(known)
    return [Grade() if s is None else grade_for(s, target, strength) for s in samples]


def spread(samples: list[Stats | None]) -> dict:
    """How far apart the clips are before anything is done — the number that
    says whether harmonising is worth switching on at all."""
    known = [s for s in samples if s is not None]
    if len(known) < 2:
        return {"measured": len(known), "black": 0.0, "white": 0.0, "colour": 0.0}
    return {
        "measured": len(known),
        "black": round(max(s.y_low for s in known) - min(s.y_low for s in known), 1),
        "white": round(max(s.y_high for s in known) - min(s.y_high for s in known), 1),
        "colour": round(max(
            max(s.u_avg for s in known) - min(s.u_avg for s in known),
            max(s.v_avg for s in known) - min(s.v_avg for s in known),
        ), 1),
    }
