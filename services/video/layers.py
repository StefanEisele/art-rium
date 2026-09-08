"""
Schichtenschnitt — one control video, many materials, cut to the music.

This is the sibling of services/video/cut.py and solves a different problem
with the same beat map. The beat cut deals shots from a bag: one clip at a
time, hard cuts, each shot a different moment. This one takes N renders that
share a control video — same geometry, same motion, same length, all looping —
and treats them as *layers of one shot* rather than as a sequence of shots.

**The rule everything follows: all tracks read the same source position at the
same output moment.** They came out of one control video, so at loop position
1.83 s every one of them shows the same form in the same place; only the
material differs. Cross-dissolving between two tracks at that moment therefore
morphs the *texture* while the shape stands still, which is the effect this
exists for. Break that rule anywhere — let one track run faster, start one
half a second late — and the dissolve becomes a dissolve between two unrelated
pictures, which is what an ordinary cross-fade already does.

So the timeline is one shared function of output time to source time,

    src(t) = the loop position every track is showing at output time t

built piecewise: constant speed inside a slot, continuous across slot
boundaries, and it never runs backwards. Speed is the second thing the music
drives — loud bars run fast, quiet bars run slow — and because the whole stack
is retimed together, a speed shift changes the *tempo of the form* without
touching the alignment. That is what makes the shift read as sync rather than
as a glitch.

**Sources loop, so the piece can be any length.** `src(t)` runs on past the
loop length, and every slot it crosses a loop boundary in is split there, so
the positions that reach the renderer are always inside [0, loop). Six
three-second loops are therefore enough material for a four-minute track, which
is the point of generating them as loops in the first place.

Splitting rather than letting ffmpeg repeat the input (`-stream_loop`) is the
difference between a correct render and a subtly wrong one. `-stream_loop`
repeats each file at its own length, so the moment two tracks are not exactly
the same duration they wrap at different times and drift apart — which is
precisely the invariant above, broken. Taking the position modulo the shared
loop keeps every track reading the same frame, and a track that happens to be
longer simply never shows its tail.

A seam is still a seam: at every whole multiple of the loop length the picture
jumps from the last frame back to the first, and that jump is invisible only if
the render actually loops. `LayerPlan.wraps` says where those moments are so
the UI can mark them and the eye can be pointed at them.

**Length is chosen in beats, on the same ladder as the beat cut**, and for the
same reason: 1, 2, 4, 8 or 16 beats read as intentional, 3.7 seconds reads as
an accident. Energy picks the rung, so the stack changes material fast in loud
bars and holds through quiet ones.
"""
from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, field

from services.video.beats import BeatMap
from services.video.cut import LADDER, _Bag
from services.video.cut_render import segment_frames

# How far the music may push the playback rate either side of 1.0. Past 2x the
# form stops being readable at all, and below 0.4 the interpolation the source
# was finished with starts to show as judder.
SPEED_MIN, SPEED_MAX = 0.35, 2.5
SPEED_LOW_DEFAULT, SPEED_HIGH_DEFAULT = 0.7, 1.6

# A dissolve shorter than this reads as a cut with a defect rather than as a
# transition, whatever it was asked to be.
MIN_BLEND_SECONDS = 0.08

# A transition may not eat more than this share of the segment it opens —
# otherwise a 1-beat segment at 90 BPM is nothing but transition and the form
# never gets a moment to simply stand there.
MAX_BLEND_SHARE = 0.6

# Same two as the beat cut, and for the same reason: the edit opens and closes
# on the music, not on a hard boundary.
OPENING_FADE_SECONDS = 0.60
CLOSING_FADE_SECONDS = 1.20


# ── Styles ───────────────────────────────────────────────────────────────────
# `quiet`/`loud` are indices into LADDER — the rung used in the least and the
# most energetic bar. `blend_beats` is how long a transition lasts, in beats,
# before it is capped against the segment it opens.

@dataclass(frozen=True)
class LayerStyle:
    key: str
    label: str
    hint: str
    quiet: int
    loud: int
    blend_beats: float
    jitter: float = 0.0


STYLES: tuple[LayerStyle, ...] = (
    LayerStyle("schweben", "Schweben",
               "Sehr lange Schichten, 16 und 8 Schläge, mit langen Überblendungen. "
               "Das Material wechselt, ohne dass man den Wechsel bemerkt.",
               quiet=0, loud=1, blend_beats=4.0, jitter=0.10),
    LayerStyle("welle", "Welle",
               "8 und 4 Schläge, weiche Übergänge über einen halben Takt. "
               "Der ruhige Grundschlag.",
               quiet=1, loud=2, blend_beats=2.0, jitter=0.15),
    LayerStyle("atmen", "Atmen",
               "Die größte Spannweite: hält 16 Schläge in den leisen Takten und "
               "wechselt jeden zweiten in den lauten. Folgt der Dramaturgie.",
               quiet=0, loud=3, blend_beats=1.0, jitter=0.15),
    LayerStyle("puls", "Puls",
               "4 und 2 Schläge, kurze Blenden. Jeder Takt bringt ein anderes "
               "Material auf dieselbe Form.",
               quiet=2, loud=3, blend_beats=0.5, jitter=0.15),
    LayerStyle("flimmern", "Flimmern",
               "2 Schläge und einzelne. Das Material wechselt schneller, als man "
               "es einzeln lesen kann — die Form trägt.",
               quiet=3, loud=4, blend_beats=0.25, jitter=0.20),
)

STYLE_BY_KEY = {s.key: s for s in STYLES}
DEFAULT_STYLE = "atmen"


def style_options() -> list[dict]:
    return [
        {"key": s.key, "label": s.label, "hint": s.hint,
         "fastest": LADDER[s.loud], "slowest": LADDER[s.quiet],
         "blend_beats": s.blend_beats}
        for s in STYLES
    ]


# ── Transitions ──────────────────────────────────────────────────────────────
# Each one is an ffmpeg `blend` expression over two time-aligned streams. They
# are written as builders rather than constants because `blend` hands an
# expression the frame's timestamp `T`, not a normalised position — so progress
# has to be computed and substituted in.
#
# Every builder receives `p`: progress through the WHOLE transition, 0..1, not
# through this slot. The two differ whenever a transition was split at a loop
# boundary, and using the slot's own clock there would restart the dissolve
# halfway through it. It also receives `variant`, a small integer taken off the
# slot's beat index, so a wipe does not travel the same way and a dissolve does
# not use the same cloud forty times over. Both halves of a split slot carry
# the same beat, so both halves pick the same variant and the seam holds.
#
#
# ── Why a builder answers per plane ──────────────────────────────────────────
#
# `blend` evaluates its expression once per SAMPLE, which in YUV means once on
# the luma plane and once on each chroma plane. Those planes do not mean the
# same thing: 0 is black on the luma plane, but on the U plane 0 is fully green
# and the neutral sits at half scale. Anything that is not a plain
# interpolation is therefore wrong on chroma, and `all_expr` — one expression
# for all three — cannot say so.
#
# That was not theoretical. The old `licht` screened A and B together, which on
# luma is a highlight bloom and on chroma drives both U and V toward their
# maxima. High U with high V is magenta. Measured 2026-09-06 with two neutral
# grey sources: at the midpoint the output was RGB (255, 139, 255), and it
# stayed magenta across most of the transition rather than blooming at its
# centre. The "white" flash was a hot pink one.
#
# So a builder returns a `BlendExpr` — one expression for luma, one for chroma
# — and the renderer emits c0/c1/c2 instead of `all_expr`.
#
#
# ── Bit depth ────────────────────────────────────────────────────────────────
#
# The whole render meets in 10 bits, and the expressions are written against
# that scale. Blending is where banding is made: a slow dissolve across a
# smooth gradient is precisely the picture that shows every quantisation step,
# and the output has been 10-bit all along, so the 8-bit intermediate was
# throwing away precision the encoder was ready to keep. Measured at 816x1440,
# 10-bit blending cost nothing over 8-bit (8.85 s against 9.56 s for 150
# frames) — it is a free two bits.

BLEND_DEPTH = 10
BLEND_PEAK = (1 << BLEND_DEPTH) - 1          # white / full scale — 1023
BLEND_NEUTRAL = 1 << (BLEND_DEPTH - 1)       # chroma neutral — 512
BLEND_PIX_FMT = "yuv420p10le"

# The exponent that takes a code value to light and back. Blending two pictures
# is physically an addition of light, not of code values, so `weich` undoes the
# transfer, mixes, and puts it back. Without that the midpoint of a dissolve
# sits darker than either side — the familiar muddy centre of a cross-fade.
DISPLAY_GAMMA = 2.2

# `licht`: how sharply the flash peaks and how far it goes. The lift is a
# raised cosine taken to a power, so raising the exponent narrows the bloom to
# the middle of the transition instead of washing the whole of it.
FLASH_SHARPNESS = 3.0
FLASH_AMOUNT = 0.88

# `wisch`: the soft edge, as a fraction of the frame. A one-pixel edge is an
# aliased staircase that crawls as it travels; a feathered band reads as a
# wipe. Four directions, picked by variant, because one direction forty times
# over is a tic.
WIPE_FEATHER = 0.055
_WIPE_AXES = ("(X/W)", "(1-X/W)", "(Y/H)", "(1-Y/H)")

# `aufloesen`: the width of the band in which the new material is arriving, and
# the phases that move the cloud around between transitions.
DISSOLVE_BAND = 0.30
_DISSOLVE_PHASES = (0.0, 1.7, 3.4, 5.1)


@dataclass(frozen=True)
class BlendExpr:
    """One transition, as the two expressions `blend` actually needs.

    `chroma` goes to both U and V. Most transitions set it equal to `luma` —
    a wipe or a straight interpolation means the same thing on every plane —
    and the ones that do not are the reason this type exists.
    """
    luma: str
    chroma: str

    @property
    def is_uniform(self) -> bool:
        return self.luma == self.chroma


def _mix(p: str) -> str:
    """Plain interpolation. Correct on any plane, which is why it is what the
    per-plane transitions fall back to on chroma."""
    return f"A*(1-{p})+B*{p}"


def _weich(p: str, variant: int) -> BlendExpr:
    """Straight cross-dissolve, mixed in light rather than in code values.

    The material fades from one to the other while the form underneath does
    not move at all. Luma is taken out of the display transfer, mixed, and put
    back, so a half-and-half moment carries the light of both pictures instead
    of the average of their code values — measured on an 8-bit pair of 64 and
    192, the linear midpoint is 145 where the naive one is 128. Chroma is
    interpolated plainly: it is not a light quantity and has no transfer to
    undo.
    """
    g = DISPLAY_GAMMA
    luma = (f"pow(pow(A/{BLEND_PEAK},{g})*(1-{p})+pow(B/{BLEND_PEAK},{g})*{p}"
            f",1/{g})*{BLEND_PEAK}")
    return BlendExpr(luma, _mix(p))


def _licht(p: str, variant: int) -> BlendExpr:
    """Dissolve that blooms through white at the halfway point.

    `screen` rather than `add`, which would clip the highlights it is made of.
    On chroma there is no screening to do — a flash is an overexposure, and an
    overexposure loses its colour — so chroma is pulled toward neutral by the
    same amount the luma is lifted. That is what makes it read as white rather
    than as the magenta this used to produce.
    """
    lift = f"(pow(4*{p}*(1-{p}),{FLASH_SHARPNESS})*{FLASH_AMOUNT})"
    mix = f"({_mix(p)})"
    return BlendExpr(
        luma=f"{mix}*(1-{lift})+(A+B-A*B/{BLEND_PEAK})*{lift}",
        chroma=f"{mix}*(1-{lift})+{BLEND_NEUTRAL}*{lift}",
    )


def _wisch(p: str, variant: int) -> BlendExpr:
    """A soft edge travelling across the frame.

    On this material it is the strongest of the four: the edge crosses a form
    that is identical on both sides, so what moves is purely the substance the
    form is made of. The edge is feathered and eased — a hard edge at pixel
    resolution aliases into a crawling staircase, and a wipe that travels at a
    constant rate reads as a machine rather than as a gesture.

    `X/W` and `Y/H` are normalised, so the edge lands in the same place on the
    half-resolution chroma planes as on luma. That is what keeps the colour
    from fringing along it, and why this one expression serves all three.
    """
    axis = _WIPE_AXES[variant % len(_WIPE_AXES)]
    f = WIPE_FEATHER
    ease = f"({p}*{p}*(3-2*{p}))"
    # The edge travels from just off one side to just off the other, so both
    # ends of the transition are fully clear of it.
    pos = f"({ease}*(1+{f})-{f}/2)"
    expr = f"A+(B-A)*clip(({pos}-{axis})/{f}+0.5,0,1)"
    return BlendExpr(expr, expr)


def _aufloesen(p: str, variant: int) -> BlendExpr:
    """The new material bleeds through in soft clouds rather than as an even
    fade.

    This used to be `random(1)` — per pixel, per frame. Two things were wrong
    with it. The draw is independent on every plane, so a speckle that arrived
    in luma had not arrived in chroma and the whole frame crawled with coloured
    confetti. And per-pixel noise is the one thing an inter-frame encoder
    cannot code: it boiled at the frame rate, ate the bitrate, and came back
    from the encoder as blocks.

    What replaced it is a smooth two-octave field built from the normalised
    coordinates, so it is identical on every plane, stands still instead of
    boiling, and moves in patches an encoder can carry. The threshold sweeps
    through it inside a soft band, so material arrives by bleeding rather than
    by popping. Two octaves rather than three: measured at 816x1440 the third
    cost 8.5 s per 150 frames and could not be seen.
    """
    ph = _DISSOLVE_PHASES[variant % len(_DISSOLVE_PHASES)]
    field = (f"(0.5+0.5*(0.65*sin(X/W*5.3+Y/H*3.7+{ph:.2f})"
             f"+0.35*sin(X/W*13.1-Y/H*9.7+{ph + 2.1:.2f})))")
    b = DISSOLVE_BAND
    expr = f"A+(B-A)*clip((({p}*(1+{b}))-{field})/{b},0,1)"
    return BlendExpr(expr, expr)


TRANSITIONS: tuple[tuple[str, str, str], ...] = (
    ("weich", "Weich",
     "Reine Überblendung, in Licht gerechnet statt in Zahlen. Die Form steht, "
     "das Material wandert — und die Mitte sackt nicht ab."),
    ("licht", "Licht",
     "Überblendung, die in der Mitte durch Weiß aufleuchtet und dabei die "
     "Farbe verliert, wie eine Überbelichtung."),
    ("wisch", "Wisch",
     "Eine weiche Kante läuft durchs Bild und wechselt von Mal zu Mal die "
     "Richtung. Der stärkste Effekt."),
    ("aufloesen", "Auflösen",
     "Das neue Material blutet in weichen Wolken durch das alte."),
    ("hart", "Hart", "Kein Übergang — der Wechsel sitzt auf dem Schlag."),
    ("mix", "Gemischt", "Pro Wechsel eine andere der vier Blenden."),
)

_BUILDERS = {
    "weich": _weich, "licht": _licht, "wisch": _wisch, "aufloesen": _aufloesen,
}
_MIXABLE = ("weich", "licht", "wisch", "aufloesen")
DEFAULT_TRANSITION = "weich"


def transition_options() -> list[dict]:
    return [{"key": k, "label": lab, "hint": h} for k, lab, h in TRANSITIONS]


def blend_expr(
    kind: str, duration: float, p0: float = 0.0, p1: float = 1.0,
    variant: int = 0,
) -> BlendExpr:
    """The ffmpeg `blend` expressions for `kind` over one slot.

    `duration` is the slot's length in seconds and (`p0`, `p1`) the part of the
    transition it covers. A whole transition is (0, 1); a transition split at a
    loop boundary is (0, 0.4) and (0.4, 1), and the dissolve carries straight
    across the seam. `variant` picks between the shapes a transition can take —
    which way a wipe runs, which cloud a dissolve uses — and must be the same
    for both halves of a split, or the seam shows.
    """
    builder = _BUILDERS.get(kind)
    if builder is None:
        raise ValueError(f"{kind!r} is not a blended transition")
    d = max(duration, MIN_BLEND_SECONDS)
    if p0 == 0.0 and p1 == 1.0:
        progress = f"(T/{d:.4f})"
    else:
        progress = f"({p0:.5f}+{p1 - p0:.5f}*T/{d:.4f})"
    return builder(progress, variant)


# ── Speed: one continuous curve, not a staircase ─────────────────────────────
#
# Speed used to be one constant per slot, stepping instantly at every slot
# boundary. Measured on a 40 s plan: 37 slots, 9 distinct speeds, 11 step
# changes, the largest of them 0.63x — the picture ran at 1.56x and then, in
# one frame, at 0.93x. The source POSITION was continuous; its DERIVATIVE was
# not, and a 40% jump in velocity does not read as rhythm. It reads as the
# picture stopping and starting again.
#
# What replaced it is a continuous function of output time with two parts:
#
#   **Schwelle** — the swell. The bar energy, smoothed between bar centres,
#   mapped onto [speed_low, speed_high]. This is the old relationship (loud
#   bars run fast) with the staircase taken out of it.
#
#   **Puls** — the accent at a cut. Approaching a change of material the
#   picture brakes; through the change it pushes. So the slowest moment is the
#   cut itself — which also gives the eye a beat to register the new material —
#   and the acceleration peaks just after it. Continuous throughout, on
#   purpose: a step at the cut would be masked by a hard change of material but
#   plainly visible in the middle of a cross-dissolve.
#
# Both have their own dial and either can be turned off.
#
# The curve is carried into the render as a piecewise-LINEAR speed: each
# rendered segment ramps from one speed to another, which the renderer
# expresses as a quadratic `setpts`. Knots are placed where the curve actually
# bends — around each cut — rather than uniformly, which keeps the segment
# count near 3 per slot instead of one per frame.

# How long the brake takes, and the push after it. Roughly a sixteenth at
# 120 BPM: long enough to feel as a movement, short enough to stay an accent
# rather than becoming the tempo.
ACCENT_SECONDS = 0.28

# The longest stretch that may be rendered as a single ramp. Only the swell
# lives out here and it bends slowly, but a 16-beat slot is eight seconds and
# one straight line across all of it would flatten what the swell is doing.
MAX_KNOT_SECONDS = 1.6

# **Measured 2026-09-06, and the reason this floor exists at all.** A branch
# asked for exactly ONE frame collapses: 200 one-frame segments rendered 0.10 s
# instead of 6.67 s — 98.5% of the piece gone, with ffmpeg exiting 0. At two
# frames and above every length came out exact. It is not the segment count
# (900 segments rendered a perfect 450 s) and not the motion mode (plain `fps=`
# fails identically), it is the one-frame branch itself.
#
# This already bit the old planner: across a sweep of styles, tempi and seeds,
# 1936 slots came out as a single frame — every one of them a blend sliver left
# by `split_at_wraps` cutting a transition a few milliseconds from a loop
# boundary. They were silently shortening renders before any of this.
MIN_SEGMENT_FRAMES = 2

SWELL_DEFAULT = 1.0
PULSE_DEFAULT = 0.35
PULSE_MAX = 0.75


def clamp_unit(value, default: float) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _smoothstep(x: float) -> float:
    """0..1 with zero slope at both ends. Used everywhere a value has to
    arrive somewhere rather than run into it."""
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


class SpeedCurve:
    """Playback rate as a continuous function of output time.

    Holds the whole piece, not one slot: the accent at a cut reaches back
    before it and forward past it, so no slot can compute its own speed
    without knowing its neighbours.
    """

    def __init__(
        self, *, marks: list[tuple[float, float]], low: float, high: float,
        swell: float, pulse: float, cuts: list[float],
    ):
        # (time, energy) at each bar's centre.
        self.marks = marks or [(0.0, 0.5)]
        self.low, self.high = low, high
        self.swell, self.pulse = swell, pulse
        self.cuts = sorted(cuts)

    # ── the swell ────────────────────────────────────────────────────────
    def energy_at(self, t: float) -> float:
        marks = self.marks
        if len(marks) == 1 or t <= marks[0][0]:
            return marks[0][1]
        if t >= marks[-1][0]:
            return marks[-1][1]
        lo = 0
        hi = len(marks) - 1
        while hi - lo > 1:                      # the marks are sorted
            mid = (lo + hi) // 2
            if marks[mid][0] <= t:
                lo = mid
            else:
                hi = mid
        (t0, e0), (t1, e1) = marks[lo], marks[hi]
        if t1 <= t0:
            return e1
        return e0 + (e1 - e0) * _smoothstep((t - t0) / (t1 - t0))

    def base_at(self, t: float) -> float:
        """Speed before the accent. `swell` at 0 holds the middle of the band,
        so turning the swell off gives a constant tempo rather than a slow one."""
        mid = (self.low + self.high) / 2.0
        return mid + (self.energy_at(t) - 0.5) * (self.high - self.low) * self.swell

    # ── the accent ───────────────────────────────────────────────────────
    def _nearest_cut(self, t: float) -> float | None:
        """The cut whose accent owns this moment.

        Nearest rather than a sum of all of them: at a fast tempo the accents
        overlap, and multiplying two together would compound into a lurch far
        deeper than the dial asks for.
        """
        if not self.cuts:
            return None
        lo, hi = 0, len(self.cuts) - 1
        best = self.cuts[0]
        while lo <= hi:
            mid = (lo + hi) // 2
            c = self.cuts[mid]
            if abs(c - t) < abs(best - t):
                best = c
            if c < t:
                lo = mid + 1
            else:
                hi = mid - 1
        return best

    def pulse_at(self, t: float) -> float:
        """Multiplier around 1.0: brake into the cut, push through it.

            t = c - A   1.0        running
            t = c       1 - d      slowest, exactly on the change
            t = c + A   1 + d      pushing
            t = c + 2A  1.0        settled
        """
        if self.pulse <= 0:
            return 1.0
        c = self._nearest_cut(t)
        if c is None:
            return 1.0
        d, a = self.pulse, ACCENT_SECONDS
        if t < c - a or t > c + 2 * a:
            return 1.0
        if t <= c:
            return 1.0 - d * _smoothstep((t - (c - a)) / a)
        if t <= c + a:
            return (1.0 - d) + 2 * d * _smoothstep((t - c) / a)
        return 1.0 + d * (1.0 - _smoothstep((t - (c + a)) / a))

    def at(self, t: float) -> float:
        return max(SPEED_MIN, min(SPEED_MAX, self.base_at(t) * self.pulse_at(t)))

    # ── where the curve needs a knot ─────────────────────────────────────
    def knots_in(self, start: float, end: float) -> list[float]:
        """Times inside (start, end) where the curve bends hard enough to need
        its own segment."""
        a = ACCENT_SECONDS
        out: set[float] = set()
        for c in self.cuts:
            if c + 2 * a < start or c - a > end:
                continue
            for k in (c - a, c, c + a, c + 2 * a):
                if start + 1e-6 < k < end - 1e-6:
                    out.add(round(k, 6))
        return sorted(out)


def _split_points(
    start: float, end: float, curve: SpeedCurve, min_span: float,
) -> list[float]:
    """Segment boundaries for one stretch: the curve's knots, a cap on plain
    runs, and nothing shorter than `min_span`.

    **The floor belongs here, not downstream.** An accent knot landing a
    fraction before the end of a stretch used to produce a 60 ms segment, and
    on one 40 s plan 135 of 312 segments were that. Absorbing them afterwards
    meant re-solving a neighbour's ramp to keep the source position exact,
    which put the speed steps straight back in — measured at 0.61, which is
    the size of the step this whole curve exists to remove. Declining to place
    the knot costs a little accent precision on a short stretch and costs the
    continuity nothing.
    """
    marks = [start] + curve.knots_in(start, end) + [end]
    kept = [marks[0]]
    for k in marks[1:-1]:
        if k - kept[-1] >= min_span:
            kept.append(k)
    # The last knot has to leave the closing segment room as well, so it goes
    # if it is too near the end rather than the end being moved.
    while len(kept) > 1 and end - kept[-1] < min_span:
        kept.pop()
    kept.append(end)

    out = [kept[0]]
    for nxt in kept[1:]:
        prev = out[-1]
        span = nxt - prev
        if span > MAX_KNOT_SECONDS:
            steps = int(math.ceil(span / MAX_KNOT_SECONDS))
            for i in range(1, steps):
                out.append(prev + span * i / steps)
        out.append(nxt)
    return out


def solve_ramp(v0: float, v1: float, duration: float, distance: float) -> float:
    """Output time at which a ramp from v0 to v1 has consumed `distance`.

    src(t) = v0*t + (v1-v0)*t^2/(2T). Inverting is a quadratic, and it is
    needed in two places: finding where a loop boundary falls inside a slot,
    and telling the renderer where the picture is. Constant speed degenerates
    to the obvious division rather than being a special case at the call site.
    """
    if duration <= 0:
        return 0.0
    a = (v1 - v0) / (2.0 * duration)
    if abs(a) < 1e-9:
        return distance / max(v0, 1e-9)
    disc = v0 * v0 + 4.0 * a * distance
    if disc <= 0:
        return duration
    return (-v0 + math.sqrt(disc)) / (2.0 * a)


# ── The plan ─────────────────────────────────────────────────────────────────

@dataclass
class Slot:
    """One stretch of output with a single speed RAMP and one layer state.

    A slot is either a hold (`from_track` is None) or a transition, and the
    two alternate. Both carry the same shared source window, which is what
    keeps every track on the same form.

    Speed is two numbers rather than one. It enters at `speed_in`, leaves at
    `speed_out`, and moves linearly in output time between them — so the
    source position is a quadratic, and its derivative is continuous across
    the whole piece instead of stepping at every boundary.
    """
    start: float            # output timeline, seconds
    end: float
    src_in: float           # shared loop position, seconds — may exceed the loop
    src_out: float
    speed_in: float         # source seconds per output second, entering
    speed_out: float        # ... and leaving
    track: int              # the material this slot ends on
    from_track: int | None  # the material it started from; None = a hold
    blend: str              # transition key; "" on a hold
    bar: int
    beat: int
    beats: float            # length in beats
    section: bool           # opens a new musical section
    # Which part of the transition this slot covers, 0..1. A transition split
    # by a loop boundary — or by a speed knot — would otherwise restart its
    # dissolve from the beginning instead of carrying on where it left off.
    #
    # Measured in OUTPUT time, not source distance. The two agree only while
    # the speed is constant, and `blend` drives its expression off the frame's
    # timestamp, so output time is the one that is actually right.
    blend_p0: float = 0.0
    blend_p1: float = 1.0

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def speed(self) -> float:
        """The mean rate across the slot — what it consumes, divided by how
        long it takes. Kept because a single number is what a timeline label
        and a sanity check want."""
        return (self.speed_in + self.speed_out) / 2.0

    @property
    def is_blend(self) -> bool:
        return self.from_track is not None and self.from_track != self.track

    def speed_at(self, t: float) -> float:
        """The rate at an absolute output time inside this slot."""
        d = self.duration
        if d <= 0:
            return self.speed_in
        f = max(0.0, min(1.0, (t - self.start) / d))
        return self.speed_in + (self.speed_out - self.speed_in) * f


@dataclass
class LayerPlan:
    slots: list[Slot]
    style: str
    transition: str
    seed: int
    tracks: int
    loop: float             # the shared loop length every track was read at
    bpm: float
    beats_per_bar: int
    song_start: float
    song_end: float
    speed_low: float
    speed_high: float
    swell: float = SWELL_DEFAULT
    pulse: float = PULSE_DEFAULT
    # How many times the material actually changes. Not the same as the number
    # of slots any more: a slot is now a render segment, and the speed curve
    # needs several of them per change.
    changes: int = 0
    # Total source walked through, before the wrap. Stored rather than read off
    # the last slot: every position was taken modulo the loop by then, so the
    # last one says where in the loop the piece ends, not how far it travelled.
    consumed: float = 0.0
    wraps: list[float] = field(default_factory=list)   # output times where the loop restarts
    warnings: list[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.song_end - self.song_start

    @property
    def source_consumed(self) -> float:
        return self.consumed

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict) -> "LayerPlan":
        payload = dict(data)
        payload["slots"] = [Slot(**s) for s in payload.get("slots", [])]
        payload.setdefault("wraps", [])
        payload.setdefault("warnings", [])
        return cls(**payload)


def clamp_speed(value: float | None, default: float) -> float:
    try:
        return max(SPEED_MIN, min(SPEED_MAX, float(value)))
    except (TypeError, ValueError):
        return default


def rung_for_energy(style: LayerStyle, energy: float, rng: random.Random) -> int:
    """Which LADDER index this bar's loudness calls for.

    Linear between the style's two endpoints, then jittered by at most one rung
    so a long stretch at constant energy does not metronome. Identical in shape
    to services/video/cut.py's — the two planners read the music the same way
    and only differ in what they do with the answer.
    """
    lo, hi = style.quiet, style.loud
    idx = lo + (hi - lo) * max(0.0, min(1.0, energy))
    rung = int(round(idx))
    if style.jitter and rng.random() < style.jitter:
        rung += rng.choice((-1, 1))
    return max(0, min(len(LADDER) - 1, rung))


def speed_for_energy(energy: float, low: float, high: float) -> float:
    """Playback rate for a bar of this loudness, before the accent.

    Linear, deliberately: the relationship between what is heard and what is
    seen should be predictable. The curve that carries it is what changed —
    this is still the value the curve passes through.
    """
    return low + (high - low) * max(0.0, min(1.0, energy))


def plan_layers(
    beatmap: BeatMap,
    *,
    tracks: int,
    loop: float,
    style: str = DEFAULT_STYLE,
    transition: str = DEFAULT_TRANSITION,
    seed: int | None = None,
    speed_low: float = SPEED_LOW_DEFAULT,
    speed_high: float = SPEED_HIGH_DEFAULT,
    swell: float = SWELL_DEFAULT,
    pulse: float = PULSE_DEFAULT,
    fps: int = 24,
    span: tuple[float, float] | None = None,
) -> LayerPlan:
    """Build the edit. Deterministic in every argument, `seed` included.

    Four passes, and the order is the design:

      1. Walk the beat grid and decide WHEN the material changes and to what.
         Nothing about speed enters here — the output timeline is the music's,
         and it would be the same at any tempo.
      2. Build the speed curve, which needs the finished list of changes
         because the accent reaches back before each one and forward past it.
      3. Cut every stretch at the curve's knots and integrate the source
         position through it. This is where slots become render segments.
      4. Wrap the positions into the loop, then absorb anything too short to
         render.
    """
    if tracks < 2:
        raise ValueError("A layer cut needs at least two tracks")
    if loop <= 0:
        raise ValueError("The loop length must be positive")

    sty = STYLE_BY_KEY.get(style, STYLE_BY_KEY[DEFAULT_STYLE])
    rng = random.Random(seed if seed is not None else random.randrange(1 << 30))
    resolved_seed = seed if seed is not None else rng.randrange(1 << 30)
    rng = random.Random(resolved_seed)

    low = clamp_speed(speed_low, SPEED_LOW_DEFAULT)
    high = clamp_speed(speed_high, SPEED_HIGH_DEFAULT)
    if high < low:
        low, high = high, low
    swell = clamp_unit(swell, SWELL_DEFAULT)
    pulse = min(PULSE_MAX, clamp_unit(pulse, PULSE_DEFAULT))

    start, end = span or (0.0, beatmap.duration)
    start = max(0.0, start)
    end = min(beatmap.duration, end)
    warnings: list[str] = []
    if end - start < 1.0:
        raise ValueError("The span is too short to cut")

    beats = beatmap.beats
    bar_starts = set(beatmap.bar_starts())
    sections = set(beatmap.sections)

    def bar_of(beat_index: int) -> int:
        return max(0, (beat_index - beatmap.downbeat_phase) // beatmap.beats_per_bar)

    def energy_at(beat_index: int) -> float:
        if not beatmap.bar_energy:
            return 0.5
        return beatmap.bar_energy[min(bar_of(beat_index), len(beatmap.bar_energy) - 1)]

    def beat_time(beat_index: int) -> float:
        """Seconds for a beat index, extrapolated past the tracked grid.

        The grid stops at the last beat the tracker could justify, which on a
        4/4 track can leave a second or more of music with no beats under it.
        Extrapolating at the last known interval keeps the edit going to the
        end of the song rather than stopping early.
        """
        if beat_index < len(beats):
            return beats[beat_index]
        if len(beats) < 2:
            return beatmap.duration
        step = beats[-1] - beats[-2]
        return beats[-1] + step * (beat_index - len(beats) + 1)

    # Open on the first beat at or after the span's start; anything before it
    # is covered by extending the first slot backwards, so the piece starts
    # with the music rather than with the grid.
    first = next((i for i, t in enumerate(beats) if t >= start - 1e-6), 0)

    bag = _Bag(tracks, rng)
    current = bag.take()

    # ── 1. when the material changes, and to what ────────────────────────────
    # Output timeline only. No speed, no source position: those come after,
    # because the accent in the curve has to know where every change is.
    stretches: list[dict] = []
    pos = start
    beat_i = first
    guard = 0

    while pos < end - 1e-6:
        guard += 1
        if guard > 20000:                      # pathological grid; stop cleanly
            warnings.append("Der Beat-Raster brach ab — der Schnitt endet früher.")
            break

        energy = energy_at(beat_i)
        step_beats = LADDER[rung_for_energy(sty, energy, rng)]

        segment_end = min(beat_time(beat_i + step_beats), end)
        if segment_end <= pos + 1e-6:
            beat_i += max(1, step_beats)
            continue

        nxt = bag.take() if stretches else current
        kind = _pick_transition(transition, rng)
        beat_len = max(beat_time(beat_i + 1) - beat_time(beat_i), 1e-6)

        # The transition opens the segment and the hold fills the rest of it,
        # so the beat carries the change and the form then has time to stand.
        blend_len = 0.0
        if kind and stretches:
            wanted = sty.blend_beats * beat_len
            blend_len = min(wanted, (segment_end - pos) * MAX_BLEND_SHARE)
            if blend_len < MIN_BLEND_SECONDS:
                blend_len = 0.0

        is_section = bar_of(beat_i) in sections and beat_i in bar_starts

        if blend_len > 0:
            stretches.append(dict(
                start=pos, end=pos + blend_len, track=nxt, from_track=current,
                blend=kind, bar=bar_of(beat_i), beat=beat_i,
                beat_len=beat_len, section=is_section,
            ))
            pos += blend_len
            current = nxt
        elif stretches:
            # A hard change still happens here; it just has no slot of its own.
            current = nxt

        if pos < segment_end - 1e-6:
            stretches.append(dict(
                start=pos, end=segment_end, track=current, from_track=None,
                blend="", bar=bar_of(beat_i), beat=beat_i, beat_len=beat_len,
                section=is_section and blend_len <= 0,
            ))
            pos = segment_end

        beat_i += step_beats

    if not stretches:
        raise ValueError("No slots were produced — is the beat grid empty?")

    # Hold the last stretch out to the end of the span rather than stopping on
    # the last beat the grid could justify.
    if stretches[-1]["end"] < end - 1e-6:
        stretches[-1]["end"] = end

    # ── 2. the speed curve ───────────────────────────────────────────────────
    # Every stretch opens on a change of material, so its start is a cut. The
    # first one is not: nothing changed there, the piece merely began.
    cuts = [st["start"] for st in stretches[1:]]
    marks: list[tuple[float, float]] = []
    if beatmap.bar_energy:
        bar_index = beatmap.bar_starts()
        for i, e in enumerate(beatmap.bar_energy):
            if i < len(bar_index):
                t0 = beat_time(bar_index[i])
                t1 = beat_time(bar_index[i] + beatmap.beats_per_bar)
                marks.append(((t0 + t1) / 2.0, e))
    curve = SpeedCurve(marks=marks, low=low, high=high, swell=swell,
                       pulse=pulse, cuts=cuts)

    # ── 3. knots, and the source position integrated through them ────────────
    slots: list[Slot] = []
    src = 0.0
    min_span = MIN_SEGMENT_FRAMES / fps if fps > 0 else 0.0
    for st in stretches:
        points = _split_points(st["start"], st["end"], curve, min_span)
        whole = st["end"] - st["start"]
        for a, b in zip(points, points[1:]):
            if b - a <= 1e-9:
                continue
            v0, v1 = curve.at(a), curve.at(b)
            consumed = (b - a) * (v0 + v1) / 2.0
            slots.append(Slot(
                start=a, end=b, src_in=src, src_out=src + consumed,
                speed_in=v0, speed_out=v1,
                track=st["track"], from_track=st["from_track"],
                blend=st["blend"], bar=st["bar"], beat=st["beat"],
                beats=(b - a) / st["beat_len"],
                section=st["section"] and a == st["start"],
                blend_p0=(a - st["start"]) / whole if whole else 0.0,
                blend_p1=(b - st["start"]) / whole if whole else 1.0,
            ))
            src += consumed

    consumed_total = src
    wraps = wrap_times(slots, loop)
    # Done on a finished timeline: every slot that crosses a loop boundary
    # becomes two, and the positions that leave here are all inside [0, loop).
    slots = split_at_wraps(slots, loop)
    # Last, because splitting is what makes the slivers: a transition cut a few
    # milliseconds from a loop boundary leaves a piece no renderer can draw.
    slots = merge_slivers(slots, fps)

    if consumed_total < loop * 0.4 and len(slots) > 2:
        warnings.append(
            f"Der Schnitt verbraucht nur {consumed_total:.1f}s der {loop:.1f}s langen "
            "Loop — ein höheres Tempo oder ein längeres Stück nutzt mehr davon."
        )

    return LayerPlan(
        slots=slots, style=sty.key, transition=transition, seed=resolved_seed,
        tracks=tracks, loop=loop, bpm=beatmap.bpm,
        beats_per_bar=beatmap.beats_per_bar,
        song_start=start, song_end=end,
        speed_low=low, speed_high=high,
        swell=swell, pulse=pulse,
        changes=len(stretches),
        consumed=consumed_total,
        wraps=wraps,
        warnings=warnings,
    )


def split_at_wraps(slots: list[Slot], loop: float) -> list[Slot]:
    """Cut every slot at the loop boundaries it crosses, wrapping its position.

    Belongs to the planner rather than the renderer because it changes what the
    plan *is*: the timeline the UI draws, the slot count, and the transition
    progress all move. A renderer that did this quietly would be showing the
    user one edit and rendering another.

    Every slot that comes out reads inside [0, loop), so the render needs no
    looping machinery at all — and a track longer than the shared loop simply
    never has its tail read, which is what keeps unequal tracks aligned.
    """
    if loop <= 0:
        return slots
    out: list[Slot] = []
    for slot in slots:
        t0 = slot.start
        src = slot.src_in
        while True:
            base = math.floor(src / loop + 1e-9) * loop
            boundary = base + loop
            if slot.src_out <= boundary + 1e-9:
                out.append(_reslice(slot, t0, slot.end, src - base,
                                    slot.src_out - base))
                break
            # Where the ramp reaches the boundary, measured from the slot's own
            # start — `solve_ramp` inverts the quadratic the speed ramp makes.
            cut_at = slot.start + solve_ramp(
                slot.speed_in, slot.speed_out, slot.duration,
                boundary - slot.src_in,
            )
            cut_at = max(t0 + 1e-9, min(slot.end, cut_at))
            out.append(_reslice(slot, t0, cut_at, src - base, loop))
            t0, src = cut_at, boundary
            if t0 >= slot.end - 1e-9:
                break
    return out


def _reslice(slot: Slot, t0: float, t1: float, src_in: float, src_out: float) -> Slot:
    """One piece of a split slot, with its share of everything.

    The transition's progress is taken from OUTPUT time, because that is what
    `blend` reads off the frame; the speeds are sampled off the parent's ramp
    so the piece continues the same curve rather than restarting it.
    """
    lo, hi = slot.blend_p0, slot.blend_p1
    whole = slot.duration or 1.0
    f0 = (t0 - slot.start) / whole
    f1 = (t1 - slot.start) / whole
    return Slot(
        start=t0, end=t1,
        # An exact boundary lands a hair either side of it in floating point,
        # and a negative trim start is an ffmpeg error rather than a rounding
        # detail.
        src_in=max(0.0, src_in), src_out=max(src_in, src_out),
        speed_in=slot.speed_at(t0), speed_out=slot.speed_at(t1),
        track=slot.track, from_track=slot.from_track,
        blend=slot.blend, bar=slot.bar, beat=slot.beat,
        beats=slot.beats * ((t1 - t0) / whole),
        section=slot.section and t0 == slot.start,
        blend_p0=lo + (hi - lo) * f0,
        blend_p1=lo + (hi - lo) * f1,
    )


def merge_slivers(slots: list[Slot], fps: int) -> list[Slot]:
    """Absorb any segment too short to survive the render into its neighbour.

    A branch asked for one frame collapses and takes most of the piece with it
    — see MIN_SEGMENT_FRAMES for the measurement. The slivers come from
    `split_at_wraps`, are a few milliseconds long, and the honest fix is to
    give their time to the segment next door rather than to draw them.

    **Which neighbour is not a free choice.** A sliver sits at a loop boundary
    by construction, and the segments on either side of that boundary read
    opposite ends of the loop — the one before is finishing at `loop`, the one
    after is starting at 0. Merging into the wrong one asks a segment to travel
    from 2.07 s to 0.04 s in a tenth of a second, which comes out as a negative
    speed and a backwards trim. So a sliver only ever joins the neighbour it is
    already source-continuous with, and the boundary stays where it was.

    The neighbour keeps the speed at its far end and has the near one re-solved,
    so the source position stays exactly continuous across the join; only the
    shape of a few milliseconds of curve moves.
    """
    if fps <= 0 or len(slots) < 2:
        return slots

    def too_short(slot: Slot) -> bool:
        """The renderer's own measure, not a duration in seconds.

        Both are rounded against the absolute timeline, so a segment can be
        0.0664 s and still be a perfectly renderable two frames. Comparing
        seconds against MIN_SEGMENT_FRAMES/fps called that one a sliver and
        merged a segment that never needed merging.
        """
        return segment_frames(slot.start, slot.end, fps) < MIN_SEGMENT_FRAMES

    if not any(too_short(s) for s in slots):
        return slots

    def joins(before: Slot, after: Slot) -> bool:
        """Do these two read one continuous stretch of source?"""
        return abs(after.src_in - before.src_out) < 1e-6

    # Forward: a sliver that continues the segment before it joins that one.
    out: list[Slot] = []
    for slot in slots:
        if out and too_short(slot) and joins(out[-1], slot):
            prev = out[-1]
            prev.end = slot.end
            prev.src_out = slot.src_out
            # The far end of the ramp comes from the sliver, because the
            # sliver's speeds are what the curve says at the new boundary.
            # Keeping the absorbing segment's own exit speed left it reading
            # the curve at a moment that is no longer its end, which put a
            # step of 0.30 back into the piece.
            prev.speed_out = slot.speed_out
            prev.blend_p1 = slot.blend_p1
            prev.beats += slot.beats
            _rebalance(prev)
            continue
        out.append(slot)

    # Backward: whatever is left is a sliver that opens a loop, so it belongs
    # to the segment after it instead.
    merged: list[Slot] = []
    for slot in reversed(out):
        if merged and too_short(slot) and joins(slot, merged[0]):
            nxt = merged[0]
            nxt.start = slot.start
            nxt.src_in = slot.src_in
            nxt.speed_in = slot.speed_in
            nxt.blend_p0 = slot.blend_p0
            nxt.beats += slot.beats
            nxt.section = nxt.section or slot.section
            _rebalance(nxt, at_start=True)
            continue
        merged.insert(0, slot)

    # Last resort: a sliver with no source-continuous neighbour at all. In
    # practice this is the tail after the final wrap, which has nothing after
    # it to join. It gives its time to the segment beside it and its few
    # milliseconds of source go with it — at the end of a piece the closing
    # fade to black is already running over that moment.
    return _absorb_leftovers(merged, fps)


def _absorb_leftovers(slots: list[Slot], fps: int) -> list[Slot]:
    out: list[Slot] = []
    for slot in slots:
        if out and segment_frames(slot.start, slot.end, fps) < MIN_SEGMENT_FRAMES:
            prev = out[-1]
            prev.end = slot.end          # its source is dropped, not its time
            prev.beats += slot.beats
            _rebalance(prev)
            continue
        out.append(slot)
    while (len(out) > 1
           and segment_frames(out[0].start, out[0].end, fps) < MIN_SEGMENT_FRAMES):
        first = out.pop(0)
        out[0].start = first.start
        out[0].beats += first.beats
        out[0].section = out[0].section or first.section
        _rebalance(out[0], at_start=True)
    return out


def _rebalance(slot: Slot, *, at_start: bool = False) -> None:
    """Re-solve ONE end of a slot's ramp so it consumes exactly what it spans.

    Source continuity is the invariant; the ramp is the thing allowed to bend
    to preserve it.

    **Which end is bent is the whole point.** A slot only ever needs
    rebalancing because it swallowed a sliver at a loop seam, and the seam is
    the one moment where the picture already jumps — the source restarts there,
    so a kink in the rate is hidden by a discontinuity the viewer is looking
    at anyway. Spreading the correction over both ends instead put half of it
    at the slot's *other* boundary, in the middle of continuous motion, which
    is precisely where it must not go. So the end nearest the seam takes all
    of it and the far end is left exactly where the curve put it.

    If the solution would run the picture backwards the slot goes flat instead,
    which is a rate this can always satisfy.
    """
    if slot.duration <= 0:
        return
    mean = (slot.src_out - slot.src_in) / slot.duration
    if mean <= 0:
        slot.speed_in = slot.speed_out = max(mean, 1e-3)
        return
    if at_start:
        entry = 2.0 * mean - slot.speed_out
        if entry < SPEED_MIN * 0.5:
            slot.speed_in = slot.speed_out = mean
        else:
            slot.speed_in = entry
        return
    exit_ = 2.0 * mean - slot.speed_in
    if exit_ < SPEED_MIN * 0.5:
        slot.speed_in = slot.speed_out = mean
    else:
        slot.speed_out = exit_


def _pick_transition(transition: str, rng: random.Random) -> str:
    if transition == "hart":
        return ""
    if transition == "mix":
        return rng.choice(_MIXABLE)
    return transition if transition in _BUILDERS else DEFAULT_TRANSITION


def wrap_times(slots: list[Slot], loop: float) -> list[float]:
    """Output times at which the shared source position crosses a loop boundary.

    Every one of them is a moment where the picture jumps from the loop's last
    frame back to its first — invisible if the render really loops, and worth
    marking on the timeline either way, because "it looks like it stutters at
    0:41" has exactly one cause and this is the list of candidates.
    """
    out: list[float] = []
    if loop <= 0:
        return out
    for slot in slots:
        if slot.speed <= 0:
            continue
        k = int(slot.src_in // loop) + 1
        while k * loop < slot.src_out - 1e-9:
            t = solve_ramp(slot.speed_in, slot.speed_out, slot.duration,
                           k * loop - slot.src_in)
            out.append(round(slot.start + t, 3))
            k += 1
    return out
