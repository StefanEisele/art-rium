"""Auto-enhance ("Zauberstab") — one click, corrections derived from the image.

Modelled on what Apple's Photos wand actually does. Apple's own description of
the flow is: analyse the tone curve via the histogram to catch backlit or
underexposed shots, detect faces, then set *initial values* for Exposure,
Brilliance, Highlights, Shadows, Contrast, Brightness, Black Point, Saturation
and Vibrance — and finally expose one macro slider that drives all of them
together. Core Image's older public API (`autoAdjustmentFilters`) shows the
same skeleton: CIToneCurve (contrast), CIHighlightShadowAdjust (local
shadow/highlight), CIVibrance (saturation that spares already-saturated
colour), plus two face filters.

So the wand is not a fixed filter — it is *analysis → per-image amounts →
one macro strength*. That is exactly the shape implemented here:

    analyze(img)            → Analysis    what the picture is (measurements)
    plan(analysis, strength) → Adjustments what to do about it (pure, tested)
    apply(img, adjustments)  → Image      do it

`plan` is deliberately pure and separate: it is where every judgement call
lives, so the calibration can be tested and tuned without touching pixels.

**Face detection is the one part deliberately left out.** It exists in Apple's
pipeline to protect skin tones; this library is generated painterly art, where
a face is a rendering, not a person to keep flattering — and a false positive
would push a colour cast onto an image whose palette is the whole point.

Calibration is *art-safe* rather than Apple-faithful, at the user's request.
Same measurements, capped amounts: Vibrance instead of flat Saturation, tight
caps on white balance, and no correction at all when the picture is already
fine. A deliberately muted rust/blue frame must come out of this still muted —
just cleaner. `STRENGTH_DEFAULT` is the one-click amount; the UI slider scales
every amount linearly from there, which is what Apple's macro slider does.

Pillow-only by design: it is the one image library the project already ships,
and every operation here reduces to a 256-entry LUT, one blurred mask, or an
HSV pass. No numpy, no OpenCV.
"""
from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path

from PIL import Image as PILImage
from PIL import ImageChops, ImageFilter, ImageStat

logger = logging.getLogger(__name__)

# ── Macro strength ───────────────────────────────────────────────────────────
# The UI slider runs 0-150 %, defaulting to 100 % = "what the analysis asked
# for". Above 100 % the same amounts are simply pushed further, which is how
# Apple's slider behaves past its auto point.
STRENGTH_DEFAULT = 100
STRENGTH_MAX = 150

# Analysis runs on a downscaled copy: histograms and channel means are
# statistics, and 512 px carries them exactly as well as 2048 px does at 1/16
# the pixels. Kept above the thumbnail size so a cached thumbnail is never
# mistaken for a valid analysis source.
ANALYSIS_EDGE = 512

# ── Caps (the art-safe calibration) ──────────────────────────────────────────
# Every one of these is a ceiling on a *correction*, not a fixed amount. An
# image that needs nothing gets nothing; these only bound the worst case.

# Black/white point: how many levels the ends may be pulled in. Apple's Black
# Point can clip hard; 12/18 levels is roughly "remove the haze", not "crush".
_MAX_BLACK_SHIFT = 12
_MAX_WHITE_SHIFT = 18
# Only chase the clipping point part of the way. Landing exactly on the
# measured percentile makes every image end up with identical ends, which is
# how auto-levels flattens a series into sameness.
_LEVEL_APPROACH = 0.8
# Percentiles that count as "the ends". Not 0/100: a handful of stray pixels
# (a specular dot, a dead-black corner) would otherwise define the whole range.
_LOW_PERCENTILE = 0.004
_HIGH_PERCENTILE = 0.996

# Exposure, expressed as gamma so highlights survive.
#
# The band is the art-safe part, and it was measured rather than guessed. A
# 12-image spread across this library reads mean luma 57-117 (median ~92):
# these are deliberately dim, moody frames. Apple's photographic target of
# ~122 would therefore declare *every single image in the library*
# underexposed and haul it up 20-50 % — which is precisely the "overrides a
# deliberately muted palette" failure this calibration exists to avoid.
#
# So exposure is corrected only towards the nearest edge of a wide acceptable
# band, and left alone inside it. Only genuine underexposure gets rescued.
_MEAN_BAND = (78.0, 168.0)
_MIN_GAMMA = 0.85   # brightening ceiling
_MAX_GAMMA = 1.10   # darkening ceiling

# Local tone mapping = Apple's "Brilliance". Strongest single lever here, and
# the one that makes the result look considered rather than levels-stretched.
# The thresholds subtract the amount of shadow/highlight that is *normal* for
# this library (measured: 20-55 % of pixels below 64 is the everyday case),
# so the lift responds to genuinely blocked-up areas instead of firing at full
# strength on every frame.
_SHADOW_FLOOR = 0.30
_HIGHLIGHT_FLOOR = 0.05
_MAX_SHADOW_LIFT = 0.28
_MAX_HIGHLIGHT_RECOVERY = 0.25

# S-curve contrast, applied only in proportion to how flat the image measures.
_MAX_CONTRAST = 0.22

# Vibrance, weighted per-pixel against existing saturation (see _vibrance_lut).
# Capped low: measured at 0.38 this pushed mean saturation up by a quarter
# (75 → 91), which is a look change, not a clean-up.
#
# `_SATURATION_ENOUGH` is the same dead-zone idea as `_MEAN_BAND`: above it the
# picture has all the colour it needs and gets none added. Without it every
# image in the library picked up a little vibrance, which is precisely the
# "applies a look to everything" behaviour being avoided here.
_MAX_VIBRANCE = 0.16
_SATURATION_ENOUGH = 85.0

# White balance. ±4 % per channel is enough to pull a genuinely grey-drifted
# render back to neutral and far too little to neutralise an intentional
# palette — which is the entire point of the tight cap here.
_MAX_WB_GAIN = 0.04

# Unsharp mask = Apple's "Definition"/"Sharpness", and the one adjustment that
# must not be applied unconditionally: sharpening an already-crisp render just
# adds halos. Driven by measured acutance instead. Across a 12-image spread of
# this library acutance runs 2.2 (soft, atmospheric) to 13.8 (hard-edged
# graphic), so 8.0 is the point above which an image is treated as crisp
# enough to leave alone.
_ACUTANCE_CRISP = 8.0
_MAX_DEFINITION = 0.5
_UNSHARP_RADIUS = 1.6
_UNSHARP_THRESHOLD = 3


@dataclass(frozen=True, slots=True)
class Analysis:
    """What the picture measures — no judgement, just numbers."""
    black_point: int        # luma level at _LOW_PERCENTILE
    white_point: int        # luma level at _HIGH_PERCENTILE
    mean: float             # mean luma, 0-255
    shadow_mass: float      # fraction of pixels below 64
    highlight_mass: float   # fraction above 208
    saturation: float       # mean HSV S, 0-255
    acutance: float         # mean |luma - blur(luma)|; micro-contrast, i.e. how crisp
    channel_means: tuple[float, float, float]   # R, G, B means over midtones

    @property
    def span(self) -> int:
        """Occupied tonal range. 255 = uses the full scale, 60 = very flat."""
        return max(1, self.white_point - self.black_point)


@dataclass(frozen=True, slots=True)
class Adjustments:
    """What to do about it. All amounts already scaled by the macro strength.

    Serialised into the DB alongside the rendition so the UI can show what the
    wand decided — the equivalent of watching Apple's sliders jump.
    """
    black_point: int                    # input level to map to 0
    white_point: int                    # input level to map to 255
    gamma: float                        # <1 brightens, >1 darkens
    contrast: float                     # 0-1 S-curve amount
    shadows: float                      # 0-1 local lift
    highlights: float                   # 0-1 local recovery
    vibrance: float                      # 0-1
    wb_gains: tuple[float, float, float]  # per-channel multipliers
    definition: float                   # 0-1 unsharp amount

    @property
    def is_noop(self) -> bool:
        """True when the analysis found nothing worth doing.

        Worth having explicitly: an already-graded image should cost no file
        and no rendition row, and the UI should be able to say so rather than
        silently producing a byte-identical copy.
        """
        return (
            self.black_point == 0
            and self.white_point == 255
            and abs(self.gamma - 1.0) < 0.01
            and self.contrast < 0.01
            and self.shadows < 0.01
            and self.highlights < 0.01
            and self.vibrance < 0.01
            and self.definition < 0.01
            and all(abs(g - 1.0) < 0.005 for g in self.wb_gains)
        )

    def as_dict(self) -> dict:
        d = asdict(self)
        d["wb_gains"] = list(self.wb_gains)
        return d


def clamp_strength(value: int | float | None) -> int:
    """UI strength onto 0-STRENGTH_MAX. None/garbage reads as the default.

    Garbage falls back to the default rather than to 0, because this value
    arrives from a query string: a typo should still enhance the image, not
    silently hand back the original and look like the feature is broken.
    """
    try:
        return max(0, min(STRENGTH_MAX, int(round(float(value)))))
    except (TypeError, ValueError):
        return STRENGTH_DEFAULT


# ── Analysis ─────────────────────────────────────────────────────────────────


def _percentile_level(histogram: list[int], fraction: float) -> int:
    """Luma level below which `fraction` of the pixels sit."""
    total = sum(histogram)
    if total <= 0:
        return 0
    target = total * fraction
    running = 0
    for level, count in enumerate(histogram):
        running += count
        if running >= target:
            return level
    return 255


def analyze(img: PILImage.Image) -> Analysis:
    """Measure `img`. Cheap: runs on a downscaled RGB copy."""
    small = img.convert("RGB")
    small.thumbnail((ANALYSIS_EDGE, ANALYSIS_EDGE), PILImage.LANCZOS)

    luma = small.convert("L")
    histogram = luma.histogram()
    total = max(1, sum(histogram))

    # Midtone-only channel means for white balance. Including clipped darks and
    # blown highlights biases grey-world towards whatever the extremes happen
    # to be — a black border alone would read as a colour cast.
    midtones = luma.point(lambda v: 255 if 48 <= v <= 208 else 0)
    if sum(midtones.histogram()[255:]) > total * 0.05:
        channel_means = tuple(ImageStat.Stat(small, mask=midtones).mean[:3])
    else:
        # Almost no midtones (a near-black or blown frame): grey-world has
        # nothing to stand on, so report neutral and let plan() skip WB.
        channel_means = (1.0, 1.0, 1.0)

    # Acutance: how far the image departs from its own blur. High = crisp,
    # low = soft. Measured on the downscaled copy, so it describes structure
    # rather than pixel-level noise, which is the scale sharpening acts on.
    acutance = ImageStat.Stat(
        ImageChops.difference(luma, luma.filter(ImageFilter.GaussianBlur(radius=2)))
    ).mean[0]

    return Analysis(
        black_point=_percentile_level(histogram, _LOW_PERCENTILE),
        white_point=_percentile_level(histogram, _HIGH_PERCENTILE),
        mean=ImageStat.Stat(luma).mean[0],
        shadow_mass=sum(histogram[:64]) / total,
        highlight_mass=sum(histogram[208:]) / total,
        saturation=ImageStat.Stat(small.convert("HSV")).mean[1],
        acutance=acutance,
        channel_means=channel_means,  # type: ignore[arg-type]
    )


# ── Planning (pure — this is the calibration) ────────────────────────────────


def plan(a: Analysis, strength: int = STRENGTH_DEFAULT) -> Adjustments:
    """Turn measurements into capped amounts. Pure function; no pixels."""
    k = clamp_strength(strength) / 100.0
    if k <= 0:
        return Adjustments(0, 255, 1.0, 0.0, 0.0, 0.0, 0.0, (1.0, 1.0, 1.0), 0.0)

    # ── Black / white point ──────────────────────────────────────────────
    # Only pull in an end that is actually unused. `_LEVEL_APPROACH` leaves a
    # little headroom so nothing that was visible becomes clipped.
    black = min(_MAX_BLACK_SHIFT, int(a.black_point * _LEVEL_APPROACH * k))
    white_gap = max(0, 255 - a.white_point)
    white = 255 - min(_MAX_WHITE_SHIFT, int(white_gap * _LEVEL_APPROACH * k))

    # ── Exposure as gamma ────────────────────────────────────────────────
    # Inside the acceptable band: nothing. Outside it: solve mean^gamma = the
    # nearest band edge in normalised space, then clamp. Guard the degenerate
    # ends — a fully black or fully white frame carries no exposure
    # information to act on.
    low, high = _MEAN_BAND
    target = low if a.mean < low else high if a.mean > high else None
    if target is not None and 4.0 < a.mean < 251.0:
        gamma = math.log(target / 255.0) / math.log(a.mean / 255.0)
        gamma = 1.0 + (gamma - 1.0) * k        # scale the *correction*, not the exponent
        gamma = max(_MIN_GAMMA, min(_MAX_GAMMA, gamma))
    else:
        gamma = 1.0

    # ── Contrast, in proportion to flatness ──────────────────────────────
    # A picture already spanning the scale gets nothing; the S-curve only
    # exists to rescue images whose tones sit in a narrow band.
    flatness = max(0.0, 1.0 - a.span / 200.0)
    contrast = min(_MAX_CONTRAST, flatness * _MAX_CONTRAST * k)

    # ── Brilliance: local shadow lift + highlight recovery ────────────────
    # Driven by how much of the frame sits at an end *beyond what is normal
    # here*. A low-key frame is a look; a low-key frame where two thirds of
    # the pixels are blocked up is a frame with detail hiding in it.
    shadows = min(_MAX_SHADOW_LIFT, max(0.0, a.shadow_mass - _SHADOW_FLOOR) * 1.2 * k)
    highlights = min(_MAX_HIGHLIGHT_RECOVERY, max(0.0, a.highlight_mass - _HIGHLIGHT_FLOOR) * 1.2 * k)

    # ── Vibrance ─────────────────────────────────────────────────────────
    # Scaled down by what is already there: a saturated frame gets almost
    # nothing, which is what separates Vibrance from Saturation.
    headroom = max(0.0, (_SATURATION_ENOUGH - a.saturation) / _SATURATION_ENOUGH)
    vibrance = min(_MAX_VIBRANCE, headroom * _MAX_VIBRANCE * k)

    # ── White balance (grey-world, tightly capped) ───────────────────────
    # Faded further the more saturated the image is: in a strongly toned
    # frame the "cast" is the palette, and correcting it would be vandalism.
    r, g, b = a.channel_means
    grey = (r + g + b) / 3.0
    if grey > 1.0:
        trust = max(0.0, 1.0 - a.saturation / 130.0) * k
        # Clamp the *final* deviation, not the pre-scaled one: `k` reaches 1.5
        # at full slider travel, and clamping first let a 4 % cap leave here
        # as a 6 % shift.
        wb = tuple(
            1.0 + max(-_MAX_WB_GAIN, min(_MAX_WB_GAIN, (grey / c - 1.0) * trust))
            if c > 1.0 else 1.0
            for c in (r, g, b)
        )
    else:
        wb = (1.0, 1.0, 1.0)

    # ── Definition ───────────────────────────────────────────────────────
    # Only what the picture is missing: a crisp render gets nothing, which is
    # what keeps `is_noop` honest and keeps halos off already-sharp edges.
    softness = max(0.0, 1.0 - a.acutance / _ACUTANCE_CRISP)
    definition = min(_MAX_DEFINITION, softness * k)

    return Adjustments(
        black_point=black,
        white_point=max(black + 1, white),
        gamma=gamma,
        contrast=contrast,
        shadows=shadows,
        highlights=highlights,
        vibrance=vibrance,
        wb_gains=wb,  # type: ignore[arg-type]
        definition=definition,
    )


# ── LUT builders ─────────────────────────────────────────────────────────────


def _s_curve(x: float, amount: float) -> float:
    """Symmetric contrast curve on 0-1, pinned at 0, 0.5 and 1.

    A sine bow rather than the usual cubic: because it is pinned at both ends
    it cannot push a tone that was inside the range out of it, and it stays
    monotone (slope 1 - 2·amount at the ends) for every amount this module
    allows — so no two input tones can ever collapse onto one output tone.
    """
    return x - amount * math.sin(2.0 * math.pi * x) / math.pi


def build_tone_lut(adj: Adjustments) -> list[int]:
    """One 768-entry LUT folding white balance, levels, gamma and contrast.

    All four are per-channel monotone point operations, so they compose into a
    single table and cost one `Image.point()` pass over the pixels no matter
    how many of them are active.
    """
    lut: list[int] = []
    span = max(1, adj.white_point - adj.black_point)
    for gain in adj.wb_gains:
        for level in range(256):
            v = level * gain                                  # white balance
            v = (v - adj.black_point) / span                  # levels
            v = max(0.0, min(1.0, v))
            v = v ** adj.gamma                                # exposure
            if adj.contrast > 0:
                v = _s_curve(v, adj.contrast)                 # contrast
            lut.append(max(0, min(255, round(v * 255))))
    return lut


def _vibrance_lut(amount: float) -> list[int]:
    """Saturation curve that spares already-saturated pixels.

    Peaks in the middle and falls to nothing at both ends: unsaturated pixels
    stay unsaturated (no colour invented out of grey), fully saturated ones
    stay put (no clipping into poster colours). Applied to the S channel of
    HSV, which is where "how colourful is this pixel" actually lives.
    """
    return [
        max(0, min(255, round(s + amount * s * (1.0 - s / 255.0))))
        for s in range(256)
    ]


def _gamma_lut(exponent: float) -> list[int]:
    """A 768-entry (3-band) gamma table.

    Tripled because the shadow/highlight copies are blended over an RGB image:
    applying the same curve to all three channels keeps the lift colour-neutral,
    which is what Apple means by Brilliance being "color neutral".
    """
    band = [max(0, min(255, round(255 * (v / 255.0) ** exponent))) for v in range(256)]
    return band * 3


def _shadow_lut(amount: float) -> list[int]:
    """Curve for the lifted copy blended in over the shadow mask."""
    return _gamma_lut(1.0 - amount * 0.55)


def _highlight_lut(amount: float) -> list[int]:
    """Curve for the pulled-down copy blended in over the highlight mask."""
    return _gamma_lut(1.0 + amount * 0.75)


# ── Rendering ────────────────────────────────────────────────────────────────


def _luma_mask(img: PILImage.Image) -> PILImage.Image:
    """Blurred luminance, used to make shadow/highlight work *local*.

    A large-radius Gaussian is what separates "Brilliance" from a plain curve:
    it means a dark pixel inside a bright area is left alone, while a dark
    *region* gets lifted. Blurring a 1/8-scale copy and scaling back up costs a
    fraction of the full-size blur and is visually identical at this radius —
    the mask is deliberately soft, so its own resampling error cannot show.
    """
    w, h = img.size
    small = img.convert("L").resize((max(1, w // 8), max(1, h // 8)), PILImage.LANCZOS)
    small = small.filter(ImageFilter.GaussianBlur(radius=4))
    return small.resize((w, h), PILImage.BILINEAR)


def apply(img: PILImage.Image, adj: Adjustments) -> PILImage.Image:
    """Apply `adj` to `img`, preserving any alpha channel."""
    alpha = img.getchannel("A") if img.mode in ("RGBA", "LA") else None
    out = img.convert("RGB")

    out = out.point(build_tone_lut(adj))

    if adj.shadows > 0.01 or adj.highlights > 0.01:
        mask = _luma_mask(out)
        if adj.shadows > 0.01:
            # Mask weight ramps from full at black to nothing at mid-grey, so
            # the lift lands only where there is actually shadow.
            weight = mask.point(
                lambda v: max(0, min(255, round(255 * adj.shadows * max(0.0, 1.0 - v / 128.0))))
            )
            out = PILImage.composite(out.point(_shadow_lut(adj.shadows)), out, weight)
        if adj.highlights > 0.01:
            weight = mask.point(
                lambda v: max(0, min(255, round(255 * adj.highlights * max(0.0, (v - 128.0) / 127.0))))
            )
            out = PILImage.composite(out.point(_highlight_lut(adj.highlights)), out, weight)

    if adj.vibrance > 0.01:
        hsv = out.convert("HSV")
        h, s, v = hsv.split()
        out = PILImage.merge("HSV", (h, s.point(_vibrance_lut(adj.vibrance)), v)).convert("RGB")

    if adj.definition > 0.01:
        out = out.filter(ImageFilter.UnsharpMask(
            radius=_UNSHARP_RADIUS,
            percent=round(60 * adj.definition),
            threshold=_UNSHARP_THRESHOLD,
        ))

    if alpha is not None:
        out.putalpha(alpha)
    return out


def enhance_image(img: PILImage.Image, strength: int = STRENGTH_DEFAULT) -> tuple[PILImage.Image, Adjustments]:
    """Analyse, plan and apply in one call. The whole wand, minus the file I/O."""
    adj = plan(analyze(img), strength)
    return apply(img, adj), adj


# ── File-level entry points ──────────────────────────────────────────────────


def _enhance_file_sync(src: Path, dest: Path, strength: int) -> dict:
    with PILImage.open(src) as opened:
        opened.load()
        out, adj = enhance_image(opened, strength)
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Write beside the target and move into place: a half-written PNG that the
    # gallery then serves would look like a corrupted enhancement rather than
    # an interrupted one.
    tmp = dest.with_suffix(dest.suffix + ".part")
    out.save(tmp, "PNG", optimize=True)
    tmp.replace(dest)
    return adj.as_dict()


async def enhance_file(src: Path, dest: Path, strength: int = STRENGTH_DEFAULT) -> dict:
    """Render the enhanced rendition of `src` into `dest`.

    Always reads the *original*, never a previous enhancement — the same rule
    the video grain pass follows, so re-running at a different strength
    replaces the correction instead of stacking it. Returns the applied
    amounts for display/persistence.
    """
    adj = await asyncio.to_thread(_enhance_file_sync, src, dest, strength)
    logger.info("Enhanced %s → %s (strength=%d)", src.name, dest.name, strength)
    return adj


def _preview_sync(src: Path, strength: int, max_edge: int) -> bytes:
    with PILImage.open(src) as opened:
        img = opened.convert("RGB")
        img.thumbnail((max_edge, max_edge), PILImage.LANCZOS)
    out, _ = enhance_image(img, strength)
    buf = BytesIO()
    out.save(buf, "JPEG", quality=88, optimize=True)
    return buf.getvalue()


async def preview_bytes(src: Path, strength: int, max_edge: int = 1200) -> bytes:
    """A downscaled JPEG of what `strength` would do, for the live slider.

    Deliberately runs the *same* analyse→plan→apply chain as the real render
    rather than a cheaper approximation, so the preview a user judges cannot
    drift from the file they then get. It analyses the downscaled copy, which
    is what `analyze` does internally anyway.
    """
    return await asyncio.to_thread(_preview_sync, src, strength, max_edge)
