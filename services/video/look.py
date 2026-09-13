"""The look pass — one ffmpeg chain that corrects, grades and roughens.

This is what the film-grain pass grew into. Grain alone answered one complaint
(Wan2.2 and AnimateLCM come out plastic-smooth) and created another: the pass
was also the only thing standing between a render and delivery, so every other
correction had nowhere to live. Now there is one post-pass with seven dials,
rendered as a single filter chain into a single sibling file, and grain is the
last of the seven rather than the whole feature.

**Order is not arbitrary.** Corrections come first, because everything after
them is a look applied *to a corrected picture*; sharpening follows the
correction so it works on final contrast; halation and vignette are optical
effects that belong on the graded image; aberration is a lens artefact, so it
sits outside the optics; and grain is last, because film grain is in the print,
on top of everything the lens and the lab did:

    eq → colortemperature → unsharp → halation → vignette → rgbashift → noise

Every dial is an int on a UI scale and every one of them is a no-op at 0, which
is what lets the chain be assembled by concatenation without special cases.
Signed dials (contrast, saturation, temperature) run -100..100 with 0 neutral.


── Why the encoder settings changed ─────────────────────────────────────────

The old pass encoded at libx265 crf 30 with `-tune grain`, chosen for file
size. It measurably softened the picture, which is what prompted this rewrite.
Measured 2026-09-04 on a real 133-frame AnimateLCM render (816x1440, 27.7 MB),
grain strength 20. "picture" is SSIM against the untouched source after both
sides are gaussian-blurred, so the synthetic grain is taken out of the
comparison and what is left is how well the picture underneath survived:

    crf 30, tune=grain  (the old default)    6.7 MB    picture 0.9585
    crf 26, tune=grain                      17.2 MB    picture 0.9707
    crf 24, tune=grain                      24.2 MB    picture 0.9742
    crf 22, tune=grain (10-bit)             31.6 MB    picture 0.9761
    crf 20, tune=grain                      39.0 MB    picture 0.9774

Two separate effects were measured underneath that, and only one of them is
about grain:

  * The encode alone softens. With the noise filter switched off entirely,
    high-frequency energy against the source fell 6.9% at crf 30, 4.9% at
    crf 26, 4.1% at crf 24 and 3.0% at crf 20. That is the encoder, not the
    filter — there was no filter.

  * The noise costs a further ~1.4% of picture SSIM, and — this is the part
    that surprised — that cost is FLAT across crf 30 to 20. Noise is not
    starving the picture at low bitrates; it is simply a fixed price for
    adding noise.

So the fix is bits, not a cleverer filter, and the returns flatten after 26.
`-tune grain` is worth its ~15% size premium only when there is grain to keep:
without it at crf 30 the picture scored 0.9710 against tune's 0.9585, but the
grain was largely encoded away (high-frequency excess over the source fell from
+12.5% to +2.4%). So the tune is applied conditionally, and a look with no
grain in it encodes cleanly at a lower crf instead.

The honest headline: this pass now costs about 2.6x what it used to on the same
material. That is the price of the sharpness, and it was worth stating rather
than burying.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from core.subproc import communicate
from core.video_thumb import probe_video_duration

logger = logging.getLogger(__name__)

PREVIEW_SECONDS = 4.0

# ── Encoder ──────────────────────────────────────────────────────────────────
# See the module docstring for the measurements behind these two numbers. The
# split exists because `-tune grain` buys nothing when there is no grain: it
# tells the encoder to protect noise, and protecting noise that is not there
# just costs bits the picture could have had.
_CODEC = "libx265"
_CRF_WITH_GRAIN = "26"
_CRF_CLEAN = "22"
_PRESET = "medium"
# hvc1 rather than the default hev1 tag: QuickTime and Safari refuse to play
# HEVC in MP4 without it, and these files are watched in a browser.
_HEVC_TAG = "hvc1"

# ── Dial ranges ──────────────────────────────────────────────────────────────
# ffmpeg's own noise scale goes to 100, but past ~60 on the luma plane it is
# static rather than grain, so the slider maps onto 0-60 instead of 1:1.
NOISE_CEILING = 60

# unsharp's amount. 1.0 is already assertive on a diffusion render, whose
# "detail" is partly the model's own texture; past that it turns crunchy.
SHARPEN_MAX = 1.0

# eq's contrast is a multiplier around 0.5. ±0.35 is a firm grade and still
# short of the point where the blacks crush.
CONTRAST_SPAN = 0.35
# eq's saturation, 1.0 neutral. -100 lands at 0.15 rather than 0 so the
# extreme is "nearly monochrome", which is a look, rather than "greyscale",
# which is a different feature.
SATURATION_SPAN = 0.85
SATURATION_FLOOR = 0.15

# colortemperature is in kelvin around a neutral 6500. Warmer is a lower
# number in the filter's terms (it corrects *for* that temperature), so the
# mapping below inverts: a positive dial reads as a warmer picture.
TEMPERATURE_NEUTRAL = 6500
TEMPERATURE_SPAN = 2500

# Halation: how far the highlight bloom spreads, and how much of it is mixed
# back in. Blur radius matters more than opacity for whether it reads as
# "light in the lens" or as "the picture is out of focus".
HALATION_BLUR_MIN, HALATION_BLUR_MAX = 6.0, 26.0
HALATION_OPACITY_MAX = 0.55
# Luma below this contributes nothing to the bloom. Highlights only — bloom
# lifted off midtones is just a veil over the whole frame.
HALATION_THRESHOLD = 165

# vignette's angle. The filter's default is PI/5; going much past PI/3.2
# darkens the corners to black on a 9:16 frame.
VIGNETTE_MAX_ANGLE = 0.98          # radians, ~PI/3.2

# rgbashift, in pixels. Two is already visible on a 1080-wide frame; four is
# the most that still reads as a lens rather than as a broken file.
ABERRATION_MAX_PX = 4


def _clamp(value, lo: int, hi: int, default: int = 0) -> int:
    try:
        return max(lo, min(hi, int(round(float(value)))))
    except (TypeError, ValueError):
        return default


def clamp_strength(value: int | float | None) -> int:
    """UI strength onto 0-100. None/garbage reads as 0 (off).

    Kept under its old name because the grain dial is still `grain_strength`
    on the row and still 0-100; only its company changed.
    """
    return _clamp(value, 0, 100)


def _signed(value: int | float | None) -> int:
    return _clamp(value, -100, 100)


@dataclass(frozen=True)
class Look:
    """The seven dials, and how they become an ffmpeg chain.

    A Look with every dial at 0 renders nothing — `is_empty` says so, and the
    callers treat that as "remove the pass" rather than "encode a copy".
    """
    sharpen: int = 0          # 0..100
    contrast: int = 0         # -100..100
    saturation: int = 0       # -100..100
    temperature: int = 0      # -100..100, positive = warmer
    halation: int = 0         # 0..100
    vignette: int = 0         # 0..100
    aberration: int = 0       # 0..100
    grain: int = 0            # 0..100

    @classmethod
    def from_dict(cls, data: dict | None) -> "Look":
        """Build from whatever the client or the row carried.

        Unknown keys are dropped and missing ones default, so a Look stored by
        an older version stays loadable after a dial is added.
        """
        d = data or {}
        return cls(
            sharpen=clamp_strength(d.get("sharpen")),
            contrast=_signed(d.get("contrast")),
            saturation=_signed(d.get("saturation")),
            temperature=_signed(d.get("temperature")),
            halation=clamp_strength(d.get("halation")),
            vignette=clamp_strength(d.get("vignette")),
            aberration=clamp_strength(d.get("aberration")),
            grain=clamp_strength(d.get("grain")),
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def is_empty(self) -> bool:
        return not any(asdict(self).values())

    @property
    def has_grain(self) -> bool:
        return self.grain > 0

    def with_grain(self, strength: int) -> "Look":
        return replace(self, grain=clamp_strength(strength))

    # ── The chain ────────────────────────────────────────────────────────────

    def filter_chain(self) -> str:
        """The whole look as one ffmpeg filtergraph, or "" when nothing is set.

        Returned as a *simple* graph — one input, one output — so it can be
        passed to `-vf` even though halation needs a split and a blend. ffmpeg
        accepts labelled links there as long as the graph has a single open end
        at each side, which this always does.
        """
        before = self._correction() + self._sharpen()
        after = self._vignette() + self._aberration() + self._grain()
        halation = self._halation()
        if halation is None:
            return ",".join(before + after)
        return _splice_halation(before, after, halation)

    # Each stage returns a list so an inactive dial contributes nothing at all
    # — no `eq=contrast=1.0` no-ops cluttering the chain or costing a pass.

    def _correction(self) -> list[str]:
        out: list[str] = []
        eq: list[str] = []
        if self.contrast:
            eq.append(f"contrast={1.0 + self.contrast / 100 * CONTRAST_SPAN:.4f}")
        if self.saturation:
            if self.saturation >= 0:
                sat = 1.0 + self.saturation / 100 * SATURATION_SPAN
            else:
                sat = 1.0 + self.saturation / 100 * (1.0 - SATURATION_FLOOR)
            eq.append(f"saturation={sat:.4f}")
        if eq:
            out.append("eq=" + ":".join(eq))
        if self.temperature:
            # Inverted on purpose: the filter corrects *for* a temperature, so
            # telling it a low number warms the picture up.
            kelvin = TEMPERATURE_NEUTRAL - self.temperature / 100 * TEMPERATURE_SPAN
            out.append(f"colortemperature=temperature={kelvin:.0f}:mix=1:pl=0")
        return out

    def _sharpen(self) -> list[str]:
        if not self.sharpen:
            return []
        amount = self.sharpen / 100 * SHARPEN_MAX
        # Luma only. Sharpening chroma on 4:2:0 amplifies the subsampling
        # rather than any detail, and shows up as coloured fringing on edges.
        return [f"unsharp=5:5:{amount:.3f}:5:5:0.0"]

    def _vignette(self) -> list[str]:
        if not self.vignette:
            return []
        angle = self.vignette / 100 * VIGNETTE_MAX_ANGLE
        return [f"vignette=a={angle:.4f}"]

    def _aberration(self) -> list[str]:
        if not self.aberration:
            return []
        px = max(1, round(self.aberration / 100 * ABERRATION_MAX_PX))
        # Red out, blue in — the direction a real uncorrected lens disperses.
        return [f"rgbashift=rh={px}:bh=-{px}"]

    def _grain(self) -> list[str]:
        if not self.grain:
            return []
        # Luma plane only (`c0`): film grain lives in luminance, and noising
        # the chroma planes produces coloured speckle that reads as a
        # compression fault rather than as texture. `c0f=t` regenerates the
        # pattern every frame — without the temporal flag the grain freezes
        # into a static dirt overlay stuck to the lens.
        return [f"noise=c0s={round(self.grain * NOISE_CEILING / 100)}:c0f=t"]

    def _halation(self) -> tuple[float, float] | None:
        if not self.halation:
            return None
        t = self.halation / 100
        blur = HALATION_BLUR_MIN + t * (HALATION_BLUR_MAX - HALATION_BLUR_MIN)
        return blur, t * HALATION_OPACITY_MAX


def _splice_halation(
    before: list[str], after: list[str], halation: tuple[float, float],
) -> str:
    """Assemble the chain with the highlight bloom in its proper place.

    Halation is the one stage that is not a straight link — it needs the
    picture twice — so the linear runs on either side of it are spliced around
    a split/blend pair rather than joined by commas.
    """
    blur, opacity = halation
    head = ",".join(before) + "," if before else ""
    tail = "," + ",".join(after) if after else ""
    # Highlights are isolated by flooring everything below the threshold to
    # black, blurred wide, and screened back on. Screen rather than add: add
    # clips the highlights it is supposed to be blooming.
    return (
        f"{head}split[hl_base][hl_src];"
        f"[hl_src]lutyuv=y='if(gt(val,{HALATION_THRESHOLD}),val,16)',"
        f"gblur=sigma={blur:.2f}[hl_glow];"
        f"[hl_base][hl_glow]blend=all_mode=screen:all_opacity={opacity:.3f}"
        f"{tail}"
    )


# ── Presets ──────────────────────────────────────────────────────────────────
# Starting points, not destinations: every one of them lands in the same seven
# sliders, which stay editable afterwards. `klar` exists because the pass's
# own encode softens the picture slightly whatever else it does, and a
# sharpen-only look is the honest answer to "just undo that".

PRESETS: tuple[tuple[str, str, str, Look], ...] = (
    ("klar", "Klar",
     "Nur Korrektur: schärft und hebt den Kontrast leicht an. Für Renders, "
     "die sonst nichts brauchen.",
     Look(sharpen=35, contrast=12, saturation=8)),
    ("korn", "Korn",
     "Der alte Grain-Pass: nur Korn, sonst nichts angefasst.",
     Look(grain=30)),
    ("kino", "Kino",
     "Kräftiger Kontrast, leicht entsättigt, warmes Licht mit Halation und "
     "Vignette. Der Standard-Filmlook.",
     Look(sharpen=20, contrast=22, saturation=-10, temperature=15,
          halation=25, vignette=30, grain=25)),
    ("analog", "Analog",
     "Warm, weich, mit Lichthof und einem Hauch Linsenfehler. Viel Korn.",
     Look(sharpen=10, contrast=8, saturation=10, temperature=35,
          halation=45, vignette=22, aberration=20, grain=45)),
    ("bleach", "Bleach",
     "Harter Kontrast, stark entsättigt, kalt. Für Metall und Beton.",
     Look(sharpen=30, contrast=42, saturation=-55, temperature=-25,
          vignette=28, grain=20)),
    ("traum", "Traum",
     "Weiches Licht, starker Lichthof, kaum Kontrast. Nimmt Schärfe bewusst "
     "zurück statt sie zu geben.",
     Look(contrast=-12, saturation=18, temperature=10,
          halation=70, vignette=18, grain=15)),
)

PRESET_BY_KEY = {key: look for key, _, _, look in PRESETS}
DEFAULT_PRESET = "kino"


def preset_options() -> list[dict]:
    """The preset list for the frontend, values included so the sliders can
    jump to a preset without a second round trip."""
    return [
        {"key": key, "label": label, "hint": hint, "look": look.to_dict()}
        for key, label, hint, look in PRESETS
    ]


# ── Rendering ────────────────────────────────────────────────────────────────

def preview_window(duration: float, seconds: float = PREVIEW_SECONDS) -> tuple[float, float]:
    """(start, length) of the excerpt to grade for a preview.

    Taken from the middle: the opening frames of an i2v clip are the source
    still barely moving, which is the least representative place to judge how
    a look sits on actual motion. Clips shorter than the window are graded
    whole rather than producing a zero-length preview.

    A duration of 0 means the probe failed (probe_video_duration swallows its
    own errors); grade from the start for the full window and let ffmpeg stop
    at whatever the real end turns out to be.
    """
    if duration <= 0:
        return 0.0, seconds
    if duration <= seconds:
        return 0.0, duration
    return (duration - seconds) / 2.0, seconds


def encode_args(look: Look) -> list[str]:
    """Filter + encoder settings shared by the preview and the full render.

    Shared deliberately: if the preview encoded differently from the render,
    the look a user dials in on the preview would not be the look they get,
    which defeats the point of previewing at all.
    """
    args = ["-vf", look.filter_chain() or "null"]
    args += ["-c:v", _CODEC, "-tag:v", _HEVC_TAG]
    if look.has_grain:
        args += ["-tune", "grain", "-crf", _CRF_WITH_GRAIN]
    else:
        args += ["-crf", _CRF_CLEAN]
    args += ["-preset", _PRESET, "-pix_fmt", "yuv420p"]
    return args


def render_command(ffmpeg: str, src: Path, out: Path, look: Look) -> list[str]:
    return [
        ffmpeg, "-y",
        "-i", str(src),
        *encode_args(look),
        # Audio (an attached soundtrack, or the model's native track) is passed
        # through untouched; this pass only ever re-encodes the picture.
        "-c:a", "copy",
        "-movflags", "+faststart",
        str(out),
    ]


def preview_command(
    ffmpeg: str, src: Path, out: Path, look: Look, *, start: float, length: float,
) -> list[str]:
    return [
        ffmpeg, "-y",
        # -ss ahead of -i seeks by index rather than decoding up to the mark,
        # which is what keeps a preview to a couple of seconds.
        "-ss", f"{start:.3f}",
        "-t", f"{length:.3f}",
        "-i", str(src),
        *encode_args(look),
        # No audio: input seeking can land the audio a few ms off the video,
        # and a look preview is a picture judgement anyway.
        "-an",
        "-movflags", "+faststart",
        str(out),
    ]


async def render_look(
    src: Path, out: Path, look: Look, *, ffmpeg_path: str = "ffmpeg",
) -> None:
    """Grade the whole of `src` into `out`. Raises RuntimeError on failure."""
    await _run_ffmpeg(render_command(ffmpeg_path, src, out, look), label="look")


async def render_look_preview(
    src: Path,
    out: Path,
    look: Look,
    *,
    ffmpeg_path: str = "ffmpeg",
    seconds: float = PREVIEW_SECONDS,
) -> float:
    """Grade a short excerpt out of the middle of `src` into `out`.

    Returns the excerpt's length in seconds. Raises RuntimeError on failure.
    """
    duration = await probe_video_duration(src)
    start, length = preview_window(duration, seconds)
    await _run_ffmpeg(
        preview_command(ffmpeg_path, src, out, look, start=start, length=length),
        label="look_preview",
    )
    return length


async def _run_ffmpeg(cmd: list[str], *, label: str) -> None:
    logger.info("ffmpeg %s: %s", label, " ".join(str(c) for c in cmd))
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await communicate(proc)
    if proc.returncode != 0:
        tail = stderr.decode(errors="replace")[-600:]
        raise RuntimeError(f"ffmpeg {label} failed (rc={proc.returncode}): {tail}")
