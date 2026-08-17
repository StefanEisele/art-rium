"""Film grain over a single image — the still-image sibling of the video pass.

services/video/grain.py exists because Wan2.2 output is smooth to the point of
looking plastic. Generated stills have the same problem, and the same cure:
monochrome noise that puts texture back without touching colour, sharpness or
geometry.

Three decisions carried over from the video pass, for the same reasons:

- **Luma only.** The identical offset goes onto R, G and B, so grain changes
  how bright a pixel is and never what colour it is. Independent per-channel
  noise reads as a compression fault, not as film.
- **A sibling file, never an overwrite.** Any strength can be tried, undone,
  or re-tried; the source stays pristine. `strength=0` and the DELETE end up
  in the same teardown.
- **Always re-rendered from the source below it,** never from its own output,
  so changing the strength replaces the grain instead of stacking it.

Two decisions specific to stills:

- **Midtone weighting.** Grain is faded out towards pure black and pure white
  (`_EDGE_KEEP`). Film behaves this way, and without it deep shadows — which
  this library has a lot of — turn into visible noise mush.
- **The preview is sigma-corrected.** Grain is per-pixel, so a 4096 px render
  and a 1200 px preview shown at the *same* size on screen do not look alike:
  downscaling averages neighbouring samples and quietly removes most of the
  noise. `preview_bytes` scales sigma by the same factor it scales the image,
  which makes the two match at display size — the point of previewing at all.

Pillow-only, like services/image/enhance.py: `Image.effect_noise` gives a
Gaussian noise field, and the rest is one composite and one point LUT.
"""
from __future__ import annotations

import asyncio
import logging
from io import BytesIO
from pathlib import Path

from PIL import Image as PILImage
from PIL import ImageChops

logger = logging.getLogger(__name__)

# The UI slider runs 0-100 so its whole travel is useful; the one-click default
# is a tasteful amount rather than the maximum, which is reserved for a
# deliberately coarse look.
STRENGTH_DEFAULT = 30
STRENGTH_MAX = 100

# Gaussian sigma, in 8-bit levels, at strength 100. Past ~26 the noise stops
# reading as grain and starts reading as a broken sensor.
SIGMA_CEILING = 24.0

# How much grain survives at pure black / pure white, relative to the midtones.
# Not 0: a hard falloff makes the grain stop at an edge, which is more visible
# than the grain itself.
_EDGE_KEEP = 0.35

# Preview edge, and the floor under the sigma correction below. A very large
# original scales sigma down so far that the preview would show no grain at
# all, which is useless even if it is technically faithful.
PREVIEW_EDGE = 1200
_MIN_PREVIEW_SIGMA = 1.2


def clamp_strength(value: int | float | None) -> int:
    """UI strength onto 0-STRENGTH_MAX. None/garbage reads as 0 (grain off)."""
    try:
        return max(0, min(STRENGTH_MAX, int(round(float(value)))))
    except (TypeError, ValueError):
        return 0


def sigma_for(strength: int) -> float:
    """Noise sigma in 8-bit levels for `strength` on the 0-100 UI scale."""
    return clamp_strength(strength) * SIGMA_CEILING / 100.0


def preview_sigma(strength: int, *, source_edge: int, preview_edge: int) -> float:
    """Sigma for a preview rendered at `preview_edge` off a `source_edge` original.

    Scaled by the resampling ratio so the preview and the eventual full-size
    render carry the *same* apparent grain once the browser has fitted either
    of them into the same box. Enlarging never happens (the preview is a
    downscale or a no-op), so the ratio is capped at 1.
    """
    ratio = min(1.0, preview_edge / max(1, source_edge))
    full = sigma_for(strength)
    if full <= 0:
        return 0.0
    return max(_MIN_PREVIEW_SIGMA, full * ratio)


def _weight_lut() -> list[int]:
    """Per-luma grain weight: full in the midtones, `_EDGE_KEEP` at the ends."""
    return [
        max(0, min(255, round(255 * (_EDGE_KEEP + (1.0 - _EDGE_KEEP) * (1.0 - abs(v - 128) / 128.0)))))
        for v in range(256)
    ]


def apply_grain(img: PILImage.Image, sigma: float) -> PILImage.Image:
    """Add monochrome grain of `sigma` to `img`, preserving any alpha channel.

    The noise field is not seeded — Pillow does not expose one — so two renders
    at the same strength differ in their exact grain, the way two film frames
    do. Nothing downstream depends on them being identical.
    """
    if sigma <= 0:
        return img.copy()

    alpha = img.getchannel("A") if img.mode in ("RGBA", "LA") else None
    base = img.convert("RGB")

    # One noise plane, replicated across the channels: same offset on R, G and
    # B is what makes this luminance grain instead of colour speckle.
    plane = PILImage.effect_noise(base.size, sigma)
    noise = PILImage.merge("RGB", (plane, plane, plane))

    # effect_noise centres on 128, so subtracting it turns the field into a
    # signed offset. ImageChops clips once, at the end.
    grained = ImageChops.add(base, noise, scale=1.0, offset=-128)

    weight = base.convert("L").point(_weight_lut())
    out = PILImage.composite(grained, base, weight)

    if alpha is not None:
        out.putalpha(alpha)
    return out


# ── File-level entry points ──────────────────────────────────────────────────


def _grain_file_sync(src: Path, dest: Path, strength: int) -> None:
    with PILImage.open(src) as opened:
        opened.load()
        out = apply_grain(opened, sigma_for(strength))
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Write beside the target and move into place, as the enhance pass does: a
    # half-written PNG that the gallery then serves would look like a corrupt
    # rendition rather than an interrupted one.
    tmp = dest.with_suffix(dest.suffix + ".part")
    out.save(tmp, "PNG", optimize=True)
    tmp.replace(dest)


async def grain_file(src: Path, dest: Path, strength: int = STRENGTH_DEFAULT) -> None:
    """Render the grained rendition of `src` into `dest`.

    `src` is the rendition *below* the grain — the enhanced file when there is
    one, the original otherwise — and never a previous grain render.
    """
    await asyncio.to_thread(_grain_file_sync, src, dest, strength)
    logger.info("Grained %s → %s (strength=%d)", src.name, dest.name, strength)


def _preview_sync(src: Path, strength: int, max_edge: int) -> bytes:
    with PILImage.open(src) as opened:
        source_edge = max(opened.size)
        img = opened.convert("RGB")
        img.thumbnail((max_edge, max_edge), PILImage.LANCZOS)
    sigma = preview_sigma(strength, source_edge=source_edge, preview_edge=max_edge)
    out = apply_grain(img, sigma)
    buf = BytesIO()
    # Quality 92 rather than the enhance preview's 88: JPEG's chroma-subsampled
    # DCT is exactly the wrong tool for fine noise, and at 88 the artefacts are
    # loud enough to be mistaken for the grain being judged.
    out.save(buf, "JPEG", quality=92, optimize=True)
    return buf.getvalue()


async def preview_bytes(src: Path, strength: int, max_edge: int = PREVIEW_EDGE) -> bytes:
    """A downscaled JPEG of what `strength` would do, for the live slider."""
    return await asyncio.to_thread(_preview_sync, src, strength, max_edge)
