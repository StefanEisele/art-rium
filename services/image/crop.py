"""Zuschnitt — a picture's own framing, chosen in the gallery.

Pure geometry: no resampling, no colour, only the pixels inside a box. That is
what decides where it sits in the rendition stack (services/image/rendition.py):

    original  →  _upscaled  →  _crop  →  _enhanced  →  _grain

- **Above the upscale.** The upscale costs GPU minutes and a crop costs
  milliseconds, and the expensive pass must never be invalidated by a cheap
  one — so re-framing a picture re-cuts the existing upscale instead of asking
  for a new one, and upscaling a cropped picture re-cuts the result.
- **Below the wand and the grain.** The wand's analysis then measures the
  picture viewers actually get — a black border that was cut away no longer
  drags its black point — and the grain lands on the delivered pixels. Both are
  re-rendered on top whenever the crop changes, exactly as they already are
  when the upscale underneath them changes.

The box is kept as fractions of the rendition it is cut from (`x, y, w, h` in
0..1), not as pixels, so one framing lands on the original and on a 2x upscale
alike and survives the upscale being added or removed. The aspect the box was
drawn at travels with it — "free", "original" or "W:H" — and a fixed aspect is
re-imposed on the *rounded* pixel box, so a 4:5 crop comes out exactly 4:5 at
whatever size it is cut from instead of 4:5 give or take a pixel.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path

from PIL import Image as PILImage

logger = logging.getLogger(__name__)

ASPECT_FREE = "free"
ASPECT_ORIGINAL = "original"
_ASPECT_RE = re.compile(r"^(\d{1,5}):(\d{1,5})$")

# Below this the "picture" is a smear of a few pixels. The editor stops long
# before it; this only guards the endpoint against a nonsense box.
MIN_EDGE_PX = 8

# Within this many pixels of every edge the box is the whole frame, which is
# no crop at all. Absorbs the float noise of a box dragged onto the edges.
_FULL_FRAME_SLACK_PX = 1


@dataclass(frozen=True)
class CropBox:
    """A framing as fractions of the rendition it is cut from."""
    x: float
    y: float
    w: float
    h: float
    aspect: str = ASPECT_FREE

    def as_dict(self) -> dict:
        return {
            "x": round(self.x, 6), "y": round(self.y, 6),
            "w": round(self.w, 6), "h": round(self.h, 6),
            "aspect": self.aspect,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CropBox":
        return normalize(d.get("x", 0), d.get("y", 0), d.get("w", 1), d.get("h", 1),
                         d.get("aspect"))


def clean_aspect(value: str | None) -> str:
    """"free", "original" or a positive "W:H"; anything else reads as free."""
    if value in (ASPECT_FREE, ASPECT_ORIGINAL):
        return value
    m = _ASPECT_RE.match(str(value or ""))
    if m and int(m.group(1)) > 0 and int(m.group(2)) > 0:
        return f"{int(m.group(1))}:{int(m.group(2))}"
    return ASPECT_FREE


def aspect_ratio(aspect: str, src_w: int, src_h: int) -> float | None:
    """Width / height the crop must have in pixels, or None when it is free."""
    if aspect == ASPECT_ORIGINAL:
        return src_w / src_h if src_w > 0 and src_h > 0 else None
    m = _ASPECT_RE.match(aspect or "")
    return int(m.group(1)) / int(m.group(2)) if m else None


def _unit(v, default: float) -> float:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(v) else max(0.0, min(1.0, v))


def normalize(x, y, w, h, aspect: str | None = None) -> CropBox:
    """Clamp a client's box into the frame: every value a fraction, the box
    entirely inside the picture. A box hanging over an edge is trimmed rather
    than slid — what the client drew is what it gets, minus what isn't there."""
    x, y = _unit(x, 0.0), _unit(y, 0.0)
    w = min(_unit(w, 1.0), 1.0 - x)
    h = min(_unit(h, 1.0), 1.0 - y)
    return CropBox(x, y, w, h, clean_aspect(aspect))


def pixel_box(src_w: int, src_h: int, box: CropBox) -> tuple[int, int, int, int]:
    """(left, top, right, bottom) of `box` on a `src_w` x `src_h` source."""
    width = min(src_w, max(1, round(box.w * src_w)))
    height = min(src_h, max(1, round(box.h * src_h)))
    ratio = aspect_ratio(box.aspect, src_w, src_h)
    if ratio:
        # The fractions were drawn on whatever the browser showed; the shape
        # is what was asked for, so it is re-derived at the real size.
        height = round(width / ratio)
        if height > src_h:
            height = src_h
            width = round(height * ratio)
        width = min(src_w, max(1, width))
        height = min(src_h, max(1, height))
    left = max(0, min(src_w - width, round(box.x * src_w)))
    top = max(0, min(src_h - height, round(box.y * src_h)))
    return left, top, left + width, top + height


def is_full_frame(src_w: int, src_h: int, box: CropBox) -> bool:
    """True when the box keeps the whole picture — no crop at all."""
    left, top, right, bottom = pixel_box(src_w, src_h, box)
    s = _FULL_FRAME_SLACK_PX
    return left <= s and top <= s and right >= src_w - s and bottom >= src_h - s


def is_too_small(src_w: int, src_h: int, box: CropBox) -> bool:
    left, top, right, bottom = pixel_box(src_w, src_h, box)
    return right - left < MIN_EDGE_PX or bottom - top < MIN_EDGE_PX


def exact_box(src_w: int, src_h: int, box: CropBox) -> CropBox:
    """The box as it was actually cut, back in fractions.

    What gets stored, so the row describes the file to the pixel — the
    gallery's before/after lays the original under the crop with it, and a
    rounded-away pixel there shows as a seam.
    """
    left, top, right, bottom = pixel_box(src_w, src_h, box)
    return CropBox(left / src_w, top / src_h, (right - left) / src_w,
                   (bottom - top) / src_h, box.aspect)


def token(box: dict | None) -> str:
    """Short, stable fingerprint of a stored box, for the `?v=` cache marker."""
    if not box:
        return "0"
    raw = json.dumps(box, sort_keys=True).encode()
    return hashlib.blake2s(raw, digest_size=4).hexdigest()


# ── File-level entry point ───────────────────────────────────────────────────


def _crop_file_sync(src: Path, dest: Path, box: CropBox) -> tuple[CropBox, int, int]:
    with PILImage.open(src) as opened:
        opened.load()
        exact = exact_box(opened.width, opened.height, box)
        out = opened.crop(pixel_box(opened.width, opened.height, box))
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    # Default compression, not optimize=True like the wand and the grain:
    # measured on a 2160x3840 cut, 0.6 s against 5.8 s for a file 6 % smaller.
    # This one sits on the interactive path and, with a wand or grain on top,
    # is only ever read by them.
    out.save(tmp, "PNG")
    tmp.replace(dest)
    return exact, out.width, out.height


async def crop_file(src: Path, dest: Path, box: CropBox) -> tuple[CropBox, int, int]:
    """Cut `box` out of `src` into `dest`.

    `src` is the rendition *below* the crop — the upscale when there is one,
    the original otherwise. Returns the box as actually cut plus the size of
    the result.
    """
    exact, w, h = await asyncio.to_thread(_crop_file_sync, src, dest, box)
    logger.info("Cropped %s → %s (%dx%d, %s)", src.name, dest.name, w, h, box.aspect)
    return exact, w, h
