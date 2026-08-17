"""
How Instagram frames a feed post — and how to pre-crop so it doesn't have to.

Meta's own media reference is the source for the numbers: a feed image "must
be within a 4:5 to 1.91:1 range", min width 320, max width 1440, JPEG, sRGB.
A carousel renders every child in ONE frame, taken from the first child.

That combination is what puts bars on a 9:16 picture. 9:16 is 0.5625 — far
outside the range — so Instagram fits it into the tallest frame it has, 4:5,
and pads the two sides. Nothing is wrong with the upload; the frame simply
cannot be that tall. The only way to fill the frame is to hand Instagram an
image that already has the frame's shape, which is what `crop_box` computes.

Deliberately mirrored in the frontend preview (frontends/tools/instagram):
`crop_box` here and CSS `object-fit: cover` + `object-position` there describe
the same window, so what the preview shows is what gets published. Change one,
change the other.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image as PILImage

# The feed's hard limits, from Meta's media reference.
FEED_MIN_RATIO = 0.8    # 4:5   — the tallest a feed image may be (max portrait)
FEED_MAX_RATIO = 1.91   # 1.91:1 — the widest

# Frames a post can be pinned to explicitly. "auto" instead means: whatever
# Instagram will use anyway, i.e. the first child's ratio pulled into range.
FRAME_RATIOS: dict[str, float] = {
    "4x5":    0.8,
    "1x1":    1.0,
    "1.91x1": 1.91,
}
FRAME_AUTO = "auto"
FRAME_CHOICES = (FRAME_AUTO, *FRAME_RATIOS)

# Used when a post's frame is "auto" but no child reports its dimensions.
# 4:5 is the safe guess: it is the frame Instagram picks for anything taller,
# which is every portrait render this tool produces.
FRAME_FALLBACK_RATIO = FRAME_RATIOS["4x5"]

CROP_MODES = ("fit", "fill")


def clamp_feed_ratio(ratio: float) -> float:
    """Pull an aspect ratio into the feed's supported range."""
    return min(max(ratio, FEED_MIN_RATIO), FEED_MAX_RATIO)


def frame_ratio(setting: str, first_child_ratio: float | None) -> float:
    """The aspect ratio Instagram will render this post in.

    An explicit setting wins. "auto" reproduces Instagram's own rule: the first
    child decides, clamped into the supported range — which is exactly why a
    9:16 first child yields a 4:5 frame with bars rather than a 9:16 post.
    """
    if setting in FRAME_RATIOS:
        return FRAME_RATIOS[setting]
    if not first_child_ratio or first_child_ratio <= 0:
        return FRAME_FALLBACK_RATIO
    return clamp_feed_ratio(first_child_ratio)


def crop_box(
    src_w: int, src_h: int, target_ratio: float, offset: float = 0.5,
) -> tuple[int, int, int, int]:
    """The largest `target_ratio` window that fits inside `src_w`×`src_h`.

    `offset` (0..1) slides that window along the axis being cut — 0 keeps the
    top (or left) edge, 1 the bottom (or right), 0.5 centres it. Only one axis
    is ever cut: the window always spans the full extent of the other.

    Returns a PIL-style (left, upper, right, lower) box.
    """
    offset = min(max(offset, 0.0), 1.0)
    if src_w <= 0 or src_h <= 0:
        raise ValueError("Source dimensions must be positive")

    if src_w / src_h > target_ratio:
        # Too wide for the frame — take a full-height slice, slide horizontally.
        new_w = min(src_w, max(1, round(src_h * target_ratio)))
        left = round((src_w - new_w) * offset)
        return (left, 0, left + new_w, src_h)

    # Too tall (or exact) — take a full-width slice, slide vertically.
    new_h = min(src_h, max(1, round(src_w / target_ratio)))
    upper = round((src_h - new_h) * offset)
    return (0, upper, src_w, upper + new_h)


def needs_crop(src_w: int, src_h: int, target_ratio: float, tolerance: float = 0.01) -> bool:
    """False when the source already has the frame's shape (±tolerance), so a
    crop would only re-encode the file for nothing."""
    if src_w <= 0 or src_h <= 0:
        return False
    return abs(src_w / src_h - target_ratio) > tolerance


def crop_rendition_name(original_filename: str, target_ratio: float, offset: float) -> str:
    """Filename for a baked crop — deterministic from the crop it represents.

    Two posts asking for the same crop of the same picture therefore share one
    file instead of accumulating a rendition per post, and re-saving a post
    overwrites its own crop rather than orphaning the previous one. Stays
    inside routers/generate.py's `_SAFE_FILENAME_RE`, without which
    `/share/image/` would refuse to serve it to Instagram.
    """
    stem = Path(original_filename).stem
    return f"{stem}_igr{round(target_ratio * 1000)}o{round(offset * 100):02d}.png"


def render_crop_sync(src: Path, dest: Path, target_ratio: float, offset: float) -> tuple[int, int]:
    """Crop `src` to `target_ratio` and write it to `dest`. Returns the output
    size. Synchronous Pillow work — call through asyncio.to_thread."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with PILImage.open(src) as img:
        box = crop_box(img.width, img.height, target_ratio, offset)
        cropped = img.crop(box)
        # PNG throughout: /share/image serves everything as image/png, and the
        # enhance and grain renditions are PNG siblings too.
        cropped.save(dest, format="PNG")
        return cropped.width, cropped.height
