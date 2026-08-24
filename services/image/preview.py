"""AVIF previews — what the browser should actually be given to look at.

The library stores PNGs, and they are big: a 1080x1920 render is around 3 MB,
an upscaled one several times that. Until now every viewing path handed those
straight to the browser — the gallery modal loaded the full original, and any
mode that pages through pictures one after another would download a few
megabytes per picture. On a phone that is the whole experience.

Nothing about the stored file needs to change. The PNG is the master: it is
what the passes read, what gets published, and what a download should hand
over. What the *screen* needs is a different thing entirely — an image at
roughly the size it will be displayed at, in a format that is small.

AVIF, measured on this library at 1600 px: ~95 KB against 3.1 MB, a factor of
about 30, and visually indistinguishable at that size. WebP came out ~15 %
bigger for the same quality and JPEG about 2.5x bigger, so AVIF it is — every
browser that can run this PWA has supported it for years.

Three decisions worth writing down:

- **A fixed ladder of widths** (`PREVIEW_WIDTHS`), not an arbitrary `?w=`.
  The cache is on disk and lives as long as the image does, so an open
  parameter would let one picture accumulate a preview per viewport width
  anyone ever opened it at. Callers ask for what they want and get the next
  size up.

- **Never upscaled.** A source smaller than the requested width is encoded at
  its own size. The preview is meant to *match* the picture, so asking for
  1600 px of a 900 px image gives 900 px — sharp, not interpolated.

- **Content-addressed on the source's mtime and size**, so nothing has to
  remember to invalidate anything. Re-running the wand, the grain or the
  upscale rewrites the file it renders, which changes the fingerprint, which
  means the next request renders a fresh preview and the old one simply stops
  being referenced. This is the same problem `_serialize`'s `?v=` marker
  solves for the thumbnail, handled at the other end: there by naming the
  version in the URL, here by naming it in the cache path.

Encoder settings are measured rather than guessed (1600 px, this library):

    speed=4   92 KB   696 ms      speed=8   102 KB    58 ms
    speed=6   98 KB   135 ms      speed=10  112 KB    29 ms

speed=6 is the knee. Below it the encoder spends five times as long to save
7 %, and above it the files grow faster than the time saved is worth — the
encode happens once and is cached, while the bytes are paid on every view by
every device.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import shutil
from pathlib import Path

from PIL import Image as PILImage

logger = logging.getLogger(__name__)

# The sizes a preview may be rendered at, ascending. 640 covers a grid cell on
# a dense screen, 1024 a tablet, 1600 a phone at 3x, 2200 a desktop viewing
# full-bleed. A request between two rungs gets the one above it.
PREVIEW_WIDTHS = (640, 1024, 1600, 2200)
DEFAULT_WIDTH = 1600

QUALITY = 58
SPEED = 6          # see the table in the module docstring
MEDIA_TYPE = "image/avif"
SUFFIX = ".avif"

# One in-flight render per cache path. Two viewers opening the same picture at
# the same moment — or a prefetch racing the view it is prefetching for —
# would otherwise both encode it, and the second would overwrite the first
# while it was being served.
#
# Never pruned, and it does not need to be: `render` returns on the cache hit
# *before* it looks in here, so an entry only appears the first time a given
# preview is actually rendered in this process. Removing entries after the
# fact would reintroduce the race it exists to prevent — a waiter still queued
# on a lock that a later caller has already replaced with a fresh one.
_locks: dict[Path, asyncio.Lock] = {}


def clamp_width(requested: int | None) -> int:
    """The ladder rung to render at: the smallest one that is >= `requested`.

    Falls back to `DEFAULT_WIDTH` for a missing value and to the top rung for
    anything larger than the ladder, so an unbounded query parameter can never
    turn into an unbounded number of cache entries.
    """
    if not requested or requested <= 0:
        return DEFAULT_WIDTH
    for width in PREVIEW_WIDTHS:
        if requested <= width:
            return width
    return PREVIEW_WIDTHS[-1]


def fingerprint(src: Path) -> str:
    """Short digest of the source's identity *and* its current contents.

    mtime and size rather than a hash of the bytes: this is called on every
    preview request, and reading several megabytes to decide whether a cached
    file is still valid would cost more than the encode it is meant to save.
    A rendition pass rewrites the whole file, so both fields move.
    """
    st = src.stat()
    raw = f"{src.name}:{st.st_mtime_ns}:{st.st_size}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def cache_dir_for(cache_root: Path, stem: str) -> Path:
    """Where every preview of one source file lives.

    Grouped by source stem — and sharded one level above it, because these
    stems all begin with a UUID and a single flat directory would end up with
    one entry per image in the library. Grouping is what makes `purge` a
    directory removal rather than a search.
    """
    return cache_root / stem[:2].lower() / stem


def cache_path(cache_root: Path, src: Path, width: int) -> Path:
    """Absolute path of the cached preview for `src` at `width`."""
    stem = src.stem
    return cache_dir_for(cache_root, stem) / f"{width}_{fingerprint(src)}{SUFFIX}"


def purge(cache_root: Path, stem: str) -> None:
    """Drop every cached preview of one source file (best-effort).

    Called when the image it belongs to is deleted. Stale entries left behind
    by a re-render are harmless — nothing references them — but the ones
    belonging to a deleted picture should not outlive it.
    """
    target = cache_dir_for(cache_root, stem)
    if not target.exists():
        return
    try:
        shutil.rmtree(target)
    except Exception as exc:
        logger.warning(f"Could not purge previews for {stem}: {exc}")


def _render_sync(src: Path, dest: Path, width: int) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with PILImage.open(src) as img:
        img = img.convert("RGB")
        # `thumbnail` is a no-op when the image already fits, which is exactly
        # the "never upscaled" rule — a 900 px source asked for at 1600 stays
        # 900 px instead of being interpolated up to a soft 1600.
        img.thumbnail((width, width), PILImage.LANCZOS)
        # Written beside the target and moved into place, so a reader that
        # arrives mid-encode never sees a half-written file.
        tmp = dest.with_suffix(dest.suffix + ".part")
        img.save(tmp, "AVIF", quality=QUALITY, speed=SPEED)
        tmp.replace(dest)


async def render(cache_root: Path, src: Path, width: int) -> Path:
    """Path to the preview of `src` at `width`, rendering it if needed.

    Cheap and idempotent once warm: the common case is a `stat` on the source
    and an `exists` on the cache entry.
    """
    dest = cache_path(cache_root, src, width)
    if dest.exists():
        return dest

    async with _locks.setdefault(dest, asyncio.Lock()):
        # Whoever held the lock before us may have just rendered it.
        if not dest.exists():
            await asyncio.to_thread(_render_sync, src, dest, width)
            logger.debug(f"Preview rendered: {dest.name} ({dest.stat().st_size // 1024} KB)")
    return dest
