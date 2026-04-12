"""
Thumbnail generation — creates a 512 px JPEG from any PIL-readable source image.
The longest side is scaled to THUMB_MAX; the other side shrinks proportionally.
"""
import asyncio
import logging
from pathlib import Path

from PIL import Image as PILImage

logger = logging.getLogger(__name__)

THUMB_MAX = 512   # longest-side cap in pixels
THUMB_QUALITY = 85


def _make_thumbnail_sync(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with PILImage.open(src) as img:
        img = img.convert("RGB")          # handles PNG / RGBA → no transparency in JPEG
        img.thumbnail((THUMB_MAX, THUMB_MAX), PILImage.LANCZOS)
        img.save(dest, "JPEG", quality=THUMB_QUALITY, optimize=True)


async def make_thumbnail(src: Path, dest: Path) -> bool:
    """
    Generate a JPEG thumbnail at *dest* from *src*.
    Runs the blocking Pillow call in a thread pool.
    Returns True on success, False on failure.
    """
    try:
        await asyncio.to_thread(_make_thumbnail_sync, src, dest)
        logger.debug(f"Thumbnail created: {dest}")
        return True
    except Exception as exc:
        logger.warning(f"Thumbnail generation failed for {src}: {exc}")
        return False


def thumb_rel_path(filename: str) -> str:
    """
    Return the relative (to storage_dir) path where the thumbnail for *filename*
    should be stored, e.g. 'thumbnails/abc123_img.jpg'.
    """
    stem = Path(filename).stem
    return f"thumbnails/{stem}.jpg"
