"""
Image preparation for web upload — re-encode any source image to a
size- and quality-bounded JPEG suitable for WordPress / Instagram.

Used by:
  - services/wordpress/media.py  (upload pipeline)
"""
import asyncio
import logging
from io import BytesIO
from pathlib import Path

from PIL import Image as PILImage

logger = logging.getLogger(__name__)


def _flatten_to_rgb(img: PILImage.Image) -> PILImage.Image:
    """Composite alpha onto white so JPEG doesn't render transparency as black."""
    if img.mode in ("RGBA", "LA"):
        bg = PILImage.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        return bg
    if img.mode == "P":
        img = img.convert("RGBA")
        bg = PILImage.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        return bg
    if img.mode != "RGB":
        return img.convert("RGB")
    return img


def _prepare_jpg_sync(src: Path, max_edge: int, quality: int) -> tuple[bytes, str]:
    with PILImage.open(src) as opened:
        img = _flatten_to_rgb(opened)
        w, h = img.size
        if max(w, h) > max_edge:
            if w >= h:
                img = img.resize((max_edge, round(h * max_edge / w)), PILImage.LANCZOS)
            else:
                img = img.resize((round(w * max_edge / h), max_edge), PILImage.LANCZOS)

        buf = BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True, progressive=True)
        return buf.getvalue(), src.stem + ".jpg"


async def prepare_jpg_for_web(
    src: Path, *, max_edge: int = 1080, quality: int = 88,
) -> tuple[bytes, str]:
    """
    Open *src*, downscale so the longest edge is ≤ max_edge, return
    (jpeg_bytes, suggested_filename). Pillow runs in a thread pool.
    """
    return await asyncio.to_thread(_prepare_jpg_sync, src, max_edge, quality)
