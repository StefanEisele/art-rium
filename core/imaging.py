"""
Image preparation for web upload — re-encode any source image to a
size- and quality-bounded JPEG suitable for WordPress / Instagram /
local VLM analysis.

Two domain wrappers (preferred entry points):
  prepare_for_upload(src)  — 1080px / Q88 — WP media & IG dispatch
  prepare_for_vlm(src)     — settings.vlm_analysis_max_edge / Q80 — Ollama analyze

The low-level prepare_jpg_for_web stays available for one-off scripts and
the titler endpoint (which has its own UX-driven dimensions).
"""
import asyncio
import logging
from io import BytesIO
from pathlib import Path

from PIL import Image as PILImage

from core.config import settings

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


async def prepare_for_upload(src: Path) -> tuple[bytes, str]:
    """Web-upload preset: 1080px longest edge, JPEG Q88 — WP media + IG dispatch."""
    return await prepare_jpg_for_web(src, max_edge=1080, quality=88)


async def prepare_for_vlm(src: Path) -> tuple[bytes, str]:
    """VLM analysis preset: settings.vlm_analysis_max_edge, JPEG Q80."""
    return await prepare_jpg_for_web(
        src, max_edge=settings.vlm_analysis_max_edge, quality=80,
    )
