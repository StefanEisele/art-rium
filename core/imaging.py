"""
Image preparation for web upload — re-encode any source image to a
size- and quality-bounded JPEG or AVIF suitable for WordPress /
Instagram / local VLM analysis.

Three domain wrappers (preferred entry points):
  prepare_for_upload(src)  — 1080px / JPEG Q88 — Instagram dispatch (Meta
                             Graph API rejects AVIF, so this stays JPEG).
  prepare_for_wp(src)      — 1080px / AVIF Q65 or JPEG Q88 per settings —
                             WordPress media library uploader.
  prepare_for_vlm(src)     — settings.vlm_analysis_max_edge / JPEG Q80 —
                             Ollama VLM input (loopback, JPEG is enough).

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


def _flatten_and_resize(src: Path, max_edge: int) -> PILImage.Image:
    """Open src, flatten to opaque RGB, downscale so the longest edge ≤ max_edge."""
    with PILImage.open(src) as opened:
        img = _flatten_to_rgb(opened)
        w, h = img.size
        if max(w, h) > max_edge:
            if w >= h:
                img = img.resize((max_edge, round(h * max_edge / w)), PILImage.LANCZOS)
            else:
                img = img.resize((round(w * max_edge / h), max_edge), PILImage.LANCZOS)
        else:
            img.load()  # force-realise pixels before fp closes
        return img


def _prepare_jpg_sync(src: Path, max_edge: int, quality: int) -> tuple[bytes, str]:
    img = _flatten_and_resize(src, max_edge)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True, progressive=True)
    return buf.getvalue(), src.stem + ".jpg"


def _prepare_avif_sync(src: Path, max_edge: int, quality: int, speed: int) -> tuple[bytes, str]:
    img = _flatten_and_resize(src, max_edge)
    buf = BytesIO()
    img.save(buf, format="AVIF", quality=quality, speed=speed)
    return buf.getvalue(), src.stem + ".avif"


async def prepare_jpg_for_web(
    src: Path, *, max_edge: int = 1080, quality: int = 88,
) -> tuple[bytes, str]:
    """
    Open *src*, downscale so the longest edge is ≤ max_edge, return
    (jpeg_bytes, suggested_filename). Pillow runs in a thread pool.
    """
    return await asyncio.to_thread(_prepare_jpg_sync, src, max_edge, quality)


async def prepare_for_upload(src: Path) -> tuple[bytes, str]:
    """Web-upload preset: 1080px longest edge, JPEG Q88 — Instagram dispatch.

    Stays JPEG-only because Meta Graph API rejects AVIF for feed/reel/story
    media. For WordPress, use prepare_for_wp() instead.
    """
    return await prepare_jpg_for_web(src, max_edge=1080, quality=88)


async def prepare_for_wp(src: Path) -> tuple[bytes, str, str]:
    """WordPress media-upload preset. Returns (bytes, filename, content_type).

    AVIF (settings.wp_avif_quality / wp_avif_speed) or JPEG Q88 per
    settings.wp_upload_format. Both variants downscale to 1080px longest edge.
    """
    if settings.wp_upload_format == "avif":
        body, filename = await asyncio.to_thread(
            _prepare_avif_sync, src, 1080,
            settings.wp_avif_quality, settings.wp_avif_speed,
        )
        return body, filename, "image/avif"
    body, filename = await asyncio.to_thread(_prepare_jpg_sync, src, 1080, 88)
    return body, filename, "image/jpeg"


async def prepare_for_vlm(src: Path) -> tuple[bytes, str]:
    """VLM analysis preset: settings.vlm_analysis_max_edge, JPEG Q80."""
    return await prepare_jpg_for_web(
        src, max_edge=settings.vlm_analysis_max_edge, quality=80,
    )
