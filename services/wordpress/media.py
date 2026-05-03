"""
WordPress media uploader — Single Source of Truth for pushing gallery
images into the WP media library with VLM-generated metadata.

Pipeline per image:
  1. Re-encode source PNG → JPG (max 1080 longest edge, quality 88)
  2. VLM analyse via Ollama → alt_text, seo_description, caption (EN — Polylang default)
  3. POST /wp/v2/media (binary) → returns {id, source_url}
  4. POST /wp/v2/media/{id} with metadata fields (WP REST accepts POST as update)
  5. Persist on the Image row

Idempotent: if image.wp_media_id is set, returns existing metadata without
re-uploading or re-analysing. A separate regenerate-metadata endpoint can
refresh VLM output without re-uploading the binary (TODO).
"""
import asyncio
import logging
import uuid
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from core.config import settings
from core.imaging import prepare_jpg_for_web
from core.models import Image
from services.ollama.client import analyze_image
from services.wordpress.client import request_json, upload_binary

logger = logging.getLogger(__name__)

# Per-image asyncio locks — guarantee that two concurrent calls to
# upload_image_to_wp() for the same image_id serialize. Without this, two
# requests racing on the same row both see wp_media_id=NULL, both run the
# expensive VLM + WP upload, and end up creating duplicate WP media items
# while the second commit overwrites the first's wp_media_id pointer.
_upload_locks: dict[uuid.UUID, asyncio.Lock] = {}
_upload_locks_meta = asyncio.Lock()


async def _get_upload_lock(image_id: uuid.UUID) -> asyncio.Lock:
    async with _upload_locks_meta:
        lock = _upload_locks.get(image_id)
        if lock is None:
            lock = asyncio.Lock()
            _upload_locks[image_id] = lock
        return lock


async def upload_image_to_wp(image: Image, db: AsyncSession) -> dict:
    """
    Upload one Image to WordPress (idempotent on image.wp_media_id).

    Returns:
      {
        "image_id":             str,
        "media_id":             int,
        "source_url":           str,
        "alt_text":             str,
        "seo_description":      str,
        "caption":              str,
        "was_already_uploaded": bool,
      }
    """
    lock = await _get_upload_lock(image.id)
    async with lock:
        # Re-fetch the row inside the lock — another concurrent caller may
        # have completed the upload while we were waiting for the lock.
        await db.refresh(image)

        if image.wp_media_id and image.wp_source_url:
            logger.info("WP upload skipped — image %s already at media_id=%s", image.id, image.wp_media_id)
            return {
                "image_id":             str(image.id),
                "media_id":             image.wp_media_id,
                "source_url":           image.wp_source_url,
                "seo_title":            image.wp_seo_title or "",
                "alt_text":             image.wp_alt_text or "",
                "seo_description":      image.wp_seo_description or "",
                "caption":              image.wp_caption or "",
                "was_already_uploaded": True,
            }

        src = settings.storage_dir / image.filepath
        if not src.exists():
            raise FileNotFoundError(f"Source image missing on disk: {src}")

        # Two encodings: full-quality for WP upload, downscaled for the VLM.
        logger.info("WP upload %s — re-encoding %s", image.id, src.name)
        jpg_bytes,    jpg_filename = await prepare_jpg_for_web(src, max_edge=1080, quality=88)
        vlm_jpg_bytes, _           = await prepare_jpg_for_web(src, max_edge=settings.vlm_analysis_max_edge, quality=80)

        logger.info(
            "WP upload %s — analysing with %s (vlm payload %d KB, edge %d)",
            image.id, settings.ollama_vlm_model,
            len(vlm_jpg_bytes) // 1024, settings.vlm_analysis_max_edge,
        )
        metadata = await analyze_image(
            vlm_jpg_bytes,
            title=image.title,
            notes=image.notes,
            language=settings.wp_default_language,
        )

        logger.info("WP upload %s — POSTing to /wp/v2/media (lang=%s)", image.id, settings.wp_default_language)
        media = await upload_binary(
            "/wp/v2/media",
            body=jpg_bytes,
            filename=jpg_filename,
            content_type="image/jpeg",
            params={"lang": settings.wp_default_language},
        )
        media_id   = media["id"]
        source_url = media["source_url"]

        title_for_wp = image.title or metadata["seo_title"] or jpg_filename
        await request_json(
            "POST",
            f"/wp/v2/media/{media_id}",
            json={
                "title":       title_for_wp,
                "alt_text":    metadata["alt_text"],
                "caption":     metadata["caption"],
                "description": metadata["seo_description"],
            },
        )

        image.wp_media_id        = media_id
        image.wp_source_url      = source_url
        image.wp_uploaded_at     = datetime.now(timezone.utc)
        image.wp_seo_title       = metadata["seo_title"]
        image.wp_alt_text        = metadata["alt_text"]
        image.wp_seo_description = metadata["seo_description"]
        image.wp_caption         = metadata["caption"]
        await db.commit()

        logger.info("WP upload %s → media_id=%s, %s", image.id, media_id, source_url)

        return {
            "image_id":             str(image.id),
            "media_id":             media_id,
            "source_url":           source_url,
            "seo_title":            metadata["seo_title"],
            "alt_text":             metadata["alt_text"],
            "seo_description":      metadata["seo_description"],
            "caption":              metadata["caption"],
            "was_already_uploaded": False,
        }
