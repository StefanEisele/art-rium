"""
WordPress integration API — uploads gallery images to the WP media library
with VLM-generated alt-text / SEO description / caption, and generates
multilingual blog post drafts in the art-rium voice.
"""
import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from core.auth import require_auth
from core.config import settings
from core.db import get_db
from core.models import Image
from services.ollama import client as ollama_client
from services.wordpress import client as wp_client
from services.wordpress.articles import generate_articles_for_image
from services.wordpress.media import upload_image_to_wp

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/wordpress", dependencies=[Depends(require_auth)])


class UploadRequest(BaseModel):
    image_ids: list[uuid.UUID]


@router.get("/status")
async def status():
    """Health + config check for WP and Ollama."""
    wp_missing = wp_client.missing_config()
    return {
        "wordpress": {
            "configured": not wp_missing,
            "missing":    wp_missing,
            "base_url":   settings.wp_base_url or None,
            "reachable":  await wp_client.reachable() if not wp_missing else False,
            "default_language": settings.wp_default_language,
        },
        "ollama": {
            "host":          settings.ollama_host,
            "vlm_model":     settings.ollama_vlm_model,
            "llm_model":     settings.ollama_llm_model,
            "reachable":     await ollama_client.reachable(),
        },
    }


@router.post("/media/upload")
async def upload_media(body: UploadRequest, db: AsyncSession = Depends(get_db)):
    """
    Upload one or more gallery images to WordPress.
    Idempotent: images already linked to a WP media item are returned as-is.

    Note: VLM analysis takes several seconds per image; large batches may
    take a minute or more. Frontends should set generous timeouts.
    """
    if wp_client.missing_config():
        raise HTTPException(
            status_code=400,
            detail=f"WordPress not configured: {wp_client.missing_config()}",
        )

    results: list[dict] = []
    errors:  list[dict] = []

    for image_id in body.image_ids:
        image = await db.get(Image, image_id)
        if not image:
            errors.append({"image_id": str(image_id), "error": "Image not found"})
            continue
        try:
            result = await upload_image_to_wp(image, db)
            results.append(result)
        except Exception as exc:
            logger.error("WP upload failed for %s: %s", image_id, exc, exc_info=True)
            errors.append({"image_id": str(image_id), "error": f"{type(exc).__name__}: {exc}"})

    return {"uploaded": results, "errors": errors}


class ArticleGenerateRequest(BaseModel):
    image_id: uuid.UUID
    publish: bool = Field(
        default=False,
        description="If true, posts go up as 'publish'. Default 'draft'.",
    )


@router.post("/article/generate")
async def generate_article(
    body: ArticleGenerateRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Generate a DE/EN/ZH blog-post triple about *image_id* and push them to
    WordPress. Synchronous (~60–120s with a warm article model). The image
    must already be uploaded to WP via /media/upload.
    """
    if wp_client.missing_config():
        raise HTTPException(
            status_code=400,
            detail=f"WordPress not configured: {wp_client.missing_config()}",
        )

    image = await db.get(Image, body.image_id)
    if not image:
        raise HTTPException(status_code=404, detail=f"Image {body.image_id} not found")
    if not image.wp_media_id:
        raise HTTPException(
            status_code=400,
            detail="Image is not yet uploaded to WordPress — call /media/upload first.",
        )

    try:
        return await generate_articles_for_image(image, db, publish=body.publish)
    except Exception as exc:
        logger.error("Article generation failed for %s: %s", body.image_id, exc, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"{type(exc).__name__}: {exc}",
        )
