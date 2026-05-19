"""
WordPress integration API — uploads gallery images to the WP media library
with VLM-generated alt-text / SEO description / caption, and generates
multilingual blog post drafts in the art-rium voice.
"""
import asyncio
import logging
import uuid
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.ext.asyncio import AsyncSession

from core.auth import require_auth
from core.config import settings
from core.db import get_db
from core.models import Image
from services.ollama import client as ollama_client
from services.wordpress import article_jobs
from services.wordpress import client as wp_client
from services.wordpress.articles import generate_articles_for_images, generate_rich_articles_for_series
from services.wordpress.media import upload_image_to_wp

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/wordpress", dependencies=[Depends(require_auth)])


def _require_wp_configured() -> None:
    """Reject the request with 400 if WordPress env vars are missing.

    Calls missing_config() once and reuses the result for the error message
    — the prior inline pattern called it twice in the same handler.
    """
    missing = wp_client.missing_config()
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"WordPress not configured: {missing}",
        )


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


@router.post("/media/upload", dependencies=[Depends(_require_wp_configured)])
async def upload_media(body: UploadRequest, db: AsyncSession = Depends(get_db)):
    """
    Upload one or more gallery images to WordPress.
    Idempotent: images already linked to a WP media item are returned as-is.

    Note: VLM analysis takes several seconds per image; large batches may
    take a minute or more. Frontends should set generous timeouts.
    """
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
    image_ids: list[uuid.UUID] = Field(
        min_length=1,
        max_length=6,
        description="1–6 image UUIDs. With more than one, the article treats them as a series.",
    )
    publish: bool = Field(
        default=False,
        description="If true, posts go up as 'publish'. Default 'draft'.",
    )


@router.post("/article/generate", dependencies=[Depends(_require_wp_configured)])
async def generate_article(
    body: ArticleGenerateRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Generate a DE/EN/ZH blog-post triple about one or more images (a series)
    and push them to WordPress. Synchronous (~60–180s warm depending on
    image count). All images must already be uploaded to WP via /media/upload.
    The first image's wp_media_id becomes the WP featured_media.
    """
    images: list[Image] = []
    for image_id in body.image_ids:
        image = await db.get(Image, image_id)
        if not image:
            raise HTTPException(status_code=404, detail=f"Image {image_id} not found")
        if not image.wp_media_id:
            raise HTTPException(
                status_code=400,
                detail=f"Image {image_id} is not yet uploaded to WordPress — call /media/upload first.",
            )
        images.append(image)

    try:
        return await generate_articles_for_images(images, db, publish=body.publish)
    except Exception as exc:
        logger.error(
            "Article generation failed for %d image(s): %s",
            len(body.image_ids), exc, exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"{type(exc).__name__}: {exc}",
        )


# ───────────────────────────────────────────────────────────────────────────
# Rich-article endpoint — SEO-friendly multi-section series posts
# ───────────────────────────────────────────────────────────────────────────


class ParentSeries(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    url:  str = Field(min_length=1, description="Full URL to the parent series page on the artist's site")


class SingulartLink(BaseModel):
    title:         str = Field(min_length=1, max_length=200)
    url:           str = Field(min_length=1, description="Singulart product page URL")
    thumbnail_url: str = Field(min_length=1, description="Image thumbnail URL (typically a cropped/sized image)")


_ALLOWED_LANGUAGES = ("de", "en", "zh")


class ArticleGenerateRichRequest(BaseModel):
    image_ids: list[uuid.UUID] = Field(
        min_length=1,
        max_length=6,
        description="1–6 image UUIDs. Featured image = first in list. Galleries auto-split for N≥4.",
    )
    series_name:       str | None = Field(default=None, description="Name of the series (used as article title and quoted in intro). Not translated across languages.")
    parent_series:     ParentSeries | None = Field(default=None, description="Parent series the LLM should link inline via [PARENT_SERIES] placeholder.")
    singulart_links:   list[SingulartLink] | None = Field(default=None, max_length=10, description="Singulart product cards rendered in 'Available Works' section.")
    notes:             str | None = Field(default=None, max_length=2000, description="Free-text intent/context for the LLM (mood, technical context, references). Anchors the prose in concrete artist intent.")
    artist_mode:       Literal["first_person", "third_person"] = Field(default="third_person", description="Narrative perspective. 'first_person' lets the LLM speak as the artist (works for process slots); 'third_person' is editorial/curatorial.")
    languages:         list[str] = Field(default=list(_ALLOWED_LANGUAGES), description="Subset of ('de','en','zh'). Only the chosen language(s) are pushed to WordPress; the LLM still generates all three internally for voice alignment.")
    publish:           bool = Field(default=False, description="If true, posts go up as 'publish'. Default 'draft'.")

    @field_validator("languages")
    @classmethod
    def _check_languages(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("languages must contain at least one of 'de','en','zh'")
        bad = [lang for lang in v if lang not in _ALLOWED_LANGUAGES]
        if bad:
            raise ValueError(f"languages contains unsupported codes: {bad}. Allowed: {_ALLOWED_LANGUAGES}")
        return list(dict.fromkeys(v))  # dedupe, preserve order


@router.post("/article/generate-rich", dependencies=[Depends(_require_wp_configured)])
async def generate_rich_article(
    body: ArticleGenerateRichRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Kick off a DE/EN/ZH SEO-friendly rich series article generation as a
    background job and return the job_id immediately. Total wall time can
    exceed 6 minutes (image upload + VLM + LLM), so we run it in an
    asyncio task and let the frontend poll /article/jobs/{job_id}.

    Images that have not yet been uploaded to WordPress are auto-uploaded
    (with VLM analysis) as the first phase of the job. Polylang's REST API
    does not auto-link translations — that step still has to be done by hand
    in WP admin after the job completes.
    """
    if not settings.artist_website_url or not settings.artist_instagram_url:
        raise HTTPException(
            status_code=400,
            detail="artist_website_url and artist_instagram_url must be set in .env / config for rich-article footers.",
        )

    # Validate image existence up-front (fast feedback, before queuing).
    image_ids = [str(iid) for iid in body.image_ids]
    for image_id in body.image_ids:
        image = await db.get(Image, image_id)
        if not image:
            raise HTTPException(status_code=404, detail=f"Image {image_id} not found")

    parent_payload = body.parent_series.model_dump() if body.parent_series else None
    singulart_payload = [link.model_dump() for link in body.singulart_links] if body.singulart_links else None

    params = {
        "image_ids":       image_ids,
        "series_name":     body.series_name,
        "parent_series":   parent_payload,
        "singulart_links": singulart_payload,
        "notes":           body.notes,
        "artist_mode":     body.artist_mode,
        "languages":       body.languages,
        "publish":         body.publish,
    }
    job_id = await article_jobs.create_job(params)
    asyncio.create_task(_run_rich_article_job(job_id, params))

    return {"job_id": job_id, "status": "queued"}


@router.get("/article/jobs/{job_id}")
async def get_article_job(job_id: str):
    """Poll the state of a rich-article job. Returns the current job dict
    or 404 if the job is unknown (typical after server restart, since the
    in-memory tracker is wiped). Persisted Article DB rows survive even when
    the job is gone — list /api/wordpress/articles for those."""
    job = article_jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found (may have been evicted or server restarted)")
    return job


@router.get("/article/jobs")
async def list_article_jobs(limit: int = 20):
    """List the most-recent rich-article jobs from the in-memory tracker."""
    return article_jobs.list_jobs(limit=limit)


async def _run_rich_article_job(job_id: str, params: dict) -> None:
    """Background runner: opens its own DB session, walks the job through
    upload → generate → done/failed phases, updates the job tracker."""
    from core.db import AsyncSessionLocal
    from services.wordpress.media import upload_image_to_wp

    async with AsyncSessionLocal() as db:
        try:
            await article_jobs.update_job(
                job_id,
                status="running",
                phase="loading",
                message="Loading images…",
            )

            images: list[Image] = []
            for iid in params["image_ids"]:
                img = await db.get(Image, uuid.UUID(iid))
                if not img:
                    raise RuntimeError(f"Image {iid} not found")
                images.append(img)

            pending = [img for img in images if not img.wp_media_id or not img.wp_source_url]
            if pending:
                for idx, img in enumerate(pending, 1):
                    await article_jobs.update_job(
                        job_id,
                        status="uploading",
                        phase="upload",
                        message=f"Uploading image {idx}/{len(pending)} to WordPress (VLM analysis…)",
                    )
                    await upload_image_to_wp(img, db)
                # Refresh all images to pick up wp_media_id / wp_source_url for the renderer.
                for img in images:
                    await db.refresh(img)

            await article_jobs.update_job(
                job_id,
                status="generating",
                phase="llm",
                message="Calling the LLM. The model writes DE/EN/ZH in one pass — this takes ~3–6 minutes.",
            )

            result = await generate_rich_articles_for_series(
                images, db,
                series_name=params["series_name"],
                parent_series=params["parent_series"],
                singulart_links=params["singulart_links"],
                user_notes=params["notes"],
                artist_mode=params["artist_mode"],
                languages=params["languages"],
                publish=params["publish"],
            )
            await article_jobs.mark_done(job_id, result)
        except Exception as exc:
            logger.error(
                "Rich article job %s failed: %s",
                job_id, exc, exc_info=True,
            )
            await article_jobs.mark_failed(job_id, f"{type(exc).__name__}: {exc}")
