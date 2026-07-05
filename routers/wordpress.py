"""
WordPress integration API — uploads gallery images to the WP media library
with VLM-generated alt-text / SEO description / caption, and generates
EN+DE blog post drafts in the art-rium voice (Essay / Work / Lab modes).
"""
import logging
import uuid
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.ext.asyncio import AsyncSession

from core.auth import require_auth
from core.config import settings
from core.db import get_db
from core.models import Image, Video
from core.tasks import safe_create_task
from services.ollama import chat as ollama_client
from services.wordpress import article_jobs
from services.wordpress import client as wp_client
from services.wordpress.orchestrator import generate_modal_article
from services.wordpress.media import upload_image_to_wp
from services.youtube import client as yt_client
from services.youtube.uploader import upload_video_to_youtube

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
    """Health + config check for WP, Ollama, and YouTube."""
    wp_missing = wp_client.missing_config()
    yt_missing = yt_client.missing_config()
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
        "youtube": {
            "configured":      not yt_missing,
            "missing":         yt_missing,
            "privacy_default": settings.youtube_privacy_default,
            "reachable":       await yt_client.reachable() if not yt_missing else False,
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


# ───────────────────────────────────────────────────────────────────────────
# Modal article endpoint — Essay / Work / Lab (EN + DE)
# ───────────────────────────────────────────────────────────────────────────


class ParentSeries(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    url:  str = Field(min_length=1, description="Full URL to the parent series page on the artist's site")


class SingulartLink(BaseModel):
    title:         str = Field(min_length=1, max_length=200)
    url:           str = Field(min_length=1, description="Singulart product page URL")
    thumbnail_url: str = Field(min_length=1, description="Image thumbnail URL (typically a cropped/sized image)")


_ALLOWED_LANGUAGES = ("en", "de")
_ALLOWED_MODES = ("essay", "work", "lab")


class ArticleGenerateRequest(BaseModel):
    image_ids: list[uuid.UUID] = Field(
        min_length=1,
        max_length=6,
        description="1–6 image UUIDs. Featured image = first in list. Galleries auto-split for N≥4.",
    )
    video_ids: list[uuid.UUID] = Field(
        default_factory=list,
        max_length=6,
        description="0–6 video UUIDs (Animate or Improv). Each is uploaded to YouTube before the LLM call; the model places [VIDEO_K] placeholders in the prose, which the renderer substitutes with wp:embed blocks.",
    )
    mode:              Literal["essay", "work", "lab"] = Field(
        description="Article category. 'essay' = thesis-driven critic-audience long form. 'work' = curatorial work/series page with metadata block. 'lab' = ComfyUI tutorial with code blocks.",
    )
    series_name:       str | None = Field(default=None, description="Series name (Work) or workflow title (Lab) or anchoring series (Essay). Not translated across languages.")
    parent_series:     ParentSeries | None = Field(default=None, description="(Work only) Parent series the LLM should link inline via [PARENT_SERIES] placeholder.")
    singulart_links:   list[SingulartLink] | None = Field(default=None, max_length=10, description="(Work only) Singulart product cards rendered in the 'Available Works' section.")
    notes:             str | None = Field(default=None, max_length=2000, description="Free-text intent/context for the LLM. Essay: thesis + reference sources. Work: mood/technical context. Lab: tech stack/hardware notes.")
    artist_mode:       Literal["first_person", "third_person"] = Field(default="third_person", description="(Work only) Narrative perspective.")
    languages:         list[str] = Field(default=list(_ALLOWED_LANGUAGES), description="Subset of ('en','de'). The LLM generates both internally for voice alignment; only the chosen language(s) are pushed to WordPress.")
    publish:           bool = Field(default=False, description="If true, posts go up as 'publish'. Default 'draft'.")

    @field_validator("languages")
    @classmethod
    def _check_languages(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("languages must contain at least one of 'en','de'")
        bad = [lang for lang in v if lang not in _ALLOWED_LANGUAGES]
        if bad:
            raise ValueError(f"languages contains unsupported codes: {bad}. Allowed: {_ALLOWED_LANGUAGES}")
        return list(dict.fromkeys(v))  # dedupe, preserve order


@router.post("/article/generate", dependencies=[Depends(_require_wp_configured)])
async def generate_article(
    body: ArticleGenerateRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Kick off an EN+DE modal article generation (Essay / Work / Lab) as a
    background job and return the job_id immediately. Total wall time can
    exceed 6 minutes (image upload + VLM + LLM), so the work runs in an
    asyncio task and the frontend polls /article/jobs/{job_id}.

    Images that have not yet been uploaded to WordPress are auto-uploaded
    (with VLM analysis) as the first phase of the job. Polylang's REST API
    does not auto-link the EN+DE pair — that step has to be done by hand
    in WP admin after the job completes.
    """
    if not settings.artist_website_url or not settings.artist_instagram_url:
        raise HTTPException(
            status_code=400,
            detail="artist_website_url and artist_instagram_url must be set in .env / config for article footers.",
        )

    # Validate image existence up-front (fast feedback, before queuing).
    image_ids = [str(iid) for iid in body.image_ids]
    for image_id in body.image_ids:
        image = await db.get(Image, image_id)
        if not image:
            raise HTTPException(status_code=404, detail=f"Image {image_id} not found")

    # Validate video existence + readiness up-front.
    video_ids = [str(vid) for vid in body.video_ids]
    for video_id in body.video_ids:
        video = await db.get(Video, video_id)
        if not video:
            raise HTTPException(status_code=404, detail=f"Video {video_id} not found")
        if video.status != "done" or not video.filepath:
            raise HTTPException(
                status_code=400,
                detail=f"Video {video_id} is not ready (status={video.status!r}). Wait for generation to finish before embedding.",
            )

    # YouTube must be configured if any videos are requested.
    if body.video_ids and yt_client.missing_config():
        raise HTTPException(
            status_code=400,
            detail=(
                "Cannot embed videos — YouTube not configured. Missing: "
                f"{yt_client.missing_config()}. Run scripts/youtube_auth.py after setting "
                "YOUTUBE_CLIENT_ID + YOUTUBE_CLIENT_SECRET in .env."
            ),
        )

    # Parent series / Singulart only meaningful in Work mode — strip silently
    # for essay/lab so the frontend can keep the fields populated without
    # forcing the user to clear them when switching modes.
    parent_payload = body.parent_series.model_dump() if (body.parent_series and body.mode == "work") else None
    singulart_payload = (
        [link.model_dump() for link in body.singulart_links]
        if (body.singulart_links and body.mode == "work")
        else None
    )

    params = {
        "image_ids":       image_ids,
        "video_ids":       video_ids,
        "mode":            body.mode,
        "series_name":     body.series_name,
        "parent_series":   parent_payload,
        "singulart_links": singulart_payload,
        "notes":           body.notes,
        "artist_mode":     body.artist_mode,
        "languages":       body.languages,
        "publish":         body.publish,
    }
    job_id = await article_jobs.create_job(params)
    safe_create_task(_run_modal_article_job(job_id, params), name=f"article_job:{job_id}")

    return {"job_id": job_id, "status": "queued"}


@router.get("/article/jobs/{job_id}")
async def get_article_job(job_id: str):
    """Poll the state of a modal-article job. Returns the current job dict
    or 404 if the job is unknown (typical after server restart, since the
    in-memory tracker is wiped). Persisted Article DB rows survive even when
    the job is gone."""
    job = article_jobs.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found (may have been evicted or server restarted)")
    return job


@router.get("/article/jobs")
async def list_article_jobs(limit: int = 20):
    """List the most-recent modal-article jobs from the in-memory tracker."""
    return article_jobs.list_jobs(limit=limit)


async def _run_modal_article_job(job_id: str, params: dict) -> None:
    """Background runner: opens its own DB session, walks the job through
    upload → youtube → generate → done/failed phases, updates the job tracker."""
    from core.db import AsyncSessionLocal

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

            videos: list[Video] = []
            for vid in params.get("video_ids") or []:
                v = await db.get(Video, uuid.UUID(vid))
                if not v:
                    raise RuntimeError(f"Video {vid} not found")
                videos.append(v)

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

            pending_videos = [v for v in videos if not v.youtube_video_id or not v.youtube_url]
            if pending_videos:
                for idx, v in enumerate(pending_videos, 1):
                    await article_jobs.update_job(
                        job_id,
                        status="uploading",
                        phase="youtube",
                        message=f"Uploading video {idx}/{len(pending_videos)} to YouTube…",
                    )
                    await upload_video_to_youtube(v, db)
                await db.commit()
                for v in videos:
                    await db.refresh(v)

            await article_jobs.update_job(
                job_id,
                status="generating",
                phase="llm",
                message=f"Calling the LLM ({params['mode']} mode). The model writes EN/DE in one pass — this takes ~3–6 minutes.",
            )

            result = await generate_modal_article(
                images, db,
                mode=params["mode"],
                series_name=params["series_name"],
                parent_series=params["parent_series"],
                singulart_links=params["singulart_links"],
                user_notes=params["notes"],
                artist_mode=params["artist_mode"],
                languages=params["languages"],
                videos=videos or None,
                publish=params["publish"],
            )
            await article_jobs.mark_done(job_id, result)
        except Exception as exc:
            logger.error(
                "Modal article job %s failed: %s",
                job_id, exc, exc_info=True,
            )
            await article_jobs.mark_failed(job_id, f"{type(exc).__name__}: {exc}")
