"""
Model-aware YouTube uploader — pushes a Video row to YouTube and persists
``youtube_*`` fields. Idempotent on ``video.youtube_video_id``.

This lives next to (not inside) ``services/youtube/client.py`` because the
client is model-agnostic (takes raw paths + metadata) and we want to keep
that boundary clean for future non-Video callers.
"""
from __future__ import annotations

import logging

from sqlalchemy.ext.asyncio import AsyncSession

from core.config import settings
from core.models import Video
from services.youtube.client import upload_video

logger = logging.getLogger(__name__)


def _title_for(video: Video) -> str:
    """Derive a YouTube title from the Video row. ≤100 chars (the client
    truncates again defensively)."""
    wf = video.workflow or ""
    prompt = (video.prompt or "").strip().replace("\n", " ")

    if wf in ("improv_synth", "improv_hands", "improv_pip"):
        # Improv mixes already carry a descriptive prompt like
        # "Improv mix (synth) of <source>" — use it verbatim.
        return prompt or "Piano improvisation — art-rium"

    if wf in ("i2v_multi", "flf2v"):
        head = (prompt[:60].rstrip() + "…") if len(prompt) > 60 else prompt
        return f"Animate — {head}" if head else "Animate — art-rium"

    return prompt[:80] if prompt else "art-rium"


def _description_for(video: Video) -> str:
    """Derive a YouTube description. Generic but useful — names the source
    workflow and (when present) the full prompt."""
    wf = video.workflow or ""
    prompt = (video.prompt or "").strip()
    lines: list[str] = []

    if wf in ("improv_synth", "improv_hands", "improv_pip"):
        lines.append("Piano improvisation recorded as part of the art-rium project.")
    elif wf in ("i2v_multi", "flf2v"):
        lines.append("Generative key-frame video produced with ComfyUI (Wan 2.2 14B).")
    else:
        lines.append("Video from the art-rium project.")

    if prompt:
        lines.append("")
        lines.append(f"Prompt: {prompt}")

    site = (settings.artist_website_url or "").strip()
    if site:
        lines.append("")
        lines.append(f"More work at {site}")

    return "\n".join(lines)


async def upload_video_to_youtube(video: Video, db: AsyncSession) -> dict:
    """Upload *video* to YouTube and persist the resulting ID + URL.

    Idempotent: if ``video.youtube_video_id`` is already set, returns the
    existing record without re-uploading.

    Raises:
      RuntimeError on YouTube API or auth failures.
      FileNotFoundError if the local MP4 has been removed since DB insert.
    """
    if video.youtube_video_id and video.youtube_url:
        logger.info(
            "YouTube upload skipped for video %s (already uploaded as %s)",
            video.id, video.youtube_video_id,
        )
        return {
            "video_id":    video.youtube_video_id,
            "url":         video.youtube_url,
            "privacy":     video.youtube_privacy,
            "uploaded_at": video.youtube_uploaded_at,
            "skipped":     True,
        }

    if not video.filepath:
        raise RuntimeError(f"Video {video.id} has no filepath — cannot upload")

    src = settings.storage_dir / video.filepath
    logger.info(
        "YouTube upload start — video=%s, file=%s, kind=%s",
        video.id, src.name, video.workflow,
    )

    result = await upload_video(
        src,
        title=_title_for(video),
        description=_description_for(video),
        privacy=settings.youtube_privacy_default,
    )

    video.youtube_video_id    = result["video_id"]
    video.youtube_url         = result["url"]
    video.youtube_privacy     = result["privacy"]
    video.youtube_uploaded_at = result["uploaded_at"]
    await db.flush()

    return result | {"skipped": False}
