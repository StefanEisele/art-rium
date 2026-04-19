"""
Instagram companion posts — Stories and Reel.

Called automatically after a feed post (single or carousel) is published.

Stories: one image-story per image in the post (up to 10).
Reel:    a crossfade slideshow MP4 generated from the same images.

Both run fire-and-update: failures are logged and written to the DB but
do NOT raise — the feed post is already live and must not be rolled back.
"""
import asyncio
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx
from sqlalchemy import select

from core.config import settings
from core.db import AsyncSessionLocal
from core.models import Image, InstagramPost
from workers.video_generator import generate_slideshow

logger = logging.getLogger(__name__)

REEL_POLL_INTERVAL = 5    # seconds between status polls
REEL_POLL_TIMEOUT  = 300  # seconds before giving up


# ── Public entry point ────────────────────────────────────────────────────────

async def publish_companion_posts(post_id: uuid.UUID) -> None:
    """
    Publish Stories and a Reel for the given (already-published) feed post.
    All errors are caught internally and written to the DB.
    """
    missing = [k for k, v in {
        "INSTAGRAM_USER_ID":    settings.instagram_user_id,
        "INSTAGRAM_ACCESS_TOKEN": settings.instagram_access_token,
        "PUBLIC_BASE_URL":      settings.public_base_url,
    }.items() if not v]
    if missing:
        logger.warning("Companion posts skipped — missing config: %s", ", ".join(missing))
        return

    await asyncio.gather(
        _publish_stories(post_id),
        _publish_reel(post_id),
        return_exceptions=True,   # log but don't raise
    )


# ── Stories ───────────────────────────────────────────────────────────────────

async def _publish_stories(post_id: uuid.UUID) -> None:
    graph = settings.instagram_graph_api_base
    uid   = settings.instagram_user_id
    token = settings.instagram_access_token

    async with AsyncSessionLocal() as db:
        post = await db.get(InstagramPost, post_id)
        if not post:
            return

        all_ids = [post.image_id] + (post.carousel_image_ids or [])
        img_result = await db.execute(select(Image).where(Image.id.in_(all_ids)))
        images = {img.id: img for img in img_result.scalars().all()}

    media_ids: list[str] = []
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            for img_id in all_ids:
                img = images.get(img_id)
                if not img:
                    logger.warning("Story: image %s not found, skipping", img_id)
                    continue
                r = await client.post(f"{graph}/{uid}/media", data={
                    "image_url":    _image_url(img.filename),
                    "media_type":   "STORIES",
                    "access_token": token,
                })
                body = r.json()
                _check(body, f"story container for {img.filename}")
                container_id = body["id"]

                r2 = await client.post(f"{graph}/{uid}/media_publish", data={
                    "creation_id":  container_id,
                    "access_token": token,
                })
                body2 = r2.json()
                _check(body2, f"story publish for {img.filename}")
                media_ids.append(body2["id"])
                logger.info("Story published: %s (%s)", body2["id"], img.filename)
                await asyncio.sleep(3)  # avoid "Media ID not available" on rapid successive stories

        status = "posted"
    except Exception as exc:
        logger.error("Stories failed for post %s: %s", post_id, exc)
        status = "posted" if media_ids else "failed"

    async with AsyncSessionLocal() as db:
        post = await db.get(InstagramPost, post_id)
        if post:
            post.story_status    = status
            post.story_media_ids = media_ids or None
            post.updated_at      = datetime.now(timezone.utc)
            await db.commit()


# ── Reel ──────────────────────────────────────────────────────────────────────

async def _publish_reel(post_id: uuid.UUID) -> None:
    graph = settings.instagram_graph_api_base
    uid   = settings.instagram_user_id
    token = settings.instagram_access_token

    async with AsyncSessionLocal() as db:
        post = await db.get(InstagramPost, post_id)
        if not post:
            return
        all_ids = [post.image_id] + (post.carousel_image_ids or [])
        img_result = await db.execute(select(Image).where(Image.id.in_(all_ids)))
        images = {img.id: img for img in img_result.scalars().all()}
        caption = post.caption or ""

    video_path: Path | None = None
    reel_media_id: str | None = None
    status = "failed"

    try:
        # ── 1. Build ordered image file paths ────────────────────────────────
        image_paths: list[Path] = []
        for img_id in all_ids:
            img = images.get(img_id)
            if not img:
                raise ValueError(f"Reel: image {img_id} not found in DB")
            image_paths.append(settings.storage_dir / img.filepath)

        # ── 2. Generate slideshow video ───────────────────────────────────────
        settings.reels_dir.mkdir(parents=True, exist_ok=True)
        video_path = await generate_slideshow(
            image_paths,
            settings.reels_dir,
            ffmpeg_path=settings.ffmpeg_path,
        )

        # ── 3. Create Reel container ──────────────────────────────────────────
        video_url = _reel_url(video_path.name)
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(f"{graph}/{uid}/media", data={
                "media_type":    "REELS",
                "video_url":     video_url,
                "caption":       caption,
                "share_to_feed": "true",
                "access_token":  token,
            })
            body = r.json()
            _check(body, "reel container")
            container_id = body["id"]
            logger.info("Reel container: %s", container_id)

            # ── 4. Poll until FINISHED ────────────────────────────────────────
            await _wait_for_video(client, container_id, token)

            # ── 5. Publish ────────────────────────────────────────────────────
            r2 = await client.post(f"{graph}/{uid}/media_publish", data={
                "creation_id":  container_id,
                "access_token": token,
            })
            body2 = r2.json()
            _check(body2, "reel publish")
            reel_media_id = body2["id"]
            logger.info("Reel published: %s", reel_media_id)

        status = "posted"

    except Exception as exc:
        logger.error("Reel failed for post %s: %s", post_id, exc)
        status = "failed"

    finally:
        # Clean up temp video regardless of outcome
        if video_path and video_path.exists():
            try:
                video_path.unlink()
            except Exception as e:
                logger.warning("Could not delete reel temp file %s: %s", video_path, e)

    async with AsyncSessionLocal() as db:
        post = await db.get(InstagramPost, post_id)
        if post:
            post.reel_status   = status
            post.reel_media_id = reel_media_id
            post.updated_at    = datetime.now(timezone.utc)
            await db.commit()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _image_url(filename: str) -> str:
    base = settings.public_base_url.rstrip("/")
    url  = f"{base}/share/image/{filename}"
    if settings.image_share_token:
        url += f"?token={settings.image_share_token}"
    return url


def _reel_url(filename: str) -> str:
    base = settings.public_base_url.rstrip("/")
    url  = f"{base}/share/reel/{filename}"
    if settings.image_share_token:
        url += f"?token={settings.image_share_token}"
    return url


def _check(body: dict, context: str) -> None:
    if "error" in body:
        err = body["error"]
        raise RuntimeError(f"{context}: {err.get('message', body)}")


async def _wait_for_video(
    client: httpx.AsyncClient,
    container_id: str,
    token: str,
) -> None:
    """Poll container status until FINISHED or timeout."""
    graph = settings.instagram_graph_api_base
    deadline = asyncio.get_event_loop().time() + REEL_POLL_TIMEOUT

    while True:
        r = await client.get(
            f"{graph}/{container_id}",
            params={"fields": "status_code", "access_token": token},
        )
        body = r.json()
        _check(body, "reel status poll")
        status_code = body.get("status_code", "")
        logger.debug("Reel container %s status: %s", container_id, status_code)

        if status_code == "FINISHED":
            return
        if status_code == "ERROR":
            raise RuntimeError(f"Reel container {container_id} entered ERROR state")

        if asyncio.get_event_loop().time() >= deadline:
            raise TimeoutError(
                f"Reel container {container_id} not ready after {REEL_POLL_TIMEOUT}s"
            )
        await asyncio.sleep(REEL_POLL_INTERVAL)
