"""
Pi posting outpost client.

The outpost (ig.stefaneisele.com) accepts a self-contained package per post:
re-encoded JPEGs, an optional pre-rendered slideshow MP4, caption, and the
intended publish times. Once enqueued, the Pi publishes autonomously even
when the desktop is offline.

This module owns:
- `dispatch_to_outpost(post_id)`  — render + upload, persist outpost_id.
- `sync_outpost_status(post_ids)` — pull /status/{id} for in-flight rows and
  mirror feed/reel state back into the local instagram_posts row.
- `cancel_on_outpost(outpost_id)` — best-effort DELETE on post deletion.
"""
from __future__ import annotations

import logging
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx
from sqlalchemy import select

from core.config import settings
from core.db import AsyncSessionLocal
from core.imaging import prepare_jpg_for_web
from core.models import Image, InstagramPost
from core.scheduling import companion_at
from workers.video_generator import generate_slideshow

logger = logging.getLogger(__name__)


def missing_config() -> list[str]:
    missing: list[str] = []
    if not settings.outpost_base_url:
        missing.append("OUTPOST_BASE_URL")
    if not settings.outpost_shared_secret:
        missing.append("OUTPOST_SHARED_SECRET")
    return missing


def _headers() -> dict[str, str]:
    return {"X-Outpost-Key": settings.outpost_shared_secret}


def _base() -> str:
    return settings.outpost_base_url.rstrip("/")


async def health() -> dict:
    """One-shot reachability probe for the frontend / status endpoint."""
    miss = missing_config()
    if miss:
        return {"reachable": False, "missing_config": miss}
    try:
        async with httpx.AsyncClient(timeout=8) as client:
            r = await client.get(f"{_base()}/health")
            return {"reachable": r.status_code == 200, "http_status": r.status_code}
    except Exception as exc:
        return {"reachable": False, "error": f"{type(exc).__name__}: {exc}"}


# ── Dispatch ────────────────────────────────────────────────────────────────


async def dispatch_to_outpost(post_id: uuid.UUID) -> None:
    """
    Build the multipart package for a scheduled post and POST it to /enqueue.

    Re-encodes images to web-grade JPEG (1080px Q88), optionally renders the
    reel slideshow inline (Pi cannot ffmpeg at speed), then uploads and
    persists `outpost_id` + `outpost_status='queued'`.

    Failures are recorded on the row (`outpost_status='failed'`, `error=...`)
    rather than raised — the caller is a fire-and-forget background task.
    """
    miss = missing_config()
    if miss:
        await _record_failure(post_id, f"Outpost not configured: {', '.join(miss)}")
        return

    async with AsyncSessionLocal() as db:
        post = await db.get(InstagramPost, post_id)
        if not post:
            return
        if post.dispatch_target != "outpost":
            return
        if post.outpost_id:
            logger.info("dispatch_to_outpost %s — already dispatched (%s), skipping",
                        post_id, post.outpost_id)
            return

        all_ids = [post.image_id] + list(post.carousel_image_ids or [])
        img_result = await db.execute(select(Image).where(Image.id.in_(all_ids)))
        images = {img.id: img for img in img_result.scalars().all()}
        ordered_images = [images[i] for i in all_ids if i in images]
        if len(ordered_images) != len(all_ids):
            await _record_failure(post_id, "One or more images not found in DB")
            return

        caption = post.caption or ""
        scheduled_at = post.scheduled_at
        reel_publish_at = (
            companion_at(post.scheduled_at, post.reel_delay_minutes, post.companion_time)
            if post.reel_delay_minutes is not None
            else None
        )
        story_publish_at = (
            companion_at(post.scheduled_at, post.story_delay_minutes, post.companion_time)
            if post.story_delay_minutes is not None
            else None
        )

    # ── Re-encode images (off the DB session) ─────────────────────────────
    try:
        encoded: list[tuple[bytes, str]] = []
        for img in ordered_images:
            src = settings.storage_dir / img.filepath
            jpg_bytes, jpg_name = await prepare_jpg_for_web(src)
            encoded.append((jpg_bytes, jpg_name))
    except Exception as exc:
        await _record_failure(post_id, f"Image re-encode failed: {exc}")
        return

    # ── Render slideshow MP4 if a reel is scheduled ───────────────────────
    reel_path: Path | None = None
    if reel_publish_at is not None:
        try:
            image_paths = [settings.storage_dir / img.filepath for img in ordered_images]
            settings.reels_dir.mkdir(parents=True, exist_ok=True)
            reel_path = await generate_slideshow(
                image_paths,
                settings.reels_dir,
                ffmpeg_path=settings.ffmpeg_path,
            )
        except Exception as exc:
            await _record_failure(post_id, f"Slideshow render failed: {exc}")
            return

    # ── Multipart upload to /enqueue ──────────────────────────────────────
    try:
        files: list[tuple[str, tuple[str, bytes, str]]] = [
            ("images", (name, data, "image/jpeg")) for data, name in encoded
        ]
        if reel_path is not None:
            with open(reel_path, "rb") as f:
                files.append(("reel_mp4", (reel_path.name, f.read(), "video/mp4")))

        data = {
            "caption": caption,
            "scheduled_at": _iso(scheduled_at),
        }
        if reel_publish_at is not None:
            data["reel_publish_at"] = _iso(reel_publish_at)
        if story_publish_at is not None:
            data["story_publish_at"] = _iso(story_publish_at)

        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(
                f"{_base()}/enqueue",
                headers=_headers(),
                data=data,
                files=files,
            )
        if r.status_code >= 300:
            await _record_failure(
                post_id,
                f"Outpost /enqueue HTTP {r.status_code}: {r.text[:300]}",
            )
            return
        body = r.json()
        outpost_id = body.get("id")
        if not outpost_id:
            await _record_failure(post_id, f"Outpost response missing id: {body}")
            return
    except Exception as exc:
        await _record_failure(
            post_id,
            f"Outpost dispatch failed: {type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        )
        return

    async with AsyncSessionLocal() as db:
        fresh = await db.get(InstagramPost, post_id)
        if not fresh:
            return
        fresh.outpost_id = outpost_id
        fresh.outpost_status = body.get("status", "queued")
        fresh.outpost_reel_status = "pending" if reel_path is not None else None
        if story_publish_at is not None and not fresh.story_status:
            fresh.story_status = "pending"
            fresh.story_scheduled_at = story_publish_at
        fresh.outpost_dispatched_at = datetime.now(timezone.utc)
        fresh.error = None
        fresh.updated_at = datetime.now(timezone.utc)
        await db.commit()
        logger.info("dispatch_to_outpost %s → outpost_id=%s (reel=%s)",
                    post_id, outpost_id, reel_path is not None)


# ── Sync ────────────────────────────────────────────────────────────────────


async def sync_outpost_status() -> None:
    """
    Mirror Pi state back into the local DB for all in-flight outpost posts.

    A row is in-flight when outpost_id is set AND (outpost_status not in
    terminal states OR outpost_reel_status not in terminal states). On every
    tick we pull /status/{id}, then update local feed status, reel status,
    and instagram_media_id accordingly.
    """
    if missing_config():
        return

    terminal_feed = {"posted", "failed", "cancelled"}
    terminal_reel = {"posted", "failed"}
    terminal_story = {"posted", "failed"}

    async with AsyncSessionLocal() as db:
        result = await db.execute(
            select(InstagramPost)
            .where(InstagramPost.dispatch_target == "outpost")
            .where(InstagramPost.outpost_id.isnot(None))
        )
        in_flight: list[InstagramPost] = []
        for post in result.scalars().all():
            feed_done = (post.outpost_status or "") in terminal_feed
            reel_done = (
                post.outpost_reel_status is None
                or post.outpost_reel_status in terminal_reel
            )
            story_done = (
                post.story_status is None
                or post.story_status in terminal_story
            )
            if not (feed_done and reel_done and story_done):
                in_flight.append(post)

    if not in_flight:
        return

    async with httpx.AsyncClient(timeout=15, headers=_headers()) as client:
        for post in in_flight:
            try:
                r = await client.get(f"{_base()}/status/{post.outpost_id}")
                if r.status_code != 200:
                    logger.debug("outpost /status/%s → HTTP %s", post.outpost_id, r.status_code)
                    continue
                remote = r.json()
            except Exception as exc:
                logger.debug("outpost /status/%s failed: %s", post.outpost_id, exc)
                continue

            await _apply_remote_state(post.id, remote)


async def _apply_remote_state(post_id: uuid.UUID, remote: dict) -> None:
    """Mirror Pi /status response into local row. Idempotent."""
    async with AsyncSessionLocal() as db:
        post = await db.get(InstagramPost, post_id)
        if not post or post.dispatch_target != "outpost":
            return

        changed = False
        now = datetime.now(timezone.utc)

        remote_feed = remote.get("status")
        if remote_feed and remote_feed != post.outpost_status:
            post.outpost_status = remote_feed
            changed = True

        remote_reel = remote.get("reel_status")
        if remote_reel != post.outpost_reel_status:
            post.outpost_reel_status = remote_reel
            changed = True

        # Mirror Instagram-side feed publication into the canonical fields.
        if remote_feed == "posted":
            if post.status != "posted":
                post.status = "posted"
                changed = True
            media_id = remote.get("instagram_media_id")
            if media_id and post.instagram_media_id != media_id:
                post.instagram_media_id = media_id
                changed = True
            if post.error:
                post.error = None
                changed = True
        elif remote_feed == "failed":
            err = remote.get("error") or "Outpost reported failure"
            if post.error != err:
                post.error = err
                changed = True

        # Reel companion mirror.
        if remote_reel == "posted":
            reel_media = remote.get("reel_media_id")
            if reel_media and post.reel_media_id != reel_media:
                post.reel_media_id = reel_media
                changed = True
            if post.reel_status != "posted":
                post.reel_status = "posted"
                if not post.reel_scheduled_at:
                    post.reel_scheduled_at = now
                changed = True
        elif remote_reel == "failed" and post.reel_status != "failed":
            post.reel_status = "failed"
            changed = True

        # Story companion mirror.
        remote_story = remote.get("story_status")
        if remote_story and remote_story != post.story_status:
            post.story_status = remote_story
            changed = True
        remote_story_media = remote.get("story_media_ids")
        if remote_story_media and post.story_media_ids != remote_story_media:
            post.story_media_ids = remote_story_media
            changed = True
        if remote_story == "posted" and not post.story_scheduled_at:
            post.story_scheduled_at = now
            changed = True

        if changed:
            post.updated_at = now
            await db.commit()
            logger.info(
                "outpost sync %s → status=%s reel=%s media=%s",
                post_id, post.status, post.reel_status, post.instagram_media_id,
            )


# ── Cancel ──────────────────────────────────────────────────────────────────


async def cancel_on_outpost(outpost_id: str) -> None:
    """Best-effort DELETE on outpost; never raises."""
    if missing_config() or not outpost_id:
        return
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.delete(f"{_base()}/{outpost_id}", headers=_headers())
            logger.info("Cancelled outpost post %s → HTTP %s", outpost_id, r.status_code)
    except Exception as exc:
        logger.warning("Failed to cancel outpost post %s: %s", outpost_id, exc)


async def update_on_outpost(
    outpost_id: str,
    *,
    caption: str | None = None,
    scheduled_at: datetime | None = None,
    reel_publish_at: datetime | None = None,
    story_publish_at: datetime | None = None,
) -> None:
    """
    PATCH the Pi-side post. Only sends fields the caller passed (None = leave
    untouched). Raises RuntimeError on non-2xx so the router can surface the
    detail back to the user.
    """
    if missing_config() or not outpost_id:
        raise RuntimeError("Outpost not configured")
    payload: dict = {}
    if caption is not None:
        payload["caption"] = caption
    if scheduled_at is not None:
        payload["scheduled_at"] = _iso(scheduled_at)
    if reel_publish_at is not None:
        payload["reel_publish_at"] = _iso(reel_publish_at)
    if story_publish_at is not None:
        payload["story_publish_at"] = _iso(story_publish_at)
    if not payload:
        return
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.patch(
            f"{_base()}/{outpost_id}",
            headers={**_headers(), "Content-Type": "application/json"},
            json=payload,
        )
    if r.status_code >= 300:
        raise RuntimeError(f"Outpost PATCH HTTP {r.status_code}: {r.text[:300]}")
    logger.info("update_on_outpost %s ← %s", outpost_id, sorted(payload.keys()))


# ── Internal ───────────────────────────────────────────────────────────────


async def _record_failure(post_id: uuid.UUID, message: str) -> None:
    logger.error("outpost dispatch %s failed: %s", post_id, message)
    async with AsyncSessionLocal() as db:
        post = await db.get(InstagramPost, post_id)
        if not post:
            return
        post.outpost_status = "failed"
        post.error = message[:1000]
        post.updated_at = datetime.now(timezone.utc)
        await db.commit()


def _iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()
