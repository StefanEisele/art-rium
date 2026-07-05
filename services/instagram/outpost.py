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
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

import httpx
from sqlalchemy import select

from core.config import settings
from core.db import AsyncSessionLocal
from core.imaging import prepare_for_upload
from core.models import InstagramPost, Video
from core.scheduling import companion_at
from services.instagram.companions import find_companion, get_or_create_companion
from services.instagram.ig_video import ensure_ig_compatible
from services.instagram.media import load_media_refs
from services.instagram.reel_concat import concat_reel_videos
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
    Route a scheduled post to the right Pi-outpost dispatch path.

    For `kind='feed'` rows we re-encode images, optionally render a reel
    slideshow, and POST to /enqueue. For `kind='reel'` rows we concatenate
    the chosen source videos into one 1080×1920 MP4 and POST to /enqueue-reel.

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
        kind = post.kind

    if kind == "reel":
        await _dispatch_reel_only(post_id)
        return
    await _dispatch_feed(post_id)


async def _dispatch_feed(post_id: uuid.UUID) -> None:
    """Re-encode images + optional reel slideshow + carousel videos, POST to /enqueue.

    Sends the legacy `images[]`-only shape when the post is image-only; adds
    `videos[]` + `media_order` (JSON describing the carousel kind/idx order)
    when the post contains any video children. The Pi handles both shapes.
    """
    import json as _json

    async with AsyncSessionLocal() as db:
        post = await db.get(InstagramPost, post_id)
        if not post or post.outpost_id:
            return
        media_refs = await load_media_refs(post, db)
        if not media_refs:
            await _record_failure(post_id, "Feed post has no media items")
            return

        caption = post.caption or ""
        scheduled_at = post.scheduled_at
        reel_companion = find_companion(post, "reel")
        story_companion = find_companion(post, "story")
        reel_video_id = reel_companion.video_id if reel_companion else None
        reel_publish_at = (
            companion_at(post.scheduled_at, reel_companion.delay_minutes, post.companion_time)
            if reel_companion and reel_companion.delay_minutes is not None
            else None
        )
        story_publish_at = (
            companion_at(post.scheduled_at, story_companion.delay_minutes, post.companion_time)
            if story_companion and story_companion.delay_minutes is not None
            else None
        )

        if reel_video_id is not None and reel_publish_at is not None:
            vid = await db.get(Video, reel_video_id)
            if not vid or vid.status != "done" or not vid.filepath:
                await _record_failure(
                    post_id,
                    f"Referenced reel video {reel_video_id} is not ready",
                )
                return
            reel_video_path: Path | None = settings.storage_dir / vid.filepath
        else:
            reel_video_path = None

    image_refs = [r for r in media_refs if r.kind == "image"]
    video_refs = [r for r in media_refs if r.kind == "video"]
    is_mixed = bool(video_refs)

    # ── Re-encode images (off the DB session) ─────────────────────────────
    encoded_images: list[tuple[bytes, str]] = []
    try:
        for ref in image_refs:
            src = settings.storage_dir / ref.filepath
            jpg_bytes, jpg_name = await prepare_for_upload(src)
            encoded_images.append((jpg_bytes, jpg_name))
    except Exception as exc:
        await _record_failure(post_id, f"Image re-encode failed: {exc}")
        return

    # ── Resolve carousel video paths (transcoded to H.264 if needed) ──────
    # Meta rejects HEVC/10-bit with error 2207082. ensure_ig_compatible
    # returns the source unchanged when it's already H.264/yuv420p,
    # otherwise produces a cached `<stem>_ig.mp4` sibling. Streamed from
    # disk at upload time below rather than read fully into RAM here — a
    # 4-video carousel + reel can be several hundred MB.
    video_paths: list[Path] = []
    try:
        for ref in video_refs:
            src = settings.storage_dir / ref.filepath
            video_paths.append(await ensure_ig_compatible(src))
    except Exception as exc:
        await _record_failure(post_id, f"Video read failed: {exc}")
        return

    # ── Resolve the reel video — reuse stored video if picked, else render slideshow ─
    reel_path: Path | None = None
    if reel_video_path is not None:
        try:
            reel_path = await ensure_ig_compatible(reel_video_path)
        except Exception as exc:
            await _record_failure(post_id, f"Reel video transcode failed: {exc}")
            return
        logger.info("dispatch_to_outpost %s — reusing stored video %s", post_id, reel_path.name)
    elif reel_publish_at is not None:
        if not image_refs:
            await _record_failure(
                post_id,
                "Reel companion slideshow needs image children — none in this post",
            )
            return
        try:
            image_paths = [settings.storage_dir / r.filepath for r in image_refs]
            settings.reels_dir.mkdir(parents=True, exist_ok=True)
            reel_path = await generate_slideshow(
                image_paths,
                settings.reels_dir,
                ffmpeg_path=settings.ffmpeg_path,
            )
        except Exception as exc:
            await _record_failure(post_id, f"Slideshow render failed: {exc}")
            return

    data = {"caption": caption, "scheduled_at": _iso(scheduled_at)}
    if reel_publish_at is not None:
        data["reel_publish_at"] = _iso(reel_publish_at)
    if story_publish_at is not None:
        data["story_publish_at"] = _iso(story_publish_at)

    # Only emit media_order on mixed posts. Image-only stays on the legacy
    # shape so an outdated Pi (without the mixed-media patch) keeps working.
    if is_mixed:
        order: list[dict] = []
        img_idx = vid_idx = 0
        for ref in media_refs:
            if ref.kind == "image":
                order.append({"kind": "image", "idx": img_idx})
                img_idx += 1
            else:
                order.append({"kind": "video", "idx": vid_idx})
                vid_idx += 1
        data["media_order"] = _json.dumps(order)

    # ── Multipart upload to /enqueue ──────────────────────────────────────
    # Videos/reel are opened as file handles (not read into bytes) so httpx
    # streams the multipart body from disk instead of holding every clip
    # resident in RAM at once.
    with ExitStack() as stack:
        files: list[tuple[str, tuple[str, object, str]]] = [
            ("images", (name, jpg, "image/jpeg")) for jpg, name in encoded_images
        ]
        for vid_path in video_paths:
            fh = stack.enter_context(open(vid_path, "rb"))
            files.append(("videos", (vid_path.name, fh, "video/mp4")))
        if reel_path is not None:
            fh = stack.enter_context(open(reel_path, "rb"))
            files.append(("reel_mp4", (reel_path.name, fh, "video/mp4")))

        result = await _post_to_outpost(
            post_id, "/enqueue",
            data=data, files=files, error_label="Outpost dispatch failed",
        )
    if result is None:
        return
    outpost_id, body = result

    await _finalize_outpost_dispatch(
        post_id,
        outpost_id=outpost_id,
        body=body,
        has_reel=reel_path is not None,
        reel_filename=None,
        story_publish_at=story_publish_at,
    )
    logger.info(
        "dispatch_to_outpost %s → outpost_id=%s (mixed=%s, reel=%s)",
        post_id, outpost_id, is_mixed, reel_path is not None,
    )


async def _dispatch_reel_only(post_id: uuid.UUID) -> None:
    """
    Concatenate the post's reel_video_ids into one 1080×1920 MP4 and POST
    to the Pi /enqueue-reel endpoint. No images, optional story companion.
    """
    async with AsyncSessionLocal() as db:
        post = await db.get(InstagramPost, post_id)
        if not post or post.outpost_id:
            return
        video_ids = list(post.reel_video_ids or [])
        if not video_ids:
            await _record_failure(post_id, "kind='reel' but reel_video_ids is empty")
            return
        if len(video_ids) > 4:
            await _record_failure(post_id, f"Too many reel_video_ids ({len(video_ids)}, max 4)")
            return

        vid_result = await db.execute(select(Video).where(Video.id.in_(video_ids)))
        videos = {v.id: v for v in vid_result.scalars().all()}
        ordered = [videos.get(vid) for vid in video_ids]
        if any(v is None or v.status != "done" or not v.filepath for v in ordered):
            await _record_failure(post_id, "One or more reel source videos are not ready")
            return

        caption = post.caption or ""
        scheduled_at = post.scheduled_at
        story_companion = find_companion(post, "story")
        story_publish_at = (
            companion_at(post.scheduled_at, story_companion.delay_minutes, post.companion_time)
            if story_companion and story_companion.delay_minutes is not None
            else None
        )
        source_paths = [settings.storage_dir / v.filepath for v in ordered]

    # ── Concatenate to a 9:16 reel MP4 (off the DB session) ────────────────
    settings.reels_dir.mkdir(parents=True, exist_ok=True)
    reel_path = settings.reels_dir / f"reel_only_{post_id.hex}.mp4"
    try:
        await concat_reel_videos(source_paths, reel_path, ffmpeg_path=settings.ffmpeg_path)
    except Exception as exc:
        await _record_failure(
            post_id,
            f"Reel concat failed: {type(exc).__name__}: {exc}",
        )
        return

    # ── Multipart upload to /enqueue-reel ──────────────────────────────────
    # Streamed from disk (not read into bytes) so httpx doesn't hold the
    # whole concatenated reel resident in RAM.
    data = {"caption": caption, "scheduled_at": _iso(scheduled_at)}
    if story_publish_at is not None:
        data["story_publish_at"] = _iso(story_publish_at)

    with open(reel_path, "rb") as fh:
        result = await _post_to_outpost(
            post_id, "/enqueue-reel",
            data=data,
            files=[("reel_mp4", (reel_path.name, fh, "video/mp4"))],
            error_label="Outpost reel dispatch failed",
        )
    if result is None:
        return
    outpost_id, body = result

    await _finalize_outpost_dispatch(
        post_id,
        outpost_id=outpost_id,
        body=body,
        has_reel=True,                      # kind='reel' always has a reel
        reel_filename=reel_path.name,
        story_publish_at=story_publish_at,
    )
    logger.info(
        "_dispatch_reel_only %s → outpost_id=%s (n_videos=%d)",
        post_id, outpost_id, len(video_ids),
    )


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
            reel = find_companion(post, "reel")
            story = find_companion(post, "story")
            feed_done = (post.outpost_status or "") in terminal_feed
            reel_done = (
                not reel or reel.outpost_status is None
                or reel.outpost_status in terminal_reel
            )
            story_done = (
                not story or story.status is None
                or story.status in terminal_story
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
        reel = find_companion(post, "reel")
        if remote_reel != (reel.outpost_status if reel else None):
            reel = get_or_create_companion(post, "reel")
            reel.outpost_status = remote_reel
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
            reel = get_or_create_companion(post, "reel")
            reel_media = remote.get("reel_media_id")
            if reel_media and reel.media_id != reel_media:
                reel.media_id = reel_media
                changed = True
            if reel.status != "posted":
                reel.status = "posted"
                if not reel.scheduled_at:
                    reel.scheduled_at = now
                changed = True
        elif remote_reel == "failed":
            reel = find_companion(post, "reel")
            if not reel or reel.status != "failed":
                reel = get_or_create_companion(post, "reel")
                reel.status = "failed"
                changed = True

        # Story companion mirror.
        remote_story = remote.get("story_status")
        story = find_companion(post, "story")
        if remote_story and remote_story != (story.status if story else None):
            story = get_or_create_companion(post, "story")
            story.status = remote_story
            changed = True
        remote_story_media = remote.get("story_media_ids")
        if remote_story_media and remote_story_media != (story.media_ids if story else None):
            story = get_or_create_companion(post, "story")
            story.media_ids = remote_story_media
            changed = True
        if remote_story == "posted" and not (story.scheduled_at if story else None):
            story = get_or_create_companion(post, "story")
            story.scheduled_at = now
            changed = True

        if changed:
            post.updated_at = now
            await db.commit()
            reel = find_companion(post, "reel")
            logger.info(
                "outpost sync %s → status=%s reel=%s media=%s",
                post_id, post.status, reel.status if reel else None, post.instagram_media_id,
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


async def _post_to_outpost(
    post_id: uuid.UUID,
    path: str,
    *,
    data: dict,
    files: list,
    error_label: str,
) -> tuple[str, dict] | None:
    """Multipart POST to the Pi outpost. On success returns (outpost_id, body).
    On any failure (HTTP/network/missing id), records the failure on the row
    and returns None."""
    try:
        async with httpx.AsyncClient(timeout=300) as client:
            r = await client.post(
                f"{_base()}{path}", headers=_headers(), data=data, files=files,
            )
        if r.status_code >= 300:
            await _record_failure(
                post_id, f"Outpost {path} HTTP {r.status_code}: {r.text[:300]}",
            )
            return None
        body = r.json()
        outpost_id = body.get("id")
        if not outpost_id:
            await _record_failure(post_id, f"Outpost response missing id: {body}")
            return None
        return outpost_id, body
    except Exception as exc:
        await _record_failure(
            post_id,
            f"{error_label}: {type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        )
        return None


async def _finalize_outpost_dispatch(
    post_id: uuid.UUID,
    *,
    outpost_id: str,
    body: dict,
    has_reel: bool,
    reel_filename: str | None,
    story_publish_at: datetime | None,
) -> None:
    """Mirror a successful dispatch into the local row. Idempotency / commit
    ownership: opens its own session, commits exactly once."""
    now = datetime.now(timezone.utc)
    async with AsyncSessionLocal() as db:
        fresh = await db.get(InstagramPost, post_id)
        if not fresh:
            return
        fresh.outpost_id          = outpost_id
        fresh.outpost_status      = body.get("status", "queued")
        if has_reel:
            get_or_create_companion(fresh, "reel").outpost_status = "pending"
        else:
            existing_reel = find_companion(fresh, "reel")
            if existing_reel:
                existing_reel.outpost_status = None
        if reel_filename is not None:
            get_or_create_companion(fresh, "reel").video_filename = reel_filename
        existing_story = find_companion(fresh, "story")
        if story_publish_at is not None and not (existing_story.status if existing_story else None):
            story = get_or_create_companion(fresh, "story")
            story.status       = "pending"
            story.scheduled_at = story_publish_at
        fresh.outpost_dispatched_at = now
        fresh.error                 = None
        fresh.updated_at            = now
        await db.commit()


def _iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()
