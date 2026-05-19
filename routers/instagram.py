"""
Instagram scheduling API — schedule, list, update, and delete planned posts.
Supports single-image and carousel (multi-image) posts.

Actual posting to the Instagram Graph API requires a public image URL (e.g.
via Cloudflare tunnel). Use the /post-now endpoint once that is configured.
"""
import asyncio
import uuid
import logging
from datetime import datetime, timezone
from typing import Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select, desc, asc
from sqlalchemy.ext.asyncio import AsyncSession

from core.auth import require_auth
from core.config import settings
from core.db import get_db
from core.models import InstagramPost, Image, Video
from core.scheduling import companion_at
from services.instagram.graph import missing_config
from services.instagram.publisher import publish_feed, schedule_feed
from services.instagram.reel import schedule_reel
from services.instagram import outpost as outpost_svc

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/instagram", dependencies=[Depends(require_auth)])


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class PostCreate(BaseModel):
    kind: Optional[str] = "feed"                          # "feed" (default, image-based) | "reel" (standalone, 1–4 videos)
    image_id: Optional[uuid.UUID] = None                  # primary / first image (required for kind='feed')
    carousel_image_ids: Optional[list[uuid.UUID]] = None  # additional images (2nd…10th) — kind='feed' only
    reel_video_ids: Optional[list[uuid.UUID]] = None      # 1–4 source videos to concat for kind='reel'
    caption: Optional[str] = None
    scheduled_at: datetime          # client computes this (incl. offset logic)
    story_delay_minutes: Optional[int] = None  # null = off; N = N min after feed/reel
    reel_delay_minutes:  Optional[int] = None  # null = off; N = N min after feed (kind='feed' only)
    companion_time: Optional[str] = "18:23"    # "HH:MM" — day+ delays snap to this time
    reel_video_id: Optional[uuid.UUID] = None  # use an existing generated video for the companion Reel (kind='feed' only)
    dispatch_target: Optional[str] = "local"   # "local" (default) | "outpost" (Pi cloud-schedule)


class PostUpdate(BaseModel):
    caption: Optional[str] = None
    scheduled_at: Optional[datetime] = None
    status: Optional[str] = None    # scheduled | posted | cancelled
    carousel_image_ids: Optional[list[uuid.UUID]] = None
    story_delay_minutes: Optional[int] = None  # use model_fields_set to detect explicit null
    reel_delay_minutes:  Optional[int] = None
    companion_time: Optional[str] = None
    reel_video_id: Optional[uuid.UUID] = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _all_image_ids(post: InstagramPost) -> list[uuid.UUID]:
    """Return ordered list of all image IDs for a feed post (primary first).
    Empty list for kind='reel' rows, which have no images."""
    if not post.image_id:
        return []
    ids = [post.image_id]
    if post.carousel_image_ids:
        ids.extend(post.carousel_image_ids)
    return ids


def _serialize(post: InstagramPost, images: dict[uuid.UUID, "Image"] | None = None) -> dict:
    primary = (images or {}).get(post.image_id) if post.image_id else None
    d = {
        "id": str(post.id),
        "kind": post.kind,
        "image_id": str(post.image_id) if post.image_id else None,
        "carousel_image_ids": [str(i) for i in post.carousel_image_ids] if post.carousel_image_ids else None,
        "reel_video_ids": [str(i) for i in post.reel_video_ids] if post.reel_video_ids else None,
        "is_carousel": bool(post.carousel_image_ids),
        "caption": post.caption,
        "scheduled_at": post.scheduled_at.isoformat(),
        "status": post.status,
        "instagram_media_id": post.instagram_media_id,
        "story_delay_minutes":  post.story_delay_minutes,
        "reel_delay_minutes":   post.reel_delay_minutes,
        "companion_time":       post.companion_time,
        "story_scheduled_at":   post.story_scheduled_at.isoformat() if post.story_scheduled_at else None,
        "reel_scheduled_at":    post.reel_scheduled_at.isoformat() if post.reel_scheduled_at else None,
        "story_status":    post.story_status,
        "story_media_ids": post.story_media_ids,
        "reel_status":     post.reel_status,
        "reel_media_id":   post.reel_media_id,
        "reel_video_id":   str(post.reel_video_id) if post.reel_video_id else None,
        "feed_creation_id": post.feed_creation_id,
        "reel_creation_id": post.reel_creation_id,
        "remote_scheduled": bool(post.feed_creation_id),
        "dispatch_target":       post.dispatch_target,
        "outpost_id":            post.outpost_id,
        "outpost_status":        post.outpost_status,
        "outpost_reel_status":   post.outpost_reel_status,
        "outpost_dispatched_at": post.outpost_dispatched_at.isoformat() if post.outpost_dispatched_at else None,
        "error":           post.error,
        "created_at": post.created_at.isoformat(),
        "updated_at": post.updated_at.isoformat(),
    }
    if images is not None:
        # Primary image info (for timeline preview)
        if primary:
            d["image"] = {
                "filename": primary.filename,
                "title": primary.title,
                "url": f"/api/image/{primary.filename}",
                "thumb_url": f"/api/image/{primary.filename}/thumb",
            }
        # All carousel images
        all_ids = _all_image_ids(post)
        d["carousel_images"] = [
            {
                "filename": images[i].filename,
                "thumb_url": f"/api/image/{images[i].filename}/thumb",
                "url": f"/api/image/{images[i].filename}",
            }
            for i in all_ids if i in images
        ]
    return d


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/posts")
async def list_posts(
    status: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
):
    """Return all scheduled posts ordered by scheduled_at ASC, with image info."""
    stmt = select(InstagramPost).order_by(asc(InstagramPost.scheduled_at))
    if status:
        stmt = stmt.where(InstagramPost.status == status)
    result = await db.execute(stmt)
    posts = result.scalars().all()

    # Batch-load all referenced images (kind='reel' rows skip — they have none)
    image_ids: set[uuid.UUID] = set()
    for p in posts:
        if p.image_id:
            image_ids.add(p.image_id)
        if p.carousel_image_ids:
            image_ids.update(p.carousel_image_ids)

    images: dict[uuid.UUID, Image] = {}
    if image_ids:
        img_result = await db.execute(select(Image).where(Image.id.in_(image_ids)))
        for img in img_result.scalars().all():
            images[img.id] = img

    return [_serialize(p, images) for p in posts]


@router.get("/last-post")
async def get_last_post(db: AsyncSession = Depends(get_db)):
    """Return the latest scheduled_at among non-cancelled posts, or null."""
    stmt = (
        select(InstagramPost)
        .where(InstagramPost.status != "cancelled")
        .order_by(desc(InstagramPost.scheduled_at))
        .limit(1)
    )
    result = await db.execute(stmt)
    post = result.scalar_one_or_none()
    return {"scheduled_at": post.scheduled_at.isoformat() if post else None}


@router.post("/posts", status_code=201)
async def create_post(body: PostCreate, db: AsyncSession = Depends(get_db)):
    kind = (body.kind or "feed").lower()
    if kind not in {"feed", "reel"}:
        raise HTTPException(status_code=400, detail="kind must be 'feed' or 'reel'")

    # Normalise to UTC
    scheduled_at = body.scheduled_at
    if scheduled_at.tzinfo is None:
        scheduled_at = scheduled_at.replace(tzinfo=timezone.utc)

    dispatch_target = (body.dispatch_target or "local").lower()
    if dispatch_target not in {"local", "outpost"}:
        raise HTTPException(status_code=400, detail="dispatch_target must be 'local' or 'outpost'")

    if kind == "reel":
        # Standalone reel: validate the source-video list; outpost is the only path.
        if dispatch_target != "outpost":
            raise HTTPException(
                status_code=400,
                detail="Reel-only posts must use dispatch_target='outpost' (the local "
                       "Graph-API path can't schedule reels).",
            )
        if not body.reel_video_ids:
            raise HTTPException(status_code=400, detail="kind='reel' requires reel_video_ids")
        if len(body.reel_video_ids) > 4:
            raise HTTPException(status_code=400, detail="At most 4 source videos per reel")
        for vid_id in body.reel_video_ids:
            vid = await db.get(Video, vid_id)
            if not vid or vid.status != "done" or not vid.filepath:
                raise HTTPException(status_code=404, detail=f"Video {vid_id} not found or not ready")
        if body.reel_delay_minutes is not None:
            raise HTTPException(
                status_code=400,
                detail="reel_delay_minutes is for feed-companion reels; on kind='reel' the post itself is the reel.",
            )
        if body.image_id or body.carousel_image_ids:
            raise HTTPException(
                status_code=400,
                detail="kind='reel' must not include image_id / carousel_image_ids.",
            )
    else:
        # Feed: keep the original image-based validation.
        if not body.image_id:
            raise HTTPException(status_code=400, detail="kind='feed' requires image_id")
        if not await db.get(Image, body.image_id):
            raise HTTPException(status_code=404, detail="Image not found")
        if body.carousel_image_ids:
            if len(body.carousel_image_ids) > 9:
                raise HTTPException(status_code=400, detail="Carousel supports at most 10 images (1 primary + 9 additional)")
            for cid in body.carousel_image_ids:
                if not await db.get(Image, cid):
                    raise HTTPException(status_code=404, detail=f"Carousel image {cid} not found")
        if body.reel_video_ids:
            raise HTTPException(status_code=400, detail="reel_video_ids is only valid for kind='reel'")

    if dispatch_target == "outpost":
        miss = outpost_svc.missing_config()
        if miss:
            raise HTTPException(
                status_code=400,
                detail=f"Outpost not configured: {', '.join(miss)} — set in .env",
            )

    post = InstagramPost(
        kind=kind,
        image_id=body.image_id if kind == "feed" else None,
        carousel_image_ids=(body.carousel_image_ids or None) if kind == "feed" else None,
        reel_video_ids=(body.reel_video_ids or None) if kind == "reel" else None,
        caption=body.caption,
        scheduled_at=scheduled_at,
        status="scheduled",
        story_delay_minutes=body.story_delay_minutes,
        reel_delay_minutes=body.reel_delay_minutes if kind == "feed" else None,
        companion_time=body.companion_time,
        reel_video_id=body.reel_video_id if kind == "feed" else None,
        dispatch_target=dispatch_target,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    db.add(post)
    await db.commit()
    await db.refresh(post)
    logger.info(
        "Scheduled Instagram %s post %s for %s (target=%s)",
        kind, post.id, post.scheduled_at, dispatch_target,
    )

    # Outpost path: package + upload to Pi. Local path: try Graph-side
    # scheduled_publish_time (whitelist-gated, falls through to local scheduler).
    if dispatch_target == "outpost":
        asyncio.create_task(outpost_svc.dispatch_to_outpost(post.id))
    else:
        asyncio.create_task(_remote_schedule(post.id))

    images = await _load_images_for_post(post, db)
    return _serialize(post, images)


async def _load_images_for_post(post: InstagramPost, db: AsyncSession) -> dict[uuid.UUID, Image]:
    """Batch-fetch all referenced images for serialization. Empty for reels."""
    if post.kind == "reel" or not post.image_id:
        return {}
    all_ids = _all_image_ids(post)
    img_result = await db.execute(select(Image).where(Image.id.in_(all_ids)))
    return {i.id: i for i in img_result.scalars().all()}


async def _remote_schedule(post_id: uuid.UUID) -> None:
    """Try to register both feed and reel containers with Instagram."""
    try:
        await schedule_feed(post_id)
    except Exception as exc:
        logger.error("remote feed scheduling failed for %s: %s", post_id, exc)
    try:
        await schedule_reel(post_id)
    except Exception as exc:
        logger.error("remote reel scheduling failed for %s: %s", post_id, exc)


@router.patch("/posts/{post_id}")
async def update_post(
    post_id: uuid.UUID,
    body: PostUpdate,
    db: AsyncSession = Depends(get_db),
):
    post = await db.get(InstagramPost, post_id)
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")

    # Outpost posts use a separate edit path — caption / scheduled_at /
    # companion_time / reel retime push to the Pi. Image swaps and
    # reel add/remove still require delete + recreate.
    if post.dispatch_target == "outpost" and post.outpost_id:
        return await _update_outpost_post(post, body, db)

    # Detect changes that invalidate an existing Instagram-side schedule —
    # caption / images / time / reel-source — so we can re-create the container.
    invalidates_remote = False

    if body.caption is not None and body.caption != post.caption:
        post.caption = body.caption
        invalidates_remote = True
    if body.scheduled_at is not None:
        scheduled_at = body.scheduled_at
        if scheduled_at.tzinfo is None:
            scheduled_at = scheduled_at.replace(tzinfo=timezone.utc)
        if scheduled_at != post.scheduled_at:
            post.scheduled_at = scheduled_at
            invalidates_remote = True
    if body.status is not None:
        allowed = {"scheduled", "posted", "cancelled", "failed"}
        if body.status not in allowed:
            raise HTTPException(status_code=400, detail=f"status must be one of {allowed}")
        post.status = body.status
    if body.carousel_image_ids is not None:
        post.carousel_image_ids = body.carousel_image_ids or None
        invalidates_remote = True
    if "story_delay_minutes" in body.model_fields_set:
        post.story_delay_minutes = body.story_delay_minutes
    if "reel_delay_minutes" in body.model_fields_set:
        if body.reel_delay_minutes != post.reel_delay_minutes:
            invalidates_remote = True
        post.reel_delay_minutes = body.reel_delay_minutes
    if "companion_time" in body.model_fields_set:
        if body.companion_time != post.companion_time:
            invalidates_remote = True
        post.companion_time = body.companion_time
    if "reel_video_id" in body.model_fields_set:
        if body.reel_video_id != post.reel_video_id:
            invalidates_remote = True
        post.reel_video_id = body.reel_video_id

    creation_ids_to_drop: list[str] = []
    if invalidates_remote and post.status == "scheduled":
        if post.feed_creation_id:
            creation_ids_to_drop.append(post.feed_creation_id)
            post.feed_creation_id = None
        if post.reel_creation_id:
            creation_ids_to_drop.append(post.reel_creation_id)
            post.reel_creation_id = None
            post.reel_status = None

    post.updated_at = datetime.now(timezone.utc)
    await db.commit()

    if creation_ids_to_drop:
        asyncio.create_task(_drop_remote_containers(creation_ids_to_drop))
    if invalidates_remote and post.status == "scheduled":
        asyncio.create_task(_remote_schedule(post.id))

    images = await _load_images_for_post(post, db)
    return _serialize(post, images)


@router.delete("/posts/{post_id}", status_code=204)
async def delete_post(post_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    post = await db.get(InstagramPost, post_id)
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    creation_ids = [c for c in (post.feed_creation_id, post.reel_creation_id) if c]
    reel_filename = post.reel_video_filename
    outpost_id = post.outpost_id
    await db.delete(post)
    await db.commit()
    logger.info(f"Deleted Instagram post {post_id}")

    if creation_ids:
        asyncio.create_task(_drop_remote_containers(creation_ids))
    if outpost_id:
        asyncio.create_task(outpost_svc.cancel_on_outpost(outpost_id))
    if reel_filename:
        try:
            (settings.reels_dir / reel_filename).unlink(missing_ok=True)
        except Exception as exc:
            logger.warning("Could not delete reel slideshow %s: %s", reel_filename, exc)


async def _update_outpost_post(
    post: InstagramPost,
    body: PostUpdate,
    db: AsyncSession,
) -> dict:
    """
    Apply edits to an outpost-dispatched post. Allowed: caption, scheduled_at,
    companion_time, reel_delay retime (existing reel only). Forbidden: image
    changes, reel add/remove, source video swap, story delays.
    """
    requested = set(body.model_fields_set) - {"status"}

    # Forbidden fields are only flagged when the *value* actually changes —
    # the frontend always serialises every field, so presence alone is noise.
    new_carousel = body.carousel_image_ids or None
    old_carousel = list(post.carousel_image_ids) if post.carousel_image_ids else None
    bad: set[str] = set()
    if "carousel_image_ids" in requested and new_carousel != old_carousel:
        bad.add("carousel_image_ids")
    if "reel_video_id" in requested and body.reel_video_id != post.reel_video_id:
        bad.add("reel_video_id")
    if bad:
        raise HTTPException(
            status_code=409,
            detail=(f"Cannot edit {sorted(bad)} on cloud-scheduled post — "
                    f"cancel and create a new post instead."),
        )

    # Reel and Story can only be retimed, not added/removed after dispatch
    # (add would need re-upload; remove would orphan an in-flight container).
    if "reel_delay_minutes" in requested:
        was_set = post.reel_delay_minutes is not None
        now_set = body.reel_delay_minutes is not None
        if was_set != now_set:
            verb = "add" if now_set else "remove"
            raise HTTPException(
                status_code=409,
                detail=(f"Cannot {verb} reel on cloud-scheduled post — "
                        f"cancel and create a new post instead."),
            )
    if "story_delay_minutes" in requested:
        was_set = post.story_delay_minutes is not None
        now_set = body.story_delay_minutes is not None
        if was_set != now_set:
            verb = "add" if now_set else "remove"
            raise HTTPException(
                status_code=409,
                detail=(f"Cannot {verb} story on cloud-scheduled post — "
                        f"cancel and create a new post instead."),
            )

    # Apply local mutations (only the editable subset).
    pi_caption = pi_scheduled_at = pi_reel_publish_at = pi_story_publish_at = None

    if "caption" in requested and body.caption != post.caption:
        post.caption = body.caption
        pi_caption = body.caption or ""

    if "scheduled_at" in requested and body.scheduled_at is not None:
        new_sched = body.scheduled_at
        if new_sched.tzinfo is None:
            new_sched = new_sched.replace(tzinfo=timezone.utc)
        if new_sched != post.scheduled_at:
            post.scheduled_at = new_sched
            pi_scheduled_at = new_sched

    if "companion_time" in requested and body.companion_time != post.companion_time:
        post.companion_time = body.companion_time

    if "reel_delay_minutes" in requested and body.reel_delay_minutes != post.reel_delay_minutes:
        post.reel_delay_minutes = body.reel_delay_minutes
    if "story_delay_minutes" in requested and body.story_delay_minutes != post.story_delay_minutes:
        post.story_delay_minutes = body.story_delay_minutes

    # If a companion exists and any timing input changed, recompute its
    # publish time and push the new value to the Pi.
    reel_timing = {"scheduled_at", "companion_time", "reel_delay_minutes"} & requested
    story_timing = {"scheduled_at", "companion_time", "story_delay_minutes"} & requested

    if post.reel_delay_minutes is not None and reel_timing:
        new_reel_at = companion_at(post.scheduled_at, post.reel_delay_minutes, post.companion_time)
        if new_reel_at != post.reel_scheduled_at:
            post.reel_scheduled_at = new_reel_at
            pi_reel_publish_at = new_reel_at

    if post.story_delay_minutes is not None and story_timing:
        new_story_at = companion_at(post.scheduled_at, post.story_delay_minutes, post.companion_time)
        if new_story_at != post.story_scheduled_at:
            post.story_scheduled_at = new_story_at
            pi_story_publish_at = new_story_at

    if all(v is None for v in (pi_caption, pi_scheduled_at, pi_reel_publish_at, pi_story_publish_at)):
        # Nothing to send to the Pi — return current state.
        return _serialize(post, await _load_images_for_post(post, db))

    try:
        await outpost_svc.update_on_outpost(
            post.outpost_id,
            caption=pi_caption,
            scheduled_at=pi_scheduled_at,
            reel_publish_at=pi_reel_publish_at,
            story_publish_at=pi_story_publish_at,
        )
    except RuntimeError as exc:
        # Don't commit local edits if the Pi rejected them — keeps state aligned.
        await db.rollback()
        raise HTTPException(status_code=502, detail=f"Outpost edit failed: {exc}")

    post.updated_at = datetime.now(timezone.utc)
    await db.commit()

    return _serialize(post, await _load_images_for_post(post, db))


async def _drop_remote_containers(creation_ids: list[str]) -> None:
    """
    Best-effort cancellation of scheduled Instagram containers. The Graph API
    accepts DELETE on a container ID as long as it has not yet published.
    Failures are logged but never raise — the post is already gone locally.
    """
    if missing_config():
        return
    token = settings.instagram_access_token
    graph = settings.instagram_graph_api_base
    async with httpx.AsyncClient(timeout=15) as client:
        for cid in creation_ids:
            try:
                r = await client.delete(f"{graph}/{cid}", params={"access_token": token})
                logger.info("Dropped IG container %s → %s", cid, r.status_code)
            except Exception as exc:
                logger.warning("Failed to drop IG container %s: %s", cid, exc)


@router.get("/outpost-status")
async def outpost_status():
    """Reachability + config probe for the cloud-schedule toggle."""
    return await outpost_svc.health()


@router.post("/posts/{post_id}/post-now")
async def post_now(post_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """Publish a scheduled post to Instagram immediately via the Graph API."""
    missing = missing_config()
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing config: {', '.join(missing)} — set these in .env",
        )

    post = await db.get(InstagramPost, post_id)
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    if post.status == "posted":
        raise HTTPException(status_code=400, detail="Post already published")

    # The publisher only acts on rows in 'scheduled' state — flip retries
    # ('failed', 'cancelled') back so the same code path handles them.
    if post.status != "scheduled":
        post.status = "scheduled"
        post.updated_at = datetime.now(timezone.utc)
        await db.commit()

    status, media_id = await publish_feed(post_id)
    if status == "posted":
        return {"media_id": media_id, "status": "posted"}

    # Pull the recorded error message back for the HTTP response
    await db.refresh(post)
    detail = post.error or "Publish failed (no error recorded)"
    logger.error("Instagram post-now %s failed: %s", post_id, detail)
    raise HTTPException(status_code=502, detail=detail)
