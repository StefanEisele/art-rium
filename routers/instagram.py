"""
Instagram scheduling API — schedule, list, update, and delete planned posts.
Supports single-image and carousel (multi-image) posts.

Actual posting to the Instagram Graph API requires a public image URL (e.g.
via Cloudflare tunnel). Use the /post-now endpoint once that is configured.
"""
import uuid
import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select, desc, asc
from sqlalchemy.ext.asyncio import AsyncSession

from core.auth import require_auth
from core.db import get_db
from core.models import InstagramPost, Image
from services.instagram.graph import missing_config
from services.instagram.publisher import publish_feed

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/instagram", dependencies=[Depends(require_auth)])


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class PostCreate(BaseModel):
    image_id: uuid.UUID                                   # primary / first image
    carousel_image_ids: Optional[list[uuid.UUID]] = None  # additional images (2nd…10th)
    caption: Optional[str] = None
    scheduled_at: datetime          # client computes this (incl. offset logic)
    story_delay_minutes: Optional[int] = None  # null = off; N = N min after feed
    reel_delay_minutes:  Optional[int] = None  # null = off; N = N min after feed
    companion_time: Optional[str] = "18:23"    # "HH:MM" — day+ delays snap to this time
    reel_video_id: Optional[uuid.UUID] = None  # use an existing generated video for the Reel


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
    """Return ordered list of all image IDs for a post (primary first)."""
    ids = [post.image_id]
    if post.carousel_image_ids:
        ids.extend(post.carousel_image_ids)
    return ids


def _serialize(post: InstagramPost, images: dict[uuid.UUID, "Image"] | None = None) -> dict:
    primary = (images or {}).get(post.image_id)
    d = {
        "id": str(post.id),
        "image_id": str(post.image_id),
        "carousel_image_ids": [str(i) for i in post.carousel_image_ids] if post.carousel_image_ids else None,
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

    # Batch-load all referenced images
    image_ids: set[uuid.UUID] = set()
    for p in posts:
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
    # Validate primary image
    img = await db.get(Image, body.image_id)
    if not img:
        raise HTTPException(status_code=404, detail="Image not found")

    # Validate carousel images if provided
    if body.carousel_image_ids:
        if len(body.carousel_image_ids) > 9:
            raise HTTPException(status_code=400, detail="Carousel supports at most 10 images (1 primary + 9 additional)")
        for cid in body.carousel_image_ids:
            if not await db.get(Image, cid):
                raise HTTPException(status_code=404, detail=f"Carousel image {cid} not found")

    # Normalise to UTC
    scheduled_at = body.scheduled_at
    if scheduled_at.tzinfo is None:
        scheduled_at = scheduled_at.replace(tzinfo=timezone.utc)

    post = InstagramPost(
        image_id=body.image_id,
        carousel_image_ids=body.carousel_image_ids or None,
        caption=body.caption,
        scheduled_at=scheduled_at,
        status="scheduled",
        story_delay_minutes=body.story_delay_minutes,
        reel_delay_minutes=body.reel_delay_minutes,
        companion_time=body.companion_time,
        reel_video_id=body.reel_video_id,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    db.add(post)
    await db.commit()
    await db.refresh(post)
    logger.info(f"Scheduled Instagram post {post.id} for {post.scheduled_at} (carousel={bool(post.carousel_image_ids)})")

    # Build images dict for serialization
    all_ids = _all_image_ids(post)
    img_result = await db.execute(select(Image).where(Image.id.in_(all_ids)))
    images = {i.id: i for i in img_result.scalars().all()}
    return _serialize(post, images)


@router.patch("/posts/{post_id}")
async def update_post(
    post_id: uuid.UUID,
    body: PostUpdate,
    db: AsyncSession = Depends(get_db),
):
    post = await db.get(InstagramPost, post_id)
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")

    if body.caption is not None:
        post.caption = body.caption
    if body.scheduled_at is not None:
        scheduled_at = body.scheduled_at
        if scheduled_at.tzinfo is None:
            scheduled_at = scheduled_at.replace(tzinfo=timezone.utc)
        post.scheduled_at = scheduled_at
    if body.status is not None:
        allowed = {"scheduled", "posted", "cancelled", "failed"}
        if body.status not in allowed:
            raise HTTPException(status_code=400, detail=f"status must be one of {allowed}")
        post.status = body.status
    if body.carousel_image_ids is not None:
        post.carousel_image_ids = body.carousel_image_ids or None
    if "story_delay_minutes" in body.model_fields_set:
        post.story_delay_minutes = body.story_delay_minutes
    if "reel_delay_minutes" in body.model_fields_set:
        post.reel_delay_minutes = body.reel_delay_minutes
    if "companion_time" in body.model_fields_set:
        post.companion_time = body.companion_time
    if "reel_video_id" in body.model_fields_set:
        post.reel_video_id = body.reel_video_id

    post.updated_at = datetime.now(timezone.utc)
    await db.commit()

    all_ids = _all_image_ids(post)
    img_result = await db.execute(select(Image).where(Image.id.in_(all_ids)))
    images = {i.id: i for i in img_result.scalars().all()}
    return _serialize(post, images)


@router.delete("/posts/{post_id}", status_code=204)
async def delete_post(post_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    post = await db.get(InstagramPost, post_id)
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    await db.delete(post)
    await db.commit()
    logger.info(f"Deleted Instagram post {post_id}")


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
