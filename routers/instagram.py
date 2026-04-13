"""
Instagram scheduling API — schedule, list, update, and delete planned posts.
Actual posting to the Instagram Graph API requires a public image URL (e.g.
via Cloudflare tunnel). Use the /post-now endpoint once that is configured.
"""
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
from core.models import InstagramPost, Image

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/instagram", dependencies=[Depends(require_auth)])


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class PostCreate(BaseModel):
    image_id: uuid.UUID
    caption: Optional[str] = None
    scheduled_at: datetime          # client computes this (incl. offset logic)


class PostUpdate(BaseModel):
    caption: Optional[str] = None
    scheduled_at: Optional[datetime] = None
    status: Optional[str] = None    # scheduled | posted | cancelled


# ── Helpers ───────────────────────────────────────────────────────────────────

def _serialize(post: InstagramPost, image: Optional[Image] = None) -> dict:
    d = {
        "id": str(post.id),
        "image_id": str(post.image_id),
        "caption": post.caption,
        "scheduled_at": post.scheduled_at.isoformat(),
        "status": post.status,
        "instagram_media_id": post.instagram_media_id,
        "created_at": post.created_at.isoformat(),
        "updated_at": post.updated_at.isoformat(),
    }
    if image:
        d["image"] = {
            "filename": image.filename,
            "title": image.title,
            "url": f"/api/image/{image.filename}",
            "thumb_url": f"/api/image/{image.filename}/thumb",
        }
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

    # Batch-load images
    image_ids = list({p.image_id for p in posts})
    images: dict[uuid.UUID, Image] = {}
    if image_ids:
        img_result = await db.execute(select(Image).where(Image.id.in_(image_ids)))
        for img in img_result.scalars().all():
            images[img.id] = img

    return [_serialize(p, images.get(p.image_id)) for p in posts]


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
    img = await db.get(Image, body.image_id)
    if not img:
        raise HTTPException(status_code=404, detail="Image not found")

    # Normalise to UTC
    scheduled_at = body.scheduled_at
    if scheduled_at.tzinfo is None:
        scheduled_at = scheduled_at.replace(tzinfo=timezone.utc)

    post = InstagramPost(
        image_id=body.image_id,
        caption=body.caption,
        scheduled_at=scheduled_at,
        status="scheduled",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    )
    db.add(post)
    await db.commit()
    await db.refresh(post)
    logger.info(f"Scheduled Instagram post {post.id} for {post.scheduled_at}")
    return _serialize(post, img)


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
        allowed = {"scheduled", "posted", "cancelled"}
        if body.status not in allowed:
            raise HTTPException(status_code=400, detail=f"status must be one of {allowed}")
        post.status = body.status

    post.updated_at = datetime.now(timezone.utc)
    await db.commit()
    img = await db.get(Image, post.image_id)
    return _serialize(post, img)


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
    """Publish a scheduled post to Instagram via the Graph API."""
    # ── Validate config ──────────────────────────────────────────────────────
    missing = [k for k, v in {
        "INSTAGRAM_USER_ID": settings.instagram_user_id,
        "INSTAGRAM_ACCESS_TOKEN": settings.instagram_access_token,
        "PUBLIC_BASE_URL": settings.public_base_url,
    }.items() if not v]
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

    img = await db.get(Image, post.image_id)
    if not img:
        raise HTTPException(status_code=404, detail="Image not found")

    # ── Build public image URL ───────────────────────────────────────────────
    base = settings.public_base_url.rstrip("/")
    image_url = f"{base}/share/image/{img.filename}"
    if settings.image_share_token:
        image_url += f"?token={settings.image_share_token}"

    logger.info(f"Posting to Instagram — image_url={image_url}")

    graph = settings.instagram_graph_api_base
    uid   = settings.instagram_user_id
    token = settings.instagram_access_token

    async with httpx.AsyncClient(timeout=30) as client:
        # Step 1 — create media container
        r1 = await client.post(
            f"{graph}/{uid}/media",
            data={
                "image_url":    image_url,
                "caption":      post.caption or "",
                "access_token": token,
            },
        )
        body1 = r1.json()
        if "error" in body1:
            err = body1["error"]
            logger.error(f"Instagram container error: {err}")
            raise HTTPException(status_code=502, detail=err.get("message", "Instagram API error"))
        container_id = body1["id"]
        logger.info(f"Container created: {container_id}")

        # Step 2 — publish
        r2 = await client.post(
            f"{graph}/{uid}/media_publish",
            data={
                "creation_id":  container_id,
                "access_token": token,
            },
        )
        body2 = r2.json()
        if "error" in body2:
            err = body2["error"]
            logger.error(f"Instagram publish error: {err}")
            raise HTTPException(status_code=502, detail=err.get("message", "Instagram publish error"))
        media_id = body2["id"]
        logger.info(f"Published — media_id={media_id}")

    # ── Update DB ────────────────────────────────────────────────────────────
    post.status = "posted"
    post.instagram_media_id = media_id
    post.updated_at = datetime.now(timezone.utc)
    await db.commit()

    return {"media_id": media_id, "status": "posted"}
