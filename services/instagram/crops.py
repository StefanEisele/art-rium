"""
Baking the pre-crops a feed post's images are published with.

`services/instagram/framing.py` decides *what* the crop is; this module makes
it exist as a file. It has to be a real file, and that is the whole design
constraint: the two dispatch paths consume media very differently — the local
Graph path hands Meta a `/share/image/<name>` URL to fetch, the Pi outpost
re-encodes the file and uploads the bytes — but both go through
`MediaRef.filename` / `.filepath`. Bake the crop into a sibling PNG, point the
MediaRef at it, and neither path needs to know that cropping exists.

Idempotent by design, because it runs twice: once when the post is saved (so
the crop exists while it sits in the timeline) and once at dispatch (so an
image enhanced *after* scheduling is still published cropped rather than as
the pixels it had at save time). A crop that is present and newer than its
source is left alone.
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import settings
from core.models import Image, InstagramPost, InstagramPostMedia, Video
from services.image.rendition import primary_filename, primary_filepath, resolve_image_path
from services.instagram.framing import (
    crop_rendition_name,
    frame_ratio,
    needs_crop,
    render_crop_sync,
)

logger = logging.getLogger(__name__)


def _child_ratio(
    m: InstagramPostMedia,
    images: dict[uuid.UUID, Image],
    videos: dict[uuid.UUID, Video],
) -> float | None:
    """Aspect ratio of one carousel child, or None when it isn't recorded."""
    row = images.get(m.image_id) if m.kind == "image" else videos.get(m.video_id)
    if not row or not row.width or not row.height:
        return None
    return row.width / row.height


async def _load_children(
    items: list[InstagramPostMedia], db: AsyncSession,
) -> tuple[dict[uuid.UUID, Image], dict[uuid.UUID, Video]]:
    image_ids = [m.image_id for m in items if m.kind == "image" and m.image_id]
    video_ids = [m.video_id for m in items if m.kind == "video" and m.video_id]
    images: dict[uuid.UUID, Image] = {}
    if image_ids:
        r = await db.execute(select(Image).where(Image.id.in_(image_ids)))
        images = {i.id: i for i in r.scalars().all()}
    videos: dict[uuid.UUID, Video] = {}
    if video_ids:
        r = await db.execute(select(Video).where(Video.id.in_(video_ids)))
        videos = {v.id: v for v in r.scalars().all()}
    return images, videos


async def post_frame_ratio(post: InstagramPost, db: AsyncSession) -> float:
    """The aspect ratio Instagram will render this post in."""
    items = sorted(post.media, key=lambda m: m.position)
    if not items:
        return frame_ratio(post.frame_ratio, None)
    images, videos = await _load_children(items, db)
    return frame_ratio(post.frame_ratio, _child_ratio(items[0], images, videos))


async def ensure_post_crops(post: InstagramPost, db: AsyncSession) -> float:
    """Make every 'fill' image child's crop rendition current, and clear the
    crop of every child that shouldn't have one. Returns the frame ratio used.

    Mutates the media rows; the caller commits. Raises RuntimeError if a crop
    cannot be rendered — publishing an uncropped image where the user asked for
    a crop would silently undo their decision, which is worse than failing the
    save.
    """
    items = sorted(post.media, key=lambda m: m.position)
    if not items:
        return frame_ratio(post.frame_ratio, None)

    images, videos = await _load_children(items, db)
    target = frame_ratio(post.frame_ratio, _child_ratio(items[0], images, videos))

    for m in items:
        img = images.get(m.image_id) if m.kind == "image" else None
        # Videos can't be cropped here (that means re-encoding), and an image
        # that is already the frame's shape needs no second file.
        if (m.kind != "image" or m.crop_mode != "fill" or not img
                or not needs_crop(img.width or 0, img.height or 0, target)):
            m.crop_filename = None
            m.crop_filepath = None
            continue

        src = resolve_image_path(img)
        if not src.exists():
            raise RuntimeError(f"Cannot crop {primary_filename(img)}: source file missing")

        name = crop_rendition_name(primary_filename(img), target, m.crop_offset)
        rel = str(Path(primary_filepath(img)).with_name(name)).replace("\\", "/")
        dest = settings.storage_dir / rel

        # Deterministic name, so an existing file is already this exact crop —
        # unless the source has been re-rendered since (the wand, the grain
        # pass), which the mtime catches.
        if not dest.exists() or dest.stat().st_mtime < src.stat().st_mtime:
            try:
                w, h = await asyncio.to_thread(render_crop_sync, src, dest, target, m.crop_offset)
            except Exception as exc:
                raise RuntimeError(f"Cannot crop {primary_filename(img)}: {exc}") from exc
            logger.info("IG crop %s → %dx%d (ratio %.3f, offset %.2f)",
                        name, w, h, target, m.crop_offset)

        m.crop_filename = name
        m.crop_filepath = rel

    return target
