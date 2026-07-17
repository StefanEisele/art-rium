"""
Helpers for reading and writing the ordered media list of an Instagram feed
post (the instagram_post_media table).

A feed post is a carousel of up to 10 children; each child is either an
image or a video. This module provides:

- `MediaRef`           — a flat, session-decoupled record of one child.
- `load_media_refs()`  — fetch the ordered list for one post, joining Image
                         and Video so callers don't need a second query.
- `share_kind_for()`   — pick the correct `/share/<kind>/` route for the
                         Graph-API `image_url` / `video_url` field.

Reel-mode posts (`kind='reel'`) don't use this — their source clips live in
`InstagramPost.reel_video_ids` and are concatenated at dispatch time.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import settings
from core.models import Image, InstagramPost, InstagramPostMedia, Video

MediaKind = Literal["image", "video"]


def resolve_video_path(video: Video) -> Path:
    """Absolute path to the file Instagram should actually ingest for `video`.

    A soundtrack attached via `/jobs/{id}/soundtrack` is muxed into a sibling
    file (`muxed_filename`) rather than overwriting the silent original
    (`filepath`) — see routers/video.py::_serialize's `primary_name` for the
    same precedence used by the video player. Every reel/companion dispatch
    path must resolve through here, or a video with an attached soundtrack
    silently gets published without audio.
    """
    if video.muxed_filename:
        return settings.videos_dir / video.muxed_filename
    return settings.storage_dir / video.filepath


@dataclass(slots=True)
class MediaRef:
    """One ordered child of a feed-post carousel.

    Carries everything a publisher needs to build a Graph-API child
    container, decoupled from any DB session so it can outlive the
    transaction it was loaded in.
    """
    kind:        MediaKind          # 'image' | 'video'
    media_id:    uuid.UUID          # Image.id or Video.id
    filename:    str                # public share filename
    filepath:    str                # storage-relative path (for re-encode / multipart)

    @property
    def share_kind(self) -> str:
        """The `/share/<kind>/` route to use for this media's public URL."""
        return "image" if self.kind == "image" else "video"


async def load_media_refs(post: InstagramPost, db: AsyncSession) -> list[MediaRef]:
    """Return the post's media children in position order, joined with their
    backing Image / Video rows. Videos that aren't yet `done` are skipped
    (the caller decides whether that's an error)."""
    items = sorted(post.media, key=lambda m: m.position)
    image_ids = [m.image_id for m in items if m.kind == "image" and m.image_id]
    video_ids = [m.video_id for m in items if m.kind == "video" and m.video_id]

    images: dict[uuid.UUID, Image] = {}
    if image_ids:
        rs = await db.execute(select(Image).where(Image.id.in_(image_ids)))
        images = {i.id: i for i in rs.scalars().all()}

    videos: dict[uuid.UUID, Video] = {}
    if video_ids:
        rs = await db.execute(select(Video).where(Video.id.in_(video_ids)))
        videos = {v.id: v for v in rs.scalars().all()}

    refs: list[MediaRef] = []
    for m in items:
        if m.kind == "image":
            img = images.get(m.image_id)
            if img:
                refs.append(MediaRef("image", img.id, img.filename, img.filepath))
        elif m.kind == "video":
            vid = videos.get(m.video_id)
            if vid and vid.status == "done" and vid.filename and vid.filepath:
                refs.append(MediaRef("video", vid.id, vid.filename, vid.filepath))
    return refs


async def replace_media_items(
    post: InstagramPost,
    items: list[tuple[MediaKind, uuid.UUID]],
    db: AsyncSession,
) -> None:
    """Replace a post's media children with the given ordered list.

    Flushes the orphan-deletes before appending the new children, so the
    UNIQUE(post_id, position) constraint does not see the same position
    occupied by both an old (to-delete) and new (to-insert) row.
    """
    if post.media:
        post.media.clear()
        await db.flush()
    for position, (kind, media_id) in enumerate(items):
        post.media.append(InstagramPostMedia(
            position=position,
            kind=kind,
            image_id=media_id if kind == "image" else None,
            video_id=media_id if kind == "video" else None,
        ))
