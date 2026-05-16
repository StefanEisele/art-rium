"""
ORM models — Single Source of Truth for the database schema.
Alembic autogenerates migrations from these definitions.
"""
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    BigInteger,
    DateTime,
    Integer,
    Numeric,
    SmallInteger,
    String,
    Text,
    ForeignKey,
)
from sqlalchemy.dialects.postgresql import ARRAY, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from core.db import Base


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ─────────────────────────────────────────────────────────────────────────────
# Images
# ─────────────────────────────────────────────────────────────────────────────

class Image(Base):
    __tablename__ = "images"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    filename: Mapped[str] = mapped_column(String(512), unique=True, nullable=False)
    filepath: Mapped[str] = mapped_column(Text, nullable=False)  # relative to storage_dir
    prompt: Mapped[str | None] = mapped_column(Text)
    seed: Mapped[int | None] = mapped_column(BigInteger)
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    workflow_name: Mapped[str | None] = mapped_column(String(128))
    batch_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    thumbnail_path: Mapped[str | None] = mapped_column(Text)        # relative to storage_dir, JPEG 512 px
    title: Mapped[str | None] = mapped_column(String(512))          # chosen display title
    tags: Mapped[list[str] | None] = mapped_column(ARRAY(String))
    rating: Mapped[int | None] = mapped_column(SmallInteger)       # 1–5, personal curation
    notes: Mapped[str | None] = mapped_column(Text)
    # WordPress media library (set when uploaded via /api/wordpress/media/upload)
    wp_media_id: Mapped[int | None] = mapped_column(Integer)
    wp_source_url: Mapped[str | None] = mapped_column(Text)
    wp_uploaded_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    wp_seo_title: Mapped[str | None] = mapped_column(String(120))     # VLM-generated fallback when image.title is None, ≤60 chars
    wp_alt_text: Mapped[str | None] = mapped_column(Text)             # VLM-generated, EN
    wp_seo_description: Mapped[str | None] = mapped_column(Text)      # VLM-generated, EN, ≤155 chars
    wp_caption: Mapped[str | None] = mapped_column(Text)              # VLM-generated, EN, ≤300 chars
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, nullable=False
    )

    shop_listings: Mapped[list["ShopListing"]] = relationship(
        back_populates="image", cascade="all, delete-orphan", lazy="selectin"
    )
    instagram_posts: Mapped[list["InstagramPost"]] = relationship(
        back_populates="image", cascade="all, delete-orphan", lazy="selectin"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Articles (WordPress drafts)
# ─────────────────────────────────────────────────────────────────────────────

class Article(Base):
    __tablename__ = "articles"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    body_md: Mapped[str | None] = mapped_column(Text)              # Markdown draft
    excerpt: Mapped[str | None] = mapped_column(Text)              # ≤155 chars, used as Yoast meta description
    tags: Mapped[list[str] | None] = mapped_column(ARRAY(String))  # 3–6 tags, generated with the article
    language: Mapped[str] = mapped_column(String(8), nullable=False, default="en")  # Polylang slug: en | de | zh
    translation_group_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False, default=uuid.uuid4
    )                                                               # shared across the DE/EN/ZH siblings of one piece
    wp_post_id: Mapped[int | None] = mapped_column(Integer)        # null until pushed
    wp_link: Mapped[str | None] = mapped_column(Text)              # canonical URL after WP push
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="draft"
    )                                                               # draft | published | failed
    image_ids: Mapped[list[uuid.UUID] | None] = mapped_column(ARRAY(UUID(as_uuid=True)))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now, nullable=False
    )


# ─────────────────────────────────────────────────────────────────────────────
# Shop listings (Singulart)
# ─────────────────────────────────────────────────────────────────────────────

class ShopListing(Base):
    __tablename__ = "shop_listings"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    image_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("images.id"), nullable=False
    )
    title: Mapped[str | None] = mapped_column(String(512))
    description: Mapped[str | None] = mapped_column(Text)
    price: Mapped[float | None] = mapped_column(Numeric(10, 2))
    singulart_id: Mapped[str | None] = mapped_column(String(128))  # their internal ID if known
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="draft"
    )                                                               # draft | ready | submitted | live
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, nullable=False
    )

    image: Mapped["Image"] = relationship(back_populates="shop_listings")


# ─────────────────────────────────────────────────────────────────────────────
# Instagram scheduled posts
# ─────────────────────────────────────────────────────────────────────────────

class InstagramPost(Base):
    __tablename__ = "instagram_posts"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    image_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("images.id"), nullable=False
    )
    carousel_image_ids: Mapped[list[uuid.UUID] | None] = mapped_column(
        ARRAY(UUID(as_uuid=True))
    )                                                               # null = single post; set = carousel (additional images after image_id)
    caption: Mapped[str | None] = mapped_column(Text)
    scheduled_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="scheduled"
    )                                                               # scheduled | posted | cancelled
    instagram_media_id: Mapped[str | None] = mapped_column(String(128))  # filled after posting
    # Companion posts (auto-created alongside the feed post)
    story_delay_minutes: Mapped[int | None] = mapped_column(Integer)      # null = disabled; 0 = post immediately after feed
    reel_delay_minutes: Mapped[int | None] = mapped_column(Integer)       # null = disabled; 0 = post immediately after feed
    story_scheduled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))  # set when feed published
    reel_scheduled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))   # set when feed published
    story_status: Mapped[str | None] = mapped_column(String(32))          # pending | processing | posted | failed
    story_media_ids: Mapped[list[str] | None] = mapped_column(ARRAY(String(128)))  # one per image
    reel_status: Mapped[str | None] = mapped_column(String(32))           # pending | processing | posted | failed
    reel_media_id: Mapped[str | None] = mapped_column(String(128))
    companion_time: Mapped[str | None] = mapped_column(String(5))              # "HH:MM" for day+ companion posts (default "18:23")
    reel_video_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))  # use an existing generated Video instead of slideshow
    # Instagram-side scheduled containers — once set, the post will publish without us
    feed_creation_id: Mapped[str | None] = mapped_column(String(128))
    reel_creation_id: Mapped[str | None] = mapped_column(String(128))
    reel_video_filename: Mapped[str | None] = mapped_column(String(512))   # slideshow MP4 in storage/reels (kept until reel publishes)
    # Pi posting outpost (cloud-scheduled posts go through here instead of the local scheduler)
    dispatch_target: Mapped[str] = mapped_column(String(16), nullable=False, default="local")  # local | outpost
    outpost_id: Mapped[str | None] = mapped_column(String(64))           # Pi-side post UUID (returned by /enqueue)
    outpost_status: Mapped[str | None] = mapped_column(String(32))       # mirrors Pi: queued|publishing|posted|failed|cancelled
    outpost_reel_status: Mapped[str | None] = mapped_column(String(32))  # mirrors Pi reel_status (when reel uploaded)
    outpost_dispatched_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error: Mapped[str | None] = mapped_column(Text)                              # last failure message
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now, nullable=False
    )

    image: Mapped["Image"] = relationship(back_populates="instagram_posts")


# ─────────────────────────────────────────────────────────────────────────────
# Key-frame videos
# ─────────────────────────────────────────────────────────────────────────────

class Video(Base):
    __tablename__ = "videos"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    filename: Mapped[str | None] = mapped_column(String(512), unique=True)   # null until done
    filepath: Mapped[str | None] = mapped_column(Text)                       # relative to storage_dir
    image_ids: Mapped[list[uuid.UUID] | None] = mapped_column(ARRAY(UUID(as_uuid=True)))  # source key frames
    prompt: Mapped[str | None] = mapped_column(Text)
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    frame_count: Mapped[int | None] = mapped_column(Integer)  # frames per transition (length param)
    n_images: Mapped[int | None] = mapped_column(Integer)     # 2–6 key frames
    fps: Mapped[int | None] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="generating")
    error: Mapped[str | None] = mapped_column(Text)
    comfy_prompt_id: Mapped[str | None] = mapped_column(String(128))
    workflow: Mapped[str | None] = mapped_column(String(32))          # "i2v_multi" | "flf2v"
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, nullable=False
    )


# ─────────────────────────────────────────────────────────────────────────────
# Piano improvisation sessions
# ─────────────────────────────────────────────────────────────────────────────

class ImprovSession(Base):
    __tablename__ = "improv_sessions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    source_video_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("videos.id", ondelete="RESTRICT"),
        nullable=False,
    )
    recording_filename: Mapped[str] = mapped_column(String(512), nullable=False)
    mix_synth_video_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("videos.id", ondelete="SET NULL"),
        nullable=True,
    )
    mix_hands_video_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("videos.id", ondelete="SET NULL"),
        nullable=True,
    )
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="queued")
    # queued | processing | done | failed
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, nullable=False
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
