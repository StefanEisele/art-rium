"""
ORM models — Single Source of Truth for the database schema.
Alembic autogenerates migrations from these definitions.
"""
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
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
    batch_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), index=True)
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


# ─────────────────────────────────────────────────────────────────────────────
# Articles (WordPress drafts)
# ─────────────────────────────────────────────────────────────────────────────

class Article(Base):
    __tablename__ = "articles"
    __table_args__ = (
        CheckConstraint("status IN ('draft', 'published', 'failed')", name="ck_articles_status"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    body_md: Mapped[str | None] = mapped_column(Text)              # Markdown draft
    excerpt: Mapped[str | None] = mapped_column(Text)              # ≤155 chars, used as Yoast meta description
    tags: Mapped[list[str] | None] = mapped_column(ARRAY(String))  # 3–6 tags, generated with the article
    language: Mapped[str] = mapped_column(String(8), nullable=False, default="en")  # Polylang slug: en | de (zh rows exist historically; no longer produced)
    translation_group_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False, default=uuid.uuid4, index=True
    )                                                               # shared across the EN+DE siblings of one piece
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
    __table_args__ = (
        CheckConstraint(
            "status IN ('draft', 'ready', 'submitted', 'live')", name="ck_shop_listings_status"
        ),
    )

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
    __table_args__ = (
        # The scheduler polls WHERE status='scheduled' AND scheduled_at <= now()
        # every 60s (workers/instagram_scheduler.py) — this is the hot path.
        Index("ix_instagram_posts_status_scheduled_at", "status", "scheduled_at"),
        CheckConstraint(
            "status IN ('scheduled', 'posted', 'cancelled', 'failed')",
            name="ck_instagram_posts_status",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    kind: Mapped[str] = mapped_column(
        String(16), nullable=False, default="feed", server_default="feed"
    )                                                               # feed (default, mixed image+video carousel) | reel (standalone, 1–4 videos concatenated)
    # Feed media items live in instagram_post_media (mixed image+video carousel,
    # ordered by position 0..9). See InstagramPostMedia below.
    reel_video_ids: Mapped[list[uuid.UUID] | None] = mapped_column(
        ARRAY(UUID(as_uuid=True))
    )                                                               # ordered list of 1–4 source videos for kind='reel' concat
    caption: Mapped[str | None] = mapped_column(Text)
    scheduled_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="scheduled", index=True
    )                                                               # scheduled | posted | cancelled | failed
    instagram_media_id: Mapped[str | None] = mapped_column(String(128))  # filled after posting
    # Story/reel companion state lives in PostCompanion (one row per kind) —
    # see below. companion_time is the one knob shared by both companions'
    # timing math, so it stays here rather than being duplicated per-row.
    companion_time: Mapped[str | None] = mapped_column(String(5))              # "HH:MM" for day+ companion posts (default "18:23")
    # Instagram-side scheduled containers — once set, the post will publish without us
    feed_creation_id: Mapped[str | None] = mapped_column(String(128))
    # Pi posting outpost (cloud-scheduled posts go through here instead of the local scheduler)
    dispatch_target: Mapped[str] = mapped_column(String(16), nullable=False, default="local")  # local | outpost
    outpost_id: Mapped[str | None] = mapped_column(String(64))           # Pi-side post UUID (returned by /enqueue)
    outpost_status: Mapped[str | None] = mapped_column(String(32))       # mirrors Pi: queued|publishing|posted|failed|cancelled
    outpost_dispatched_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error: Mapped[str | None] = mapped_column(Text)                              # last failure message
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now, nullable=False
    )

    media: Mapped[list["InstagramPostMedia"]] = relationship(
        back_populates="post",
        cascade="all, delete-orphan",
        order_by="InstagramPostMedia.position",
        lazy="selectin",
    )
    companions: Mapped[list["PostCompanion"]] = relationship(
        back_populates="post",
        cascade="all, delete-orphan",
        lazy="selectin",
    )


class PostCompanion(Base):
    """One companion (story or reel) attached to a feed InstagramPost.

    Replaces the old story_*/reel_*/outpost_reel_status flat columns that
    used to live on InstagramPost — one row per companion kind, so a future
    companion type is a new `kind` value, not new parent-table columns."""
    __tablename__ = "post_companions"
    __table_args__ = (
        UniqueConstraint("post_id", "kind", name="uq_post_companions_post_kind"),
        Index("ix_post_companions_post_id", "post_id"),
        CheckConstraint("kind IN ('story', 'reel')", name="ck_post_companions_kind"),
        # NOT constrained the same way the vocabulary comment on the old
        # reel_status column warned about: `status` here can also arrive
        # verbatim from the Pi outpost's /status response for some paths, so
        # this CHECK covers the LOCAL vocabulary only — same trade-off as
        # instagram_posts.story_status/outpost_status before this migration.
        CheckConstraint(
            "status IS NULL OR status IN "
            "('pending', 'processing', 'posted', 'failed', 'remote_scheduled')",
            name="ck_post_companions_status",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    post_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("instagram_posts.id", ondelete="CASCADE"), nullable=False,
    )
    kind: Mapped[str] = mapped_column(String(16), nullable=False)  # 'story' | 'reel'

    delay_minutes: Mapped[int | None] = mapped_column(Integer)      # null = disabled; 0 = immediately after feed
    scheduled_at:  Mapped[datetime | None] = mapped_column(DateTime(timezone=True))  # planned publish time, or backfilled once observed posted
    status:        Mapped[str | None] = mapped_column(String(32))   # pending | processing | posted | failed | remote_scheduled
    outpost_status: Mapped[str | None] = mapped_column(String(32))  # mirrors Pi reel_status verbatim; reel only today (was outpost_reel_status)

    # Reel-only fields (NULL for kind='story')
    creation_id:    Mapped[str | None] = mapped_column(String(128))          # Instagram-side scheduled container id
    media_id:       Mapped[str | None] = mapped_column(String(128))          # published reel media id
    video_id:       Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))  # use an existing generated Video instead of slideshow
    video_filename: Mapped[str | None] = mapped_column(String(512))          # slideshow MP4 in storage/reels (kept until reel publishes)

    # Story-only field (NULL for kind='reel')
    media_ids: Mapped[list[str] | None] = mapped_column(ARRAY(String(128)))  # one per image (story only)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now, nullable=False
    )

    post: Mapped["InstagramPost"] = relationship(back_populates="companions")


class InstagramPostMedia(Base):
    """One ordered child of a feed post's carousel. kind='image' references
    an Image row; kind='video' references a Video row. Up to 10 per post."""
    __tablename__ = "instagram_post_media"
    __table_args__ = (
        UniqueConstraint("post_id", "position", name="uq_ig_post_media_position"),
        Index("ix_ig_post_media_post_id", "post_id"),
        CheckConstraint(
            "(image_id IS NOT NULL) != (video_id IS NOT NULL)",
            name="ck_ig_post_media_exactly_one_ref",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    post_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("instagram_posts.id", ondelete="CASCADE"),
        nullable=False,
    )
    position: Mapped[int] = mapped_column(Integer, nullable=False)        # 0..9
    kind: Mapped[str] = mapped_column(String(8), nullable=False)          # 'image' | 'video'
    image_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("images.id", ondelete="CASCADE"), nullable=True
    )
    video_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("videos.id", ondelete="CASCADE"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, nullable=False
    )

    post: Mapped["InstagramPost"] = relationship(back_populates="media")


# ─────────────────────────────────────────────────────────────────────────────
# Key-frame videos
# ─────────────────────────────────────────────────────────────────────────────

class Video(Base):
    __tablename__ = "videos"
    __table_args__ = (
        CheckConstraint(
            "status IN ('generating', 'review', 'assembling', 'done', 'failed')",
            name="ck_videos_status",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    filename: Mapped[str | None] = mapped_column(String(512), unique=True)   # null until done
    filepath: Mapped[str | None] = mapped_column(Text)                       # relative to storage_dir
    image_ids: Mapped[list[uuid.UUID] | None] = mapped_column(ARRAY(UUID(as_uuid=True)))  # source key frames
    prompt: Mapped[str | None] = mapped_column(Text)
    title: Mapped[str | None] = mapped_column(String(255))   # user-editable display title
    notes: Mapped[str | None] = mapped_column(Text)          # user-editable free-form notes
    # Optional muxed soundtrack (Song attached via /tools/video detail modal)
    soundtrack_song_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("songs.id", ondelete="SET NULL"), nullable=True
    )
    muxed_filename: Mapped[str | None] = mapped_column(String(512))   # in storage/videos/, sibling of `filename`
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    frame_count: Mapped[int | None] = mapped_column(Integer)  # representative/fallback frame count (flf2v: per-transition; i2v_multi/ltx_i2v: per-image)
    n_images: Mapped[int | None] = mapped_column(Integer)     # total selected images (bounds workflow-dependent: i2v_multi 1–10, ltx_i2v 1–6, flf2v 2–20)
    fps: Mapped[int | None] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="generating", index=True)
    error: Mapped[str | None] = mapped_column(Text)
    comfy_prompt_id: Mapped[str | None] = mapped_column(String(128))
    workflow: Mapped[str | None] = mapped_column(String(32))          # "i2v_multi" | "ltx_i2v" | "flf2v"
    # YouTube upload (set when pushed via services/youtube/client.py)
    youtube_video_id: Mapped[str | None] = mapped_column(String(32))    # e.g. "dQw4w9WgXcQ"
    youtube_url: Mapped[str | None] = mapped_column(Text)               # canonical watch URL
    youtube_privacy: Mapped[str | None] = mapped_column(String(16))     # "public" | "unlisted" | "private"
    youtube_uploaded_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, nullable=False
    )


# ─────────────────────────────────────────────────────────────────────────────
# Piano improvisation sessions
# ─────────────────────────────────────────────────────────────────────────────

class ImprovSession(Base):
    __tablename__ = "improv_sessions"
    __table_args__ = (
        CheckConstraint(
            "status IN ('queued', 'processing', 'done', 'failed')",
            name="ck_improv_sessions_status",
        ),
    )

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
    mix_pip_video_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("videos.id", ondelete="SET NULL"),
        nullable=True,
    )
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="queued")
    # queued | processing | done | failed
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, nullable=False, index=True
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


# ─────────────────────────────────────────────────────────────────────────────
# Songs (ACE-Step 1.5 Turbo audio generation)
# ─────────────────────────────────────────────────────────────────────────────

class Song(Base):
    __tablename__ = "songs"
    __table_args__ = (
        CheckConstraint("status IN ('generating', 'done', 'failed')", name="ck_songs_status"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    filename: Mapped[str | None] = mapped_column(String(512), unique=True)   # null until done
    filepath: Mapped[str | None] = mapped_column(Text)                       # relative to storage_dir
    tags: Mapped[str] = mapped_column(Text, nullable=False)                  # ACE "tags" prompt (style / genre / mood)
    lyrics: Mapped[str | None] = mapped_column(Text)
    duration_seconds: Mapped[int] = mapped_column(Integer, nullable=False)   # 5–240
    bpm: Mapped[int | None] = mapped_column(Integer)
    musical_key: Mapped[str | None] = mapped_column(String(32))              # e.g. "E minor"
    language: Mapped[str | None] = mapped_column(String(8))                  # "en" | "de" | "zh" | ...
    seed: Mapped[int | None] = mapped_column(BigInteger)
    steps: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=8, server_default="8")
    cfg: Mapped[float | None] = mapped_column(Numeric(4, 2))
    shift: Mapped[float | None] = mapped_column(Numeric(4, 2))               # ModelSamplingAuraFlow shift
    title: Mapped[str | None] = mapped_column(String(255))                   # user-editable display title
    notes: Mapped[str | None] = mapped_column(Text)                          # user-editable free-form notes
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="generating")
    error: Mapped[str | None] = mapped_column(Text)
    comfy_prompt_id: Mapped[str | None] = mapped_column(String(128))
    workflow: Mapped[str | None] = mapped_column(String(32))                 # "ace_step_1.5_turbo"
    waveform_path: Mapped[str | None] = mapped_column(Text)                  # optional PNG thumbnail
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, nullable=False
    )
