"""
ORM models — Single Source of Truth for the database schema.
Alembic autogenerates migrations from these definitions.
"""
import uuid
from datetime import datetime, timezone
from decimal import Decimal

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
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
    # [{"name": "<filename>.safetensors", "strength": 0.500}, ...] — one entry
    # per LoRA active at generation time, applied as a chain (order matters
    # for Z-Image; SDXL/Ernie only ever populate 0 or 1 entry today).
    loras: Mapped[list[dict] | None] = mapped_column(JSONB)
    workflow_name: Mapped[str | None] = mapped_column(String(128))
    batch_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), index=True)
    thumbnail_path: Mapped[str | None] = mapped_column(Text)        # relative to storage_dir, JPEG 512 px
    title: Mapped[str | None] = mapped_column(String(512))          # chosen display title
    tags: Mapped[list[str] | None] = mapped_column(ARRAY(String))
    rating: Mapped[int | None] = mapped_column(SmallInteger)       # 1–5, personal curation
    notes: Mapped[str | None] = mapped_column(Text)
    # Auto-enhance ("Zauberstab", services/image/enhance.py). Like the video
    # grain pass, this writes a *sibling* file and never overwrites the
    # original, and is always re-rendered from the rendition below it — the
    # upscale when there is one, the original otherwise — so re-running at a
    # different strength replaces the correction instead of stacking it.
    # Precedence for publishing lives in services/image/rendition.py.
    enhance_strength: Mapped[int | None] = mapped_column(SmallInteger)  # 0–150 UI scale; null = not enhanced
    enhanced_filename: Mapped[str | None] = mapped_column(String(512))  # basename, sibling of `filename`
    enhanced_filepath: Mapped[str | None] = mapped_column(Text)         # relative to storage_dir
    enhance_params: Mapped[dict | None] = mapped_column(JSONB)          # the Adjustments the analysis chose, for the UI
    # Diffusion upscale (services/image/upscale.py) — Ultimate SD Upscale
    # driven by Z-Image Turbo. The *lowest* derived rendition: it always reads
    # the untouched original, and the two Pillow passes (wand, grain) render on
    # top of it at the new resolution. Deliberately so — re-running this costs
    # GPU minutes, so nothing cheap is allowed to invalidate it, and only
    # deleting the upscale takes the image back to its generated size.
    upscale_scale: Mapped[float | None] = mapped_column(Float)          # 1.5–4.0; null = not upscaled
    upscale_denoise: Mapped[float | None] = mapped_column(Float)        # the creativity the tiles were redrawn at
    upscale_model: Mapped[str | None] = mapped_column(String(32))       # key into services/image/upscale.py::UPSCALE_MODELS
    upscaled_filename: Mapped[str | None] = mapped_column(String(512))  # basename, sibling of `filename`
    upscaled_filepath: Mapped[str | None] = mapped_column(Text)         # relative to storage_dir
    upscale_width: Mapped[int | None] = mapped_column(Integer)
    upscale_height: Mapped[int | None] = mapped_column(Integer)
    # Film grain (services/image/grain.py) — the last derived rendition: it
    # renders on top of the enhancement / upscale when there is one (grain
    # belongs at the delivery resolution), as the video pass does too.
    # Removing it puts whatever was underneath back untouched.
    grain_strength: Mapped[int | None] = mapped_column(SmallInteger)    # 1–100 UI scale; null = no grain
    grained_filename: Mapped[str | None] = mapped_column(String(512))   # basename, sibling of `filename`
    grained_filepath: Mapped[str | None] = mapped_column(Text)          # relative to storage_dir
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
    collaborators: Mapped[list[str] | None] = mapped_column(
        ARRAY(String(30))
    )                                                               # up to 3 IG usernames invited as co-authors; not supported on stories
    scheduled_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="scheduled", index=True
    )                                                               # scheduled | posted | cancelled | failed
    instagram_media_id: Mapped[str | None] = mapped_column(String(128))  # filled after posting
    # Which frame Instagram will render this post in — "auto" (the first
    # child's ratio, clamped into the feed's 4:5…1.91:1 range, which is what
    # Instagram does anyway) or a pinned choice. See services/instagram/framing.py.
    frame_ratio: Mapped[str] = mapped_column(String(8), nullable=False, default="auto")
    # Meta's `is_ai_generated` self-disclosure, set per post at container
    # creation. New posts default to settings.instagram_ai_label_default.
    ai_label: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
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
    # How this child meets the post's frame. 'fit' publishes the picture as it
    # is and lets Instagram pad it (bars); 'fill' pre-crops it to the frame so
    # there is nothing left to pad. `crop_offset` (0..1) slides the crop window
    # along whichever axis is being cut. Images only — cropping a video would
    # mean re-encoding it, so video children stay 'fit'.
    crop_mode: Mapped[str] = mapped_column(String(8), nullable=False, default="fit")
    crop_offset: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)
    # The baked crop that actually gets published, a sibling of the source
    # image under images_dir. Null whenever crop_mode is 'fit'.
    crop_filename: Mapped[str | None] = mapped_column(String(512))
    crop_filepath: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, nullable=False
    )

    post: Mapped["InstagramPost"] = relationship(back_populates="media")


# ─────────────────────────────────────────────────────────────────────────────
# Key-frame videos
# ─────────────────────────────────────────────────────────────────────────────

# Workflows whose clips carry a generated audio track (the model samples sound
# alongside the picture). Everything else — the Wan-based i2v_multi/flf2v — is
# silent. "ltx_i2v" is retired but kept here so clips generated before the
# MiniMax H3 switch keep reporting their audio to the merge/mux paths.
AUDIO_WORKFLOWS = frozenset({"minimax_i2v", "minimax_flf", "ltx_i2v"})

# Key-frame animation workflows, as opposed to improv mixes or merges. Used to
# label a video for YouTube and for the article LLM.
ANIMATE_WORKFLOWS = frozenset(
    {"i2v_multi", "minimax_i2v", "minimax_flf", "ltx_i2v", "flf2v", "minimax_api"}
)

# Videos rendered by a paid cloud provider rather than the local GPU.
API_WORKFLOW = "minimax_api"

# Ledger states. 'released' never counts against the limit; nothing is ever
# deleted, because the ledger *is* the audit trail.
LEDGER_STATES = ("reserved", "settled", "released")
COMMITTED_STATES = ("reserved", "settled")
LEDGER_KINDS = ("generate_768p", "generate_2k", "regenerate_2k")


class Video(Base):
    __tablename__ = "videos"
    __table_args__ = (
        CheckConstraint(
            "status IN ('queued', 'generating', 'review', 'assembling', 'done', 'failed')",
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
    # Ambient bed: how loudly the video's OWN generated audio plays under the
    # attached song. Null = the song replaces the clip's sound entirely, which
    # is the historical behaviour and stays the default. Persisted because the
    # mux is re-run whenever its source changes (upscale, grain) and would
    # otherwise silently drop the bed on the next re-render.
    soundtrack_bed_volume: Mapped[float | None] = mapped_column(Float)
    # Optional muxed soundtrack (Song attached via /tools/video detail modal)
    soundtrack_song_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("songs.id", ondelete="SET NULL"), nullable=True
    )
    muxed_filename: Mapped[str | None] = mapped_column(String(512))   # in storage/videos/, sibling of `filename`
    # Where in the song the picture starts. Null/0 for every ordinary
    # soundtrack; only a beat cut that was told to begin at a later bar carries
    # one. Persisted rather than passed, because the mux is re-run from scratch
    # after an upscale or a grain pass and would otherwise reset the offset and
    # slide the whole edit off its music.
    soundtrack_start_seconds: Mapped[float | None] = mapped_column(Float)
    # Optional SEEDVR2 upscale / retime pass (/tools/video detail modal). Runs
    # before the grain pass — grain belongs at the delivery resolution, and
    # feeding a grained picture to a restorer would have it reconstruct the
    # noise. All three dials below are independent: 0 resolution runs the
    # timing stages alone, and no interpolation with a target rate is a plain
    # conform.
    upscale_resolution: Mapped[int | None] = mapped_column(SmallInteger)  # target SHORT edge in px; 0 = size kept; null = no pass
    upscale_filename: Mapped[str | None] = mapped_column(String(512))     # in storage/videos/, sibling of `filename`
    # RIFE interpolation applied *after* the restoration (1 = off). Ordered
    # there so the upscale's cost stays independent of the factor — SEEDVR2
    # only ever restores the source's real frames.
    upscale_rife: Mapped[int | None] = mapped_column(SmallInteger)
    # Playback rate the pass was asked to write (null = keep the source's,
    # which with interpolation means slow motion). Persisted because a
    # re-render after a soundtrack change replays these settings, and without
    # it a same-length interpolation would come back stretched.
    upscale_fps: Mapped[int | None] = mapped_column(SmallInteger)
    # Optional look pass (/tools/video detail modal) — correction, grade,
    # optics and grain in one ffmpeg chain. Always re-rendered from the
    # rendition *below* it (upscale_filename or muxed_filename or filename) so
    # changing a dial replaces the look instead of stacking a second one on it.
    #
    # `grain_strength` predates the other six dials and is kept as the grain
    # one, mirrored out of `look_params` on every render: services/improv and
    # the gallery both read it, and a look is "grained" exactly when that dial
    # is up. `grain_filename` is likewise still the output slot — the file the
    # pass writes, whatever the pass has grown into.
    grain_strength: Mapped[int | None] = mapped_column(SmallInteger)   # 0–100 UI scale; null = no look
    grain_filename: Mapped[str | None] = mapped_column(String(512))    # in storage/videos/, sibling of `filename`
    look_params: Mapped[dict | None] = mapped_column(JSONB)            # services/video/look.py::Look.to_dict()
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    frame_count: Mapped[int | None] = mapped_column(Integer)  # representative/fallback frame count (flf2v: per-transition; i2v_multi/minimax_i2v: per-image)
    n_images: Mapped[int | None] = mapped_column(Integer)     # total selected images (bounds workflow-dependent: i2v_multi 1–10, minimax_i2v 1–6, flf2v 2–20)
    fps: Mapped[int | None] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="generating", index=True)
    error: Mapped[str | None] = mapped_column(Text)
    comfy_prompt_id: Mapped[str | None] = mapped_column(String(128))
    workflow: Mapped[str | None] = mapped_column(String(32))          # "i2v_multi" | "minimax_i2v" | "flf2v" | "minimax_flf" | "merge" | "beatcut" | "minimax_api" (legacy rows may hold "ltx_i2v")
    # The edit that produced a workflow="beatcut" row: style, seed, tempo and
    # every shot (services/video/cut.py::EditPlan). Kept so the card can say
    # what the piece is, and so the same cut can be rebuilt or re-rolled from
    # the numbers it was made with instead of from a remembered UI state.
    cut_plan: Mapped[dict | None] = mapped_column(JSONB)
    # ── Cloud generation (workflow == API_WORKFLOW) ──────────────────────────
    # These rows are rendered by MiniMax, not by the local GPU, so they behave
    # differently in one important way: the job keeps running when this process
    # dies. core/startup_sweep.py therefore leaves them alone and
    # services/video_api/queue.py reconciles them against the provider instead,
    # the same division of labour the Instagram outpost already uses.
    api_task_id: Mapped[str | None] = mapped_column(String(128), index=True)  # provider task id, null until submitted
    api_resolution: Mapped[str | None] = mapped_column(String(8))             # "768P" | "2K"
    api_ratio: Mapped[str | None] = mapped_column(String(16))                 # "adaptive" | "16:9" | …
    duration_s: Mapped[int | None] = mapped_column(SmallInteger)              # requested seconds (4–15)
    # A 2K regeneration points at the 768P take it was pulled up from, so the
    # pair is shown as one item rather than as two unrelated videos.
    source_video_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("videos.id", ondelete="SET NULL"), nullable=True, index=True
    )
    # YouTube upload (set when pushed via services/youtube/client.py)
    youtube_video_id: Mapped[str | None] = mapped_column(String(32))    # e.g. "dQw4w9WgXcQ"
    youtube_url: Mapped[str | None] = mapped_column(Text)               # canonical watch URL
    youtube_privacy: Mapped[str | None] = mapped_column(String(16))     # "public" | "unlisted" | "private"
    youtube_uploaded_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, nullable=False
    )


class ControlTrack(Base):
    """An uploaded control track for the VACE structure-video workflow.

    A first-class asset rather than a job input, because that is how it is
    used: the same Blender turntable gets rendered at a dozen strengths while
    the look is being found, and re-uploading a 40 MB depth pass each time
    would be the slowest part of the loop.

    `kind` says how the track is read, and getting it wrong is a silent
    failure rather than an error:
      - "depth"   a rendered depth pass. Blender writes these with the
                  background white, so they are inverted before use.
      - "footage" ordinary video; DepthAnythingV2 derives the depth in-graph,
                  and the result must NOT also be inverted.
      - "mask"    flat object-ID colours, keyed into regions by ColorToMask.

    Files live under storage/control/.
    """
    __tablename__ = "control_tracks"
    __table_args__ = (
        CheckConstraint("kind IN ('depth', 'footage', 'mask')", name="ck_control_tracks_kind"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    filepath: Mapped[str] = mapped_column(Text, nullable=False)      # relative to storage_dir
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    title: Mapped[str | None] = mapped_column(String(255))
    thumbnail_path: Mapped[str | None] = mapped_column(Text)         # relative to storage_dir
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    # Probed on upload, not trusted from the client: `length` is clamped to it
    # so VACE is never asked for frames the track cannot guide, which it would
    # otherwise pad with flat grey.
    frame_count: Mapped[int | None] = mapped_column(Integer)
    fps: Mapped[float | None] = mapped_column(Float)
    # Where a derived track came from: a mask segmented out of some footage, or
    # a trimmed copy of a longer take. Kept because the two must stay frame
    # aligned — a mask keyed against frame 40 of its source is meaningless
    # beside a differently-trimmed version of that source — so the UI offers a
    # mask together with the exact track it was cut from and nothing else.
    source_track_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("control_tracks.id", ondelete="SET NULL"),
        index=True,
    )
    # For kind="mask": what each colour means, as
    # [{"color": [255,0,0], "label": "Tomate", "coverage": 0.31, "frames": 81}].
    # Without it a mask video is three anonymous silhouettes and the user has to
    # remember which primary they asked for which thing.
    regions: Mapped[list[dict] | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, nullable=False
    )


class VideoClip(Base):
    """One generated segment clip of a Video job — a first-class library item.

    Every workflow (i2v_multi / minimax_i2v / flf2v) persists each segment it
    renders as a clip row; the job's clips form its "stack" in the UI. Clips
    from any number of jobs can then be merged (in any order) into a new
    Video row with workflow="merge". Files live under
    storage/videos/segments/{video_id}/.
    """
    __tablename__ = "video_clips"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    video_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("videos.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    idx: Mapped[int] = mapped_column(Integer, nullable=False)          # position within the job
    filename: Mapped[str] = mapped_column(String(512), nullable=False) # e.g. "seg_0.mp4" (in the job's segments dir)
    thumb: Mapped[str] = mapped_column(String(512), nullable=False)    # e.g. "seg_0_thumb.jpg"
    prompt: Mapped[str | None] = mapped_column(Text)
    frame_count: Mapped[int | None] = mapped_column(Integer)
    workflow: Mapped[str] = mapped_column(String(32), nullable=False)  # source job workflow
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    fps: Mapped[int | None] = mapped_column(Integer)
    has_audio: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)  # true for AUDIO_WORKFLOWS clips
    # What the Wan sampler was told. Both are null for a MiniMax clip (that
    # model runs its own fixed recipe) and for anything rendered before these
    # became settings. Kept for the same reason the prompt is: they are dials
    # the user is expected to hunt for a value on, and a clip that cannot say
    # which value produced it cannot be compared with the next one.
    wan_steps: Mapped[int | None] = mapped_column(SmallInteger)      # total sampler steps, both experts
    wan_lora_high: Mapped[float | None] = mapped_column(Float)       # distill on the high-noise expert; LOWER = more motion
    # Optional SEEDVR2 upscale of this individual clip, rendered *before* the
    # merge. Upscaling the merged video instead makes the restorer (and RIFE)
    # work across the hard cuts between segments, which it interpolates into
    # visible morphs — a clip has no cuts inside it, so this is the pass that
    # is safe to run. Sibling file in the same segments dir; the merge reads
    # it, and grain still belongs afterwards on the merged result.
    upscale_resolution: Mapped[int | None] = mapped_column(SmallInteger)  # target SHORT edge in px; 0 = size kept; null = no pass
    upscale_rife: Mapped[int | None] = mapped_column(SmallInteger)        # RIFE factor applied after the restore (1 = off)
    upscale_fps: Mapped[int | None] = mapped_column(SmallInteger)         # playback rate written; null = the source's own
    upscale_filename: Mapped[str | None] = mapped_column(String(512))     # e.g. "seg_0_up.mp4", sibling of `filename`
    # Dimensions the pass actually produced. Persisted rather than derived so
    # the merge can size its canvas from the rendition it is really feeding in.
    upscale_width: Mapped[int | None] = mapped_column(Integer)
    upscale_height: Mapped[int | None] = mapped_column(Integer)
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


# ─────────────────────────────────────────────────────────────────────────────
# Cloud-render budget (services/video_api/)
# ─────────────────────────────────────────────────────────────────────────────
# MiniMax bills per second of generated video, so the price of a call is known
# before it is made. That is what lets the limit be a hard gate instead of an
# after-the-fact warning — but only if money is *reserved* before the job is
# submitted and settled afterwards. Adding cost up after the fact breaks the
# moment two jobs run at once: both pass the check, both run, the limit is
# gone. See services/video_api/budget.py.


class BudgetPeriod(Base):
    """One month's spending limit, and the conversion it was set with."""

    __tablename__ = "budget_periods"
    __table_args__ = (
        CheckConstraint("limit_eur >= 0", name="ck_budget_periods_limit_positive"),
        CheckConstraint(
            "warn_threshold_pct BETWEEN 1 AND 100", name="ck_budget_periods_warn_pct"
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    month: Mapped[str] = mapped_column(String(7), nullable=False, unique=True)  # "2026-08"
    limit_eur: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False)
    warn_threshold_pct: Mapped[int] = mapped_column(
        SmallInteger, nullable=False, default=80, server_default="80"
    )
    # The USD→EUR rate this period books at. Stored per period rather than in
    # the config so the ledger stays reproducible: entries keep the amount they
    # were priced at, and changing the rate later cannot rewrite history.
    usd_eur_rate: Mapped[Decimal] = mapped_column(Numeric(8, 4), nullable=False)
    rate_updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, nullable=False
    )

    entries: Mapped[list["LedgerEntry"]] = relationship(
        back_populates="period", cascade="all, delete-orphan"
    )


class LedgerEntry(Base):
    """One reservation, and what became of it.

    Rows are never deleted and amounts are never rewritten — a finished job
    moves 'reserved' → 'settled' (possibly at a different amount, once the
    provider reports what it actually billed), a failed one moves
    'reserved' → 'released'.
    """

    __tablename__ = "ledger_entries"
    __table_args__ = (
        CheckConstraint(
            "state IN ('reserved', 'settled', 'released')", name="ck_ledger_entries_state"
        ),
        CheckConstraint(
            "kind IN ('generate_768p', 'generate_2k', 'regenerate_2k')",
            name="ck_ledger_entries_kind",
        ),
        # The hot query is "everything committed in this period".
        Index("ix_ledger_entries_period_state", "period_id", "state"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    period_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("budget_periods.id", ondelete="CASCADE"), nullable=False
    )
    video_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("videos.id", ondelete="SET NULL"), nullable=True, index=True
    )
    # Null until the provider accepted the submit. A reserved row that still has
    # no task id after a few minutes is one whose submit never landed, and the
    # startup reconciliation releases it — without that check, a crash between
    # reserving and submitting would eat budget permanently.
    task_id: Mapped[str | None] = mapped_column(String(128))
    # Guards against a retried submit booking twice.
    idempotency_key: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    kind: Mapped[str] = mapped_column(String(24), nullable=False)
    duration_s: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    ref_image_count: Mapped[int] = mapped_column(
        SmallInteger, nullable=False, default=0, server_default="0"
    )
    input_video_seconds: Mapped[Decimal] = mapped_column(
        Numeric(6, 2), nullable=False, default=Decimal("0"), server_default="0"
    )
    amount_usd: Mapped[Decimal] = mapped_column(Numeric(10, 4), nullable=False)
    amount_eur: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False)  # incl. safety factor while reserved
    # The rate this entry was priced at. Kept per entry, not read back off the
    # period: settling a job weeks later must convert at the rate it was booked
    # with, or a rate edit would silently re-price work already done.
    usd_eur_rate: Mapped[Decimal] = mapped_column(Numeric(8, 4), nullable=False)
    state: Mapped[str] = mapped_column(
        String(16), nullable=False, default="reserved", server_default="reserved", index=True
    )
    note: Mapped[str | None] = mapped_column(Text)   # why it was released, for the ledger view
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, nullable=False, index=True
    )
    settled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    period: Mapped["BudgetPeriod"] = relationship(back_populates="entries")
