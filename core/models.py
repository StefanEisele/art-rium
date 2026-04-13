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
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, nullable=False
    )

    shop_listings: Mapped[list["ShopListing"]] = relationship(
        back_populates="image", cascade="all, delete-orphan"
    )
    instagram_posts: Mapped[list["InstagramPost"]] = relationship(
        back_populates="image", cascade="all, delete-orphan"
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
    wp_post_id: Mapped[int | None] = mapped_column(Integer)        # null until pushed
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="draft"
    )                                                               # draft | pushed | published
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
    caption: Mapped[str | None] = mapped_column(Text)
    scheduled_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="scheduled"
    )                                                               # scheduled | posted | cancelled
    instagram_media_id: Mapped[str | None] = mapped_column(String(128))  # filled after posting
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now, nullable=False
    )

    image: Mapped["Image"] = relationship(back_populates="instagram_posts")
