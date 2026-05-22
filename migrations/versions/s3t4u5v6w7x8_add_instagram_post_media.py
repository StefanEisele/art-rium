"""add instagram_post_media table for mixed image+video carousels

Introduces an ordered media-items table so a feed post can mix images and
videos in one carousel (Meta Graph API supports up to 10 mixed children).

This is a fresh-start migration: existing `scheduled` feed posts get
cancelled before the old image columns are dropped, because there's no
backfill into the new table. Outpost-dispatched posts that have already
been uploaded to the Pi keep their outpost_id and continue posting from
there — the local row just loses its image references.

Revision ID: s3t4u5v6w7x8
Revises: r2s3t4u5v6w7
Create Date: 2026-05-22
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

revision = 's3t4u5v6w7x8'
down_revision = 'r2s3t4u5v6w7'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. New media-items table (image OR video per position, 0..9 per post).
    op.create_table(
        'instagram_post_media',
        sa.Column('id',         UUID(as_uuid=True), primary_key=True),
        sa.Column('post_id',    UUID(as_uuid=True), sa.ForeignKey('instagram_posts.id', ondelete='CASCADE'), nullable=False),
        sa.Column('position',   sa.Integer(),       nullable=False),
        sa.Column('kind',       sa.String(8),       nullable=False),
        sa.Column('image_id',   UUID(as_uuid=True), sa.ForeignKey('images.id', ondelete='CASCADE'), nullable=True),
        sa.Column('video_id',   UUID(as_uuid=True), sa.ForeignKey('videos.id', ondelete='CASCADE'), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint('post_id', 'position', name='uq_ig_post_media_position'),
        sa.CheckConstraint(
            "(kind = 'image' AND image_id IS NOT NULL AND video_id IS NULL) OR "
            "(kind = 'video' AND video_id IS NOT NULL AND image_id IS NULL)",
            name='ck_ig_post_media_kind_xor',
        ),
    )
    op.create_index('ix_ig_post_media_post_id', 'instagram_post_media', ['post_id'])

    # 2. Cancel any feed posts that depended on the old image columns.
    #    Outpost rows keep going — their media is already at the Pi.
    op.execute("""
        UPDATE instagram_posts
        SET status = 'cancelled',
            error = COALESCE(error, '') ||
                    CASE WHEN error IS NULL OR error = '' THEN '' ELSE E'\n' END ||
                    'Cancelled by schema migration: mixed-media refactor dropped image_id/carousel_image_ids. Re-create the post.',
            updated_at = now()
        WHERE kind = 'feed'
          AND status = 'scheduled'
          AND (dispatch_target IS NULL OR dispatch_target <> 'outpost' OR outpost_id IS NULL)
    """)

    # 3. Drop the old image columns. Reel-mode (reel_video_ids) is unaffected.
    op.drop_column('instagram_posts', 'carousel_image_ids')
    op.drop_column('instagram_posts', 'image_id')


def downgrade() -> None:
    op.add_column(
        'instagram_posts',
        sa.Column('image_id', UUID(as_uuid=True), sa.ForeignKey('images.id'), nullable=True),
    )
    op.add_column(
        'instagram_posts',
        sa.Column('carousel_image_ids', sa.dialects.postgresql.ARRAY(UUID(as_uuid=True)), nullable=True),
    )
    op.drop_index('ix_ig_post_media_post_id', table_name='instagram_post_media')
    op.drop_table('instagram_post_media')
