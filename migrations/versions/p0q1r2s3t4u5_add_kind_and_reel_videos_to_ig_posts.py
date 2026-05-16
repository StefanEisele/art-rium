"""add kind + reel_video_ids to instagram_posts, make image_id nullable

Adds support for standalone reel posts: an InstagramPost row can now hold
either a feed-style post (kind='feed', image_id set, optional carousel +
companions) or a reel-only post (kind='reel', image_id NULL, 1–4 source
videos in reel_video_ids that get concatenated into a single 1080×1920
MP4 before dispatch to the Pi outpost).

Revision ID: p0q1r2s3t4u5
Revises: o9p0q1r2s3t4
Create Date: 2026-05-16
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import ARRAY, UUID

revision = 'p0q1r2s3t4u5'
down_revision = 'o9p0q1r2s3t4'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        'instagram_posts',
        sa.Column('kind', sa.String(16), nullable=False, server_default='feed'),
    )
    op.add_column(
        'instagram_posts',
        sa.Column('reel_video_ids', ARRAY(UUID(as_uuid=True)), nullable=True),
    )
    op.alter_column('instagram_posts', 'image_id', nullable=True)


def downgrade() -> None:
    op.alter_column('instagram_posts', 'image_id', nullable=False)
    op.drop_column('instagram_posts', 'reel_video_ids')
    op.drop_column('instagram_posts', 'kind')
