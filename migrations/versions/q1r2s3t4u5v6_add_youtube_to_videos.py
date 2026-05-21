"""add youtube_* columns to videos

Adds idempotent tracking for YouTube uploads. Once a Video row has
`youtube_video_id` set, the article job's youtube-phase skips it and reuses
the existing URL for the wp:embed block.

Revision ID: q1r2s3t4u5v6
Revises: p0q1r2s3t4u5
Create Date: 2026-05-21
"""
from alembic import op
import sqlalchemy as sa

revision = 'q1r2s3t4u5v6'
down_revision = 'p0q1r2s3t4u5'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('videos', sa.Column('youtube_video_id', sa.String(length=32), nullable=True))
    op.add_column('videos', sa.Column('youtube_url',      sa.Text(),            nullable=True))
    op.add_column('videos', sa.Column('youtube_privacy',  sa.String(length=16), nullable=True))
    op.add_column('videos', sa.Column('youtube_uploaded_at', sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    op.drop_column('videos', 'youtube_uploaded_at')
    op.drop_column('videos', 'youtube_privacy')
    op.drop_column('videos', 'youtube_url')
    op.drop_column('videos', 'youtube_video_id')
