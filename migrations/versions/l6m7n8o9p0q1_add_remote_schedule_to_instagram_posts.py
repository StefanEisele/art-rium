"""add remote-schedule columns to instagram_posts

Tracks Instagram-side scheduled containers so the local server can be
offline between scheduling and publication. See services/instagram/publisher.py.

Revision ID: l6m7n8o9p0q1
Revises: k5l6m7n8o9p0
Create Date: 2026-05-06
"""
from alembic import op
import sqlalchemy as sa

revision = 'l6m7n8o9p0q1'
down_revision = 'k5l6m7n8o9p0'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('instagram_posts', sa.Column('feed_creation_id', sa.String(128), nullable=True))
    op.add_column('instagram_posts', sa.Column('reel_creation_id', sa.String(128), nullable=True))
    op.add_column('instagram_posts', sa.Column('reel_video_filename', sa.String(512), nullable=True))


def downgrade() -> None:
    op.drop_column('instagram_posts', 'reel_video_filename')
    op.drop_column('instagram_posts', 'reel_creation_id')
    op.drop_column('instagram_posts', 'feed_creation_id')
