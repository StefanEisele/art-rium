"""add reel_video_id to instagram_posts

Revision ID: a1b2c3d4e5f6
Revises: f1a3b5c7d9e2
Create Date: 2026-04-20
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

revision = 'a1b2c3d4e5f6'
down_revision = 'f1a3b5c7d9e2'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('instagram_posts', sa.Column('reel_video_id', UUID(as_uuid=True), nullable=True))


def downgrade() -> None:
    op.drop_column('instagram_posts', 'reel_video_id')
