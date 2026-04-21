"""add companion scheduling to instagram_posts

Revision ID: c3d5e7f2a1b9
Revises: b9e4a2f1c8d7
Create Date: 2026-04-19

Adds story_delay_minutes, reel_delay_minutes, story_scheduled_at,
reel_scheduled_at so that companion posts can be scheduled independently
from the feed post.
"""
from alembic import op
import sqlalchemy as sa

revision = 'c3d5e7f2a1b9'
down_revision = 'b9e4a2f1c8d7'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('instagram_posts', sa.Column('story_delay_minutes', sa.Integer(), nullable=True))
    op.add_column('instagram_posts', sa.Column('reel_delay_minutes',  sa.Integer(), nullable=True))
    op.add_column('instagram_posts', sa.Column('story_scheduled_at',  sa.DateTime(timezone=True), nullable=True))
    op.add_column('instagram_posts', sa.Column('reel_scheduled_at',   sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    op.drop_column('instagram_posts', 'reel_scheduled_at')
    op.drop_column('instagram_posts', 'story_scheduled_at')
    op.drop_column('instagram_posts', 'reel_delay_minutes')
    op.drop_column('instagram_posts', 'story_delay_minutes')
