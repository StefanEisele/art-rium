"""add outpost dispatch columns to instagram_posts

Tracks which posts were dispatched to the Pi posting outpost
(ig.stefaneisele.com) so the local scheduler does not also publish them.

Revision ID: m7n8o9p0q1r2
Revises: l6m7n8o9p0q1
Create Date: 2026-05-08
"""
from alembic import op
import sqlalchemy as sa

revision = 'm7n8o9p0q1r2'
down_revision = 'l6m7n8o9p0q1'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        'instagram_posts',
        sa.Column('dispatch_target', sa.String(16), nullable=False, server_default='local'),
    )
    op.add_column('instagram_posts', sa.Column('outpost_id', sa.String(64), nullable=True))
    op.add_column('instagram_posts', sa.Column('outpost_status', sa.String(32), nullable=True))
    op.add_column('instagram_posts', sa.Column('outpost_reel_status', sa.String(32), nullable=True))
    op.add_column(
        'instagram_posts',
        sa.Column('outpost_dispatched_at', sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('instagram_posts', 'outpost_dispatched_at')
    op.drop_column('instagram_posts', 'outpost_reel_status')
    op.drop_column('instagram_posts', 'outpost_status')
    op.drop_column('instagram_posts', 'outpost_id')
    op.drop_column('instagram_posts', 'dispatch_target')
