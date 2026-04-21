"""add companion_time to instagram_posts

Revision ID: d4e6f0a2b8c1
Revises: c3d5e7f2a1b9
Create Date: 2026-04-19

Adds companion_time (HH:MM string) so the scheduler can snap day+ companion
posts to a specific time of day instead of using a pure minute offset.
"""
from alembic import op
import sqlalchemy as sa

revision = 'd4e6f0a2b8c1'
down_revision = 'c3d5e7f2a1b9'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('instagram_posts', sa.Column('companion_time', sa.String(5), nullable=True))


def downgrade() -> None:
    op.drop_column('instagram_posts', 'companion_time')
