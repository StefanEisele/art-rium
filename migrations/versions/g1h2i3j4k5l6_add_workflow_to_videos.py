"""add workflow column to videos

Revision ID: g1h2i3j4k5l6
Revises: a1b2c3d4e5f6
Create Date: 2026-04-24
"""
from alembic import op
import sqlalchemy as sa

revision = 'g1h2i3j4k5l6'
down_revision = 'a1b2c3d4e5f6'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('videos', sa.Column('workflow', sa.String(32), nullable=True))


def downgrade() -> None:
    op.drop_column('videos', 'workflow')
