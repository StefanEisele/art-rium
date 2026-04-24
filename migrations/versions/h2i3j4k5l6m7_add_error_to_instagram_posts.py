"""add error column to instagram_posts

Revision ID: h2i3j4k5l6m7
Revises: g1h2i3j4k5l6
Create Date: 2026-04-24
"""
from alembic import op
import sqlalchemy as sa

revision = 'h2i3j4k5l6m7'
down_revision = 'g1h2i3j4k5l6'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('instagram_posts', sa.Column('error', sa.Text, nullable=True))


def downgrade() -> None:
    op.drop_column('instagram_posts', 'error')
