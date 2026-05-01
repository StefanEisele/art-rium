"""add wp_seo_title to images

Revision ID: j4k5l6m7n8o9
Revises: i3j4k5l6m7n8
Create Date: 2026-05-01
"""
from alembic import op
import sqlalchemy as sa

revision = 'j4k5l6m7n8o9'
down_revision = 'i3j4k5l6m7n8'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('images', sa.Column('wp_seo_title', sa.String(length=120), nullable=True))


def downgrade() -> None:
    op.drop_column('images', 'wp_seo_title')
