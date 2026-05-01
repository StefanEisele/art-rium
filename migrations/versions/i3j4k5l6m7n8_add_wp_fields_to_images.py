"""add WordPress fields to images

Revision ID: i3j4k5l6m7n8
Revises: h2i3j4k5l6m7
Create Date: 2026-04-26
"""
from alembic import op
import sqlalchemy as sa

revision = 'i3j4k5l6m7n8'
down_revision = 'h2i3j4k5l6m7'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('images', sa.Column('wp_media_id', sa.Integer, nullable=True))
    op.add_column('images', sa.Column('wp_source_url', sa.Text, nullable=True))
    op.add_column('images', sa.Column('wp_uploaded_at', sa.DateTime(timezone=True), nullable=True))
    op.add_column('images', sa.Column('wp_alt_text', sa.Text, nullable=True))
    op.add_column('images', sa.Column('wp_seo_description', sa.Text, nullable=True))
    op.add_column('images', sa.Column('wp_caption', sa.Text, nullable=True))


def downgrade() -> None:
    op.drop_column('images', 'wp_caption')
    op.drop_column('images', 'wp_seo_description')
    op.drop_column('images', 'wp_alt_text')
    op.drop_column('images', 'wp_uploaded_at')
    op.drop_column('images', 'wp_source_url')
    op.drop_column('images', 'wp_media_id')
