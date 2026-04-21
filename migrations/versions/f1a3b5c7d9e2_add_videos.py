"""add videos table

Revision ID: f1a3b5c7d9e2
Revises: d4e6f0a2b8c1
Create Date: 2026-04-19
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import ARRAY, UUID

revision = 'f1a3b5c7d9e2'
down_revision = 'd4e6f0a2b8c1'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'videos',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('filename', sa.String(512), unique=True, nullable=True),
        sa.Column('filepath', sa.Text, nullable=True),
        sa.Column('image_ids', ARRAY(UUID(as_uuid=True)), nullable=True),
        sa.Column('prompt', sa.Text, nullable=True),
        sa.Column('width', sa.Integer, nullable=True),
        sa.Column('height', sa.Integer, nullable=True),
        sa.Column('frame_count', sa.Integer, nullable=True),
        sa.Column('n_images', sa.Integer, nullable=True),
        sa.Column('fps', sa.Integer, nullable=True),
        sa.Column('status', sa.String(32), nullable=False, server_default='generating'),
        sa.Column('error', sa.Text, nullable=True),
        sa.Column('comfy_prompt_id', sa.String(128), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    op.drop_table('videos')
