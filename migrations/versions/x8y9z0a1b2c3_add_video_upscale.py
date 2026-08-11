"""add upscale_resolution + upscale_filename to videos

Revision ID: x8y9z0a1b2c3
Revises: w7x8y9z0a1b2
Create Date: 2026-08-10
"""
from alembic import op
import sqlalchemy as sa

revision = 'x8y9z0a1b2c3'
down_revision = 'w7x8y9z0a1b2'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        'videos',
        sa.Column('upscale_resolution', sa.SmallInteger(), nullable=True),
    )
    op.add_column(
        'videos',
        sa.Column('upscale_filename', sa.String(512), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('videos', 'upscale_filename')
    op.drop_column('videos', 'upscale_resolution')
