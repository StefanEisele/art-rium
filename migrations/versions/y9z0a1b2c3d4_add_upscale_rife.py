"""add upscale_rife to videos

Revision ID: y9z0a1b2c3d4
Revises: x8y9z0a1b2c3
Create Date: 2026-08-10
"""
from alembic import op
import sqlalchemy as sa

revision = 'y9z0a1b2c3d4'
down_revision = 'x8y9z0a1b2c3'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        'videos',
        sa.Column('upscale_rife', sa.SmallInteger(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('videos', 'upscale_rife')
