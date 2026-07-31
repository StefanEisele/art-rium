"""add grain_strength + grain_filename to videos

Revision ID: v6w7x8y9z0a1
Revises: d4e5f6a7b8c9
Create Date: 2026-07-31
"""
from alembic import op
import sqlalchemy as sa

revision = 'v6w7x8y9z0a1'
down_revision = 'd4e5f6a7b8c9'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        'videos',
        sa.Column('grain_strength', sa.SmallInteger(), nullable=True),
    )
    op.add_column(
        'videos',
        sa.Column('grain_filename', sa.String(512), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('videos', 'grain_filename')
    op.drop_column('videos', 'grain_strength')
