"""add auto-enhance rendition columns to images

Revision ID: z0a1b2c3d4e5
Revises: y9z0a1b2c3d4
Create Date: 2026-08-11
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = 'z0a1b2c3d4e5'
down_revision = 'y9z0a1b2c3d4'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('images', sa.Column('enhance_strength', sa.SmallInteger(), nullable=True))
    op.add_column('images', sa.Column('enhanced_filename', sa.String(512), nullable=True))
    op.add_column('images', sa.Column('enhanced_filepath', sa.Text(), nullable=True))
    op.add_column('images', sa.Column('enhance_params', postgresql.JSONB(), nullable=True))


def downgrade() -> None:
    op.drop_column('images', 'enhance_params')
    op.drop_column('images', 'enhanced_filepath')
    op.drop_column('images', 'enhanced_filename')
    op.drop_column('images', 'enhance_strength')
