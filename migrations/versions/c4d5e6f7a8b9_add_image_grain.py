"""add image grain rendition

Revision ID: c4d5e6f7a8b9
Revises: b3c4d5e6f7a8
Create Date: 2026-08-15

The film-grain pass over a single image, mirroring the enhance columns added
in z0a1b2c3d4e5. Nullable throughout: no grain is the normal state.
"""
from alembic import op
import sqlalchemy as sa

revision = 'c4d5e6f7a8b9'
down_revision = 'b3c4d5e6f7a8'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('images', sa.Column('grain_strength', sa.SmallInteger(), nullable=True))
    op.add_column('images', sa.Column('grained_filename', sa.String(length=512), nullable=True))
    op.add_column('images', sa.Column('grained_filepath', sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column('images', 'grained_filepath')
    op.drop_column('images', 'grained_filename')
    op.drop_column('images', 'grain_strength')
