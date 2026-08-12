"""add soundtrack_bed_volume to videos

Revision ID: b3c4d5e6f7a8
Revises: a2b3c4d5e6f7
Create Date: 2026-08-12

Null keeps the historical behaviour (the song replaces the clip's own audio);
a value is the volume the clip's generated sound plays at underneath it.
"""
from alembic import op
import sqlalchemy as sa

revision = 'b3c4d5e6f7a8'
down_revision = 'a2b3c4d5e6f7'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('videos', sa.Column('soundtrack_bed_volume', sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column('videos', 'soundtrack_bed_volume')
