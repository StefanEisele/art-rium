"""add soundtrack_song_id + muxed_filename to videos

Revision ID: u5v6w7x8y9z0
Revises: t4u5v6w7x8y9
Create Date: 2026-05-30
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

revision = 'u5v6w7x8y9z0'
down_revision = 't4u5v6w7x8y9'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        'videos',
        sa.Column(
            'soundtrack_song_id',
            UUID(as_uuid=True),
            sa.ForeignKey('songs.id', ondelete='SET NULL'),
            nullable=True,
        ),
    )
    op.add_column(
        'videos',
        sa.Column('muxed_filename', sa.String(512), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('videos', 'muxed_filename')
    op.drop_column('videos', 'soundtrack_song_id')
