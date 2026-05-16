"""add mix_pip_video_id to improv_sessions

Adds the third improv-output reference: source video with the hands
recording as a small picture-in-picture inset.

Revision ID: o9p0q1r2s3t4
Revises: n8o9p0q1r2s3
Create Date: 2026-05-16
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

revision = 'o9p0q1r2s3t4'
down_revision = 'n8o9p0q1r2s3'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        'improv_sessions',
        sa.Column(
            'mix_pip_video_id',
            UUID(as_uuid=True),
            sa.ForeignKey('videos.id', ondelete='SET NULL'),
            nullable=True,
        ),
    )


def downgrade() -> None:
    op.drop_column('improv_sessions', 'mix_pip_video_id')
