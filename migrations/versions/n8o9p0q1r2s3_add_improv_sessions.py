"""add improv_sessions table

Stores piano-improvisation sessions: a generated source video, the iPhone
recording uploaded by the user, and references to the two muxed output
videos (synth: source-video + piano-audio; hands: iPhone-clip with
normalised audio).

Revision ID: n8o9p0q1r2s3
Revises: m7n8o9p0q1r2
Create Date: 2026-05-14
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

revision = 'n8o9p0q1r2s3'
down_revision = 'm7n8o9p0q1r2'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'improv_sessions',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('source_video_id', UUID(as_uuid=True),
                  sa.ForeignKey('videos.id', ondelete='RESTRICT'), nullable=False),
        sa.Column('recording_filename', sa.String(512), nullable=False),
        sa.Column('mix_synth_video_id', UUID(as_uuid=True),
                  sa.ForeignKey('videos.id', ondelete='SET NULL'), nullable=True),
        sa.Column('mix_hands_video_id', UUID(as_uuid=True),
                  sa.ForeignKey('videos.id', ondelete='SET NULL'), nullable=True),
        sa.Column('status', sa.String(32), nullable=False, server_default='queued'),
        sa.Column('error', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.func.now()),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        'ix_improv_sessions_created_at',
        'improv_sessions', ['created_at'],
    )


def downgrade() -> None:
    op.drop_index('ix_improv_sessions_created_at', table_name='improv_sessions')
    op.drop_table('improv_sessions')
