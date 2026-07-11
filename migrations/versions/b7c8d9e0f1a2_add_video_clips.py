"""add video_clips table (per-segment clip library)

Revision ID: b7c8d9e0f1a2
Revises: f0a7f5dc6258
Create Date: 2026-07-09

Segment clips become first-class library items: every video job persists
each rendered segment as a video_clips row (its "stack"), and clips from
any number of jobs can be merged into a new Video row (workflow="merge").
Replaces the per-job review/assemble flow; legacy status='review' jobs are
backfilled from their segments meta.json at startup (core.startup_sweep).
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision = 'b7c8d9e0f1a2'
down_revision = 'f0a7f5dc6258'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'video_clips',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('video_id', UUID(as_uuid=True), sa.ForeignKey('videos.id', ondelete='CASCADE'), nullable=False),
        sa.Column('idx', sa.Integer, nullable=False),
        sa.Column('filename', sa.String(512), nullable=False),
        sa.Column('thumb', sa.String(512), nullable=False),
        sa.Column('prompt', sa.Text, nullable=True),
        sa.Column('frame_count', sa.Integer, nullable=True),
        sa.Column('workflow', sa.String(32), nullable=False),
        sa.Column('width', sa.Integer, nullable=True),
        sa.Column('height', sa.Integer, nullable=True),
        sa.Column('fps', sa.Integer, nullable=True),
        sa.Column('has_audio', sa.Boolean, nullable=False, server_default=sa.false()),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index('ix_video_clips_video_id', 'video_clips', ['video_id'])
    op.create_unique_constraint('uq_video_clips_video_idx', 'video_clips', ['video_id', 'idx'])


def downgrade() -> None:
    op.drop_table('video_clips')
