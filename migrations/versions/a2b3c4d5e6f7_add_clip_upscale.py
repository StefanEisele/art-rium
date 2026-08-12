"""add per-clip SEEDVR2 upscale columns to video_clips

Revision ID: a2b3c4d5e6f7
Revises: z0a1b2c3d4e5
Create Date: 2026-08-12

NB: the obvious next id in this file's naming series, a1b2c3d4e5f6, is
already taken by add_reel_video_id_to_instagram_posts — the sequence wrapped
around. Alembic reports a duplicate as "multiple heads", which reads like a
branching problem rather than a name clash.
"""
from alembic import op
import sqlalchemy as sa

revision = 'a2b3c4d5e6f7'
down_revision = 'z0a1b2c3d4e5'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('video_clips', sa.Column('upscale_resolution', sa.SmallInteger(), nullable=True))
    op.add_column('video_clips', sa.Column('upscale_rife', sa.SmallInteger(), nullable=True))
    op.add_column('video_clips', sa.Column('upscale_filename', sa.String(512), nullable=True))
    op.add_column('video_clips', sa.Column('upscale_width', sa.Integer(), nullable=True))
    op.add_column('video_clips', sa.Column('upscale_height', sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column('video_clips', 'upscale_height')
    op.drop_column('video_clips', 'upscale_width')
    op.drop_column('video_clips', 'upscale_filename')
    op.drop_column('video_clips', 'upscale_rife')
    op.drop_column('video_clips', 'upscale_resolution')
