"""add songs table (ACE-Step 1.5 Turbo audio generation)

Revision ID: t4u5v6w7x8y9
Revises: s3t4u5v6w7x8
Create Date: 2026-05-30
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

revision = 't4u5v6w7x8y9'
down_revision = 's3t4u5v6w7x8'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'songs',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('filename', sa.String(512), unique=True, nullable=True),
        sa.Column('filepath', sa.Text, nullable=True),
        sa.Column('tags', sa.Text, nullable=False),
        sa.Column('lyrics', sa.Text, nullable=True),
        sa.Column('duration_seconds', sa.Integer, nullable=False),
        sa.Column('bpm', sa.Integer, nullable=True),
        sa.Column('musical_key', sa.String(32), nullable=True),
        sa.Column('language', sa.String(8), nullable=True),
        sa.Column('seed', sa.BigInteger, nullable=True),
        sa.Column('steps', sa.SmallInteger, nullable=False, server_default='8'),
        sa.Column('cfg', sa.Numeric(4, 2), nullable=True),
        sa.Column('shift', sa.Numeric(4, 2), nullable=True),
        sa.Column('title', sa.String(255), nullable=True),
        sa.Column('notes', sa.Text, nullable=True),
        sa.Column('status', sa.String(32), nullable=False, server_default='generating'),
        sa.Column('error', sa.Text, nullable=True),
        sa.Column('comfy_prompt_id', sa.String(128), nullable=True),
        sa.Column('workflow', sa.String(32), nullable=True),
        sa.Column('waveform_path', sa.Text, nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    op.drop_table('songs')
