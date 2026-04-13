"""add instagram_posts table

Revision ID: e5b2d9f1a3c6
Revises: c7f3a1b2d4e8
Create Date: 2026-04-13

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

revision: str = 'e5b2d9f1a3c6'
down_revision: Union[str, Sequence[str], None] = 'c7f3a1b2d4e8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'instagram_posts',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('image_id', UUID(as_uuid=True), sa.ForeignKey('images.id'), nullable=False),
        sa.Column('caption', sa.Text(), nullable=True),
        sa.Column('scheduled_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('status', sa.String(32), nullable=False, server_default='scheduled'),
        sa.Column('instagram_media_id', sa.String(128), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index('ix_instagram_posts_scheduled_at', 'instagram_posts', ['scheduled_at'])
    op.create_index('ix_instagram_posts_status', 'instagram_posts', ['status'])


def downgrade() -> None:
    op.drop_index('ix_instagram_posts_status', 'instagram_posts')
    op.drop_index('ix_instagram_posts_scheduled_at', 'instagram_posts')
    op.drop_table('instagram_posts')
