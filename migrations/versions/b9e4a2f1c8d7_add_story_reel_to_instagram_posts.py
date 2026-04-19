"""add story and reel fields to instagram_posts

Revision ID: b9e4a2f1c8d7
Revises: f2a4c8e1b7d3
Create Date: 2026-04-17

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import ARRAY

revision: str = 'b9e4a2f1c8d7'
down_revision: Union[str, Sequence[str], None] = 'f2a4c8e1b7d3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('instagram_posts', sa.Column('story_status',   sa.String(32),  nullable=True))
    op.add_column('instagram_posts', sa.Column('story_media_ids', ARRAY(sa.String(128)), nullable=True))
    op.add_column('instagram_posts', sa.Column('reel_status',    sa.String(32),  nullable=True))
    op.add_column('instagram_posts', sa.Column('reel_media_id',  sa.String(128), nullable=True))


def downgrade() -> None:
    op.drop_column('instagram_posts', 'reel_media_id')
    op.drop_column('instagram_posts', 'reel_status')
    op.drop_column('instagram_posts', 'story_media_ids')
    op.drop_column('instagram_posts', 'story_status')
