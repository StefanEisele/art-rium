"""add carousel_image_ids to instagram_posts

Revision ID: f2a4c8e1b7d3
Revises: e5b2d9f1a3c6
Create Date: 2026-04-17

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import ARRAY, UUID

revision: str = 'f2a4c8e1b7d3'
down_revision: Union[str, Sequence[str], None] = 'e5b2d9f1a3c6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'instagram_posts',
        sa.Column('carousel_image_ids', ARRAY(UUID(as_uuid=True)), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('instagram_posts', 'carousel_image_ids')
