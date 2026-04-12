"""add thumbnail_path to images

Revision ID: c7f3a1b2d4e8
Revises: b3e9f1a2c4d5
Create Date: 2026-04-12

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = 'c7f3a1b2d4e8'
down_revision: Union[str, Sequence[str], None] = 'b3e9f1a2c4d5'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('images', sa.Column('thumbnail_path', sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column('images', 'thumbnail_path')
