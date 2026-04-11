"""add title to images

Revision ID: b3e9f1a2c4d5
Revises: 5e4e2eac785b
Create Date: 2026-04-11

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = 'b3e9f1a2c4d5'
down_revision: Union[str, Sequence[str], None] = '5e4e2eac785b'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('images', sa.Column('title', sa.String(512), nullable=True))


def downgrade() -> None:
    op.drop_column('images', 'title')
