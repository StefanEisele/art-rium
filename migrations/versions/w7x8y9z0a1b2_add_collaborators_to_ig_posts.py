"""add collaborators to instagram_posts

Revision ID: w7x8y9z0a1b2
Revises: v6w7x8y9z0a1
Create Date: 2026-08-01
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = 'w7x8y9z0a1b2'
down_revision = 'v6w7x8y9z0a1'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        'instagram_posts',
        sa.Column('collaborators', postgresql.ARRAY(sa.String(30)), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('instagram_posts', 'collaborators')
