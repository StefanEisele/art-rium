"""add title + notes columns to videos

User-editable display title and free-form notes for generated videos.
Separate from `prompt` (the LoRA-augmented generation input) so the user can
overwrite without losing the original prompt for reference.

Revision ID: r2s3t4u5v6w7
Revises: q1r2s3t4u5v6
Create Date: 2026-05-22
"""
from alembic import op
import sqlalchemy as sa

revision = 'r2s3t4u5v6w7'
down_revision = 'q1r2s3t4u5v6'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('videos', sa.Column('title', sa.String(length=255), nullable=True))
    op.add_column('videos', sa.Column('notes', sa.Text(),             nullable=True))


def downgrade() -> None:
    op.drop_column('videos', 'notes')
    op.drop_column('videos', 'title')
