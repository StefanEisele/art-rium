"""add lora_name/lora_strength columns to images

Revision ID: c1d2e3f4a5b6
Revises: b7c8d9e0f1a2
Create Date: 2026-07-10

Reconstruction metadata: every z-Image generation already carried a LoRA +
strength, but only the seed was persisted. Adding these lets a source image's
own LoRA/strength be reused as the default for consistency-critical follow-up
generations (e.g. story key-frames), and be reconstructed/audited later.
"""
import sqlalchemy as sa
from alembic import op

revision = 'c1d2e3f4a5b6'
down_revision = 'b7c8d9e0f1a2'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('images', sa.Column('lora_name', sa.String(256), nullable=True))
    op.add_column('images', sa.Column('lora_strength', sa.Numeric(4, 3), nullable=True))


def downgrade() -> None:
    op.drop_column('images', 'lora_strength')
    op.drop_column('images', 'lora_name')
