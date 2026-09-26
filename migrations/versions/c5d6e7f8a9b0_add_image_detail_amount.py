"""add images.detail_amount — the Detail Daemon dial a picture was sampled at

Reconstruction metadata, in the same spirit as `loras`: "Weiterarbeiten" opens
the generator with the recipe a picture was made from, and without this the
one dial that changes how much material the model invents would silently reset
to 0 every time.

Nullable, and null means two different-looking things that are the same thing:
rows generated before this existed, and rows generated with the dial at 0.
Both sampled identically — the sampler chain at detail 0 is pixel-identical to
the KSampler it replaced — so there is nothing to backfill.

Revision ID: c5d6e7f8a9b0
Revises: b4c5d6e7f8a9
Create Date: 2026-09-19
"""
from alembic import op
import sqlalchemy as sa

revision = 'c5d6e7f8a9b0'
down_revision = 'b4c5d6e7f8a9'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('images', sa.Column('detail_amount', sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column('images', 'detail_amount')
