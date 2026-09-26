"""add the image crop rendition (Zuschnitt)

A picture's own framing, chosen in the gallery and cut as a sibling file
between the upscale and the wand (services/image/crop.py). `crop_box` holds the
framing as fractions of the rendition it was cut from plus the aspect it was
drawn at; the other four columns describe the file it produced, the same
shape the upscale's columns have.

All nullable, nothing to backfill: null is "not cropped", which is every row
that exists today.

Revision ID: d6e7f8a9b0c1
Revises: c5d6e7f8a9b0
Create Date: 2026-09-27
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = 'd6e7f8a9b0c1'
down_revision = 'c5d6e7f8a9b0'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('images', sa.Column('crop_box', postgresql.JSONB(), nullable=True))
    op.add_column('images', sa.Column('cropped_filename', sa.String(length=512), nullable=True))
    op.add_column('images', sa.Column('cropped_filepath', sa.Text(), nullable=True))
    op.add_column('images', sa.Column('crop_width', sa.Integer(), nullable=True))
    op.add_column('images', sa.Column('crop_height', sa.Integer(), nullable=True))


def downgrade() -> None:
    for col in ('crop_height', 'crop_width', 'cropped_filepath', 'cropped_filename', 'crop_box'):
        op.drop_column('images', col)
