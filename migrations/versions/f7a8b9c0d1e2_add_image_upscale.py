"""Add the diffusion-upscale rendition to images

Ultimate SD Upscale driven by Z-Image Turbo (services/image/upscale.py). The
rendition sits between the enhancement and the grain, so the chain becomes
original → enhanced → upscaled → grained.

Revision ID: f7a8b9c0d1e2
Revises: e6f7a8b9c0d1
"""
import sqlalchemy as sa
from alembic import op

revision = 'f7a8b9c0d1e2'
down_revision = 'e6f7a8b9c0d1'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('images', sa.Column('upscale_scale', sa.Float()))
    op.add_column('images', sa.Column('upscale_denoise', sa.Float()))
    op.add_column('images', sa.Column('upscale_model', sa.String(length=32)))
    op.add_column('images', sa.Column('upscaled_filename', sa.String(length=512)))
    op.add_column('images', sa.Column('upscaled_filepath', sa.Text()))
    op.add_column('images', sa.Column('upscale_width', sa.Integer()))
    op.add_column('images', sa.Column('upscale_height', sa.Integer()))


def downgrade() -> None:
    for col in (
        'upscale_height', 'upscale_width', 'upscaled_filepath',
        'upscaled_filename', 'upscale_model', 'upscale_denoise', 'upscale_scale',
    ):
        op.drop_column('images', col)
