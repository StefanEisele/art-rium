"""Add Instagram frame/crop settings and the AI-label flag

Instagram renders a feed post in one frame between 4:5 and 1.91:1 (the first
carousel child decides), so a 9:16 render gets padded into 4:5. `frame_ratio`
records which frame the post is planned for and the per-child `crop_*` columns
record whether that child should be pre-cropped to fill it, plus the baked
rendition that is actually published.

`ai_label` carries Meta's `is_ai_generated` self-disclosure per post.

Revision ID: e6f7a8b9c0d1
Revises: d5e6f7a8b9c0
"""
import sqlalchemy as sa
from alembic import op

revision = 'e6f7a8b9c0d1'
down_revision = 'd5e6f7a8b9c0'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        'instagram_posts',
        sa.Column('frame_ratio', sa.String(length=8), nullable=False,
                  server_default='auto'),
    )
    # Existing rows were planned without a label; only new posts pick up the
    # configured default, so a scheduled post never changes meaning under the
    # user's feet.
    op.add_column(
        'instagram_posts',
        sa.Column('ai_label', sa.Boolean(), nullable=False, server_default=sa.false()),
    )

    op.add_column(
        'instagram_post_media',
        sa.Column('crop_mode', sa.String(length=8), nullable=False,
                  server_default='fit'),
    )
    op.add_column(
        'instagram_post_media',
        sa.Column('crop_offset', sa.Float(), nullable=False, server_default='0.5'),
    )
    op.add_column('instagram_post_media', sa.Column('crop_filename', sa.String(length=512)))
    op.add_column('instagram_post_media', sa.Column('crop_filepath', sa.Text()))
    op.create_check_constraint(
        'ck_ig_post_media_crop_mode',
        'instagram_post_media',
        "crop_mode IN ('fit', 'fill')",
    )


def downgrade() -> None:
    op.drop_constraint('ck_ig_post_media_crop_mode', 'instagram_post_media', type_='check')
    op.drop_column('instagram_post_media', 'crop_filepath')
    op.drop_column('instagram_post_media', 'crop_filename')
    op.drop_column('instagram_post_media', 'crop_offset')
    op.drop_column('instagram_post_media', 'crop_mode')
    op.drop_column('instagram_posts', 'ai_label')
    op.drop_column('instagram_posts', 'frame_ratio')
