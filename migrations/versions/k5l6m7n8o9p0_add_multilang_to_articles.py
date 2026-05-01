"""add multilingual + WP-link fields to articles

Revision ID: k5l6m7n8o9p0
Revises: j4k5l6m7n8o9
Create Date: 2026-05-01

Adds language tagging, translation grouping, excerpt, tags, and the WP
canonical link to the articles table — needed for Phase 3 multilingual
article generation.
"""
import uuid

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import ARRAY, UUID

revision = 'k5l6m7n8o9p0'
down_revision = 'j4k5l6m7n8o9'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('articles', sa.Column('excerpt', sa.Text, nullable=True))
    op.add_column('articles', sa.Column('tags', ARRAY(sa.String()), nullable=True))
    op.add_column(
        'articles',
        sa.Column('language', sa.String(length=8), nullable=False, server_default='en'),
    )
    op.add_column(
        'articles',
        sa.Column(
            'translation_group_id',
            UUID(as_uuid=True),
            nullable=False,
            server_default=sa.text('gen_random_uuid()'),
        ),
    )
    op.add_column('articles', sa.Column('wp_link', sa.Text, nullable=True))

    # Drop the server defaults — model-side defaults take over for new rows.
    op.alter_column('articles', 'language', server_default=None)
    op.alter_column('articles', 'translation_group_id', server_default=None)

    op.create_index(
        'ix_articles_translation_group_id', 'articles', ['translation_group_id']
    )


def downgrade() -> None:
    op.drop_index('ix_articles_translation_group_id', table_name='articles')
    op.drop_column('articles', 'wp_link')
    op.drop_column('articles', 'translation_group_id')
    op.drop_column('articles', 'language')
    op.drop_column('articles', 'tags')
    op.drop_column('articles', 'excerpt')
