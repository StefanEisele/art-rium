"""add image_series + image_series_items — curated picture packages

A *series* is the package a set of pictures is posted as: named, ordered, and
kept, so it can be handed to the titler, the video tool, an article and an
Instagram carousel without being reassembled by hand each time.

Until now the only grouping on `images` was `batch_id`, minted per
/api/generate call — which records how pictures were *made*, not how they
belong together. Curating is the whole point, so this is its own pair of
tables rather than a column on `images`.

A picture may belong to several series: the gallery shows series in their own
mode, so nothing disappears from the picture grid and a second membership
costs nothing.

Two constraints carry real weight:
  * UNIQUE(series_id, position) — the order is the post's order, so it has to
    be an order. Reordering therefore goes clear → flush → re-append (see
    services/instagram/media.py::replace_media_items), never UPDATE-in-place:
    swapping two positions would collide mid-flush.
  * UNIQUE(series_id, image_id) — the same picture twice in one carousel is
    always a mistake.

`position` is allowed to have gaps. Deleting a picture cascades its item row
away and leaves 0,1,3 behind; members are read ORDER BY position and numbered
by their index in the answer, the same convention the clip strip uses.
Appending must therefore be MAX(position)+1, never len(items).

Revision ID: b4c5d6e7f8a9
Revises: a3b4c5d6e7f8
Create Date: 2026-09-18
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

revision = 'b4c5d6e7f8a9'
down_revision = 'a3b4c5d6e7f8'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'image_series',
        sa.Column('id',         UUID(as_uuid=True), primary_key=True),
        sa.Column('title',      sa.String(512),     nullable=True),
        sa.Column('notes',      sa.Text(),          nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )

    op.create_table(
        'image_series_items',
        sa.Column('id',         UUID(as_uuid=True), primary_key=True),
        sa.Column('series_id',  UUID(as_uuid=True), sa.ForeignKey('image_series.id', ondelete='CASCADE'), nullable=False),
        sa.Column('image_id',   UUID(as_uuid=True), sa.ForeignKey('images.id', ondelete='CASCADE'), nullable=False),
        sa.Column('position',   sa.Integer(),       nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint('series_id', 'position', name='uq_image_series_items_position'),
        sa.UniqueConstraint('series_id', 'image_id', name='uq_image_series_items_image'),
    )
    op.create_index('ix_image_series_items_series_id', 'image_series_items', ['series_id'])
    # The gallery asks "which series is this picture in?" for every tile on a
    # page, which is a lookup by image, not by series.
    op.create_index('ix_image_series_items_image_id', 'image_series_items', ['image_id'])


def downgrade() -> None:
    op.drop_index('ix_image_series_items_image_id', table_name='image_series_items')
    op.drop_index('ix_image_series_items_series_id', table_name='image_series_items')
    op.drop_table('image_series_items')
    op.drop_table('image_series')
