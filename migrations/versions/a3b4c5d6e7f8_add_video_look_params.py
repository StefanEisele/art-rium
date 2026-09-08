"""Store the whole look, not just the grain strength.

The film-grain pass became a look pass: correction, grade, optics and grain in
one ffmpeg chain (services/video/look.py). Six of its seven dials had nowhere
to live, so they live here as one JSON object.

`grain_strength` stays and keeps meaning the grain dial — services/improv and
the gallery both ask that column whether a video is grained, and neither should
have to learn about this one to get an answer. It is mirrored out of
`look_params` on every render.

Null on every existing row. Rows written before this read back as a grain-only
look built from `grain_strength`, which is exactly what they were.

Revision ID: a3b4c5d6e7f8
Revises: f2a3b4c5d6e7
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = 'a3b4c5d6e7f8'
down_revision = 'f2a3b4c5d6e7'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "videos",
        sa.Column("look_params", postgresql.JSONB(astext_type=sa.Text()),
                  nullable=True),
    )


def downgrade() -> None:
    op.drop_column("videos", "look_params")
