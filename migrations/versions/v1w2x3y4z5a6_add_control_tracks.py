"""Add control_tracks for the VACE structure-video workflow.

A control track (Blender depth pass, object-ID mask render, or plain footage)
is an asset, not a job input — the same turntable is rendered at many strengths
while the look is being found, so it is uploaded once and referenced by id.

Revision ID: v1w2x3y4z5a6
Revises: f7a8b9c0d1e2
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import UUID

revision = 'v1w2x3y4z5a6'
down_revision = 'f7a8b9c0d1e2'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "control_tracks",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("filename", sa.String(512), nullable=False),
        sa.Column("filepath", sa.Text(), nullable=False),
        sa.Column("kind", sa.String(16), nullable=False),
        sa.Column("title", sa.String(255)),
        sa.Column("thumbnail_path", sa.Text()),
        sa.Column("width", sa.Integer()),
        sa.Column("height", sa.Integer()),
        sa.Column("frame_count", sa.Integer()),
        sa.Column("fps", sa.Float()),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "kind IN ('depth', 'footage', 'mask')", name="ck_control_tracks_kind"
        ),
    )
    op.create_index("ix_control_tracks_created_at", "control_tracks", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_control_tracks_created_at", table_name="control_tracks")
    op.drop_table("control_tracks")
