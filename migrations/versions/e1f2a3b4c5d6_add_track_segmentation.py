"""Link a derived control track to its source, and name a mask's colours.

Filmed footage can now be segmented into a colour-ID mask automatically (SAM
3.1, services/segment/), which turns a control track from something a user
uploads into something this tool also produces. Two columns follow from that:

  source_track_id  the track this one was derived from — the footage a mask was
                   keyed out of, or the longer take a trimmed copy came from.
                   A mask is only valid against the exact frames it was
                   segmented from, so the pairing has to be recorded rather
                   than left for the user to remember. ON DELETE SET NULL: the
                   mask stays usable as a file after its source is gone, it
                   just loses the link.

  regions          for kind="mask", what each colour is:
                   [{"color": [255,0,0], "label": "Tomate", "coverage": 0.31}].
                   A mask video is otherwise three anonymous silhouettes, and
                   the render UI needs the labels to ask which picture goes
                   into which region.

Null on every existing row: hand-authored Blender masks predate both and are
unaffected.

Revision ID: e1f2a3b4c5d6
Revises: d9e0f1a2b3c4
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = 'e1f2a3b4c5d6'
down_revision = 'd9e0f1a2b3c4'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "control_tracks",
        sa.Column("source_track_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.add_column(
        "control_tracks",
        sa.Column("regions", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )
    op.create_index(
        "ix_control_tracks_source_track_id", "control_tracks", ["source_track_id"],
    )
    op.create_foreign_key(
        "fk_control_tracks_source_track_id",
        "control_tracks", "control_tracks",
        ["source_track_id"], ["id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    op.drop_constraint(
        "fk_control_tracks_source_track_id", "control_tracks", type_="foreignkey",
    )
    op.drop_index("ix_control_tracks_source_track_id", table_name="control_tracks")
    op.drop_column("control_tracks", "regions")
    op.drop_column("control_tracks", "source_track_id")
