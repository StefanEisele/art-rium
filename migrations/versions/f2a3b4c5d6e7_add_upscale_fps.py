"""Record the frame rate an upscale/retime pass was asked to write.

The SEEDVR2 pass grew two dials that are independent of the restoration:
interpolation and a target playback rate. `upscale_rife` was already stored;
the rate was not, and re-rendering a pass (which happens whenever its source
changes — a soundtrack attached or dropped) replayed the stored settings
without it. A pass that had been asked for "3x, same length" therefore came
back three times longer, as slow motion.

Null on every existing row, which is the honest reading of them: they were all
rendered before a target rate could be asked for, so they all kept the source's
own.

Revision ID: f2a3b4c5d6e7
Revises: e1f2a3b4c5d6
"""
import sqlalchemy as sa
from alembic import op

revision = 'f2a3b4c5d6e7'
down_revision = 'e1f2a3b4c5d6'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "videos",
        sa.Column("upscale_fps", sa.SmallInteger(), nullable=True),
    )
    op.add_column(
        "video_clips",
        sa.Column("upscale_fps", sa.SmallInteger(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("video_clips", "upscale_fps")
    op.drop_column("videos", "upscale_fps")
