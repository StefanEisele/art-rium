"""Record what the Wan sampler was told, per clip.

Wan 2.2's step count and the distill strength on its high-noise expert stopped
being fixed numbers in the builder and became the two dials in the UI: steps buy
resolved detail, and the high-noise distill decides whether the clip moves at
all (lower = more motion). They are dials worth finding the right value for, and
a clip whose settings are not written down cannot be compared against the next
one — so both land on the row, next to the prompt and the frame count that were
already kept for the same reason.

Null on every existing row and on every MiniMax clip: MiniMax H3 runs its own
fixed recipe and neither dial reaches it. The frontend reads null as "before
this was recorded" rather than as a value.

  wan_steps      total sampler steps across both experts (4-16)
  wan_lora_high  distill LoRA strength on the high-noise expert (0.0-1.0)

Revision ID: d9e0f1a2b3c4
Revises: c8d9e0f1a2b3
"""
import sqlalchemy as sa
from alembic import op

revision = 'd9e0f1a2b3c4'
down_revision = 'c8d9e0f1a2b3'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("video_clips", sa.Column("wan_steps", sa.SmallInteger(), nullable=True))
    op.add_column("video_clips", sa.Column("wan_lora_high", sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("video_clips", "wan_lora_high")
    op.drop_column("video_clips", "wan_steps")
