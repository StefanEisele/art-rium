"""Add the beat-cut columns to videos.

A beat cut (workflow='beatcut') is a merge whose segment boundaries were placed
on a song's beat grid rather than at whatever length each clip happened to be.
Two things about it have to survive the request that made it:

  cut_plan                 the edit itself — style, seed, tempo and every shot.
                           Kept so the card can say what it is ("28 Schnitte ·
                           120 BPM · Dramaturgie") and so a plan can be rebuilt
                           or re-rolled later from the same numbers.

  soundtrack_start_seconds where in the song the picture begins. Only non-zero
                           when the user deliberately started at a later bar,
                           but then it is not optional: the soundtrack mux is
                           re-run from scratch on every upscale and grain pass,
                           and without this the offset would silently reset and
                           the whole edit would slide off its music.

Revision ID: c8d9e0f1a2b3
Revises: v1w2x3y4z5a6
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = 'c8d9e0f1a2b3'
down_revision = 'v1w2x3y4z5a6'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("videos", sa.Column("cut_plan", JSONB(), nullable=True))
    op.add_column("videos", sa.Column("soundtrack_start_seconds", sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column("videos", "soundtrack_start_seconds")
    op.drop_column("videos", "cut_plan")
