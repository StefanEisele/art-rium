"""add embedding_trainings (textual inversion from gallery pictures)

One row per embedding trained with kohya sd-scripts (services/embedding/).
The files live in ComfyUI's embeddings folder; this records what they were
trained from, with which settings, and which snapshot is the active one.

Revision ID: e7f8a9b0c1d2
Revises: d6e7f8a9b0c1
Create Date: 2026-09-28
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = 'e7f8a9b0c1d2'
down_revision = 'd6e7f8a9b0c1'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'embedding_trainings',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('name', sa.String(length=64), nullable=False, unique=True),
        sa.Column('token', sa.String(length=64), nullable=False),
        sa.Column('series_id', postgresql.UUID(as_uuid=True),
                  sa.ForeignKey('image_series.id', ondelete='SET NULL'), nullable=True),
        sa.Column('image_ids', postgresql.JSONB(), nullable=False),
        sa.Column('template', sa.String(length=16), nullable=False),
        sa.Column('init_word', sa.String(length=64), nullable=False),
        sa.Column('vectors', sa.SmallInteger(), nullable=False),
        sa.Column('steps', sa.Integer(), nullable=False),
        sa.Column('learning_rate', sa.Float(), nullable=False),
        sa.Column('status', sa.String(length=16), nullable=False),
        sa.Column('error', sa.Text(), nullable=True),
        sa.Column('chosen_step', sa.Integer(), nullable=True),
        sa.Column('seconds', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('finished_at', sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "status IN ('queued', 'training', 'done', 'failed', 'cancelled')",
            name='ck_embedding_trainings_status',
        ),
        sa.CheckConstraint("template IN ('style', 'object')",
                           name='ck_embedding_trainings_template'),
    )


def downgrade() -> None:
    op.drop_table('embedding_trainings')
