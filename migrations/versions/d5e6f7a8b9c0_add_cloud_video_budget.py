"""add cloud video budget ledger and API job columns

Revision ID: d5e6f7a8b9c0
Revises: c4d5e6f7a8b9
Create Date: 2026-08-15

Two-phase budget booking for MiniMax H3 cloud renders (services/video_api/):
a per-month limit and a ledger of reservations, plus the columns a Video row
needs when it is rendered remotely instead of on the local GPU.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = 'd5e6f7a8b9c0'
down_revision = 'c4d5e6f7a8b9'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'budget_periods',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('month', sa.String(length=7), nullable=False, unique=True),
        sa.Column('limit_eur', sa.Numeric(10, 2), nullable=False),
        sa.Column('warn_threshold_pct', sa.SmallInteger(), nullable=False, server_default='80'),
        sa.Column('usd_eur_rate', sa.Numeric(8, 4), nullable=False),
        sa.Column('rate_updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint('limit_eur >= 0', name='ck_budget_periods_limit_positive'),
        sa.CheckConstraint(
            'warn_threshold_pct BETWEEN 1 AND 100', name='ck_budget_periods_warn_pct'
        ),
    )

    op.create_table(
        'ledger_entries',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('period_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('video_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('task_id', sa.String(length=128), nullable=True),
        sa.Column('idempotency_key', sa.String(length=64), nullable=False, unique=True),
        sa.Column('kind', sa.String(length=24), nullable=False),
        sa.Column('duration_s', sa.SmallInteger(), nullable=False),
        sa.Column('ref_image_count', sa.SmallInteger(), nullable=False, server_default='0'),
        sa.Column('input_video_seconds', sa.Numeric(6, 2), nullable=False, server_default='0'),
        sa.Column('amount_usd', sa.Numeric(10, 4), nullable=False),
        sa.Column('amount_eur', sa.Numeric(10, 2), nullable=False),
        sa.Column('usd_eur_rate', sa.Numeric(8, 4), nullable=False),
        sa.Column('state', sa.String(length=16), nullable=False, server_default='reserved'),
        sa.Column('note', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('settled_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['period_id'], ['budget_periods.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['video_id'], ['videos.id'], ondelete='SET NULL'),
        sa.CheckConstraint(
            "state IN ('reserved', 'settled', 'released')", name='ck_ledger_entries_state'
        ),
        sa.CheckConstraint(
            "kind IN ('generate_768p', 'generate_2k', 'regenerate_2k')",
            name='ck_ledger_entries_kind',
        ),
    )
    op.create_index('ix_ledger_entries_period_state', 'ledger_entries', ['period_id', 'state'])
    op.create_index('ix_ledger_entries_video_id', 'ledger_entries', ['video_id'])
    op.create_index('ix_ledger_entries_state', 'ledger_entries', ['state'])
    op.create_index('ix_ledger_entries_created_at', 'ledger_entries', ['created_at'])

    # ── Video: the cloud-render columns ──────────────────────────────────────
    op.add_column('videos', sa.Column('api_task_id', sa.String(length=128), nullable=True))
    op.add_column('videos', sa.Column('api_resolution', sa.String(length=8), nullable=True))
    op.add_column('videos', sa.Column('api_ratio', sa.String(length=16), nullable=True))
    op.add_column('videos', sa.Column('duration_s', sa.SmallInteger(), nullable=True))
    op.add_column('videos', sa.Column('source_video_id', postgresql.UUID(as_uuid=True), nullable=True))
    op.create_foreign_key(
        'fk_videos_source_video_id', 'videos', 'videos', ['source_video_id'], ['id'],
        ondelete='SET NULL',
    )
    op.create_index('ix_videos_api_task_id', 'videos', ['api_task_id'])
    op.create_index('ix_videos_source_video_id', 'videos', ['source_video_id'])

    # A cloud job waits in a local FIFO before it is submitted, which is a
    # state the local pipeline never had.
    op.drop_constraint('ck_videos_status', 'videos', type_='check')
    op.create_check_constraint(
        'ck_videos_status', 'videos',
        "status IN ('queued', 'generating', 'review', 'assembling', 'done', 'failed')",
    )


def downgrade() -> None:
    op.drop_constraint('ck_videos_status', 'videos', type_='check')
    op.create_check_constraint(
        'ck_videos_status', 'videos',
        "status IN ('generating', 'review', 'assembling', 'done', 'failed')",
    )
    op.drop_index('ix_videos_source_video_id', table_name='videos')
    op.drop_index('ix_videos_api_task_id', table_name='videos')
    op.drop_constraint('fk_videos_source_video_id', 'videos', type_='foreignkey')
    for col in ('source_video_id', 'duration_s', 'api_ratio', 'api_resolution', 'api_task_id'):
        op.drop_column('videos', col)

    op.drop_table('ledger_entries')
    op.drop_table('budget_periods')
