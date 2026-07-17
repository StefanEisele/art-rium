"""replace lora_name/lora_strength columns with a loras JSONB list

Revision ID: d4e5f6a7b8c9
Revises: c1d2e3f4a5b6
Create Date: 2026-07-17

Z-Image Turbo can now chain multiple LoRAs (each with its own strength).
lora_name/lora_strength only ever held one, so they're replaced with a
`loras` JSONB column: [{"name": ..., "strength": ...}, ...]. Existing
single-LoRA rows are migrated in place as a one-item list.
"""
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = 'd4e5f6a7b8c9'
down_revision = 'c1d2e3f4a5b6'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('images', sa.Column('loras', JSONB(), nullable=True))
    op.execute("""
        UPDATE images
        SET loras = jsonb_build_array(
            jsonb_build_object('name', lora_name, 'strength', lora_strength)
        )
        WHERE lora_name IS NOT NULL
    """)
    op.drop_column('images', 'lora_strength')
    op.drop_column('images', 'lora_name')


def downgrade() -> None:
    op.add_column('images', sa.Column('lora_name', sa.String(256), nullable=True))
    op.add_column('images', sa.Column('lora_strength', sa.Numeric(4, 3), nullable=True))
    op.execute("""
        UPDATE images
        SET lora_name = loras->0->>'name',
            lora_strength = (loras->0->>'strength')::numeric
        WHERE loras IS NOT NULL AND jsonb_array_length(loras) > 0
    """)
    op.drop_column('images', 'loras')
