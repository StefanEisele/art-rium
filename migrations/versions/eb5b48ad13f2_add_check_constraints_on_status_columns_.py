"""add CHECK constraints on status columns and instagram_post_media

Revision ID: eb5b48ad13f2
Revises: 0359b4bf9dff
Create Date: 2026-07-05

Hand-written: alembic's autogenerate does not diff CheckConstraint objects,
so this doesn't come from `alembic revision --autogenerate`. Value sets were
audited against actual code paths (not just column comments, which had
drifted — e.g. InstagramPost.status is missing "failed" and reel_status is
missing "remote_scheduled" in the pre-existing comments) and cross-checked
against the live DB's current distinct values before writing this migration.

story_status / outpost_status / outpost_reel_status on instagram_posts are
deliberately NOT constrained — they're written verbatim from the Pi
outpost's own /status response, a vocabulary this repo doesn't own.
"""
from alembic import op

revision = 'eb5b48ad13f2'
down_revision = '0359b4bf9dff'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_check_constraint(
        "ck_articles_status", "articles", "status IN ('draft', 'published', 'failed')"
    )
    op.create_check_constraint(
        "ck_shop_listings_status", "shop_listings",
        "status IN ('draft', 'ready', 'submitted', 'live')",
    )
    op.create_check_constraint(
        "ck_instagram_posts_status", "instagram_posts",
        "status IN ('scheduled', 'posted', 'cancelled', 'failed')",
    )
    op.create_check_constraint(
        "ck_instagram_posts_reel_status", "instagram_posts",
        "reel_status IS NULL OR reel_status IN "
        "('pending', 'processing', 'posted', 'failed', 'remote_scheduled')",
    )
    op.create_check_constraint(
        "ck_ig_post_media_exactly_one_ref", "instagram_post_media",
        "(image_id IS NOT NULL) != (video_id IS NOT NULL)",
    )
    op.create_check_constraint(
        "ck_videos_status", "videos",
        "status IN ('generating', 'review', 'assembling', 'done', 'failed')",
    )
    op.create_check_constraint(
        "ck_songs_status", "songs", "status IN ('generating', 'done', 'failed')"
    )
    op.create_check_constraint(
        "ck_improv_sessions_status", "improv_sessions",
        "status IN ('queued', 'processing', 'done', 'failed')",
    )


def downgrade() -> None:
    op.drop_constraint("ck_improv_sessions_status", "improv_sessions", type_="check")
    op.drop_constraint("ck_songs_status", "songs", type_="check")
    op.drop_constraint("ck_videos_status", "videos", type_="check")
    op.drop_constraint("ck_ig_post_media_exactly_one_ref", "instagram_post_media", type_="check")
    op.drop_constraint("ck_instagram_posts_reel_status", "instagram_posts", type_="check")
    op.drop_constraint("ck_instagram_posts_status", "instagram_posts", type_="check")
    op.drop_constraint("ck_shop_listings_status", "shop_listings", type_="check")
    op.drop_constraint("ck_articles_status", "articles", type_="check")
