"""
Shared accessors for InstagramPost companion rows (story/reel).

See core/models.py::PostCompanion — one row per companion kind, replacing
the old flat story_*/reel_* columns on InstagramPost (code review A2).
"""
from core.models import InstagramPost, PostCompanion


def find_companion(post: InstagramPost, kind: str) -> PostCompanion | None:
    return next((c for c in post.companions if c.kind == kind), None)


def get_or_create_companion(post: InstagramPost, kind: str) -> PostCompanion:
    """Return the existing companion row, or create+attach a new one.

    Safe to call more than once for the same (post, kind) before a flush —
    `post.companions` is a live Python list, so a second call finds the
    first-created companion already appended rather than creating a dupe.
    """
    existing = find_companion(post, kind)
    if existing is not None:
        return existing
    companion = PostCompanion(post_id=post.id, kind=kind)
    post.companions.append(companion)
    return companion
