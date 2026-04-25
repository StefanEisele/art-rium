"""
Scheduling helpers — Single Source of Truth for companion-post timing.

Used by both the Instagram router (post-now) and the InstagramScheduler worker
to compute when Stories/Reels should be published relative to a feed post.
"""
from datetime import datetime, timedelta


def companion_at(
    published_at: datetime,
    delay_minutes: int,
    companion_time: str | None,
) -> datetime:
    """
    Compute a companion post's scheduled_at relative to the feed publication.

    For day+ delays (>= 24 h) the result snaps to the HH:MM target time so
    multi-day campaigns post at a predictable hour rather than drifting.
    """
    dt = published_at + timedelta(minutes=delay_minutes)
    if delay_minutes >= 1440 and companion_time:
        try:
            h, m = map(int, companion_time.split(":"))
            dt = dt.replace(hour=h, minute=m, second=0, microsecond=0)
        except (ValueError, AttributeError):
            pass
    return dt
