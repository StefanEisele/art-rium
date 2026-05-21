"""
YouTube Data API v3 — resumable video upload.

Auth: OAuth 2.0 with a long-lived refresh token (obtained once via
``scripts/youtube_auth.py``). Access tokens last 1h and are refreshed on
demand inside ``upload_video``.

Quota: a single upload costs 1600 units. Default per-project quota is
10,000/day → ~6 uploads/day before you need to request an increase.

Privacy values: 'public' | 'unlisted' | 'private'.

The single-PUT upload path is used (the Resumable endpoint, but uploaded
in one request). YouTube accepts this for any file size we realistically
hand it. If a future Improv mix gets large enough to fail mid-PUT, switch
to chunked PUTs against the same upload URL.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import httpx

from core.config import settings

logger = logging.getLogger(__name__)


_TOKEN_URL  = "https://oauth2.googleapis.com/token"
_UPLOAD_URL = "https://www.googleapis.com/upload/youtube/v3/videos"
_WATCH_URL  = "https://www.youtube.com/watch?v="

_ALLOWED_PRIVACY = ("public", "unlisted", "private")


def missing_config() -> list[str]:
    """Return the names of YT env vars that are unset (empty list = OK)."""
    return [
        k for k, v in {
            "YOUTUBE_CLIENT_ID":     settings.youtube_client_id,
            "YOUTUBE_CLIENT_SECRET": settings.youtube_client_secret,
            "YOUTUBE_REFRESH_TOKEN": settings.youtube_refresh_token,
        }.items() if not v
    ]


async def _exchange_refresh_token() -> str:
    """Trade the long-lived refresh token for a fresh 1h access token."""
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            _TOKEN_URL,
            data={
                "client_id":     settings.youtube_client_id,
                "client_secret": settings.youtube_client_secret,
                "refresh_token": settings.youtube_refresh_token,
                "grant_type":    "refresh_token",
            },
        )
    if r.status_code >= 400:
        raise RuntimeError(
            f"YouTube token refresh failed → {r.status_code}: {r.text[:500]}. "
            "Refresh token may have been revoked — re-run scripts/youtube_auth.py."
        )
    data = r.json()
    token = data.get("access_token")
    if not token:
        raise RuntimeError(f"YouTube token refresh returned no access_token: {data}")
    return token


async def upload_video(
    file_path: Path,
    *,
    title: str,
    description: str,
    privacy: str | None = None,
    tags: list[str] | None = None,
    category_id: str = "22",  # "People & Blogs" — sensible default for art context
) -> dict:
    """
    Upload an MP4 to YouTube and return ``{video_id, url, privacy, uploaded_at}``.

    Raises RuntimeError on any non-2xx response. Caller is responsible for
    idempotency (skip when ``video.youtube_video_id`` is already set).
    """
    if not file_path.exists():
        raise FileNotFoundError(f"YouTube upload source missing: {file_path}")

    missing = missing_config()
    if missing:
        raise RuntimeError(f"YouTube not configured: {missing}")

    privacy = privacy or settings.youtube_privacy_default
    if privacy not in _ALLOWED_PRIVACY:
        raise ValueError(f"privacy must be one of {_ALLOWED_PRIVACY}, got {privacy!r}")

    access_token = await _exchange_refresh_token()

    # YouTube enforces a 100-char title cap and a 5000-char description cap.
    # Truncate defensively so the API doesn't reject the whole upload.
    title = (title or "Untitled").strip()[:100]
    description = (description or "").strip()[:5000]

    body = file_path.read_bytes()
    metadata = {
        "snippet": {
            "title":       title,
            "description": description,
            "tags":        tags or [],
            "categoryId":  category_id,
        },
        "status": {
            "privacyStatus":           privacy,
            "selfDeclaredMadeForKids": False,
        },
    }

    async with httpx.AsyncClient(timeout=None) as client:
        # Step 1 — initiate resumable upload.
        init_resp = await client.post(
            _UPLOAD_URL,
            params={"uploadType": "resumable", "part": "snippet,status"},
            headers={
                "Authorization":             f"Bearer {access_token}",
                "Content-Type":              "application/json; charset=UTF-8",
                "X-Upload-Content-Type":     "video/mp4",
                "X-Upload-Content-Length":   str(len(body)),
            },
            json=metadata,
        )
        if init_resp.status_code >= 400:
            raise RuntimeError(
                f"YouTube upload init → {init_resp.status_code}: {init_resp.text[:500]}"
            )
        upload_url = init_resp.headers.get("Location")
        if not upload_url:
            raise RuntimeError("YouTube upload init returned no Location header")

        # Step 2 — single PUT of the whole file. Resumable in spirit (we have
        # the URL and could chunk), but YouTube is happy to take it in one go
        # for the file sizes we deal with.
        put_resp = await client.put(
            upload_url,
            content=body,
            headers={
                "Content-Type":   "video/mp4",
                "Content-Length": str(len(body)),
            },
        )
        if put_resp.status_code >= 400:
            raise RuntimeError(
                f"YouTube upload PUT → {put_resp.status_code}: {put_resp.text[:500]}"
            )

    payload = put_resp.json()
    video_id = payload.get("id")
    if not video_id:
        raise RuntimeError(f"YouTube upload succeeded but response has no id: {payload}")

    logger.info("YouTube upload OK: %s (%s, %d bytes)", video_id, privacy, len(body))
    return {
        "video_id":    video_id,
        "url":         _WATCH_URL + video_id,
        "privacy":     privacy,
        "uploaded_at": datetime.now(timezone.utc),
    }


async def reachable() -> bool:
    """Best-effort check that the OAuth credentials still work. Used by /status."""
    if missing_config():
        return False
    try:
        await _exchange_refresh_token()
        return True
    except Exception as exc:
        logger.warning("YouTube reachable() failed: %s", exc)
        return False
