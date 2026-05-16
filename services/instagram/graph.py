"""
Shared Instagram Graph API helpers — Single Source of Truth.

Used by:
  - routers/instagram.py        (synchronous /post-now endpoint)
  - workers/instagram_scheduler.py  (background auto-poster)
  - workers/instagram_companion.py  (Stories + Reel companions)

Centralises:
  - Public share URL construction (image / reel / video) with token query
  - Graph API error checking (raises RuntimeError on `error` payloads)
  - Container readiness polling (FINISHED / ERROR / timeout)
  - Configuration completeness check (missing env vars list)
"""
import asyncio
import logging
from typing import Literal

import httpx

from core.config import settings

logger = logging.getLogger(__name__)

# Default polling parameters for media containers (image / carousel)
CONTAINER_POLL_INTERVAL = 3
CONTAINER_POLL_TIMEOUT = 60

# Reel containers take longer (video upload + transcode)
REEL_POLL_INTERVAL = 5
REEL_POLL_TIMEOUT = 300


ShareKind = Literal["image", "reel", "video", "video-loop"]


# ── URL construction ──────────────────────────────────────────────────────────

def share_url(filename: str, kind: ShareKind = "image") -> str:
    """
    Build a public share URL that the Graph API can fetch.

    `kind` selects the endpoint prefix:
      image       → /share/image/<file>      PNG from storage/images or comfy output
      reel        → /share/reel/<file>       temp slideshow MP4 in storage/reels
      video       → /share/video/<file>      curated video in storage/videos
      video-loop  → /share/video-loop/<file> tiny HTML page with a looping video
                                             (Improv tool's QR-code target)

    The IMAGE_SHARE_TOKEN is appended as ?token=… when configured.
    """
    base = settings.public_base_url.rstrip("/")
    url = f"{base}/share/{kind}/{filename}"
    if settings.image_share_token:
        url += f"?token={settings.image_share_token}"
    return url


# ── Error handling ────────────────────────────────────────────────────────────

def check_response(body: dict, context: str) -> None:
    """Raise RuntimeError if Graph API response contains an `error` field."""
    if "error" in body:
        err = body["error"]
        raise RuntimeError(f"{context}: {err.get('message', body)}")


# ── Configuration check ──────────────────────────────────────────────────────

def missing_config() -> list[str]:
    """Return the names of required env vars that are unset (empty list = OK)."""
    return [
        k for k, v in {
            "INSTAGRAM_USER_ID":      settings.instagram_user_id,
            "INSTAGRAM_ACCESS_TOKEN": settings.instagram_access_token,
            "PUBLIC_BASE_URL":        settings.public_base_url,
        }.items() if not v
    ]


# ── Container polling ────────────────────────────────────────────────────────

async def wait_container_ready(
    client: httpx.AsyncClient,
    container_id: str,
    *,
    max_wait: int = CONTAINER_POLL_TIMEOUT,
    poll_interval: int = CONTAINER_POLL_INTERVAL,
) -> None:
    """
    Poll the Graph API until the media container reaches FINISHED status.

    Raises:
      RuntimeError on container ERROR state
      TimeoutError if not FINISHED within max_wait seconds
    """
    graph = settings.instagram_graph_api_base
    token = settings.instagram_access_token

    deadline = asyncio.get_event_loop().time() + max_wait
    while True:
        r = await client.get(
            f"{graph}/{container_id}",
            params={"fields": "status_code", "access_token": token},
        )
        body = r.json()
        status = body.get("status_code", "")
        logger.debug("Container %s status: %s", container_id, status)

        if status == "FINISHED":
            return
        if status == "ERROR":
            raise RuntimeError(
                f"Instagram container {container_id} failed processing: {body}"
            )
        if asyncio.get_event_loop().time() >= deadline:
            raise TimeoutError(
                f"Instagram container {container_id} not ready after "
                f"{max_wait}s (status={status!r})"
            )
        await asyncio.sleep(poll_interval)
