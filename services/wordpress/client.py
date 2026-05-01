"""
WordPress REST API client — base auth, URL, and request helpers.

Used by:
  - services/wordpress/media.py
  - (later) article-publishing code

Auth: WordPress Application Password (Users → Profile → Application Passwords).
Username + app password are sent via HTTP Basic auth.
"""
import logging

import httpx

from core.config import settings

logger = logging.getLogger(__name__)


def missing_config() -> list[str]:
    """Return the names of WP env vars that are unset (empty list = OK)."""
    return [
        k for k, v in {
            "WP_BASE_URL":     settings.wp_base_url,
            "WP_USERNAME":     settings.wp_username,
            "WP_APP_PASSWORD": settings.wp_app_password,
        }.items() if not v
    ]


def _auth() -> tuple[str, str]:
    return (settings.wp_username, settings.wp_app_password)


def _url(path: str) -> str:
    base = settings.wp_base_url.rstrip("/")
    if not path.startswith("/"):
        path = "/" + path
    return f"{base}/wp-json{path}"


async def reachable() -> bool:
    """Best-effort: check that /wp-json/ responds."""
    try:
        async with httpx.AsyncClient(timeout=4) as client:
            r = await client.get(_url("/"))
            return r.status_code == 200
    except Exception:
        return False


async def request_json(
    method: str,
    path: str,
    *,
    json: dict | None = None,
    params: dict | None = None,
    timeout: float = 30.0,
) -> dict:
    """JSON request helper. Raises RuntimeError on non-2xx with body excerpt."""
    async with httpx.AsyncClient(timeout=timeout, auth=_auth()) as client:
        r = await client.request(method, _url(path), json=json, params=params)
    if r.status_code >= 400:
        raise RuntimeError(f"WP {method} {path} → {r.status_code}: {r.text[:500]}")
    return r.json()


async def upload_binary(
    path: str,
    *,
    body: bytes,
    filename: str,
    content_type: str,
    params: dict | None = None,
    timeout: float = 180.0,
) -> dict:
    """Binary upload (e.g. /wp/v2/media). Sends Content-Disposition for filename."""
    headers = {
        "Content-Type":        content_type,
        "Content-Disposition": f'attachment; filename="{filename}"',
    }
    async with httpx.AsyncClient(timeout=timeout, auth=_auth()) as client:
        r = await client.post(_url(path), content=body, headers=headers, params=params)
    if r.status_code >= 400:
        raise RuntimeError(f"WP POST {path} → {r.status_code}: {r.text[:500]}")
    return r.json()
