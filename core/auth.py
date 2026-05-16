import logging

from fastapi import HTTPException, Request, WebSocket

from core.config import settings

logger = logging.getLogger(__name__)

AUTH_COOKIE_NAME = "art_rium_auth"
AUTH_COOKIE_MAX_AGE = 60 * 60 * 24 * 30  # 30 days

_LOCAL_PREFIXES = ("127.", "192.168.", "10.")
_172_OCTET_MIN = 16
_172_OCTET_MAX = 31
# Headers a reverse proxy (Cloudflare tunnel) sets — their presence means the
# request originated remotely even though request.client.host shows loopback.
_PROXY_HEADERS = ("cf-connecting-ip", "x-forwarded-for", "x-real-ip")


def is_local_ip(ip: str) -> bool:
    if any(ip.startswith(p) for p in _LOCAL_PREFIXES):
        return True
    if ip.startswith("172."):
        try:
            second = int(ip.split(".")[1])
            return _172_OCTET_MIN <= second <= _172_OCTET_MAX
        except (ValueError, IndexError):
            pass
    return False


def _is_direct_local_request(request: Request) -> bool:
    """True only for genuine local-network access — not a proxied request."""
    ip = request.client.host if request.client else ""
    if not is_local_ip(ip):
        return False
    return not any(h in request.headers for h in _PROXY_HEADERS)


def _extract_key(request: Request) -> str:
    return (
        request.headers.get("X-API-Key")
        or request.query_params.get("api_key")
        or request.cookies.get(AUTH_COOKIE_NAME)
        or ""
    )


def auth_ok(request: Request) -> bool:
    """Direct local access bypasses auth; everything else needs the API key."""
    if _is_direct_local_request(request):
        return True
    if not settings.api_key:
        return True
    return _extract_key(request) == settings.api_key


def require_auth(request: Request) -> None:
    """FastAPI dependency — raises 401 when auth_ok() returns False."""
    if not auth_ok(request):
        raise HTTPException(status_code=401, detail="Invalid API key")


def ws_auth_ok(websocket: WebSocket) -> bool:
    ip = websocket.client.host if websocket.client else ""
    if is_local_ip(ip) and not any(h in websocket.headers for h in _PROXY_HEADERS):
        return True
    if not settings.api_key:
        return True
    key = websocket.query_params.get("api_key", "")
    return key == settings.api_key
