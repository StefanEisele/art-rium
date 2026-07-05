import hashlib
import hmac
import logging
import secrets
import time

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


def verify_api_key(key: str | None) -> bool:
    """Constant-time check against the configured bearer key."""
    return bool(key) and bool(settings.api_key) and secrets.compare_digest(key, settings.api_key)


# ── Session tokens ────────────────────────────────────────────────────────────
#
# The auth cookie used to hold the raw API key verbatim (main.py, pre-2026-07),
# so any log/leak of the cookie was a full credential leak and the key
# couldn't be rotated without invalidating it. Instead, the cookie now holds
# a derived, time-limited, HMAC-signed token that proves the browser saw the
# real key at some point without the cookie itself being one — leaking the
# cookie only leaks a token, not the bearer credential used by the header/
# query-param paths (X-API-Key, ?api_key= for <img>/<video> src).

def _session_secret() -> bytes:
    return hashlib.sha256(f"art-rium-session-v1:{settings.api_key}".encode()).digest()


def create_session_token() -> str:
    issued_at = str(int(time.time()))
    sig = hmac.new(_session_secret(), issued_at.encode(), hashlib.sha256).hexdigest()
    return f"{issued_at}.{sig}"


def verify_session_token(token: str | None) -> bool:
    if not token or not settings.api_key:
        return False
    issued_at, _, sig = token.partition(".")
    if not sig:
        return False
    try:
        age = time.time() - int(issued_at)
    except ValueError:
        return False
    if age < 0 or age > AUTH_COOKIE_MAX_AGE:
        return False
    expected = hmac.new(_session_secret(), issued_at.encode(), hashlib.sha256).hexdigest()
    return secrets.compare_digest(sig, expected)


def _extract_bearer_key(request: Request) -> str | None:
    """Raw-key sources only — NOT the session cookie, which is a derived token."""
    return request.headers.get("X-API-Key") or request.query_params.get("api_key")


def auth_ok(request: Request) -> bool:
    """Direct local access bypasses auth; everything else needs the API key
    (header/query-param) or a valid session cookie."""
    if _is_direct_local_request(request):
        return True
    if not settings.api_key:
        return True
    if verify_api_key(_extract_bearer_key(request)):
        return True
    return verify_session_token(request.cookies.get(AUTH_COOKIE_NAME))


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
    # Browsers send cookies on the WS handshake same as any same-origin
    # request, so the session cookie set at login covers this — no need for
    # the API key to also travel in the WS URL's query string.
    return verify_session_token(websocket.cookies.get(AUTH_COOKIE_NAME))
