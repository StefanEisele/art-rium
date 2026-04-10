import logging

from fastapi import Request, WebSocket

from core.config import settings

logger = logging.getLogger(__name__)

_LOCAL_PREFIXES = ("127.", "192.168.", "10.")
_172_RANGE = (16, 31)


def is_local_ip(ip: str) -> bool:
    if any(ip.startswith(p) for p in _LOCAL_PREFIXES):
        return True
    if ip.startswith("172."):
        try:
            second = int(ip.split(".")[1])
            return _172_RANGE[0] <= second <= _172_RANGE[1]
        except (ValueError, IndexError):
            pass
    return False


def auth_ok(request: Request) -> bool:
    """Local IPs bypass auth; remote requests require X-API-Key or ?api_key=."""
    ip = request.client.host if request.client else ""
    if is_local_ip(ip):
        return True
    if not settings.api_key:
        return True
    key = request.headers.get("X-API-Key") or request.query_params.get("api_key", "")
    return key == settings.api_key


def ws_auth_ok(websocket: WebSocket) -> bool:
    ip = websocket.client.host if websocket.client else ""
    if is_local_ip(ip):
        return True
    if not settings.api_key:
        return True
    key = websocket.query_params.get("api_key", "")
    return key == settings.api_key
