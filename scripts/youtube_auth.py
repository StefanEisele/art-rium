"""
One-shot OAuth bootstrap for the YouTube uploader.

Run this once after setting YOUTUBE_CLIENT_ID and YOUTUBE_CLIENT_SECRET in
.env:

    python scripts/youtube_auth.py

A browser tab opens, you grant the youtube.upload scope, and the script
writes (or updates) YOUTUBE_REFRESH_TOKEN in .env.

The refresh token survives indefinitely as long as the OAuth consent
screen in Google Cloud Console is set to "In production" (NOT "Testing"
— testing-mode tokens expire after 7 days).

No external dependencies — uses stdlib's http.server for the loopback
redirect.
"""
from __future__ import annotations

import json
import secrets
import sys
import urllib.parse
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT / ".env"

AUTH_URL  = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
SCOPE     = "https://www.googleapis.com/auth/youtube.upload"
PORT      = 8765   # loopback redirect port — must match the URI we send to Google


def _read_env() -> dict[str, str]:
    if not ENV_PATH.exists():
        sys.exit(f".env not found at {ENV_PATH}. Copy .env.example to .env first.")
    out: dict[str, str] = {}
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        out[k.strip()] = v.strip()
    return out


def _write_env_value(key: str, value: str) -> None:
    """Set or update KEY=VALUE in .env, preserving everything else."""
    lines = ENV_PATH.read_text(encoding="utf-8").splitlines()
    replaced = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(f"{key}=") or stripped.startswith(f"{key} ="):
            lines[i] = f"{key}={value}"
            replaced = True
            break
    if not replaced:
        if lines and lines[-1].strip():
            lines.append("")
        lines.append(f"{key}={value}")
    ENV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


class _RedirectHandler(BaseHTTPRequestHandler):
    captured: dict[str, str] = {}

    def log_message(self, format: str, *args) -> None:  # silence stdout noise
        return

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        params = dict(urllib.parse.parse_qsl(parsed.query))
        _RedirectHandler.captured.update(params)

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        msg = "OK — you can close this tab." if "code" in params else f"Auth failed: {params}"
        self.wfile.write(
            f"<html><body style='font-family:sans-serif;padding:40px;color:#222'>"
            f"<h2>{msg}</h2></body></html>".encode("utf-8")
        )


def _wait_for_code(state: str) -> str:
    server = HTTPServer(("127.0.0.1", PORT), _RedirectHandler)
    print(f"Waiting for Google redirect on http://127.0.0.1:{PORT} …")
    while True:
        server.handle_request()
        cap = _RedirectHandler.captured
        if "error" in cap:
            sys.exit(f"Google returned an error: {cap['error']} ({cap.get('error_description','')})")
        if cap.get("state") and cap["state"] != state:
            sys.exit(f"State mismatch (possible CSRF). got {cap.get('state')!r}, expected {state!r}")
        if "code" in cap:
            return cap["code"]


def _exchange_code(code: str, client_id: str, client_secret: str, redirect_uri: str) -> dict:
    data = urllib.parse.urlencode({
        "code":          code,
        "client_id":     client_id,
        "client_secret": client_secret,
        "redirect_uri":  redirect_uri,
        "grant_type":    "authorization_code",
    }).encode("utf-8")
    req = urllib.request.Request(
        TOKEN_URL,
        data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> None:
    env = _read_env()
    client_id     = env.get("YOUTUBE_CLIENT_ID", "")
    client_secret = env.get("YOUTUBE_CLIENT_SECRET", "")
    if not client_id or not client_secret:
        sys.exit(
            "YOUTUBE_CLIENT_ID and YOUTUBE_CLIENT_SECRET must be set in .env first.\n"
            "Get them from Google Cloud Console → APIs & Services → Credentials → "
            "OAuth 2.0 Client IDs (Desktop app)."
        )

    redirect_uri = f"http://127.0.0.1:{PORT}"
    state = secrets.token_urlsafe(16)
    auth_params = urllib.parse.urlencode({
        "client_id":     client_id,
        "redirect_uri":  redirect_uri,
        "response_type": "code",
        "scope":         SCOPE,
        "access_type":   "offline",
        "prompt":        "consent",   # force a fresh refresh_token even if previously granted
        "state":         state,
    })
    url = f"{AUTH_URL}?{auth_params}"
    print("Opening browser for Google consent …")
    print(f"If nothing opens, paste this URL manually:\n  {url}")
    webbrowser.open(url)

    code = _wait_for_code(state)
    print("Got auth code, exchanging for refresh token …")
    tokens = _exchange_code(code, client_id, client_secret, redirect_uri)

    refresh = tokens.get("refresh_token")
    if not refresh:
        sys.exit(
            f"No refresh_token in response. Google only returns one on the FIRST consent.\n"
            f"Revoke access at https://myaccount.google.com/permissions and retry.\n"
            f"Response: {tokens}"
        )

    _write_env_value("YOUTUBE_REFRESH_TOKEN", refresh)
    print(f"\nRefresh token written to {ENV_PATH}")
    print("Done — the uploader is now operational.")


if __name__ == "__main__":
    main()
