import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from core.auth import (
    AUTH_COOKIE_NAME,
    AUTH_COOKIE_MAX_AGE,
    auth_ok,
    create_session_token,
    verify_api_key,
    verify_session_token,
)
from core.config import settings
from core.startup_sweep import sweep_stuck_jobs
from core.tasks import safe_create_task
from services.ollama.client import warm_titler_model
from workers.comfy_listener import ComfyListener
from workers.instagram_scheduler import InstagramScheduler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _quiet_exception_handler(loop, context):
    exc = context.get("exception")
    if isinstance(exc, (ConnectionResetError, BrokenPipeError)):
        return
    loop.default_exception_handler(context)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if sys.platform == "win32":
        asyncio.get_running_loop().set_exception_handler(_quiet_exception_handler)

    # Ensure storage dirs exist
    settings.images_dir.mkdir(parents=True, exist_ok=True)
    settings.shop_prep_dir.mkdir(parents=True, exist_ok=True)
    settings.videos_dir.mkdir(parents=True, exist_ok=True)
    settings.improv_dir.mkdir(parents=True, exist_ok=True)
    settings.songs_dir.mkdir(parents=True, exist_ok=True)

    # Mark jobs orphaned by the previous process's death as failed before
    # anything else starts polling/dispatching.
    await sweep_stuck_jobs()

    # Start ComfyUI WebSocket listener
    listener = ComfyListener(app.state)
    comfy_task = safe_create_task(listener.run(), name="comfy_listener")
    app.state.comfy_listener = listener

    # Start Instagram auto-poster
    scheduler = InstagramScheduler()
    scheduler_task = safe_create_task(scheduler.run(), name="instagram_scheduler")

    # Warm the titler VLM in the background — cold load is ~2.5 min, which
    # exceeds the Cloudflare tunnel's ~100s upstream timeout for the first
    # frontend request. Fire-and-forget; never blocks server startup.
    warm_task = safe_create_task(warm_titler_model(), name="titler_warmup")
    logger.info("Titler warm-up scheduled in background")

    yield

    comfy_task.cancel()
    scheduler_task.cancel()
    warm_task.cancel()
    for t in (comfy_task, scheduler_task, warm_task):
        try:
            await t
        except asyncio.CancelledError:
            pass


app = FastAPI(title="art-rium", lifespan=lifespan)


@app.middleware("http")
async def no_cache_html(request, call_next):
    response = await call_next(request)
    if response.headers.get("content-type", "").startswith("text/html"):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


# Always-public path prefixes — Meta needs /share/* to fetch media, and the
# login stub is itself public so users can submit their API key.
_PUBLIC_PREFIXES = ("/share/",)

_LOGIN_HTML = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>art-rium</title>
<style>
  html,body{height:100%;margin:0}
  body{font-family:-apple-system,BlinkMacSystemFont,system-ui,sans-serif;background:#0f0f10;color:#eee;display:flex;align-items:center;justify-content:center;padding:24px;box-sizing:border-box}
  .card{max-width:340px;width:100%;background:#1a1a1c;border-radius:14px;padding:28px 24px;box-shadow:0 4px 24px rgba(0,0,0,.4)}
  h1{margin:0 0 4px;font-size:22px;font-weight:600}
  p{margin:0 0 20px;color:#888;font-size:14px}
  label{display:block;font-size:12px;color:#888;margin-bottom:6px}
  input{width:100%;box-sizing:border-box;padding:10px 12px;border-radius:8px;border:1px solid #333;background:#0f0f10;color:#eee;font-size:14px}
  input:focus{outline:none;border-color:#666}
  button{width:100%;margin-top:14px;padding:11px;border:none;border-radius:8px;background:#fff;color:#000;font-size:14px;font-weight:600;cursor:pointer}
  .err{color:#e85;font-size:13px;margin-top:12px;text-align:center}
</style>
</head><body>
<form class="card" method="get" action="/" onsubmit="try{localStorage.setItem('z_apikey',document.getElementById('k').value)}catch(e){}">
  <h1>art-rium</h1>
  <p>Enter your API key to continue.</p>
  <label>API Key</label>
  <input id="k" name="api_key" type="password" autofocus required autocomplete="off" autocorrect="off" spellcheck="false">
  <button type="submit">Unlock</button>
  __ERROR__
</form>
</body></html>"""


def _login_response(error: bool) -> HTMLResponse:
    body = _LOGIN_HTML.replace(
        "__ERROR__",
        '<div class="err">Invalid key — try again.</div>' if error else "",
    )
    return HTMLResponse(body, status_code=401 if error else 200)


@app.middleware("http")
async def gate_frontend(request: Request, call_next):
    """
    Block every static frontend / API request until the user has supplied the
    API key (via header, ?api_key=, or persisted cookie). /share/* stays open
    so Meta can fetch image/reel URLs.
    """
    path = request.url.path
    if any(path.startswith(p) for p in _PUBLIC_PREFIXES):
        return await call_next(request)

    if auth_ok(request):
        # Fresh GET with ?api_key= on a page load → set cookie + bounce to a
        # clean URL so the key doesn't linger in browser history. Skips the
        # wasted static-file lookup for the dirty URL.
        if (
            request.method == "GET"
            and request.query_params.get("api_key")
            and "text/html" in request.headers.get("accept", "")
        ):
            clean_qs = "&".join(
                f"{k}={v}" for k, v in request.query_params.multi_items() if k != "api_key"
            )
            target = path + (f"?{clean_qs}" if clean_qs else "")
            redirect = RedirectResponse(target, status_code=303)
            redirect.set_cookie(
                AUTH_COOKIE_NAME,
                create_session_token(),
                max_age=AUTH_COOKIE_MAX_AGE,
                httponly=True,
                secure=request.url.scheme == "https",
                samesite="lax",
            )
            return redirect

        response = await call_next(request)
        # Promote a freshly-supplied header key to a session cookie too (handy
        # when an external tool calls the API with X-API-Key from a browser
        # context). Skip if a valid session already exists, so we don't churn
        # a fresh token (and reset its max-age) on every single request.
        header_key = request.headers.get("X-API-Key")
        if verify_api_key(header_key) and not verify_session_token(
            request.cookies.get(AUTH_COOKIE_NAME)
        ):
            response.set_cookie(
                AUTH_COOKIE_NAME,
                create_session_token(),
                max_age=AUTH_COOKIE_MAX_AGE,
                httponly=True,
                secure=request.url.scheme == "https",
                samesite="lax",
            )
        return response

    accepts_html = "text/html" in request.headers.get("accept", "")
    if request.method == "GET" and accepts_html:
        # A query-param key was supplied but didn't match → show "invalid" hint.
        return _login_response(error=request.query_params.get("api_key") is not None)
    return JSONResponse({"detail": "Auth required"}, status_code=401)


# ── Routers ──────────────────────────────────────────────────────────────────
from routers import generate, images, titler, instagram, video, wordpress, system, improv, music  # noqa: E402  (after app is created)

app.include_router(generate.router)
app.include_router(images.router)
app.include_router(titler.router)
app.include_router(instagram.router)
app.include_router(video.router)
app.include_router(wordpress.router)
app.include_router(system.router)
app.include_router(improv.router)
app.include_router(music.router)

# ── Static frontends ──────────────────────────────────────────────────────────
# Mount order matters: /shared and /tools/* must precede the / catch-all.
_frontends_dir = Path(__file__).parent / "frontends"

_shared = _frontends_dir / "shared"
if _shared.exists():
    app.mount("/shared", StaticFiles(directory=str(_shared)), name="shared")

_TOOL_NAMES = ("z-image", "gallery", "titler", "instagram", "video", "articles", "improv", "music")
for _tool in _TOOL_NAMES:
    _dir = _frontends_dir / "tools" / _tool
    if _dir.exists():
        app.mount(f"/tools/{_tool}", StaticFiles(directory=str(_dir), html=True), name=_tool)

_dashboard = _frontends_dir / "dashboard"
if _dashboard.exists():
    app.mount("/", StaticFiles(directory=str(_dashboard), html=True), name="dashboard")

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cert_dir = Path(__file__).parent / "certs"
    ssl_key = cert_dir / "key.pem"
    ssl_cert = cert_dir / "cert.pem"
    use_ssl = ssl_key.exists() and ssl_cert.exists() and "--http" not in sys.argv

    logger.info(f"Starting {'HTTPS' if use_ssl else 'HTTP'} on port {settings.port}")

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.port,
        ssl_keyfile=str(ssl_key) if use_ssl else None,
        ssl_certfile=str(ssl_cert) if use_ssl else None,
        log_level="info",
    )
