import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from core.config import settings
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

    # Start ComfyUI WebSocket listener
    listener = ComfyListener(app.state)
    comfy_task = asyncio.create_task(listener.run())
    app.state.comfy_listener = listener

    # Start Instagram auto-poster
    scheduler = InstagramScheduler()
    scheduler_task = asyncio.create_task(scheduler.run())

    # Warm the titler VLM in the background — cold load is ~2.5 min, which
    # exceeds the Cloudflare tunnel's ~100s upstream timeout for the first
    # frontend request. Fire-and-forget; never blocks server startup.
    warm_task = asyncio.create_task(warm_titler_model())
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


# ── Routers ──────────────────────────────────────────────────────────────────
from routers import generate, images, titler, instagram, video, wordpress, system  # noqa: E402  (after app is created)

app.include_router(generate.router)
app.include_router(images.router)
app.include_router(titler.router)
app.include_router(instagram.router)
app.include_router(video.router)
app.include_router(wordpress.router)
app.include_router(system.router)

# ── Static frontends ──────────────────────────────────────────────────────────
# Shared assets (CSS / JS) must be mounted before tool and root catch-alls
_shared = Path(__file__).parent / "frontends" / "shared"
if _shared.exists():
    app.mount("/shared", StaticFiles(directory=str(_shared)), name="shared")

# Tools must be mounted before the root catch-all
_z_image = Path(__file__).parent / "frontends" / "tools" / "z-image"
if _z_image.exists():
    app.mount("/tools/z-image", StaticFiles(directory=str(_z_image), html=True), name="z-image")

_gallery = Path(__file__).parent / "frontends" / "tools" / "gallery"
if _gallery.exists():
    app.mount("/tools/gallery", StaticFiles(directory=str(_gallery), html=True), name="gallery")

_titler = Path(__file__).parent / "frontends" / "tools" / "titler"
if _titler.exists():
    app.mount("/tools/titler", StaticFiles(directory=str(_titler), html=True), name="titler")

_instagram = Path(__file__).parent / "frontends" / "tools" / "instagram"
if _instagram.exists():
    app.mount("/tools/instagram", StaticFiles(directory=str(_instagram), html=True), name="instagram")

_video = Path(__file__).parent / "frontends" / "tools" / "video"
if _video.exists():
    app.mount("/tools/video", StaticFiles(directory=str(_video), html=True), name="video")

_articles = Path(__file__).parent / "frontends" / "tools" / "articles"
if _articles.exists():
    app.mount("/tools/articles", StaticFiles(directory=str(_articles), html=True), name="articles")

_dashboard = Path(__file__).parent / "frontends" / "dashboard"
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
