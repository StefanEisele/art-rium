import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from core.config import settings
from workers.comfy_listener import ComfyListener

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

    # Start ComfyUI WebSocket listener
    listener = ComfyListener(app.state)
    task = asyncio.create_task(listener.run())
    app.state.comfy_listener = listener

    yield

    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


app = FastAPI(title="art-rium", lifespan=lifespan)

# ── Routers ──────────────────────────────────────────────────────────────────
from routers import generate, images  # noqa: E402  (after app is created)

app.include_router(generate.router)
app.include_router(images.router)

# ── Static frontends ──────────────────────────────────────────────────────────
# Tools must be mounted before the root catch-all
_z_image = Path(__file__).parent / "frontends" / "tools" / "z-image"
if _z_image.exists():
    app.mount("/tools/z-image", StaticFiles(directory=str(_z_image), html=True), name="z-image")

_gallery = Path(__file__).parent / "frontends" / "tools" / "gallery"
if _gallery.exists():
    app.mount("/tools/gallery", StaticFiles(directory=str(_gallery), html=True), name="gallery")

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
