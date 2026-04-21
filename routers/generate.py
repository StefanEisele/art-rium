import copy
import json
import logging
import random
import uuid
from pathlib import Path

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.auth import require_auth, ws_auth_ok
from core.comfy import post_prompt
from core.config import settings
from core.db import get_db
from core.models import Image
from core.thumbnail import make_thumbnail, thumb_rel_path

logger = logging.getLogger(__name__)
router = APIRouter()

# Workflow template loaded once at import time
_TEMPLATE = json.loads(
    (Path(__file__).parent.parent / "workflows" / "z-image_turbo.json").read_text()
)


_ALLOWED_LORAS = {
    "rustorangeanddimblue_lora_copy.safetensors",
    "art_vision_ZIG.safetensors",
    "zImageT_zidiusArt_melancholy.safetensors",
}


def _build_workflow(
    prompt: str, seed: int, width: int, height: int,
    lora_name: str, lora_strength: float,
) -> dict:
    wf = copy.deepcopy(_TEMPLATE)
    wf["45"]["inputs"]["text"] = prompt
    wf["44"]["inputs"]["seed"] = seed if seed >= 0 else random.randint(0, 2**32 - 1)
    wf["41"]["inputs"]["width"] = width
    wf["41"]["inputs"]["height"] = height
    wf["51"]["inputs"]["lora_name"] = lora_name
    wf["51"]["inputs"]["strength_model"] = round(max(0.0, min(1.0, lora_strength)), 3)
    return wf


class GenerateRequest(BaseModel):
    prompt: str
    seed: int = -1
    width: int = 1024
    height: int = 1024
    client_id: str
    batch_count: int = 1
    lora_name: str = "rustorangeanddimblue_lora_copy.safetensors"
    lora_strength: float = 0.5


@router.post("/api/generate", dependencies=[Depends(require_auth)])
async def generate(req: GenerateRequest, request: Request):
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt is required")
    if req.lora_name not in _ALLOWED_LORAS:
        raise HTTPException(status_code=400, detail=f"Unknown LoRA: {req.lora_name}")

    listener = request.app.state.comfy_listener
    batch_count = max(1, min(10, req.batch_count))
    batch_id = str(uuid.uuid4())
    prompt_ids = []

    for i in range(batch_count):
        seed = (req.seed + i) if req.seed >= 0 else random.randint(0, 2**32 - 1)
        workflow = _build_workflow(req.prompt, seed, req.width, req.height, req.lora_name, req.lora_strength)

        result = await post_prompt(workflow, listener.client_id)

        prompt_id = result.get("prompt_id")
        if not prompt_id:
            raise HTTPException(status_code=500, detail="ComfyUI did not return prompt_id")

        listener.register_prompt(
            prompt_id=prompt_id,
            client_id=req.client_id,
            index=i + 1,
            total=batch_count,
            batch_id=batch_id,
            prompt_text=req.prompt,
            seed=seed,
            width=req.width,
            height=req.height,
        )
        prompt_ids.append(prompt_id)
        logger.info(f"Queued [{i+1}/{batch_count}] prompt={prompt_id}")

    return {"batch_id": batch_id, "prompt_ids": prompt_ids, "batch_count": batch_count}


@router.get("/share/image/{filename}")
async def get_shared_image(filename: str, token: str = ""):
    """Public image endpoint for external services (e.g. Instagram).
    Protected by IMAGE_SHARE_TOKEN instead of the regular API key."""
    if settings.image_share_token and token != settings.image_share_token:
        raise HTTPException(status_code=403, detail="Invalid share token")
    safe_name = Path(filename).name
    for search_dir in [settings.images_dir, settings.comfyui_output_dir]:
        for candidate in search_dir.rglob(safe_name):
            if candidate.exists():
                return FileResponse(candidate, media_type="image/png")
    raise HTTPException(status_code=404, detail="Image not found")


@router.get("/share/reel/{filename}")
async def get_shared_reel(filename: str, token: str = ""):
    """Public video endpoint for Reel uploads to Instagram Graph API.
    Protected by IMAGE_SHARE_TOKEN (same token as images)."""
    if settings.image_share_token and token != settings.image_share_token:
        raise HTTPException(status_code=403, detail="Invalid share token")
    safe_name = Path(filename).name
    candidate = settings.reels_dir / safe_name
    if candidate.exists():
        return FileResponse(candidate, media_type="video/mp4")
    raise HTTPException(status_code=404, detail="Reel not found")


@router.get("/share/video/{filename}")
async def get_shared_video(filename: str, token: str = ""):
    """Public endpoint for serving generated videos to Instagram Graph API."""
    if settings.image_share_token and token != settings.image_share_token:
        raise HTTPException(status_code=403, detail="Invalid share token")
    safe_name = Path(filename).name
    candidate = settings.videos_dir / safe_name
    if candidate.exists():
        return FileResponse(candidate, media_type="video/mp4")
    raise HTTPException(status_code=404, detail="Video not found")


@router.get("/api/image/{filename}", dependencies=[Depends(require_auth)])
async def get_image(filename: str):
    safe_name = Path(filename).name
    for search_dir in [settings.images_dir, settings.comfyui_output_dir]:
        for candidate in search_dir.rglob(safe_name):
            if candidate.exists():
                return FileResponse(candidate, media_type="image/png")
    raise HTTPException(status_code=404, detail="Image not found")


@router.get("/api/image/{filename}/thumb", dependencies=[Depends(require_auth)])
async def get_image_thumb(filename: str, db: AsyncSession = Depends(get_db)):
    """Serve the JPEG thumbnail; fall back to the full image if no thumbnail exists."""
    safe_name = Path(filename).name

    # Look up DB record to find stored thumbnail_path
    result = await db.execute(select(Image).where(Image.filename == safe_name))
    img = result.scalar_one_or_none()

    if img and img.thumbnail_path:
        thumb = settings.storage_dir / img.thumbnail_path
        if thumb.exists():
            return FileResponse(thumb, media_type="image/jpeg")

    # Fall back: serve full image (same logic as get_image)
    for search_dir in [settings.images_dir, settings.comfyui_output_dir]:
        for candidate in search_dir.rglob(safe_name):
            if candidate.exists():
                return FileResponse(candidate, media_type="image/png")
    raise HTTPException(status_code=404, detail="Image not found")


@router.post("/api/images/backfill-thumbnails", dependencies=[Depends(require_auth)])
async def backfill_thumbnails(db: AsyncSession = Depends(get_db)):
    """
    Generate missing thumbnails for all images that don't have one yet.
    Safe to call multiple times — skips images that already have a thumbnail.
    """
    result = await db.execute(select(Image).where(Image.thumbnail_path.is_(None)))
    images = result.scalars().all()

    done, failed = 0, 0
    for img in images:
        src = settings.storage_dir / img.filepath
        if not src.exists():
            failed += 1
            continue
        rel = thumb_rel_path(img.filename)
        dest = settings.storage_dir / rel
        ok = await make_thumbnail(src, dest)
        if ok:
            img.thumbnail_path = rel
            done += 1
        else:
            failed += 1

    await db.commit()
    return {"backfilled": done, "failed": failed, "total": len(images)}


@router.post("/api/clear_pending_images/{client_id}", dependencies=[Depends(require_auth)])
async def clear_pending_images(client_id: str, request: Request):
    listener = request.app.state.comfy_listener
    count = len(listener._pending.get(client_id, []))
    listener._pending.pop(client_id, None)
    return {"cleared": count}


@router.get("/api/health")
async def health():
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(f"http://{settings.comfyui_host}/system_stats")
            comfy_ok = r.status_code == 200
    except Exception:
        comfy_ok = False
    return {
        "status": "ok",
        "comfyui": "connected" if comfy_ok else "unreachable",
        "auth_required": bool(settings.api_key),
    }


@router.websocket("/ws/{client_id}")
async def ws_endpoint(websocket: WebSocket, client_id: str):
    if not ws_auth_ok(websocket):
        logger.warning(f"Rejected WS connection from {websocket.client.host}")
        await websocket.close(code=4001)
        return

    await websocket.accept()
    listener = websocket.app.state.comfy_listener
    listener.add_ws(client_id, websocket)
    logger.info(f"Frontend connected: {client_id}")

    # Replay any images that arrived while client was disconnected
    await listener.replay_pending(client_id, websocket)

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        listener.remove_ws(client_id)
        logger.info(f"Frontend disconnected: {client_id}")
