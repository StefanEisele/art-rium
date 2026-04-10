import copy
import json
import logging
import random
import uuid
from pathlib import Path

import httpx
from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from core.auth import auth_ok, ws_auth_ok
from core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

# Workflow template loaded once at import time
_TEMPLATE = json.loads(
    (Path(__file__).parent.parent / "workflows" / "z-image_turbo.json").read_text()
)

COMFYUI_TIMEOUT = 10.0


def _build_workflow(prompt: str, seed: int, width: int, height: int) -> dict:
    wf = copy.deepcopy(_TEMPLATE)
    wf["45"]["inputs"]["text"] = prompt
    wf["44"]["inputs"]["seed"] = seed if seed >= 0 else random.randint(0, 2**32 - 1)
    wf["41"]["inputs"]["width"] = width
    wf["41"]["inputs"]["height"] = height
    return wf


class GenerateRequest(BaseModel):
    prompt: str
    seed: int = -1
    width: int = 1024
    height: int = 1024
    client_id: str
    batch_count: int = 1


@router.post("/api/generate")
async def generate(req: GenerateRequest, request: Request):
    if not auth_ok(request):
        raise HTTPException(status_code=401, detail="Invalid API key")
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt is required")

    listener = request.app.state.comfy_listener
    batch_count = max(1, min(10, req.batch_count))
    batch_id = str(uuid.uuid4())
    prompt_ids = []

    for i in range(batch_count):
        seed = (req.seed + i) if req.seed >= 0 else random.randint(0, 2**32 - 1)
        workflow = _build_workflow(req.prompt, seed, req.width, req.height)

        try:
            async with httpx.AsyncClient(timeout=COMFYUI_TIMEOUT) as client:
                resp = await client.post(
                    f"http://{settings.comfyui_host}/prompt",
                    json={"prompt": workflow, "client_id": listener.client_id},
                )
                resp.raise_for_status()
                result = resp.json()
        except httpx.ConnectError:
            raise HTTPException(status_code=503, detail=f"ComfyUI unreachable: {settings.comfyui_host}")
        except httpx.HTTPStatusError as exc:
            raise HTTPException(status_code=500, detail=f"ComfyUI error: {exc.response.status_code}")
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

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


@router.get("/api/image/{filename}")
async def get_image(filename: str, request: Request):
    if not auth_ok(request):
        raise HTTPException(status_code=401, detail="Invalid API key")
    safe_name = Path(filename).name
    # Serve from managed storage; fall back to ComfyUI output dir
    for search_dir in [settings.images_dir, settings.comfyui_output_dir]:
        for candidate in search_dir.rglob(safe_name):
            if candidate.exists():
                from fastapi.responses import FileResponse
                return FileResponse(candidate, media_type="image/png")
    raise HTTPException(status_code=404, detail="Image not found")


@router.post("/api/clear_pending_images/{client_id}")
async def clear_pending_images(client_id: str, request: Request):
    if not auth_ok(request):
        raise HTTPException(status_code=401, detail="Invalid API key")
    listener = request.app.state.comfy_listener
    count = len(listener._pending.get(client_id, []))
    listener._pending.pop(client_id, None)
    return {"cleared": count}


@router.get("/api/health")
async def health(request: Request):
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
