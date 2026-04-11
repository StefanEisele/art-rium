"""
Image Titler — runs a ComfyUI/Ollama workflow that generates title suggestions
for a given image and returns them as a text stream via WebSocket.
"""
import asyncio
import base64
import copy
import json
import logging
import uuid
from pathlib import Path

import httpx
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from core.auth import auth_ok
from core.config import settings
from core.db import AsyncSessionLocal
from core.models import Image

logger = logging.getLogger(__name__)
router = APIRouter()

_WORKFLOW = json.loads(
    (Path(__file__).parent.parent / "workflows" / "qwen_3_5_image_titler.json").read_text()
)

COMFYUI_TIMEOUT = 20.0
OUTPUT_NODE = "1"  # OllamaChat node that produces the text


def _build_workflow(image_b64: str) -> dict:
    wf = copy.deepcopy(_WORKFLOW)
    wf["12"]["inputs"]["image"] = image_b64
    return wf


async def _read_b64(path: Path) -> str:
    def _read():
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return await asyncio.to_thread(_read)


class TitlerRequest(BaseModel):
    image_id: str
    client_id: str


@router.post("/api/titler/run")
async def run_titler(req: TitlerRequest, request: Request):
    if not auth_ok(request):
        raise HTTPException(status_code=401, detail="Invalid API key")

    # Load image record
    async with AsyncSessionLocal() as session:
        try:
            img = await session.get(Image, uuid.UUID(req.image_id))
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid image_id")
        if not img:
            raise HTTPException(status_code=404, detail="Image not found")
        filepath = settings.storage_dir / img.filepath

    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Image file not found on disk")

    image_b64 = await _read_b64(filepath)
    workflow = _build_workflow(image_b64)
    listener = request.app.state.comfy_listener

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
        raise HTTPException(status_code=500, detail=f"ComfyUI error {exc.response.status_code}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    prompt_id = result.get("prompt_id")
    if not prompt_id:
        raise HTTPException(status_code=500, detail="ComfyUI did not return prompt_id")

    listener.register_text_prompt(
        prompt_id=prompt_id,
        client_id=req.client_id,
        output_node=OUTPUT_NODE,
    )
    logger.info(f"Titler queued: prompt={prompt_id} image={req.image_id}")

    return {"prompt_id": prompt_id}
