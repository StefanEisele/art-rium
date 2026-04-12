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

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from core.auth import require_auth
from core.comfy import post_prompt
from core.config import settings
from core.db import get_db
from core.models import Image

logger = logging.getLogger(__name__)
router = APIRouter(dependencies=[Depends(require_auth)])

_WORKFLOW = json.loads(
    (Path(__file__).parent.parent / "workflows" / "qwen_3_5_image_titler.json").read_text()
)

OUTPUT_NODE = "6"  # PreviewAny node — this is what fires the executed event with text
_TITLER_TIMEOUT = 20.0  # Ollama inference is slow


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
async def run_titler(
    req: TitlerRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    try:
        img = await db.get(Image, uuid.UUID(req.image_id))
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid image_id")
    if not img:
        raise HTTPException(status_code=404, detail="Image not found")

    # Prefer thumbnail for analysis — smaller file is sufficient for Ollama
    src_rel = img.thumbnail_path or img.filepath
    filepath = settings.storage_dir / src_rel
    if not filepath.exists():
        # thumbnail missing but full image might exist — fall back gracefully
        if img.thumbnail_path:
            filepath = settings.storage_dir / img.filepath
        if not filepath.exists():
            raise HTTPException(status_code=404, detail="Image file not found on disk")

    image_b64 = await _read_b64(filepath)
    workflow = _build_workflow(image_b64)
    listener = request.app.state.comfy_listener

    result = await post_prompt(workflow, listener.client_id, timeout=_TITLER_TIMEOUT)

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
