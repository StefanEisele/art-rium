"""
ComfyUI client — worker-side helpers for long-running background generation.

Used by:
  - routers/video.py::_run_generation   (key-frame video pipeline)

Differs from core.comfy.post_prompt (router-side) in that errors raise
RuntimeError rather than HTTPException — these helpers run inside background
asyncio tasks where FastAPI exception types are inappropriate.

For interactive routers that submit a workflow and return a prompt_id to a
WebSocket consumer (generate, titler), keep using core.comfy.post_prompt.
"""
import asyncio
import logging
from pathlib import Path

import httpx

from core.config import settings

logger = logging.getLogger(__name__)


async def upload_image(client: httpx.AsyncClient, filepath: Path, name: str) -> str:
    """Upload a PNG to ComfyUI's input folder, return the assigned filename."""
    with open(filepath, "rb") as f:
        data = f.read()
    r = await client.post(
        f"http://{settings.comfyui_host}/upload/image",
        files={"image": (name, data, "image/png")},
        data={"type": "input", "overwrite": "true"},
    )
    r.raise_for_status()
    return r.json()["name"]


async def post_workflow(client: httpx.AsyncClient, workflow: dict) -> str:
    """Submit a workflow to ComfyUI's /prompt endpoint, return the prompt_id."""
    r = await client.post(
        f"http://{settings.comfyui_host}/prompt",
        json={"prompt": workflow},
        timeout=30,
    )
    if r.status_code != 200:
        body = r.text
        logger.error("ComfyUI rejected workflow (%d): %s", r.status_code, body[:3000])
        raise RuntimeError(f"ComfyUI rejected workflow ({r.status_code}): {body[:500]}")
    data = r.json()
    pid = data.get("prompt_id")
    if not pid:
        raise RuntimeError(f"ComfyUI did not return prompt_id: {data}")
    return pid


async def poll_history(
    client: httpx.AsyncClient,
    prompt_id: str,
    *,
    timeout: int,
    interval: int,
) -> dict:
    """Poll /history/<prompt_id> until status='success' or timeout. Returns outputs dict."""
    deadline = asyncio.get_event_loop().time() + timeout
    while True:
        r = await client.get(
            f"http://{settings.comfyui_host}/history/{prompt_id}",
            timeout=10,
        )
        if r.status_code == 200:
            body = r.json()
            entry = body.get(prompt_id, {})
            status = entry.get("status", {})
            if status.get("completed"):
                if status.get("status_str") != "success":
                    msgs = [str(m) for m in status.get("messages", [])]
                    raise RuntimeError(f"ComfyUI job failed: {' | '.join(msgs)}")
                return entry.get("outputs", {})
        if asyncio.get_event_loop().time() >= deadline:
            raise TimeoutError(f"ComfyUI job {prompt_id} timed out after {timeout}s")
        await asyncio.sleep(interval)


async def queue_info(prompt_id: str) -> dict:
    """Return queue status for a prompt_id (best-effort, never raises)."""
    try:
        async with httpx.AsyncClient(timeout=4) as client:
            r = await client.get(f"http://{settings.comfyui_host}/queue")
            q = r.json()
        for item in q.get("queue_running", []):
            if len(item) > 1 and item[1] == prompt_id:
                return {"status": "running"}
        for i, item in enumerate(q.get("queue_pending", [])):
            if len(item) > 1 and item[1] == prompt_id:
                return {"status": "pending", "position": i + 1}
        return {"status": "not_in_queue"}
    except Exception:
        return {"status": "unknown"}
