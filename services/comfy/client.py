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
    """Poll /history/<prompt_id> until status='success' or timeout. Returns outputs dict.

    Transient network errors are absorbed and retried — ComfyUI's HTTP thread can
    stall for several seconds during model load/unload, and a single ReadTimeout
    should not kill a multi-minute job. After MAX_CONSECUTIVE_MISSES failed polls
    in a row (~90s), we assume ComfyUI is genuinely down and raise.
    """
    deadline = asyncio.get_event_loop().time() + timeout
    MAX_CONSECUTIVE_MISSES = 6
    misses = 0
    while True:
        try:
            r = await client.get(
                f"http://{settings.comfyui_host}/history/{prompt_id}",
                timeout=30,
            )
            misses = 0
            if r.status_code == 200:
                body = r.json()
                entry = body.get(prompt_id, {})
                status = entry.get("status", {})
                status_str = status.get("status_str")
                # Fail fast on error state — some ComfyUI versions never set completed=True after a node exception
                if status_str == "error" or any(
                    isinstance(m, (list, tuple)) and m and m[0] == "execution_error"
                    for m in status.get("messages", [])
                ):
                    msgs = [str(m) for m in status.get("messages", [])]
                    raise RuntimeError(f"ComfyUI job failed: {' | '.join(msgs)}")
                if status.get("completed"):
                    if status_str != "success":
                        msgs = [str(m) for m in status.get("messages", [])]
                        raise RuntimeError(f"ComfyUI job failed: {' | '.join(msgs)}")
                    return entry.get("outputs", {})
        except httpx.RequestError as e:
            misses += 1
            logger.warning(
                "ComfyUI history poll failed (%s: %s) — miss %d/%d",
                type(e).__name__, e, misses, MAX_CONSECUTIVE_MISSES,
            )
            if misses >= MAX_CONSECUTIVE_MISSES:
                raise RuntimeError(
                    f"ComfyUI unreachable after {misses} consecutive polls — likely crashed"
                ) from e
        if asyncio.get_event_loop().time() >= deadline:
            raise TimeoutError(f"ComfyUI job {prompt_id} timed out after {timeout}s")
        await asyncio.sleep(interval)


async def free_memory(client: httpx.AsyncClient) -> None:
    """Ask ComfyUI to fully unload all models and free VRAM. Best-effort, never raises.

    Useful between independent submissions in a multi-segment pipeline. Without
    this, ComfyUI partially evicts a model after one prompt and partially reloads
    it for the next — on Wan 14B fp8 + 16 GB GPU the partial-reload path can hit
    an mmap access violation in load_torch_file and crash the ComfyUI process.
    Forcing a clean unload makes the next prompt do a cold load instead.
    """
    try:
        r = await client.post(
            f"http://{settings.comfyui_host}/free",
            json={"unload_models": True, "free_memory": True},
            timeout=10,
        )
        if r.status_code != 200:
            logger.warning("ComfyUI /free returned %d: %s", r.status_code, r.text[:200])
    except Exception as e:
        logger.warning("ComfyUI /free failed (%s): %s", type(e).__name__, e)


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
