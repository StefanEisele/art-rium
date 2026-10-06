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
import time
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


async def post_workflow(
    client: httpx.AsyncClient, workflow: dict, client_id: str | None = None,
) -> str:
    """Submit a workflow to ComfyUI's /prompt endpoint, return the prompt_id.

    The submission carries the listener's `client_id`, and that is not
    cosmetic. ComfyUI routes execution events to the session that submitted the
    prompt — `server.send_sync("executing", …, server.client_id)` in
    execution.py — so a prompt posted without one leaves the listener with no
    `executing` or `progress` events for it at all. Every caller here registers
    node labels and expects `GET …/progress` to read out ComfyUI's live stage;
    without the id those readouts stay empty for the whole render, which is
    exactly what a multi-minute job can least afford.
    """
    if client_id is None:
        # Imported here: workers.comfy_listener pulls in services.comfy.ingest,
        # and a module-level import back the other way would close the loop.
        from workers.comfy_listener import get_listener

        listener = get_listener()
        client_id = listener.client_id if listener else None

    payload: dict = {"prompt": workflow}
    if client_id:
        payload["client_id"] = client_id
    r = await client.post(
        f"http://{settings.comfyui_host}/prompt",
        json=payload,
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


# Which files a loader node is currently offering. ComfyUI builds these enums by
# scanning its models directory at startup, so the answer changes when ComfyUI
# restarts — and a workflow naming a file the enum does not hold is rejected at
# submit time with a validation error, which is a worse way to find out than
# asking first.
#
# The cache is short-lived on purpose. A process-lifetime cache would be the
# obvious choice and it is the wrong one: the interesting answer is "that model
# is not installed", and it is interesting precisely at the moment someone is
# installing it. Caching that for as long as this server happens to run means
# downloading a checkpoint, restarting ComfyUI, and still being told it is
# missing until art-rium is restarted too — for no reason the user can see.
# A minute is long enough that a job asking once per clip costs one request,
# and short enough that a ComfyUI restart is noticed on its own.
_LOADER_CACHE_TTL = 60.0
_loader_choices: dict[tuple[str, str], tuple[float, set[str]]] = {}


async def loader_choices(node: str, field: str) -> set[str]:
    """The set of file names `node`'s `field` combo box offers.

    Returns an empty set if ComfyUI cannot be reached or does not know the
    node — callers treat that as "cannot confirm" and fall back to whatever
    they were going to do anyway, rather than failing a render over a probe.
    A failed probe is never cached, for the same reason.
    """
    key = (node, field)
    hit = _loader_choices.get(key)
    if hit and (time.monotonic() - hit[0]) < _LOADER_CACHE_TTL:
        return hit[1]
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(f"http://{settings.comfyui_host}/object_info/{node}")
            info = r.json()
        raw = info[node]["input"]["required"][field][0]
        choices = {str(x) for x in raw}
    except Exception as e:
        logger.warning("ComfyUI /object_info/%s failed (%s): %s", node, type(e).__name__, e)
        return set()
    _loader_choices[key] = (time.monotonic(), choices)
    return choices


async def embedding_names() -> list[str] | None:
    """Every embedding ComfyUI can load right now, `/`-separated, or None when
    ComfyUI cannot be reached.

    Not cached, unlike `loader_choices`: a training finishing is exactly the
    moment someone reaches for the new name, and a minute-old list would say
    it does not exist. The call is a directory listing and costs nothing.
    """
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(f"http://{settings.comfyui_host}/embeddings")
            r.raise_for_status()
            return [str(n).replace("\\", "/") for n in r.json()]
    except Exception as e:
        logger.info("ComfyUI /embeddings unavailable (%s): %s", type(e).__name__, e)
        return None
