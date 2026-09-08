"""Shared VRAM pre-flight for ComfyUI submissions.

First hardened for the video pipeline: Ollama's titler VLM (or a leftover
render from the job before) can sit in VRAM right up until a Wan/MiniMax
submission needs the whole card, and ComfyUI's dynamic-VRAM loader does not
fall back to a slower path for everything — MiniMax dies with a CUDA OOM that
takes its prompt worker thread with it, and other workflows silently stream
weights from CPU instead, which is not an error but is much slower than a
render that actually fit. Any workflow submitted through services.comfy.client
can hit either version of this, so the check lives here rather than only in
routers/video.py.
"""
import asyncio
import logging

import httpx

from core.config import settings
from services.comfy.client import free_memory
from services.ollama.analysis import cancel_titler_warmup
from services.ollama.chat import unload_model, wait_until_unloaded

logger = logging.getLogger(__name__)

# How long to wait for someone else to let go of the card. Sized for the
# thing that actually holds it: an Ollama cold load of the titler VLM, which
# takes ~150 s and cannot be aborted once it has started.
_VRAM_WAIT_TIMEOUT = 210.0
_VRAM_WAIT_POLL = 5.0

# ComfyUI's /free returns as soon as it has dropped its references; CUDA hands
# the memory back a moment later. Re-measuring immediately reads the old
# number — the same reason the per-segment loop elsewhere sleeps after
# free_memory.
_COMFY_FREE_SETTLE = 3.0


async def evict_ollama() -> None:
    """Ask Ollama to give the card back, and wait for it to confirm.

    Three steps, each because the previous one is not enough:

    1. Call off any in-flight startup warm-up. Evicting first does nothing
       against it — during a cold load the model is not resident yet, so the
       eviction finds an empty server and the warm-up then loads 5.3 GB *into*
       the render. See services/ollama/analysis.py::cancel_titler_warmup.
    2. Evict whatever is resident. The "✨ Suggest prompts" step just before a
       generation leaves the titler VLM in VRAM on a 30 min keep-alive, and
       Ollama sizes its KV cache from total VRAM, so even a 3B model holds
       gigabytes of a 16 GB card.
    3. Wait for confirmation — `unload_model` returns once Ollama accepts the
       request (~250 ms measured) while the runner keeps its VRAM longer.

    Still not a guarantee: none of this can stop an Ollama cold load that is
    already under way, which is why the caller checks free VRAM afterwards
    rather than trusting this to have worked.
    """
    await cancel_titler_warmup()
    for model in (
        settings.ollama_titler_model,
        settings.ollama_vlm_model,
        settings.ollama_prompt_model,
    ):
        if model:
            await unload_model(model)
    await wait_until_unloaded()


async def release_comfy_models() -> None:
    """Ask ComfyUI to unload its resident models. Best-effort, never raises."""
    async with httpx.AsyncClient(timeout=15) as client:
        await free_memory(client)


async def comfy_devices() -> list[dict]:
    """ComfyUI's device list, or a clear error explaining why there isn't one."""
    try:
        async with httpx.AsyncClient(timeout=8) as client:
            r = await client.get(f"http://{settings.comfyui_host}/system_stats")
    except Exception as exc:
        raise RuntimeError(f"ComfyUI is unreachable at {settings.comfyui_host} ({exc})") from exc

    # A CUDA OOM kills ComfyUI's `prompt_worker` *thread*, not its process: the
    # HTTP server keeps answering and /prompt keeps handing out prompt_ids while
    # nothing ever executes, so a job sits "generating" until the poll times out
    # half an hour later. /system_stats returning 500 is the reliable tell.
    if r.status_code != 200:
        raise RuntimeError(
            f"ComfyUI is up but not working (/system_stats → {r.status_code}). Its worker "
            "thread most likely died on an earlier out-of-memory error; restart ComfyUI."
        )
    try:
        return r.json().get("devices", []) or []
    except Exception:
        return []


async def preflight_vram(required: float, label: str) -> None:
    """Refuse to submit until the GPU actually has `required` bytes free.

    Evicting Ollama is not sufficient, and neither is cancelling its warm-up.
    An Ollama *cold* load takes ~150 s, allocates VRAM progressively as it
    goes, and cannot be stopped from the client side — `/api/ps` does not even
    list the model until the load finishes, so both `wait_until_unloaded` and
    `cancel_titler_warmup` see an idle server and wave the job through while
    several GB are quietly being taken. Measured on a post-reboot start:
    ComfyUI up at 14:51:18, job submitted 14:52:05, dead at 14:52:12.

    So stop reasoning about *who* holds the card and check the only thing that
    matters — how much is free — retrying the eviction while we wait. Failing
    here with a number is strictly better than the alternative: a CUDA OOM
    that kills ComfyUI's worker thread and needs a restart, or a render that
    silently takes several times longer than it should.

    Ollama is not the only holder, though, and for a post-pass it is usually
    not the holder at all: ComfyUI keeps the models from the render that just
    finished, so a job queued straight after one finds only a few GB free and
    no amount of evicting Ollama changes that. On the first shortfall we
    therefore ask ComfyUI to let go too, before starting to wait.
    """
    deadline = asyncio.get_event_loop().time() + _VRAM_WAIT_TIMEOUT
    warned = False
    released = False

    while True:
        devices = await comfy_devices()
        if not devices:
            logger.warning("ComfyUI reported no devices — skipping the VRAM pre-flight")
            return
        dev = devices[0]
        free, total = dev.get("vram_free", 0), dev.get("vram_total", 0)

        if free >= required:
            logger.info(
                "Pre-flight VRAM on %s: %.1f GB free of %.1f GB (need %.1f)",
                dev.get("name"), free / 1e9, total / 1e9, required / 1e9,
            )
            return

        # Before waiting on anyone, make ComfyUI drop what the previous render
        # left resident. Waiting cannot fix that on its own — nothing else
        # frees it — so a missing release here is a guaranteed timeout, not a
        # slow start.
        if not released:
            released = True
            logger.info(
                "Only %.1f GB free on %s — asking ComfyUI to unload before waiting",
                free / 1e9, dev.get("name"),
            )
            await release_comfy_models()
            await asyncio.sleep(_COMFY_FREE_SETTLE)
            continue

        if not warned:
            logger.warning(
                "Only %.1f GB free on %s, %s needs %.1f GB — waiting for the card",
                free / 1e9, dev.get("name"), label, required / 1e9,
            )
            warned = True

        if asyncio.get_event_loop().time() >= deadline:
            raise RuntimeError(
                f"GPU still busy after {_VRAM_WAIT_TIMEOUT:.0f}s: only "
                f"{free / 1e9:.1f} GB free of {total / 1e9:.1f} GB, {label} "
                f"needs {required / 1e9:.1f} GB. Something else is holding the card "
                "(an Ollama model — check `ollama ps` — or a ComfyUI render that "
                "will not unload; restarting ComfyUI clears the latter)."
            )

        # Whoever it is may only just have finished loading; try both holders
        # again before the next check.
        await evict_ollama()
        await release_comfy_models()
        await asyncio.sleep(_VRAM_WAIT_POLL)


async def free_vram_for(required: float, label: str) -> None:
    """Make the GPU ready for a ComfyUI render, or fail loudly trying.

    Evict Ollama first, then verify the card is actually free before
    submitting — falling short is not a slow path for every workflow (see
    `preflight_vram`), so check rather than trust that the eviction worked.
    """
    await evict_ollama()
    await preflight_vram(required, label)
