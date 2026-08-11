"""
Ollama chat transport — the generic POST /api/chat wrapper used by every
higher-level Ollama call (analysis, article writers), plus basic
health/VRAM-management helpers that talk to Ollama directly.
"""
import asyncio
import base64
import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import httpx

from core.config import settings

logger = logging.getLogger(__name__)

# Ollama ≥0.17 sizes the default context window from total VRAM (32k on this
# machine) and multiplies it by the parallel-slot count, so a model loaded
# without an explicit num_ctx balloons to its full trained context — the 3B
# titler then holds 13 GB of KV cache and starves ComfyUI. 16k covers the
# largest real payload (20 VLM images at 512px ≈ 8k tokens plus prompt text)
# at ~1.3 GB of KV. Callers that need a different window pass their own
# num_ctx, which wins over this default.
DEFAULT_NUM_CTX = 16384

_PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"


@lru_cache(maxsize=64)
def _read_prompt_cached(filename: str, _mtime_ns: int) -> str:
    return (_PROMPTS_DIR / filename).read_text(encoding="utf-8")


def _read_prompt(filename: str) -> str:
    """Load a file from prompts/, cached per (name, mtime).

    Keying on mtime means editing a prompt file takes effect on the next call
    instead of at the next server restart. That matters because these files
    are the tuning surface for prompt behaviour — the previous
    cache-forever-per-process version silently served the pre-edit text for
    the life of the process, so a tuning round could be evaluated against the
    prompt it was meant to replace.
    """
    path = _PROMPTS_DIR / filename
    try:
        mtime_ns = path.stat().st_mtime_ns
    except OSError:
        mtime_ns = 0  # let read_text raise the real error below
    return _read_prompt_cached(filename, mtime_ns)


def _b64_jpgs(jpgs: list[bytes]) -> list[str]:
    return [base64.b64encode(j).decode("ascii") for j in jpgs]


def _strip_json_fences(content: str) -> str:
    """qwen3-family models occasionally wrap JSON in ```json fences despite
    the no-fences instruction. Idempotent for fence-free content."""
    s = content.strip()
    if s.startswith("```"):
        first_nl = s.find("\n")
        if first_nl != -1:
            s = s[first_nl + 1:]
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3].rstrip()
    return s


async def _chat_json(
    *,
    model: str,
    user_text: str,
    system: str | None = None,
    jpgs: list[bytes] | None = None,
    options: dict[str, Any] | None = None,
    keep_alive: str | None = None,
    think: bool | None = None,
    timeout: float,
    label: str,
    salvage: Callable[[str], str] | None = None,
    max_attempts: int = 1,
) -> dict[str, Any]:
    """
    POST to Ollama /api/chat with format=json and return the parsed message.

    On JSONDecodeError, apply `salvage(content) -> repaired` (when given) and
    retry parsing. Up to `max_attempts` fresh model calls are made when content
    is empty or unparseable. The fence-stripper is always applied as a pre-step.

    Raises:
      RuntimeError on Ollama-side error, empty content past last attempt, or
        non-parseable JSON past last attempt.
      httpx.HTTPError on network/timeout failures (propagated).
    """
    messages: list[dict[str, Any]] = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    user_msg: dict[str, Any] = {"role": "user", "content": user_text}
    if jpgs:
        user_msg["images"] = _b64_jpgs(jpgs)
    messages.append(user_msg)

    payload: dict[str, Any] = {
        "model":    model,
        "messages": messages,
        "format":   "json",
        "stream":   False,
        "options":  {"num_ctx": DEFAULT_NUM_CTX, **(options or {})},
    }
    if keep_alive is not None:
        payload["keep_alive"] = keep_alive
    if think is not None:
        payload["think"] = think

    last_exc: json.JSONDecodeError | None = None
    last_content = ""

    for attempt in range(1, max_attempts + 1):
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(f"{settings.ollama_host}/api/chat", json=payload)
            if r.status_code >= 400:
                # Ollama 5xx usually carries the real cause (OOM, ctx overflow,
                # model-load failure) in the response body. raise_for_status()
                # would drop it. Surface it before re-raising.
                body_snippet = (r.text or "")[:1000]
                logger.error(
                    "%s: Ollama returned HTTP %d — body: %s",
                    label, r.status_code, body_snippet or "(empty)",
                )
                r.raise_for_status()
            data = r.json()

        if "error" in data:
            raise RuntimeError(f"{label}: Ollama error: {data['error']}")

        content = data.get("message", {}).get("content", "") or ""
        if not content.strip():
            logger.warning("%s: attempt %d returned empty content", label, attempt)
            if attempt < max_attempts:
                continue
            raise RuntimeError(f"{label}: Ollama returned empty content")
        last_content = content

        stripped = _strip_json_fences(content)
        try:
            parsed = json.loads(stripped)
            if attempt > 1:
                logger.info("%s: attempt %d produced valid JSON.", label, attempt)
            return parsed
        except json.JSONDecodeError as exc:
            if salvage is not None:
                salvaged = salvage(stripped)
                if salvaged != stripped:
                    try:
                        parsed = json.loads(salvaged)
                        logger.warning(
                            "%s: JSON parse failed but salvage succeeded (orig err: %s)",
                            label, exc,
                        )
                        return parsed
                    except json.JSONDecodeError as exc2:
                        logger.warning(
                            "%s: salvage made progress but did not parse (orig=%s, salvaged=%s)",
                            label, exc, exc2,
                        )
            last_exc = exc
            if attempt < max_attempts:
                logger.warning(
                    "%s: attempt %d failed JSON parse (%s); retrying (%d/%d).",
                    label, attempt, exc, attempt + 1, max_attempts,
                )

    logger.error(
        "%s: returned non-JSON after %d attempt(s) — last err: %s — content head: %s",
        label, max_attempts, last_exc, last_content[:500],
    )
    snippet = (last_content[:300] + "...") if len(last_content) > 300 else (last_content or "(empty)")
    raise RuntimeError(
        f"{label}: Ollama returned non-JSON after {max_attempts} attempt(s) "
        f"(parse err: {last_exc}) — start: {snippet!r}"
    ) from last_exc


async def reachable() -> bool:
    """Best-effort Ollama health check."""
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            r = await client.get(f"{settings.ollama_host}/api/tags")
            return r.status_code == 200
    except Exception:
        return False


async def unload_model(model: str, timeout: float = 30.0) -> None:
    """
    Tell Ollama to unload *model* from VRAM immediately.

    Best-effort: errors are logged and swallowed. Calling this on a model
    that isn't loaded is a no-op on Ollama's side. We use POST /api/generate
    with no prompt and keep_alive=0 — the documented way to evict a model.

    Used before the heavy modal-article LLM call (qwen3.6:27b, 17 GB) to
    free VRAM that the small titler model (qwen2.5vl:3b) holds resident
    for 30 min after warm-up. Without this, loading the 17 GB model on top
    of the resident titler crashes Ollama's model-runner subprocess on
    constrained GPUs.
    """
    if not model:
        return
    payload = {"model": model, "keep_alive": 0}
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(f"{settings.ollama_host}/api/generate", json=payload)
        if r.status_code == 200:
            logger.info("Ollama: unloaded model %s from VRAM", model)
        else:
            logger.warning(
                "Ollama: unload of %s returned %d — %s",
                model, r.status_code, (r.text or "")[:200],
            )
    except Exception as exc:
        logger.warning("Ollama: unload of %s failed (%s) — proceeding anyway", model, exc)


async def wait_until_unloaded(timeout: float = 30.0, poll: float = 0.25) -> bool:
    """Block until Ollama reports no model resident. Returns True if the card
    came free, False on timeout.

    `unload_model` only waits for Ollama to *accept* the eviction, not to carry
    it out: measured on this box, the POST returns after ~250 ms while the
    runner subprocess still holds its VRAM for a few hundred ms more. Whoever
    grabs the GPU next — ComfyUI loading a 19.5 GB model — must not start
    inside that window, so callers that are about to hand the card over need a
    real confirmation rather than a best-effort request.

    Best-effort like the rest of this module: a probe that cannot reach Ollama
    reports success, because a dead Ollama is holding nothing.
    """
    deadline = asyncio.get_event_loop().time() + timeout
    while True:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.get(f"{settings.ollama_host}/api/ps")
            resident = [m.get("name") for m in r.json().get("models", [])]
        except Exception as exc:
            logger.warning("Ollama: /api/ps probe failed (%s) — assuming free", exc)
            return True
        if not resident:
            return True
        if asyncio.get_event_loop().time() >= deadline:
            logger.warning(
                "Ollama: still holding %s after %.0fs — proceeding anyway",
                ", ".join(str(m) for m in resident), timeout,
            )
            return False
        await asyncio.sleep(poll)
