"""
Z-Image Turbo prompt enhancer — idea -> N styled prompts.
Backed by prompts/zimage-enhancer.md (system template, with {STYLE_BLOCK}
placeholder) and prompts/zimage-styles.md (four-style signature library).
"""
import asyncio
import logging
import re
from functools import lru_cache
from typing import Any

import httpx

from core.config import settings
from services.ollama.chat import _read_prompt

logger = logging.getLogger(__name__)

# Deterministic rotation across the full four-style set — all equally
# weighted defaults, no optional/overflow tier.
_ZIMAGE_STYLE_SECTIONS = ("A", "B", "C", "D")


@lru_cache(maxsize=1)
def _zimage_style_blocks() -> dict[str, str]:
    """Parse prompts/zimage-styles.md into {style_letter: block_text} pairs.

    Each block is the heading `## Style X — ...` plus all following lines
    until the next `## ` heading or EOF. Cached for the process lifetime.
    """
    text = _read_prompt("zimage-styles.md")
    pattern = re.compile(
        r"^##\s+Style\s+([A-D])\b.*?(?=^##\s+|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    return {m.group(1): m.group(0).strip() for m in pattern.finditer(text)}


def _zimage_style_name(letter: str) -> str:
    """Pull the display name from the style block's first heading line.
    `## Style A — Wire-Wrapped Tenebrism` → `Wire-Wrapped Tenebrism`."""
    block = _zimage_style_blocks().get(letter, "")
    first_line = block.splitlines()[0] if block else ""
    if "—" in first_line:
        return first_line.split("—", 1)[1].strip()
    return f"Style {letter}"


def list_zimage_styles() -> list[dict[str, str]]:
    """Public style catalogue: [{style, name}, ...] in A..D order. Used by
    GET /api/prompts/styles so pickers render from the markdown SSOT."""
    return [
        {"style": letter, "name": _zimage_style_name(letter)}
        for letter in _ZIMAGE_STYLE_SECTIONS
        if letter in _zimage_style_blocks()
    ]


def get_zimage_style_block(letter: str) -> str | None:
    """Full markdown block for one style letter, or None when unknown."""
    return _zimage_style_blocks().get(letter)


async def _enhance_one_zimage(idea: str, style_letter: str, timeout: float) -> str:
    """Single-style enhancement call. Returns the enhanced prompt string."""
    blocks = _zimage_style_blocks()
    if style_letter not in blocks:
        raise ValueError(f"Unknown style letter: {style_letter}")
    system = _read_prompt("zimage-enhancer.md").replace("{STYLE_BLOCK}", blocks[style_letter])

    payload: dict[str, Any] = {
        "model":   settings.ollama_prompt_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": idea.strip()},
        ],
        "stream":  False,
        # qwen3 instruct family: disable thinking — chain-of-thought tokens
        # would burn num_predict and slow this down without quality gains
        # on a creative rewrite task.
        "think":   False,
        "options": {
            "temperature":    0.7,
            "top_p":          0.9,
            "repeat_penalty": 1.05,
            "num_ctx":        4096,
            "num_predict":    512,
        },
    }
    # Connect-level retry. All N styles fire concurrently in one burst, so a
    # momentary Ollama hiccup (model swap, brief restart, the app's own VRAM-free
    # pass) can refuse every connection in the same instant and 502 the whole
    # request. Ollama recovers within a second or two, so retry connect failures
    # with a short backoff. HTTP-status / empty-content errors are NOT retried —
    # those are genuine and retrying 6× concurrently would only pile up.
    attempts = 3
    for attempt in range(1, attempts + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.post(f"{settings.ollama_host}/api/chat", json=payload)
                r.raise_for_status()
                data = r.json()
            break
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            if attempt == attempts:
                raise
            delay = 1.5 * attempt
            logger.warning(
                "enhance: style %s connect failed (attempt %d/%d), retrying in %.1fs: %s",
                style_letter, attempt, attempts, delay, exc,
            )
            await asyncio.sleep(delay)
    if "error" in data:
        raise RuntimeError(f"Ollama error: {data['error']}")
    content = (data.get("message", {}).get("content", "") or "").strip()
    if not content:
        raise RuntimeError("empty content")

    # Defensive: strip the model's occasional self-quoting (`"…"`) or leading
    # `Output:` / `Prompt:` labels despite the OUTPUT CONTRACT clause.
    if content.startswith('"') and content.endswith('"') and content.count('"') == 2:
        content = content[1:-1].strip()
    for prefix in ("Output:", "Prompt:", "Enhanced prompt:"):
        if content.lower().startswith(prefix.lower()):
            content = content[len(prefix):].lstrip(" :\n")
            break
    return content


async def enhance_zimage_prompts(
    idea: str,
    *,
    n: int = 4,
    timeout: float = 120.0,
) -> list[dict[str, str]]:
    """
    Enhance a short user idea into *n* Z-Image Turbo prompts, one per style.

    Iterates through styles A..D and dispatches one Qwen call per style.
    Calls run concurrently — Ollama serialises on its own queue if the model
    is single-instance, so wall time is roughly n × single_call_latency on
    shared hardware.

    Returns a list of {style, name, prompt} dicts in style order. Styles
    that fail (timeout, empty response, Ollama error) are SKIPPED, so the
    returned length may be < n. The caller decides whether to retry or
    proceed with a partial set.
    """
    idea = idea.strip()
    if not idea:
        raise ValueError("enhance_zimage_prompts: idea is empty")
    if not (1 <= n <= len(_ZIMAGE_STYLE_SECTIONS)):
        raise ValueError(f"enhance_zimage_prompts: n must be 1..{len(_ZIMAGE_STYLE_SECTIONS)}")

    selected = _ZIMAGE_STYLE_SECTIONS[:n]

    async def _enhance_safe(letter: str) -> dict[str, str] | None:
        try:
            prompt = await _enhance_one_zimage(idea, letter, timeout)
            return {"style": letter, "name": _zimage_style_name(letter), "prompt": prompt}
        except Exception as exc:
            logger.warning("enhance_zimage_prompts: style %s failed: %s", letter, exc)
            return None

    logger.info(
        "enhance_zimage_prompts: model=%s, idea len=%d, n=%d",
        settings.ollama_prompt_model, len(idea), n,
    )
    results = await asyncio.gather(*(_enhance_safe(s) for s in selected))
    successful = [r for r in results if r is not None]
    logger.info(
        "enhance_zimage_prompts: produced %d/%d prompts (%s)",
        len(successful), n, ",".join(r["style"] for r in successful),
    )
    return successful
