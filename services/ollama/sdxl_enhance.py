"""
ArtiVision XL (SDXL) prompt enhancer — idea -> N styled positive+negative
prompt pairs. Reuses the same four-style library as the Z-Image Turbo
enhancer (prompts/zimage-styles.md, via services.ollama.zimage_enhance's
public style catalogue) so both generator models share one signature-style
SSOT, but drives it through prompts/sdxl-enhancer.md — a system template
written for SDXL's tag/phrase idiom and an active negative prompt, rather
than Z-Image Turbo's flowing-prose, negative-ignoring one.
"""
import asyncio
import logging
import re
from typing import Any

import httpx

from core.config import settings
from services.ollama.chat import _read_prompt
from services.ollama.zimage_enhance import get_zimage_style_block, list_zimage_styles

logger = logging.getLogger(__name__)

_POSITIVE_RE = re.compile(r"^positive:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
_NEGATIVE_RE = re.compile(r"^negative:\s*(.+)$", re.IGNORECASE | re.MULTILINE)


def _parse_positive_negative(content: str) -> tuple[str, str]:
    pos_m = _POSITIVE_RE.search(content)
    neg_m = _NEGATIVE_RE.search(content)
    if not pos_m or not neg_m:
        raise RuntimeError(f"enhancer output missing POSITIVE/NEGATIVE lines: {content[:200]!r}")
    return pos_m.group(1).strip(), neg_m.group(1).strip()


async def _enhance_one_sdxl(idea: str, style_letter: str, timeout: float) -> dict[str, str]:
    """Single-style enhancement call. Returns {"prompt": ..., "negative_prompt": ...}."""
    block = get_zimage_style_block(style_letter)
    if block is None:
        raise ValueError(f"Unknown style letter: {style_letter}")
    system = _read_prompt("sdxl-enhancer.md").replace("{STYLE_BLOCK}", block)

    payload: dict[str, Any] = {
        "model":   settings.ollama_prompt_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": idea.strip()},
        ],
        "stream":  False,
        "think":   False,
        "options": {
            "temperature":    0.7,
            "top_p":          0.9,
            "repeat_penalty": 1.05,
            "num_ctx":        4096,
            "num_predict":    512,
        },
    }
    # Same connect-level retry rationale as zimage_enhance: all N styles fire
    # concurrently, so a momentary Ollama hiccup can refuse every connection
    # in the same instant. HTTP-status / empty-content errors are not retried.
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
                "sdxl enhance: style %s connect failed (attempt %d/%d), retrying in %.1fs: %s",
                style_letter, attempt, attempts, delay, exc,
            )
            await asyncio.sleep(delay)
    if "error" in data:
        raise RuntimeError(f"Ollama error: {data['error']}")
    content = (data.get("message", {}).get("content", "") or "").strip()
    if not content:
        raise RuntimeError("empty content")

    positive, negative = _parse_positive_negative(content)
    return {"prompt": positive, "negative_prompt": negative}


async def enhance_sdxl_prompts(
    idea: str,
    *,
    n: int = 4,
    timeout: float = 120.0,
) -> list[dict[str, str]]:
    """
    Enhance a short user idea into *n* ArtiVision XL prompt pairs, one per
    style. Mirrors enhance_zimage_prompts's concurrency/skip-on-failure
    behaviour — see that function for the rationale. Returns a list of
    {style, name, prompt, negative_prompt} dicts in style order; failed
    styles are skipped, so the result may be shorter than n.
    """
    idea = idea.strip()
    if not idea:
        raise ValueError("enhance_sdxl_prompts: idea is empty")

    catalogue = list_zimage_styles()
    if not (1 <= n <= len(catalogue)):
        raise ValueError(f"enhance_sdxl_prompts: n must be 1..{len(catalogue)}")

    selected = catalogue[:n]

    async def _enhance_safe(entry: dict[str, str]) -> dict[str, str] | None:
        try:
            result = await _enhance_one_sdxl(idea, entry["style"], timeout)
            return {"style": entry["style"], "name": entry["name"], **result}
        except Exception as exc:
            logger.warning("enhance_sdxl_prompts: style %s failed: %s", entry["style"], exc)
            return None

    logger.info(
        "enhance_sdxl_prompts: model=%s, idea len=%d, n=%d",
        settings.ollama_prompt_model, len(idea), n,
    )
    results = await asyncio.gather(*(_enhance_safe(entry) for entry in selected))
    successful = [r for r in results if r is not None]
    logger.info(
        "enhance_sdxl_prompts: produced %d/%d prompts (%s)",
        len(successful), n, ",".join(r["style"] for r in successful),
    )
    return successful
