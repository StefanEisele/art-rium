"""
Ollama client — local LLM/VLM calls for image analysis and article writing.

Used by:
  - services/wordpress/media.py     (alt-text + SEO metadata at upload time)
  - services/wordpress/articles.py  (multilingual blog post generation)

Models:
  OLLAMA_VLM_MODEL — vision; per-image metadata (default qwen2.5vl:latest)
  OLLAMA_LLM_MODEL — vision; multilingual article writer  (default qwen3.5:latest)

Both models must be vision-capable; the article writer needs to see the image
to honour the voice guide's "concrete first" rule.
"""
import base64
import json
import logging
from pathlib import Path
from typing import Any

import httpx

from core.config import settings

logger = logging.getLogger(__name__)

_VOICE_GUIDE_PATH = Path(__file__).resolve().parent.parent.parent / "prompts" / "voice.md"
_voice_guide_cache: str | None = None


def _voice_guide() -> str:
    """Lazy-load prompts/voice.md once per process."""
    global _voice_guide_cache
    if _voice_guide_cache is None:
        _voice_guide_cache = _VOICE_GUIDE_PATH.read_text(encoding="utf-8")
    return _voice_guide_cache


_SYSTEM_PROMPT = """You are an art-blog image analyst. Analyze the artwork image and return STRICT JSON with exactly four fields:
  seo_title: 3 to 7 words, max 60 characters. An evocative title for the artwork — concrete (subject, palette, mood), in the art-rium voice. Title Case. No clickbait, no ALL CAPS, no colons cramming two ideas, no trailing period.
  alt_text: one sentence describing what is visibly in the image, suitable for screen readers. Maximum 125 characters. Concrete (subject, composition, palette). No interpretation, no marketing language.
  seo_description: one sentence describing the artwork's mood and subject for use as a meta description on a blog post. Maximum 155 characters. Concrete imagery, no buzzwords, no superlatives.
  caption: 1-2 sentences for the WordPress media library, slightly more descriptive than alt_text, may include subject and atmosphere. Maximum 300 characters.

Use the provided title and notes as context but do not quote them verbatim. Write in the requested language. Return ONLY the JSON object — no prose, no code fences, no commentary."""


async def analyze_image(
    jpg_bytes: bytes,
    *,
    title: str | None = None,
    notes: str | None = None,
    language: str = "en",
    timeout: float = 600.0,
) -> dict[str, str]:
    """
    Run the local VLM on *jpg_bytes* and return alt_text / seo_description / caption.

    Raises:
      RuntimeError if Ollama returns an error or the model output is not valid JSON.
      httpx.HTTPError on network/timeout failures.
    """
    user_text = (
        f"Title: {title or '(none)'}\n"
        f"Notes: {notes or '(none)'}\n\n"
        f"Language for output: {language}"
    )
    payload: dict[str, Any] = {
        "model": settings.ollama_vlm_model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": user_text,
                "images": [base64.b64encode(jpg_bytes).decode("ascii")],
            },
        ],
        "format": "json",
        "stream": False,
        "options": {"temperature": 0.4},
    }

    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(f"{settings.ollama_host}/api/chat", json=payload)
        r.raise_for_status()
        data = r.json()

    if "error" in data:
        raise RuntimeError(f"Ollama error: {data['error']}")

    content = data.get("message", {}).get("content", "")
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        logger.error("VLM returned non-JSON content: %s", content[:500])
        raise RuntimeError(f"VLM returned non-JSON content: {exc}") from exc

    return {
        "seo_title":       (parsed.get("seo_title") or "").strip().rstrip(".").strip()[:60],
        "alt_text":        (parsed.get("alt_text") or "").strip()[:125],
        "seo_description": (parsed.get("seo_description") or "").strip()[:155],
        "caption":         (parsed.get("caption") or "").strip()[:300],
    }


_ARTICLE_TASK = """You are writing a single blog post about the artwork shown in the image, in three sibling languages: German (de), English (en), and 简体中文 (zh). Generate all three in one pass so the voice stays aligned — same structure, same key images observed in the artwork, same mood, idiomatic in each language (NEVER word-for-word translation).

Follow the art-rium voice guide above strictly. Every poetic move must be traceable to a concrete element actually visible in the image.

Return STRICT JSON with exactly this shape:
{
  "de": {"title": str, "body_md": str, "excerpt": str, "tags": [str, ...]},
  "en": {"title": str, "body_md": str, "excerpt": str, "tags": [str, ...]},
  "zh": {"title": str, "body_md": str, "excerpt": str, "tags": [str, ...]}
}

For each language:
  title    — 1 to 7 words, evocative, per the voice guide's "Titles" rules above. No trailing period. No quote marks.
  body_md  — the article body, ~330 words target, plain Markdown with paragraph breaks only (no headings, no bold/italic, no links, no bullets). Follow the 4-part structure from the voice guide: concrete entry → reflection/mood → series/project context → quiet close.
  excerpt  — one sentence, ≤155 characters, in the article's voice. Used as the meta description; concrete imagery, no marketing language.
  tags     — 3 to 6 short lowercase tags relevant to the artwork. Single words or short phrases. Same set across languages where it makes sense (proper nouns may differ).

Return ONLY the JSON object — no prose around it, no code fences, no commentary."""


async def write_article(
    jpg_bytes: bytes,
    *,
    title_hint: str | None = None,
    alt_text: str | None = None,
    notes: str | None = None,
    timeout: float = 1200.0,
) -> dict[str, dict[str, Any]]:
    """
    Generate DE/EN/ZH blog posts about the artwork in *jpg_bytes*.

    Returns a dict {"de": {...}, "en": {...}, "zh": {...}} where each value
    has keys: title, body_md, excerpt, tags. The model is OLLAMA_LLM_MODEL,
    which must be vision-capable.

    Raises RuntimeError if Ollama errors or the output isn't valid JSON.
    """
    system_prompt = _voice_guide() + "\n\n---\n\n" + _ARTICLE_TASK

    user_text_parts = ["Image metadata (use as context, do not quote verbatim):"]
    if title_hint:
        user_text_parts.append(f"  title hint: {title_hint}")
    if alt_text:
        user_text_parts.append(f"  alt text:   {alt_text}")
    if notes:
        user_text_parts.append(f"  notes:      {notes}")
    if len(user_text_parts) == 1:
        user_text_parts.append("  (none)")
    user_text_parts.append("\nWrite the article now. JSON only.")
    user_text = "\n".join(user_text_parts)

    payload: dict[str, Any] = {
        "model": settings.ollama_llm_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": user_text,
                "images": [base64.b64encode(jpg_bytes).decode("ascii")],
            },
        ],
        "format": "json",
        "stream": False,
        "think": False,                # thinking + format:json hangs on Qwen3-family models
        "options": {"temperature": 0.7},
    }

    logger.info("Article generation: model=%s, image=%dKB", settings.ollama_llm_model, len(jpg_bytes) // 1024)
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(f"{settings.ollama_host}/api/chat", json=payload)
        r.raise_for_status()
        data = r.json()

    if "error" in data:
        raise RuntimeError(f"Ollama error: {data['error']}")

    content = data.get("message", {}).get("content", "")
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        logger.error("Article LLM returned non-JSON content: %s", content[:500])
        raise RuntimeError(f"Article LLM returned non-JSON content: {exc}") from exc

    out: dict[str, dict[str, Any]] = {}
    for lang in ("de", "en", "zh"):
        block = parsed.get(lang) or {}
        title = (block.get("title") or "").strip().rstrip(".").strip()
        body = (block.get("body_md") or "").strip()
        excerpt = (block.get("excerpt") or "").strip()[:155]
        tags = block.get("tags") or []
        if not isinstance(tags, list):
            tags = []
        tags = [str(t).strip().lower() for t in tags if str(t).strip()][:6]

        if not title or not body:
            raise RuntimeError(f"Article LLM produced empty title/body for lang={lang}")

        out[lang] = {"title": title, "body_md": body, "excerpt": excerpt, "tags": tags}

    return out


async def reachable() -> bool:
    """Best-effort Ollama health check."""
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            r = await client.get(f"{settings.ollama_host}/api/tags")
            return r.status_code == 200
    except Exception:
        return False
