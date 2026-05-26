"""
Ollama client — local LLM/VLM calls for image analysis, article writing,
and Z-Image Turbo prompt enhancement.

Used by:
  - services/wordpress/media.py     (alt-text + SEO metadata at upload time)
  - services/wordpress/articles.py  (multilingual blog post generation)
  - routers/titler.py               (title brainstorming)
  - routers/generate.py             (z-Image enhance mode)

Models:
  OLLAMA_VLM_MODEL    — vision; per-image metadata (default qwen2.5vl:latest)
  OLLAMA_LLM_MODEL    — vision; multilingual article writer (default qwen3.6:27b)
  OLLAMA_TITLER_MODEL — vision; lightweight title brainstorming
  OLLAMA_PROMPT_MODEL — text-only; Z-Image Turbo prompt enhancer (qwen3:4b-instruct-2507)

Both LLM/VLM models for analyze/article must be vision-capable; the article
writer needs to see the image to honour the voice guide's "concrete first" rule.
"""
import asyncio
import base64
import json
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import httpx

from core.config import settings

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"


@lru_cache(maxsize=4)
def _read_prompt(filename: str) -> str:
    """Lazy-load a file from prompts/ once per process."""
    return (_PROMPTS_DIR / filename).read_text(encoding="utf-8")


# ── Generic Ollama /api/chat helper ──────────────────────────────────────────


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
    }
    if options is not None:
        payload["options"] = options
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


# ── Per-image metadata (alt-text + SEO) ──────────────────────────────────────


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
    parsed = await _chat_json(
        model=settings.ollama_vlm_model,
        system=_SYSTEM_PROMPT,
        user_text=user_text,
        jpgs=[jpg_bytes],
        options={"temperature": 0.4},
        timeout=timeout,
        label="analyze_image",
    )
    return {
        "seo_title":       (parsed.get("seo_title") or "").strip().rstrip(".").strip()[:60],
        "alt_text":        (parsed.get("alt_text") or "").strip()[:125],
        "seo_description": (parsed.get("seo_description") or "").strip()[:155],
        "caption":         (parsed.get("caption") or "").strip()[:300],
    }


# ── Per-image metadata block — shared by both article writers ────────────────


def _normalize_meta_lists(
    n: int,
    title_hints: list[str | None] | None,
    alt_texts:   list[str | None] | None,
    notes_list:  list[str | None] | None,
) -> tuple[list, list, list]:
    th = title_hints if title_hints is not None else [None] * n
    at = alt_texts   if alt_texts   is not None else [None] * n
    nl = notes_list  if notes_list  is not None else [None] * n
    if not (len(th) == len(at) == len(nl) == n):
        raise ValueError("metadata lists must be parallel to jpgs")
    return th, at, nl


def _format_image_meta_block(
    title_hints: list[str | None],
    alt_texts:   list[str | None],
    notes_list:  list[str | None],
    *,
    always_per_image_header: bool,
) -> list[str]:
    """Render the 'Image metadata' block consumed by both article prompts.

    When *always_per_image_header* is False (single-image legacy article path),
    a single-image input renders without an 'Image 1:' header — matches the
    pre-refactor user_text exactly. Multi-image always shows headers.
    """
    n = len(title_hints)
    parts: list[str] = []
    use_header = always_per_image_header or n > 1
    indent = "    " if use_header else "  "
    for i, (t, a, no) in enumerate(zip(title_hints, alt_texts, notes_list), 1):
        if use_header:
            parts.append(f"  Image {i}:")
        any_meta = False
        if t:
            parts.append(f"{indent}title hint: {t}")
            any_meta = True
        if a:
            parts.append(f"{indent}alt text:   {a}")
            any_meta = True
        if no:
            parts.append(f"{indent}notes:      {no}")
            any_meta = True
        if not any_meta:
            parts.append(f"{indent}(none)")
    return parts


# ── Voice-aligned single/series article (legacy path) ────────────────────────
# Prompt lives in prompts/voice-task.md, loaded via _read_prompt (lru_cache).


async def write_article(
    jpgs: list[bytes],
    *,
    title_hints: list[str | None] | None = None,
    alt_texts: list[str | None] | None = None,
    notes_list: list[str | None] | None = None,
    timeout: float = 1200.0,
) -> dict[str, dict[str, Any]]:
    """
    Generate DE/EN/ZH blog posts about the artwork(s) in *jpgs*.

    For a single image (len(jpgs)==1), writes an article about that piece.
    For multiple images, writes ONE article that treats them as a connected
    series — same artist, related materials/forms/palette across the set.

    Optional metadata lists are parallel to *jpgs* (one entry per image).
    Pass None for any list to omit that field across all images, or include
    None for individual images that lack the field.

    Returns {"de": {...}, "en": {...}, "zh": {...}} where each value has
    keys: title, body_md, excerpt, tags. The model is OLLAMA_LLM_MODEL,
    which must be vision-capable.

    Raises ValueError if jpgs is empty, RuntimeError on Ollama or JSON errors.
    """
    if not jpgs:
        raise ValueError("write_article requires at least one image")

    n = len(jpgs)
    title_hints, alt_texts, notes_list = _normalize_meta_lists(
        n, title_hints, alt_texts, notes_list,
    )

    parts: list[str] = []
    if n > 1:
        parts.append(
            f"You are writing a single article about a SERIES of {n} artworks. "
            f"The {n} images attached are the series — treat them as a connected "
            f"body of work, not separate pieces. Anchor paragraph 1 in one "
            f"specific image, weave concrete detail from the others through "
            f"paragraph 2, and use paragraph 3 for series-level framing "
            f"(recurring forms, palette, materials across the set)."
        )
        parts.append("")
    parts.append("Image metadata (use as context, do not quote verbatim):")
    parts.extend(_format_image_meta_block(
        title_hints, alt_texts, notes_list, always_per_image_header=False,
    ))
    parts.append("\nWrite the article now. JSON only.")
    user_text = "\n".join(parts)

    logger.info(
        "Article generation: model=%s, %d image(s), total=%dKB",
        settings.ollama_llm_model, n, sum(len(j) for j in jpgs) // 1024,
    )
    parsed = await _chat_json(
        model=settings.ollama_llm_model,
        system=_read_prompt("voice.md") + "\n\n---\n\n" + _read_prompt("voice-task.md"),
        user_text=user_text,
        jpgs=jpgs,
        options={"temperature": 0.7, "num_ctx": 8192, "num_predict": 4096},
        think=False,                # thinking + format:json hangs on Qwen3-family models
        timeout=timeout,
        label="write_article",
    )

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


# ── Rich-article writer — SEO-friendly multi-section series articles ─────────


def _rich_article_task(
    n_images: int,
    series_name: str | None,
    parent_series: dict[str, str] | None,
    has_singulart: bool,
    artist_mode: str = "third_person",
) -> str:
    """Build the rich-article task spec for the given inputs.

    Conditionally includes/omits the available_works_intro and larger_practice
    slots based on whether singulart links and a parent series are provided.
    """
    has_parent = parent_series is not None
    is_first_person = artist_mode == "first_person"

    perspective_clause = (
        "PERSPECTIVE: Write in the FIRST person of the artist (English 'I', German 'ich', Chinese 我). The voice is the artist speaking about the work — process, intent, decisions. The technical_approach slots especially benefit from first-person ('I render…', 'ich arbeite mit…', '我使用…'). Concept and visual_language can lean editorial-third-person within the same article. NEVER write 'the artist' in the first-person mode — say 'I'."
        if is_first_person else
        "PERSPECTIVE: Write in the editorial THIRD person — curatorial voice describing the work and its concerns. Refer to the artist as 'the artist' (English), 'der Künstler' (German, masculine; do NOT default to 'die Künstlerin'), or 我 the equivalent third-person framing in Chinese ('艺术家'). NEVER use 'I' or first-person pronouns for the artist."
    )

    larger_practice_field = (
        '\n  "larger_practice":      [str, ...],   // 1–2 paragraphs; one MUST contain the literal text [PARENT_SERIES] where the parent series name belongs'
        if has_parent else ""
    )
    available_works_field = (
        '\n  "available_works_intro": str,         // single sentence introducing the works-for-sale section'
        if has_singulart else ""
    )

    series_clause = (
        f"The series is titled \"{series_name}\". Use this title verbatim wherever you mention the series — do NOT translate it across languages."
        if series_name
        else "No series_name was provided. Invent a 2–4 word working title that suits the body of work; use the same title across all three languages."
    )
    parent_clause = (
        f'A parent series \"{parent_series["name"]}\" exists. In the larger_practice section, write at least one paragraph that mentions the parent series, using the literal placeholder [PARENT_SERIES] where the name belongs (the orchestrator substitutes a hyperlink). Do NOT translate the parent series name. Do NOT write a URL.'
        if has_parent else
        "No parent series was provided. Do NOT include a larger_practice slot in your output."
    )
    works_clause = (
        "Singulart product listings exist for some works in this series. In the available_works_intro slot, write a single sentence (12–25 words) that introduces the buying section — neutral and informative, no hard sell. The orchestrator renders the per-product cards."
        if has_singulart else
        "No Singulart products were provided. Do NOT include an available_works_intro slot in your output."
    )
    series_label = "series" if n_images > 1 else "single artwork (treat as a body of work even though only one piece is shown)"

    return f"""You are writing a SINGLE rich, SEO-friendly WordPress article about a {series_label} of {n_images} artwork(s), in three sibling languages: German (de), English (en), and 简体中文 (zh). Generate all three in one pass so the voice stays aligned — same structure, same observations, same key images, idiomatic in each language (NEVER word-for-word translation).

==========================================================================
CRITICAL JSON SAFETY — read before composing any string value.
==========================================================================
The output is STRICT JSON. Inside any JSON string value (every "intro", "concept" item, etc.):
- NEVER use ANY kind of double quotation mark — neither ASCII " (U+0022) nor curly „ " " (U+201E / U+201C / U+201D), nor Chinese book-title brackets 《》. ALL of these break parsing or cause downstream rendering bugs. The ONLY acceptable double-quote characters in your output are the JSON string delimiters themselves around each value.
- When you need to refer to the series name, a parent series, a quoted phrase, or anything that would normally take quotes: WRITE IT WITHOUT QUOTATION MARKS. Use the bare name. Examples:
  - WRONG (DE): „Oxidation und Fluss" ist eine Serie…
  - WRONG (DE): "Oxidation und Fluss" ist eine Serie…
  - RIGHT (DE): Oxidation und Fluss ist eine Serie…
  - RIGHT (DE): Die Serie Oxidation und Fluss untersucht…
  - WRONG (EN): "Recursive Identities" examines the loop of identity…
  - RIGHT (EN): Recursive Identities examines the loop of identity…
  - RIGHT (EN): The series Recursive Identities examines the loop…
  - WRONG (ZH): 《材料的静止》探索了…
  - RIGHT (ZH): 材料的静止 探索了…
- ASCII apostrophe ' (U+0027) is fine and encouraged for contractions: don't, the artist's, l'œuvre.
- The bare title is unambiguous because:
  (a) the article has a dedicated `title` field that names the series formally, and
  (b) the renderer can typeset it as italic / bold / link from context — quote marks are not needed for clarity.
- All JSON keys remain in ASCII (e.g. "intro", "concept") — only the *values* must avoid double quotes.

{perspective_clause}

{series_clause}

{parent_clause}

{works_clause}

Follow the voice-rich guide above strictly. Every claim about the series must be traceable to recurring observations across the images.

==========================================================================
OUTPUT SHAPE
==========================================================================

Return STRICT JSON with exactly this top-level shape:
{{
  "de": {{ language block }},
  "en": {{ language block }},
  "zh": {{ language block }}
}}

Each language block contains EXACTLY these fields, in this order:
{{
  "title":                str,                  // 2–6 words; equals the series name when one is provided
  "intro":                str,                  // opening paragraph, 60–100 words; names the series and states the thesis
  "concept":              [str, ...],           // 1–2 paragraphs; total 100–160 words
  "visual_language":      [str, ...],           // 1–2 paragraphs; total 100–160 words
  "technical_approach":   {{
    "intro": str,                               // 1 paragraph, 40–70 words; names the tool stack
    "steps": [str, ...],                        // 2–4 short bullet items (10–20 words each); concrete process steps
    "outro": str                                // 1 paragraph, 30–60 words; what the workflow enables
  }},{available_works_field}{larger_practice_field}
  "excerpt":              str,                  // ≤155 characters; the WP post excerpt — voice-faithful, names the series and one concrete detail
  "meta_description":     str,                  // 130–155 characters; Yoast SEO snippet — search-active phrasing, contains the focus_keyphrase verbatim, distinct sentence from excerpt
  "focus_keyphrase":      str,                  // 2–4 lowercase words; SEO target phrase, localised per language. Must appear verbatim in intro, meta_description, and at least one body slot. NEVER the bare series title alone.
  "tags":                 [str, ...]            // 3–6 short lowercase tags
}}

Hard rules carried over from the voice-rich guide:
- NO hedging verbs (seems / appears / wirkt / scheint).
- NO precious adverbs (etwas / fast / somewhat / slightly).
- NO AI-marketing buzzwords (harnessing, next-generation, cutting-edge, revolutionary, bahnbrechend, atemberaubend, stunning).
- NO exclamation marks. NO hashtags. NO bold/italic decoration in the prose.
- Every paragraph in concept / visual_language / technical_approach must be plain prose — NO embedded headings, NO embedded bullet lists. Bullets appear ONLY in technical_approach.steps.
- NO raw URLs anywhere. Links flow only through the [PARENT_SERIES] placeholder when a parent series is provided.

Return ONLY the JSON object — no prose around it, no code fences, no commentary."""


# Last-resort JSON salvage for the qwen3.6 quote failure modes.
# Despite the prompt forbidding all double quotes inside string values, the
# model occasionally regresses and emits things like:
#   "intro": "„Oxidation und Fluss" ist eine Serie…"   (DE low-quote pair, ASCII close)
#   "intro": ""Recursive Identities" examines…"        (ASCII pair around emphasized phrase)
# The DE pattern is unambiguous (U+201E never appears outside string content)
# so we can replace its ASCII close with U+201D safely. The EN-style ASCII
# pair is harder — we conservatively only fix the case where an ASCII quote
# directly follows another ASCII quote that started a JSON value (i.e., `: ""`).
_CURLY_QUOTE_FIX_DE = re.compile(r'„([^"\n]{1,200}?)"')


def _salvage_curly_quotes(s: str) -> str:
    """Convert ASCII closing-quotes after a German „ opener to U+201D
    so the JSON parser stops treating them as string terminators."""
    return _CURLY_QUOTE_FIX_DE.sub(lambda m: f"„{m.group(1)}”", s)


def _iterative_salvage(s: str, max_passes: int = 6) -> str:
    """Apply the salvage repeatedly. Each pass may expose new patterns that
    the previous pass un-blocked (nested DE quotes, multiple quoted phrases
    in the same JSON value, etc.). Stops when a pass produces no change."""
    cur = s
    for _ in range(max_passes):
        nxt = _salvage_curly_quotes(cur)
        if nxt == cur:
            return cur
        cur = nxt
    return cur


async def write_rich_article(
    jpgs: list[bytes],
    *,
    series_name: str | None = None,
    parent_series: dict[str, str] | None = None,  # {"name": str, "url": str}
    has_singulart: bool = False,
    title_hints: list[str | None] | None = None,
    alt_texts:   list[str | None] | None = None,
    notes_list:  list[str | None] | None = None,
    user_notes:  str | None = None,
    artist_mode: str = "third_person",            # "first_person" | "third_person"
    timeout: float = 1800.0,
) -> dict[str, dict[str, Any]]:
    """
    Generate DE/EN/ZH rich, SEO-friendly series articles about *jpgs*.

    Returns a {"de": {...}, "en": {...}, "zh": {...}} dict where each value
    has the slot keys: title, intro, concept, visual_language,
    technical_approach{intro,steps,outro}, excerpt, tags — plus
    available_works_intro (when has_singulart=True) and
    larger_practice (when parent_series is set).

    Used by services/wordpress/articles.py:generate_rich_articles_for_series
    which stitches Gutenberg blocks (H2 headings, paragraphs, galleries,
    Singulart product cards, inline parent-series anchor) around the prose.
    """
    if not jpgs:
        raise ValueError("write_rich_article requires at least one image")
    if parent_series is not None and not (parent_series.get("name") and parent_series.get("url")):
        raise ValueError("parent_series must have both 'name' and 'url' keys")

    n = len(jpgs)
    title_hints, alt_texts, notes_list = _normalize_meta_lists(
        n, title_hints, alt_texts, notes_list,
    )

    task_spec = _rich_article_task(n, series_name, parent_series, has_singulart, artist_mode)
    system_prompt = _read_prompt("voice-rich.md") + "\n\n---\n\n" + task_spec

    parts: list[str] = []
    if series_name:
        parts.append(f'Series name: "{series_name}"')
    if parent_series:
        parts.append(f'Parent series: "{parent_series["name"]}" (use the [PARENT_SERIES] placeholder; do NOT write the URL)')
    if has_singulart:
        parts.append("Singulart products: YES (orchestrator renders the cards; you write only the available_works_intro sentence)")
    if user_notes and user_notes.strip():
        parts.append("")
        parts.append("Artist's intent / context for this article (anchor the prose in these specifics — do not quote verbatim):")
        parts.append(user_notes.strip())
    parts.append("")
    parts.append(f"Number of images attached: {n}")
    parts.append("")
    parts.append("Per-image metadata (use as context, do not quote verbatim):")
    parts.extend(_format_image_meta_block(
        title_hints, alt_texts, notes_list, always_per_image_header=True,
    ))
    parts.append("\nWrite the article now. JSON only.")
    user_text = "\n".join(parts)

    logger.info(
        "Rich article: model=%s, %d image(s), %dKB total, series=%s, parent=%s, singulart=%s",
        settings.ollama_llm_model, n, sum(len(j) for j in jpgs) // 1024,
        series_name or "(none)",
        parent_series["name"] if parent_series else "(none)",
        has_singulart,
    )

    parsed = await _chat_json(
        model=settings.ollama_llm_model,
        system=system_prompt,
        user_text=user_text,
        jpgs=jpgs,
        options={"temperature": 0.65, "num_ctx": 16384, "num_predict": 6000},
        think=False,                # thinking + format:json hangs on Qwen3-family models
        timeout=timeout,
        label="write_rich_article",
        salvage=_iterative_salvage,
        max_attempts=3,             # qwen3.6 is non-deterministic; fresh samples often parse cleanly
    )

    out: dict[str, dict[str, Any]] = {}
    for lang in ("de", "en", "zh"):
        block = parsed.get(lang) or {}
        title = (block.get("title") or "").strip().rstrip(".").strip()
        intro = (block.get("intro") or "").strip()
        concept = block.get("concept") or []
        visual_language = block.get("visual_language") or []
        technical_approach = block.get("technical_approach") or {}
        ta_intro = (technical_approach.get("intro") or "").strip()
        ta_steps = technical_approach.get("steps") or []
        ta_outro = (technical_approach.get("outro") or "").strip()
        available_works_intro = (block.get("available_works_intro") or "").strip() if has_singulart else ""
        larger_practice = block.get("larger_practice") or [] if parent_series else []
        excerpt = (block.get("excerpt") or "").strip()[:155]
        meta_description = (block.get("meta_description") or excerpt).strip()[:155]
        focus_keyphrase = (block.get("focus_keyphrase") or "").strip().lower()
        tags = block.get("tags") or []

        if not isinstance(concept, list):
            concept = [str(concept)]
        if not isinstance(visual_language, list):
            visual_language = [str(visual_language)]
        if not isinstance(ta_steps, list):
            ta_steps = []
        if not isinstance(larger_practice, list):
            larger_practice = [str(larger_practice)]
        if not isinstance(tags, list):
            tags = []

        concept = [str(p).strip() for p in concept if str(p).strip()]
        visual_language = [str(p).strip() for p in visual_language if str(p).strip()]
        ta_steps = [str(s).strip() for s in ta_steps if str(s).strip()][:4]
        larger_practice = [str(p).strip() for p in larger_practice if str(p).strip()]
        tags = [str(t).strip().lower() for t in tags if str(t).strip()][:6]

        if not title or not intro:
            raise RuntimeError(f"Rich-article LLM produced empty title/intro for lang={lang}")
        if not concept or not visual_language:
            raise RuntimeError(f"Rich-article LLM missing concept/visual_language for lang={lang}")
        if not ta_intro or not ta_steps:
            raise RuntimeError(f"Rich-article LLM missing technical_approach for lang={lang}")
        if parent_series and not larger_practice:
            raise RuntimeError(f"Rich-article LLM missing larger_practice for lang={lang} despite parent_series")

        out[lang] = {
            "title":                 title,
            "intro":                 intro,
            "concept":               concept,
            "visual_language":       visual_language,
            "technical_approach":    {"intro": ta_intro, "steps": ta_steps, "outro": ta_outro},
            "available_works_intro": available_works_intro,
            "larger_practice":       larger_practice,
            "excerpt":               excerpt,
            "meta_description":      meta_description,
            "focus_keyphrase":       focus_keyphrase,
            "tags":                  tags,
        }

    return out


# ── Modal article writer — Essay / Work / Lab (EN + DE) ──────────────────────
# Replaces the trilingual rich-article path. Loads voice-system.md (universal
# voice) + mode-{mode}.md (mode-specific task spec) and validates the slot
# shape per mode. Existing `write_article` / `write_rich_article` stay around
# for back-compat but are no longer routed to from the Articles tool.


_MODAL_LANGS = ("en", "de")
_MODAL_MODES = ("essay", "work", "lab")

# Serialise modal article generation. Two concurrent jobs hitting qwen3.6:27b
# with num_ctx=24576 + 9 images each crash Ollama's model-runner subprocess
# (KV-cache allocation fails). One generation at a time; everyone else waits.
_MODAL_ARTICLE_LOCK = asyncio.Lock()


def _modal_user_text(
    *,
    mode: str,
    n_images: int,
    series_name: str | None,
    parent_series: dict[str, str] | None,
    has_singulart: bool,
    user_notes: str | None,
    artist_mode: str,
    title_hints: list[str | None],
    alt_texts:   list[str | None],
    notes_list:  list[str | None],
    video_descriptions: list[str] | None = None,
    video_frame_index_ranges: list[tuple[int, int]] | None = None,
) -> str:
    """Build the user-message block for the modal article writer."""
    parts: list[str] = [f"MODE: {mode}", ""]

    if mode == "work":
        # Work mode is the only one that uses series_name / parent_series /
        # has_singulart / artist_mode meaningfully.
        if series_name:
            parts.append(f"Series name: {series_name}")
        else:
            parts.append("No series name provided. Invent a 2–6 word working title; use the same title across en and de.")
        if parent_series:
            parts.append(
                f"Parent series: {parent_series['name']} (use the [PARENT_SERIES] placeholder in larger_practice; "
                f"do NOT write the URL; do NOT translate the parent name)."
            )
        else:
            parts.append("No parent series. Set larger_practice to null.")
        if has_singulart:
            parts.append("Singulart products: YES. Write a single available_works_intro sentence (12–25 words).")
        else:
            parts.append("No Singulart products. Set available_works_intro to null.")
        parts.append(
            "Perspective: "
            + ("FIRST person (the artist speaking)." if artist_mode == "first_person"
               else "THIRD person (editorial / curatorial).")
        )
    elif mode == "essay":
        parts.append("Perspective: first person — the artist's own argument. Cite real sources with date + venue only.")
        if series_name:
            parts.append(f"Optional anchoring series: {series_name} (mention only if it serves the thesis).")
    elif mode == "lab":
        parts.append("Perspective: first-person workmanlike. Peer-to-peer with ComfyUI / generative-art practitioners.")
        if series_name:
            parts.append(f"Workflow context name: {series_name} (use as anchor for naming the technique).")
    else:
        raise ValueError(f"Unknown mode: {mode!r}; expected one of {_MODAL_MODES}")

    if user_notes and user_notes.strip():
        parts.append("")
        parts.append("Author's intent / context for this article (anchor the prose in these specifics — do not quote verbatim):")
        parts.append(user_notes.strip())

    parts.append("")
    parts.append(f"Number of images attached: {n_images}")
    parts.append("")
    parts.append("Per-image metadata (use as context, do not quote verbatim):")
    parts.extend(_format_image_meta_block(
        title_hints, alt_texts, notes_list, always_per_image_header=True,
    ))

    if video_descriptions:
        parts.append("")
        parts.append(f"Number of videos attached for embedding: {len(video_descriptions)}")
        if video_frame_index_ranges:
            parts.append(
                f"After the {n_images} gallery images above, additional images in this message "
                f"are sample frames extracted from the videos at even intervals through each clip "
                f"(in playback order). Use them to describe what the viewer will actually see — "
                f"colors, composition, motion arc, hands/keys for piano clips. Do NOT invent motion "
                f"or content the frames don't show."
            )
        parts.append("Video manifest (index → kind/description). Each [VIDEO_K] must be placed in the")
        parts.append("prose EXACTLY ONCE on its own paragraph line — the renderer substitutes it with")
        parts.append("the YouTube embed at that position. Place each video where the prose naturally")
        parts.append("introduces, pauses on, or extends what the video shows. Do NOT quote the [VIDEO_K]")
        parts.append("token in surrounding text; do NOT write 'see [VIDEO_1] below'; just emit the bare")
        parts.append("token as its own paragraph line. Use ALL videos. Same placement in EN and DE.")
        for i, desc in enumerate(video_descriptions):
            line = f"  {desc}"
            if video_frame_index_ranges:
                start, end = video_frame_index_ranges[i] if i < len(video_frame_index_ranges) else (0, 0)
                if start and end and end >= start:
                    n_frames = end - start + 1
                    line += f" — sample frames: images {start}–{end} ({n_frames} frame{'s' if n_frames != 1 else ''})"
                else:
                    line += " — (no frame samples available)"
            parts.append(line)

    parts.append("")
    parts.append("Write the article now. JSON only.")
    return "\n".join(parts)


def _strip_str(v: Any, *, max_chars: int | None = None) -> str:
    s = (str(v) if v is not None else "").strip()
    return s[:max_chars] if max_chars else s


def _strip_list(v: Any, *, max_items: int | None = None, lower: bool = False) -> list[str]:
    if not isinstance(v, list):
        return []
    out: list[str] = []
    for item in v:
        s = (str(item) if item is not None else "").strip()
        if not s:
            continue
        if lower:
            s = s.lower()
        out.append(s)
    return out[:max_items] if max_items else out


def _clamp_focus_keyphrase(raw: str, *, max_words: int = 4) -> str:
    """Trim a focus_keyphrase to ≤max_words. The LLM occasionally returns a
    full title-sized phrase (6+ words) which Yoast flags as too long. We
    keep the first max_words tokens — usually still semantically useful
    since the model puts the most distinctive words first."""
    s = (raw or "").strip().lower()
    if not s:
        return ""
    words = s.split()
    return " ".join(words[:max_words])


def _truncate_at_word(text: str, max_chars: int) -> str:
    """Truncate text to ≤max_chars, cutting at the last whitespace inside
    the budget so we don't end mid-word. Falls back to a hard slice when
    no whitespace exists in the budget."""
    s = (text or "").strip()
    if len(s) <= max_chars:
        return s
    head = s[:max_chars]
    last_space = head.rfind(" ")
    if last_space > max_chars * 0.6:   # only trust a word-break if it's not too far back
        return head[:last_space].rstrip(",;:.- \t")
    return head


def _count_substring(haystack: str, needle: str) -> int:
    """Case-insensitive non-overlapping substring count."""
    if not needle:
        return 0
    return haystack.lower().count(needle.lower())


def _validate_essay_block(lang: str, block: dict) -> dict:
    title    = _strip_str(block.get("title"), max_chars=180).rstrip(".").strip()
    intro    = _strip_list(block.get("intro"))
    movements_raw = block.get("movements") or []
    closing  = _strip_str(block.get("closing"))
    excerpt  = _truncate_at_word(_strip_str(block.get("excerpt")), 150)
    meta_desc_raw = _strip_str(block.get("meta_description")) or excerpt
    meta_desc = _truncate_at_word(meta_desc_raw, 150)
    focus_kp = _clamp_focus_keyphrase(_strip_str(block.get("focus_keyphrase")))
    tags     = _strip_list(block.get("tags"), max_items=6, lower=True)
    og_idea  = _strip_str(block.get("og_image_idea"))

    if not title or not intro:
        raise RuntimeError(f"Essay LLM produced empty title/intro for lang={lang}")
    if not isinstance(movements_raw, list) or not movements_raw:
        raise RuntimeError(f"Essay LLM produced empty movements for lang={lang}")

    movements: list[dict] = []
    for i, mv in enumerate(movements_raw, 1):
        if not isinstance(mv, dict):
            continue
        heading = _strip_str(mv.get("heading"))
        body    = _strip_list(mv.get("body"))
        if not heading or not body:
            raise RuntimeError(f"Essay LLM produced movement {i} without heading/body for lang={lang}")
        movements.append({"heading": heading, "body": body})

    _log_essay_seo_placement(lang, focus_kp, title, intro, movements, closing, meta_desc)
    _log_essay_readability(lang, intro, movements, closing)

    return {
        "title":            title,
        "intro":            intro,
        "movements":        movements,
        "closing":          closing,
        "excerpt":          excerpt,
        "meta_description": meta_desc,
        "focus_keyphrase":  focus_kp,
        "tags":             tags,
        "og_image_idea":    og_idea,
    }


def _log_essay_seo_placement(
    lang: str,
    focus_kp: str,
    title: str,
    intro: list[str],
    movements: list[dict],
    closing: str,
    meta_desc: str,
) -> None:
    """Diagnose Yoast keyphrase placement and log warnings. Non-fatal —
    the post still publishes, but the operator sees what slipped through
    the prompt rules and can regenerate if it matters."""
    if not focus_kp:
        logger.warning("Essay SEO[%s]: focus_keyphrase missing", lang)
        return

    issues: list[str] = []

    if not title.lower().startswith(focus_kp.lower()):
        issues.append(f"title does not start with keyphrase (title={title!r})")

    intro_p1 = intro[0] if intro else ""
    if _count_substring(intro_p1, focus_kp) == 0:
        issues.append("keyphrase not in intro[0]")

    if _count_substring(meta_desc, focus_kp) == 0:
        issues.append("keyphrase not in meta_description")

    heading_hits = sum(1 for mv in movements if _count_substring(mv["heading"], focus_kp))
    if heading_hits == 0:
        issues.append("keyphrase not in any movement heading")

    body_text = "\n".join(
        [*intro] +
        [p for mv in movements for p in mv["body"]] +
        ([closing] if closing else [])
    )
    density = _count_substring(body_text, focus_kp)
    if density < 3:
        issues.append(f"keyphrase density {density} (<3) across body")

    if issues:
        logger.warning(
            "Essay SEO[%s] focus_keyphrase=%r: %s",
            lang, focus_kp, "; ".join(issues),
        )
    else:
        logger.info("Essay SEO[%s] focus_keyphrase=%r: placement OK", lang, focus_kp)


# Yoast-compatible transition wordlists. Single-word entries are matched on
# word boundaries; multi-word entries are matched as substrings (lowercase).
# Kept in sync with the lists in prompts/voice-system.md.
_TRANSITIONS_EN = {
    "but", "yet", "still", "however", "instead", "so", "then", "because",
    "while", "after", "before", "since", "until", "although", "despite",
    "if", "when", "also", "even", "only", "just", "and", "as", "where",
    "whereas", "whereby", "unless",
}
_TRANSITIONS_EN_MULTI = ("in fact", "so that", "such that")

_TRANSITIONS_DE = {
    "aber", "doch", "dennoch", "trotzdem", "deshalb", "daher", "weil",
    "zwar", "schließlich", "allerdings", "sondern", "obwohl", "während",
    "indem", "sodass", "damit", "falls", "sobald", "statt", "dafür",
    "noch", "auch", "denn", "also", "eben", "nur", "und", "da", "wenn",
    "als", "solange", "bevor", "nachdem", "bis",
}
_TRANSITIONS_DE_MULTI: tuple[str, ...] = ()


def _split_sentences(text: str) -> list[str]:
    """Cheap sentence splitter: cut on .!? followed by whitespace. Good
    enough for German + English prose; ignores abbreviation edge cases
    which Yoast's checker also handles only approximately."""
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return [p for p in (s.strip() for s in parts) if p]


def _sentence_has_transition(sentence: str, lang: str) -> bool:
    s = sentence.lower()
    multi = _TRANSITIONS_EN_MULTI if lang == "en" else _TRANSITIONS_DE_MULTI
    for phrase in multi:
        if phrase in s:
            return True
    words = re.findall(r"\b[\wäöüß']+\b", s, flags=re.UNICODE)
    bag = _TRANSITIONS_EN if lang == "en" else _TRANSITIONS_DE
    return any(w in bag for w in words)


def _first_word(sentence: str) -> str:
    m = re.match(r"\s*([\wäöüß']+)", sentence, flags=re.UNICODE)
    return m.group(1).lower() if m else ""


def _log_essay_readability(
    lang: str,
    intro: list[str],
    movements: list[dict],
    closing: str,
) -> None:
    """Mirror Yoast's readability heuristics — transition density,
    consecutive-start runs, and movement length. Logs warnings only; the
    post still publishes. Operator can decide whether to regenerate."""
    issues: list[str] = []

    # ── Movement length: cap is 250 words (matches prompt). Yoast flags at
    # 300 but our prompt asks the model to self-cap at 250, so we warn at
    # both thresholds with different severities.
    for i, mv in enumerate(movements, 1):
        wc = sum(len(p.split()) for p in mv.get("body", []))
        if wc > 300:
            issues.append(
                f"movement {i} ({mv.get('heading')!r}) is {wc} words "
                "(>300; Yoast will flag — needs a split)"
            )
        elif wc > 250:
            issues.append(
                f"movement {i} ({mv.get('heading')!r}) is {wc} words "
                "(>250 prompt cap; consider a split)"
            )

    # ── Build the flat sentence list for transitions + consecutive-start.
    all_paragraphs: list[str] = []
    all_paragraphs.extend(intro)
    for mv in movements:
        all_paragraphs.extend(mv.get("body", []))
    if closing:
        all_paragraphs.append(closing)
    sentences = [s for p in all_paragraphs for s in _split_sentences(p)]

    # ── Transition density (Yoast wants ≥30%).
    n_sentences = len(sentences) or 1
    transition_hits = sum(1 for s in sentences if _sentence_has_transition(s, lang))
    transition_pct = round(100.0 * transition_hits / n_sentences, 1)
    if transition_pct < 30.0:
        issues.append(
            f"transitions {transition_pct}% ({transition_hits}/{n_sentences}; Yoast wants ≥30%)"
        )

    # ── Consecutive-start runs (Yoast flags 3+ in a row sharing an opener).
    runs: list[tuple[str, int]] = []
    prev_word = ""
    run_len = 0
    for s in sentences:
        w = _first_word(s)
        if w and w == prev_word:
            run_len += 1
        else:
            if run_len >= 3:
                runs.append((prev_word, run_len))
            prev_word = w
            run_len = 1
    if run_len >= 3:
        runs.append((prev_word, run_len))
    if runs:
        issues.append(
            f"{len(runs)} consecutive-start run(s) of 3+ sentences: "
            + ", ".join(f"{w!r}×{n}" for w, n in runs[:5])
        )

    if issues:
        logger.warning("Essay readability[%s]: %s", lang, "; ".join(issues))
    else:
        logger.info("Essay readability[%s]: OK (transitions %s%%)", lang, transition_pct)


def _validate_work_block(
    lang: str, block: dict, *, has_singulart: bool, has_parent: bool,
) -> dict:
    title         = _strip_str(block.get("title"), max_chars=180).rstrip(".").strip()
    opening       = _strip_str(block.get("opening_line"))
    body          = _strip_list(block.get("body"))
    excerpt       = _truncate_at_word(_strip_str(block.get("excerpt")), 150)
    meta_desc_raw = _strip_str(block.get("meta_description")) or excerpt
    meta_desc     = _truncate_at_word(meta_desc_raw, 150)
    focus_kp      = _clamp_focus_keyphrase(_strip_str(block.get("focus_keyphrase")))
    tags          = _strip_list(block.get("tags"), max_items=6, lower=True)
    og_idea       = _strip_str(block.get("og_image_idea"))
    metadata_raw  = block.get("metadata") or {}

    aw_intro: str | None = None
    if has_singulart:
        aw_intro = _strip_str(block.get("available_works_intro")) or None

    larger: list[str] | None = None
    if has_parent:
        larger = _strip_list(block.get("larger_practice"))
        if not larger:
            raise RuntimeError(f"Work LLM missing larger_practice for lang={lang} despite parent_series")

    if not title or not opening or not body:
        raise RuntimeError(f"Work LLM produced empty title/opening/body for lang={lang}")

    metadata = {
        "year":       _strip_str(metadata_raw.get("year"))      or "",
        "medium":     _strip_str(metadata_raw.get("medium"))    or "",
        "dimensions": _strip_str(metadata_raw.get("dimensions")) or "",
        "edition":    _strip_str(metadata_raw.get("edition"))   or "",
        "status":     _strip_str(metadata_raw.get("status"))    or "",
    }

    return {
        "title":                 title,
        "opening_line":          opening,
        "body":                  body,
        "available_works_intro": aw_intro,
        "larger_practice":       larger,
        "metadata":              metadata,
        "excerpt":               excerpt,
        "meta_description":      meta_desc,
        "focus_keyphrase":       focus_kp,
        "tags":                  tags,
        "og_image_idea":         og_idea,
    }


_LAB_CODE_LANGS = {"bash", "python", "json", "yaml", "text", "shell", "sh", "js", "javascript", "ts", "typescript"}


def _validate_lab_block(lang: str, block: dict) -> dict:
    title        = _strip_str(block.get("title"), max_chars=180).rstrip(".").strip()
    problem      = _strip_str(block.get("problem"))
    solution     = _strip_str(block.get("solution_intro"))
    tool_stack   = _strip_list(block.get("tool_stack"), max_items=10, lower=True)
    hardware     = block.get("hardware_context")
    hardware_s   = _strip_str(hardware) if hardware else ""
    steps        = _strip_list(block.get("steps"), max_items=10)
    result       = _strip_str(block.get("result"))
    why          = _strip_str(block.get("why_it_matters"))
    excerpt      = _truncate_at_word(_strip_str(block.get("excerpt")), 150)
    meta_desc_raw = _strip_str(block.get("meta_description")) or excerpt
    meta_desc    = _truncate_at_word(meta_desc_raw, 150)
    focus_kp     = _clamp_focus_keyphrase(_strip_str(block.get("focus_keyphrase")))
    tags         = _strip_list(block.get("tags"), max_items=6, lower=True)
    og_idea      = _strip_str(block.get("og_image_idea"))

    code_raw     = block.get("code_blocks") or []
    code_blocks: list[dict] = []
    if isinstance(code_raw, list):
        for cb in code_raw[:8]:
            if not isinstance(cb, dict):
                continue
            code = _strip_str(cb.get("code"))
            lang_tag = _strip_str(cb.get("language")).lower()
            if not code:
                continue
            if lang_tag not in _LAB_CODE_LANGS:
                lang_tag = "text"
            cap_v = cb.get("caption")
            caption = _strip_str(cap_v) if cap_v else ""
            code_blocks.append({"language": lang_tag, "code": code, "caption": caption})

    if not title or not problem or not solution or not steps:
        raise RuntimeError(f"Lab LLM produced empty title/problem/solution/steps for lang={lang}")

    return {
        "title":             title,
        "problem":           problem,
        "solution_intro":    solution,
        "tool_stack":        tool_stack,
        "hardware_context":  hardware_s or None,
        "steps":             steps,
        "code_blocks":       code_blocks,
        "result":            result,
        "why_it_matters":    why,
        "excerpt":           excerpt,
        "meta_description":  meta_desc,
        "focus_keyphrase":   focus_kp,
        "tags":              tags,
        "og_image_idea":     og_idea,
    }


# ── DE revision pass — fix EN→DE carry-over after the main bilingual call ───

_PLACEHOLDER_VIDEO_RE = re.compile(r"\[VIDEO_\d+\]")


def _count_placeholders(block: dict) -> tuple[int, int]:
    """Count [VIDEO_K] and [PARENT_SERIES] occurrences anywhere in a block.
    Serializes via json so nested arrays/dicts are covered uniformly."""
    s = json.dumps(block, ensure_ascii=False)
    return (len(_PLACEHOLDER_VIDEO_RE.findall(s)), s.count("[PARENT_SERIES]"))


async def _revise_german_block(
    de_block: dict,
    *,
    mode: str,
    has_singulart: bool,
    has_parent: bool,
    timeout: float = 600.0,
) -> dict | None:
    """
    Second pass: rewrite the German block to remove anti-translation tells
    that appear when EN+DE are generated in the same model call.

    Returns the revised block on success, or None on any failure — caller
    falls back to the original DE block.

    Invariants enforced (revision is rejected if violated):
      - focus_keyphrase byte-equal to original (Yoast target locked).
      - [VIDEO_K] and [PARENT_SERIES] placeholder counts unchanged.
      - metadata{} (Work mode) restored verbatim from original.
      - code_blocks[i].code and .language (Lab mode) restored verbatim;
        only captions may be revised.
    """
    original_focus_kp = de_block.get("focus_keyphrase", "")
    original_metadata = de_block.get("metadata") if mode == "work" else None
    original_code_blocks = de_block.get("code_blocks") if mode == "lab" else None
    orig_video_count, orig_parent_count = _count_placeholders(de_block)

    user_text = (
        f"MODE: {mode}\n\n"
        "Hier ist der zu überarbeitende deutsche Sprachblock als JSON. "
        "Überarbeite die Prosa idiomatisch nach den Regeln im System-Prompt. "
        "Gib EXAKT dieselben JSON-Schlüssel zurück — keine entfernten, keine neuen. "
        "Behalte focus_keyphrase, metadata, code_blocks[].code/.language und alle "
        "[VIDEO_K] / [PARENT_SERIES] Platzhalter unverändert.\n\n"
        f"{json.dumps(de_block, ensure_ascii=False, indent=2)}\n\n"
        "Antworte mit dem überarbeiteten JSON-Objekt — kein Vorspann, keine Fences."
    )

    try:
        revised = await _chat_json(
            model=settings.ollama_llm_model,
            system=_read_prompt("de-revision.md"),
            user_text=user_text,
            options={"temperature": 0.5, "num_ctx": 16384, "num_predict": 6000},
            think=False,
            timeout=timeout,
            label=f"revise_german[{mode}]",
            salvage=_iterative_salvage,
            max_attempts=2,
        )
    except Exception as exc:
        logger.warning("DE revision pass failed (%s) — keeping original DE block", exc)
        return None

    try:
        if mode == "essay":
            revised = _validate_essay_block("de", revised)
        elif mode == "work":
            revised = _validate_work_block(
                "de", revised, has_singulart=has_singulart, has_parent=has_parent,
            )
        else:
            revised = _validate_lab_block("de", revised)
    except RuntimeError as exc:
        logger.warning("DE revision returned invalid block (%s) — keeping original", exc)
        return None

    if revised.get("focus_keyphrase") != original_focus_kp:
        logger.warning(
            "DE revision changed focus_keyphrase (%r → %r) — keeping original",
            original_focus_kp, revised.get("focus_keyphrase"),
        )
        return None

    rev_video_count, rev_parent_count = _count_placeholders(revised)
    if rev_video_count != orig_video_count:
        logger.warning(
            "DE revision changed [VIDEO_K] count (%d → %d) — keeping original",
            orig_video_count, rev_video_count,
        )
        return None
    if rev_parent_count != orig_parent_count:
        logger.warning(
            "DE revision changed [PARENT_SERIES] count (%d → %d) — keeping original",
            orig_parent_count, rev_parent_count,
        )
        return None

    # Metadata + code blocks are data, not prose — restore verbatim defensively
    # even if the model already left them alone.
    if mode == "work" and original_metadata is not None:
        revised["metadata"] = original_metadata

    if mode == "lab" and original_code_blocks is not None:
        revised_blocks = revised.get("code_blocks") or []
        if len(revised_blocks) != len(original_code_blocks):
            logger.warning(
                "DE revision changed code_blocks count (%d → %d) — keeping original",
                len(original_code_blocks), len(revised_blocks),
            )
            return None
        for orig_cb, new_cb in zip(original_code_blocks, revised_blocks):
            if isinstance(new_cb, dict) and isinstance(orig_cb, dict):
                new_cb["code"] = orig_cb.get("code", "")
                new_cb["language"] = orig_cb.get("language", "text")

    logger.info("DE revision pass succeeded for mode=%s", mode)
    return revised


async def write_modal_article(
    jpgs: list[bytes],
    *,
    mode: str,                                                   # "essay" | "work" | "lab"
    series_name: str | None = None,
    parent_series: dict[str, str] | None = None,                  # {"name": str, "url": str} — Work mode only
    has_singulart: bool = False,                                  # Work mode only
    title_hints: list[str | None] | None = None,
    alt_texts:   list[str | None] | None = None,
    notes_list:  list[str | None] | None = None,
    user_notes:  str | None = None,
    artist_mode: str = "third_person",                            # Work mode only ("first_person" | "third_person")
    video_descriptions: list[str] | None = None,                  # one entry per [VIDEO_K]; embeds via wp:embed
    video_frame_jpgs: list[list[bytes]] | None = None,            # per-video sample frames; concatenated after gallery jpgs
    timeout: float = 1800.0,
) -> dict[str, dict[str, Any]]:
    """
    Generate an EN+DE article in one of three modes (essay / work / lab).

    Returns {"en": {...}, "de": {...}} where each value has the slot shape
    defined by prompts/mode-{mode}.md and validated by _validate_*_block.

    Voice alignment is preserved by generating both languages in a single
    model call. The system prompt is voice-system.md + mode-{mode}.md.

    Raises:
      ValueError on unknown mode or empty jpgs.
      RuntimeError if Ollama returns an error / non-JSON / missing required slots.
      httpx.HTTPError on network / timeout failures.
    """
    if mode not in _MODAL_MODES:
        raise ValueError(f"write_modal_article: mode must be one of {_MODAL_MODES}, got {mode!r}")
    if not jpgs:
        raise ValueError("write_modal_article requires at least one image")
    if parent_series is not None and not (parent_series.get("name") and parent_series.get("url")):
        raise ValueError("parent_series must have both 'name' and 'url' keys")

    if _MODAL_ARTICLE_LOCK.locked():
        logger.info("Modal article: another generation in progress, queueing (mode=%s)", mode)
    async with _MODAL_ARTICLE_LOCK:
        return await _write_modal_article_locked(
            jpgs,
            mode=mode,
            series_name=series_name,
            parent_series=parent_series,
            has_singulart=has_singulart,
            title_hints=title_hints,
            alt_texts=alt_texts,
            notes_list=notes_list,
            user_notes=user_notes,
            artist_mode=artist_mode,
            video_descriptions=video_descriptions,
            video_frame_jpgs=video_frame_jpgs,
            timeout=timeout,
        )


async def _write_modal_article_locked(
    jpgs: list[bytes],
    *,
    mode: str,
    series_name: str | None,
    parent_series: dict[str, str] | None,
    has_singulart: bool,
    title_hints: list[str | None] | None,
    alt_texts:   list[str | None] | None,
    notes_list:  list[str | None] | None,
    user_notes:  str | None,
    artist_mode: str,
    video_descriptions: list[str] | None,
    video_frame_jpgs: list[list[bytes]] | None,
    timeout: float,
) -> dict[str, dict[str, Any]]:
    """Body of write_modal_article, running under _MODAL_ARTICLE_LOCK.
    All Ollama I/O for one article (main pass + DE revision) runs here."""
    n = len(jpgs)
    title_hints, alt_texts, notes_list = _normalize_meta_lists(
        n, title_hints, alt_texts, notes_list,
    )

    # Combine gallery images + per-video sample frames into a single flat list
    # for Ollama, and remember which slice belongs to which video so the user
    # prompt can label them ("Images N+1..N+M are frames from [VIDEO_K]").
    all_jpgs: list[bytes] = list(jpgs)
    video_frame_index_ranges: list[tuple[int, int]] | None = None
    if video_frame_jpgs:
        ranges: list[tuple[int, int]] = []
        for frames in video_frame_jpgs:
            if not frames:
                ranges.append((0, 0))
                continue
            start = len(all_jpgs) + 1   # 1-based for the prompt
            all_jpgs.extend(frames)
            end = len(all_jpgs)
            ranges.append((start, end))
        video_frame_index_ranges = ranges

    system_prompt = (
        _read_prompt("voice-system.md")
        + "\n\n---\n\n"
        + _read_prompt(f"mode-{mode}.md")
    )

    user_text = _modal_user_text(
        mode=mode,
        n_images=n,
        series_name=series_name,
        parent_series=parent_series,
        has_singulart=has_singulart,
        user_notes=user_notes,
        artist_mode=artist_mode,
        title_hints=title_hints,
        alt_texts=alt_texts,
        notes_list=notes_list,
        video_descriptions=video_descriptions,
        video_frame_index_ranges=video_frame_index_ranges,
    )

    n_frames = len(all_jpgs) - n
    logger.info(
        "Modal article: mode=%s, model=%s, %d gallery image(s) + %d video frame(s) = %d total, %dKB, series=%s, parent=%s, singulart=%s",
        mode, settings.ollama_llm_model, n, n_frames, len(all_jpgs),
        sum(len(j) for j in all_jpgs) // 1024,
        series_name or "(none)",
        parent_series["name"] if parent_series else "(none)",
        has_singulart,
    )

    # Bump context budget when video frames are present — each VL frame costs
    # ~1000–2000 tokens and the default 16k cap gets tight past 2–3 videos.
    num_ctx = 24576 if n_frames else 16384

    # Free VRAM for the 17 GB article LLM. The titler model (qwen2.5vl:3b)
    # holds ~3 GB resident for 30 min after warm-up; loading qwen3.6:27b on
    # top of that crashes Ollama's model-runner subprocess on constrained
    # GPUs. Unloading the titler first is the cleanest fix.
    if (
        settings.ollama_titler_model
        and settings.ollama_titler_model != settings.ollama_llm_model
    ):
        await unload_model(settings.ollama_titler_model)

    parsed = await _chat_json(
        model=settings.ollama_llm_model,
        system=system_prompt,
        user_text=user_text,
        jpgs=all_jpgs,
        options={"temperature": 0.65, "num_ctx": num_ctx, "num_predict": 6000},
        think=False,                # thinking + format:json hangs on Qwen3-family models
        timeout=timeout,
        label=f"write_modal_article[{mode}]",
        salvage=_iterative_salvage,
        max_attempts=3,             # qwen3.6 is non-deterministic; fresh samples often parse cleanly
    )

    has_parent = parent_series is not None
    out: dict[str, dict[str, Any]] = {}
    for lang in _MODAL_LANGS:
        block = parsed.get(lang) or {}
        if mode == "essay":
            out[lang] = _validate_essay_block(lang, block)
        elif mode == "work":
            out[lang] = _validate_work_block(lang, block, has_singulart=has_singulart, has_parent=has_parent)
        else:  # lab
            out[lang] = _validate_lab_block(lang, block)

    # DE revision pass — Qwen-family models carry EN syntax into the DE block
    # when both languages are generated together. The revision rewrites
    # idiomatically; on any failure or invariant violation we keep the original.
    if "de" in out:
        revised_de = await _revise_german_block(
            out["de"],
            mode=mode,
            has_singulart=has_singulart,
            has_parent=has_parent,
        )
        if revised_de is not None:
            out["de"] = revised_de

    return out


# ── Titler — title suggestions + warm-up ─────────────────────────────────────


_TITLER_SYSTEM = "You are an art curator specialising in contemporary media art."
_TITLER_KEEP_ALIVE = "30m"  # keep VLM resident in VRAM after each call — cold-load is ~2.5 min


async def warm_titler_model(timeout: float = 600.0) -> None:
    """
    Fire-and-forget Ollama call that loads the titler model into VRAM.

    Called from the FastAPI lifespan as a background task so the first real
    request from the frontend doesn't pay the ~150s cold-load and hit
    upstream timeouts (Cloudflare tunnel caps at ~100s).

    Uses a tiny synthetic JPG so the vision tower warms up too, with
    num_predict=1 to keep wall time near the pure load cost.
    """
    from io import BytesIO
    from PIL import Image as PILImage

    buf = BytesIO()
    PILImage.new("RGB", (32, 32), (128, 128, 128)).save(buf, "JPEG", quality=50)
    jpg = buf.getvalue()

    payload = {
        "model":      settings.ollama_titler_model,
        "messages":   [{"role": "user", "content": "ok", "images": [base64.b64encode(jpg).decode("ascii")]}],
        "stream":     False,
        "keep_alive": _TITLER_KEEP_ALIVE,
        "options":    {"num_predict": 1, "temperature": 0.0},
    }
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(f"{settings.ollama_host}/api/chat", json=payload)
        if r.status_code != 200:
            logger.warning("Titler warm-up: ollama returned %s — %s", r.status_code, r.text[:200])
            return
        data = r.json()
        load_ms = (data.get("load_duration") or 0) / 1_000_000
        total_ms = (data.get("total_duration") or 0) / 1_000_000
        logger.info(
            "Titler warm-up complete: model=%s, load=%.1fs, total=%.1fs",
            settings.ollama_titler_model, load_ms / 1000, total_ms / 1000,
        )
    except Exception as exc:
        logger.warning("Titler warm-up failed (will retry on first request): %s", exc)


async def generate_titles(
    jpg_bytes: bytes,
    *,
    n: int = 5,
    timeout: float = 120.0,
) -> list[str]:
    """
    Generate *n* short title suggestions for the artwork in *jpg_bytes*.

    Uses OLLAMA_TITLER_MODEL (must be vision-capable). Returns a deduplicated
    list of cleaned title strings, capped at *n*.

    Raises RuntimeError on Ollama error / non-JSON output, httpx.HTTPError
    on network failures.
    """
    user_text = (
        f"Suggest {n} short, evocative titles for this artwork. "
        f"Each title: 2 to 6 words, Title Case, no trailing punctuation, "
        f"no surrounding quotes, no numbering, no commentary.\n\n"
        f'Return STRICT JSON: {{"titles": ["title one", "title two", ...]}}'
    )
    parsed = await _chat_json(
        model=settings.ollama_titler_model,
        system=_TITLER_SYSTEM,
        user_text=user_text,
        jpgs=[jpg_bytes],
        options={"temperature": 0.8},
        keep_alive=_TITLER_KEEP_ALIVE,
        timeout=timeout,
        label="generate_titles",
    )
    return _clean_titles(parsed, n=n)


async def generate_video_titles(
    jpgs: list[bytes],
    *,
    n: int = 5,
    timeout: float = 180.0,
) -> list[str]:
    """
    Generate *n* short title suggestions for a short video, given a handful of
    evenly-spaced sample frames in playback order.

    Reuses OLLAMA_TITLER_MODEL with a video-aware user prompt that tells the
    model the frames belong to one work (so titles describe the piece as a
    whole, not each frame separately). Returns a deduplicated, cleaned list
    capped at *n*.
    """
    if not jpgs:
        raise RuntimeError("generate_video_titles requires at least one frame")

    user_text = (
        f"The {len(jpgs)} images below are evenly-spaced frames from one short "
        f"video artwork, in playback order. Suggest {n} short, evocative titles "
        f"for the video as a whole — not the individual frames. "
        f"Each title: 2 to 6 words, Title Case, no trailing punctuation, "
        f"no surrounding quotes, no numbering, no commentary.\n\n"
        f'Return STRICT JSON: {{"titles": ["title one", "title two", ...]}}'
    )
    parsed = await _chat_json(
        model=settings.ollama_titler_model,
        system=_TITLER_SYSTEM,
        user_text=user_text,
        jpgs=jpgs,
        options={"temperature": 0.8},
        keep_alive=_TITLER_KEEP_ALIVE,
        timeout=timeout,
        label="generate_video_titles",
    )
    return _clean_titles(parsed, n=n)


def _clean_titles(parsed: dict, *, n: int) -> list[str]:
    raw = parsed.get("titles") or []
    if not isinstance(raw, list):
        raise RuntimeError(f"Titler VLM 'titles' field is not a list: {type(raw).__name__}")

    seen: set[str] = set()
    out: list[str] = []
    for t in raw:
        s = str(t).strip().strip('"').strip("'").rstrip(".").strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out[:n]


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


# ── Z-Image Turbo prompt enhancer ────────────────────────────────────────────
# Backed by prompts/zimage-enhancer.md (system template, with {STYLE_BLOCK}
# placeholder) and prompts/zimage-styles.md (six-style signature library + G).

# Deterministic rotation. A..F is the default 6-style set; G is the optional
# aerial overlay, only chosen if the caller asks for n >= 7 styles.
_ZIMAGE_STYLE_SECTIONS = ("A", "B", "C", "D", "E", "F", "G")


@lru_cache(maxsize=1)
def _zimage_style_blocks() -> dict[str, str]:
    """Parse prompts/zimage-styles.md into {style_letter: block_text} pairs.

    Each block is the heading `## Style X — ...` plus all following lines
    until the next `## ` heading or EOF. Cached for the process lifetime.
    """
    text = _read_prompt("zimage-styles.md")
    pattern = re.compile(
        r"^##\s+Style\s+([A-G])\b.*?(?=^##\s+|\Z)",
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
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.post(f"{settings.ollama_host}/api/chat", json=payload)
        r.raise_for_status()
        data = r.json()
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
    n: int = 6,
    timeout: float = 120.0,
) -> list[dict[str, str]]:
    """
    Enhance a short user idea into *n* Z-Image Turbo prompts, one per style.

    Iterates through styles A..F (and G for n=7) and dispatches one Qwen
    call per style. Calls run concurrently — Ollama serialises on its own
    queue if the model is single-instance, so wall time is roughly
    n × single_call_latency on shared hardware.

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
