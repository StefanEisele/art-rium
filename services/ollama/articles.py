"""
Ollama article writers — legacy voice article, rich SEO series article, and
the modal (Essay/Work/Lab) writer used by the Articles tool.

Live flavour: write_modal_article (+ its DE revision pass).
Back-compat flavours, kept for direct callers, no longer routed via the
HTTP API: write_article, write_rich_article.
"""
import asyncio
import json
import logging
import re
from typing import Any

import httpx

from core.config import settings
from services.comfy.client import free_memory as comfy_free_memory
from services.ollama.chat import _chat_json, _read_prompt, unload_model
from services.ollama.validators import (
    _validate_essay_block,
    _validate_lab_block,
    _validate_work_block,
)

logger = logging.getLogger(__name__)


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

    # Context budget capped at 16k. The article model (gemma3:27b Q3) runs at
    # the VRAM edge (spilling onto the 2070); bumping num_ctx to 24576 for
    # video frames overflows the KV/compute buffer and crashes the runner
    # (GGML_ASSERT(buffer) failed). Gemma 3 encodes each image with a fixed
    # ~256 tokens regardless of resolution, so even multi-video essays fit in
    # 16k — the old 24576 bump was for Qwen VL frames (~1-2k tokens each).
    num_ctx = 16384

    # Free VRAM for the 17 GB article LLM. The titler, VLM, and prompt-enhancer
    # models all may sit resident in VRAM (titler warm-up = 30 min keep-alive;
    # VLM stays for the default 5 min after the per-image analyze_image batch
    # during WP upload; prompt-enhancer is fired by the Z-Image tool). ComfyUI
    # also holds the z-Image-Turbo UNet/VAE/CLIP from the latest generation.
    # On constrained GPUs (12-16 GB) the article LLM load crashes with an OOM
    # that surfaces as `model runner has unexpectedly stopped` from /api/chat.
    article_model = settings.ollama_llm_model
    for other_model in (
        settings.ollama_titler_model,
        settings.ollama_vlm_model,
        settings.ollama_prompt_model,
    ):
        if other_model and other_model != article_model:
            await unload_model(other_model)

    # Tell ComfyUI to release its loaded checkpoints. The next ComfyUI prompt
    # pays a cold-load (~10-30s), but the article LLM gets the headroom it needs.
    async with httpx.AsyncClient() as client:
        await comfy_free_memory(client)

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
