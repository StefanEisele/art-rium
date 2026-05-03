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
import re
from pathlib import Path
from typing import Any

import httpx

from core.config import settings

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"
_VOICE_GUIDE_PATH = _PROMPTS_DIR / "voice.md"
_VOICE_RICH_PATH  = _PROMPTS_DIR / "voice-rich.md"
_voice_guide_cache:      str | None = None
_voice_rich_cache:       str | None = None


def _voice_guide() -> str:
    """Lazy-load prompts/voice.md once per process."""
    global _voice_guide_cache
    if _voice_guide_cache is None:
        _voice_guide_cache = _VOICE_GUIDE_PATH.read_text(encoding="utf-8")
    return _voice_guide_cache


def _voice_rich() -> str:
    """Lazy-load prompts/voice-rich.md once per process."""
    global _voice_rich_cache
    if _voice_rich_cache is None:
        _voice_rich_cache = _VOICE_RICH_PATH.read_text(encoding="utf-8")
    return _voice_rich_cache


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

==========================================================================
VOICE RULES — non-negotiable. These are the most-violated rules in past drafts. Read them BEFORE you start writing each paragraph and check the paragraph against them BEFORE moving on.
==========================================================================

RULE 1 — NO hedging verbs.
  Replace "seems / appears / feels like / wirkt / scheint / fühlt sich an wie" with direct verbs of observation ("is / holds / rests / falls / ist / hält / liegt / fällt"). This applies to EVERY paragraph, not just the entry.
    ✗ "a silence that seems heavy to carry"      → ✓ "the silence holds"
    ✗ "the light seems to come from a source"    → ✓ "the light comes from above" (or just describe what's there)
    ✗ "scheint der Träger gebeugt"               → ✓ "der Träger ist gebeugt"
    ✗ "die Atmosphäre wirkt schwer"              → ✓ "die Atmosphäre ist schwer" (or cut — the image already shows it)
    ✗ "what it seems / was sie zu sein scheint"  → ✓ cut entirely; the phrase carries no observation

RULE 2 — NO Fazit closes. Paragraph 4 has SPECIFIC bans:
  - No sentence in p4 may begin with "It is" / "It's" / "Es ist" / "Sie ist" / "这是" / "它是" / "那是".
  - No p4 sentence may contain "the answer to" / "a symbol of" / "a metaphor for" / "what happens when" / "die Antwort auf" / "ein Symbol für" / "eine Metapher für" / "was passiert, wenn".
  - No anaphora (do not start three sentences in a row with the same subject + verb pattern).
  - No meta-commentary about looking at the image ("a moment that does not need to be explained, only seen" — forbidden).
  Instead, paragraph 4 ends on ONE concrete observation drawn from the image that has not already been named in paragraphs 1–3 — a colour shifting, a shadow's edge, a texture, a fragment of light. Two sentences max. Then stop.
    ✗ "It is an image of the journey, not of the destination. It is an image of the burden we carry…"
    ✗ "Es ist ein Bild von der Reise, nicht von dem Ziel."
    ✗ "It is a moment that does not need to be explained, but only seen. The cloud is a mirror of the silence that rules the room. It is the answer to the question of what happens when we try to carry the impossible."
    ✓ "The pause holds. It does not go empty, because the book is still open and the light keeps shifting. Muted ochre, a broken green. The colours hold the room together without closing it."
    ✓ "Die Pause hält sich. Sie wird nicht leer, weil das Buch noch offen ist und das Licht sich verschiebt. Gedämpftes Ocker, ein gebrochenes Grün. Die Farben halten den Raum zusammen, ohne ihn zu schließen."

RULE 3 — NO precious adverbs ("etwas / ein wenig / fast / beinahe / somewhat / a little / slightly"). Cut them. If the observation needs one, the noun or verb is wrong — rewrite that instead.

RULE 4 — NO interior projection beyond what's visible. If the image does not show a decision, a longing, a memory — do not write one. Stay in the room. Series/project context (paragraph 3) is the ONLY place where slightly broader framing is allowed, and even there, frame it concretely (materials, recurring forms, palette across the series), not psychologically.

RULE 5 — Each language NATIVELY. The German must read as German prose; the English as English prose; the Chinese as 中文书面语. Same observations, same structure, idiomatic phrasing in each. NEVER lift constructions across languages.
  - English-specific: do not lift German participle constructions or compound nouns. "The man is not stuffed" was a translation error from "gestopft" — write English the model would write if German didn't exist.
  - Chinese-specific: 中文 must use 中文 vocabulary throughout — no English words left untranslated (e.g. "muted" → "暗淡 / 沉静"). Check negations carefully: 关乎 ("is about") and 无关 ("is not about") are opposites; do not invert the meaning.

==========================================================================
OUTPUT SHAPE
==========================================================================

Return STRICT JSON with exactly this shape:
{
  "de": {"title": str, "body_md": str, "excerpt": str, "tags": [str, ...]},
  "en": {"title": str, "body_md": str, "excerpt": str, "tags": [str, ...]},
  "zh": {"title": str, "body_md": str, "excerpt": str, "tags": [str, ...]}
}

For each language:
  title    — 1 to 7 words, evocative, per the voice guide's "Titles" rules above. No trailing period. No quote marks.
  body_md  — the article body. Length: 260–320 words; 240 is a hard floor. Plain Markdown, paragraph breaks only (no headings, no bold/italic, no links, no bullets). Write FOUR distinct paragraphs separated by blank lines, in this order with these per-paragraph word budgets: (1) concrete entry, 40–60 words — one specific observation, not a thesis; (2) reflection/mood, 100–140 words — anchor every move in concrete detail from the image; (3) series/project context, 50–80 words — how this piece sits in a wider body of work; (4) quiet close, 20–40 words — see Rule 2 above; one or two sentences ending on a concrete observation. All four paragraphs are required.
  excerpt  — one sentence, ≤155 characters, in the article's voice. Used as the meta description; concrete imagery, no marketing language.
  tags     — 3 to 6 short lowercase tags relevant to the artwork. Single words or short phrases. Same set across languages where it makes sense (proper nouns may differ).

==========================================================================
FINAL CHECKLIST — verify before emitting JSON.
==========================================================================
1. Each body_md is 260–320 words (240 hard floor).
2. Each body_md has exactly 4 paragraphs separated by blank lines.
3. NO sentence anywhere in paragraph 4 starts with "It is" / "It's" / "Es ist" / "Sie ist" / "这是" / "它是" / "那是".
4. NO sentence anywhere in paragraph 4 contains "a decision" / "an acceptance" / "a longing" / "a memory" / "eine Entscheidung" / "eine Annahme" — these project interiority not visible in the image.

Return ONLY the JSON object — no prose around it, no code fences, no commentary."""


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
    title_hints = title_hints if title_hints is not None else [None] * n
    alt_texts   = alt_texts   if alt_texts   is not None else [None] * n
    notes_list  = notes_list  if notes_list  is not None else [None] * n
    if not (len(title_hints) == len(alt_texts) == len(notes_list) == n):
        raise ValueError("metadata lists must be parallel to jpgs")

    system_prompt = _voice_guide() + "\n\n---\n\n" + _ARTICLE_TASK

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
    for i, (t, a, no) in enumerate(zip(title_hints, alt_texts, notes_list), 1):
        header = f"  Image {i}:" if n > 1 else None
        indent = "    " if n > 1 else "  "
        if header:
            parts.append(header)
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
    parts.append("\nWrite the article now. JSON only.")
    user_text = "\n".join(parts)

    payload: dict[str, Any] = {
        "model": settings.ollama_llm_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": user_text,
                "images": [base64.b64encode(j).decode("ascii") for j in jpgs],
            },
        ],
        "format": "json",
        "stream": False,
        "think": False,                # thinking + format:json hangs on Qwen3-family models
        "options": {"temperature": 0.7, "num_ctx": 8192, "num_predict": 4096},
    }

    logger.info(
        "Article generation: model=%s, %d image(s), total=%dKB",
        settings.ollama_llm_model, n, sum(len(j) for j in jpgs) // 1024,
    )
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


# ───────────────────────────────────────────────────────────────────────────
# Rich-article writer — SEO-friendly multi-section series articles
# ───────────────────────────────────────────────────────────────────────────


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
    title_hints = title_hints if title_hints is not None else [None] * n
    alt_texts   = alt_texts   if alt_texts   is not None else [None] * n
    notes_list  = notes_list  if notes_list  is not None else [None] * n
    if not (len(title_hints) == len(alt_texts) == len(notes_list) == n):
        raise ValueError("metadata lists must be parallel to jpgs")

    task_spec = _rich_article_task(n, series_name, parent_series, has_singulart, artist_mode)
    system_prompt = _voice_rich() + "\n\n---\n\n" + task_spec

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
    for i, (t, a, no) in enumerate(zip(title_hints, alt_texts, notes_list), 1):
        parts.append(f"  Image {i}:")
        any_meta = False
        if t:
            parts.append(f"    title hint: {t}")
            any_meta = True
        if a:
            parts.append(f"    alt text:   {a}")
            any_meta = True
        if no:
            parts.append(f"    notes:      {no}")
            any_meta = True
        if not any_meta:
            parts.append(f"    (none)")
    parts.append("\nWrite the article now. JSON only.")
    user_text = "\n".join(parts)

    payload: dict[str, Any] = {
        "model": settings.ollama_llm_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": user_text,
                "images": [base64.b64encode(j).decode("ascii") for j in jpgs],
            },
        ],
        "format": "json",
        "stream": False,
        "think": False,                # thinking + format:json hangs on Qwen3-family models
        "options": {"temperature": 0.65, "num_ctx": 16384, "num_predict": 6000},
    }

    logger.info(
        "Rich article: model=%s, %d image(s), %dKB total, series=%s, parent=%s, singulart=%s",
        settings.ollama_llm_model, n, sum(len(j) for j in jpgs) // 1024,
        series_name or "(none)",
        parent_series["name"] if parent_series else "(none)",
        has_singulart,
    )

    async def _call_ollama() -> tuple[str, dict[str, Any]]:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(f"{settings.ollama_host}/api/chat", json=payload)
            r.raise_for_status()
            d = r.json()
        if "error" in d:
            raise RuntimeError(f"Ollama error: {d['error']}")
        return d.get("message", {}).get("content", "") or "", d

    def _strip_fences(content: str) -> str:
        # qwen3.6 occasionally wraps the JSON in ```json fences despite the no-fences instruction.
        s = content.strip()
        if s.startswith("```"):
            first_nl = s.find("\n")
            if first_nl != -1:
                s = s[first_nl + 1:]
            if s.rstrip().endswith("```"):
                s = s.rstrip()[:-3].rstrip()
        return s

    def _parse_with_salvage(content: str) -> dict:
        """Parse JSON from model output. On JSONDecodeError try iterative
        salvage; on still-failure raise the JSONDecodeError."""
        stripped = _strip_fences(content)
        try:
            return json.loads(stripped)
        except json.JSONDecodeError as exc:
            salvaged = _iterative_salvage(stripped)
            if salvaged != stripped:
                try:
                    result = json.loads(salvaged)
                    logger.warning(
                        "Rich-article LLM returned non-JSON; iterative quote-salvage succeeded (orig parse err: %s)",
                        exc,
                    )
                    return result
                except json.JSONDecodeError as exc2:
                    logger.warning(
                        "Rich-article LLM returned non-JSON; salvage made progress but did not parse (orig=%s, salvaged=%s)",
                        exc, exc2,
                    )
            raise

    # Up to 3 attempts total — the model is non-deterministic, fresh samples
    # often parse cleanly even when previous attempts produced unfixable JSON.
    MAX_ATTEMPTS = 3
    parsed = None
    last_exc: json.JSONDecodeError | None = None
    last_content = ""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        content, data = await _call_ollama()
        if not content.strip():
            logger.warning("Rich-article LLM attempt %d returned empty content; retrying.", attempt)
            continue
        last_content = content
        try:
            parsed = _parse_with_salvage(content)
            if attempt > 1:
                logger.info("Rich-article LLM attempt %d produced valid JSON.", attempt)
            break
        except json.JSONDecodeError as exc:
            last_exc = exc
            if attempt < MAX_ATTEMPTS:
                logger.warning(
                    "Rich-article LLM attempt %d failed JSON parse (%s); retrying (attempt %d/%d).",
                    attempt, exc, attempt + 1, MAX_ATTEMPTS,
                )

    if parsed is None:
        logger.error(
            "Rich-article LLM failed JSON parse after %d attempts — last err: %s — content head: %s",
            MAX_ATTEMPTS, last_exc, last_content[:500],
        )
        snippet = (last_content[:300] + "...") if len(last_content) > 300 else (last_content or "(empty)")
        raise RuntimeError(
            f"Rich-article LLM returned non-JSON after {MAX_ATTEMPTS} attempts (len={len(last_content)}, parse err: {last_exc}) — start: {snippet!r}"
        ) from last_exc

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


async def reachable() -> bool:
    """Best-effort Ollama health check."""
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            r = await client.get(f"{settings.ollama_host}/api/tags")
            return r.status_code == 200
    except Exception:
        return False
