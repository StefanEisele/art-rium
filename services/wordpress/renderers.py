"""
Per-mode article renderers — stitch LLM slot prose + Gutenberg blocks
(services/wordpress/gutenberg.py) into a WordPress post body. One renderer
per article flavour: rich (back-compat) and essay/work/lab (modal).
"""
import html
import re

from core.models import Image, Video
from services.wordpress.gutenberg import (
    _bullet_list_block,
    _code_block,
    _footer_block,
    _gallery_block,
    _h2_block,
    _h3_block,
    _maybe_video_block,
    _metadata_block,
    _ordered_list_block,
    _para_block,
    _render_parent_placeholder,
    _separator_block,
    _singulart_image_block,
    _split_for_galleries,
    _SINGULART_CAPTION,
    _trailing_video_blocks,
)

_HEADINGS = {
    # Rich-article (back-compat) + Work-mode shared
    "concept":            {"en": "The Concept",                "de": "Das Konzept"},
    "visual_language":    {"en": "Visual Language",            "de": "Visuelle Sprache"},
    "technical_approach": {"en": "Technical Approach",         "de": "Technischer Ansatz"},
    "available_works":    {"en": "Available Works",            "de": "Verfügbare Werke"},
    "larger_practice":    {"en": "Part of a Larger Practice",  "de": "Teil einer größeren Praxis"},
    # Lab-mode
    "problem":            {"en": "Problem",                    "de": "Das Problem"},
    "approach":           {"en": "Approach",                   "de": "Ansatz"},
    "steps":              {"en": "Steps",                      "de": "Schritte"},
    "result":             {"en": "Result",                     "de": "Ergebnis"},
    "why":                {"en": "Why this matters",           "de": "Warum das zählt"},
    "tool_stack":         {"en": "Tool stack",                 "de": "Werkzeuge"},
}


# ── Rich renderer (back-compat) ──────────────────────────────────────────────


def _render_rich_post(
    slots: dict,
    lang: str,
    images: list[Image],
    singulart_links: list[dict] | None,
    parent_series: dict[str, str] | None,
) -> str:
    """Stitch the LLM slot prose, heading translations, gallery blocks, conditional
    Singulart product cards, and conditional parent-series link into a Gutenberg post body."""
    blocks: list[str] = []

    blocks.append(_para_block(html.escape(slots["intro"])))

    blocks.append(_h2_block(_HEADINGS["concept"][lang]))
    for p in slots["concept"]:
        blocks.append(_para_block(html.escape(p)))

    gallery_a, gallery_b = _split_for_galleries(images)
    if gallery_a:
        blocks.append(_gallery_block(gallery_a))

    blocks.append(_h2_block(_HEADINGS["visual_language"][lang]))
    for p in slots["visual_language"]:
        blocks.append(_para_block(html.escape(p)))

    if gallery_b:
        blocks.append(_gallery_block(gallery_b))

    blocks.append(_h2_block(_HEADINGS["technical_approach"][lang]))
    ta = slots["technical_approach"]
    blocks.append(_para_block(html.escape(ta["intro"])))
    if ta["steps"]:
        blocks.append(_bullet_list_block(ta["steps"]))
    if ta.get("outro"):
        blocks.append(_para_block(html.escape(ta["outro"])))

    if singulart_links:
        blocks.append(_h2_block(_HEADINGS["available_works"][lang]))
        works_intro = slots.get("available_works_intro") or ""
        if works_intro:
            blocks.append(_para_block(html.escape(works_intro)))
        for link in singulart_links:
            blocks.append(_h3_block(link.get("title") or ""))
            blocks.append(_singulart_image_block(link, _SINGULART_CAPTION[lang]))

    if parent_series and slots.get("larger_practice"):
        blocks.append(_h2_block(_HEADINGS["larger_practice"][lang]))
        for p in slots["larger_practice"]:
            escaped = html.escape(p)
            blocks.append(_para_block(_render_parent_placeholder(escaped, parent_series)))

    blocks.append(_separator_block())
    blocks.append(_footer_block(lang))

    return "\n\n".join(blocks)


def _flatten_rich_body_for_search(slots: dict, parent_series: dict[str, str] | None) -> str:
    """Flatten LLM slot prose into a single paragraph-separated string for Article.body_md.
    Used for full-text search and regeneration debugging — NOT shown to readers (WP holds the HTML)."""
    parts: list[str] = [slots["intro"]]
    parts.extend(slots["concept"])
    parts.extend(slots["visual_language"])
    ta = slots["technical_approach"]
    parts.append(ta["intro"])
    parts.extend(f"- {s}" for s in ta["steps"])
    if ta.get("outro"):
        parts.append(ta["outro"])
    if parent_series and slots.get("larger_practice"):
        parts.extend(slots["larger_practice"])
    return "\n\n".join(p for p in parts if p)


# ── Essay renderer ──────────────────────────────────────────────────────────


def _render_essay_post(
    slots: dict,
    lang: str,
    images: list[Image],
    videos: list[Video] | None = None,
) -> str:
    """Stitch Essay-mode prose into a Gutenberg post body."""
    blocks: list[str] = []
    consumed: set[int] = set()

    for paragraph in slots["intro"]:
        embed = _maybe_video_block(paragraph, videos, consumed)
        blocks.append(embed if embed is not None else _para_block(html.escape(paragraph)))

    gallery_a, gallery_b = _split_for_galleries(images)
    if gallery_a:
        blocks.append(_gallery_block(gallery_a))

    for idx, movement in enumerate(slots["movements"]):
        blocks.append(_h2_block(movement["heading"]))
        for paragraph in movement["body"]:
            embed = _maybe_video_block(paragraph, videos, consumed)
            blocks.append(embed if embed is not None else _para_block(html.escape(paragraph)))
        # Drop the second gallery (when present) between movements 2 and 3 — keeps
        # the page from being a wall of text in image-heavy essays.
        if gallery_b and idx == max(0, len(slots["movements"]) // 2 - 1):
            blocks.append(_gallery_block(gallery_b))
            gallery_b = []

    if gallery_b:
        blocks.append(_gallery_block(gallery_b))

    if slots.get("closing"):
        closing = slots["closing"]
        embed = _maybe_video_block(closing, videos, consumed)
        blocks.append(embed if embed is not None else _para_block(html.escape(closing)))

    blocks.extend(_trailing_video_blocks(videos, consumed))

    blocks.append(_separator_block())
    blocks.append(_footer_block(lang))
    return "\n\n".join(blocks)


# ── Work renderer ──────────────────────────────────────────────────────────


def _render_work_post(
    slots: dict,
    lang: str,
    images: list[Image],
    singulart_links: list[dict] | None,
    parent_series: dict[str, str] | None,
    videos: list[Video] | None = None,
) -> str:
    """Stitch Work/Series-mode prose into a Gutenberg post body."""
    blocks: list[str] = []
    consumed: set[int] = set()

    # Opening line — styled as a lead paragraph (italic) so it visually carries weight.
    opening = html.escape(slots["opening_line"])
    blocks.append(
        '<!-- wp:paragraph {"className":"has-drop-cap"} -->\n'
        f'<p class="has-drop-cap"><em>{opening}</em></p>\n'
        '<!-- /wp:paragraph -->'
    )

    for paragraph in slots["body"]:
        embed = _maybe_video_block(paragraph, videos, consumed)
        blocks.append(embed if embed is not None else _para_block(html.escape(paragraph)))

    gallery_a, gallery_b = _split_for_galleries(images)
    if gallery_a:
        blocks.append(_gallery_block(gallery_a))

    if singulart_links and slots.get("available_works_intro"):
        blocks.append(_h2_block(_HEADINGS["available_works"][lang]))
        blocks.append(_para_block(html.escape(slots["available_works_intro"])))
        for link in singulart_links:
            blocks.append(_h3_block(link.get("title") or ""))
            blocks.append(_singulart_image_block(link, _SINGULART_CAPTION[lang]))

    if parent_series and slots.get("larger_practice"):
        blocks.append(_h2_block(_HEADINGS["larger_practice"][lang]))
        for paragraph in slots["larger_practice"]:
            embed = _maybe_video_block(paragraph, videos, consumed)
            if embed is not None:
                blocks.append(embed)
            else:
                escaped = html.escape(paragraph)
                blocks.append(_para_block(_render_parent_placeholder(escaped, parent_series)))

    if gallery_b:
        blocks.append(_gallery_block(gallery_b))

    blocks.extend(_trailing_video_blocks(videos, consumed))

    blocks.append(_separator_block())
    # Pick the Singulart URL for the metadata-block trailing line, if any.
    singulart_link = (singulart_links[0]["url"] if singulart_links else None)
    metadata_block = _metadata_block(slots.get("metadata") or {}, lang, singulart_link)
    if metadata_block:
        blocks.append(metadata_block)

    blocks.append(_separator_block())
    blocks.append(_footer_block(lang))
    return "\n\n".join(blocks)


# ── Lab renderer ───────────────────────────────────────────────────────────


def _render_lab_post(
    slots: dict,
    lang: str,
    images: list[Image],
    videos: list[Video] | None = None,
) -> str:
    """Stitch Lab/Tutorial-mode prose into a Gutenberg post body."""
    blocks: list[str] = []
    consumed: set[int] = set()

    # PROBLEM
    blocks.append(_h2_block(_HEADINGS["problem"][lang]))
    problem = slots["problem"]
    embed = _maybe_video_block(problem, videos, consumed)
    blocks.append(embed if embed is not None else _para_block(html.escape(problem)))

    # APPROACH
    blocks.append(_h2_block(_HEADINGS["approach"][lang]))
    # solution_intro can be one paragraph or several joined by blank lines — split & render each.
    for paragraph in re.split(r"\n\s*\n", slots["solution_intro"].strip()):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        embed = _maybe_video_block(paragraph, videos, consumed)
        blocks.append(embed if embed is not None else _para_block(html.escape(paragraph)))

    # Tool stack — inline as a single paragraph for a tight read.
    if slots.get("tool_stack"):
        label = html.escape(_HEADINGS["tool_stack"][lang])
        stack = ", ".join(html.escape(t) for t in slots["tool_stack"])
        blocks.append(_para_block(f"<strong>{label}:</strong> {stack}"))

    # Hardware context — italic callout when present.
    if slots.get("hardware_context"):
        blocks.append(_para_block(f"<em>{html.escape(slots['hardware_context'])}</em>"))

    # Optional gallery — useful when the post documents a visual workflow.
    gallery_a, _gallery_b = _split_for_galleries(images)
    if gallery_a:
        blocks.append(_gallery_block(gallery_a))

    # STEPS
    blocks.append(_h2_block(_HEADINGS["steps"][lang]))
    if slots.get("steps"):
        blocks.append(_ordered_list_block(slots["steps"]))

    # CODE BLOCKS — all rendered immediately after the steps list.
    for cb in slots.get("code_blocks") or []:
        blocks.append(_code_block(cb["language"], cb["code"], cb.get("caption", "")))

    # RESULT
    if slots.get("result"):
        blocks.append(_h2_block(_HEADINGS["result"][lang]))
        result = slots["result"]
        embed = _maybe_video_block(result, videos, consumed)
        blocks.append(embed if embed is not None else _para_block(html.escape(result)))

    # WHY IT MATTERS
    if slots.get("why_it_matters"):
        blocks.append(_h2_block(_HEADINGS["why"][lang]))
        why = slots["why_it_matters"]
        embed = _maybe_video_block(why, videos, consumed)
        blocks.append(embed if embed is not None else _para_block(html.escape(why)))

    blocks.extend(_trailing_video_blocks(videos, consumed))

    blocks.append(_separator_block())
    blocks.append(_footer_block(lang))
    return "\n\n".join(blocks)


# ── body_md flattener (for Article.body_md full-text search / debugging) ────


def _flatten_modal_body_for_search(slots: dict, mode: str) -> str:
    """Flatten a modal-mode language block into a single string for body_md.

    Not shown to readers (WP holds the rendered HTML). Used for FTS + debugging
    + the future regenerate-metadata endpoint."""
    parts: list[str] = []
    if mode == "essay":
        parts.extend(slots.get("intro") or [])
        for mv in slots.get("movements") or []:
            parts.append(f"## {mv['heading']}")
            parts.extend(mv.get("body") or [])
        if slots.get("closing"):
            parts.append(slots["closing"])
    elif mode == "work":
        if slots.get("opening_line"):
            parts.append(slots["opening_line"])
        parts.extend(slots.get("body") or [])
        if slots.get("available_works_intro"):
            parts.append(slots["available_works_intro"])
        parts.extend(slots.get("larger_practice") or [])
    elif mode == "lab":
        if slots.get("problem"):
            parts.append(slots["problem"])
        if slots.get("solution_intro"):
            parts.append(slots["solution_intro"])
        parts.extend(f"- {s}" for s in slots.get("steps") or [])
        for cb in slots.get("code_blocks") or []:
            parts.append(f"```{cb['language']}\n{cb['code']}\n```")
        if slots.get("result"):
            parts.append(slots["result"])
        if slots.get("why_it_matters"):
            parts.append(slots["why_it_matters"])
    return "\n\n".join(p for p in parts if p)
