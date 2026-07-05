"""
Modal-article slot validators (essay/work/lab) + Yoast-style SEO and
readability diagnostics used by services/ollama/articles.py.
"""
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


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
