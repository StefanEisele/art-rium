"""The rules every prompt core passes in code before it costs a render.

The rules are also in the prompts (BLOCK: ZIMAGE_RULES), but a model can read
a rule and break it anyway. Measured 2026-10-08 with these exact rules in
front of it, the 4 B prompt model still wrote 44 words in a 40-word mode, put
"rust-orange" into a formula core, kept a German noun and dropped the
must-keep "lighthouse". Each check below answers one of those.

`check_core()` returns the first broken rule as a `Violation` — a short key
for the run history and a message addressed to the editor, naming only that
rule, which is what its single retry is sent back with.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

LENGTH = {"formula": (8, 40), "prose": (120, 220)}

# Colour and capture words that belong to the anchor in formula mode. "rust"
# alone is deliberately absent: "rust growing over a leaf" is a material, and
# the material physics are half of the artist's signature.
_FORMULA_BANNED = (
    "orange", "blue", "teal", "grey", "gray", "copper", "amber", "red", "gold", "golden",
    "turquoise", "cyan", "magenta", "vhs", "grain", "grainy", "film grain", "35mm",
    "polaroid", "monochrome", "desaturated", "sepia",
)
_BANNED_RE = re.compile(r"\b(" + "|".join(re.escape(w) for w in _FORMULA_BANNED) + r")\b", re.I)

_NEGATION_RE = re.compile(r"\b(no|not|without|never)\b", re.I)
_UMLAUT_RE = re.compile(r"[äöüÄÖÜß]")
# Function words that are German and never English. "die", "den", "was" are
# English words too and would flag good cores.
_GERMAN_RE = re.compile(r"\b(und|mit|eine|einer|einem|einen|nicht|ist|sind|der|das|auf|über|ein)\b", re.I)
_WORD_RE = re.compile(r"[A-Za-z0-9'’-]+")

_STOP = {
    "a", "an", "the", "one", "two", "three", "four", "five", "of", "in", "on", "at", "to",
    "and", "with", "single", "lone", "some", "many",
}


@dataclass(frozen=True)
class Violation:
    rule: str      # length | must_keep | formula | anchor | language | negation | repeat
    message: str   # what the editor's retry is told


def words(text: str) -> list[str]:
    return _WORD_RE.findall(text)


def normalize_core(text: str) -> str:
    """The form cores are compared in: lowercase words, punctuation dropped."""
    return " ".join(w.lower() for w in words(text))


def head_noun(phrase: str) -> str:
    """"a reaching hand" → "hand"; "kintsugi seams" → "seam". The stem is
    matched as a substring, so "figure" finds "figures" and "pour" "poured"."""
    toks = [w.lower() for w in words(phrase) if w.lower() not in _STOP]
    if not toks:
        return ""
    head = toks[-1]
    if head.endswith("ies") and len(head) > 4:
        return head[:-3]
    if head.endswith("s") and not head.endswith("ss") and len(head) > 3:
        return head[:-1]
    return head


def missing_must_keeps(core: str, must_keep: list[str]) -> list[str]:
    low = core.lower()
    return [m for m in must_keep if (h := head_noun(m)) and h not in low]


def _anchor_shingles(anchor_text: str) -> set[str]:
    toks = [w.lower() for w in words(re.sub(r"\{\w+\}", " ", anchor_text))]
    return {" ".join(toks[i:i + 3]) for i in range(len(toks) - 2)}


def check_core(
    core: str,
    *,
    mode: str,
    must_keep: list[str],
    anchor_head: str,
    anchor_tail: str,
    triggers: list[str],
    seen_cores: dict[str, int] | None = None,
) -> Violation | None:
    core = core.strip()
    n = len(words(core))
    lo, hi = LENGTH[mode]
    if not (lo <= n <= hi):
        return Violation("length", f"Your core has {n} words; {mode} mode needs {lo}–{hi}. "
                                   "Rewrite it to fix only that, keeping the change you made.")

    gone = missing_must_keeps(core, must_keep)
    if gone:
        return Violation("must_keep", "Your core no longer names these must-keeps by their noun: "
                                      f"{', '.join(gone)}. Put them back; change nothing else.")

    if mode == "formula":
        hit = sorted({m.group(1).lower() for m in _BANNED_RE.finditer(core)})
        if hit:
            return Violation("formula", f"Formula mode: the core contains palette/capture words ({', '.join(hit)}). "
                                        "The anchor owns colour, light and capture. Remove only those words.")

    low_core = normalize_core(core)
    for t in triggers:
        if t and t.lower() in core.lower():
            return Violation("anchor", f"The core contains the trigger word '{t}'. The program adds it; remove it.")
    copied = sorted(s for s in _anchor_shingles(f"{anchor_head} {anchor_tail}") if s in low_core)
    if copied:
        return Violation("anchor", f"The core repeats the locked anchor ('{copied[0]}'). "
                                   "The anchor is added around the core; remove the repeat.")

    if _UMLAUT_RE.search(core) or _GERMAN_RE.search(core):
        return Violation("language", "The core contains German words. Write it in English only, same content.")

    neg = _NEGATION_RE.search(core)
    if neg:
        return Violation("negation", f"The core uses '{neg.group(1)}'. Z-Image ignores negations; "
                                     "write the constraint as presence (\"an empty plaza\", not \"no people\").")

    if seen_cores is not None and (k := seen_cores.get(low_core)) is not None:
        return Violation("repeat", f"This exact core was already rendered in round {k}. "
                                   "Make the one change you chose; an identical core renders an identical image.")
    return None
