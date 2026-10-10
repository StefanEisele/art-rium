"""Scoring, the accept decision, phases and the plateau jolt — prompts/iterate.md,
Part 1, "Accepting a candidate" and "Phases". Pure functions, no I/O.
"""
from __future__ import annotations

WEIGHTS = {"intent": 0.30, "surreal": 0.25, "signature": 0.20, "composition": 0.15, "craft": 0.10}
CRITERIA = tuple(WEIGHTS)

# Above this margin a score difference is trusted on its own; inside it the
# critic's absolute numbers are noise and the both-orders duel decides.
CLEAR_MARGIN = 0.6
GATE_INTENT, GATE_CRAFT = 5, 4

OPERATORS = (
    "SCALE", "REPEAT", "DOUBLE", "MATERIAL", "GRAVITY", "RELOCATE",
    "MERGE", "LIGHT", "CAMERA", "CAPTURE", "REDUCE", "CLARIFY",
)
_BOLD = ("SCALE", "REPEAT", "DOUBLE", "MATERIAL", "GRAVITY", "RELOCATE", "MERGE", "LIGHT", "CAMERA", "CAPTURE")
# In formula mode the anchor owns light and capture — a LIGHT or CAPTURE edit
# could only be written with the very words the formula check forbids.
_PROSE_ONLY = ("LIGHT", "CAPTURE")


def weighted_total(scores: dict[str, int]) -> float:
    return round(sum(WEIGHTS[k] * scores[k] for k in CRITERIA), 2)


def gate_failure(report: dict) -> str | None:
    """Why this report can never become the best, or None if it may."""
    s = report["scores"]
    if s["intent"] < GATE_INTENT:
        return f"intent {s['intent']} < {GATE_INTENT}"
    if s["craft"] < GATE_CRAFT:
        return f"craft {s['craft']} < {GATE_CRAFT}"
    gone = [m["item"] for m in report.get("must_keep", []) if m.get("status") == "missing"]
    if gone:
        return f"must-keep missing: {', '.join(gone)}"
    return None


def triage(candidate_total: float, best_total: float | None) -> str:
    """"first" (no best yet), "clear" (accept outright), "close" (duel decides)
    or "reject". Gates are checked by the caller before this."""
    if best_total is None:
        return "first"
    diff = round(candidate_total - best_total, 2)   # 6.6 − 6.0 is 0.5999… in floats
    if diff >= CLEAR_MARGIN:
        return "clear"
    if diff > -CLEAR_MARGIN:
        return "close"
    return "reject"


def duel_verdict(candidate_first: str, candidate_second: str) -> str:
    """Combine the two orders of one duel into "candidate", "best" or "split".

    `candidate_first` is the winner label ("A"/"B"/"tie") when the candidate
    was shown as A; `candidate_second` when it was shown as B. Only an answer
    that holds both ways counts — measured, Gemma 27B picks by position.
    """
    first = {"A": "candidate", "B": "best"}.get(candidate_first, "tie")
    second = {"A": "best", "B": "candidate"}.get(candidate_second, "tie")
    return first if first == second and first != "tie" else "split"


def phase_counts(n: int) -> tuple[int, int]:
    """(explore, polish) round counts for a run of n rounds."""
    if n <= 5:
        return 2, 1
    return round(0.35 * n), max(1, round(0.15 * n))


def phase_for(round_no: int, n: int) -> str:
    """The phase of round `round_no` (1-based). Round 1 is the brief's own
    core, so the first phase an editor ever sees is that of round 2."""
    explore, polish = phase_counts(n)
    if round_no <= explore:
        return "explore"
    if round_no > n - polish:
        return "polish"
    return "refine"


def plateau_after(n: int) -> int:
    """Rounds without an accepted candidate that trigger the jolt."""
    return max(3, round(n / 5))


def allowed_operators(
    *, mode: str, phase: str, plateau: bool, recent: list[str], intent_weak: bool,
) -> list[str]:
    """The operators the editor may use this round, in the md's order.

    `recent` is the operators of the last rounds, newest last; under a plateau
    the last four are excluded so the jolt is a different kind of change. A
    weak or missing must-keep always allows CLARIFY — fixing intent comes
    first in every phase.
    """
    if phase == "explore":
        ops = list(_BOLD)
    elif phase == "polish":
        ops = ["CLARIFY", "REDUCE"]
    else:
        ops = list(OPERATORS)
    if mode == "formula":
        ops = [o for o in ops if o not in _PROSE_ONLY]
    if plateau:
        fresh = [o for o in (_BOLD if mode == "prose" else [b for b in _BOLD if b not in _PROSE_ONLY])
                 if o not in recent[-4:]]
        ops = fresh or ops
    if intent_weak and "CLARIFY" not in ops:
        ops.append("CLARIFY")
    return ops
