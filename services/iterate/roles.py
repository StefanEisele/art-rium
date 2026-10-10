"""The four model calls of one run — brief, critic, duel, editor — and the
validation that stands between a model's JSON and the loop.

A malformed report must never become the best and a broken core must never
cost a render, so every reply is checked here: scores coerced to integers
1–10, a misplaced nested key lifted back, a missing key retried once with
"reply with the exact keys", a core that breaks a rule (services.iterate.checks)
sent back once naming only that rule.

Call settings are the ones measured 2026-10-08 (prompts/iterate.md, "Models
and hardware"): `think: false`, `format: json`, images at 768 px / q85, and
`num_ctx` always set — without it Ollama sized qwen3-vl at 34.6 GB. It is
16384 and not smaller on purpose: at 8192 Ollama decides Gemma 27B (12.8 GB of
weights) fits the empty 4060 Ti alone, then fails to allocate the vision
projector on top (`GGML_ASSERT(buffer)`, cudaMalloc OOM). At 16384 its
estimate no longer fits one card and it splits across both, which works.
"""
from __future__ import annotations

import asyncio
import io
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import httpx
from PIL import Image as PILImage

from services.iterate.checks import Violation, check_core
from services.iterate.judge import CRITERIA
from services.iterate.spec import Spec
from services.comfy.vram import release_comfy_models
from services.ollama.chat import _chat_json, unload_model, wait_until_unloaded

logger = logging.getLogger(__name__)

NUM_CTX = 16384
KEEP_ALIVE = "5m"   # the runner unloads explicitly before every render
TIMEOUT = 600.0
VLM_EDGE, VLM_QUALITY = 768, 85
_GPU_SETTLE = 5.0   # CUDA hands freed memory back a moment after /free and unload

T_BRIEF, T_CRITIC, T_DUEL = 0.6, 0.2, 0.1
T_EDITOR = {"explore": 0.7, "refine": 0.5, "polish": 0.3}


@dataclass(frozen=True)
class Models:
    critic: str   # VLM: critic + duel
    editor: str   # LLM: brief + editor


def vlm_jpg(path: Path) -> bytes:
    with PILImage.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((VLM_EDGE, VLM_EDGE))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=VLM_QUALITY)
    return buf.getvalue()


async def load_jpg(path: Path) -> bytes:
    return await asyncio.to_thread(vlm_jpg, path)


async def _call(model: str, system: str, user: str, *, temperature: float, num_predict: int,
                label: str, jpgs: list[bytes] | None = None) -> dict:
    kwargs = dict(
        model=model, system=system, user_text=user, jpgs=jpgs,
        options={"num_ctx": NUM_CTX, "temperature": temperature, "num_predict": num_predict},
        keep_alive=KEEP_ALIVE, think=False, timeout=TIMEOUT, label=label, max_attempts=2,
    )
    try:
        return await _chat_json(**kwargs)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code < 500:
            raise
        # A 500 here is almost always a model load that found the GPU taken
        # (cudaMalloc OOM → "llama runner process has terminated"): measured
        # 2026-10-08 when a second job rendered on ComfyUI during a run. Clear
        # the cards completely once and try again.
        logger.warning("%s: Ollama 500 on load — freeing the GPU and retrying once", label)
        await release_comfy_models()
        await unload_model(model)
        await wait_until_unloaded(timeout=60)
        await asyncio.sleep(_GPU_SETTLE)
        return await _chat_json(**kwargs)


def _retry_text(user: str, previous: dict, problem: str) -> str:
    return (f"{user}\n\nYOUR PREVIOUS ANSWER:\n{json.dumps(previous, ensure_ascii=False)}\n\n"
            f"PROBLEM: {problem}\nReply with the complete JSON again, fixing only that.")


def _str_list(v, limit: int) -> list[str]:
    if isinstance(v, str):
        v = [p for p in v.split(",")]
    if not isinstance(v, list):
        return []
    out: list[str] = []
    for item in v:
        s = str(item).strip().strip("'\"")
        if s and s.lower() not in (o.lower() for o in out):
            out.append(s)
    return out[:limit]


def fmt_list(items: list[str]) -> str:
    return json.dumps(items, ensure_ascii=False)


def fmt_scores(scores: dict | None) -> str:
    if not scores:
        return "n/a"
    return " · ".join(f"{k} {scores[k]}" for k in CRITERIA)


# ── Brief ────────────────────────────────────────────────────────────────────

def validate_brief(r: dict) -> dict:
    for key in ("subject", "twist", "core"):
        if not str(r.get(key, "")).strip():
            raise ValueError(f"key '{key}' missing or empty")
    return {
        "subject": str(r["subject"]).strip(),
        "twist": str(r["twist"]).strip(),
        "twist_source": "proposed" if str(r.get("twist_source", "")).strip().lower() == "proposed" else "user",
        "must_keep": _str_list(r.get("must_keep"), 4),
        "free": _str_list(r.get("free"), 8),
        "mood": str(r.get("mood", "")).strip(),
        "core": str(r["core"]).strip(),
    }


async def run_brief(spec: Spec, models: Models, *, idea: str, mode: str, source_prompt: str,
                    check_kwargs: dict) -> tuple[dict, Violation | None]:
    """The frozen brief and the first core. A core that still breaks a rule
    after two retries is used anyway and the violation returned: round 1 has
    no best to fall back on, and refusing to start is worse than a 42-word
    first core the editor will tighten."""
    system = spec.prompts["BRIEF"]
    user = spec.templates["BRIEF_USER"].format(idea=idea or "(none — develop the source prompt)",
                                               mode=mode, source_prompt=source_prompt or "")
    text, brief, violation = user, None, None
    for attempt in range(3):
        raw = await _call(models.editor, system, text, temperature=T_BRIEF, num_predict=900, label="iterate-brief")
        try:
            brief = validate_brief(raw)
        except ValueError as exc:
            text = _retry_text(user, raw, f"{exc}. Reply with exactly the keys of the OUTPUT line.")
            continue
        violation = check_core(brief["core"], mode=mode, must_keep=brief["must_keep"], **check_kwargs)
        if violation is None:
            return brief, None
        logger.info("iterate brief: core broke rule %s (attempt %d)", violation.rule, attempt + 1)
        text = _retry_text(user, raw, violation.message)
    if brief is None:
        raise RuntimeError("The brief model returned no usable brief after three attempts")
    return brief, violation


# ── Critic ───────────────────────────────────────────────────────────────────

_REPORT_KEYS = "observed, must_keep, scores {intent, surreal, signature, composition, craft}, defects, keep"


def validate_report(r: dict, must_keep: list[str]) -> dict:
    scores = r.get("scores")
    if not isinstance(scores, dict):   # sometimes the five criteria land at top level
        scores = {k: r.get(k) for k in CRITERIA}
    out: dict[str, int] = {}
    for k in CRITERIA:
        v = scores.get(k)
        try:
            out[k] = max(1, min(10, int(round(float(v)))))
        except (TypeError, ValueError):
            raise ValueError(f"score '{k}' missing or not a number: {v!r}")
    listed = {}
    for m in r.get("must_keep") or []:
        if isinstance(m, dict) and m.get("status") in ("present", "weak", "missing"):
            listed[str(m.get("item", "")).strip().lower()] = m["status"]
    # An item the critic skipped is reported as weak, not missing: unknown
    # must not gate a candidate out, but the editor should look at it.
    mk = [{"item": item, "status": listed.get(item.lower(), "weak")} for item in must_keep]
    return {
        "observed": str(r.get("observed", "")).strip(),
        "must_keep": mk,
        "scores": out,
        "defects": [str(d) for d in (r.get("defects") or []) if str(d).strip()][:3],
        "keep": [str(k) for k in (r.get("keep") or []) if str(k).strip()][:3],
    }


async def run_critic(spec: Spec, models: Models, jpg: bytes, brief: dict, *, round_no: int,
                     rounds: int, fragile: bool = False) -> dict:
    user = spec.templates["CRITIC_USER"].format(
        subject=brief["subject"], twist=brief["twist"], must_keep=fmt_list(brief["must_keep"]),
        mood=brief["mood"], round=round_no, rounds=rounds,
        fragile_line=("Note: the best prompt so far is seed-fragile — a second seed scored much lower."
                      if fragile else ""),
    )
    text = user
    for _ in range(2):
        raw = await _call(models.critic, spec.prompts["CRITIC"], text, temperature=T_CRITIC,
                          num_predict=900, label="iterate-critic", jpgs=[jpg])
        try:
            return validate_report(raw, brief["must_keep"])
        except ValueError as exc:
            text = _retry_text(user, raw, f"{exc}. Reply with the exact keys only: {_REPORT_KEYS}.")
    raise RuntimeError("The critic returned no valid report twice")


# ── Duel ─────────────────────────────────────────────────────────────────────

async def run_duel(spec: Spec, models: Models, a: bytes, b: bytes, brief: dict) -> dict:
    """One order of a duel. Returns {"winner": "A"|"B"|"tie", "reason": str}."""
    user = spec.templates["DUEL_USER"].format(
        subject=brief["subject"], twist=brief["twist"], must_keep=fmt_list(brief["must_keep"]),
        mood=brief["mood"],
    )
    raw = await _call(models.critic, spec.prompts["DUEL"], user, temperature=T_DUEL,
                      num_predict=500, label="iterate-duel", jpgs=[a, b])
    winner = {"A": "A", "B": "B"}.get(str(raw.get("winner", "")).strip().upper(), "tie")
    return {"winner": winner, "reason": str(raw.get("reason", "")).strip()}


# ── Editor ───────────────────────────────────────────────────────────────────

def validate_edit(r: dict, allowed: list[str]) -> dict:
    core = str(r.get("core", "")).strip()
    if not core:
        raise ValueError("key 'core' missing or empty")
    op = str(r.get("operator", "")).strip().upper()
    if op not in allowed:
        raise ValueError(f"operator '{op or '—'}' is not allowed this round; choose one of {', '.join(allowed)}")
    return {
        "diagnosis": str(r.get("diagnosis", "")).strip(),
        "operator": op,
        "change": str(r.get("change", "")).strip(),
        "expect": str(r.get("expect", "")).strip(),
        "core": core,
    }


async def run_editor(spec: Spec, models: Models, fields: dict, *, phase: str, allowed: list[str],
                     mode: str, must_keep: list[str], check_kwargs: dict,
                     seen_cores: dict[str, int]) -> tuple[dict | None, Violation | None]:
    """The next core, or (None, violation) when it broke a rule twice — the
    round is then skipped without a render."""
    user = spec.templates["EDITOR_USER"].format(**fields, operators_allowed=", ".join(allowed))
    text, last_violation = user, None
    for _ in range(2):
        raw = await _call(models.editor, spec.prompts["EDITOR"], text, temperature=T_EDITOR[phase],
                          num_predict=700, label="iterate-editor")
        try:
            edit = validate_edit(raw, allowed)
        except ValueError as exc:
            last_violation = Violation("schema", str(exc))
            text = _retry_text(user, raw, str(exc))
            continue
        last_violation = check_core(edit["core"], mode=mode, must_keep=must_keep,
                                    seen_cores=seen_cores, **check_kwargs)
        if last_violation is None:
            return edit, None
        text = _retry_text(user, raw, last_violation.message)
    return None, last_violation
