"""The iterate background job — one idea in, N rounds, one strong image out.

What a round costs is mostly not the models but the hand-over between them:
Gemma 27B spreads over both cards (~8.9 GB on the 4060 Ti, ~3.9 GB on the
2070) and cannot share the 4060 Ti with Z-Image Turbo, so every round is

  think   editor writes the next core            (Ollama on the GPU)
  render  Ollama unloaded → ComfyUI renders      (ComfyUI on the GPU)
  think   ComfyUI /free → critic (→ duel)        (Ollama on the GPU)

and the hand-over is explicit both ways: `wait_until_unloaded` before a render
(Ollama's unload returns before the runner lets go of its VRAM) and ComfyUI's
/free plus a settle before a model load (a load into memory that is still
being released is what killed Gemma's projector allocation once already).

State lives in `_runs` while the job runs and in a sidecar
storage/iterations/<run_id>.json that is rewritten after every step, so a run
is readable after it finished or after a restart cut it short. The pictures
themselves are gallery rows from the moment they are rendered (batch_id = the
run id) and are collected into one image series, best first.
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from core.comfy import WORKFLOW_NAME as ZIMAGE_WORKFLOW_NAME
from core.config import settings
from core.db import AsyncSessionLocal
from core.loras import LORAS
from core.models import Image, ImageSeries, ImageSeriesItem
from services.comfy.client import free_memory, poll_history, post_workflow
from services.comfy.ingest import ingest_comfy_image
from services.comfy.progress import attach_live_stage
from services.comfy.vram import comfy_devices, evict_ollama
from services.comfy.zimage import ZIMAGE_SAVE_NODE, build_zimage_workflow
from services.iterate import judge
from services.iterate.checks import normalize_core
from services.iterate.roles import (
    Models, fmt_list, fmt_scores, load_jpg, run_brief, run_critic, run_duel, run_editor,
)
from services.iterate.spec import Anchor, Spec, compose_prompt
from services.ollama.chat import unload_model, wait_until_unloaded
from workers.comfy_listener import get_listener

logger = logging.getLogger(__name__)

RUNS_DIR_NAME = "iterations"
ROUNDS_MIN, ROUNDS_MAX, ROUNDS_DEFAULT = 1, 30, 8
SWEEP_MIN, SWEEP_MAX, SWEEP_DEFAULT = 0, 6, 3
SERIES_MAX = 20   # routers/series.py::MAX_IMAGES — what every consumer downstream accepts

# The LoRA stack of the published portrait series ("Desert Reflections"),
# which BLOCK: ANCHORS in prompts/iterate.md names as the signature preset.
SIGNATURE_LORAS = [
    {"name": "rustorangeanddimblue_lora_copy.safetensors", "strength": 0.35},
    {"name": "art_vision_ZIG.safetensors", "strength": 0.20},
    {"name": "zImageT_zidiusArt_melancholy.safetensors", "strength": 0.45},
    {"name": "art_vision_v2_epoch_10.safetensors", "strength": 0.35},
]
PRESET_FOR_MODE = {"formula": "signature", "prose": "prose"}

_RENDER_TIMEOUT = 360       # a cold load + 2048² render is ~90 s; past this the prompt is lost
_COMFY_BACK_TIMEOUT = 240.0 # how long a restarting ComfyUI gets before the run gives up
_RENDER_POLL = 2
_COMFY_FREE_SETTLE = 3.0    # CUDA hands /free'd memory back a moment later
_TRANSIENT = (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError, ConnectionError)

# run_id → live record. The sidecar is the durable copy.
_runs: dict[str, dict] = {}


def runs_dir() -> Path:
    d = settings.storage_dir / RUNS_DIR_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def active_run_id() -> str | None:
    for rid, run in _runs.items():
        if run["status"] in ("queued", "running"):
            return rid
    return None


def live_run(run_id: str) -> dict | None:
    """The in-memory record of a run this process started, or None."""
    return _runs.get(run_id)


def forget_run(run_id: str) -> bool:
    """Drop a finished run's sidecar (its pictures and series stay)."""
    _runs.pop(run_id, None)
    path = runs_dir() / f"{run_id}.json"
    if not path.exists():
        return False
    path.unlink()
    return True


def triggers_for(loras: list[dict]) -> list[str]:
    by_file = {lo["filename"]: lo["trigger"] for lo in LORAS}
    out: list[str] = []
    for sel in loras:
        t = by_file.get(sel["name"], "")
        if t and t not in out:
            out.append(t)
    return out


# ── Persistence ──────────────────────────────────────────────────────────────

def public_view(run: dict) -> dict:
    """The record without its private keys, with ComfyUI's live stage folded in."""
    view = {k: v for k, v in run.items() if not k.startswith("_")}
    live = {k: run[k] for k in ("_prompt_id", "_band") if k in run}
    if live:
        prog = attach_live_stage({"pct": run.get("pct", 0), **live})
        view["pct"] = prog.get("pct", view.get("pct"))
        if prog.get("detail"):
            view["detail"] = prog["detail"]
    return view


def _save(run: dict) -> None:
    path = runs_dir() / f"{run['id']}.json"
    tmp = path.with_suffix(".tmp")
    data = {k: v for k, v in run.items() if not k.startswith("_")}
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


def load_run(run_id: str) -> dict | None:
    if run_id in _runs:
        return public_view(_runs[run_id])
    path = runs_dir() / f"{run_id}.json"
    if not path.exists():
        return None
    try:
        run = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if run.get("status") in ("queued", "running"):
        # Not in memory but claims to be running: the server stopped under it.
        run["status"], run["message"] = "interrupted", "Server wurde während des Laufs beendet"
    return run


def list_runs(limit: int = 30) -> list[dict]:
    files = sorted(runs_dir().glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
    out = []
    for f in files:
        run = load_run(f.stem)
        if not run:
            continue
        best = next((r for r in run.get("rounds", []) if r.get("round") == run.get("best_round")), None)
        final = run.get("final") or {}
        out.append({
            "id": run["id"], "status": run["status"], "created_at": run.get("created_at"),
            "idea": run["params"].get("idea", ""), "title": run.get("title"),
            "rounds": run["params"]["rounds"], "done_rounds": len(run.get("rounds", [])),
            "best_total": (best or {}).get("total"),
            "thumb_url": final.get("thumb_url") or (best or {}).get("thumb_url"),
        })
    return out


def _set(run: dict, message: str, pct: float, **extra) -> None:
    run.update(message=message, pct=int(pct), updated_at=_now(), **extra)
    run.pop("_prompt_id", None)
    run.pop("_band", None)


# ── GPU hand-over ────────────────────────────────────────────────────────────

class _Gpu:
    """Who holds the card right now, so a hand-over happens only when needed."""

    def __init__(self, models: Models):
        self.models = models
        self.resident: str | None = None   # the iterate model Ollama holds, if any
        self.comfy = True                  # assume the worst at the start

    async def for_comfy(self) -> None:
        for m in {self.models.critic, self.models.editor}:
            await unload_model(m)
        await evict_ollama()
        await wait_until_unloaded(timeout=60)
        self.resident = None

    async def for_ollama(self, model: str) -> None:
        if self.comfy:
            async with httpx.AsyncClient(timeout=15) as client:
                await free_memory(client)
            await asyncio.sleep(_COMFY_FREE_SETTLE)
            self.comfy = False
        if self.resident and self.resident != model:
            # Two different models never share the cards: Gemma alone spreads
            # over both, and a load racing an unload is the GGML_ASSERT crash.
            await unload_model(self.resident)
            await wait_until_unloaded(timeout=60)
        self.resident = model


# ── Rendering ────────────────────────────────────────────────────────────────

async def _render(run: dict, gpu: _Gpu, prompt: str, seed: int, band: tuple[float, float]) -> dict:
    """Render, ingest, return the gallery row. Survives one ComfyUI restart.

    Measured on the first live run: the artist restarted ComfyUI mid-run and
    round 5 died on a ConnectError. A restarted ComfyUI also forgets the
    prompt it was given, so the history poll would wait out its whole timeout
    for a picture that is never coming — hence a timeout sized to a render
    (cold load included) rather than to a video job, then one resubmit once
    ComfyUI answers again.
    """
    p = run["params"]
    wf = build_zimage_workflow(prompt, seed, p["width"], p["height"], p["loras"], p["detail"])
    for attempt in (1, 2):
        await gpu.for_comfy()
        gpu.comfy = True
        try:
            outputs = await _submit_and_wait(run, wf, band)
            break
        except (httpx.HTTPError, TimeoutError, RuntimeError) as exc:
            down = isinstance(exc, (httpx.TransportError, TimeoutError)) or "unreachable" in str(exc)
            if attempt == 2 or not down:
                raise
            logger.warning("iterate %s: ComfyUI went away mid-render (%s) — waiting to resubmit", run["id"], exc)
            run["message"] = run.get("message", "") + " — ComfyUI nicht erreichbar, warte…"
            await _wait_for_comfy()
    images = (outputs.get(ZIMAGE_SAVE_NODE) or {}).get("images") or []
    if not images:
        raise RuntimeError("ComfyUI returned no image (SaveImage output missing)")
    entry = images[0]
    rel = f"{entry['subfolder']}/{entry['filename']}" if entry.get("subfolder") else entry["filename"]
    dest, image_id = await ingest_comfy_image(
        rel, prompt=prompt, seed=seed, width=p["width"], height=p["height"], loras=p["loras"],
        detail_amount=p["detail"] or None, workflow_name=ZIMAGE_WORKFLOW_NAME,
        batch_id=uuid.UUID(run["id"]),
    )
    if not dest or not image_id:
        raise RuntimeError("Ingest of the rendered image failed (see server log)")
    return {"image_id": image_id, "path": str(dest), "filename": dest.name,
            "url": f"/api/image/{dest.name}", "thumb_url": f"/api/image/{dest.name}/thumb"}


async def _submit_and_wait(run: dict, wf: dict, band: tuple[float, float]) -> dict:
    async with httpx.AsyncClient(timeout=60) as client:
        for attempt in range(3):
            try:
                prompt_id = await post_workflow(client, wf)
                break
            except _TRANSIENT:
                if attempt == 2:
                    raise
                await asyncio.sleep(4 * (attempt + 1))
        listener = get_listener()
        if listener:
            listener.register_node_labels(prompt_id, wf)
        run["_prompt_id"], run["_band"] = prompt_id, (int(band[0]), int(band[1]))
        try:
            return await poll_history(client, prompt_id, timeout=_RENDER_TIMEOUT, interval=_RENDER_POLL)
        finally:
            run.pop("_prompt_id", None)
            run.pop("_band", None)


async def _wait_for_comfy() -> None:
    deadline = time.monotonic() + _COMFY_BACK_TIMEOUT
    while True:
        try:
            await comfy_devices()
            return
        except RuntimeError:
            if time.monotonic() > deadline:
                raise RuntimeError(f"ComfyUI did not come back within {_COMFY_BACK_TIMEOUT:.0f} s")
            await asyncio.sleep(5)


async def _set_notes(image_id: str, notes: str) -> None:
    """Best-effort: gallery bookkeeping must never cost a run its rounds."""
    try:
        async with AsyncSessionLocal() as db:
            img = await db.get(Image, uuid.UUID(image_id))
            if img is not None:
                img.notes = notes
                await db.commit()
    except Exception:
        logger.warning("iterate: could not write notes on %s", image_id, exc_info=True)


def _series_members(run: dict) -> list[str]:
    """Best first (position 0 is the cover), then every rendered round in
    order, then the sweep. Over the 20 a series may hold, rejected rounds go
    first — they stay in the gallery either way."""
    final = (run.get("final") or {}).get("image_id")
    best = next((r["image_id"] for r in run["rounds"] if r.get("round") == run.get("best_round")), None)
    cover = final or best
    rendered = [r for r in run["rounds"] if r.get("image_id")]
    rest = [r["image_id"] for r in rendered] + [s["image_id"] for s in run.get("sweep", []) if s.get("image_id")]
    rest = [i for i in rest if i != cover]
    if cover and len(rest) + 1 > SERIES_MAX:
        rejected = {r["image_id"] for r in rendered if r.get("decision") not in ("accepted", "first")}
        while len(rest) + 1 > SERIES_MAX and any(i in rejected for i in rest):
            drop = next(i for i in reversed(rest) if i in rejected)
            rest.remove(drop)
        rest = rest[:SERIES_MAX - 1]
    return ([cover] if cover else []) + rest


async def _sync_series(run: dict) -> None:
    """Best-effort, like the notes: the sidecar is the run's record either way."""
    try:
        await _write_series(run)
    except Exception:
        logger.warning("iterate: could not update the series of run %s", run["id"], exc_info=True)


async def _write_series(run: dict) -> None:
    ids = _series_members(run)
    if len(ids) < 2 or run.get("series_gone"):
        return
    async with AsyncSessionLocal() as db:
        existing = set((await db.execute(
            select(Image.id).where(Image.id.in_([uuid.UUID(i) for i in ids]))
        )).scalars().all())
        ids = [i for i in ids if uuid.UUID(i) in existing]
        if len(ids) < 2:
            return
        if run.get("series_id"):
            series = (await db.execute(
                select(ImageSeries).where(ImageSeries.id == uuid.UUID(run["series_id"]))
                .options(selectinload(ImageSeries.items))
            )).scalar_one_or_none()
            if series is None:   # deleted from the gallery mid-run — respect that
                run["series_gone"] = True
                return
            if series.items:
                series.items.clear()
                await db.flush()
        else:
            series = ImageSeries(title=run["title"])
            db.add(series)
        series.notes = _series_notes(run)
        for pos, iid in enumerate(ids):
            series.items.append(ImageSeriesItem(position=pos, image_id=uuid.UUID(iid)))
        await db.commit()
        run["series_id"] = str(series.id)


def _series_notes(run: dict) -> str:
    b = run.get("brief") or {}
    lines = [f"Iterate · {run['params']['rounds']} Runden · {run['params']['mode']}"]
    if b:
        lines.append(f"Subjekt: {b.get('subject', '')} · Twist: {b.get('twist', '')}")
    if run.get("best_round"):
        lines.append(f"Bestes Bild: Runde {run['best_round']} · {run.get('best_total')}")
    if run.get("seeds_passed") is not None:
        lines.append(f"Gegenprobe: {run['seeds_passed']} von {run['seeds_tested']} Seeds bestehen")
    return "\n".join(lines)


# ── Editor context ───────────────────────────────────────────────────────────

def _history_lines(run: dict) -> str:
    lines = []
    for r in run["rounds"]:
        total = f"{r['total']:.2f}" if r.get("total") is not None else "—"
        lines.append(f"{r['round']} · {r.get('operator') or '—'} · {r.get('change') or ''} · "
                     f"{r.get('decision')} · {total}")
    return "\n".join(lines) or "(none)"


def _accepted_line(r: dict) -> str:
    return r.get("reason") or r.get("decision", "")


def _editor_fields(run: dict, anchor: Anchor, triggers: list[str], next_round: int,
                   phase: str, plateau: bool) -> dict:
    b, p = run["brief"], run["params"]
    best = next(r for r in run["rounds"] if r.get("round") == run["best_round"])
    latest = next(r for r in reversed(run["rounds"]) if r.get("report"))
    lr = latest["report"]
    return dict(
        subject=b["subject"], twist=b["twist"], must_keep=fmt_list(b["must_keep"]),
        free=fmt_list(b["free"]), mood=b["mood"], mode=p["mode"],
        anchor_head=anchor.head.replace("{triggers}", ", ".join(triggers)).strip(" ,"),
        anchor_tail=anchor.tail,
        next_round=next_round, rounds=p["rounds"], phase=phase, plateau=str(plateau).lower(),
        best_round=best["round"], best_total=f"{best['total']:.2f}", best_core=best["core"],
        best_scores=fmt_scores(best["report"]["scores"]), best_keep=fmt_list(best["report"]["keep"]),
        round=latest["round"], accepted_line=_accepted_line(latest), latest_core=latest["core"],
        latest_observed=lr["observed"],
        latest_must_keep="; ".join(f"{m['item']}: {m['status']}" for m in lr["must_keep"]) or "(none)",
        latest_scores=fmt_scores(lr["scores"]),
        latest_defects="\n".join(f"{i + 1}. {d}" for i, d in enumerate(lr["defects"])) or "(none)",
        latest_expect=latest.get("expect") or "n/a (round 1)",
        history=_history_lines(run),
    )


def _rounds_since_accept(run: dict) -> int:
    n = 0
    for r in reversed(run["rounds"]):
        if r.get("decision") in ("accepted", "first"):
            break
        n += 1
    return n


# ── The job ──────────────────────────────────────────────────────────────────

def new_run(params: dict, models: Models, title: str) -> dict:
    run_id = str(uuid.uuid4())
    run = {
        "id": run_id, "status": "queued", "message": "Wartet…", "pct": 0,
        "created_at": _now(), "updated_at": _now(), "finished_at": None,
        "title": title, "params": params,
        "models": {"critic": models.critic, "editor": models.editor},
        "anchor": None, "brief": None, "brief_violation": None,
        "rounds": [], "best_round": None, "best_total": None,
        "sweep": [], "final": None, "seeds_passed": None, "seeds_tested": None,
        "series_id": None, "error": None,
    }
    _runs[run_id] = run
    _save(run)
    return run


async def run_iterate(run_id: str, spec: Spec) -> None:
    run = _runs[run_id]
    p = run["params"]
    models = Models(**run["models"])
    gpu = _Gpu(models)
    anchor = spec.anchors[PRESET_FOR_MODE[p["mode"]]]
    triggers = triggers_for(p["loras"])
    run["anchor"] = {"name": anchor.name,
                     "head": anchor.head.replace("{triggers}", ", ".join(triggers)).strip(" ,"),
                     "tail": anchor.tail}
    check_kwargs = dict(anchor_head=anchor.head, anchor_tail=anchor.tail, triggers=triggers)
    n = p["rounds"]
    seed = p["seed"]
    span = 88.0 / n   # rounds own 5–93 %, the sweep 93–99 %

    try:
        run["status"] = "running"
        _set(run, "Brief wird geschrieben…", 1)
        await gpu.for_ollama(models.editor)
        t0 = time.time()
        brief, violation = await run_brief(spec, models, idea=p["idea"], mode=p["mode"],
                                           source_prompt=p.get("source_prompt") or "",
                                           check_kwargs=check_kwargs)
        run["brief"] = brief
        run["brief_secs"] = round(time.time() - t0, 1)
        if violation:
            run["brief_violation"] = {"rule": violation.rule, "message": violation.message}
        _save(run)

        seen: dict[str, int] = {}
        best_jpg: bytes | None = None
        jolts: set[str] = set()
        polish_rejects = 0

        for rnd in range(1, n + 1):
            base = 5 + span * (rnd - 1)
            entry: dict = {"round": rnd, "phase": judge.phase_for(rnd, n) if rnd > 1 else "brief",
                           "timings": {}}
            label = f"Runde {rnd}/{n}"

            # 1 · editor — every round but the first, which is the brief's core
            if rnd == 1:
                entry.update(operator=None, change="first core from the brief", expect=None,
                             diagnosis=None, core=brief["core"])
            else:
                phase = entry["phase"]
                plateau = (_rounds_since_accept(run) >= judge.plateau_after(n) and phase not in jolts)
                if plateau:
                    jolts.add(phase)
                best_r = next(r for r in run["rounds"] if r["round"] == run["best_round"])
                intent_weak = any(m["status"] != "present" for m in best_r["report"]["must_keep"])
                recent = [r["operator"] for r in run["rounds"] if r.get("operator")]
                allowed = judge.allowed_operators(mode=p["mode"], phase=phase, plateau=plateau,
                                                  recent=recent, intent_weak=intent_weak)
                entry.update(plateau=plateau, allowed=allowed)
                _set(run, f"{label} · Regie schreibt den nächsten Prompt…", base)
                await gpu.for_ollama(models.editor)
                t0 = time.time()
                edit, violation = await run_editor(
                    spec, models, _editor_fields(run, anchor, triggers, rnd, phase, plateau),
                    phase=phase, allowed=allowed, mode=p["mode"], must_keep=brief["must_keep"],
                    check_kwargs=check_kwargs, seen_cores=seen,
                )
                entry["timings"]["editor"] = round(time.time() - t0, 1)
                if edit is None:
                    entry.update(decision="skipped", total=None, operator=None, core=None,
                                 change=f"rule broken twice: {violation.rule}",
                                 reason=f"skipped (rule: {violation.rule}) — {violation.message}")
                    run["rounds"].append(entry)
                    _save(run)
                    continue
                entry.update(edit)

            # 2 · render
            entry["prompt"] = compose_prompt(anchor, entry["core"], triggers)
            entry["seed"] = seed
            _set(run, f"{label} · Rendern…", base + span * 0.15)
            t0 = time.time()
            shot = await _render(run, gpu, entry["prompt"], seed, (base + span * 0.15, base + span * 0.6))
            entry["timings"]["render"] = round(time.time() - t0, 1)
            entry.update({k: shot[k] for k in ("image_id", "filename", "url", "thumb_url")})
            seen[normalize_core(entry["core"])] = rnd

            # 3 · critic
            _set(run, f"{label} · Kritik…", base + span * 0.6)
            await gpu.for_ollama(models.critic)
            jpg = await load_jpg(Path(shot["path"]))
            t0 = time.time()
            report = await run_critic(spec, models, jpg, brief, round_no=rnd, rounds=n)
            entry["timings"]["critic"] = round(time.time() - t0, 1)
            entry["report"] = report
            entry["total"] = judge.weighted_total(report["scores"])

            # 4 · decide
            gate = judge.gate_failure(report)
            verdict = judge.triage(entry["total"], run["best_total"])
            if verdict == "first":
                entry["decision"], entry["reason"] = "first", "accepted (round 1 is always the best)"
            elif gate:
                entry["decision"], entry["reason"] = "rejected", f"rejected (gate: {gate})"
            elif verdict == "clear":
                entry["decision"] = "accepted"
                entry["reason"] = f"accepted (clear win, {entry['total'] - run['best_total']:+.2f})"
            elif verdict == "reject":
                entry["decision"] = "rejected"
                entry["reason"] = f"rejected (score {entry['total'] - run['best_total']:+.2f} below the best)"
            else:
                _set(run, f"{label} · Duell (knapp)…", base + span * 0.9)
                t0 = time.time()
                d1 = await run_duel(spec, models, jpg, best_jpg, brief)   # candidate = A
                d2 = await run_duel(spec, models, best_jpg, jpg, brief)   # candidate = B
                entry["timings"]["duel"] = round(time.time() - t0, 1)
                outcome = judge.duel_verdict(d1["winner"], d2["winner"])
                entry["duel"] = {"candidate_first": d1, "candidate_second": d2, "outcome": outcome}
                if outcome == "candidate":
                    entry["decision"] = "accepted"
                    entry["reason"] = "accepted (close call, won the duel in both orders)"
                else:
                    entry["decision"] = "rejected"
                    entry["reason"] = ("rejected (close call, lost the duel in both orders)" if outcome == "best"
                                       else "rejected (close call, duel split — the best stays)")

            if entry["decision"] in ("accepted", "first"):
                run["best_round"], run["best_total"] = rnd, entry["total"]
                best_jpg = jpg
                polish_rejects = 0
            elif entry["phase"] == "polish":
                polish_rejects += 1

            run["rounds"].append(entry)
            await _set_notes(entry["image_id"],
                             f"iterate · round {rnd}/{n} · {entry.get('operator') or 'brief'} · "
                             f"{entry['decision']} · {entry['total']:.2f}")
            await _sync_series(run)
            _save(run)

            if p.get("early_stop") and polish_rejects >= 2:
                run["stopped_early_at"] = rnd
                break

        await _sweep(run, spec, models, gpu, anchor, triggers, brief, best_jpg)

        run["status"] = "done"
        _set(run, "Fertig", 100, finished_at=_now())
        await _sync_series(run)
        _save(run)
    except asyncio.CancelledError:
        run["status"] = "cancelled"
        _set(run, "Abgebrochen", run.get("pct", 0), finished_at=_now())
        _save(run)
        raise
    except Exception as exc:
        logger.exception("Iterate run %s failed", run_id)
        msg = str(exc).strip()
        run["status"] = "failed"
        run["error"] = (f"{type(exc).__name__}: {msg}" if msg else type(exc).__name__)[:1000]
        _set(run, "Fehlgeschlagen", run.get("pct", 0), finished_at=_now())
        _save(run)
    finally:
        # Leave the cards free for whatever the artist does next.
        for m in {models.critic, models.editor}:
            await unload_model(m)


async def _sweep(run: dict, spec: Spec, models: Models, gpu: _Gpu, anchor: Anchor,
                 triggers: list[str], brief: dict, best_jpg: bytes | None) -> None:
    """Render the best prompt with more seeds: does the *prompt* hold up, or
    did one seed carry it? The strongest sweep image duels the best."""
    p = run["params"]
    k = p["sweep"]
    best = next((r for r in run["rounds"] if r.get("round") == run["best_round"]), None)
    if not k or best is None or best_jpg is None:
        return
    rng = random.Random(p["seed"])
    passed = 0 if judge.gate_failure(best["report"]) else 1
    candidates = []
    for i in range(k):
        seed = rng.randint(0, 2**32 - 1)
        lo = 93 + 6 * i / k
        _set(run, f"Gegenprobe · Seed {i + 1}/{k} · Rendern…", lo)
        shot = await _render(run, gpu, best["prompt"], seed, (lo, lo + 3 / k))
        _set(run, f"Gegenprobe · Seed {i + 1}/{k} · Kritik…", lo + 3 / k)
        await gpu.for_ollama(models.critic)
        jpg = await load_jpg(Path(shot["path"]))
        report = await run_critic(spec, models, jpg, brief, round_no=p["rounds"], rounds=p["rounds"])
        total = judge.weighted_total(report["scores"])
        gate = judge.gate_failure(report)
        passed += 0 if gate else 1
        item = {"seed": seed, **{k2: shot[k2] for k2 in ("image_id", "filename", "url", "thumb_url")},
                "report": report, "total": total, "gate": gate}
        run["sweep"].append(item)
        await _set_notes(shot["image_id"], f"iterate · sweep seed {i + 1}/{k} · best prompt of round "
                                           f"{best['round']} · {total:.2f}" + (f" · {gate}" if gate else ""))
        if not gate:
            candidates.append((total, jpg, item))
        _save(run)

    run["seeds_passed"], run["seeds_tested"] = passed, k + 1
    final = {k2: best[k2] for k2 in ("image_id", "filename", "url", "thumb_url")}
    final.update(source="round", round=best["round"], total=best["total"])
    if candidates:
        total, jpg, item = max(candidates, key=lambda c: c[0])
        _set(run, "Gegenprobe · Duell gegen das beste Bild…", 99)
        await gpu.for_ollama(models.critic)
        d1 = await run_duel(spec, models, jpg, best_jpg, brief)
        d2 = await run_duel(spec, models, best_jpg, jpg, brief)
        outcome = judge.duel_verdict(d1["winner"], d2["winner"])
        run["sweep_duel"] = {"seed": item["seed"], "candidate_first": d1, "candidate_second": d2,
                             "outcome": outcome}
        if outcome == "candidate":
            final = {k2: item[k2] for k2 in ("image_id", "filename", "url", "thumb_url")}
            final.update(source="sweep", seed=item["seed"], total=total)
    run["final"] = final
    _save(run)

