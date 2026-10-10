"""Iterate — start, watch and stop the render → critic → editor loop.

The loop itself is services/iterate/runner.py; its design and the prompts it
sends are prompts/iterate.md. One run at a time: every round hands the GPU
back and forth between ComfyUI and Ollama, and a second run would fight the
first for the same card.
"""
from __future__ import annotations

import logging
import uuid
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from core.auth import require_auth
from core.config import settings
from core.db import AsyncSessionLocal
from core.job_control import cancel_job
from core.loras import ALLOWED_LORAS
from core.models import Image
from core.tasks import safe_create_task
from services.comfy import zimage as zimage_service
from services.iterate import runner
from services.iterate.roles import Models
from services.iterate.spec import load_spec

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/iterate", dependencies=[Depends(require_auth)])

IDEA_MAX = 1000
SIZE_MIN, SIZE_MAX = 256, 2048


def _default_model(configured: str) -> str:
    return configured or settings.ollama_llm_model


def critic_choices() -> list[dict]:
    """What the critic may be. Measured 2026-10-08 against the artist's own
    ratings (prompts/iterate.md, "Calibration") — see the notes."""
    choices = [
        {"model": "qwen3-vl:8b-instruct", "label": "Qwen3-VL 8B", "note": "empfohlen · 8 s je Bild"},
        {"model": settings.ollama_llm_model, "label": "Gemma 3 27B", "note": "strenger · 41 s, Duell-blind"},
    ]
    default = _default_model(settings.ollama_iterate_critic_model)
    if default not in {c["model"] for c in choices}:
        choices.insert(0, {"model": default, "label": default, "note": "aus .env"})
    return choices


class IterateLora(BaseModel):
    name: str
    strength: float = 0.5


class IterateRequest(BaseModel):
    idea: str = ""
    source_image_id: uuid.UUID | None = None
    mode: Literal["formula", "prose"] = "formula"
    rounds: int = runner.ROUNDS_DEFAULT
    sweep: int = runner.SWEEP_DEFAULT
    # None = the signature stack (formula) or no LoRA (prose); a source
    # picture's own stack wins over both.
    loras: list[IterateLora] | None = None
    width: int = 1080
    height: int = 1920
    seed: int = -1
    detail: float | None = None
    early_stop: bool = False
    critic_model: str | None = None


@router.get("/options")
async def iterate_options():
    return {
        "rounds": {"min": runner.ROUNDS_MIN, "max": runner.ROUNDS_MAX, "default": runner.ROUNDS_DEFAULT},
        "sweep": {"min": runner.SWEEP_MIN, "max": runner.SWEEP_MAX, "default": runner.SWEEP_DEFAULT},
        "signature_loras": runner.SIGNATURE_LORAS,
        "critics": critic_choices(),
        "default_critic": _default_model(settings.ollama_iterate_critic_model),
        "editor": _default_model(settings.ollama_iterate_editor_model),
        "active_run": runner.active_run_id(),
    }


@router.post("", status_code=202)
async def start_iterate(body: IterateRequest):
    if (active := runner.active_run_id()) is not None:
        raise HTTPException(status_code=409, detail=f"Ein Lauf ist schon aktiv ({active})")
    idea = body.idea.strip()
    if len(idea) > IDEA_MAX:
        raise HTTPException(status_code=400, detail=f"Idee höchstens {IDEA_MAX} Zeichen")
    if not idea and body.source_image_id is None:
        raise HTTPException(status_code=400, detail="Eine Idee oder ein Startbild ist nötig")
    if not (runner.ROUNDS_MIN <= body.rounds <= runner.ROUNDS_MAX):
        raise HTTPException(status_code=400, detail=f"Runden {runner.ROUNDS_MIN}–{runner.ROUNDS_MAX}")
    if not (runner.SWEEP_MIN <= body.sweep <= runner.SWEEP_MAX):
        raise HTTPException(status_code=400, detail=f"Gegenprobe {runner.SWEEP_MIN}–{runner.SWEEP_MAX} Seeds")
    if not (SIZE_MIN <= body.width <= SIZE_MAX and SIZE_MIN <= body.height <= SIZE_MAX):
        raise HTTPException(status_code=400, detail=f"Größe {SIZE_MIN}–{SIZE_MAX} px")
    critic = body.critic_model or _default_model(settings.ollama_iterate_critic_model)
    if critic not in {c["model"] for c in critic_choices()}:
        raise HTTPException(status_code=400, detail=f"Unbekannter Kritiker: {critic}")

    # A broken edit of prompts/iterate.md fails here, not three rounds in.
    try:
        spec = load_spec()
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    source_prompt, source_seed, source_loras, source_title = "", None, None, None
    if body.source_image_id is not None:
        async with AsyncSessionLocal() as db:
            img = await db.get(Image, body.source_image_id)
            if img is None:
                raise HTTPException(status_code=404, detail="Startbild nicht gefunden")
            source_prompt, source_seed, source_loras, source_title = img.prompt or "", img.seed, img.loras, img.title

    if body.loras is not None:
        loras = [{"name": sel.name, "strength": round(max(0.0, min(1.5, sel.strength)), 3)} for sel in body.loras]
    elif source_loras:
        loras = source_loras
    else:
        loras = list(runner.SIGNATURE_LORAS) if body.mode == "formula" else []
    for sel in loras:
        if sel["name"] not in ALLOWED_LORAS:
            raise HTTPException(status_code=400, detail=f"Unbekannte LoRA: {sel['name']}")

    seed = body.seed if body.seed >= 0 else (source_seed if source_seed is not None and source_seed >= 0
                                               else uuid.uuid4().int % 2**32)
    params = {
        "idea": idea, "source_image_id": str(body.source_image_id) if body.source_image_id else None,
        "source_prompt": source_prompt, "mode": body.mode, "rounds": body.rounds, "sweep": body.sweep,
        "loras": loras, "width": body.width, "height": body.height, "seed": seed,
        "detail": zimage_service.clamp_detail(body.detail), "early_stop": body.early_stop,
    }
    title = "Iterate · " + ((idea[:60] + ("…" if len(idea) > 60 else "")) if idea else (source_title or "Startbild"))
    models = Models(critic=critic, editor=_default_model(settings.ollama_iterate_editor_model))
    run = runner.new_run(params, models, title)
    safe_create_task(runner.run_iterate(run["id"], spec), name=f"iterate:{run['id']}")
    logger.info("Queued iterate run %s (%d rounds, %s, critic=%s)", run["id"], body.rounds, body.mode, critic)
    return runner.public_view(run)


@router.get("")
async def list_iterate_runs(limit: int = 30):
    return {"runs": runner.list_runs(max(1, min(100, limit)))}


@router.get("/{run_id}")
async def get_iterate_run(run_id: uuid.UUID):
    run = runner.load_run(str(run_id))
    if run is None:
        raise HTTPException(status_code=404, detail="Lauf nicht gefunden")
    return run


@router.post("/{run_id}/cancel")
async def cancel_iterate_run(run_id: uuid.UUID):
    live = runner.live_run(str(run_id))
    if live is None:
        raise HTTPException(status_code=404, detail="Lauf läuft nicht (mehr)")
    await cancel_job(run_id, prompt_ids=[live.get("_prompt_id")])
    return runner.public_view(live)


@router.delete("/{run_id}", status_code=204)
async def delete_iterate_run(run_id: uuid.UUID):
    """Forget a run's record. Its pictures and series stay in the gallery."""
    rid = str(run_id)
    if runner.active_run_id() == rid:
        raise HTTPException(status_code=409, detail="Lauf ist aktiv — erst abbrechen")
    if not runner.forget_run(rid):
        raise HTTPException(status_code=404, detail="Lauf nicht gefunden")
