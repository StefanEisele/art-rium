"""
Music generation — ACE-Step 1.5 Turbo workflow.

Single workflow type: a text-to-audio pipeline that takes a style/genre tags
prompt, optional lyrics, duration, bpm, language, and musical key, and
produces an MP3.

POST /api/music/generate            → enqueue job, return {song_id}
GET  /api/music/jobs/{id}           → poll status
GET  /api/music/jobs/{id}/progress  → lightweight progress (ComfyUI queue + phase)
GET  /api/music/thumb/{id}          → waveform PNG (404 if not generated)
GET  /api/music/file/{fname}        → serve MP3
GET  /api/music                     → list all songs
PATCH /api/music/jobs/{id}          → update user-editable fields (title/notes)
DELETE /api/music/{id}              → delete

Mirrors the architectural shape of routers/video.py: the router file holds
both the HTTP endpoints and the background ComfyUI submission/poll loop,
with a module-level _progress dict for live status.
"""
import asyncio
import logging
import random
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from core.auth import require_auth
from core.config import settings
from core.db import AsyncSessionLocal, get_db
from core.models import Song
from services.comfy.client import poll_history, post_workflow, queue_info

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/music", dependencies=[Depends(require_auth)])

# ── Per-job progress (module-level, single-process safe) ──────────────────────
# str(song_id) → {"phase": str, "message": str, "pct": int}
_progress: dict[str, dict] = {}

# ── Workflow constants ───────────────────────────────────────────────────────

_UNET_NAME = "acestep_v1.5_turbo.safetensors"
_CLIP_NAME_1 = "qwen_0.6b_ace15.safetensors"
_CLIP_NAME_2 = "qwen_1.7b_ace15.safetensors"
_VAE_NAME = "ace_1.5_vae.safetensors"
_WORKFLOW_NAME = "ace_step_1.5_turbo"

# KSampler / model defaults pulled from the source workflow
# (workflows/audio_ace_step_1_5_split (1).json).
_DEFAULT_STEPS = 8           # turbo
_DEFAULT_CFG = 1.0
_DEFAULT_SHIFT = 3.0         # ModelSamplingAuraFlow
_SAMPLER_NAME = "euler"
_SCHEDULER = "simple"

SAVE_NODE_ID = "save"
POLL_INTERVAL = 5            # ACE turbo is fast; tighter polling than video
POLL_TIMEOUT = 600           # 10 minutes max

# ── Pydantic ─────────────────────────────────────────────────────────────────

class GenerateMusicRequest(BaseModel):
    tags: str = Field(min_length=1, max_length=2000,
                      description="ACE 'tags' prompt — style / genre / mood / instruments.")
    lyrics: str | None = Field(default=None, max_length=4000)
    duration_seconds: int = Field(default=25, ge=5, le=240)
    bpm: int = Field(default=120, ge=40, le=240)
    musical_key: str | None = Field(default=None, max_length=32,
                                    description='e.g. "E minor", "C major", or empty for auto.')
    language: str = Field(default="en", max_length=8)
    time_signature: str = Field(default="4", max_length=8)
    seed: int | None = Field(default=None, ge=0, le=2**32 - 1,
                             description="Optional fixed seed; random if omitted.")
    steps: int = Field(default=_DEFAULT_STEPS, ge=4, le=50)
    cfg: float = Field(default=_DEFAULT_CFG, ge=0.0, le=10.0)
    shift: float = Field(default=_DEFAULT_SHIFT, ge=0.0, le=10.0)


class SongUpdate(BaseModel):
    title: str | None = None
    notes: str | None = None


# ── Workflow builder ─────────────────────────────────────────────────────────

def _build_ace_step_workflow(req: GenerateMusicRequest, seed: int, save_prefix: str) -> dict:
    """Convert the source workflow JSON into ComfyUI API form.

    Nodes mirror workflows/audio_ace_step_1_5_split (1).json exactly:
      UNETLoader → ModelSamplingAuraFlow → DualCLIPLoader → TextEncodeAceStepAudio1.5
      → ConditioningZeroOut (negative) → EmptyAceStep1.5LatentAudio → KSampler
      → VAELoader → VAEDecodeAudio → SaveAudioMP3

    The TextEncodeAceStepAudio1.5 widget surface has more knobs than we expose
    (use_lyric_normalization, guidance scales, etc). We send the documented
    musical params and let the rest fall back to node defaults.
    """
    duration = float(req.duration_seconds)
    return {
        "unet": {"class_type": "UNETLoader", "inputs": {
            "unet_name":    _UNET_NAME,
            "weight_dtype": "default",
        }},
        "clip": {"class_type": "DualCLIPLoader", "inputs": {
            "clip_name1": _CLIP_NAME_1,
            "clip_name2": _CLIP_NAME_2,
            "type":       "ace",
            "device":     "default",
        }},
        "vae": {"class_type": "VAELoader", "inputs": {
            "vae_name": _VAE_NAME,
        }},
        "sampling": {"class_type": "ModelSamplingAuraFlow", "inputs": {
            "model": ["unet", 0],
            "shift": float(req.shift),
        }},
        "latent": {"class_type": "EmptyAceStep1.5LatentAudio", "inputs": {
            "seconds":    duration,
            "batch_size": 1,
        }},
        "encode": {"class_type": "TextEncodeAceStepAudio1.5", "inputs": {
            "clip":                 ["clip", 0],
            "tags":                 req.tags,
            "lyrics":               req.lyrics or "",
            "seed":                 int(seed),
            "bpm":                  int(req.bpm),
            "duration":             duration,
            # COMBO inputs — must match the node's exact option strings.
            "timesignature":        req.time_signature or "4",
            "language":             req.language or "en",
            "keyscale":             req.musical_key or "C major",
            # Advanced LLM-sampler params; node treats them as required even
            # though they are marked "advanced". Defaults mirror the workflow.
            "generate_audio_codes": True,
            "cfg_scale":            2.0,
            "temperature":          0.85,
            "top_p":                0.9,
            "top_k":                0,
            "min_p":                0.0,
        }},
        "neg": {"class_type": "ConditioningZeroOut", "inputs": {
            "conditioning": ["encode", 0],
        }},
        "sampler": {"class_type": "KSampler", "inputs": {
            "model":        ["sampling", 0],
            "positive":     ["encode", 0],
            "negative":     ["neg", 0],
            "latent_image": ["latent", 0],
            "seed":         int(seed),
            "steps":        int(req.steps),
            "cfg":          float(req.cfg),
            "sampler_name": _SAMPLER_NAME,
            "scheduler":    _SCHEDULER,
            "denoise":      1.0,
        }},
        "decode": {"class_type": "VAEDecodeAudio", "inputs": {
            "samples": ["sampler", 0],
            "vae":     ["vae", 0],
        }},
        SAVE_NODE_ID: {"class_type": "SaveAudioMP3", "inputs": {
            "audio":           ["decode", 0],
            "filename_prefix": save_prefix,
            "quality":         "V0",
        }},
    }


# ── Helpers ──────────────────────────────────────────────────────────────────

def _set_progress(song_key: str, phase: str, message: str, pct: int) -> None:
    _progress[song_key] = {"phase": phase, "message": message, "pct": pct}


async def _finalize_failure(song_id: uuid.UUID, exc: Exception, song_key: str) -> None:
    """Clear progress, record the exception on the Song row."""
    _progress.pop(song_key, None)
    msg = str(exc).strip()
    err = f"{type(exc).__name__}: {msg}" if msg else type(exc).__name__
    async with AsyncSessionLocal() as db:
        song = await db.get(Song, song_id)
        if song:
            song.status = "failed"
            song.error = err[:1000]
            await db.commit()


async def _save_comfy_prompt_id(song_id: uuid.UUID, prompt_id: str) -> None:
    async with AsyncSessionLocal() as db:
        song = await db.get(Song, song_id)
        if song:
            song.comfy_prompt_id = prompt_id
            await db.commit()


def _resolve_audio_output(save_out: dict) -> Path:
    """Resolve the ComfyUI-side output filename from a SaveAudioMP3 save node.

    The output dict shape mirrors VHS_VideoCombine but uses the 'audio' key
    instead of 'gifs'/'videos'. Falls through several common keys so a future
    ComfyUI version that renames the slot still works."""
    entries = (
        save_out.get("audio")
        or save_out.get("audios")
        or save_out.get("ui", {}).get("audio")
        or []
    )
    if not entries:
        raise RuntimeError(f"SaveAudioMP3 output missing: {save_out}")
    entry = entries[0]
    src = settings.comfyui_output_dir / entry.get("subfolder", "") / entry["filename"]
    if not src.exists():
        raise FileNotFoundError(f"Audio output not found at {src}")
    return src


# ── Background generation task ───────────────────────────────────────────────

async def _run_generation(song_id: uuid.UUID, req: GenerateMusicRequest, seed: int) -> None:
    song_key = str(song_id)
    save_prefix = f"audio/artrium_{song_id.hex[:10]}"

    try:
        _set_progress(song_key, "submitting", "Submitting workflow to ComfyUI…", 5)
        wf = _build_ace_step_workflow(req, seed, save_prefix)

        async with httpx.AsyncClient(timeout=60) as client:
            prompt_id = await post_workflow(client, wf)
            logger.info("Music job %s → ComfyUI prompt %s", song_id, prompt_id)
            await _save_comfy_prompt_id(song_id, prompt_id)

            _set_progress(song_key, "queued", "Waiting in ComfyUI queue…", 15)
            outputs = await poll_history(
                client, prompt_id, timeout=POLL_TIMEOUT, interval=POLL_INTERVAL,
            )

        _set_progress(song_key, "finalizing", "Saving audio…", 90)
        comfy_src = _resolve_audio_output(outputs.get(SAVE_NODE_ID, {}))

        settings.songs_dir.mkdir(parents=True, exist_ok=True)
        dest = settings.songs_dir / f"{song_id}_artrium.mp3"
        await asyncio.to_thread(shutil.copy2, comfy_src, dest)
        rel_path = dest.relative_to(settings.storage_dir)
        logger.info("Music stored: %s", dest)

        async with AsyncSessionLocal() as db:
            song = await db.get(Song, song_id)
            if song:
                song.filename = dest.name
                song.filepath = str(rel_path)
                song.status   = "done"
                song.error    = None
                await db.commit()

        _progress.pop(song_key, None)

    except Exception as exc:
        logger.exception("Music generation %s failed", song_id)
        await _finalize_failure(song_id, exc, song_key)


# ── Endpoints ────────────────────────────────────────────────────────────────

@router.post("/generate", status_code=202)
async def generate_music(body: GenerateMusicRequest, db: AsyncSession = Depends(get_db)):
    seed = body.seed if body.seed is not None else random.randint(0, 2**32 - 1)

    song = Song(
        id=uuid.uuid4(),
        tags=body.tags,
        lyrics=body.lyrics,
        duration_seconds=body.duration_seconds,
        bpm=body.bpm,
        musical_key=body.musical_key,
        language=body.language,
        seed=seed,
        steps=body.steps,
        cfg=body.cfg,
        shift=body.shift,
        status="generating",
        workflow=_WORKFLOW_NAME,
        created_at=datetime.now(timezone.utc),
    )
    db.add(song)
    await db.commit()
    await db.refresh(song)

    asyncio.create_task(_run_generation(song.id, body, seed))
    logger.info("Queued music generation job %s (%ds, bpm=%d, key=%s)",
                song.id, body.duration_seconds, body.bpm, body.musical_key or "(auto)")

    return {"song_id": str(song.id), "status": "generating", "seed": seed}


@router.get("/jobs/{song_id}/progress")
async def get_job_progress(song_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """Lightweight progress — reads module-level dict + optional ComfyUI queue check."""
    song = await db.get(Song, song_id)
    if not song:
        raise HTTPException(status_code=404, detail="Song job not found")

    if song.status == "done":
        return {"phase": "done", "message": "Complete", "pct": 100}
    if song.status == "failed":
        return {"phase": "failed", "message": song.error or "Generation failed", "pct": 0}

    prog = dict(_progress.get(str(song_id), {"phase": "processing", "message": "Processing…", "pct": 25}))

    # Enrich with live ComfyUI queue info while the prompt is submitted
    if song.comfy_prompt_id and prog["phase"] in ("queued", "submitting", "processing"):
        qi = await queue_info(song.comfy_prompt_id)
        if qi["status"] == "running":
            prog["phase"]   = "running"
            prog["message"] = "ComfyUI: generating audio…"
            prog["pct"]     = max(prog["pct"], 40)
        elif qi["status"] == "pending":
            pos = qi.get("position", "?")
            prog["phase"]   = "queued"
            prog["message"] = f"Queued in ComfyUI (position {pos})…"
            prog["pct"]     = 15
        prog["queue"] = qi

    return prog


@router.get("/jobs/{song_id}")
async def get_job(song_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    song = await db.get(Song, song_id)
    if not song:
        raise HTTPException(status_code=404, detail="Song job not found")
    return _serialize(song)


@router.patch("/jobs/{song_id}")
async def update_song(
    song_id: uuid.UUID,
    body: SongUpdate,
    db: AsyncSession = Depends(get_db),
):
    """Update user-editable fields. Empty string clears the field; null is ignored."""
    song = await db.get(Song, song_id)
    if not song:
        raise HTTPException(status_code=404, detail="Song not found")
    if "title" in body.model_fields_set:
        song.title = (body.title or "").strip()[:255] or None
    if "notes" in body.model_fields_set:
        song.notes = (body.notes or "").strip() or None
    await db.commit()
    await db.refresh(song)
    return _serialize(song)


@router.get("/thumb/{song_id}")
async def song_thumbnail(song_id: uuid.UUID):
    """Optional waveform PNG. 404 when not generated — the UI falls back to a static icon."""
    p = settings.songs_dir / f"{song_id}_waveform.png"
    if p.exists():
        return FileResponse(p, media_type="image/png")
    raise HTTPException(status_code=404, detail="Waveform not available")


@router.get("")
async def list_songs(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Song).order_by(desc(Song.created_at)))
    return [_serialize(s) for s in result.scalars().all()]


@router.delete("/{song_id}", status_code=204)
async def delete_song(song_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    song = await db.get(Song, song_id)
    if not song:
        raise HTTPException(status_code=404, detail="Song not found")
    if song.filepath:
        p = settings.storage_dir / song.filepath
        if p.exists():
            p.unlink(missing_ok=True)
    waveform = settings.songs_dir / f"{song_id}_waveform.png"
    waveform.unlink(missing_ok=True)
    await db.delete(song)
    await db.commit()


@router.get("/file/{filename}")
async def serve_song(filename: str):
    safe = Path(filename).name
    p = settings.songs_dir / safe
    if p.exists():
        return FileResponse(p, media_type="audio/mpeg")
    raise HTTPException(status_code=404, detail="Song not found")


# ── Serializer ───────────────────────────────────────────────────────────────

def _serialize(s: Song) -> dict:
    return {
        "id":               str(s.id),
        "status":           s.status,
        "filename":         s.filename,
        "url":              f"/api/music/file/{s.filename}" if s.filename else None,
        "thumb_url":        f"/api/music/thumb/{s.id}" if s.status == "done" else None,
        "tags":             s.tags,
        "lyrics":           s.lyrics,
        "duration_seconds": s.duration_seconds,
        "bpm":              s.bpm,
        "musical_key":      s.musical_key,
        "language":         s.language,
        "seed":             s.seed,
        "steps":            s.steps,
        "cfg":              float(s.cfg) if s.cfg is not None else None,
        "shift":            float(s.shift) if s.shift is not None else None,
        "title":            s.title,
        "notes":            s.notes,
        "error":            s.error,
        "workflow":         s.workflow,
        "created_at":       s.created_at.isoformat(),
    }
