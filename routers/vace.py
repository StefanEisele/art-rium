"""
VACE structure-video API — geometry decides the form, reference images decide
the material.

Two halves, and the split matters:

**Control tracks are assets.** A Blender turntable gets rendered at a dozen
strengths while the look is being found, so it is uploaded once, probed, and
referenced by id from then on. That is what `control_tracks` is for.

**A job is a sequence of passes.** With no regions it is one pass. With regions
it is one pass per region, each taking the *previous pass's render* as its
control video — see services/comfy/vace.py for why chaining masked blocks into
a single pass renders the depth map instead.

**AnimateLCM renders in two stages by default.** Its base pass costs about a
third of the graph and already decides everything worth deciding — the
composition, the motion, whether the reference picture's material came through
at all — while the hires pass that follows is the expensive two thirds and only
adds detail to whatever the base produced. So the job stops after the base,
parks in status `review` with a watchable preview, and waits: finish it, keep
the preview as it is, or throw it away. See `_run_lcm_preview` and `finish_lcm`.

Progress deliberately reuses routers/video.py's dict and its
`GET /api/video/jobs/{id}/progress`, rather than growing a second polling
mechanism the frontend would have to learn.
"""
import asyncio
import json
import logging
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.auth import require_auth
from core.config import settings
from core.db import AsyncSessionLocal, get_db
from core.models import ControlTrack, Image, Video
from core.tasks import safe_create_task
from core.video_thumb import (
    make_video_thumbnail,
    probe_video_dimensions,
    probe_video_frames,
)
from routers.video import _progress, _segments_dir, _set_progress
from services.comfy.client import free_memory, poll_history, post_workflow
from services.comfy.animatelcm import (
    DEPTH_SWEEP as LCM_DEPTH_SWEEP,
)
from services.comfy.animatelcm import (
    HIRES_SWEEP as LCM_HIRES_SWEEP,
)
from services.comfy.animatelcm import (
    IP_SWEEP as LCM_IP_SWEEP,
)
from services.comfy.animatelcm import (
    AnimateLcmRequest,
)
from services.comfy.animatelcm import (
    DEPTH_DEFAULT as LCM_DEPTH_DEFAULT,
)
from services.comfy.animatelcm import (
    HIRES_DENOISE_DEFAULT as LCM_HIRES_DEFAULT,
)
from services.comfy.animatelcm import (
    IP_DEFAULT as LCM_IP_DEFAULT,
)
from services.comfy.animatelcm import (
    Region as LcmRegion,
)
from services.comfy.animatelcm import (
    build_animatelcm_base_workflow,
    build_animatelcm_hires_workflow,
    build_animatelcm_workflow,
    resolve_seed,
)
from services.comfy.animatelcm import (
    canvas_size as lcm_canvas,
)
from services.comfy.vace import (
    DEFAULT_ASPECT,
    DEFAULT_CANVAS,
    DEFAULT_FIT,
    DEFAULT_RECIPE,
    FITS,
    FPS,
    STRENGTH_DEFAULT,
    STRENGTH_MAX,
    STRENGTH_MIN,
    SWEET_MAX,
    SWEET_MIN,
    Region,
    VaceRequest,
    aspect_options,
    build_vace_workflow,
    canvas_options,
    canvas_size,
    clamp_strength,
    describe_strength,
    estimate_seconds,
    loop_length,
    plan_region_passes,
    recipe_options,
    snap_length,
    strength_bands,
)
from services.image.rendition import original_path
from workers.comfy_listener import get_listener

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/vace", dependencies=[Depends(require_auth)])

WORKFLOW_NAME = "vace_control"
_KINDS = ("depth", "footage", "mask")
_JOB_TIMEOUT = 7200          # a 720 region sequence is three ~12-minute passes
_MAX_UPLOAD_BYTES = 600 * 1024 * 1024


class RegionSpec(BaseModel):
    color: tuple[int, int, int]
    image_id: uuid.UUID              # reference picture, from the gallery
    strength: float = STRENGTH_DEFAULT
    threshold: int = 20


class GenerateRequest(BaseModel):
    control_track_id: uuid.UUID
    prompt: str
    mask_track_id: Optional[uuid.UUID] = None
    image_id: Optional[uuid.UUID] = None      # reference for the no-region case
    regions: list[RegionSpec] = []
    # Which graph renders this. `animatelcm` is the rebuild of the user's own
    # AnimateDiff workflow and is the default: measured 2026-08-19, VACE cannot
    # carry material from a reference picture, because it repaints from noise
    # where the other refines an existing frame at denoise 0.4.
    engine: str = "animatelcm"                # animatelcm | vace
    canvas: str = DEFAULT_CANVAS              # vace: render size
    aspect: str = DEFAULT_ASPECT              # vace: wide | tall | square
    recipe: str = DEFAULT_RECIPE              # vace: sampler
    fit: str = DEFAULT_FIT                    # pad | crop | stretch

    # ── AnimateLCM dials ─────────────────────────────────────────────────────
    # "auto" follows the control track's own shape, which is what makes a
    # portrait clip come back portrait without anyone choosing a format.
    lcm_aspect: str = "auto"
    depth_strength: float = LCM_DEPTH_DEFAULT
    ip_weight: float = LCM_IP_DEFAULT
    hires: bool = True
    hires_denoise: float = LCM_HIRES_DEFAULT
    rife: int = 1
    # Stop after the base pass and wait for a verdict. Only meaningful with
    # `hires` on — without it there is no second stage to decide about, and the
    # job simply finishes.
    preview_first: bool = True
    # Ordinary footage rather than a rendered depth pass: DepthAnythingV2
    # derives the depth in-graph. Set from the track's own `kind`, not by hand.
    derive_depth: bool = False
    strength: float = STRENGTH_DEFAULT
    length: Optional[int] = None              # None = as much of the track as Wan takes
    seed: int = -1
    negative: Optional[str] = None


# Valid values for the three axes, read off the builder's own tables so a new
# canvas or recipe never has to be declared twice.
CANVASES = {c["key"] for c in canvas_options()}
ASPECTS = {a["key"] for a in aspect_options()}
RECIPES = {r["key"] for r in recipe_options()}


# ── Control tracks ───────────────────────────────────────────────────────────

def _track_rel(track_id: uuid.UUID, suffix: str) -> str:
    return f"control/{track_id}{suffix}"


def _serialize_track(t: ControlTrack) -> dict:
    return {
        "id": str(t.id),
        "filename": t.filename,
        "kind": t.kind,
        "title": t.title,
        "width": t.width,
        "height": t.height,
        "frame_count": t.frame_count,
        "fps": t.fps,
        # What a job on this track can actually ask for: Wan samples 4n+1
        # frames and VACE pads anything beyond the track with flat grey.
        "usable_frames": loop_length(t.frame_count) if t.frame_count else None,
        "url": f"/api/vace/tracks/{t.id}/file",
        "thumb_url": f"/api/vace/tracks/{t.id}/thumb" if t.thumbnail_path else None,
        "created_at": t.created_at.isoformat(),
    }


async def _transcode_to_mp4(src: Path, dest: Path) -> None:
    """Re-wrap a recording as H.264/mp4. `-an` because a control track's audio
    is never used and would only travel through the rest of the pipeline."""
    proc = await asyncio.create_subprocess_exec(
        settings.ffmpeg_path, "-y", "-v", "error", "-i", str(src),
        "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "16",
        "-pix_fmt", "yuv420p", str(dest),
        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
    )
    _, err = await proc.communicate()
    if proc.returncode != 0 or not dest.is_file():
        raise RuntimeError(err.decode(errors="replace")[:400] or "ffmpeg failed")


@router.post("/tracks", status_code=201)
async def upload_track(
    file: UploadFile = File(...),
    kind: str = Form("depth"),
    title: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db),
):
    """Take a control track into the library: store, probe, thumbnail.

    The probe is not a nicety. `frame_count` is what later clamps a job's
    length, and without it a request for more frames than the track holds comes
    back with a silently unguided tail.
    """
    if kind not in _KINDS:
        raise HTTPException(status_code=400, detail=f"kind must be one of {_KINDS}")

    track_id = uuid.uuid4()
    suffix = Path(file.filename or "").suffix.lower() or ".mp4"
    if suffix not in (".mp4", ".mov", ".webm", ".mkv", ".avi"):
        raise HTTPException(status_code=400, detail=f"Unsupported container: {suffix}")

    settings.control_dir.mkdir(parents=True, exist_ok=True)
    rel = _track_rel(track_id, suffix)
    dest = settings.storage_dir / rel

    size = 0
    try:
        with open(dest, "wb") as out:
            while chunk := await file.read(4 * 1024 * 1024):
                size += len(chunk)
                if size > _MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="Control track too large")
                out.write(chunk)
    except Exception:
        dest.unlink(missing_ok=True)
        raise

    # A browser recording arrives as VP8/VP9 in a webm container, and
    # VHS_LoadVideoPath does not read those reliably — it is the loader for the
    # whole structure pipeline, so the container is normalised here rather than
    # discovered to be unreadable ten minutes into a render.
    if suffix == ".webm":
        transcoded = dest.with_suffix(".mp4")
        try:
            await _transcode_to_mp4(dest, transcoded)
            dest.unlink(missing_ok=True)
            dest, rel = transcoded, _track_rel(track_id, ".mp4")
        except Exception as exc:
            dest.unlink(missing_ok=True)
            transcoded.unlink(missing_ok=True)
            logger.warning("Control-track transcode failed for %s: %s", track_id, exc)
            raise HTTPException(
                status_code=400, detail="Could not convert that recording"
            ) from exc

    width, height = await probe_video_dimensions(dest)
    frames, fps = await probe_video_frames(dest)
    if not frames:
        dest.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail="Could not read any video frames from that file",
        )

    thumb_rel = _track_rel(track_id, "_thumb.jpg")
    try:
        await make_video_thumbnail(dest, settings.storage_dir / thumb_rel)
    except Exception as exc:
        logger.warning("Control-track thumbnail failed for %s: %s", track_id, exc)
        thumb_rel = None

    track = ControlTrack(
        id=track_id,
        filename=file.filename or f"{track_id}{suffix}",
        filepath=rel,
        kind=kind,
        title=title or None,
        thumbnail_path=thumb_rel,
        width=width, height=height, frame_count=frames, fps=fps,
    )
    db.add(track)
    await db.commit()
    await db.refresh(track)
    logger.info(
        "Control track %s stored (%s, %sx%s, %s frames @ %.4g fps)",
        track_id, kind, width, height, frames, fps or 0,
    )
    return _serialize_track(track)


@router.get("/tracks")
async def list_tracks(kind: Optional[str] = None, db: AsyncSession = Depends(get_db)):
    stmt = select(ControlTrack).order_by(desc(ControlTrack.created_at))
    if kind:
        stmt = stmt.where(ControlTrack.kind == kind)
    return [_serialize_track(t) for t in (await db.execute(stmt)).scalars().all()]


async def _get_track(track_id: uuid.UUID, db: AsyncSession) -> ControlTrack:
    track = await db.get(ControlTrack, track_id)
    if not track:
        raise HTTPException(status_code=404, detail="Control track not found")
    return track


@router.get("/tracks/{track_id}/file")
async def get_track_file(track_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    track = await _get_track(track_id, db)
    path = settings.storage_dir / track.filepath
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Control track missing on disk")
    return FileResponse(path, media_type="video/mp4")


@router.get("/tracks/{track_id}/thumb")
async def get_track_thumb(track_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    track = await _get_track(track_id, db)
    if not track.thumbnail_path:
        raise HTTPException(status_code=404, detail="No thumbnail")
    path = settings.storage_dir / track.thumbnail_path
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Thumbnail missing on disk")
    return FileResponse(path, media_type="image/jpeg")


@router.delete("/tracks/{track_id}", status_code=204)
async def delete_track(track_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    track = await _get_track(track_id, db)
    for rel in filter(None, (track.filepath, track.thumbnail_path)):
        path = settings.storage_dir / rel
        if path.exists():
            try:
                path.unlink()
            except Exception as exc:
                logger.warning("Could not delete %s: %s", path, exc)
    await db.delete(track)
    await db.commit()


# ── Options ──────────────────────────────────────────────────────────────────

@router.get("/options")
async def vace_options():
    """The strength range and what each part of it does.

    Server-side because the bands were measured off this workflow (see
    services/comfy/vace.py) and would drift the moment the sampler is retuned.
    """
    return {
        "strength_min": STRENGTH_MIN,
        "strength_max": STRENGTH_MAX,
        "strength_default": STRENGTH_DEFAULT,
        "recommended_min": SWEET_MIN,
        "recommended_max": SWEET_MAX,
        "bands": strength_bands(),
        "fps": FPS,
        "max_regions": 3,
        # The two engines, and what each one is actually good at. Written here
        # because both sentences are measurements, not opinions.
        "engines": [
            {"key": "animatelcm", "label": "AnimateLCM · Refine",
             "hint": "Der Nachbau des eigenen Workflows. Zieht Material aus dem "
                     "Referenzbild und verdoppelt die Auflösung in einem zweiten "
                     "Durchgang bei denoise 0.4. Die erste Wahl."},
            {"key": "vace", "label": "Wan VACE · Repaint",
             "hint": "Erfindet Umgebung und Licht frei, überschreibt dabei aber "
                     "das Material der Vorlage — gemessen. Nur für Fälle, in denen "
                     "die Vorlage wirklich verschwinden soll."},
        ],
        "lcm": {
            "depth": {"default": LCM_DEPTH_DEFAULT, "sweep": list(LCM_DEPTH_SWEEP),
                      "hint": "Wie wörtlich die Geometrie genommen wird."},
            "ip": {"default": LCM_IP_DEFAULT, "sweep": list(LCM_IP_SWEEP),
                   "hint": "Wie hart das Referenzbild sein Material aufprägt."},
            "hires": {"default": LCM_HIRES_DEFAULT, "sweep": list(LCM_HIRES_SWEEP),
                      "hint": "Wieviel Detail der zweite Durchgang erfindet. "
                              "Das Original stand fest auf 0.40."},
            "rife": [1, 2, 3, 4],
        },
        "canvases": canvas_options(),
        "aspects": aspect_options(),
        "recipes": recipe_options(),
        "defaults": {
            "canvas": DEFAULT_CANVAS,
            "aspect": DEFAULT_ASPECT,
            "recipe": DEFAULT_RECIPE,
            "fit": DEFAULT_FIT,
        },
    }


@router.get("/estimate")
async def vace_estimate(
    canvas: str = DEFAULT_CANVAS,
    aspect: str = DEFAULT_ASPECT,
    recipe: str = DEFAULT_RECIPE,
    length: int = 81,
    passes: int = 1,
):
    """Wall-clock guess for the whole job, so nobody starts a 50-minute render
    by accident. Server-side because the cost model is anchored on renders
    measured off this graph, and a client copy of it would drift silently."""
    width, height = canvas_size(canvas, aspect)
    frames = snap_length(length)
    passes = max(1, passes)
    seconds = estimate_seconds(canvas, aspect, recipe, frames) * passes
    return {
        "seconds": seconds,
        "width": width,
        "height": height,
        "frames": frames,
        "passes": passes,
        "label": _duration_label(seconds),
    }


def _duration_label(seconds: int) -> str:
    if seconds < 90:
        return f"{seconds} s"
    minutes = round(seconds / 60)
    return f"rund {minutes} Min." if minutes < 60 else f"rund {minutes // 60} h {minutes % 60} Min."


# ── Generation ───────────────────────────────────────────────────────────────

async def _reference_for(image_id: uuid.UUID, db: AsyncSession) -> Path:
    """Absolute path of a gallery picture to hand VACE as a reference.

    The untouched original, per services/image/rendition.py: a model reading
    the picture is not a viewer, and an enhanced or grained rendition would
    have it reproduce the correction as if it were the subject.
    """
    img = await db.get(Image, image_id)
    if not img:
        raise HTTPException(status_code=404, detail=f"Reference image not found: {image_id}")
    path = original_path(img)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Reference image missing on disk")
    return path


@router.post("/generate", status_code=202)
async def generate(body: GenerateRequest, db: AsyncSession = Depends(get_db)):
    """Queue a structure-video job. Returns immediately; poll
    `GET /api/video/jobs/{video_id}/progress`."""
    control = await _get_track(body.control_track_id, db)
    if not (settings.storage_dir / control.filepath).is_file():
        raise HTTPException(status_code=404, detail="Control track missing on disk")
    if control.kind == "mask":
        raise HTTPException(
            status_code=400,
            detail="That track is a colour-ID mask — pick it as mask_track_id, not as the control",
        )

    if len(body.regions) > 3:
        raise HTTPException(status_code=400, detail="At most 3 regions")
    if body.regions and not body.mask_track_id:
        raise HTTPException(status_code=400, detail="Regions need a mask track")
    if not body.regions and not body.image_id:
        raise HTTPException(status_code=400, detail="A reference image is required")
    if body.engine not in ("animatelcm", "vace"):
        raise HTTPException(status_code=400, detail=f"Unknown engine: {body.engine}")
    if body.engine == "animatelcm" and body.regions and len(body.regions) > 3:
        raise HTTPException(status_code=400, detail="At most 3 regions")
    if body.canvas not in CANVASES:
        raise HTTPException(status_code=400, detail=f"Unknown canvas: {body.canvas}")
    if body.aspect not in ASPECTS:
        raise HTTPException(status_code=400, detail=f"Unknown aspect: {body.aspect}")
    if body.recipe not in RECIPES:
        raise HTTPException(status_code=400, detail=f"Unknown recipe: {body.recipe}")
    if body.fit not in FITS:
        raise HTTPException(status_code=400, detail=f"Unknown fit: {body.fit}")

    mask = None
    if body.mask_track_id:
        mask = await _get_track(body.mask_track_id, db)
        if not (settings.storage_dir / mask.filepath).is_file():
            raise HTTPException(status_code=404, detail="Mask track missing on disk")

    # Never ask for frames the track cannot guide — VACE pads the surplus with
    # flat grey and the tail of the clip drifts with nothing steering it.
    usable = loop_length(control.frame_count or 0)
    length = snap_length(min(body.length, usable)) if body.length else usable
    if length < 5:
        raise HTTPException(status_code=400, detail="Control track is too short")

    # Resolve every reference now, so a typo fails the request instead of the
    # background job ten minutes later.
    if body.regions:
        references = [await _reference_for(r.image_id, db) for r in body.regions]
        image_ids = [r.image_id for r in body.regions]
    else:
        references = [await _reference_for(body.image_id, db)]
        image_ids = [body.image_id]

    video = Video(
        id=uuid.uuid4(),
        image_ids=image_ids,
        workflow=WORKFLOW_NAME,
        prompt=body.prompt or None,
        frame_count=length,
        n_images=0,          # no key frames; keeps _expected_clip_count at None
        fps=FPS,
        status="generating",
        created_at=datetime.now(timezone.utc),
    )
    db.add(video)
    await db.commit()

    two_stage = body.engine == "animatelcm" and body.preview_first and body.hires
    if body.engine == "animatelcm":
        # The image ids travel with the job, not just their resolved paths:
        # stage two runs in a different process lifetime and has to find the
        # same pictures again from the sidecar.
        runner = _run_lcm(
            video.id, body, control.filepath, mask.filepath if mask else None,
            references, length, (control.width, control.height), control.kind,
            image_ids=image_ids,
        )
    else:
        runner = _run_vace(
            video.id, body, control.filepath, mask.filepath if mask else None,
            references, length, (control.width, control.height), control.kind,
        )
    safe_create_task(runner, name=f"{body.engine}:{video.id}")
    # AnimateLCM renders every region in one sampler pass; VACE needs one full
    # render per region, each repainting the last.
    passes = 1 if body.engine == "animatelcm" else max(1, len(body.regions))
    total_seconds = (
        _lcm_seconds(body, control.width, control.height, length)
        if body.engine == "animatelcm"
        else estimate_seconds(body.canvas, body.aspect, body.recipe, length) * passes
    )
    # With a preview stage the number the user is waiting on is the base pass,
    # not the whole graph — quoting the total would make the wait look three
    # times longer than the one that actually ends in something to look at.
    seconds = max(20, int(total_seconds * _LCM_BASE_SHARE)) if two_stage else total_seconds
    logger.info(
        "Queued VACE job %s — %s/%s/%s, %d frames, %d pass(es), ~%d s",
        video.id, body.canvas, body.aspect, body.recipe, length, passes, seconds,
    )
    return {
        "video_id": str(video.id),
        "status": "generating",
        "length": length,
        "passes": passes,
        "seconds": seconds,
        "total_seconds": total_seconds,
        "two_stage": two_stage,
        **describe_strength(body.strength),
    }


def _stage_image(src: Path, name: str) -> str:
    """Copy a gallery picture into ComfyUI's input folder and return its name.

    LoadImage resolves by bare filename inside that folder, so a reference
    living in art-rium storage has to be put where the node can see it.
    """
    inp = settings.comfyui_output_dir.parent / "input"
    inp.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, inp / name)
    return name


async def _submit_outputs(
    wf: dict, save_nodes: list[str], key: str, band: tuple[int, int], message: str,
) -> dict[str, Path]:
    """Run one pass and return the file ComfyUI wrote for each save node.

    The sampler is named as the band owner. Without that the VHS loader's
    "93/93" — reported within a second of the job starting — would drive the
    bar to the top of the band and then fall back once sampling began.

    A graph can have more than one output: the two-stage base pass writes both
    the raw frames the finishing stage will read and the interpolated version
    the user watches, and both come out of the same submission.
    """
    sampler_node = next(
        (nid for nid, node in wf.items() if node["class_type"] == "KSampler"), None
    )
    async with httpx.AsyncClient(timeout=240) as client:
        prompt_id = await post_workflow(client, wf)
        listener = get_listener()
        if listener:
            listener.register_node_labels(prompt_id, wf)
        _set_progress(key, "generating", message, band[0],
                      prompt_id=prompt_id, band=band, band_node=sampler_node)
        outputs = await poll_history(client, prompt_id, timeout=_JOB_TIMEOUT, interval=4)

    produced: dict[str, Path] = {}
    for node in dict.fromkeys(save_nodes):
        entry = (outputs.get(node) or {}).get("gifs") or []
        if not entry:
            raise RuntimeError(f"ComfyUI returned no video for {node}: {list(outputs)}")
        path = settings.comfyui_output_dir / entry[0].get("subfolder", "") / entry[0]["filename"]
        if not path.exists():
            raise FileNotFoundError(f"Rendered file not found at {path}")
        produced[node] = path
    return produced


async def _submit(
    wf: dict, save_node: str, key: str, band: tuple[int, int], message: str,
) -> Path:
    """Single-output `_submit_outputs`."""
    return (await _submit_outputs(wf, [save_node], key, band, message))[save_node]


# Cost model for the AnimateLCM graph, anchored on two measured renders on the
# 4060 Ti: 48 frames at 544->1088 took 418 s, 94 frames the same shape took 880 s
# including RIFE 4x. Both land near 8.7 s per frame, and the relationship is
# linear in frames and in pixels, so one constant carries it:
#
#   418 / (48 * 544 * 544)  = 2.94e-5 s per base pixel-frame
#
# The hires pass is already inside that number; `hires=False` removes roughly
# two thirds of it, measured against the base pass alone.
_LCM_SECONDS_PER_PIXEL_FRAME = 2.94e-5
_LCM_BASE_SHARE = 0.35          # what is left when the refine is switched off


def _lcm_seconds(req: "GenerateRequest", width: int | None, height: int | None,
                 length: int) -> int:
    canvas_w, canvas_h = lcm_canvas(req.lcm_aspect, (width, height))
    cost = canvas_w * canvas_h * length * _LCM_SECONDS_PER_PIXEL_FRAME
    if not req.hires:
        cost *= _LCM_BASE_SHARE
    return max(30, int(cost))


# ── AnimateLCM: one graph, optionally in two stages ──────────────────────────
# Stage one renders the base pass and stops. Stage two reads that render back
# and refines it. Everything either stage needs beyond the render itself lives
# in a sidecar next to it, because the decision in between can take days and
# has to survive a restart.

_LCM_PLAN_VERSION = 1


def _lcm_plan_path(video_id: uuid.UUID) -> Path:
    return _segments_dir(video_id) / "lcm_plan.json"


def _lcm_base_path(video_id: uuid.UUID) -> Path:
    """The un-interpolated base render — stage two's input, not the preview.

    In the job's segments directory rather than beside the finished video,
    because that directory is already removed wholesale when the job is
    deleted; a sibling in videos_dir would have to be remembered by name in
    routers/video.py's cleanup, which is not that module's business.
    """
    return _segments_dir(video_id) / "lcm_base.mp4"


def _lcm_preview_name(video_id: uuid.UUID) -> str:
    return f"lcm_{video_id}_preview.mp4"


def _lcm_final_name(video_id: uuid.UUID) -> str:
    return f"lcm_{video_id}.mp4"


def _stage_references(video_id: uuid.UUID, references: list[Path]) -> list[str]:
    return [
        _stage_image(path, f"artrium_lcm_ref_{video_id.hex[:8]}_{i}.png")
        for i, path in enumerate(references)
    ]


def _unstage(names: list[str]) -> None:
    inp = settings.comfyui_output_dir.parent / "input"
    for name in names:
        (inp / name).unlink(missing_ok=True)


def _lcm_request(
    req: GenerateRequest,
    video_id: uuid.UUID,
    control_rel: str,
    mask_rel: str | None,
    staged: list[str],
    length: int,
    source_size: tuple[int | None, int | None],
    track_kind: str,
    seed: int,
) -> AnimateLcmRequest:
    """The builder request, from the API request. Both stages go through here,
    which is what guarantees the second one refines the first rather than
    something adjacent to it."""
    # Footage carries no depth of its own, so it goes through DepthAnythingV2
    # in-graph — and must then not also be inverted.
    derive = req.derive_depth or track_kind == "footage"
    lcm = AnimateLcmRequest(
        control_video=str(settings.storage_dir / control_rel),
        prompt=req.prompt,
        reference_image=staged[0] if staged else None,
        mask_video=str(settings.storage_dir / mask_rel) if mask_rel else None,
        aspect=req.lcm_aspect,
        source_width=source_size[0],
        source_height=source_size[1],
        length=length,
        source_frames=length,
        depth_strength=req.depth_strength,
        ip_weight=req.ip_weight,
        hires=req.hires,
        hires_denoise=req.hires_denoise,
        rife=req.rife,
        seed=seed,
        derive_depth=derive,
        invert_depth=not derive,
        filename_prefix=f"artrium_lcm_{video_id.hex[:8]}",
        regions=[
            LcmRegion(color=tuple(r.color), reference=staged[i],
                      weight=req.ip_weight, threshold=r.threshold)
            for i, r in enumerate(req.regions)
        ],
    )
    if req.negative:
        lcm.negative = req.negative
    return lcm


def _write_lcm_plan(
    video_id: uuid.UUID,
    req: GenerateRequest,
    control_rel: str,
    mask_rel: str | None,
    image_ids: list[uuid.UUID],
    length: int,
    source_size: tuple[int | None, int | None],
    track_kind: str,
    seed: int,
) -> None:
    """Everything stage two needs, on disk.

    The seed above all: the hires pass has to sample with the seed the base
    pass used, and a seed resolved inside the builder would be gone by the time
    anyone pressed the button.
    """
    plan = {
        "version": _LCM_PLAN_VERSION,
        "request": req.model_dump(mode="json"),
        "control_rel": control_rel,
        "mask_rel": mask_rel,
        "image_ids": [str(i) for i in image_ids],
        "length": length,
        "source_size": list(source_size),
        "track_kind": track_kind,
        "seed": seed,
    }
    path = _lcm_plan_path(video_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, indent=1), encoding="utf-8")


def _read_lcm_plan(video_id: uuid.UUID) -> dict:
    path = _lcm_plan_path(video_id)
    if not path.is_file():
        raise HTTPException(
            status_code=409,
            detail="Für diesen Job liegt kein Vorschau-Plan vor — er lässt sich nicht fertigstellen.",
        )
    try:
        plan = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"Vorschau-Plan unlesbar: {exc}") from exc
    if plan.get("version") != _LCM_PLAN_VERSION:
        raise HTTPException(status_code=409, detail="Vorschau-Plan stammt aus einer älteren Version.")
    return plan


async def _persist_render(
    video_id: uuid.UUID, produced: Path, dest: Path, status: str,
) -> tuple[int | None, int | None]:
    """Move a ComfyUI output into storage and point the row at it."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(produced, dest)
    produced.unlink(missing_ok=True)
    width, height = await probe_video_dimensions(dest)
    # The gallery card looks for this by convention — videos_dir /
    # "<id>_thumb.jpg" — and without it the preview is a black rectangle
    # with a play button, which is exactly what a 404'd <img> looks like.
    try:
        await make_video_thumbnail(dest, settings.videos_dir / f"{video_id}_thumb.jpg")
    except Exception as exc:
        logger.warning("Thumbnail failed for %s: %s", video_id, exc)

    async with AsyncSessionLocal() as db:
        video = await db.get(Video, video_id)
        if video:
            video.filename = dest.name
            video.filepath = str(dest.relative_to(settings.storage_dir)).replace("\\", "/")
            video.width, video.height = width, height
            video.status = status
            video.error = None
            await db.commit()
    return width, height


async def _fail_lcm(video_id: uuid.UUID, key: str, exc: Exception) -> None:
    async with AsyncSessionLocal() as db:
        video = await db.get(Video, video_id)
        if video:
            video.status = "failed"
            video.error = f"{type(exc).__name__}: {exc}"[:2000]
            await db.commit()
    _set_progress(key, "failed", f"{type(exc).__name__}: {exc}", 0)


async def _run_lcm(
    video_id: uuid.UUID,
    req: GenerateRequest,
    control_rel: str,
    mask_rel: str | None,
    references: list[Path],
    length: int,
    source_size: tuple[int | None, int | None] = (None, None),
    track_kind: str = "depth",
    image_ids: list[uuid.UUID] | None = None,
) -> None:
    """Background job: one graph, one sampler pass per stage, whatever the
    region count. Simpler than the VACE runner precisely because the adapters
    carry their own masks instead of needing a render each.

    Splits in two when the caller asked for a preview *and* there is a hires
    pass to hold back — otherwise there is nothing to decide and the job runs
    straight through.
    """
    key = str(video_id)
    settings.videos_dir.mkdir(parents=True, exist_ok=True)
    seed = resolve_seed(req.seed)
    staged: list[str] = []
    two_stage = req.preview_first and req.hires
    try:
        staged = _stage_references(video_id, references)
        lcm = _lcm_request(req, video_id, control_rel, mask_rel, staged,
                           length, source_size, track_kind, seed)

        if not two_stage:
            _set_progress(key, "generating", "Struktur wird gerendert…", 5)
            wf, save_node = build_animatelcm_workflow(lcm)
            produced = await _submit(wf, save_node, key, (5, 95),
                                     "Struktur wird gerendert…")
            _set_progress(key, "finalizing", "Wird gespeichert…", 96)
            width, height = await _persist_render(
                video_id, produced, settings.videos_dir / _lcm_final_name(video_id), "done",
            )
            _set_progress(key, "done", "Fertig", 100)
            logger.info("AnimateLCM job %s finished (%sx%s, %d frames)",
                        video_id, width, height, length)
            return

        _set_progress(key, "generating", "Vorschau wird gerendert…", 5)
        wf, preview_node, base_node = build_animatelcm_base_workflow(lcm)
        produced = await _submit_outputs(
            wf, [preview_node, base_node], key, (5, 92), "Vorschau wird gerendert…",
        )

        _set_progress(key, "finalizing", "Vorschau wird gespeichert…", 94)
        # The raw base is kept as-is: it is the input the hires pass reads, and
        # re-encoding it here would cost a generation of quality for nothing.
        base_dest = _lcm_base_path(video_id)
        base_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(produced[base_node], base_dest)

        _write_lcm_plan(video_id, req, control_rel, mask_rel,
                        image_ids or [], length, source_size, track_kind, seed)
        width, height = await _persist_render(
            video_id, produced[preview_node],
            settings.videos_dir / _lcm_preview_name(video_id), "review",
        )
        # Without interpolation the graph has one output and both names point at
        # the same file — which `_persist_render` has just consumed. Only a
        # genuinely separate base render is left to clean up.
        if produced[base_node] != produced[preview_node]:
            produced[base_node].unlink(missing_ok=True)
        _set_progress(key, "review", "Vorschau bereit", 100)
        logger.info(
            "AnimateLCM job %s parked for review (%sx%s, %d frames) — base at %s",
            video_id, width, height, length, base_dest.name,
        )

    except Exception as exc:
        logger.exception("AnimateLCM job %s failed", video_id)
        await _fail_lcm(video_id, key, exc)
    finally:
        _unstage(staged)
        if _progress.get(key, {}).get("phase") == "done":
            _progress.pop(key, None)


async def _run_lcm_hires(video_id: uuid.UUID, plan: dict) -> None:
    """Stage two: refine the parked base render into the final clip."""
    key = str(video_id)
    staged: list[str] = []
    try:
        base = _lcm_base_path(video_id)
        if not base.is_file():
            raise RuntimeError("Der Basis-Render zu dieser Vorschau fehlt auf der Platte")

        async with AsyncSessionLocal() as db:
            references = [
                await _reference_for(uuid.UUID(i), db) for i in plan["image_ids"]
            ]
        staged = _stage_references(video_id, references)
        req = GenerateRequest(**plan["request"])
        lcm = _lcm_request(
            req, video_id, plan["control_rel"], plan["mask_rel"], staged,
            plan["length"], tuple(plan["source_size"]), plan["track_kind"],
            int(plan["seed"]),
        )

        _set_progress(key, "generating", "Detailpass läuft…", 5)
        wf, save_node = build_animatelcm_hires_workflow(lcm, str(base))
        produced = await _submit(wf, save_node, key, (5, 95), "Detailpass läuft…")

        _set_progress(key, "finalizing", "Wird gespeichert…", 96)
        width, height = await _persist_render(
            video_id, produced, settings.videos_dir / _lcm_final_name(video_id), "done",
        )
        # The preview and the base only existed to make the decision that has
        # now been made.
        (settings.videos_dir / _lcm_preview_name(video_id)).unlink(missing_ok=True)
        base.unlink(missing_ok=True)
        _lcm_plan_path(video_id).unlink(missing_ok=True)

        _set_progress(key, "done", "Fertig", 100)
        logger.info("AnimateLCM job %s finished after review (%sx%s)", video_id, width, height)

    except Exception as exc:
        logger.exception("AnimateLCM hires pass %s failed", video_id)
        await _fail_lcm(video_id, key, exc)
    finally:
        _unstage(staged)
        if _progress.get(key, {}).get("phase") == "done":
            _progress.pop(key, None)


async def _get_review_job(video_id: uuid.UUID, db: AsyncSession) -> Video:
    video = await db.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Job nicht gefunden")
    if video.status != "review":
        raise HTTPException(
            status_code=409,
            detail=f"Dieser Job wartet nicht auf eine Entscheidung (Status: {video.status})",
        )
    return video


@router.post("/jobs/{video_id}/finish", status_code=202)
async def finish_lcm(video_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """Run the hires pass on a parked preview. Poll the usual progress endpoint."""
    video = await _get_review_job(video_id, db)
    plan = _read_lcm_plan(video_id)
    video.status = "generating"
    video.error = None
    await db.commit()

    _set_progress(str(video_id), "generating", "Detailpass wird eingereiht…", 2)
    safe_create_task(_run_lcm_hires(video_id, plan), name=f"lcm_hires:{video_id}")
    seconds = max(30, int(_lcm_seconds(GenerateRequest(**plan["request"]),
                                       plan["source_size"][0], plan["source_size"][1],
                                       plan["length"]) * (1 - _LCM_BASE_SHARE)))
    return {"video_id": str(video_id), "status": "generating", "seconds": seconds}


@router.post("/jobs/{video_id}/keep")
async def keep_lcm_preview(video_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """Accept the preview as the finished clip and skip the hires pass.

    A real outcome, not a fallback: the base pass is softer and dreamier than
    the refined render, and for some material that is the wanted picture. The
    base render and the plan go — nothing will ask for them again.
    """
    video = await _get_review_job(video_id, db)
    video.status = "done"
    video.error = None
    await db.commit()

    _lcm_base_path(video_id).unlink(missing_ok=True)
    _lcm_plan_path(video_id).unlink(missing_ok=True)
    _progress.pop(str(video_id), None)
    logger.info("AnimateLCM job %s: preview kept as the final clip", video_id)
    return {"video_id": str(video_id), "status": "done"}


async def _run_vace(
    video_id: uuid.UUID,
    req: GenerateRequest,
    control_rel: str,
    mask_rel: str | None,
    references: list[Path],
    length: int,
    source_size: tuple[int | None, int | None] = (None, None),
    track_kind: str = "depth",
) -> None:
    """Background job: one pass without regions, one pass per region with them.

    Each region pass takes the previous pass's *render* as its control video —
    a mask asks VACE to preserve the unmasked part of whatever it is given, and
    preserving a depth pass just reproduces the grey.
    """
    key = str(video_id)
    settings.videos_dir.mkdir(parents=True, exist_ok=True)
    staged: list[str] = []
    try:
        control_abs = str(settings.storage_dir / control_rel)
        mask_abs = str(settings.storage_dir / mask_rel) if mask_rel else None

        for i, path in enumerate(references):
            staged.append(_stage_image(path, f"artrium_vace_ref_{video_id.hex[:8]}_{i}.png"))

        base = VaceRequest(
            control_video=control_abs,
            prompt=req.prompt,
            canvas=req.canvas,
            aspect=req.aspect,
            recipe=req.recipe,
            fit=req.fit,
            reference_image=staged[0],
            mask_video=mask_abs,
            length=length,
            source_frames=length,
            strength=clamp_strength(req.strength),
            seed=req.seed,
            derive_depth=False,      # set per-pass below
            filename_prefix=f"artrium_vace_{video_id.hex[:8]}",
            regions=[
                Region(color=tuple(r.color), reference=staged[i],
                       strength=clamp_strength(r.strength), threshold=r.threshold)
                for i, r in enumerate(req.regions)
            ],
        )
        if req.negative:
            base.negative = req.negative

        passes = plan_region_passes(base)
        total = len(passes)
        produced: Path | None = None

        for i, step in enumerate(passes):
            if i:
                # From pass 1 on the control track is a finished render, so it
                # is neither inverted nor run through DepthAnything — that is
                # already set by plan_region_passes, and the path is what only
                # the runner can know.
                step.control_video = str(produced)
                step.source_frames = length
            message = (
                f"Durchlauf {i + 1} / {total}…" if total > 1
                else "Struktur wird gerendert…"
            )
            _set_progress(key, "generating", message, 5 + int(90 * i / total))
            wf, save_node = build_vace_workflow(step)
            band = (5 + int(90 * i / total), 5 + int(90 * (i + 1) / total))
            produced = await _submit(wf, save_node, key, band, message)
            if i + 1 < total:
                async with httpx.AsyncClient(timeout=60) as client:
                    await free_memory(client)

        _set_progress(key, "finalizing", "Wird gespeichert…", 96)
        filename = f"vace_{video_id}.mp4"
        dest = settings.videos_dir / filename
        shutil.copy2(produced, dest)
        produced.unlink(missing_ok=True)
        width, height = await probe_video_dimensions(dest)
        # The gallery card looks for this by convention — videos_dir /
        # "<id>_thumb.jpg" — and without it the preview is a black rectangle
        # with a play button, which is exactly what a 404'd <img> looks like.
        try:
            await make_video_thumbnail(dest, settings.videos_dir / f"{video_id}_thumb.jpg")
        except Exception as exc:
            logger.warning("Thumbnail failed for %s: %s", video_id, exc)

        async with AsyncSessionLocal() as db:
            video = await db.get(Video, video_id)
            if video:
                video.filename = filename
                video.filepath = str(dest.relative_to(settings.storage_dir)).replace("\\", "/")
                video.width, video.height = width, height
                video.status = "done"
                video.error = None
                await db.commit()

        _set_progress(key, "done", "Fertig", 100)
        logger.info("VACE job %s finished (%sx%s, %d frames)", video_id, width, height, length)

    except Exception as exc:
        logger.exception("VACE job %s failed", video_id)
        async with AsyncSessionLocal() as db:
            video = await db.get(Video, video_id)
            if video:
                video.status = "failed"
                video.error = f"{type(exc).__name__}: {exc}"[:2000]
                await db.commit()
        _set_progress(key, "failed", f"{type(exc).__name__}: {exc}", 0)
    finally:
        inp = settings.comfyui_output_dir.parent / "input"
        for name in staged:
            (inp / name).unlink(missing_ok=True)
        if _progress.get(key, {}).get("phase") == "done":
            _progress.pop(key, None)
