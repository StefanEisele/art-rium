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
from core.subproc import communicate
from core.tasks import safe_create_task
from core.video_thumb import (
    make_video_thumbnail,
    probe_video_dimensions,
    probe_video_frames,
)
from routers.video import _progress, _segments_dir, _set_progress
from services.segment import (
    DURATIONS as SEGMENT_DURATIONS,
)
from services.segment import (
    STRIDES as SEGMENT_STRIDES,
)
from services.segment import (
    FRAME_CEILING,
    MAX_REGIONS,
    REGION_COLORS,
    REGION_LABELS,
    SegmentError,
    budget_dict,
    build_spec,
    plan_frames,
    run_segmentation,
    trim_filters,
)
from services.comfy.client import free_memory, poll_history, post_workflow
from services.comfy.vram import free_vram_for
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
    BASE_IP_DEFAULT as LCM_BASE_DEFAULT,
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
# What the card needs free before a structure render is posted. Well below
# routers/video.py's 9 GB default because this stack is SD 1.5 — Juggernaut, a
# motion module, two ControlNets and an IP-Adapter — and even the 1088² hires
# pass stays under this. Set low on purpose: the point is to catch a dead
# ComfyUI and a card still held by the render before, not to gate a job that
# would have fit.
_LCM_MIN_FREE_VRAM = 6.0 * 1024**3
_JOB_TIMEOUT = 7200          # a 720 region sequence is three ~12-minute passes
_MAX_UPLOAD_BYTES = 600 * 1024 * 1024


class RegionSpec(BaseModel):
    color: tuple[int, int, int]
    image_id: uuid.UUID              # reference picture, from the gallery
    # VACE only: how freely that region's pass may repaint. AnimateLCM has no
    # per-region denoise — it renders every region in one sampler pass — so it
    # reads `ip_weight` instead, and reading this one there would silently wire
    # the "Wörtlich ↔ Frei" slider to a reference weight.
    strength: float = STRENGTH_DEFAULT
    # AnimateLCM only: how hard this region's picture is pressed into its mask.
    # None follows the job's global ip_weight, which is what every render did
    # before the dial existed.
    ip_weight: Optional[float] = None
    threshold: int = 20


class GenerateRequest(BaseModel):
    control_track_id: uuid.UUID
    prompt: str
    mask_track_id: Optional[uuid.UUID] = None
    image_id: Optional[uuid.UUID] = None      # reference for the no-region case
    # The picture everything the masks do NOT cover is made of. Its own image
    # rather than one of the regions': it is the ground they sit in. Only
    # meaningful with regions; without them `image_id` already covers the frame.
    base_image_id: Optional[uuid.UUID] = None
    base_weight: float = LCM_BASE_DEFAULT
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
        # A mask is only meaningful against the frames it was keyed out of, so
        # the pairing travels with it and the UI can offer the two together.
        "source_track_id": str(t.source_track_id) if t.source_track_id else None,
        "regions": t.regions or None,
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
    _, err = await communicate(proc)
    if proc.returncode != 0 or not dest.is_file():
        raise RuntimeError(err.decode(errors="replace")[:400] or "ffmpeg failed")


async def _apply_budget(src: Path, dest: Path, budget, lossless: bool = False) -> None:
    """Write `src` back out holding only the frames a budget allows.

    Both dials are baked into the file here rather than asked for at render
    time. `VHS_LoadVideoPath`'s own `force_rate` is the alternative and it is a
    trap: against a 30 fps source it returns a different frame count than the
    job asked for, and the surplus is padded with flat grey that steers
    nothing. A file that already holds exactly the wanted frames cannot do that.

    `-fps_mode passthrough` is what makes `select` stick — ffmpeg otherwise
    re-times the thinned stream back up to the source rate by duplicating the
    frames just dropped.

    `lossless` for mask videos: ColorToMask keys on an exact RGB triple, and a
    re-encode at 4:2:0 would put a ramp of unkeyable in-between colours around
    every region.
    """
    args = [settings.ffmpeg_path, "-y", "-v", "error"]
    if budget.start:
        args += ["-ss", f"{budget.start:g}"]
    args += ["-i", str(src)]
    if budget.seconds:
        args += ["-t", f"{budget.seconds:g}"]
    args += ["-vf", ",".join(trim_filters(budget))]
    # No `-r`: it is a CFR conversion that may duplicate frames, and ffmpeg
    # rejects it alongside `-fps_mode passthrough` anyway. The rate comes from
    # the timestamps `setpts` writes. See services/segment/plan.py.
    args += ["-frames:v", str(budget.kept), "-fps_mode", "passthrough", "-an"]
    args += (["-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv444p"] if lossless
             else ["-c:v", "libx264", "-preset", "veryfast", "-crf", "16",
                   "-pix_fmt", "yuv420p"])
    args.append(str(dest))

    proc = await asyncio.create_subprocess_exec(
        *args, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
    )
    _, err = await communicate(proc)
    if proc.returncode != 0 or not dest.is_file():
        raise RuntimeError(err.decode(errors="replace")[:400] or "ffmpeg failed")


async def _register_track(
    db: AsyncSession,
    track_id: uuid.UUID,
    dest: Path,
    rel: str,
    kind: str,
    filename: str,
    title: Optional[str],
    source_track_id: Optional[uuid.UUID] = None,
    regions: Optional[list[dict]] = None,
) -> ControlTrack:
    """Probe, thumbnail and store one track that is already on disk.

    The probe is not a nicety: `frame_count` is what later clamps a job's
    length, and without it a request for more frames than the track holds comes
    back with a silently unguided tail.
    """
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
        filename=filename,
        filepath=rel,
        kind=kind,
        title=title or None,
        thumbnail_path=thumb_rel,
        width=width, height=height, frame_count=frames, fps=fps,
        source_track_id=source_track_id,
        regions=regions,
    )
    db.add(track)
    await db.commit()
    await db.refresh(track)
    logger.info(
        "Control track %s stored (%s, %sx%s, %s frames @ %.4g fps)",
        track_id, kind, width, height, frames, fps or 0,
    )
    return track


@router.post("/tracks", status_code=201)
async def upload_track(
    file: UploadFile = File(...),
    kind: str = Form("depth"),
    title: Optional[str] = Form(None),
    seconds: Optional[float] = Form(None),
    stride: int = Form(1),
    start: float = Form(0.0),
    db: AsyncSession = Depends(get_db),
):
    """Take a control track into the library: store, trim, probe, thumbnail.

    `seconds` and `stride` are the frame budget, and they are applied here so
    that the stored file *is* the budget — see `_apply_budget` for why asking
    the render's loader for a different rate instead is a trap. A 20 s phone
    clip at 30 fps is 600 frames of render; 6 s at every 2nd frame is 90.
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

    # Trim and thin before anything else looks at the file, so every number
    # recorded from here on describes what the render will actually see.
    if seconds or stride > 1 or start:
        frames, fps = await probe_video_frames(dest)
        budget = plan_frames(frames or 0, fps, seconds, stride, start)
        if budget.kept < 1:
            dest.unlink(missing_ok=True)
            raise HTTPException(
                status_code=400,
                detail="Mit diesen Einstellungen bleibt kein einziges Bild übrig",
            )
        trimmed = dest.with_name(f"{dest.stem}_cut{dest.suffix}")
        try:
            await _apply_budget(dest, trimmed, budget, lossless=(kind == "mask"))
            dest.unlink(missing_ok=True)
            trimmed.replace(dest)
        except Exception as exc:
            trimmed.unlink(missing_ok=True)
            dest.unlink(missing_ok=True)
            logger.warning("Control-track trim failed for %s: %s", track_id, exc)
            raise HTTPException(
                status_code=400, detail="Zuschneiden ist fehlgeschlagen"
            ) from exc

    track = await _register_track(
        db, track_id, dest, rel, kind,
        file.filename or f"{track_id}{suffix}", title,
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
    for rel in filter(None, (track.filepath, track.thumbnail_path,
                             _track_rel(track_id, "_preview.mp4"))):
        path = settings.storage_dir / rel
        if path.exists():
            try:
                path.unlink()
            except Exception as exc:
                logger.warning("Could not delete %s: %s", path, exc)
    await db.delete(track)
    await db.commit()


# ── Frame budget, trimming, segmentation ─────────────────────────────────────

class DeriveRequest(BaseModel):
    start: float = 0.0
    seconds: Optional[float] = None
    stride: int = 1
    title: Optional[str] = None


class Concept(BaseModel):
    """One thing to find in the footage, and the colour it becomes."""
    text: str
    color: Optional[tuple[int, int, int]] = None


class SegmentRequest(BaseModel):
    concepts: list[Concept]
    # Applied to the footage *before* segmenting, producing a trimmed track that
    # the mask is then keyed against — see `_run_segmentation` on why the mask
    # cannot be trimmed on its own.
    start: float = 0.0
    seconds: Optional[float] = None
    stride: int = 1
    score_threshold: Optional[float] = None
    title: Optional[str] = None


@router.get("/tracks/{track_id}/budget")
async def track_budget(
    track_id: uuid.UUID,
    seconds: Optional[float] = None,
    stride: int = 1,
    start: float = 0.0,
    db: AsyncSession = Depends(get_db),
):
    """What a trim/stride setting would leave, in frames.

    The same function the ingest uses, so the number shown before pressing the
    button is the number that comes out of it.
    """
    track = await _get_track(track_id, db)
    return budget_dict(
        plan_frames(track.frame_count or 0, track.fps, seconds, stride, start)
    )


@router.post("/tracks/{track_id}/derive", status_code=201)
async def derive_track(
    track_id: uuid.UUID,
    body: DeriveRequest,
    db: AsyncSession = Depends(get_db),
):
    """Cut a shorter, thinner copy out of a track already in the library.

    The upload form does this too; this is the same operation for takes that
    are already stored — and for the case that matters, producing the trimmed
    footage a mask will be keyed against.
    """
    source = await _get_track(track_id, db)
    src_path = settings.storage_dir / source.filepath
    if not src_path.is_file():
        raise HTTPException(status_code=404, detail="Control track missing on disk")

    budget = plan_frames(
        source.frame_count or 0, source.fps, body.seconds, body.stride, body.start,
    )
    if budget.kept < 1:
        raise HTTPException(
            status_code=400,
            detail="Mit diesen Einstellungen bleibt kein einziges Bild übrig",
        )

    new_id = uuid.uuid4()
    rel = _track_rel(new_id, ".mp4")
    dest = settings.storage_dir / rel
    settings.control_dir.mkdir(parents=True, exist_ok=True)
    try:
        await _apply_budget(src_path, dest, budget, lossless=(source.kind == "mask"))
    except Exception as exc:
        dest.unlink(missing_ok=True)
        logger.warning("Derive failed for %s: %s", track_id, exc)
        raise HTTPException(status_code=400, detail="Zuschneiden ist fehlgeschlagen") from exc

    track = await _register_track(
        db, new_id, dest, rel, source.kind,
        f"{Path(source.filename).stem}_cut.mp4",
        body.title or f"{source.title or source.filename} · {budget.kept}f",
        source_track_id=source.id,
    )
    return _serialize_track(track)


@router.post("/tracks/{track_id}/segment", status_code=202)
async def segment_track(
    track_id: uuid.UUID,
    body: SegmentRequest,
    db: AsyncSession = Depends(get_db),
):
    """Turn filmed footage into a colour-ID mask, one colour per named thing.

    This is the automatic version of what used to be a Blender job: name the
    things in the shot ("Tomate", "Hand", "Holzbrett") and SAM 3 finds every
    instance of each and tracks it through the clip, so the render can key a
    different reference picture into each one.

    Returns immediately; poll `GET /api/video/jobs/{job_id}/progress`. The mask
    track appears in the library when it lands.
    """
    source = await _get_track(track_id, db)
    if source.kind == "mask":
        raise HTTPException(
            status_code=400,
            detail="Das ist schon eine Maske — segmentiert wird gefilmtes Material.",
        )
    if not (settings.storage_dir / source.filepath).is_file():
        raise HTTPException(status_code=404, detail="Control track missing on disk")

    concepts = [c for c in body.concepts if c.text.strip()]
    if not concepts:
        raise HTTPException(status_code=400, detail="Mindestens ein Begriff wird gebraucht")
    if len(concepts) > MAX_REGIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Höchstens {MAX_REGIONS} Regionen — der Renderpfad kettet drei IP-Adapter.",
        )
    # SAM 3 keys its prompts by their text, so two identical words collapse into
    # one object set and the second colour would never be painted.
    lowered = [c.text.strip().lower() for c in concepts]
    if len(set(lowered)) != len(lowered):
        raise HTTPException(status_code=400, detail="Jeder Begriff darf nur einmal vorkommen")

    resolved = [
        {"text": c.text.strip(), "color": list(c.color or REGION_COLORS[i])}
        for i, c in enumerate(concepts)
    ]

    budget = plan_frames(
        source.frame_count or 0, source.fps, body.seconds, body.stride, body.start,
    )
    if budget.kept < 1:
        raise HTTPException(
            status_code=400,
            detail="Mit diesen Einstellungen bleibt kein einziges Bild übrig",
        )

    job_id = uuid.uuid4()
    _set_progress(str(job_id), "generating", "Segmentierung wird eingereiht…", 2)
    safe_create_task(
        _run_segmentation(job_id, source.id, resolved, budget,
                          body.score_threshold, body.title),
        name=f"segment:{job_id}",
    )
    logger.info(
        "Queued segmentation %s on track %s — %s, %d frames",
        job_id, track_id, ", ".join(c["text"] for c in resolved), budget.kept,
    )
    return {
        "job_id": str(job_id),
        "status": "generating",
        "frames": budget.kept,
        "concepts": resolved,
    }


async def _run_segmentation(
    job_id: uuid.UUID,
    source_id: uuid.UUID,
    concepts: list[dict],
    budget,
    score_threshold: Optional[float],
    title: Optional[str],
) -> None:
    """Segment in the background, then register the mask — and, when the clip
    was trimmed, the trimmed footage it belongs to.

    **The trim happens once and both tracks come out of it.** A mask is keyed
    frame by frame against particular pixels, so it is only meaningful beside
    the exact footage it was cut from: segmenting a thinned version of a track
    the user then renders at full length would slide every region off its
    object. Producing the pair together is what makes that unrepresentable.
    """
    key = str(job_id)
    control_dir = settings.control_dir
    control_dir.mkdir(parents=True, exist_ok=True)

    mask_rel = _track_rel(job_id, ".mp4")
    mask_path = settings.storage_dir / mask_rel
    preview_rel = _track_rel(job_id, "_preview.mp4")

    trimmed_id: uuid.UUID | None = None
    trimmed_path: Path | None = None

    try:
        async with AsyncSessionLocal() as db:
            source = await db.get(ControlTrack, source_id)
            if source is None:
                raise SegmentError("Die Vorlage ist verschwunden")
            source_path = settings.storage_dir / source.filepath
            source_name, source_title = source.filename, source.title
            source_kind = source.kind

        # A trim means the render must use the trimmed copy, so it becomes a
        # track of its own rather than a temporary file. `capped` counts as a
        # trim: without it a track longer than the ceiling would be segmented
        # in full, which is exactly the runaway the ceiling exists to stop.
        trims = bool(
            budget.stride > 1 or budget.seconds or budget.start or budget.capped
        )
        segment_source = source_path
        if trims:
            _set_progress(key, "generating", "Material wird zugeschnitten…", 4)
            trimmed_id = uuid.uuid4()
            trimmed_rel = _track_rel(trimmed_id, ".mp4")
            trimmed_path = settings.storage_dir / trimmed_rel
            await _apply_budget(source_path, trimmed_path, budget)
            segment_source = trimmed_path

        # A term that matched nothing on the first frame is worth saying at
        # once — it is nearly always a non-English word, and the fix is one
        # edit. It has to *stick*, though: the next tracking tick lands
        # milliseconds later, so a one-shot message would be overwritten before
        # anyone read it. It rides along with every later line instead.
        warning = ""

        def on_event(event: dict) -> None:
            nonlocal warning
            kind = event.get("event")
            if kind == "stage":
                pct = {"decode": 6, "load": 10, "encode": 93}.get(event.get("stage"), 8)
                _set_progress(key, "generating", event.get("message") or "…", pct)
            elif kind == "unmatched":
                names = ", ".join(event.get("concepts") or [])
                warning = f" · nicht gefunden: {names} — englische Begriffe?"
            elif kind == "progress":
                done, total = event.get("done") or 0, event.get("total") or 1
                _set_progress(
                    key, "generating",
                    f"Objekte werden verfolgt… {done}/{total}{warning}",
                    12 + int(80 * done / max(1, total)),
                )

        spec = build_spec(
            segment_source, mask_path, settings.storage_dir / preview_rel,
            concepts,
            # The clip handed to the worker is already trimmed, so it reads all
            # of it — trimming twice would compound the stride.
            budget=None,
            score_threshold=score_threshold,
        )
        spec["fps"] = budget.fps
        result = await run_segmentation(spec, on_progress=on_event)

        report = result.get("report") or {}
        regions = [
            {
                "color": c["color"],
                "label": c["text"],
                "frames": (report.get(c["text"]) or {}).get("frames", 0),
                "coverage": (report.get(c["text"]) or {}).get("coverage", 0.0),
            }
            for c in concepts
        ]
        missing = [r["label"] for r in regions if not r["frames"]]

        _set_progress(key, "finalizing", "Maske wird gespeichert…", 96)
        async with AsyncSessionLocal() as db:
            if trimmed_id and trimmed_path:
                await _register_track(
                    db, trimmed_id, trimmed_path,
                    _track_rel(trimmed_id, ".mp4"), source_kind,
                    f"{Path(source_name).stem}_cut.mp4",
                    f"{source_title or source_name} · {budget.kept}f",
                    source_track_id=source_id,
                )
            await _register_track(
                db, job_id, mask_path, mask_rel, "mask",
                f"{Path(source_name).stem}_mask.mp4",
                title or f"Maske · {', '.join(c['text'] for c in concepts)}",
                source_track_id=trimmed_id or source_id,
                regions=regions,
            )

        note = "Fertig"
        if missing:
            # Worth saying plainly: a concept nobody could find is the single
            # most likely disappointment here, and it comes back as an empty
            # colour layer that otherwise looks like a bug.
            note = "Nicht gefunden: " + ", ".join(missing)
        _set_progress(key, "done", note, 100)
        logger.info("Segmentation %s done — %s", job_id, regions)

    except Exception as exc:
        logger.exception("Segmentation %s failed", job_id)
        for path in filter(None, (mask_path, trimmed_path,
                                  settings.storage_dir / preview_rel)):
            Path(path).unlink(missing_ok=True)
        _set_progress(key, "failed", f"{exc}", 0)
    finally:
        await asyncio.sleep(120)
        if _progress.get(key, {}).get("phase") in ("done", "failed"):
            _progress.pop(key, None)


@router.get("/jobs/open")
async def open_jobs(db: AsyncSession = Depends(get_db)):
    """Structure jobs that are still waiting on something.

    The two-stage render parks after its base pass in status `review` and waits
    for a verdict — finish it, keep the preview, or discard it. That verdict is
    the point of the whole two-stage design, and until now the only thing that
    remembered which job was waiting was a variable in the browser tab. Closing
    the tab lost it: the job stayed parked in the database with a preview on
    disk and no way left in the UI to reach it.

    So the browser asks the server on load instead of remembering. `review`
    first — that is the one a person is being kept from — then anything still
    rendering, which the page can simply resume polling.
    """
    stmt = (
        select(Video)
        .where(Video.workflow == WORKFLOW_NAME,
               Video.status.in_(("review", "generating")))
        .order_by(desc(Video.created_at))
    )
    jobs = (await db.execute(stmt)).scalars().all()
    return [
        {
            "id": str(v.id),
            "status": v.status,
            "prompt": v.prompt,
            "width": v.width,
            "height": v.height,
            "frame_count": v.frame_count,
            "created_at": v.created_at.isoformat(),
            # What the poller would have shown, when the job that owns it is
            # still running in this server process. A job left over from an
            # earlier process has no live entry, which is itself the answer:
            # nothing is working on it any more.
            "live": _progress.get(str(v.id)) or None,
        }
        for v in jobs
    ]


@router.get("/segment/{job_id}/progress")
async def segment_progress(job_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """Where a segmentation job has got to.

    Its own endpoint rather than routers/video.py's: that one resolves the id
    against the `videos` table first, and a segmentation has no video row — it
    produces a control track. The shape of the answer is the same, so the
    frontend's poller does not have to learn a second one.
    """
    live = dict(_progress.get(str(job_id), {}))
    if live:
        return {
            "phase": live.get("phase", "generating"),
            "message": live.get("message", ""),
            "pct": live.get("pct", 0),
        }
    # The entry is dropped a couple of minutes after finishing, so a poller
    # that reconnects late is told the outcome by the track's existence.
    track = await db.get(ControlTrack, job_id)
    if track:
        return {"phase": "done", "message": "Fertig", "pct": 100,
                "track": _serialize_track(track)}
    raise HTTPException(status_code=404, detail="Segmentation job not found")


@router.get("/tracks/{track_id}/preview")
async def get_track_preview(track_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """The footage with its regions tinted over it.

    A flat colour-ID video is close to unreadable on its own — three silhouettes
    on black say nothing about whether the right things were caught — so the
    segmenter writes this alongside it, and this is what the library shows.
    """
    await _get_track(track_id, db)
    path = settings.storage_dir / _track_rel(track_id, "_preview.mp4")
    if not path.is_file():
        raise HTTPException(status_code=404, detail="No preview for this track")
    return FileResponse(path, media_type="video/mp4")


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
        "max_regions": MAX_REGIONS,
        # Automatic colour-ID masks. The palette is not a preference: ColorToMask
        # measures euclidean RGB distance, and pure primaries are the furthest
        # apart three keys can be, which is what buys tolerance for codec and
        # resampling error.
        "segment": {
            "colors": [list(c) for c in REGION_COLORS],
            "labels": list(REGION_LABELS),
            "max_regions": MAX_REGIONS,
            "durations": list(SEGMENT_DURATIONS),
            "strides": list(SEGMENT_STRIDES),
            "frame_ceiling": FRAME_CEILING,
            "hint": "Benenne auf Englisch, was im Bild zu sehen ist — je ein "
                    "kurzer Begriff pro Farbe. SAM 3 findet jedes Vorkommen "
                    "und verfolgt es durch den Clip; danach bekommt jede Farbe "
                    "ihr eigenes Referenzbild. Deutsche Wörter findet das "
                    "Modell nicht: „tomato\" deckt den halben Frame ab, "
                    "„Tomate\" gar nichts.",
        },
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


# A mask is only valid against the exact frames it was keyed out of. The
# segmentation path produces a matched PAIR — a decimated copy of the footage
# and the mask keyed from it — and `source_track_id` records which is which.
# Nothing used to check it at render time, and the failure is silent and
# expensive: measured 2026-09-04, a 30-frame 5 fps mask was rendered against
# the 80-frame 10 fps original it had been decimated from, so the masks ran at
# half the picture's rate and stopped entirely after frame 30. The references
# then land wherever the arithmetic puts them, which is nowhere in particular.
#
# Refused rather than quietly corrected: substituting a different control track
# changes what gets rendered, and that is the user's call, not this function's.
FRAME_RATE_TOLERANCE = 0.02


def _validate_mask_pairing(control: ControlTrack, mask: ControlTrack) -> None:
    """Refuse a mask that was not keyed out of this control track.

    Names the track that would work, because "wrong mask" is not actionable on
    its own and the right answer is always already in the database.
    """
    if mask.kind != "mask":
        raise HTTPException(
            status_code=400,
            detail=f"'{mask.title or mask.id}' ist keine Maske, sondern "
                   f"{mask.kind}-Material.",
        )

    if mask.source_track_id:
        if mask.source_track_id != control.id:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Diese Maske wurde aus einer anderen Spur gekeyt. Sie gehört "
                    f"zu {mask.source_track_id} — wähle die als Controlvideo, dann "
                    f"sitzen die Regionen auf den Bildern, für die sie berechnet "
                    f"wurden. (Aktuell gewählt: {control.id})"
                ),
            )
        # Lineage settles it. The frame and rate checks below exist to catch a
        # mismatch this one cannot see, and running them anyway would reject a
        # correct pair whose stored fps is stale — several rows in this library
        # carry 1000 and 2000 from the old r_frame_rate probe.
        return

    # A hand-authored mask carries no lineage, so the only thing left to check
    # is whether it can physically line up: same number of frames, same rate.
    if mask.frame_count and control.frame_count and mask.frame_count < control.frame_count:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Die Maske hat {mask.frame_count} Frames, das Controlvideo "
                f"{control.frame_count}. Ab Frame {mask.frame_count + 1} gäbe es "
                f"keine Maske mehr — kürze das Controlvideo oder segmentiere neu."
            ),
        )
    if mask.fps and control.fps:
        drift = abs(mask.fps - control.fps) / max(control.fps, 1e-6)
        if drift > FRAME_RATE_TOLERANCE:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Die Maske läuft mit {mask.fps:g} fps, das Controlvideo mit "
                    f"{control.fps:g}. Die Masken würden dem Bild davonlaufen."
                ),
            )


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
        _validate_mask_pairing(control, mask)

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
        # Appended, never inserted: `_lcm_request` reads the regions off the
        # front of this list by index, so the base has to be the tail.
        if body.base_image_id:
            references.append(await _reference_for(body.base_image_id, db))
            image_ids.append(body.base_image_id)
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
    # Check the card — and that ComfyUI is there at all — before posting.
    # Without this the two ways a submission fails both surface as
    # `httpx.ConnectError: All connection attempts failed`, which says nothing
    # about ComfyUI being down; and the worse one does not even fail here — a
    # CUDA OOM kills ComfyUI's prompt-worker *thread* while its HTTP server
    # keeps handing out prompt ids, so the job sits "generating" until the poll
    # times out two hours later. `free_vram_for` names both.
    #
    # This path had no pre-flight at all while it was the only GPU consumer in
    # the tool. Segmentation made it a second one, so it needs the same manners
    # as routers/video.py.
    await free_vram_for(_LCM_MIN_FREE_VRAM, "AnimateLCM/VACE")
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
        # Per-region weight when one was sent, the job's global one otherwise.
        # Never `r.strength`: that is VACE's denoise and means nothing here.
        regions=[
            LcmRegion(
                color=tuple(r.color), reference=staged[i], threshold=r.threshold,
                weight=(r.ip_weight if r.ip_weight is not None else req.ip_weight),
            )
            for i, r in enumerate(req.regions)
        ],
        # The staged list is [region refs…, base?] — see `generate`. Anything
        # past the regions is the base picture.
        base_reference=(staged[len(req.regions)]
                        if len(staged) > len(req.regions) else None),
        base_weight=req.base_weight,
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
