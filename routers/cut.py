"""
Rhythmusschnitt — cut a selection of clips to a song's beat grid.

Three endpoints, and the split between them is the whole design:

  GET  /api/cut/styles   what the planner can be asked for
  POST /api/cut/plan     the edit, as data — no render, no Video row
  POST /api/cut/render   the same edit, this time to a file

`plan` is cheap (a cached beat map plus some arithmetic — tens of
milliseconds) and `render` is not. That is why they are separate: the user
re-rolls the seed and switches styles against `plan` until the timeline looks
right, and only then spends a minute of ffmpeg on it.

**The render does not accept a plan.** It accepts the same parameters the
preview was built from, including the seed the preview resolved, and rebuilds
it. `plan_cut` is deterministic in those parameters, so what was previewed is
what gets rendered — without a round-trip of client-supplied timings that would
all have to be re-validated against the sources anyway.

The song is not part of the render. The picture is cut silent and the existing
soundtrack path in routers/video.py muxes the track onto it afterwards, which
means a beat cut is an ordinary finished video from that moment on: upscale,
grain, Instagram, YouTube and every re-render already know what to do with it.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from core.auth import require_auth
from core.config import settings
from core.db import AsyncSessionLocal, get_db
from core.models import Song, Video
from core.tasks import safe_create_task
from core.video_thumb import probe_video_duration
from routers.video import (
    MergeItem,
    _finalize_video_failure,
    _finalize_video_done,
    _merge_canvas,
    _progress,
    _resolve_merge_sources,
    _run_soundtrack_mux,
    _set_progress,
)
from services.video import beats
from services.video.audio_bed import BED_VOLUME_DEFAULT, clamp_bed_volume
from services.video.cut import (
    DEFAULT_STYLE,
    LADDER,
    MAX_STRETCH_CEILING,
    MAX_STRETCH_DEFAULT,
    STYLE_BY_KEY,
    EditPlan,
    Source,
    plan_cut,
    style_options,
)
from services.video.cut_render import RenderSource, plan_duration, render_cut

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/cut", dependencies=[Depends(require_auth)])

WORKFLOW_NAME = "beatcut"
MAX_SOURCES = 50

# A cut point can only land on a rendered frame, so half a frame is the floor on
# how early a cut can be pulled. 200 ms is roughly a sixteenth at 75 BPM — past
# that the picture stops anticipating the beat and starts missing it.
ANTICIPATION_MAX = 0.20


# ── Request bodies ───────────────────────────────────────────────────────────

class PlanRequest(BaseModel):
    song_id: uuid.UUID
    items: list[MergeItem] = Field(default_factory=list)
    style: str = DEFAULT_STYLE
    seed: Optional[int] = None            # None = roll a fresh one
    start_bar: int = 0
    end_bar: Optional[int] = None
    max_stretch: float = MAX_STRETCH_DEFAULT
    anticipation: float = 0.0             # seconds to cut ahead of the beat
    beats_per_bar: Literal[2, 3, 4, 6] = 4


class RenderRequest(PlanRequest):
    # Same parameters, plus what happens to the finished picture. The seed is
    # not optional here in practice — the client sends back the one its preview
    # resolved — but a missing one still renders something rather than failing.
    include_bed: bool = False
    bed_volume: float = BED_VOLUME_DEFAULT
    title: Optional[str] = None


# ── Shared resolution ────────────────────────────────────────────────────────

async def _song_beatmap(song: Song, beats_per_bar: int) -> beats.BeatMap:
    """The song's beat map, from the sidecar cache when there is one.

    `songs.bpm` is what ACE-Step was asked for when the track was generated,
    and handing it over as a hint is the single biggest accuracy win available
    here — see the module docstring of services/video/beats.py.
    """
    if not song.filepath:
        raise HTTPException(status_code=409, detail="Song has no file")
    path = settings.storage_dir / song.filepath
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Song file missing on disk")
    try:
        return await beats.load_or_analyze(
            path,
            beats.cache_path(settings.songs_dir, song.id),
            bpm_hint=float(song.bpm) if song.bpm else None,
            beats_per_bar=beats_per_bar,
            ffmpeg_path=settings.ffmpeg_path,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=422, detail=f"Could not analyse the song: {exc}") from exc


async def _resolve_sources(items: list[MergeItem]):
    """Selection → (planner sources, render sources, canvas, fps).

    Reuses the merge resolver, which already knows that a clip's real file may
    be its upscaled sibling and that a finished video has to be probed rather
    than believed. The one thing it does not report is duration, and the
    planner needs nothing else — so that is probed here.
    """
    if not items:
        raise HTTPException(status_code=400, detail="Nothing selected to cut")
    if len(items) > MAX_SOURCES:
        raise HTTPException(status_code=400, detail=f"At most {MAX_SOURCES} sources")
    keys = [(i.kind, i.id) for i in items]
    if len(set(keys)) != len(keys):
        raise HTTPException(status_code=400, detail="Duplicate source in selection")

    try:
        resolved = await _resolve_merge_sources(keys)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    planner: list[Source] = []
    renderer: list[RenderSource] = []
    for position, ((kind, sid), src) in enumerate(zip(keys, resolved), start=1):
        duration = await probe_video_duration(src.inp.path)
        if duration <= 0:
            raise HTTPException(
                status_code=422,
                detail=f"Could not read the length of {src.inp.path.name}",
            )
        # Named by its place in the selection, which is the number the merge
        # bar already shows on the row the user clicked. Filenames would be no
        # use here — every job writes "seg_0", so a legend built from them
        # reads "seg_0, seg_0, seg_0".
        label = f"{'Clip' if kind == 'clip' else 'Video'} {position}"
        planner.append(Source(key=f"{kind}:{sid}", duration=duration, label=label))
        renderer.append(RenderSource(path=src.inp.path, duration=duration))

    width, height = _merge_canvas([(s.width, s.height) for s in resolved])
    # The highest rate in the selection, because it is the one that quantises
    # the cut points finest: at 24 fps a cut lands within 21 ms of its beat, at
    # 30 within 17. Sources below the target are frame-doubled by `fps=`.
    fps = max((s.fps or 24) for s in resolved)
    return planner, renderer, (width, height), fps


def _build_plan(body: PlanRequest, beatmap: beats.BeatMap, sources: list[Source]) -> EditPlan:
    if body.style not in STYLE_BY_KEY:
        raise HTTPException(status_code=400, detail=f"Unknown style: {body.style}")
    try:
        return plan_cut(
            beatmap, sources,
            style=body.style,
            seed=body.seed,
            start_bar=max(0, body.start_bar),
            end_bar=body.end_bar,
            max_stretch=body.max_stretch,
            anticipation=max(0.0, min(body.anticipation, ANTICIPATION_MAX)),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


# ── Endpoints ────────────────────────────────────────────────────────────────

@router.get("/styles")
async def cut_styles():
    """The styles and the dials, read off the planner's own tables so a
    retuned style never has to be declared twice."""
    return {
        "styles": style_options(),
        "default_style": DEFAULT_STYLE,
        "ladder": list(LADDER),
        "max_stretch_default": MAX_STRETCH_DEFAULT,
        "max_stretch_ceiling": MAX_STRETCH_CEILING,
        "anticipation_max": ANTICIPATION_MAX,
        "max_sources": MAX_SOURCES,
    }


@router.post("/plan")
async def make_plan(body: PlanRequest, db: AsyncSession = Depends(get_db)):
    """Build an edit and return it, without rendering anything.

    The response carries the beat map alongside the plan because the timeline
    the user judges this by draws both: the bars and their energy underneath,
    the shots on top.
    """
    song = await db.get(Song, body.song_id)
    if not song:
        raise HTTPException(status_code=404, detail="Song not found")
    if song.status != "done":
        raise HTTPException(status_code=409, detail="Song is not ready")

    beatmap = await _song_beatmap(song, body.beats_per_bar)
    sources, _, (width, height), fps = await _resolve_sources(body.items)
    plan = _build_plan(body, beatmap, sources)

    return {
        "plan": plan.to_json(),
        "beatmap": {
            "bpm": beatmap.bpm,
            "duration": beatmap.duration,
            "beats_per_bar": beatmap.beats_per_bar,
            "bar_count": beatmap.bar_count,
            "bar_energy": beatmap.bar_energy,
            # When each bar starts, so the timeline can place the energy row on
            # the same axis as the shots instead of spacing it evenly and
            # drifting by up to a bar against them.
            "bar_times": [beatmap.beats[i] for i in beatmap.bar_starts()],
            "sections": beatmap.sections,
            "confidence": beatmap.confidence,
        },
        "sources": [{"key": s.key, "label": s.label, "duration": round(s.duration, 3)}
                    for s in sources],
        "output": {
            "width": width, "height": height, "fps": fps,
            "duration": round(plan_duration(plan, fps), 3),
        },
        "song": {
            "id": str(song.id),
            "title": song.title,
            "url": f"/api/music/file/{song.filename}" if song.filename else None,
            "requested_bpm": song.bpm,
        },
    }


@router.post("/render", status_code=202)
async def render(body: RenderRequest, db: AsyncSession = Depends(get_db)):
    """Render the edit and attach the song. Returns immediately; poll
    `GET /api/video/jobs/{video_id}/progress` like any other video job."""
    song = await db.get(Song, body.song_id)
    if not song:
        raise HTTPException(status_code=404, detail="Song not found")
    if song.status != "done" or not song.filename:
        raise HTTPException(status_code=409, detail="Song is not ready")

    beatmap = await _song_beatmap(song, body.beats_per_bar)
    sources, _, (width, height), fps = await _resolve_sources(body.items)
    plan = _build_plan(body, beatmap, sources)

    video = Video(
        id=uuid.uuid4(),
        workflow=WORKFLOW_NAME,
        status="assembling",
        title=(body.title or "").strip()[:255] or None,
        width=width,
        height=height,
        fps=fps,
        n_images=len(sources),
        frame_count=round(plan_duration(plan, fps) * fps),
        cut_plan=plan.to_json(),
        soundtrack_start_seconds=plan.song_start or None,
        created_at=datetime.now(timezone.utc),
    )
    db.add(video)
    await db.commit()

    bed = clamp_bed_volume(body.bed_volume) if body.include_bed else None
    safe_create_task(
        _run_beat_cut(video.id, plan, body.song_id, bed, [(i.kind, i.id) for i in body.items],
                      width, height, fps),
        name=f"beatcut:{video.id}",
    )
    logger.info(
        "Queued beat cut %s — %d cuts from %d source(s), %s at %.1f BPM, %.1f s",
        video.id, len(plan.cuts), len(sources), plan.style, plan.bpm,
        plan_duration(plan, fps),
    )
    return {
        "video_id": str(video.id),
        "status": "assembling",
        "cuts": len(plan.cuts),
        "duration": round(plan_duration(plan, fps), 3),
    }


# ── Background render ────────────────────────────────────────────────────────

async def _run_beat_cut(
    video_id: uuid.UUID,
    plan: EditPlan,
    song_id: uuid.UUID,
    bed_volume: float | None,
    keys: list[tuple[str, uuid.UUID]],
    width: int,
    height: int,
    fps: int,
) -> None:
    """Render the picture, mux the song onto it, and only then call it done.

    **The order matters and the status matters.** A Rhythmusschnitt without its
    song is not the thing that was asked for, so the row stays 'assembling'
    across the mux. Flipping it to 'done' after the picture — which is what this
    used to do — let the frontend's poller stop one tick too early, latch the
    silent rendition into the card, and never learn that a muxed sibling had
    appeared a second later. It also opened a window in which an upscale could
    start from the file the mux was about to supersede.

    Sources are resolved a second time rather than passed in: minutes can pass
    between the request and the render starting, and a clip that was deleted or
    re-upscaled in between must be seen as it is now.
    """
    key = str(video_id)
    try:
        _set_progress(key, "assembling", f"Schnitt wird gerendert — {len(plan.cuts)} Einstellungen…", 10)
        resolved = await _resolve_merge_sources(keys)
        renderer = [
            RenderSource(path=s.inp.path, duration=await probe_video_duration(s.inp.path))
            for s in resolved
        ]

        settings.videos_dir.mkdir(parents=True, exist_ok=True)
        dest = settings.videos_dir / f"{video_id}_artrium.mp4"
        await render_cut(
            plan, renderer, dest, width, height, fps, ffmpeg_path=settings.ffmpeg_path,
        )

        # The file lands and the thumbnail is written, but the row stays
        # 'assembling': there is still a song to put on it.
        await _finalize_video_done(video_id, dest, key, status="assembling")
        logger.info("Beat cut %s rendered (%d cuts)", video_id, len(plan.cuts))

    except Exception as exc:
        logger.exception("Beat cut %s failed", video_id)
        await _finalize_video_failure(video_id, exc, key)
        return

    # The song goes on through the same path every other soundtrack takes, so
    # the offset, the bed, and every later re-render behave identically.
    _set_progress(key, "muxing", "Song wird untergelegt…", 97)
    await _run_soundtrack_mux(video_id, song_id, bed_volume)
    await _finish_beat_cut(video_id, key)


async def _finish_beat_cut(video_id: uuid.UUID, key: str) -> None:
    """Flip the row to 'done' once the song is on it.

    A mux that failed leaves `error` set and `muxed_filename` empty. The cut
    itself is still a perfectly good silent video, so it is finished rather
    than thrown away — but the error stays on the row, because "your song is
    not on this" is exactly what the card needs to say.
    """
    async with AsyncSessionLocal() as db:
        video = await db.get(Video, video_id)
        if not video:
            _progress.pop(key, None)
            return
        muxed = bool(video.muxed_filename)
        video.status = "done"
        await db.commit()

    _progress.pop(key, None)
    if muxed:
        logger.info("Beat cut %s finished with its soundtrack", video_id)
    else:
        logger.warning("Beat cut %s finished WITHOUT its soundtrack", video_id)
