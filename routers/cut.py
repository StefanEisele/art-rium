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

**Sources come in ungrained, and the finished cut is what gets grained.** A
video that was grained on its own carries that grain into the edit, and graining
the edit afterwards would give that one shot two passes and its neighbours one.
So `_resolve_merge_sources(ungrained=True)` reads the rendition below the grain
— see `_video_ungrained_name` — and the grain belongs to the piece.

Colour harmonisation is measured here and applied in the render: every source is
sampled, the group's median is the target, and each clip moves a fraction of the
way there. See services/video/grade.py for why a fraction and not all the way.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
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
from services.improv.piano_song import is_recording
from services.video import beats, grade, piano
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
from services.video.cut_render import (
    DEFAULT_MOTION,
    RenderSource,
    clamp_motion,
    motion_options,
    plan_duration,
    render_cut,
)
from services.video.layer_render import (
    LayerSource,
    render_layers,
)
from services.video.layer_render import plan_duration as layer_duration
from services.video.layers import DEFAULT_STYLE as LAYER_DEFAULT_STYLE
from services.video.layers import STYLE_BY_KEY as LAYER_STYLE_BY_KEY
from services.video.layers import (
    ACCENT_SECONDS,
    DEFAULT_TRANSITION,
    GLOW_DEFAULT,
    PULSE_DEFAULT,
    PULSE_MAX,
    SPEED_HIGH_DEFAULT,
    SPEED_LOW_DEFAULT,
    SPEED_MAX,
    SPEED_MIN,
    SWELL_DEFAULT,
    TOUCH_DEFAULT,
    LayerPlan,
    plan_layers,
    transition_options,
)
from services.video.layers import style_options as layer_style_options

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/cut", dependencies=[Depends(require_auth)])

WORKFLOW_NAME = "beatcut"
LAYER_WORKFLOW_NAME = "layercut"
MAX_SOURCES = 50

# More than this and the stack stops being readable as one form with changing
# material and starts being a slideshow nobody can follow.
MAX_LAYERS = 12

# Below this the transitions are two or three frames long and the speed shifts
# judder. The sources are meant to be finalised through the retime pass first
# (POST /api/video/jobs/{id}/upscale with resolution 0), and the planner says so
# rather than silently producing something that looks broken.
SMOOTH_FPS = 16

# Tracks are supposed to be the same length — they come out of one control
# video. A spread wider than this means something else got into the selection,
# which is worth saying out loud because the shared loop becomes the shortest
# of them and the rest are simply never seen to the end.
LOOP_SPREAD_TOLERANCE = 0.25

# A cut point can only land on a rendered frame, so half a frame is the floor on
# how early a cut can be pulled. 200 ms is roughly a sixteenth at 75 BPM — past
# that the picture stops anticipating the beat and starts missing it.
ANTICIPATION_MAX = 0.20

# The timeline is a few hundred pixels wide and a four-minute recording has two
# thousand dynamics samples. Thinned rather than averaged — the curve is
# already smoothed, and thinning keeps its peaks where they are.
TIMELINE_POINTS = 480


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
    # How hard to pull the clips' colour and contrast together. A key from
    # services/video/grade.py::HARMONIES, never a raw number: the useful range
    # is narrow and its top end is a place nobody wants to be.
    harmonize: str = grade.DEFAULT_HARMONY
    # How a retimed shot is put back on the frame grid. Not a planning
    # parameter — the plan is identical whatever this says — but it rides on
    # the plan request so the client can carry one body to both endpoints.
    motion: str = DEFAULT_MOTION


class RenderRequest(PlanRequest):
    # Same parameters, plus what happens to the finished picture. The seed is
    # not optional here in practice — the client sends back the one its preview
    # resolved — but a missing one still renders something rather than failing.
    include_bed: bool = False
    bed_volume: float = BED_VOLUME_DEFAULT
    title: Optional[str] = None
    # Off by default: a picture edit that opens on a black frame is a choice,
    # and it used to be made unconditionally for every render.
    fade_in: bool = False


# ── Colour measurement ───────────────────────────────────────────────────────
# The planner is re-run on every knob the user touches, and sampling six clips
# costs a second or two — far too much to pay per keystroke and completely
# unnecessary, because a rendered file never changes under its own name. Keyed
# on (path, size, mtime) so a re-upscaled clip is re-measured and nothing else
# is. Bounded, because the tool runs for weeks at a time.

_STATS_CACHE: dict[tuple[str, int, int], grade.Stats | None] = {}
_STATS_CACHE_MAX = 512


async def _measure_sources(paths: list[Path]) -> list[grade.Stats | None]:
    """Colour statistics per source file, cached across planner calls."""
    out: list[grade.Stats | None] = []
    for path in paths:
        try:
            stat = path.stat()
            key = (str(path), stat.st_size, int(stat.st_mtime))
        except OSError:
            out.append(None)
            continue
        if key not in _STATS_CACHE:
            if len(_STATS_CACHE) >= _STATS_CACHE_MAX:
                _STATS_CACHE.clear()
            _STATS_CACHE[key] = await grade.measure(
                path, ffmpeg_path=settings.ffmpeg_path,
            )
        out.append(_STATS_CACHE[key])
    return out


def _played_fields(beatmap: beats.BeatMap) -> dict:
    """What the timeline draws for a recorded performance, on top of the bars:
    its dynamics curve, the accents, and where the phrases begin. A generated
    song answers with its profile alone."""
    if not beatmap.is_played:
        return {"profile": beatmap.profile}
    step = max(1, -(-len(beatmap.dynamics) // TIMELINE_POINTS))
    return {
        "profile": beatmap.profile,
        "dynamics": beatmap.dynamics[::step],
        "dynamics_rate": beatmap.dynamics_rate / step,
        "accents": beatmap.accents,
        "accent_strength": beatmap.accent_strength,
        "phrases": beatmap.phrases,
    }


def _harmony_strength(key: str) -> float:
    """Preset key → strength. An unknown key harmonises nothing rather than
    guessing, so a stale client cannot silently regrade a piece."""
    return grade.HARMONY_BY_KEY.get(key, 0.0)


# ── Shared resolution ────────────────────────────────────────────────────────

async def _song_beatmap(song: Song, beats_per_bar: int) -> beats.BeatMap:
    """The song's beat map, from the sidecar cache when there is one.

    `songs.bpm` is what ACE-Step was asked for when the track was generated,
    and handing it over as a hint is the single biggest accuracy win available
    here — see the module docstring of services/video/beats.py.

    A piano recording taken in by the improv tool is read as a performance
    instead: a tempo that is allowed to move, a continuous dynamics curve,
    accents and phrases (services/video/piano.py). Everything downstream reads
    the same BeatMap either way.
    """
    if not song.filepath:
        raise HTTPException(status_code=409, detail="Song has no file")
    path = settings.storage_dir / song.filepath
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Song file missing on disk")
    try:
        if is_recording(song):
            return await piano.load_or_analyze_piano(
                path,
                beats.cache_path(settings.songs_dir, song.id),
                beats_per_bar=beats_per_bar,
                ffmpeg_path=settings.ffmpeg_path,
            )
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
        # Ungrained: the finished edit is what gets grained, not the shots in it.
        resolved = await _resolve_merge_sources(keys, ungrained=True)
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
        "harmonies": grade.harmony_options(),
        "default_harmony": grade.DEFAULT_HARMONY,
        "motions": motion_options(),
        "default_motion": DEFAULT_MOTION,
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
    sources, renderers, (width, height), fps = await _resolve_sources(body.items)
    plan = _build_plan(body, beatmap, sources)

    samples = await _measure_sources([r.path for r in renderers])
    grades = grade.harmonise(samples, _harmony_strength(body.harmonize))

    return {
        "plan": plan.to_json(),
        # What the grade would do, per clip, before a minute of ffmpeg is spent
        # on it — "how far is this moving my material" is the only question
        # worth asking about an automatic colour pass.
        "harmonize": {
            "key": body.harmonize,
            "strength": _harmony_strength(body.harmonize),
            "spread": grade.spread(samples),
            "clips": [g.describe() for g in grades],
            "touched": sum(1 for g in grades if not g.is_identity),
        },
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
            **_played_fields(beatmap),
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
                      width, height, fps, _harmony_strength(body.harmonize),
                      fade_in=body.fade_in, motion=clamp_motion(body.motion)),
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
    harmony: float = 0.0,
    *,
    fade_in: bool = False,
    motion: str = DEFAULT_MOTION,
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
        resolved = await _resolve_merge_sources(keys, ungrained=True)
        paths = [s.inp.path for s in resolved]

        # Re-measured rather than carried from the plan request: the cache makes
        # it free when nothing moved, and a clip that was re-upscaled in between
        # has to be graded as it is now, not as it was when the plan was drawn.
        grades = grade.harmonise(await _measure_sources(paths), harmony)
        if any(not g.is_identity for g in grades):
            _set_progress(key, "assembling", "Clips werden angeglichen…", 12)
            logger.info("Beat cut %s colour grades: %s", video_id,
                        "; ".join(g.describe() for g in grades))

        renderer = [
            RenderSource(path=p, duration=await probe_video_duration(p), grade=g.filter_chain())
            for p, g in zip(paths, grades)
        ]

        settings.videos_dir.mkdir(parents=True, exist_ok=True)
        dest = settings.videos_dir / f"{video_id}_artrium.mp4"
        await render_cut(
            plan, renderer, dest, width, height, fps, ffmpeg_path=settings.ffmpeg_path,
            fade_in=fade_in, motion=motion,
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


# ── Schichtenschnitt ─────────────────────────────────────────────────────────
# The same beat map, read a different way: N renders of ONE control video,
# stacked rather than sequenced, cross-dissolved at a shared source position so
# the form stands still while the material changes. See services/video/layers.py
# for why that shared position is the whole feature.
#
# Everything downstream is deliberately identical to the beat cut — same source
# resolution, same colour harmonisation, same silent render with the song muxed
# on afterwards — so a layer cut is an ordinary finished video from the moment
# it lands, and upscale, look, Instagram and YouTube already know what to do
# with it.


class LayerPlanRequest(BaseModel):
    song_id: uuid.UUID
    items: list[MergeItem] = Field(default_factory=list)
    style: str = LAYER_DEFAULT_STYLE
    transition: str = DEFAULT_TRANSITION
    seed: Optional[int] = None
    start_bar: int = 0
    end_bar: Optional[int] = None
    beats_per_bar: Literal[2, 3, 4, 6] = 4
    speed_low: float = SPEED_LOW_DEFAULT
    speed_high: float = SPEED_HIGH_DEFAULT
    # The two halves of the speed curve. `swell` is how far the bar energy
    # moves the tempo, `pulse` how hard the picture brakes into a change of
    # material and pushes through it. Either at 0 removes that half.
    swell: float = SWELL_DEFAULT
    pulse: float = PULSE_DEFAULT
    # Only read when the music is a recorded performance: how far the playing
    # shapes each transition, and how strongly accents between changes light
    # the picture up. See services/video/layers.py.
    touch: float = TOUCH_DEFAULT
    glow: float = GLOW_DEFAULT
    harmonize: str = grade.DEFAULT_HARMONY
    # See PlanRequest.motion. It matters more here than in a beat cut: the
    # whole stack is retimed by the energy curve, so nearly every slot in a
    # layer edit is being resampled.
    motion: str = DEFAULT_MOTION


class LayerRenderRequest(LayerPlanRequest):
    include_bed: bool = False
    bed_volume: float = BED_VOLUME_DEFAULT
    title: Optional[str] = None
    # Off by default — see RenderRequest.fade_in.
    fade_in: bool = False


def _bar_span(beatmap: beats.BeatMap, start_bar: int, end_bar: int | None):
    """Bar numbers → seconds on the song's timeline.

    Bar 0 starts at 0.0 rather than at the first tracked downbeat: a grid
    begins where the tracker could first justify one, which on a 4/4 track can
    be a second and a half in, and an edit that opened there would leave the
    intro with no picture on it.
    """
    starts = beatmap.bar_starts()
    if not starts:
        return 0.0, beatmap.duration
    lo = max(0, min(start_bar, len(starts) - 1))
    start = 0.0 if lo == 0 else beatmap.beats[starts[lo]]
    if end_bar is None or end_bar >= len(starts):
        return start, beatmap.duration
    hi = max(lo + 1, end_bar)
    end = beatmap.beats[starts[hi]] if hi < len(starts) else beatmap.duration
    return start, end


async def _resolve_layers(items: list[MergeItem]):
    """Selection → (layer sources, canvas, fps, loop, warnings).

    Reuses the beat cut's resolver, so a clip's real file is still its upscaled
    sibling and a finished video is still probed rather than believed. What is
    added here is the shared loop length and the checks that only matter when
    the tracks are supposed to be interchangeable.
    """
    sources, renderers, (width, height), fps = await _resolve_sources(items)
    if len(sources) < 2:
        raise HTTPException(
            status_code=400,
            detail="Ein Schichtenschnitt braucht mindestens zwei Spuren",
        )
    if len(sources) > MAX_LAYERS:
        raise HTTPException(
            status_code=400, detail=f"Höchstens {MAX_LAYERS} Spuren",
        )

    durations = [r.duration for r in renderers]
    loop = min(durations)
    warnings: list[str] = []
    spread = (max(durations) - loop) / loop if loop > 0 else 0.0
    if spread > LOOP_SPREAD_TOLERANCE:
        warnings.append(
            f"Die Spuren sind unterschiedlich lang ({loop:.1f}s bis "
            f"{max(durations):.1f}s). Geschnitten wird auf die kürzeste — von den "
            "längeren wird das Ende nie gezeigt."
        )
    if fps < SMOOTH_FPS:
        warnings.append(
            f"Die Spuren laufen mit {fps} fps. Überblendungen sind dabei nur "
            "wenige Frames lang und die Tempowechsel ruckeln — finalisiere die "
            "Spuren zuerst mit Interpolation auf 24 fps."
        )
    return sources, renderers, (width, height), fps, loop, warnings


def _build_layer_plan(
    body: LayerPlanRequest, beatmap: beats.BeatMap, tracks: int, loop: float,
    fps: int = 24,
) -> LayerPlan:
    if body.style not in LAYER_STYLE_BY_KEY:
        raise HTTPException(status_code=400, detail=f"Unknown style: {body.style}")
    span = _bar_span(beatmap, max(0, body.start_bar), body.end_bar)
    try:
        return plan_layers(
            beatmap,
            tracks=tracks,
            loop=loop,
            style=body.style,
            transition=body.transition,
            seed=body.seed,
            speed_low=body.speed_low,
            speed_high=body.speed_high,
            swell=body.swell,
            pulse=body.pulse,
            touch=body.touch,
            glow=body.glow,
            # The planner needs the target rate: a segment that comes out one
            # frame long collapses in the filtergraph, so the floor it keeps
            # its knots above is measured in frames.
            fps=fps,
            span=span,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/layer/styles")
async def layer_styles():
    """What the layer planner can be asked for, read off its own tables."""
    return {
        "styles": layer_style_options(),
        "default_style": LAYER_DEFAULT_STYLE,
        "transitions": transition_options(),
        "default_transition": DEFAULT_TRANSITION,
        "harmonies": grade.harmony_options(),
        "default_harmony": grade.DEFAULT_HARMONY,
        "motions": motion_options(),
        "default_motion": DEFAULT_MOTION,
        "ladder": list(LADDER),
        "speed": {
            "min": SPEED_MIN, "max": SPEED_MAX,
            "low_default": SPEED_LOW_DEFAULT, "high_default": SPEED_HIGH_DEFAULT,
            "swell_default": SWELL_DEFAULT,
            "pulse_default": PULSE_DEFAULT, "pulse_max": PULSE_MAX,
            "accent_seconds": ACCENT_SECONDS,
        },
        "played": {"touch_default": TOUCH_DEFAULT, "glow_default": GLOW_DEFAULT},
        "max_layers": MAX_LAYERS,
        "smooth_fps": SMOOTH_FPS,
    }


@router.post("/layer/plan")
async def make_layer_plan(body: LayerPlanRequest, db: AsyncSession = Depends(get_db)):
    """Build a layer edit and return it, without rendering anything.

    Cheap — a cached beat map plus arithmetic — so the client can re-roll the
    seed and switch styles against this until the timeline reads right, and
    only then spend ffmpeg on it.
    """
    song = await db.get(Song, body.song_id)
    if not song:
        raise HTTPException(status_code=404, detail="Song not found")
    if song.status != "done":
        raise HTTPException(status_code=409, detail="Song is not ready")

    beatmap = await _song_beatmap(song, body.beats_per_bar)
    sources, renderers, (width, height), fps, loop, warnings = \
        await _resolve_layers(body.items)
    plan = _build_layer_plan(body, beatmap, len(sources), loop, fps)
    plan.warnings = warnings + plan.warnings

    samples = await _measure_sources([r.path for r in renderers])
    grades = grade.harmonise(samples, _harmony_strength(body.harmonize))

    return {
        "plan": plan.to_json(),
        "harmonize": {
            "key": body.harmonize,
            "strength": _harmony_strength(body.harmonize),
            "spread": grade.spread(samples),
            "clips": [g.describe() for g in grades],
            "touched": sum(1 for g in grades if not g.is_identity),
        },
        "beatmap": {
            "bpm": beatmap.bpm,
            "duration": beatmap.duration,
            "beats_per_bar": beatmap.beats_per_bar,
            "bar_count": beatmap.bar_count,
            "bar_energy": beatmap.bar_energy,
            "bar_times": [beatmap.beats[i] for i in beatmap.bar_starts()],
            "sections": beatmap.sections,
            "confidence": beatmap.confidence,
            **_played_fields(beatmap),
        },
        "sources": [{"key": s.key, "label": s.label, "duration": round(s.duration, 3)}
                    for s in sources],
        "output": {
            "width": width, "height": height, "fps": fps,
            "duration": round(layer_duration(plan, fps), 3),
            "loop": round(loop, 3),
            # How much of the loop the piece actually walks through, and how
            # many times it comes back around. Both are the answer to "will
            # this feel repetitive", which is the question this mode invites.
            "source_consumed": round(plan.source_consumed, 3),
            "loops_used": round(plan.source_consumed / loop, 2) if loop else 0,
        },
        "song": {
            "id": str(song.id),
            "title": song.title,
            "url": f"/api/music/file/{song.filename}" if song.filename else None,
            "requested_bpm": song.bpm,
            "recording": is_recording(song),
        },
    }


@router.post("/layer/render", status_code=202)
async def render_layer(body: LayerRenderRequest, db: AsyncSession = Depends(get_db)):
    """Render the layer edit and attach the song. Returns immediately; poll
    `GET /api/video/jobs/{video_id}/progress` like any other video job."""
    song = await db.get(Song, body.song_id)
    if not song:
        raise HTTPException(status_code=404, detail="Song not found")
    if song.status != "done" or not song.filename:
        raise HTTPException(status_code=409, detail="Song is not ready")

    beatmap = await _song_beatmap(song, body.beats_per_bar)
    sources, _, (width, height), fps, loop, _warnings = await _resolve_layers(body.items)
    plan = _build_layer_plan(body, beatmap, len(sources), loop, fps)

    video = Video(
        id=uuid.uuid4(),
        workflow=LAYER_WORKFLOW_NAME,
        status="assembling",
        title=(body.title or "").strip()[:255] or None,
        width=width,
        height=height,
        fps=fps,
        n_images=len(sources),
        frame_count=round(layer_duration(plan, fps) * fps),
        cut_plan=plan.to_json(),
        soundtrack_start_seconds=plan.song_start or None,
        created_at=datetime.now(timezone.utc),
    )
    db.add(video)
    await db.commit()

    bed = clamp_bed_volume(body.bed_volume) if body.include_bed else None
    safe_create_task(
        _run_layer_cut(video.id, plan, body.song_id, bed,
                       [(i.kind, i.id) for i in body.items],
                       width, height, fps, _harmony_strength(body.harmonize),
                       fade_in=body.fade_in, motion=clamp_motion(body.motion)),
        name=f"layercut:{video.id}",
    )
    blends = sum(1 for s in plan.slots if s.is_blend)
    logger.info(
        "Queued layer cut %s — %d changes in %d segments (%d transitions) over "
        "%d tracks, %s/%s at %.1f BPM, %.1f s, Schwelle=%.2f Puls=%.2f, %s "
        "(Anschlag=%.2f, %d glows)",
        video.id, plan.changes, len(plan.slots), blends, len(sources),
        plan.style, plan.transition, plan.bpm, layer_duration(plan, fps),
        plan.swell, plan.pulse, plan.profile, plan.touch, len(plan.glows),
    )
    return {
        "video_id": str(video.id),
        "status": "assembling",
        "slots": plan.changes,
        "segments": len(plan.slots),
        "transitions": blends,
        "duration": round(layer_duration(plan, fps), 3),
    }


async def _run_layer_cut(
    video_id: uuid.UUID,
    plan: LayerPlan,
    song_id: uuid.UUID,
    bed_volume: float | None,
    keys: list[tuple[str, uuid.UUID]],
    width: int,
    height: int,
    fps: int,
    harmony: float = 0.0,
    *,
    fade_in: bool = False,
    motion: str = DEFAULT_MOTION,
) -> None:
    """Render the stack, mux the song onto it, and only then call it done.

    Deliberately the same shape as `_run_beat_cut`, down to staying
    'assembling' across the mux — see that function for why the status matters
    more than it looks like it should.
    """
    key = str(video_id)
    try:
        blends = sum(1 for s in plan.slots if s.is_blend)
        _set_progress(
            key, "assembling",
            f"Schichten werden gerendert — {len(plan.slots)} Abschnitte, "
            f"{blends} Überblendungen…", 10,
        )
        resolved = await _resolve_merge_sources(keys, ungrained=True)
        paths = [s.inp.path for s in resolved]

        # Harmonisation matters more here than in a beat cut: two tracks that
        # sit at different black levels announce every dissolve as a brightness
        # step, which is exactly the thing the shared form is supposed to hide.
        grades = grade.harmonise(await _measure_sources(paths), harmony)
        if any(not g.is_identity for g in grades):
            _set_progress(key, "assembling", "Spuren werden angeglichen…", 12)
            logger.info("Layer cut %s colour grades: %s", video_id,
                        "; ".join(g.describe() for g in grades))

        renderer = [
            LayerSource(path=p, duration=await probe_video_duration(p),
                        grade=g.filter_chain())
            for p, g in zip(paths, grades)
        ]

        settings.videos_dir.mkdir(parents=True, exist_ok=True)
        dest = settings.videos_dir / f"{video_id}_artrium.mp4"
        await render_layers(
            plan, renderer, dest, width, height, fps, ffmpeg_path=settings.ffmpeg_path,
            fade_in=fade_in, motion=motion,
        )

        await _finalize_video_done(video_id, dest, key, status="assembling")
        logger.info("Layer cut %s rendered (%d slots)", video_id, len(plan.slots))

    except Exception as exc:
        logger.exception("Layer cut %s failed", video_id)
        await _finalize_video_failure(video_id, exc, key)
        return

    _set_progress(key, "muxing", "Song wird untergelegt…", 97)
    await _run_soundtrack_mux(video_id, song_id, bed_volume)
    await _finish_beat_cut(video_id, key)
