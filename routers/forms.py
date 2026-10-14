"""
Formen API — a loop drawn in the browser becomes a Struktur control track.

The geometry is generated client-side: a WebGL raymarcher draws signed-distance
scenes whose every motion is periodic in the loop phase, so the last frame
leads back into the first by construction (frontends/tools/forms/). The server's
part is small and has to be exact:

  - take the frames as PNGs, one per loop step, depth and optionally a mask
    drawn from the same scene in the same pass of the same frame,
  - refuse anything AnimateLCM would misread (wrong count, mixed sizes, a mask
    that anti-aliased its edges into colours that key into nothing),
  - encode both at the render's own rate (services/forms/plan.py) and register
    them as ordinary control tracks, the mask linked to its depth by
    `source_track_id` so the Struktur tool's pairing check passes on lineage,
  - keep the generator recipe beside the depth track, so a loop can be opened
    in the editor again and varied rather than rebuilt.

The tracks then live in the same library as Blender passes and SAM 3 masks;
nothing downstream needs to know they were generated.
"""
import asyncio
import json
import logging
import shutil
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from PIL import Image as PILImage
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.auth import require_auth
from core.config import settings
from core.db import get_db
from core.models import ControlTrack
from core.subproc import communicate
from routers.vace import _register_track, _serialize_track, _track_rel
from services.forms import (
    DEFAULT_FORMAT,
    DEFAULT_SECONDS,
    DURATIONS,
    GROUP_LABELS,
    LOOP_FPS,
    MAX_FRAMES,
    MIN_FRAMES,
    encode_command,
    format_table,
    loop_frames,
    mask_regions,
)
from services.segment import REGION_COLORS

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/forms", dependencies=[Depends(require_auth)])

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_MAX_FRAME_BYTES = 16 * 1024 * 1024
_MAX_RECIPE_BYTES = 64 * 1024
_RECIPE_VERSION = 1
# Pixels in a mask that key into no region. The page draws its mask without
# anti-aliasing, so the honest number is zero; a little slack covers a codec
# that touched the PNGs on the way, and anything beyond it is a page bug that
# would otherwise surface as holes in a render twenty minutes later.
_MAX_STRAY = 0.005


def recipe_path(track_id: uuid.UUID) -> Path:
    return settings.storage_dir / _track_rel(track_id, "_recipe.json")


@router.get("/options")
async def forms_options():
    """Formats, loop lengths and mask colours — server-side so the page and
    the render path cannot disagree about what a "6 s 9:16 loop" is."""
    return {
        "fps": LOOP_FPS,
        "durations": [{"seconds": s, "frames": loop_frames(s)} for s in DURATIONS],
        "default_seconds": DEFAULT_SECONDS,
        "formats": format_table(),
        "default_format": DEFAULT_FORMAT,
        "colors": [list(c) for c in REGION_COLORS],
        "group_labels": list(GROUP_LABELS),
        "min_frames": MIN_FRAMES,
        "max_frames": MAX_FRAMES,
    }


async def _save_frames(files: list[UploadFile], folder: Path, stem: str) -> list[Path]:
    paths = []
    for i, upload in enumerate(files):
        data = await upload.read(_MAX_FRAME_BYTES + 1)
        if len(data) > _MAX_FRAME_BYTES:
            raise HTTPException(status_code=413, detail=f"Frame {i} ist zu groß")
        if not data.startswith(_PNG_MAGIC):
            raise HTTPException(status_code=400, detail=f"Frame {i} ist kein PNG")
        path = folder / f"{stem}_{i:04d}.png"
        path.write_bytes(data)
        paths.append(path)
    return paths


def _frame_size(paths: list[Path]) -> tuple[int, int]:
    sizes = set()
    for path in paths:
        try:
            with PILImage.open(path) as im:
                sizes.add(im.size)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"{path.name} ist unlesbar") from exc
    if len(sizes) != 1:
        raise HTTPException(status_code=400, detail="Die Frames haben unterschiedliche Größen")
    width, height = sizes.pop()
    # 4:2:0 needs both edges even; ffmpeg would refuse later with a less
    # useful message.
    if width % 2 or height % 2:
        raise HTTPException(status_code=400, detail="Breite und Höhe müssen gerade sein")
    return width, height


def _parse_labels(raw: str) -> list[str]:
    try:
        data = json.loads(raw or "[]")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="labels ist kein JSON") from exc
    if not isinstance(data, list):
        raise HTTPException(status_code=400, detail="labels muss eine Liste sein")
    return [str(x).strip()[:40] or GROUP_LABELS[i]
            for i, x in enumerate(data[:len(REGION_COLORS)])]


def _parse_recipe(raw: str) -> Optional[dict]:
    if not raw:
        return None
    if len(raw.encode()) > _MAX_RECIPE_BYTES:
        raise HTTPException(status_code=413, detail="Rezept ist zu groß")
    try:
        data = json.loads(raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Rezept ist kein JSON") from exc
    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail="Rezept muss ein Objekt sein")
    return data


async def _encode(pattern: Path, dest: Path, frames: int, lossless: bool) -> None:
    proc = await asyncio.create_subprocess_exec(
        *encode_command(pattern, dest, frames, lossless),
        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
    )
    _, err = await communicate(proc)
    if proc.returncode != 0 or not dest.is_file():
        raise RuntimeError(err.decode(errors="replace")[:400] or "ffmpeg failed")


@router.post("/loops", status_code=201)
async def save_loop(
    depth: list[UploadFile] = File(...),
    mask: Optional[list[UploadFile]] = File(None),
    title: str = Form(""),
    recipe: str = Form(""),
    labels: str = Form("[]"),
    db: AsyncSession = Depends(get_db),
):
    """Store one loop: N depth frames, optionally N mask frames of the same
    scene. Returns both tracks and the Struktur link that opens them."""
    mask = mask or []
    count = len(depth)
    if not MIN_FRAMES <= count <= MAX_FRAMES:
        raise HTTPException(
            status_code=400,
            detail=f"Ein Loop hat {MIN_FRAMES}–{MAX_FRAMES} Frames, nicht {count}",
        )
    if mask and len(mask) != count:
        raise HTTPException(
            status_code=400,
            detail=f"Maske ({len(mask)}) und Tiefe ({count}) brauchen gleich viele Frames",
        )
    recipe_data = _parse_recipe(recipe)
    label_list = _parse_labels(labels)

    depth_id = uuid.uuid4()
    mask_id = uuid.uuid4() if mask else None
    depth_rel = _track_rel(depth_id, ".mp4")
    mask_rel = _track_rel(mask_id, ".mp4") if mask_id else None
    depth_path = settings.storage_dir / depth_rel
    mask_path = settings.storage_dir / mask_rel if mask_rel else None

    settings.control_dir.mkdir(parents=True, exist_ok=True)
    work = settings.control_dir / f".forms_{depth_id.hex}"
    work.mkdir(parents=True, exist_ok=True)
    regions = None
    try:
        depth_frames = await _save_frames(depth, work, "d")
        size = _frame_size(depth_frames)
        if mask:
            mask_frames = await _save_frames(mask, work, "m")
            if _frame_size(mask_frames) != size:
                raise HTTPException(status_code=400,
                                    detail="Maske und Tiefe brauchen dieselbe Größe")
            regions, stray = await asyncio.to_thread(mask_regions, mask_frames, label_list)
            if stray > _MAX_STRAY:
                raise HTTPException(
                    status_code=400,
                    detail=f"Die Maske ist nicht sauber: {stray:.1%} der Pixel passen "
                           f"zu keiner Regionsfarbe und würden nicht gekeyt.",
                )
            if stray:
                logger.warning("Forms mask %s: %.3f%% stray pixels", mask_id, stray * 100)
        await _encode(work / "d_%04d.png", depth_path, count, lossless=False)
        if mask_path:
            await _encode(work / "m_%04d.png", mask_path, count, lossless=True)
    except Exception as exc:
        for path in filter(None, (depth_path, mask_path)):
            path.unlink(missing_ok=True)
        if isinstance(exc, HTTPException):
            raise
        logger.exception("Forms loop %s failed to encode", depth_id)
        raise HTTPException(status_code=500, detail=f"Kodieren fehlgeschlagen: {exc}") from exc
    finally:
        shutil.rmtree(work, ignore_errors=True)

    name = (title or "").strip()[:200] or "Formen-Loop"
    stem = f"forms_{depth_id.hex[:8]}"
    track = await _register_track(db, depth_id, depth_path, depth_rel, "depth",
                                  f"{stem}.mp4", name)
    mask_track = None
    if mask_id and mask_path and mask_rel:
        mask_track = await _register_track(
            db, mask_id, mask_path, mask_rel, "mask", f"{stem}_mask.mp4",
            f"{name} · Maske", source_track_id=depth_id, regions=regions,
        )
    if recipe_data is not None:
        recipe_path(depth_id).write_text(json.dumps({
            "version": _RECIPE_VERSION,
            "recipe": recipe_data,
            "mask_track_id": str(mask_id) if mask_id else None,
        }), encoding="utf-8")

    logger.info("Forms loop %s stored — %d frames %sx%s%s", depth_id, count, *size,
                f", mask {mask_id}" if mask_id else "")
    return {
        "depth": _serialize_track(track),
        "mask": _serialize_track(mask_track) if mask_track else None,
        "struktur_url": f"/tools/vace/?track={depth_id}"
                        + (f"&mask={mask_id}" if mask_id else ""),
    }


@router.get("/loops")
async def list_loops(db: AsyncSession = Depends(get_db)):
    """Loops made here, newest first, each with the recipe that draws it
    again. A depth track from anywhere else has no recipe and is left to the
    Struktur library."""
    tracks = (await db.execute(
        select(ControlTrack).where(ControlTrack.kind == "depth")
        .order_by(desc(ControlTrack.created_at))
    )).scalars().all()
    out = []
    for track in tracks:
        path = recipe_path(track.id)
        if not path.is_file():
            continue
        try:
            sidecar = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            continue
        mask = None
        if sidecar.get("mask_track_id"):
            try:
                mask = await db.get(ControlTrack, uuid.UUID(sidecar["mask_track_id"]))
            except ValueError:
                mask = None
        out.append({
            **_serialize_track(track),
            "recipe": sidecar.get("recipe"),
            "mask": _serialize_track(mask) if mask else None,
        })
    return out
