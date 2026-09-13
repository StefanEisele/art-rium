"""Drive `scripts/sam3_segment.py` from art-rium, and report what it is doing.

The worker runs in ComfyUI's venv rather than this one — see that file's
docstring for why — so this is a subprocess boundary, and everything crossing
it is JSON: a spec file in, one JSON object per stdout line back.

The card is the reason this is more than `subprocess.run`. SAM 3 wants about
5 GB in fp16 and it competes with exactly the things art-rium is otherwise
doing: a ComfyUI render holds its weights after finishing, and Ollama's titler
sits resident on a 30-minute keep-alive. Both are asked to let go first, in the
same order and by the same helpers a ComfyUI submission uses.
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path

from core.config import settings
from core.subproc import kill
from services.segment.plan import FrameBudget

logger = logging.getLogger(__name__)

# fp16 weights are ~3.4 GB; the session, the 1008x1008 working tensors and the
# per-frame masks account for the rest. Measured peak sits under 6 GB.
SEGMENT_VRAM = 6.0e9

# The worker prints one line per event; anything else on stdout is a library
# being chatty and is logged rather than parsed.
ProgressFn = Callable[[dict], Awaitable[None] | None]


class SegmentError(RuntimeError):
    """The worker failed, with the reason it gave."""


def build_spec(
    source: Path,
    dest: Path,
    preview: Path | None,
    concepts: list[dict],
    budget: FrameBudget | None = None,
    score_threshold: float | None = None,
) -> dict:
    """The JSON the worker reads. Kept in one place because the two ends of it
    live in different interpreters and nothing type-checks across that gap."""
    spec: dict = {
        "source": str(source),
        "dest": str(dest),
        "model_dir": str(settings.sam3_model_dir),
        "concepts": concepts,
        "device": settings.sam3_device,
    }
    if preview:
        spec["preview"] = str(preview)
    if score_threshold is not None:
        spec["score_threshold"] = float(score_threshold)
    if budget:
        spec.update({
            "start": budget.start,
            "seconds": budget.seconds,
            "stride": budget.stride,
            "max_frames": budget.kept,
            "fps": budget.fps,
        })
    return spec


async def _free_the_card() -> None:
    """Best-effort eviction. Segmentation does not need ComfyUI, so ComfyUI
    being down is not a reason to refuse to segment."""
    try:
        from services.comfy.vram import free_vram_for
        await free_vram_for(SEGMENT_VRAM, "SAM 3 segmentation")
    except Exception as exc:                        # noqa: BLE001
        logger.info("VRAM pre-flight skipped for segmentation: %s", exc)


async def run_segmentation(
    spec: dict,
    on_progress: ProgressFn | None = None,
    timeout: float = 3600.0,
) -> dict:
    """Segment one clip. Returns the worker's `done` payload.

    Raises `SegmentError` with the worker's own message on failure — the
    interesting ones are a concept nobody could find and a CUDA OOM, and both
    are worth showing verbatim rather than as "segmentation failed".
    """
    await _free_the_card()

    scratch = settings.storage_dir / "control" / "_segment"
    scratch.mkdir(parents=True, exist_ok=True)
    spec_path = scratch / f"{uuid.uuid4().hex}.json"
    spec_path.write_text(json.dumps(spec, indent=1), encoding="utf-8")

    cmd = [str(settings.sam3_python), str(settings.sam3_script), str(spec_path)]
    logger.info("Segmenting %s -> %s (%d concepts)",
                spec["source"], spec["dest"], len(spec["concepts"]))

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(Path(settings.sam3_script).parent.parent),
    )

    result: dict | None = None
    failure: str | None = None

    async def pump() -> None:
        nonlocal result, failure
        assert proc.stdout is not None
        async for raw in proc.stdout:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                logger.debug("segment worker: %s", line[:200])
                continue
            kind = event.get("event")
            if kind == "done":
                result = event
            elif kind == "error":
                failure = event.get("message") or "unknown error"
            if on_progress:
                out = on_progress(event)
                if asyncio.iscoroutine(out):
                    await out

    try:
        await asyncio.wait_for(
            asyncio.gather(pump(), proc.wait()), timeout=timeout,
        )
    except asyncio.TimeoutError:
        proc.kill()
        raise SegmentError(
            f"Segmentierung hat das Zeitlimit von {timeout / 60:.0f} min überschritten"
        ) from None
    except asyncio.CancelledError:
        # The job was deleted or cancelled. The worker holds the GPU for the
        # whole run and would keep holding it — see core/subproc.py, which
        # makes the same guarantee for the ffmpeg passes.
        kill(proc)
        raise

    stderr = (await proc.stderr.read()).decode("utf-8", errors="replace") if proc.stderr else ""

    if failure:
        raise SegmentError(failure)
    if proc.returncode != 0 or result is None:
        tail = "\n".join(stderr.strip().splitlines()[-6:])
        raise SegmentError(tail or f"Segmentierung endete mit Code {proc.returncode}")

    spec_path.unlink(missing_ok=True)
    return result
