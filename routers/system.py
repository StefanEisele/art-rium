"""
Machine-level controls: power, and the render queue as a whole.

Power — the Pi outpost calls POST /api/system/shutdown with X-API-Key when the
user hits ig.stefaneisele.com/pc/shutdown. We return 202 immediately and then
fire the Windows shutdown command from a background task so the HTTP response
gets back to the Pi before the network stack goes down.

Queue — the two escape hatches the dashboard offers when a render session has
gone wrong. Both are deliberately here rather than in the video tool: they cut
across every tool at once, and the point of reaching for them is that you no
longer trust the tool you were in.

GET  /api/system/comfy        → is ComfyUI up, and what is in its queue
POST /api/system/comfy/restart→ kill ComfyUI and start it again
POST /api/system/jobs/cancel  → cancel every running job + empty the queue
"""
from __future__ import annotations

import asyncio
import logging
import subprocess
import sys
from typing import Literal

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel, Field

from core.auth import require_auth
from core.config import settings
from core.job_control import cancel_all_jobs
from core.tasks import job_tasks
from routers.video import forget_progress
from services.comfy import control as comfy_control

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/system", dependencies=[Depends(require_auth)])


class ShutdownRequest(BaseModel):
    mode: Literal["shutdown", "hibernate", "sleep"] = "shutdown"
    delay_seconds: int = Field(default=5, ge=0, le=300)


@router.get("/status")
def status():
    return {"ok": True, "platform": sys.platform}


# ── Render queue ──────────────────────────────────────────────────────────────

@router.get("/comfy")
async def comfy_status():
    """What the dashboard's queue strip shows: ComfyUI's queue and our own.

    `jobs` counts art-rium's live background tasks, which is not the same
    number as ComfyUI's queue and is worth showing beside it — a job doing an
    ffmpeg pass is busy while ComfyUI is idle, and a queue with entries and no
    jobs behind it is exactly the leftover state the cancel button is for.
    """
    q = await comfy_control.queue_state()
    return {
        "reachable": q["reachable"],
        "host": settings.comfyui_host,
        "running": len(q["running"]),
        "pending": len(q["pending"]),
        "jobs": len(job_tasks()),
    }


@router.post("/jobs/cancel")
async def cancel_all():
    """Stop every running job and empty ComfyUI's queue.

    Cloud renders are left alone on purpose — a MiniMax task is already paid
    for and keeps running on their side whatever we do here, so cancelling it
    locally would only lose the clip. See core/startup_sweep.py.
    """
    result = await cancel_all_jobs()
    result["progress_cleared"] = forget_progress()
    return result


@router.post("/comfy/restart", status_code=202)
async def restart_comfy(cancel_jobs: bool = True):
    """Kill ComfyUI and start it again from scripts/start-comfy.bat.

    `cancel_jobs` defaults to true and should stay that way: every job waiting
    on a prompt is dead the moment the process is killed, and without the
    cancel each one would poll a corpse for 90 seconds before failing with
    "ComfyUI unreachable" instead of saying what actually happened.
    """
    cancelled = await cancel_all_jobs() if cancel_jobs else None
    if cancel_jobs:
        forget_progress()
    try:
        result = await comfy_control.restart()
    except RuntimeError as exc:
        raise HTTPException(status_code=501, detail=str(exc))
    return {**result, "cancelled": cancelled}


@router.post("/shutdown", status_code=202)
def shutdown(req: ShutdownRequest, bg: BackgroundTasks):
    if sys.platform != "win32":
        raise HTTPException(501, f"Shutdown not implemented for platform {sys.platform!r}")

    cmd = _build_cmd(req.mode, req.delay_seconds)
    logger.warning("system.shutdown requested mode=%s delay=%ds → %s",
                   req.mode, req.delay_seconds, cmd)

    bg.add_task(_run_shutdown, cmd, req.delay_seconds)
    return {
        "accepted": True,
        "mode": req.mode,
        "delay_seconds": req.delay_seconds,
        "command": " ".join(cmd),
    }


@router.post("/shutdown/abort")
def shutdown_abort():
    """Cancel an in-flight `shutdown /s /t N` while the timer is still running."""
    if sys.platform != "win32":
        raise HTTPException(501, "Not implemented for this platform")
    try:
        subprocess.run(["shutdown", "/a"], check=True, capture_output=True, text=True)
        return {"aborted": True}
    except subprocess.CalledProcessError as exc:
        # exit 1116 = no shutdown in progress
        return {"aborted": False, "stderr": exc.stderr.strip()}


def _build_cmd(mode: str, delay: int) -> list[str]:
    if mode == "hibernate":
        return ["shutdown", "/h"]
    if mode == "sleep":
        # Windows has no first-class CLI for S3 sleep; rundll32 is the standard idiom.
        # Note: only works reliably when hibernation is OFF (powercfg -h off), otherwise
        # this hibernates instead. User chose S5 as the default anyway.
        return ["rundll32.exe", "powrprof.dll,SetSuspendState", "0,1,0"]
    # shutdown — /f forces apps to close so leftover `cmd /k` windows from
    # start-remote.bat (ComfyUI, art-rium server) don't block the shutdown.
    return ["shutdown", "/s", "/f", "/t", str(delay)]


async def _run_shutdown(cmd: list[str], delay: int) -> None:
    # tiny sleep so the 202 response gets flushed back to the Pi before the
    # `shutdown` command (which itself has a delay) starts ticking. This is
    # belt-and-suspenders — the /t delay alone is enough in practice.
    await asyncio.sleep(0.5)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            logger.error("shutdown cmd failed rc=%d stderr=%s",
                         result.returncode, result.stderr.strip())
        else:
            logger.info("shutdown cmd issued, system going down in ~%ds", delay)
    except Exception:
        logger.exception("shutdown cmd raised")
