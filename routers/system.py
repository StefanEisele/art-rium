"""
System power control — remote shutdown / hibernate of the desktop.

The Pi outpost calls POST /api/system/shutdown with X-API-Key when the user
hits ig.stefaneisele.com/pc/shutdown. We return 202 immediately and then
fire the Windows shutdown command from a background task so the HTTP
response gets back to the Pi before the network stack goes down.
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

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/system", dependencies=[Depends(require_auth)])


class ShutdownRequest(BaseModel):
    mode: Literal["shutdown", "hibernate", "sleep"] = "shutdown"
    delay_seconds: int = Field(default=5, ge=0, le=300)


@router.get("/status")
def status():
    return {"ok": True, "platform": sys.platform}


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
