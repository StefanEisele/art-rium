"""
Stopping a job for real.

A job is three things at once: a DB row, a background asyncio task, and
whatever that task has handed to somebody else — a ComfyUI prompt, an ffmpeg
child. Deleting the row used to stop only the first, which is why a deleted
video went on occupying the GPU for its full six minutes and then wrote its
segments into a directory that no longer existed.

`cancel_job()` stops all three, in the order that matters:

  1. the task, and *wait* for it to unwind (core.tasks.cancel), so nothing is
     still writing files when the caller starts deleting them. The task's own
     `await` on ffmpeg takes the child down with it — see core/subproc.py.
  2. then the ComfyUI prompt. Task first, because a prompt yanked out from
     under a task that is still polling surfaces as "ComfyUI job failed", and
     a job the user cancelled should not report a failure at all.

Callers pass the prompt ids they know about; this module deliberately knows
nothing about which model carries a `comfy_prompt_id`, so a router can hand it
the one on the row plus the one in its live progress entry without core/
growing an import of every table.
"""
from __future__ import annotations

import logging
import uuid
from typing import Iterable

from core import tasks
from core.startup_sweep import sweep_stuck_jobs
from services.comfy import control as comfy_control

logger = logging.getLogger(__name__)

CANCELLED_MSG = "Abgebrochen"


async def cancel_job(
    job_id: str | uuid.UUID, *, prompt_ids: Iterable[str | None] = (),
) -> dict:
    """Stop everything running for one job. Safe to call for an idle job."""
    names = await tasks.cancel(tasks.tasks_for(str(job_id)))
    dropped = await comfy_control.drop(prompt_ids)
    if names or dropped:
        logger.info("Cancelled job %s — %d task(s), %d ComfyUI prompt(s)",
                    job_id, len(names), dropped)
    return {"tasks": names, "prompts_dropped": dropped}


async def cancel_all_jobs() -> dict:
    """Stop every running job and empty ComfyUI's queue.

    The DB sweep afterwards is the same one that runs at startup, and it is
    correct here for the same reason: once every job task has been cancelled,
    a row still claiming to be in flight is by definition an orphan. It leaves
    cloud renders alone (they are billed and keep running at MiniMax) and
    leaves status='review' alone (an AnimateLCM job waiting on the user).
    """
    names = await tasks.cancel(tasks.job_tasks())
    comfy = await comfy_control.clear_queue()
    swept = await sweep_stuck_jobs(reason=CANCELLED_MSG)
    logger.warning(
        "Cancel-all: %d task(s), %d queued prompt(s) dropped, %d row(s) marked failed",
        len(names), comfy.get("cleared", 0), swept,
    )
    return {
        "tasks_cancelled": names,
        "comfy": comfy,
        "jobs_marked_failed": swept,
    }
