"""
Fire-and-forget background tasks.

Plain `asyncio.create_task()` has two failure modes that matter for the
generation/publish jobs spawned throughout the routers: (1) if nothing keeps
a reference to the returned Task, the event loop is free to garbage-collect
it mid-run; (2) an unhandled exception inside the task never surfaces
anywhere — it just sits on the Task object until something calls
`.result()`, which for these jobs never happens. Both mean a job can die
silently with no log line, leaving its DB row stuck in "processing" forever.

`safe_create_task()` fixes both: it holds a strong reference in a
module-level set until the task finishes, and its done-callback logs any
exception that isn't a plain CancelledError.

That same set is what makes a job *stoppable*. Task names follow one
convention — `"<label>:<job id>"`, e.g. `"video_generation:<uuid>"` — so
`tasks_for(job_id)` can find every task belonging to a job the user just
deleted or aborted without any router keeping its own registry. Keep the
convention when adding a task: a name with no colon is read as
infrastructure (the listener, the schedulers, the warm-up) and is never
touched by a cancel-all.
"""
import asyncio
import logging
from typing import Coroutine

logger = logging.getLogger(__name__)

_background_tasks: set[asyncio.Task] = set()


def safe_create_task(coro: Coroutine, *, name: str | None = None) -> asyncio.Task:
    task = asyncio.create_task(coro, name=name)
    _background_tasks.add(task)
    task.add_done_callback(_on_task_done)
    return task


def _on_task_done(task: asyncio.Task) -> None:
    _background_tasks.discard(task)
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error(f"Background task {task.get_name()!r} failed", exc_info=exc)


# ── Cancellation ──────────────────────────────────────────────────────────────

def tasks_for(job_id: str) -> list[asyncio.Task]:
    """Every live task named `<label>:<job_id>`.

    Matching the id rather than the label on purpose: one job can be carried by
    several tasks over its life (a video generation, then an upscale, then a
    look pass), and "stop this job" means all of them.
    """
    suffix = f":{job_id}"
    return [t for t in _background_tasks if not t.done() and t.get_name().endswith(suffix)]


def job_tasks() -> list[asyncio.Task]:
    """Every live *job* task — i.e. everything except the infrastructure ones.

    The listener, the two schedulers and the titler warm-up are named without a
    colon and are excluded: cancelling them would not stop a render, it would
    stop the server from noticing that one finished.
    """
    return [t for t in _background_tasks if not t.done() and ":" in t.get_name()]


async def cancel(tasks: list[asyncio.Task], *, timeout: float = 10.0) -> list[str]:
    """Cancel these tasks and wait for them to actually unwind.

    The wait is the point. `Task.cancel()` only schedules the CancelledError
    for the next await, so a caller that deletes files immediately after would
    race a job still writing segments into the directory it is removing. Ten
    seconds is generous for a render loop whose await points are an HTTP poll
    and a subprocess wait; a task still alive after that is left to finish
    unwinding on its own rather than blocking the request further.
    """
    if not tasks:
        return []
    names = [t.get_name() for t in tasks]
    for t in tasks:
        t.cancel()
    _, pending = await asyncio.wait(tasks, timeout=timeout)
    if pending:
        logger.warning(
            "Cancelled task(s) still unwinding after %.0fs: %s",
            timeout, ", ".join(sorted(t.get_name() for t in pending)),
        )
    logger.info("Cancelled %d background task(s): %s", len(names), ", ".join(names))
    return names
