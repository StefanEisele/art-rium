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
