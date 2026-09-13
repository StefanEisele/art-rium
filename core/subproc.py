"""
Cancellation-aware child-process waiting.

`await proc.communicate()` is not cancellation-safe: when the awaiting task is
cancelled, the coroutine unwinds but the child keeps running. For this project
that child is an ffmpeg encode or the SAM 3 worker — minutes of CPU or a whole
GPU — so a deleted job would go on burning the machine with nothing left to
report its result to. Every long-running subprocess in a background job waits
through `communicate()` here instead, which takes the child down with it.

Short-lived probes (ffprobe in core/video_thumb.py) deliberately do not: they
finish in milliseconds, and a kill path there would be code without a case.
"""
import asyncio
import logging

logger = logging.getLogger(__name__)


async def communicate(proc: asyncio.subprocess.Process) -> tuple[bytes, bytes]:
    """`proc.communicate()` that kills the child if this task is cancelled."""
    try:
        return await proc.communicate()
    except asyncio.CancelledError:
        kill(proc)
        raise


def kill(proc: asyncio.subprocess.Process) -> None:
    """Terminate a child if it is still alive. Never raises.

    Deliberately does not await `proc.wait()`. The one caller is unwinding a
    cancellation, where the next await is free to raise `CancelledError`
    straight back — reaping is left to the event loop's child watcher, which
    does it anyway.
    """
    if proc.returncode is not None:
        return
    try:
        proc.kill()
        logger.info("Killed child process %s (job cancelled)", proc.pid)
    except ProcessLookupError:
        pass
    except Exception as exc:            # pragma: no cover — platform edge cases
        logger.warning("Could not kill child process %s: %s", proc.pid, exc)
