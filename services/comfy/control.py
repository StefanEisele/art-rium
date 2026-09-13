"""
Stopping ComfyUI, at three levels of severity.

art-rium submits a prompt and then waits on `/history`. Nothing in that
arrangement stops the render — cancelling the waiting task only stops the
*waiting*, and ComfyUI happily finishes a 6-minute Wan clip nobody will ever
look at, then starts the next one queued behind it. These are the three ways to
actually make it stop:

  `drop(prompt_ids)`  one job's prompts: removed from the pending queue, and
                      interrupted if one of them is the running one. ComfyUI's
                      `/interrupt` takes a prompt_id and no-ops when that
                      prompt is not the one executing, which is what makes this
                      safe to call for a single deleted job while other jobs
                      keep rendering.

  `clear_queue()`     everything pending, plus a global interrupt. The "Queue
                      leeren" button.

  `restart()`         kill the process and start it again from
                      scripts/start-comfy.bat. For when ComfyUI is wedged
                      rather than busy — a dead prompt_worker still answers
                      HTTP, so /interrupt reaches a server that will never act
                      on it (that script's --cuda-device note describes how
                      this box gets into exactly that state).

Every HTTP call here is best-effort: ComfyUI being unreachable is a perfectly
ordinary answer to "stop rendering", and must never turn a delete into a 500.
"""
from __future__ import annotations

import asyncio
import logging
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import httpx

from core.config import settings

logger = logging.getLogger(__name__)

_TIMEOUT = 8.0


async def queue_state() -> dict:
    """What ComfyUI is doing: running/pending prompt ids, or reachable=False."""
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            r = await client.get(f"http://{settings.comfyui_host}/queue")
            r.raise_for_status()
            q = r.json()
    except Exception as exc:
        logger.info("ComfyUI queue unreachable: %s: %s", type(exc).__name__, exc)
        return {"reachable": False, "running": [], "pending": []}
    # Queue entries are [number, prompt_id, prompt, extra_data, outputs].
    return {
        "reachable": True,
        "running": [i[1] for i in q.get("queue_running", []) if len(i) > 1],
        "pending": [i[1] for i in q.get("queue_pending", []) if len(i) > 1],
    }


async def drop(prompt_ids: Iterable[str | None]) -> int:
    """Remove these prompts from ComfyUI and interrupt whichever one is running.

    Returns how many ids were acted on. Unknown or already-finished ids cost
    one no-op request each and are not an error — a job's last known prompt is
    routinely one that finished seconds ago.
    """
    ids = [p for p in dict.fromkeys(prompt_ids) if p]
    if not ids:
        return 0
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            await client.post(f"http://{settings.comfyui_host}/queue", json={"delete": ids})
            # Targeted, not global: /interrupt with a prompt_id only fires if
            # that prompt is the one executing, so another job's render
            # survives this call.
            for pid in ids:
                await client.post(
                    f"http://{settings.comfyui_host}/interrupt", json={"prompt_id": pid}
                )
    except Exception as exc:
        logger.warning("ComfyUI drop failed (%s): %s", type(exc).__name__, exc)
        return 0
    logger.info("Dropped %d prompt(s) from ComfyUI: %s", len(ids), ", ".join(ids))
    return len(ids)


async def clear_queue() -> dict:
    """Wipe everything pending and interrupt what is running.

    Reports what was in the queue *before* the wipe, because afterwards there
    is nothing to count and "cleared 0" would read like a failure.
    """
    before = await queue_state()
    if not before["reachable"]:
        return {"reachable": False, "cleared": 0, "interrupted": 0}
    n_pending, n_running = len(before["pending"]), len(before["running"])
    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            await client.post(f"http://{settings.comfyui_host}/queue", json={"clear": True})
            # No prompt_id — a global interrupt, since the point is that
            # nothing should still be rendering when this returns.
            await client.post(f"http://{settings.comfyui_host}/interrupt", json={})
    except Exception as exc:
        logger.warning("ComfyUI clear_queue failed (%s): %s", type(exc).__name__, exc)
        return {"reachable": True, "cleared": 0, "interrupted": 0,
                "error": f"{type(exc).__name__}: {exc}"}
    logger.warning("ComfyUI queue cleared: %d pending dropped, %d running interrupted",
                   n_pending, n_running)
    return {"reachable": True, "cleared": n_pending, "interrupted": n_running}


# ── Process control ───────────────────────────────────────────────────────────

async def restart() -> dict:
    """Kill ComfyUI and start it again. Windows only.

    Killing by *port* rather than by image name is deliberate: `python.exe` on
    this box is also art-rium itself, and a name-based kill would take the
    server issuing the request down with it. Whatever listens on
    comfyui_host's port is ComfyUI by definition.
    """
    if sys.platform != "win32":
        raise RuntimeError(f"ComfyUI restart is not implemented for {sys.platform!r}")

    script = Path(settings.comfyui_start_script)
    if not script.is_file():
        raise RuntimeError(f"Launch script missing: {script}")

    killed = await asyncio.to_thread(_kill_comfy)
    # ComfyUI binds its port at startup; relaunching before the old process has
    # released it gives the new one a bind error and no server at all.
    await asyncio.sleep(2.0)
    await asyncio.to_thread(_launch, script)
    logger.warning("ComfyUI restart issued (killed %d process(es))", len(killed))
    return {"killed_pids": killed, "script": str(script)}


def _kill_comfy() -> list[int]:
    """Kill whatever is listening on the ComfyUI port, plus its console."""
    pids = _pids_on_port(_comfy_port())
    for pid in pids:
        # /T so the venv's python goes down with the cmd wrapper (and vice
        # versa); /F because a busy CUDA process ignores a polite close.
        _run(["taskkill", "/PID", str(pid), "/T", "/F"])
    # The `cmd /k` console from start-comfy.bat outlives its python child and
    # would leave a dead window behind on every restart. Best-effort, and
    # harmless when ComfyUI was started some other way: nothing matches.
    _run(["taskkill", "/FI", "WINDOWTITLE eq ComfyUI", "/T", "/F"])
    return pids


def _launch(script: Path) -> None:
    # The script's own `start` detaches ComfyUI into its own visible console;
    # this wrapper only has to run the script and exit, so it stays hidden.
    subprocess.Popen(
        ["cmd", "/c", str(script)],
        cwd=str(script.parent.parent),
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _comfy_port() -> str:
    _, _, port = settings.comfyui_host.rpartition(":")
    return port or "8188"


def _pids_on_port(port: str) -> list[int]:
    """PIDs listening on `port`, via netstat — no third-party dependency.

    Two things this deliberately does not do.

    It does not read the state column. netstat is localised: on this German
    box that column says "ABHÖREN", not "LISTENING", and a check against the
    English word silently matched nothing — which would have made the restart
    button kill zero processes and then start a *second* ComfyUI on top of the
    first. A listening socket is identified structurally instead, by its
    wildcard foreign address (`0.0.0.0:0` / `[::]:0`), which no locale
    translates.

    It does not match the port anywhere in the line. A client connected *to*
    ComfyUI — this very server — carries that port in its foreign-address
    column, so a looser match would put art-rium's own pid on the kill list.
    """
    out = _run(["netstat", "-ano", "-p", "TCP"])
    pids: list[int] = []
    for line in out.splitlines():
        parts = line.split()
        # proto  local-address  foreign-address  state  pid
        if len(parts) < 5 or parts[0].upper() != "TCP":
            continue
        local, foreign = parts[1], parts[2]
        if not local.endswith(f":{port}") or not foreign.endswith(":0"):
            continue
        try:
            pid = int(parts[-1])
        except ValueError:
            continue
        if pid and pid not in pids:
            pids.append(pid)
    return pids


def _run(cmd: list[str]) -> str:
    """Run a Windows console tool, returning stdout. Never raises.

    Decoding is done here rather than by `text=True`, and that is not a style
    choice. These tools write in the console's OEM codepage — cp850 on this
    German box — while `text=True` decodes with the locale's ANSI one, cp1252.
    They disagree on exactly the bytes German uses: taskkill's "konnte nicht
    beendet werden" carries 0x81, which cp1252 has no character for, and the
    UnicodeDecodeError is raised on subprocess's reader *thread*, where it
    cannot be caught here — it prints a traceback and hands back empty output.
    An empty netstat means no pids, which means the restart kills nothing and
    then starts a second ComfyUI beside the first. Only the ASCII structure of
    these outputs is ever parsed, so replacement characters cost nothing.

    taskkill also exits non-zero for "no such process" and netstat can be
    missing on a stripped image; neither is worth failing a restart over.
    """
    try:
        out = subprocess.run(cmd, capture_output=True, timeout=20).stdout
    except Exception as exc:
        logger.warning("%s failed (%s): %s", cmd[0], type(exc).__name__, exc)
        return ""
    return out.decode("utf-8", errors="replace")
