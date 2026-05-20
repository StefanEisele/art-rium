"""
Rich-article background-job tracker.

The /article/generate-rich endpoint takes 6+ minutes end-to-end (image
upload + VLM + LLM). Mobile browsers drop the connection on screen lock,
so we run the work in a background asyncio task and let the frontend
poll for status. State is in-memory only — survives page reloads within
the same server process, but a server restart wipes it. Final results
(WordPress posts, Article rows) are persisted in the DB regardless.

Lifecycle:
  queued       — job submitted, asyncio.create_task() not yet picked up
  uploading    — running upload_image_to_wp() for each not-yet-uploaded image
  generating   — calling write_rich_article() and pushing posts to WordPress
  done         — completed; result has the {translation_group_id, articles[…]} payload
  failed       — error; error field has the exception string

Each job dict (returned to the frontend) shape:
  {
    "id":            str (uuid),
    "status":        "queued" | "uploading" | "generating" | "done" | "failed",
    "phase":         human-readable phase label
    "message":       short status line for the UI
    "created_at":    ISO timestamp,
    "completed_at":  ISO timestamp | null,
    "params":        original request body (image_ids, series_name, …)
    "result":        the generate_rich_articles_for_series return dict | null
    "error":         exception string | null
  }
"""
import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# Tunable: how many recent jobs to keep in memory. Older ones get evicted.
_MAX_JOBS = 50

# In-memory store. Key = job_id (str).
_jobs: dict[str, dict[str, Any]] = {}
_lock = asyncio.Lock()


async def create_job(params: dict[str, Any]) -> str:
    """Register a new job and return its id. Caller is responsible for
    spawning the runner task."""
    job_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    async with _lock:
        _jobs[job_id] = {
            "id":           job_id,
            "status":       "queued",
            "phase":        "queued",
            "message":      "Queued",
            "created_at":   now,
            "completed_at": None,
            "params":       params,
            "result":       None,
            "error":        None,
        }
        # Evict oldest if we're over the cap.
        if len(_jobs) > _MAX_JOBS:
            ordered = sorted(_jobs.items(), key=lambda kv: kv[1]["created_at"])
            for jid, _ in ordered[: len(_jobs) - _MAX_JOBS]:
                _jobs.pop(jid, None)
    logger.info("Article job %s queued (params: %s)", job_id, _summarize(params))
    return job_id


async def update_job(job_id: str, **fields: Any) -> None:
    """Patch a job dict with new fields. Silently no-ops if the job is gone."""
    async with _lock:
        if job_id in _jobs:
            _jobs[job_id].update(fields)


async def mark_done(job_id: str, result: dict[str, Any]) -> None:
    await update_job(
        job_id,
        status="done",
        phase="done",
        message=f"Generated {len(result.get('articles', []))} post(s)",
        completed_at=datetime.now(timezone.utc).isoformat(),
        result=result,
    )
    logger.info("Article job %s done — %d post(s)", job_id, len(result.get("articles", [])))


async def mark_failed(job_id: str, error: str | Exception) -> None:
    msg = str(error)
    await update_job(
        job_id,
        status="failed",
        phase="failed",
        message=msg[:200],
        completed_at=datetime.now(timezone.utc).isoformat(),
        error=msg,
    )
    logger.error("Article job %s failed: %s", job_id, msg)


def get_job(job_id: str) -> dict[str, Any] | None:
    """Synchronous read — dict access is atomic enough for the polling endpoint."""
    return _jobs.get(job_id)


def list_jobs(limit: int = 20) -> list[dict[str, Any]]:
    """Return the most recent jobs, newest first."""
    return sorted(_jobs.values(), key=lambda j: j["created_at"], reverse=True)[:limit]


def _summarize(params: dict[str, Any]) -> str:
    n = len(params.get("image_ids") or [])
    series = params.get("series_name") or "(none)"
    mode = params.get("mode") or "?"
    langs = params.get("languages") or ["en", "de"]
    return f"mode={mode}, {n} image(s), series={series}, langs={','.join(langs)}"
