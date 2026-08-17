"""Video backends — one interface, MiniMax behind it today.

The point of the interface is that the frontend never talks to a provider. When
a rented GPU turns up later, a `ComfyUIBackend` implements the same four calls
and nothing above this module changes.

Contract verified 2026-08-15 against
https://platform.minimax.io/docs/api-reference/video-generation-v2-create
and .../video-generation-v2-regeneration:

    POST https://api.minimax.io/v2/video_generation   → {"task_id": "…"}
    POST https://api.minimax.io/v2/video_regeneration → {"task_id": "…"}
    GET  https://api.minimax.io/v2/query/video_generation/{task_id}
         → {"task": {"status": "queued|running|succeeded|failed|cancelled",
                     "content": {"url": …}, "usage": {…}, …}}
    Authorization: Bearer <key>

Errors are typed rather than status-coded: the body carries
`{"error": {"type": "insufficient_balance_error", …}}`. `_error_of` turns that
into a message worth showing a human, because "HTTP 402" is not one.
"""
from __future__ import annotations

import base64
import logging
import mimetypes
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

import httpx

from core.config import settings
from services.video_api import pricing

logger = logging.getLogger(__name__)

MODEL = "MiniMax-H3"

# The docs ask for 10 s between polls. Ramp up to it rather than starting there,
# so a short clip that is already done is not held back by a fixed wait.
POLL_BACKOFF = (5, 10, 15)
POLL_INTERVAL_MAX = 15
# Past this a task is treated as lost and its reservation is released. Generous:
# a 15 s 2K clip can legitimately take several minutes plus queue time.
TASK_TIMEOUT_SECONDS = 20 * 60

Status = Literal["queued", "running", "succeeded", "failed", "cancelled"]
TERMINAL = ("succeeded", "failed", "cancelled")


class VideoBackendError(RuntimeError):
    """Provider said no. `retryable` separates a blip from a refusal."""

    def __init__(self, message: str, *, kind: str = "error", retryable: bool = False):
        super().__init__(message)
        self.kind = kind
        self.retryable = retryable


@dataclass(frozen=True, slots=True)
class VideoRequest:
    """One generation, in provider-neutral terms."""
    prompt: str
    duration_s: int = 6
    resolution: str = "2K"
    ratio: str = "adaptive"
    first_frame: Path | None = None
    last_frame: Path | None = None
    reference_images: list[Path] = field(default_factory=list)
    reference_videos: list[Path] = field(default_factory=list)
    reference_audio: list[Path] = field(default_factory=list)

    @property
    def image_count(self) -> int:
        """Every image counts towards the free-five allowance, frames included."""
        return (
            len(self.reference_images)
            + (1 if self.first_frame else 0)
            + (1 if self.last_frame else 0)
        )

    @property
    def mode(self) -> str:
        if self.reference_videos or self.reference_audio or self.reference_images:
            return "reference"
        if self.first_frame or self.last_frame:
            return "image"
        return "text"


@dataclass(frozen=True, slots=True)
class TaskState:
    status: Status
    output_url: str | None = None
    error: str | None = None
    billed_seconds: float | None = None
    billed_image_count: int | None = None
    billed_input_video_seconds: float | None = None

    @property
    def done(self) -> bool:
        return self.status in TERMINAL


class VideoBackend(Protocol):
    def estimate_cost(self, request: VideoRequest, **kw) -> dict: ...
    async def submit(self, request: VideoRequest) -> str: ...
    async def submit_regeneration(self, *, source_task_id: str) -> str: ...
    async def poll(self, task_id: str) -> TaskState: ...
    async def cancel(self, task_id: str) -> None: ...


# ── Validation ───────────────────────────────────────────────────────────────


def validate(request: VideoRequest) -> None:
    """Reject what the API would reject, before spending a request on finding out.

    Bounds are the documented ones. Checking here rather than letting the API
    refuse keeps a bad form from costing a round trip — and, more importantly,
    from being reserved against the budget first.
    """
    if not request.prompt or not request.prompt.strip():
        raise VideoBackendError("A prompt is required", kind="invalid_request")
    if len(request.prompt) > pricing.PROMPT_MAX_CHARS:
        raise VideoBackendError(
            f"Prompt is longer than {pricing.PROMPT_MAX_CHARS} characters",
            kind="invalid_request",
        )
    if not pricing.DURATION_MIN <= request.duration_s <= pricing.DURATION_MAX:
        raise VideoBackendError(
            f"Duration must be {pricing.DURATION_MIN}–{pricing.DURATION_MAX} seconds",
            kind="invalid_request",
        )
    if request.resolution not in pricing.RESOLUTIONS:
        raise VideoBackendError(
            f"Resolution must be one of {pricing.RESOLUTIONS}", kind="invalid_request",
        )
    if request.ratio not in pricing.RATIOS:
        raise VideoBackendError(f"Unknown ratio {request.ratio!r}", kind="invalid_request")
    if len(request.reference_images) > pricing.MAX_REFERENCE_IMAGES:
        raise VideoBackendError(
            f"At most {pricing.MAX_REFERENCE_IMAGES} reference images",
            kind="invalid_request",
        )
    if len(request.reference_videos) > pricing.MAX_REFERENCE_VIDEOS:
        raise VideoBackendError(
            f"At most {pricing.MAX_REFERENCE_VIDEOS} reference videos", kind="invalid_request",
        )
    if len(request.reference_audio) > pricing.MAX_REFERENCE_AUDIO:
        raise VideoBackendError(
            f"At most {pricing.MAX_REFERENCE_AUDIO} reference audio files",
            kind="invalid_request",
        )

    total = sum(
        p.stat().st_size
        for p in _all_files(request)
        if p.exists()
    )
    if total > pricing.MAX_REQUEST_BYTES:
        raise VideoBackendError(
            f"Attachments total {total // (1024 * 1024)} MB; the API caps a request at "
            f"{pricing.MAX_REQUEST_BYTES // (1024 * 1024)} MB",
            kind="invalid_request",
        )


def _all_files(request: VideoRequest) -> list[Path]:
    return [
        p for p in (
            request.first_frame, request.last_frame,
            *request.reference_images, *request.reference_videos, *request.reference_audio,
        ) if p is not None
    ]


def _data_uri(path: Path) -> str:
    """Inline a local file. The API takes public URLs, `mm_file://` ids or data
    URIs; art-rium's storage is behind an API key, so a public URL would mean
    handing MiniMax a share token. Inlining keeps the credentials at home."""
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def build_content(request: VideoRequest) -> list[dict[str, Any]]:
    """The `content` array. Exactly one text item is mandatory and goes first."""
    content: list[dict[str, Any]] = [{"type": "text", "text": request.prompt}]

    def image(path: Path, role: str) -> dict:
        return {"type": "image_url", "image_url": {"url": _data_uri(path)}, "role": role}

    if request.first_frame:
        content.append(image(request.first_frame, "first_frame"))
    if request.last_frame:
        content.append(image(request.last_frame, "last_frame"))
    for path in request.reference_images:
        content.append(image(path, "reference_image"))
    for path in request.reference_videos:
        content.append(
            {"type": "video_url", "video_url": {"url": _data_uri(path)}, "role": "reference_video"}
        )
    for path in request.reference_audio:
        content.append(
            {"type": "audio_url", "audio_url": {"url": _data_uri(path)}, "role": "reference_audio"}
        )
    return content


def build_payload(request: VideoRequest) -> dict[str, Any]:
    return {
        "model": MODEL,
        "content": build_content(request),
        "resolution": request.resolution,
        "duration": request.duration_s,
        "ratio": request.ratio,
    }


def _error_of(response: httpx.Response) -> VideoBackendError:
    """Turn the provider's typed error into something worth reading."""
    kind, message = "error", response.text[:300]
    try:
        body = response.json()
        err = body.get("error") or {}
        kind = err.get("type") or kind
        message = err.get("message") or message
    except Exception:
        pass

    friendly = {
        "authorized_error": "MiniMax rejected the API key.",
        "insufficient_balance_error": "The MiniMax account is out of credit.",
        "unprocessable_entity_error": "MiniMax refused the prompt or an attachment as sensitive content.",
        "rate_limit_error": "MiniMax is rate-limiting; the job stays queued.",
    }.get(kind)

    return VideoBackendError(
        f"{friendly} ({message})" if friendly else message,
        kind=kind,
        # 429 and 5xx are worth another go; a rejected key or prompt is not.
        retryable=kind in ("rate_limit_error", "server_error") or response.status_code >= 500,
    )


class MiniMaxBackend:
    """MiniMax H3 over the v2 REST API."""

    def __init__(self, *, api_key: str | None = None, base_url: str | None = None):
        self.api_key = api_key if api_key is not None else settings.minimax_api_key
        self.base_url = (base_url or settings.minimax_api_base).rstrip("/")

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _require_key(self) -> None:
        if not self.configured:
            raise VideoBackendError(
                "MINIMAX_API_KEY is not set — add it to .env to generate over the API.",
                kind="not_configured",
            )

    def estimate_cost(
        self,
        request: VideoRequest,
        *,
        usd_eur_rate: float = 1.08,
        safety_factor: float = 1.10,
        input_video_seconds: float = 0,
    ) -> dict:
        return pricing.price(
            kind=pricing.kind_for(request.resolution),
            duration_s=request.duration_s,
            image_count=request.image_count,
            input_video_seconds=input_video_seconds,
            usd_eur_rate=usd_eur_rate,
            safety_factor=safety_factor,
        ).as_dict()

    async def submit(self, request: VideoRequest) -> str:
        self._require_key()
        validate(request)
        payload = build_payload(request)
        # Long write timeout: the body carries base64 attachments, up to 64 MB.
        timeout = httpx.Timeout(connect=15.0, read=120.0, write=300.0, pool=15.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(
                f"{self.base_url}/v2/video_generation",
                headers=self._headers(), json=payload,
            )
        if r.status_code != 200:
            raise _error_of(r)
        task_id = r.json().get("task_id")
        if not task_id:
            raise VideoBackendError(f"No task_id in the response: {r.text[:200]}")
        logger.info(
            "MiniMax submit ok: task=%s %ss %s %s",
            task_id, request.duration_s, request.resolution, request.mode,
        )
        return task_id

    async def submit_regeneration(self, *, source_task_id: str) -> str:
        """Pull a finished 768P task up to 2K.

        Only works on tasks this account produced, within 7 days, and the docs
        note that this mode needs whitelist access — so a refusal here is
        expected rather than a bug, and the caller must release the reservation.
        """
        self._require_key()
        payload = {
            "model": MODEL,
            "source_task_id": source_task_id,
            "resolution": "2K",
        }
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(
                f"{self.base_url}/v2/video_regeneration",
                headers=self._headers(), json=payload,
            )
        if r.status_code != 200:
            raise _error_of(r)
        task_id = r.json().get("task_id")
        if not task_id:
            raise VideoBackendError(f"No task_id in the response: {r.text[:200]}")
        logger.info("MiniMax regeneration ok: source=%s task=%s", source_task_id, task_id)
        return task_id

    async def poll(self, task_id: str) -> TaskState:
        self._require_key()
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.get(
                f"{self.base_url}/v2/query/video_generation/{task_id}",
                headers=self._headers(),
            )
        if r.status_code != 200:
            raise _error_of(r)
        return parse_task(r.json())

    async def cancel(self, task_id: str) -> None:
        """MiniMax v2 exposes no cancel endpoint.

        Said out loud rather than swallowed: a submitted task runs to
        completion and *will* be billed. The UI therefore only offers cancel
        while a job is still in the local queue, where cancelling is free.
        """
        raise VideoBackendError(
            "A submitted MiniMax task cannot be cancelled — it will be billed.",
            kind="not_supported",
        )


def parse_task(body: dict) -> TaskState:
    """Read a query response into a TaskState.

    Kept module-level and pure so the status mapping can be tested against
    recorded payloads without touching the network.
    """
    task = body.get("task") or {}
    status = task.get("status") or "queued"
    usage = task.get("usage") or {}
    content = task.get("content") or {}

    if status not in ("queued", "running", "succeeded", "failed", "cancelled"):
        # Unknown state: treat as still running rather than inventing a
        # terminal answer that would settle or release money wrongly.
        logger.warning("Unknown MiniMax task status %r — treating as running", status)
        status = "running"

    error = None
    if status == "failed":
        error = task.get("error") or task.get("status_msg") or "MiniMax reported a failed task"

    return TaskState(
        status=status,  # type: ignore[arg-type]
        output_url=content.get("url"),
        error=error,
        billed_seconds=usage.get("output_seconds") or usage.get("total_seconds"),
        billed_image_count=usage.get("input_image_count"),
        billed_input_video_seconds=usage.get("input_seconds"),
    )


def poll_delay(attempt: int) -> int:
    """5 s, 10 s, 15 s, then 15 s forever — the docs ask for ~10 s."""
    if attempt < len(POLL_BACKOFF):
        return POLL_BACKOFF[attempt]
    return POLL_INTERVAL_MAX
