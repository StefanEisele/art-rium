"""
ComfyUI WebSocket listener — event relay + active ingestion pipeline.

Responsibilities:
  1. Maintain a persistent WS connection to ComfyUI
  2. Route events to the correct frontend WebSocket client
  3. On execution_success: copy image to managed storage + write DB record
"""
import asyncio
import json
import logging
import uuid
from pathlib import Path
from typing import Any

import websockets
from fastapi import WebSocket

from core.comfy import WORKFLOW_NAME
from core.config import settings
from services.comfy.ingest import ingest_comfy_image
from services.comfy.node_labels import build_label_map

logger = logging.getLogger(__name__)

WS_PING_INTERVAL = 20
WS_PING_TIMEOUT = 20

# Label maps outlive their prompt only until it finishes, but a prompt that
# never reports (ComfyUI restarted mid-render) would leak one. Cap the dict and
# evict oldest-first — it is a display nicety, not state anything depends on.
MAX_LABEL_MAPS = 64


# The process's live listener. Request handlers reach it through
# `request.app.state.comfy_listener`; background tasks (video generation runs
# for many minutes after its request returned) have no request to go through,
# so they use `get_listener()` instead. One instance per process, created in
# main.py's lifespan.
_active_listener: "ComfyListener | None" = None


def get_listener() -> "ComfyListener | None":
    """The live ComfyListener, or None before startup / after teardown."""
    return _active_listener


class ComfyListener:
    def __init__(self, app_state: Any):
        global _active_listener
        self.app_state = app_state
        self.client_id = str(uuid.uuid4())
        _active_listener = self

        # prompt_id → metadata (image generation jobs)
        self._prompt_meta: dict[str, dict] = {}

        # client_id → WebSocket
        self._active_ws: dict[str, WebSocket] = {}

        # client_id → [image_data, ...] buffered while client is offline
        self._pending: dict[str, list] = {}

        # prompt_id → {"value": int, "max": int, "node": str, "label": str} —
        # latest stage seen for ANY prompt running through ComfyUI, updated on
        # both `executing` (which node) and `progress` (how far into it). Read
        # by tool routers (video / music) to surface ComfyUI's own per-node and
        # per-sampler-step detail without opening their own WebSocket
        # subscription. Entries are evicted on execution_success/error.
        self._step_progress: dict[str, dict] = {}

        # prompt_id → {node_id: "Sampling…"} — see services/comfy/node_labels.py.
        # Registered by whoever submits the workflow, since only they know it.
        self._node_labels: dict[str, dict[str, str]] = {}

    # ── Read-only progress query (called by music/video routers) ─────────────

    def get_step_progress(self, prompt_id: str | None) -> dict | None:
        """Most recent stage/step event for this prompt_id, or None."""
        if not prompt_id:
            return None
        return self._step_progress.get(prompt_id)

    # ── Registration API (called by generate router) ─────────────────────────

    def register_node_labels(self, prompt_id: str, workflow: dict) -> None:
        """Remember node_id → stage label so progress events can name the stage.

        Called by every submitter (image batches, video segments, music,
        upscales); without it a progress event carries a bare node id that
        means nothing outside the workflow it came from.
        """
        while len(self._node_labels) >= MAX_LABEL_MAPS:
            self._node_labels.pop(next(iter(self._node_labels)))
        self._node_labels[prompt_id] = build_label_map(workflow)

    def _label_for(self, prompt_id: str, node: str | None) -> str | None:
        if not node:
            return None
        return self._node_labels.get(prompt_id, {}).get(str(node))

    def register_prompt(
        self,
        prompt_id: str,
        client_id: str,
        index: int,
        total: int,
        batch_id: str,
        prompt_text: str,
        seed: int,
        width: int,
        height: int,
        loras: list[dict] | None = None,
        workflow_name: str = WORKFLOW_NAME,
    ) -> None:
        self._prompt_meta[prompt_id] = {
            "client_id": client_id,
            "index": index,
            "total": total,
            "batch_id": batch_id,
            "prompt_text": prompt_text,
            "seed": seed,
            "width": width,
            "height": height,
            "loras": loras,
            "workflow_name": workflow_name,
            "filename": None,
        }

    def add_ws(self, client_id: str, ws: WebSocket) -> None:
        self._active_ws[client_id] = ws

    def remove_ws(self, client_id: str) -> None:
        self._active_ws.pop(client_id, None)

    async def replay_pending(self, client_id: str, ws: WebSocket) -> None:
        pending = self._pending.pop(client_id, [])
        if pending:
            logger.info(f"Replaying {len(pending)} pending images to {client_id}")
            for img_data in pending:
                try:
                    await ws.send_json({"type": "image_ready", "data": img_data})
                except Exception as e:
                    logger.error(f"Replay failed: {e}")

    # ── Main loop ─────────────────────────────────────────────────────────────

    async def run(self) -> None:
        while True:
            uri = f"ws://{settings.comfyui_host}/ws?clientId={self.client_id}"
            try:
                logger.info(f"Connecting to ComfyUI at {uri}")
                async with websockets.connect(
                    uri,
                    ping_interval=WS_PING_INTERVAL,
                    ping_timeout=WS_PING_TIMEOUT,
                ) as ws:
                    logger.info("ComfyUI WebSocket connected")
                    async for raw in ws:
                        if isinstance(raw, bytes):
                            continue  # Skip binary preview frames
                        try:
                            msg = json.loads(raw)
                        except Exception as exc:
                            logger.error(f"ComfyUI WS: malformed message dropped: {exc}")
                            continue
                        try:
                            await self._route(msg)
                        except Exception:
                            logger.exception(
                                f"ComfyUI WS: _route failed for message type "
                                f"{msg.get('type')!r} (prompt_id={msg.get('data', {}).get('prompt_id')!r})"
                            )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(f"ComfyUI WS error: {exc} — reconnecting in 3s")
                await asyncio.sleep(3)

    # ── Event routing ─────────────────────────────────────────────────────────

    async def _route(self, msg: dict) -> None:
        msg_type = msg.get("type")
        data = msg.get("data", {})
        prompt_id = data.get("prompt_id")

        if not prompt_id:
            return

        # Stash the live stage for ALL prompts, regardless of whether the
        # image-gen pipeline owns them. Tool routers read this to render the
        # same node-by-node detail ComfyUI's own UI shows.
        #
        # `executing` marks entry into a node and carries no counters, so it
        # resets value/max — a stale "step 18/20" left over from the sampler
        # would otherwise keep reading as progress all through the decode.
        if msg_type == "executing":
            node = data.get("node")
            if node is None:
                self._step_progress.pop(prompt_id, None)   # queue went idle
            else:
                self._step_progress[prompt_id] = {
                    "value": None, "max": None, "node": node,
                    "label": self._label_for(prompt_id, node),
                }
        elif msg_type == "progress":
            node = data.get("node")
            self._step_progress[prompt_id] = {
                "value": data.get("value"),
                "max":   data.get("max"),
                "node":  node,
                "label": self._label_for(prompt_id, node),
            }
        elif msg_type in ("execution_success", "execution_error"):
            self._step_progress.pop(prompt_id, None)
            self._node_labels.pop(prompt_id, None)

        if prompt_id not in self._prompt_meta:
            return

        # Frontends get the same label rather than re-deriving it from a node
        # id they cannot interpret (workflow-local ids differ per model).
        if msg_type in ("executing", "progress"):
            label = self._label_for(prompt_id, data.get("node"))
            if label:
                data["label"] = label

        meta = self._prompt_meta[prompt_id]
        client_id = meta["client_id"]
        ws = self._active_ws.get(client_id)

        # Forward raw event for progress display
        if ws:
            try:
                await ws.send_json(msg)
            except Exception as e:
                logger.error(f"Forward failed ({msg_type}): {e}")

        if msg_type == "execution_start":
            await self._on_start(ws, meta)
        elif msg_type == "executed":
            self._capture_filename(prompt_id, data)
        elif msg_type == "execution_success":
            await self._on_success(prompt_id, meta, client_id, ws)
        elif msg_type == "execution_error":
            await self._on_error(ws, prompt_id, meta, data)

    async def _on_start(self, ws, meta: dict) -> None:
        if ws:
            try:
                await ws.send_json({
                    "type": "batch_start",
                    "data": {"index": meta["index"], "total": meta["total"]},
                })
            except Exception as e:
                logger.error(f"batch_start send failed: {e}")

    def _capture_filename(self, prompt_id: str, data: dict) -> None:
        images = data.get("output", {}).get("images", [])
        if images:
            filename = images[0].get("filename", "")
            if filename:
                self._prompt_meta[prompt_id]["filename"] = filename
                logger.debug(f"Captured filename: {filename}")

    async def _on_success(
        self, prompt_id: str, meta: dict, client_id: str, ws
    ) -> None:
        filename = meta.get("filename")
        del self._prompt_meta[prompt_id]

        if not filename:
            return

        # ── Active ingestion pipeline ─────────────────────────────────────────
        ingested_path, db_id = await self._ingest(filename, meta)
        if not ingested_path:
            return

        img_data = {
            "id": db_id,
            "url": f"/api/image/{ingested_path.name}",
            "filename": ingested_path.name,
            "batch_index": meta["index"],
            "batch_total": meta["total"],
        }

        delivered = False
        if ws:
            try:
                await ws.send_json({"type": "image_ready", "data": img_data})
                logger.info(f"Delivered image_ready: {ingested_path.name}")
                delivered = True
            except Exception as e:
                logger.error(f"image_ready send failed: {e}")

        # Only buffer for replay if live delivery did not succeed —
        # otherwise a WS reconnect mid-batch would re-deliver the same image.
        if not delivered:
            self._pending.setdefault(client_id, []).append(img_data)

    async def _on_error(self, ws, prompt_id: str, meta: dict, data: dict) -> None:
        self._prompt_meta.pop(prompt_id, None)
        if ws:
            try:
                await ws.send_json({
                    "type": "execution_error",
                    "data": {
                        "message": data.get("exception_message", "Generation failed"),
                        "batch_index": meta["index"],
                        "batch_total": meta["total"],
                    },
                })
            except Exception as e:
                logger.error(f"execution_error send failed: {e}")

    # ── Ingestion ─────────────────────────────────────────────────────────────

    async def _ingest(self, filename: str, meta: dict) -> tuple[Path | None, str | None]:
        """
        Copy image from ComfyUI output dir to managed storage and
        write a record to the database.

        Returns (destination_path, image_id_str) or (None, None) on failure.
        """
        return await ingest_comfy_image(
            filename,
            prompt=meta.get("prompt_text"),
            seed=meta.get("seed"),
            width=meta.get("width"),
            height=meta.get("height"),
            loras=meta.get("loras"),
            workflow_name=meta.get("workflow_name", WORKFLOW_NAME),
            batch_id=uuid.UUID(meta["batch_id"]) if meta.get("batch_id") else None,
        )
