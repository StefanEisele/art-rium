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

logger = logging.getLogger(__name__)

WS_PING_INTERVAL = 20
WS_PING_TIMEOUT = 20


class ComfyListener:
    def __init__(self, app_state: Any):
        self.app_state = app_state
        self.client_id = str(uuid.uuid4())

        # prompt_id → metadata (image generation jobs)
        self._prompt_meta: dict[str, dict] = {}

        # client_id → WebSocket
        self._active_ws: dict[str, WebSocket] = {}

        # client_id → [image_data, ...] buffered while client is offline
        self._pending: dict[str, list] = {}

        # prompt_id → {"value": int, "max": int, "node": str} — last `progress`
        # event seen for ANY prompt running through ComfyUI. Read by tool
        # routers (video / music) to surface per-sampler-step progress without
        # opening their own WebSocket subscription. Entries are evicted in
        # `_route` when a `executing` event with no node fires (idle).
        self._step_progress: dict[str, dict] = {}

    # ── Read-only progress query (called by music/video routers) ─────────────

    def get_step_progress(self, prompt_id: str | None) -> dict | None:
        """Most recent `progress` event for this prompt_id, or None."""
        if not prompt_id:
            return None
        return self._step_progress.get(prompt_id)

    # ── Registration API (called by generate router) ─────────────────────────

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
        lora_name: str | None = None,
        lora_strength: float | None = None,
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
            "lora_name": lora_name,
            "lora_strength": lora_strength,
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

        # Stash live step progress for ALL prompts, regardless of whether the
        # image-gen pipeline owns them. Tool routers read this to render fine-
        # grained per-sampler-step progress.
        if msg_type == "progress":
            self._step_progress[prompt_id] = {
                "value": data.get("value"),
                "max":   data.get("max"),
                "node":  data.get("node"),
            }
        elif msg_type in ("execution_success", "execution_error"):
            self._step_progress.pop(prompt_id, None)

        if prompt_id not in self._prompt_meta:
            return

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
            lora_name=meta.get("lora_name"),
            lora_strength=meta.get("lora_strength"),
            workflow_name=WORKFLOW_NAME,
            batch_id=uuid.UUID(meta["batch_id"]) if meta.get("batch_id") else None,
        )
