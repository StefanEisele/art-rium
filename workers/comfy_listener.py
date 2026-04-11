"""
ComfyUI WebSocket listener — event relay + active ingestion pipeline.

Responsibilities:
  1. Maintain a persistent WS connection to ComfyUI
  2. Route events to the correct frontend WebSocket client
  3. On execution_success: copy image to managed storage + write DB record
"""
import asyncio
import logging
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import websockets
from fastapi import WebSocket

from core.config import settings
from core.db import AsyncSessionLocal
from core.models import Image

logger = logging.getLogger(__name__)

WS_PING_INTERVAL = 20
WS_PING_TIMEOUT = 20


class ComfyListener:
    def __init__(self, app_state: Any):
        self.app_state = app_state
        self.client_id = str(uuid.uuid4())

        # prompt_id → metadata (image generation jobs)
        self._prompt_meta: dict[str, dict] = {}

        # prompt_id → {client_id, output_node} (text-only jobs)
        self._text_prompts: dict[str, dict] = {}

        # client_id → WebSocket
        self._active_ws: dict[str, WebSocket] = {}

        # client_id → [image_data, ...] buffered while client is offline
        self._pending: dict[str, list] = {}

    # ── Registration API (called by generate router) ─────────────────────────

    def register_text_prompt(
        self,
        prompt_id: str,
        client_id: str,
        output_node: str,
    ) -> None:
        self._text_prompts[prompt_id] = {
            "client_id": client_id,
            "output_node": output_node,
        }

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
            "filename": None,
        }

    def add_ws(self, client_id: str, ws: WebSocket) -> None:
        self._active_ws[client_id] = ws

    def remove_ws(self, client_id: str) -> None:
        self._active_ws.pop(client_id, None)

    async def replay_pending(self, client_id: str, ws: WebSocket) -> None:
        pending = self._pending.get(client_id, [])
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
                            import json
                            msg = json.loads(raw)
                            await self._route(msg)
                        except Exception:
                            pass
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

        # Text-only jobs (e.g. titler workflow)
        if prompt_id in self._text_prompts:
            await self._route_text(msg, prompt_id, msg_type, data)
            return

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

    async def _route_text(self, msg: dict, prompt_id: str, msg_type: str, data: dict) -> None:
        meta = self._text_prompts[prompt_id]
        client_id = meta["client_id"]
        ws = self._active_ws.get(client_id)

        # Forward raw event so frontend can show progress
        if ws:
            try:
                await ws.send_json(msg)
            except Exception as e:
                logger.error(f"Text forward failed ({msg_type}): {e}")

        if msg_type == "executed" and data.get("node") == meta["output_node"]:
            text = ""
            output = data.get("output", {})
            texts = output.get("text", [])
            if texts:
                text = texts[0] if isinstance(texts[0], str) else str(texts[0])

            if ws and text:
                try:
                    await ws.send_json({"type": "text_ready", "data": {"text": text}})
                    logger.info(f"Delivered text_ready for prompt {prompt_id}")
                except Exception as e:
                    logger.error(f"text_ready send failed: {e}")
            del self._text_prompts[prompt_id]

        elif msg_type == "execution_success":
            self._text_prompts.pop(prompt_id, None)

        elif msg_type == "execution_error":
            if ws:
                try:
                    await ws.send_json({
                        "type": "execution_error",
                        "data": {"message": data.get("exception_message", "Workflow failed")},
                    })
                except Exception:
                    pass
            self._text_prompts.pop(prompt_id, None)

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

        # Buffer for replay if client is offline
        self._pending.setdefault(client_id, []).append(img_data)

        if ws:
            try:
                await ws.send_json({"type": "image_ready", "data": img_data})
                logger.info(f"Delivered image_ready: {ingested_path.name}")
            except Exception as e:
                logger.error(f"image_ready send failed: {e}")

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
        src = settings.comfyui_output_dir / filename
        if not src.exists():
            logger.error(f"Ingest: source file not found: {src}")
            return None, None

        # Destination: storage/images/YYYY/MM/{uuid}_{original_name}
        now = datetime.now(timezone.utc)
        dest_dir = settings.images_dir / now.strftime("%Y/%m")
        dest_dir.mkdir(parents=True, exist_ok=True)

        image_id = uuid.uuid4()
        dest_filename = f"{image_id}_{filename}"
        dest = dest_dir / dest_filename

        try:
            await asyncio.to_thread(shutil.copy2, src, dest)
            logger.info(f"Ingested: {filename} → {dest.relative_to(settings.storage_dir)}")
        except Exception as e:
            logger.error(f"Ingest copy failed: {e}")
            return None, None

        # Relative path stored in DB (portable across machines)
        rel_path = dest.relative_to(settings.storage_dir)

        try:
            async with AsyncSessionLocal() as session:
                record = Image(
                    id=image_id,
                    filename=dest_filename,
                    filepath=str(rel_path),
                    prompt=meta.get("prompt_text"),
                    seed=meta.get("seed"),
                    width=meta.get("width"),
                    height=meta.get("height"),
                    workflow_name="z-image_turbo",
                    batch_id=uuid.UUID(meta["batch_id"]) if meta.get("batch_id") else None,
                    created_at=now,
                )
                session.add(record)
                await session.commit()
                logger.info(f"DB record created: {image_id}")
        except Exception as e:
            logger.error(f"DB insert failed: {e}")
            # File was copied — don't delete it, just log the failure
            return dest, str(image_id)

        return dest, str(image_id)
