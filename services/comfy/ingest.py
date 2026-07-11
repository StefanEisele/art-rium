"""
Shared image ingestion — copy a ComfyUI output PNG into managed storage,
generate the thumbnail and write the Image DB row.

Used by:
  - workers/comfy_listener.py  (WebSocket execution_success path)
  - routers/video.py           (story-frames background job, poll_history path)
"""
import asyncio
import logging
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from core.config import settings
from core.db import AsyncSessionLocal
from core.models import Image
from core.thumbnail import make_thumbnail, thumb_rel_path

logger = logging.getLogger(__name__)


async def ingest_comfy_image(
    comfy_rel_path: str,
    *,
    prompt: str | None = None,
    seed: int | None = None,
    width: int | None = None,
    height: int | None = None,
    lora_name: str | None = None,
    lora_strength: float | None = None,
    workflow_name: str | None = None,
    batch_id: uuid.UUID | None = None,
) -> tuple[Path | None, str | None]:
    """
    Copy *comfy_rel_path* (relative to the ComfyUI output dir, may include a
    subfolder) into storage/images/YYYY/MM/, thumbnail it and insert an Image
    row carrying the generation metadata.

    Returns (destination_path, image_id_str) or (None, None) when the source
    file is missing or the copy fails. A DB-insert failure still returns the
    copied file (logged) — same trade-off as the original listener ingest.
    """
    src = settings.comfyui_output_dir / comfy_rel_path
    if not src.exists():
        logger.error(f"Ingest: source file not found: {src}")
        return None, None

    # Destination: storage/images/YYYY/MM/{uuid}_{original_name}
    now = datetime.now(timezone.utc)
    dest_dir = settings.images_dir / now.strftime("%Y/%m")
    dest_dir.mkdir(parents=True, exist_ok=True)

    image_id = uuid.uuid4()
    dest_filename = f"{image_id}_{src.name}"
    dest = dest_dir / dest_filename

    try:
        await asyncio.to_thread(shutil.copy2, src, dest)
        logger.info(f"Ingested: {src.name} → {dest.relative_to(settings.storage_dir)}")
    except Exception as e:
        logger.error(f"Ingest copy failed: {e}")
        return None, None

    # Relative path stored in DB (portable across machines)
    rel_path = dest.relative_to(settings.storage_dir)

    # Generate JPEG thumbnail (512 px longest side)
    thumb_rel = thumb_rel_path(dest_filename)
    thumb_dest = settings.storage_dir / thumb_rel
    thumb_ok = await make_thumbnail(dest, thumb_dest)

    try:
        async with AsyncSessionLocal() as session:
            record = Image(
                id=image_id,
                filename=dest_filename,
                filepath=str(rel_path),
                thumbnail_path=thumb_rel if thumb_ok else None,
                prompt=prompt,
                seed=seed,
                width=width,
                height=height,
                lora_name=lora_name,
                lora_strength=lora_strength,
                workflow_name=workflow_name,
                batch_id=batch_id,
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
