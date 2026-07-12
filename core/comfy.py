"""
Shared ComfyUI helpers — workflow name constant and HTTP prompt dispatch.

Both generate and titler routers submit workflows via post_prompt() so the
httpx error-handling logic lives in exactly one place.
"""
import logging
from typing import Any

import httpx
from fastapi import HTTPException

from core.config import settings

logger = logging.getLogger(__name__)

WORKFLOW_NAME = "z-image_turbo"
ARTIVISION_WORKFLOW_NAME = "artivision_xl"
ERNIE_WORKFLOW_NAME = "ernie_image_turbo"


async def post_prompt(
    workflow: dict[str, Any],
    client_id: str,
    timeout: float = 10.0,
) -> dict:
    """POST a workflow to ComfyUI /prompt and return the parsed JSON response.

    Raises HTTPException on any ComfyUI communication failure so callers
    don't need to duplicate try/except httpx blocks.
    """
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                f"http://{settings.comfyui_host}/prompt",
                json={"prompt": workflow, "client_id": client_id},
            )
            resp.raise_for_status()
            return resp.json()
    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail=f"ComfyUI unreachable: {settings.comfyui_host}",
        )
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"ComfyUI error: {exc.response.status_code}",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
