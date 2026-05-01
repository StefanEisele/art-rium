"""Print the most recently created Image row's id + filepath + WP status."""
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sqlalchemy import select

from core.db import AsyncSessionLocal
from core.models import Image


async def main():
    async with AsyncSessionLocal() as db:
        row = (await db.execute(
            select(Image).order_by(Image.created_at.desc()).limit(1)
        )).scalar_one_or_none()
        if not row:
            print("(no images)")
            return
        print(f"id          = {row.id}")
        print(f"filepath    = {row.filepath}")
        print(f"title       = {row.title!r}")
        print(f"created_at  = {row.created_at}")
        print(f"wp_media_id = {row.wp_media_id}")


asyncio.run(main())
