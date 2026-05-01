"""Print the most recent Image row that hasn't been pushed to WP yet."""
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
            select(Image)
            .where(Image.wp_media_id.is_(None))
            .order_by(Image.created_at.desc())
            .limit(1)
        )).scalar_one_or_none()
        if not row:
            print("(no unuploaded images)")
            return
        print(f"id          = {row.id}")
        print(f"filepath    = {row.filepath}")
        print(f"title       = {row.title!r}")


asyncio.run(main())
