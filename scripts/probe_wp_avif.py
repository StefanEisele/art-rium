"""
Non-destructive probe: upload a tiny AVIF to WordPress, inspect what
sub-sizes the server generates, then immediately delete the test media.

Tells us whether the WP host has Imagick (or GD) compiled with libavif —
which is required for WP to generate responsive sub-sizes (thumbnail,
medium, large). Without sub-sizes, srcset/lazy-load break for AVIF
uploads and the SEO outcome is *negative*, not positive.

Also runs the same probe with JPEG as a baseline so we can compare what
"healthy" sub-size generation looks like on this host.
"""
import asyncio
import sys
from io import BytesIO
from pathlib import Path

# Allow `python scripts/probe_wp_avif.py` from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image

from services.wordpress.client import request_json, upload_binary


def _test_image(size: tuple[int, int] = (1280, 1280)) -> Image.Image:
    """1280px so WP triggers all default sub-sizes (thumbnail/medium/medium_large/large)."""
    img = Image.new("RGB", size, (40, 80, 200))
    # Add a non-flat gradient so AVIF doesn't collapse to a few bytes
    px = img.load()
    w, h = size
    for y in range(h):
        for x in range(0, w, 8):
            px[x, y] = ((x * 255) // w, (y * 255) // h, 128)
    return img


def _make_avif() -> bytes:
    buf = BytesIO()
    _test_image().save(buf, format="AVIF", quality=60)
    return buf.getvalue()


def _make_jpeg() -> bytes:
    buf = BytesIO()
    _test_image().save(buf, format="JPEG", quality=85)
    return buf.getvalue()


async def probe(label: str, body: bytes, content_type: str, filename: str) -> None:
    print(f"\n=== {label} ===")
    print(f"upload: {len(body)} bytes, content-type={content_type}")
    try:
        media = await upload_binary(
            "/wp/v2/media",
            body=body,
            filename=filename,
            content_type=content_type,
        )
    except RuntimeError as e:
        print(f"UPLOAD FAILED: {e}")
        return

    media_id = media["id"]
    print(f"media_id={media_id}")
    print(f"mime_type={media.get('mime_type')}")
    print(f"source_url={media.get('source_url')}")

    details = media.get("media_details") or {}
    sizes = details.get("sizes") or {}
    print(f"sub-sizes generated: {list(sizes.keys()) or '(none)'}")
    if sizes:
        for name, info in sizes.items():
            print(f"  - {name}: {info.get('width')}x{info.get('height')} "
                  f"mime={info.get('mime_type')} file={info.get('file')}")

    try:
        await request_json("DELETE", f"/wp/v2/media/{media_id}", params={"force": "true"})
        print(f"deleted media_id={media_id}")
    except RuntimeError as e:
        print(f"DELETE FAILED (please remove manually): {e}")


async def main() -> None:
    print("Probing WordPress for AVIF capability...")
    await probe(
        "JPEG baseline",
        _make_jpeg(),
        "image/jpeg",
        "art-rium-probe.jpg",
    )
    await probe(
        "AVIF test",
        _make_avif(),
        "image/avif",
        "art-rium-probe.avif",
    )


if __name__ == "__main__":
    asyncio.run(main())
