"""
Generate a short slideshow MP4 from images using ffmpeg.
Used to produce the Reel companion post for Instagram.

Output format: 1080×1920 (9:16), H.264, 30 fps.
Each image is shown for CLIP_DURATION seconds; adjacent images are
connected by a FADE_DURATION-second crossfade (ffmpeg xfade filter).
"""
import asyncio
import logging
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

CLIP_DURATION = 3.0   # seconds each image is visible
FADE_DURATION = 0.5   # crossfade transition duration
VIDEO_W       = 1080
VIDEO_H       = 1920


async def generate_slideshow(
    image_paths: list[Path],
    output_dir: Path,
    ffmpeg_path: str = "ffmpeg",
) -> Path:
    """
    Build an MP4 slideshow with fade transitions.
    Returns the path of the generated file inside output_dir.
    Raises RuntimeError if ffmpeg exits non-zero.
    """
    if not image_paths:
        raise ValueError("No images provided")

    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"reel_{uuid.uuid4().hex}.mp4"

    cmd = (
        _single_cmd(ffmpeg_path, image_paths[0], out)
        if len(image_paths) == 1
        else _slideshow_cmd(ffmpeg_path, image_paths, out)
    )

    logger.info("Generating reel: %s", " ".join(str(c) for c in cmd))
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed (rc={proc.returncode}): {stderr.decode(errors='replace')[-600:]}"
        )

    mb = out.stat().st_size / 1_048_576
    logger.info("Reel video ready: %s (%.1f MB)", out.name, mb)
    return out


# ── Private helpers ───────────────────────────────────────────────────────────

def _scale_pad() -> str:
    """Scale to 9:16, letterbox/pillarbox with black, force SAR=1."""
    return (
        f"scale={VIDEO_W}:{VIDEO_H}:force_original_aspect_ratio=decrease,"
        f"pad={VIDEO_W}:{VIDEO_H}:(ow-iw)/2:(oh-ih)/2:color=black,"
        f"setsar=1"
    )


def _single_cmd(ffmpeg: str, img: Path, out: Path) -> list[str]:
    return [
        ffmpeg, "-y",
        "-loop", "1", "-t", str(CLIP_DURATION), "-i", str(img),
        "-vf", _scale_pad(),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "30",
        str(out),
    ]


def _slideshow_cmd(ffmpeg: str, imgs: list[Path], out: Path) -> list[str]:
    """
    Build a filter_complex that chains N images with xfade crossfades.

    Each input is looped for CLIP_DURATION seconds.  The xfade offset for
    transition i (0-based) is (i+1) * (CLIP_DURATION - FADE_DURATION),
    measured from the start of the combined output stream.
    """
    n   = len(imgs)
    d   = CLIP_DURATION
    f   = FADE_DURATION
    sp  = _scale_pad()

    cmd: list[str] = [ffmpeg, "-y"]
    for img in imgs:
        cmd += ["-loop", "1", "-t", str(d), "-i", str(img)]

    parts: list[str] = []

    # Scale/pad every input
    for i in range(n):
        parts.append(f"[{i}:v]{sp}[v{i}]")

    # Chain xfade filters
    for i in range(n - 1):
        offset  = round((i + 1) * (d - f), 6)
        in_a    = f"[v{i}]"   if i == 0 else f"[xf{i - 1}]"
        in_b    = f"[v{i + 1}]"
        out_lbl = "[out]"     if i == n - 2 else f"[xf{i}]"
        parts.append(
            f"{in_a}{in_b}xfade=transition=fade"
            f":duration={f}:offset={offset}{out_lbl}"
        )

    cmd += [
        "-filter_complex", ";".join(parts),
        "-map", "[out]",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "30",
        str(out),
    ]
    return cmd
