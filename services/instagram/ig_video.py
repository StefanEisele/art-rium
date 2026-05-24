"""On-demand transcode of stored videos into Instagram-compatible MP4.

Instagram Graph API rejects videos that don't match its feed-video specs
with error code 2207082 ("Media upload has failed"). The two failure modes
we've hit:

  1. Codec/pixel-format — Meta only accepts H.264 in 8-bit yuv420p. Our
     Video tool writes 10-bit HEVC masters and Improv's synth pass copies
     that stream verbatim, so most stored videos land outside the spec.
  2. Aspect ratio — feed/carousel videos must be between 4:5 (0.8) and
     1.91:1; anything more portrait or landscape is silently rejected.
     The Video tool produces ~9:16 portrait (≈0.56), which is too tall.

`ensure_ig_compatible(src)` returns either the original path (already a
match) or a cached sibling `<stem>_ig.mp4` that has been re-encoded to
H.264/yuv420p AND letterbox-padded into the legal aspect range. The
cache lives alongside the source so the existing /share/video/ endpoint
serves it without any new routing; only the first publish of a given
video pays the transcode.
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from core.config import settings

logger = logging.getLogger(__name__)

IG_VIDEO_SUFFIX = "_ig"   # <stem>_ig.mp4 sits next to the master
IG_AR_MIN = 4 / 5         # 0.80 — Instagram's most-portrait carousel limit
IG_AR_MAX = 1.91          # widest IG accepts


async def ensure_ig_compatible(src: Path) -> Path:
    """Return a path to an Instagram-compatible MP4 variant of `src`.

    If the source is already H.264 + 8-bit yuv420p AND its aspect ratio
    sits in IG's legal feed-video range, returns `src` unchanged.
    Otherwise produces (or reuses) `<stem>_ig.mp4` in the same directory.
    """
    if await _is_ig_compatible(src):
        return src
    out = src.with_name(f"{src.stem}{IG_VIDEO_SUFFIX}.mp4")
    # Re-validate any cached sibling against the current spec — a previous
    # transcode may have predated a spec change (e.g. the aspect-pad rule).
    if out.exists() and await _is_ig_compatible(out):
        return out
    await _transcode(src, out)
    return out


def _ffprobe_path() -> str:
    p = Path(settings.ffmpeg_path)
    sibling = p.parent / ("ffprobe" + p.suffix)
    return str(sibling) if sibling.exists() else "ffprobe"


async def _is_ig_compatible(src: Path) -> bool:
    """True iff codec is H.264 + 8-bit yuv420p AND aspect is in IG's legal range."""
    cmd = [
        _ffprobe_path(), "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,pix_fmt,width,height",
        "-of", "json", str(src),
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await proc.communicate()
    if proc.returncode != 0:
        return False
    try:
        streams = json.loads(stdout).get("streams") or []
        if not streams:
            return False
        s = streams[0]
        if s.get("codec_name") != "h264" or s.get("pix_fmt") != "yuv420p":
            return False
        w, h = int(s.get("width", 0)), int(s.get("height", 0))
        if w <= 0 or h <= 0:
            return False
        ar = w / h
        return IG_AR_MIN <= ar <= IG_AR_MAX
    except (ValueError, KeyError, IndexError):
        return False


async def _transcode(src: Path, out: Path) -> None:
    """Re-encode src → out as H.264 + yuv420p + faststart, audio AAC if present.

    Pads the frame into IG's legal aspect range with black bars:
      - if too portrait (iw/ih < 4/5), widens to 4:5
      - if too landscape (iw/ih > 1.91), heightens to 1.91:1
      - otherwise pad is a no-op
    The trunc(.../2)*2 wrapper keeps both output dimensions even so libx264
    can encode in yuv420p without further resampling.
    """
    pad_filter = (
        "pad="
        f"w='if(lt(iw/ih,{IG_AR_MIN}),trunc(ih*{IG_AR_MIN}/2)*2,iw)':"
        f"h='if(gt(iw/ih,{IG_AR_MAX}),trunc(iw/{IG_AR_MAX}/2)*2,ih)':"
        "x='(ow-iw)/2':y='(oh-ih)/2':color=black,setsar=1"
    )
    tmp = out.with_suffix(".mp4.part")
    cmd = [
        settings.ffmpeg_path, "-y",
        "-i", str(src),
        "-map", "0:v:0",
        "-map", "0:a:0?",  # optional audio — many tool outputs are silent
        "-vf", pad_filter,
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-preset", "medium", "-crf", "20",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        "-f", "mp4",  # explicit — ffmpeg can't infer from `.mp4.part` extension
        str(tmp),
    ]
    logger.info("ig_video transcoding %s → %s", src.name, out.name)
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        tmp.unlink(missing_ok=True)
        tail = stderr.decode(errors="replace")[-1500:]
        raise RuntimeError(f"ig_video transcode failed (rc={proc.returncode}): {tail}")
    tmp.replace(out)
