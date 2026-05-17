"""
Concatenate 1–4 generated videos into a single Instagram-Reels-native
MP4 (1080×1920, 9:16, H.264/AAC).

Audio handling: each input keeps its own audio track when present;
inputs without audio get matching-length silence inserted, so the
concat filter sees a video+audio stream pair for every segment. The
final track is a continuous AAC stream whose length matches the
concatenated video.

Each input is independently scaled to fit-inside 1080×1920 and centred
on a black canvas (`scale=...:force_original_aspect_ratio=decrease,
pad=...`), then the normalised streams are stitched with the concat
filter and re-encoded once at the end. Stream-copy concat is unreliable
across HEVC inputs (see [[project_artrium_video_per_segment]]).
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

REEL_WIDTH  = 1080
REEL_HEIGHT = 1920
REEL_FPS    = 30
REEL_SR     = 48000


def _ffprobe_path(ffmpeg_path: str) -> str:
    p = Path(ffmpeg_path)
    sibling = p.parent / ("ffprobe" + p.suffix)
    return str(sibling) if sibling.exists() else "ffprobe"


async def _probe(src: Path, ffmpeg_path: str) -> tuple[float, bool]:
    """Return (duration_seconds, has_audio) for a video file.

    Falls back to (0.0, False) if ffprobe can't read the file — the caller
    will surface this as an error before ffmpeg ever runs.
    """
    cmd = [
        _ffprobe_path(ffmpeg_path),
        "-v", "error",
        "-show_entries", "stream=codec_type",
        "-show_entries", "format=duration",
        "-of", "json",
        str(src),
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    out, _ = await proc.communicate()
    if proc.returncode != 0:
        return 0.0, False
    try:
        info = json.loads(out.decode("utf-8", errors="replace"))
        duration = float(info.get("format", {}).get("duration", 0.0))
        codecs = [s.get("codec_type") for s in info.get("streams", [])]
        return duration, "audio" in codecs
    except Exception:
        return 0.0, False


async def concat_reel_videos(
    sources: list[Path],
    out_path: Path,
    *,
    ffmpeg_path: str = "ffmpeg",
) -> None:
    """
    Stitch the given source videos into one 1080×1920 H.264 MP4 at `out_path`.

    Raises ValueError if no sources are given, RuntimeError if ffmpeg fails.
    """
    if not sources:
        raise ValueError("concat_reel_videos called with empty source list")
    for s in sources:
        if not s.exists():
            raise FileNotFoundError(f"Reel source video missing: {s}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Probe every input upfront so we know which need silent-audio synthesis.
    probes = [await _probe(s, ffmpeg_path) for s in sources]

    # Build per-input video + audio chains. Every chain ends with a stream
    # whose duration is exactly the input's video duration — concat is finicky
    # with unbounded audio streams, so we explicitly bound each one.
    chains: list[str] = []
    concat_pairs: list[str] = []
    for i, (src, (duration, has_audio)) in enumerate(zip(sources, probes)):
        v_lbl = f"v{i}"
        a_lbl = f"a{i}"
        # Defensive: if probe reported 0 duration but the file exists, fall
        # back to a sane default; ffmpeg will still try the input.
        dur = max(duration, 0.1)

        # Video: fit-into-9:16, pad-with-black, lock sar+fps so concat is happy.
        chains.append(
            f"[{i}:v]scale={REEL_WIDTH}:{REEL_HEIGHT}:force_original_aspect_ratio=decrease,"
            f"pad={REEL_WIDTH}:{REEL_HEIGHT}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={REEL_FPS}[{v_lbl}]"
        )
        if has_audio:
            # aformat normalises rate/layout in one filter (avoids aresample's
            # async drift compensation, which has a heavier code path). apad
            # extends short tracks with silence; atrim caps the result at the
            # input's video duration so concat sees a bounded audio stream.
            chains.append(
                f"[{i}:a:0]aformat=sample_fmts=fltp:sample_rates={REEL_SR}:channel_layouts=stereo,"
                f"apad,atrim=duration={dur:.3f},asetpts=PTS-STARTPTS[{a_lbl}]"
            )
        else:
            # anullsrc's d= parameter produces exact-length silence in one go.
            chains.append(
                f"anullsrc=channel_layout=stereo:sample_rate={REEL_SR}:d={dur:.3f},"
                f"asetpts=PTS-STARTPTS[{a_lbl}]"
            )
        concat_pairs.append(f"[{v_lbl}][{a_lbl}]")

    n = len(sources)
    if n == 1:
        # Single input — no concat needed, but still normalise + ensure audio.
        chains.append(f"{concat_pairs[0]}concat=n=1:v=1:a=1[vout][aout]")
    else:
        chains.append(
            "".join(concat_pairs) + f"concat=n={n}:v=1:a=1[vout][aout]"
        )
    filter_complex = ";".join(chains)

    cmd: list[str] = [ffmpeg_path, "-y"]
    for src in sources:
        cmd += ["-i", str(src)]
    cmd += [
        "-filter_complex", filter_complex,
        "-map", "[vout]",
        "-map", "[aout]",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "medium", "-crf", "20",
        "-c:a", "aac", "-b:a", "192k", "-ar", str(REEL_SR),
        "-movflags", "+faststart",
        str(out_path),
    ]
    await _run(cmd, f"reel_concat[n={n}, audio={[h for _,h in probes]}]")


async def _run(cmd: list[str], label: str) -> None:
    logger.info("ffmpeg %s: %s", label, " ".join(str(c) for c in cmd))
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        tail = stderr.decode(errors="replace")[-800:]
        raise RuntimeError(f"ffmpeg {label} failed (rc={proc.returncode}): {tail}")
