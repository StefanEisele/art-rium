"""
Concatenate 1–4 generated videos into a single Instagram-Reels-native
MP4 (1080×1920, 9:16, H.264/AAC).

Each input is independently scaled to fit-inside 1080×1920 and centred on
a black canvas (`scale=...:force_original_aspect_ratio=decrease,pad=...`),
then the normalised streams are stitched with the concat filter and
re-encoded once at the end. Stream-copy concat is unreliable across
HEVC inputs (see [[project_artrium_video_per_segment]]), so we always
re-encode.

Audio: the desktop's video pipeline doesn't write usable audio onto its
MP4s, so we synthesise silence per clip and concat that alongside. This
keeps Instagram's Reels validator happy (it expects an audio track).
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

REEL_WIDTH  = 1080
REEL_HEIGHT = 1920
REEL_FPS    = 30


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

    n = len(sources)
    # Per-input normaliser: scale to fit 1080×1920, pad to fill, lock fps + SAR
    # so all inputs share identical timing metadata for the concat filter.
    norm_chains: list[str] = []
    norm_labels: list[str] = []
    silence_labels: list[str] = []
    for i in range(n):
        v_lbl = f"v{i}"
        a_lbl = f"a{i}"
        norm_chains.append(
            f"[{i}:v]scale={REEL_WIDTH}:{REEL_HEIGHT}:"
            f"force_original_aspect_ratio=decrease,"
            f"pad={REEL_WIDTH}:{REEL_HEIGHT}:(ow-iw)/2:(oh-ih)/2,"
            f"setsar=1,fps={REEL_FPS}[{v_lbl}]"
        )
        # Synthesise silent stereo audio matching this clip's duration.
        norm_chains.append(
            f"aevalsrc=0|0:s=48000:d=10[{a_lbl}_pre];"
            f"[{a_lbl}_pre]atrim=duration=ref[{a_lbl}]".replace("ref", "10")
        )
        # The atrim above gives us a silent 10s track — we let ffmpeg cap the
        # final output via the per-video frame count instead. Simpler: feed
        # the concat with the synthesised silence trimmed by frame_count.
        norm_labels.append(f"[{v_lbl}]")
        silence_labels.append(f"[{a_lbl}]")

    # ── Single-input fast-path ──────────────────────────────────────────────
    if n == 1:
        # No concat needed — just normalise + add silent audio.
        filter_complex = "; ".join([
            f"[0:v]scale={REEL_WIDTH}:{REEL_HEIGHT}:force_original_aspect_ratio=decrease,"
            f"pad={REEL_WIDTH}:{REEL_HEIGHT}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={REEL_FPS}[v]",
        ])
        cmd = [
            ffmpeg_path, "-y",
            "-i", str(sources[0]),
            "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
            "-filter_complex", filter_complex,
            "-map", "[v]", "-map", "1:a:0",
            "-shortest",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "medium", "-crf", "20",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            str(out_path),
        ]
        await _run(cmd, "reel_concat[n=1]")
        return

    # ── Multi-input concat ──────────────────────────────────────────────────
    # Each video gets its own normaliser; a single lavfi anullsrc supplies a
    # silent audio source that all clips share. The concat filter glues the
    # normalised video streams together; the silent audio is appended once
    # via -shortest at the end (concat'ing per-clip silence is overkill).
    chains = [
        f"[{i}:v]scale={REEL_WIDTH}:{REEL_HEIGHT}:force_original_aspect_ratio=decrease,"
        f"pad={REEL_WIDTH}:{REEL_HEIGHT}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={REEL_FPS}[v{i}]"
        for i in range(n)
    ]
    concat_inputs = "".join(f"[v{i}]" for i in range(n))
    chains.append(f"{concat_inputs}concat=n={n}:v=1:a=0[vout]")
    filter_complex = ";".join(chains)

    cmd = [
        ffmpeg_path, "-y",
    ]
    for src in sources:
        cmd += ["-i", str(src)]
    cmd += [
        "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000",
        "-filter_complex", filter_complex,
        "-map", "[vout]",
        "-map", f"{n}:a:0",          # last input is the silent audio source
        "-shortest",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "medium", "-crf", "20",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        str(out_path),
    ]
    await _run(cmd, f"reel_concat[n={n}]")


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
