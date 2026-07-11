"""
Cross-job clip merge — concatenate library clips (possibly from different
workflows) into one video.

Clips from i2v_multi / flf2v (Wan, silent, h265 10-bit) and ltx_i2v (h264,
native audio) may differ in resolution, fps and audio presence, so every
input is normalized in one ffmpeg pass:

  video  scale to the target size (aspect preserved, letterbox pad), unify
         fps, then concat-filter re-encode. The concat *demuxer* with
         -c copy is unreliable for HEVC (see routers/video.py history), so
         a single re-encode through the concat filter is used instead.

  audio  only when at least one clip carries audio: silent clips get an
         anullsrc silence track trimmed to their probed duration so the
         concat filter can run with a=1. Probed durations are container
         durations (±ms); any drift is far below a frame at 24 fps.
"""
import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path

from core.video_thumb import probe_video_duration

logger = logging.getLogger(__name__)

_AUDIO_RATE = 44100


@dataclass(frozen=True)
class MergeInput:
    path: Path
    has_audio: bool
    duration: float = 0.0   # seconds; only required for silent clips in a mixed-audio merge


def build_merge_command(
    ffmpeg_path: str,
    inputs: list[MergeInput],
    dest: Path,
    width: int,
    height: int,
    fps: int,
) -> list[str]:
    """Build the full ffmpeg argv for a normalized concat of *inputs*.

    Pure function — no I/O — so the graph wiring is unit-testable. Encoding
    parameters match the tool's final-video convention (libx265 crf 22,
    yuv420p10le, hvc1 tag; aac 192k when audio is present).
    """
    n = len(inputs)
    if n < 2:
        raise ValueError("build_merge_command requires at least 2 inputs")

    any_audio = any(inp.has_audio for inp in inputs)
    cmd: list[str] = [ffmpeg_path, "-y"]
    for inp in inputs:
        cmd += ["-i", str(inp.path)]

    # Silent clips in a mixed-audio merge get a lavfi silence input each,
    # appended after the real inputs; remember which lavfi index feeds which clip.
    silence_input_for: dict[int, int] = {}
    if any_audio:
        next_idx = n
        for i, inp in enumerate(inputs):
            if inp.has_audio:
                continue
            if inp.duration <= 0:
                raise ValueError(f"Silent clip {inp.path.name} needs a probed duration for the silence track")
            cmd += [
                "-f", "lavfi", "-t", f"{inp.duration:.3f}",
                "-i", f"anullsrc=channel_layout=stereo:sample_rate={_AUDIO_RATE}",
            ]
            silence_input_for[i] = next_idx
            next_idx += 1

    filters: list[str] = []
    concat_feed = ""
    for i, inp in enumerate(inputs):
        filters.append(
            f"[{i}:v]scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps}[v{i}]"
        )
        concat_feed += f"[v{i}]"
        if any_audio:
            if inp.has_audio:
                filters.append(
                    f"[{i}:a]aresample={_AUDIO_RATE},"
                    f"aformat=channel_layouts=stereo[a{i}]"
                )
            else:
                filters.append(f"[{silence_input_for[i]}:a]anull[a{i}]")
            concat_feed += f"[a{i}]"

    a_flag = 1 if any_audio else 0
    out_labels = "[v][a]" if any_audio else "[v]"
    filters.append(f"{concat_feed}concat=n={n}:v=1:a={a_flag}{out_labels}")

    cmd += ["-filter_complex", ";".join(filters), "-map", "[v]"]
    if any_audio:
        cmd += ["-map", "[a]"]
    cmd += [
        "-c:v",     "libx265",
        "-preset",  "medium",
        "-crf",     "22",
        "-pix_fmt", "yuv420p10le",
        "-tag:v",   "hvc1",
        "-r",       str(fps),
    ]
    if any_audio:
        cmd += ["-c:a", "aac", "-b:a", "192k"]
    cmd += [str(dest)]
    return cmd


async def merge_clips(
    inputs: list[MergeInput],
    dest: Path,
    width: int,
    height: int,
    fps: int,
    *,
    ffmpeg_path: str = "ffmpeg",
) -> None:
    """Normalize + concatenate *inputs* into *dest*. Raises RuntimeError on
    ffmpeg failure. Probes durations for silent clips when the merge carries
    audio (the only case that needs them)."""
    if any(inp.has_audio for inp in inputs):
        probed: list[MergeInput] = []
        for inp in inputs:
            if inp.has_audio or inp.duration > 0:
                probed.append(inp)
                continue
            dur = await probe_video_duration(inp.path)
            if dur <= 0:
                raise RuntimeError(f"Could not probe duration of silent clip {inp.path.name}")
            probed.append(MergeInput(path=inp.path, has_audio=False, duration=dur))
        inputs = probed

    cmd = build_merge_command(ffmpeg_path, inputs, dest, width, height, fps)
    logger.info("Merging %d clips → %s", len(inputs), dest.name)
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr_b = await proc.communicate()
    if proc.returncode != 0:
        tail = stderr_b.decode(errors="replace")[-1500:]
        raise RuntimeError(f"ffmpeg merge failed (rc={proc.returncode}): {tail}")
