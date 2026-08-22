"""
Rendering an EditPlan — one ffmpeg pass, and the timing has to be exact.

Two things make this different from services/video/merge.py, which also
concatenates clips:

**Every segment's length is decided in frames, not seconds.** A cut at 3.4783 s
cannot be rendered at 24 fps; something has to round. Rounding each segment's
*duration* independently accumulates error — thirty cuts at up to half a frame
each and the last one is a frame and a half off the music. So each segment's
length is `round(end * fps) - round(start * fps)`: the errors telescope away
and every cut lands within half a frame of its beat no matter how many precede
it.

**A shot may be asked for more than it has.** The planner already slowed it to
fit, and `setpts` carries that out; but slow motion plus frame rounding can
still leave a segment one frame short, so `tpad=stop_mode=clone` holds the last
frame and the frame-exact `trim` immediately after cuts back to the target.
The clone is never seen unless it is needed.

Audio is deliberately absent. A beat cut is a picture edit and the song is its
soundtrack — chopping the clips' own audio into sub-second pieces at varying
speeds produces clicks, not sound design. The song is muxed afterwards by the
ordinary soundtrack path in routers/video.py, which means upscale, grain and
every later re-render keep working unchanged.

Colour harmonisation (services/video/grade.py) rides on `RenderSource.grade`
and is applied ONCE PER SOURCE, before the split — not per segment. A clip used
forty times in a stakkato edit is one clip, and grading it forty times would
cost forty times as much for exactly the same picture.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path

from services.video.cut import CLOSING_FADE_SECONDS, OPENING_FADE_SECONDS, EditPlan

logger = logging.getLogger(__name__)

# How much extra source a branch may hold onto before the frame-exact trim.
# Only ever consumed when rounding leaves a segment a frame short.
_TAIL_PAD_SECONDS = 0.5


@dataclass(frozen=True)
class RenderSource:
    """One file the edit reads from, and how it is corrected on the way in."""
    path: Path
    duration: float
    # ffmpeg filters from services.video.grade.Grade.filter_chain(), or "" to
    # take the clip exactly as it is. Lives here rather than in a parallel list
    # so a grade cannot be paired with the wrong file.
    grade: str = ""


def segment_frames(start: float, end: float, fps: int) -> int:
    """Frames a segment occupies, measured on the absolute timeline.

    Both ends are rounded against the same origin, so summing over the whole
    edit gives exactly `round(total * fps)` — no accumulated drift.
    """
    return max(1, round(end * fps) - round(start * fps))


def build_cut_command(
    ffmpeg_path: str,
    plan: EditPlan,
    sources: list[RenderSource],
    dest: Path,
    width: int,
    height: int,
    fps: int,
) -> list[str]:
    """Full ffmpeg argv for `plan`. Pure — no I/O — so it can be unit-tested.

    One input per source however often it appears; `split` fans it out to a
    branch per segment. Decoding a clip once instead of once per appearance is
    the difference between a fast render and a slow one on a stakkato edit that
    uses six clips forty times.
    """
    if not plan.cuts:
        raise ValueError("An edit plan with no cuts cannot be rendered")

    uses: dict[int, int] = {}
    for cut in plan.cuts:
        uses[cut.source] = uses.get(cut.source, 0) + 1
    order = sorted(uses)                      # source index → input index
    input_of = {src: i for i, src in enumerate(order)}

    cmd: list[str] = [ffmpeg_path, "-y"]
    for src in order:
        cmd += ["-i", str(sources[src].path)]

    filters: list[str] = []
    for src in order:
        n = uses[src]
        labels = "".join(f"[s{src}_{k}]" for k in range(n))
        # The grade goes in front of the split, so it runs once however many
        # times the clip appears.
        head = sources[src].grade
        if n > 1:
            prefix = head + "," if head else ""
            filters.append(f"[{input_of[src]}:v]{prefix}split={n}{labels}")
        else:
            filters.append(f"[{input_of[src]}:v]{head or 'null'}[s{src}_0]")

    taken: dict[int, int] = {}
    concat_feed = ""
    for i, cut in enumerate(plan.cuts):
        k = taken.get(cut.source, 0)
        taken[cut.source] = k + 1
        frames = segment_frames(cut.start, cut.end, fps)
        speed = max(1e-3, cut.speed)

        steps = [
            f"trim=start={cut.src_in:.4f}:end={cut.src_out:.4f}",
            f"setpts=(PTS-STARTPTS)/{speed:.5f}",
            f"fps={fps}",
            f"tpad=stop_mode=clone:stop_duration={_TAIL_PAD_SECONDS}",
            f"trim=start_frame=0:end_frame={frames}",
            "setpts=PTS-STARTPTS",
            f"scale={width}:{height}:force_original_aspect_ratio=decrease",
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2",
            "setsar=1",
        ]
        if cut.fade_in > 0:
            steps.append(f"fade=t=in:st=0:d={cut.fade_in:.3f}")
        filters.append(f"[s{cut.source}_{k}]" + ",".join(steps) + f"[v{i}]")
        concat_feed += f"[v{i}]"

    filters.append(f"{concat_feed}concat=n={len(plan.cuts)}:v=1:a=0[vc]")

    total = sum(segment_frames(c.start, c.end, fps) for c in plan.cuts) / fps
    fade_out_at = max(0.0, total - CLOSING_FADE_SECONDS)
    filters.append(
        f"[vc]fade=t=in:st=0:d={OPENING_FADE_SECONDS:.3f},"
        f"fade=t=out:st={fade_out_at:.3f}:d={CLOSING_FADE_SECONDS:.3f}[vout]"
    )

    cmd += [
        "-filter_complex", ";".join(filters),
        "-map", "[vout]",
        "-an",
        "-c:v",     "libx265",
        "-preset",  "medium",
        "-crf",     "22",
        "-pix_fmt", "yuv420p10le",
        "-tag:v",   "hvc1",
        "-r",       str(fps),
        "-movflags", "+faststart",
        str(dest),
    ]
    return cmd


def plan_duration(plan: EditPlan, fps: int) -> float:
    """What the rendered file will actually be, to the frame."""
    return sum(segment_frames(c.start, c.end, fps) for c in plan.cuts) / fps


async def render_cut(
    plan: EditPlan,
    sources: list[RenderSource],
    dest: Path,
    width: int,
    height: int,
    fps: int,
    *,
    ffmpeg_path: str = "ffmpeg",
) -> None:
    """Render `plan` to `dest`. Raises RuntimeError on ffmpeg failure."""
    cmd = build_cut_command(ffmpeg_path, plan, sources, dest, width, height, fps)
    logger.info(
        "Rendering beat cut: %d cuts from %d source(s) → %s (%dx%d @ %d fps, %.2f s)",
        len(plan.cuts), len({c.source for c in plan.cuts}), dest.name,
        width, height, fps, plan_duration(plan, fps),
    )
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        tail = stderr.decode(errors="replace")[-1500:]
        raise RuntimeError(f"ffmpeg beat cut failed (rc={proc.returncode}): {tail}")
