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

**Retiming does not invent frames, so something else has to.** `setpts` moves
the frames the source already had; a shot at half speed therefore has half the
frames the grid wants, and until 2026-09-06 the gaps were filled by repeating
the nearest one. That is judder, and on a stretch dialled to the 4x ceiling it
is three frames in four standing still. The motion block below is what replaced
it, and what it was measured against.

Audio is deliberately absent. A beat cut is a picture edit and the song is its
soundtrack — chopping the clips' own audio into sub-second pieces at varying
speeds produces clicks, not sound design. The song is muxed afterwards by the
ordinary soundtrack path in routers/video.py, which means upscale, grain and
every later re-render keep working unchanged.

Colour harmonisation (services/video/grade.py) rides on `RenderSource.grade`
and is applied ONCE PER SOURCE, before the split — not per segment. A clip used
forty times in a stakkato edit is one clip, and grading it forty times would
cost forty times as much for exactly the same picture.

The opening fade-from-black (`OPENING_FADE_SECONDS`) is opt-in via `fade_in`,
default off — a picture edit starting on a black frame is a look someone asks
for, not something every render should do. The closing fade-to-black stays
unconditional: it is what keeps the last shot from ending on a hard cut to
nothing, which reads as a mistake in a way an unfaded opening does not.
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


# ── Motion: how a retimed branch is put back onto the output grid ────────────
# `setpts` retimes without inventing frames. A branch played at 0.7x still has
# only the source's own frames, now spread further apart, and something has to
# fill the gaps; played at 1.6x it has more frames than the grid has slots, and
# something has to throw some away. `fps=` does both by picking the nearest
# frame, so it duplicates and drops — at an irregular cadence, which is exactly
# what the eye reads as judder.
#
# Measured 2026-09-06, a 30 fps source retimed onto a 30 fps grid:
#
#     speed 0.70  →  18 of 60 output frames were duplicates of the frame
#                    before them. Nearly a third of the picture stands still,
#                    in an uneven pattern.
#     speed 1.60  →  no duplicates, but frames dropped unevenly: the per-frame
#                    motion delta scattered half again as wide as an unretimed
#                    reference (cv 0.50 against 0.38).
#
# Both remedies below remove every duplicate. The differences that matter:
#
#   * `framerate` blends the two frames the output moment falls between. It is
#     bit-exact pass-through when nothing is being retimed — verified against a
#     frame hash of the untouched source — so it costs nothing on a slot at
#     speed 1.0 and engages only where the judder actually was. Measured at
#     816x1440: 0.14 s against `fps=`'s 0.10 s for 214 output frames.
#
#   * `minterpolate` estimates motion and synthesises genuinely new frames.
#     Truer movement, but it costs 0.42 s per frame at 816x1440 — about fifty
#     minutes for a four-minute piece — and it smears where the estimate
#     fails. Offered, not defaulted.
#
# So the default is the blend: strictly better than judder on this material,
# and free where there is no judder to fix.

MOTION_ORIGINAL = "original"    # nearest frame — duplicates and drops
MOTION_BLEND = "misch"          # blend across the gap
MOTION_FLOW = "fluss"           # motion-compensated interpolation
DEFAULT_MOTION = MOTION_BLEND

MOTION_MODES: tuple[tuple[str, str, str], ...] = (
    (MOTION_BLEND, "Mischen",
     "Zwischenbilder werden aus den beiden Nachbarbildern gemischt. "
     "Tempowechsel laufen weich, statt zu ruckeln. Der Normalfall."),
    (MOTION_FLOW, "Fluss",
     "Zwischenbilder werden aus der erkannten Bewegung neu gerechnet. Die "
     "sauberste Bewegung — und mit Abstand die längste Rechenzeit."),
    (MOTION_ORIGINAL, "Original",
     "Keine Zwischenbilder: in Zeitlupe stehen Bilder doppelt, im Zeitraffer "
     "fallen welche weg. Am schnellsten, sichtbar ruckelig."),
)

_MOTION_KEYS = frozenset(key for key, _, _ in MOTION_MODES)


def clamp_motion(mode: str | None) -> str:
    """An unknown mode falls back to the default rather than being refused: a
    stale client should render smoothly, not fail."""
    return mode if mode in _MOTION_KEYS else DEFAULT_MOTION


def motion_options() -> list[dict]:
    return [{"key": k, "label": lab, "hint": h} for k, lab, h in MOTION_MODES]


# ── The filtergraph outgrew the command line ────────────────────────────────
# Windows caps an entire command line at 32767 characters, and a song-length
# edit is nowhere near that small: measured on a four-minute 120 BPM plan —
# 149 slots, 66 of them transitions — the graph came to 78 KB for a plain
# dissolve and 96 KB for the flash. The limit is not a tuning parameter, it is
# a wall, and every one of those renders would have died on it.
#
# `-filter_complex_script` reads the same graph out of a file, so the argv
# stays short however long the piece is. Only long graphs are spilled, because
# a command that can be read in a log is worth keeping readable.
FILTER_ARG_LIMIT = 8000


def spill_filtergraph(cmd: list[str], near: Path) -> tuple[list[str], Path | None]:
    """Move an oversized `-filter_complex` into a sidecar script file.

    Returns the argv to actually run, and the file the caller has to delete
    afterwards — or None when the graph was small enough to leave alone.
    """
    try:
        i = cmd.index("-filter_complex")
    except ValueError:
        return cmd, None
    graph = cmd[i + 1]
    if len(graph) <= FILTER_ARG_LIMIT:
        return cmd, None
    path = near.with_name(near.stem + ".filtergraph.txt")
    path.write_text(graph, encoding="utf-8")
    spilled = list(cmd)
    spilled[i:i + 2] = ["-filter_complex_script", str(path)]
    return spilled, path


def motion_filter(mode: str, fps: int) -> str:
    """The filter that puts a retimed branch back onto the output grid."""
    resolved = clamp_motion(mode)
    if resolved == MOTION_FLOW:
        return (f"minterpolate=fps={fps}:mi_mode=mci:mc_mode=aobmc"
                ":me_mode=bidir:vsbmc=1")
    if resolved == MOTION_BLEND:
        # interp_start/end wide open: the default 15..240 window snaps the ends
        # of the blend back onto a single frame, which puts some of the judder
        # back at exactly the moments the blend was meant to smooth.
        return f"framerate=fps={fps}:interp_start=0:interp_end=255"
    return f"fps={fps}"


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
    *,
    fade_in: bool = False,
    motion: str = DEFAULT_MOTION,
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
            motion_filter(motion, fps),
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
    # The opening fade is opt-in: a picture edit that starts on a black frame
    # is a choice, not a default, and it used to be applied unconditionally.
    tail = [f"fade=t=in:st=0:d={OPENING_FADE_SECONDS:.3f}"] if fade_in else []
    tail.append(f"fade=t=out:st={fade_out_at:.3f}:d={CLOSING_FADE_SECONDS:.3f}")
    filters.append(f"[vc]{','.join(tail)}[vout]")

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
    fade_in: bool = False,
    motion: str = DEFAULT_MOTION,
) -> None:
    """Render `plan` to `dest`. Raises RuntimeError on ffmpeg failure."""
    cmd = build_cut_command(ffmpeg_path, plan, sources, dest, width, height, fps,
                            fade_in=fade_in, motion=motion)
    cmd, spilled = spill_filtergraph(cmd, dest)
    logger.info(
        "Rendering beat cut: %d cuts from %d source(s) → %s "
        "(%dx%d @ %d fps, %.2f s, Bewegung=%s)",
        len(plan.cuts), len({c.source for c in plan.cuts}), dest.name,
        width, height, fps, plan_duration(plan, fps), clamp_motion(motion),
    )
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
    finally:
        if spilled:
            spilled.unlink(missing_ok=True)
    if proc.returncode != 0:
        tail = stderr.decode(errors="replace")[-1500:]
        raise RuntimeError(f"ffmpeg beat cut failed (rc={proc.returncode}): {tail}")
