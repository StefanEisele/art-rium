"""
Rendering a LayerPlan — one ffmpeg pass, N looping inputs, one shared clock.

The sibling of services/video/cut_render.py, and it inherits that module's two
hard-won rules unchanged:

**Every slot's length is decided in frames, not seconds**, against the absolute
timeline, so rounding errors telescope away instead of accumulating over fifty
transitions.

**A branch may be asked for more than it has**, so `tpad=stop_mode=clone` holds
the last frame and the frame-exact `trim` immediately after cuts back to the
target. The clone is never seen unless it is needed.

Two things are new.

**Nothing loops here.** The planner already split every slot at the loop
boundaries it crossed and wrapped its position, so each trim lands inside
[0, loop) and the inputs are read exactly once. That is deliberate: the obvious
alternative, `-stream_loop`, repeats each file at *its own* length, so two
tracks that are not exactly the same duration wrap at different moments and
drift apart — which is the alignment invariant, broken, and invisible until
someone watches the last minute of a four-minute piece.

**Two branches can be alive at once.** A transition slot trims the SAME source
window out of two different tracks and hands both to `blend`, whose expression
carries the transition's shape. Both branches are scaled, padded and formatted
identically before they meet, because `blend` requires it and because anything
that treated them differently would break the alignment this whole feature is
built on.

Audio is deliberately absent, exactly as in the beat cut: the piece is a
picture edit and the song is muxed onto it afterwards by the ordinary
soundtrack path in routers/video.py. That is what keeps upscale, look,
Instagram and YouTube working on the result without knowing anything about it.
"""
from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass
from pathlib import Path

from core.subproc import communicate
from core.video_thumb import probe_video_duration
from services.video.cut_render import (
    DEFAULT_MOTION,
    clamp_motion,
    motion_filter,
    segment_frames,
    spill_filtergraph,
)
from services.video.layers import (
    BLEND_NEUTRAL,
    BLEND_PIX_FMT,
    CLOSING_FADE_SECONDS,
    OPENING_FADE_SECONDS,
    LayerPlan,
    blend_expr,
)

# Below this difference two speeds are the same speed, and the branch takes the
# plain linear `setpts` instead of the quadratic one. Not just an optimisation:
# the quadratic divides by (v1 - v0).
_FLAT_SPEED_EPS = 1e-4

logger = logging.getLogger(__name__)

# How much extra source a branch may hold onto before the frame-exact trim.
# Only ever consumed when rounding leaves a slot a frame short.
_TAIL_PAD_SECONDS = 0.5

# Everything meets in this format before a blend: the filter requires matching
# formats, and pinning it here means the two branches cannot be negotiated into
# different ones by whatever preceded them. Ten bits, matching the encoder —
# see the bit-depth note in services/video/layers.py for why a dissolve is
# exactly the picture that shows an 8-bit intermediate.
_BLEND_FORMAT = BLEND_PIX_FMT


# ── Accent glow ──────────────────────────────────────────────────────────────
# Strong accents between changes of material light the picture up: the
# highlights travel toward white with the attack and fall back with the note's
# decay. Planned in services/video/layers.py::plan_glows; only a recorded
# performance has any.
#
# **It is a LUT on the one stream, rewritten once per glowing frame.** The
# envelope is computed here, in Python; `sendcmd` hands `lutyuv` a new curve on
# every frame that glows and the identity on the frame after, so the per-pixel
# cost is a table lookup and nothing has a second clock.
#
# Two things that look simpler and are wrong, both checked 2026-09-13:
#
#   `eq=brightness` is 8-bit only. ffmpeg silently converts the 10-bit stack
#   down to feed it (-v verbose: auto_scale yuv420p10le -> yuv420p), and a
#   dissolve through 8 bits is exactly the banding the 10-bit blend avoids.
#
#   Blending the stack with a generated envelope stream (a 2x2 `nullsrc` ->
#   `geq` -> `scale` -> `blend`) was exact in isolation and wrong inside the
#   layer graph: measured against the same plan rendered without glow, luma
#   rose +30..+40 at moments where the envelope was 0.003, with and without
#   `enable`. Whatever the second input's frames were paired with, it was not
#   the picture's clock. A LUT cannot pair anything with anything.
GLOW_ATTACK_SECONDS = 0.04
GLOW_DECAY_SECONDS = 0.30
# A glow's window closes once its decay is below e^-3 (5%) of its peak.
GLOW_TAIL_DECAYS = 3.0
# How far the brightest highlights travel toward white at full intensity. The
# lift is weighted by the pixel's own level, so the shadows barely move: a
# bloom, not a flash.
GLOW_LIFT = 0.55
# Chroma pulled toward neutral at full intensity; light that bright loses its
# colour, as in the `licht` transition.
GLOW_DESATURATE = 0.30
# A command is stamped this far ahead of its frame, so rounding the time to
# four decimals can never push it one frame late.
_CMD_LEAD_SECONDS = 0.0005


def glow_level(glows: list[list[float]], t: float) -> float:
    """The summed glow envelope at output time `t`, 0..1.

    A linear attack into each accent, an exponential decay out of it, and
    every decay still running is carried rather than cut off by the next one.
    """
    level = 0.0
    for at, amount in glows:
        start = at - GLOW_ATTACK_SECONDS
        if t < start:
            break                              # glows are in time order
        if t < at:
            level += amount * (t - start) / GLOW_ATTACK_SECONDS
        else:
            x = (t - at) / GLOW_DECAY_SECONDS
            if x < 8.0:
                level += amount * math.exp(-x)
    return min(1.0, level)


def glow_windows(glows: list[list[float]], total: float) -> list[tuple[float, float]]:
    """Merged [start, end] spans in which anything glows at all."""
    out: list[tuple[float, float]] = []
    for t, _ in glows:
        a = max(0.0, t - GLOW_ATTACK_SECONDS)
        b = min(total, t + GLOW_TAIL_DECAYS * GLOW_DECAY_SECONDS)
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def _glow_command(t: float, level: float) -> str:
    if level <= 1e-4:
        y = c = "val"
    else:
        y = f"val+(maxval-val)*val/maxval*{GLOW_LIFT * level:.4f}"
        c = f"val+({BLEND_NEUTRAL}-val)*{GLOW_DESATURATE * level:.4f}"
    return f"{t:.4f} lutyuv@glow y '{y}', lutyuv@glow u '{c}', lutyuv@glow v '{c}';"


def glow_commands(glows: list[list[float]], total: float, fps: int) -> str:
    """The `sendcmd` script: one LUT per frame inside every glow window, and
    the identity on the frame after each window closes."""
    lines: list[str] = []
    last_frame = max(0, math.ceil(total * fps) - 1)
    for a, b in glow_windows(glows, total):
        first = max(0, math.floor(a * fps))
        last = min(math.ceil(b * fps), last_frame)
        for k in range(first, last + 2):
            t = k / fps
            level = glow_level(glows, t) if k <= last else 0.0
            lines.append(_glow_command(max(0.0, t - _CMD_LEAD_SECONDS), level))
    return "\n".join(lines) + "\n"


def glow_commands_name(dest: Path) -> str:
    """The script's filename, referenced relative to the output directory —
    ffmpeg runs there, so no Windows drive colon has to survive filtergraph
    escaping."""
    return f"{dest.stem}.glow.txt"


@dataclass(frozen=True)
class LayerSource:
    """One looping track the edit reads from."""
    path: Path
    duration: float
    # ffmpeg filters from services.video.grade.Grade.filter_chain(), or "" to
    # take the track exactly as it is. Applied once per track, before the
    # split, so a track used forty times is corrected once.
    grade: str = ""


def _branch(
    label_in: str, label_out: str, slot, frames: int, fps: int,
    width: int, height: int, motion: str = DEFAULT_MOTION,
) -> str:
    """One track's contribution to one slot: the shared window, retimed.

    Identical for a hold and for either half of a transition — that sameness is
    the alignment guarantee, so it lives in one function rather than being
    written twice and kept in step by hand. That includes the motion mode: the
    two halves of a transition have to be resampled the same way or they stop
    showing the same form, which is the whole feature.
    """
    steps = [
        f"trim=start={slot.src_in:.4f}:end={slot.src_out:.4f}",
        _retime(slot),
        motion_filter(motion, fps),
        f"tpad=stop_mode=clone:stop_duration={_TAIL_PAD_SECONDS}",
        f"trim=start_frame=0:end_frame={frames}",
        "setpts=PTS-STARTPTS",
        f"scale={width}:{height}:force_original_aspect_ratio=decrease",
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2",
        "setsar=1",
        f"format={_BLEND_FORMAT}",
    ]
    return f"[{label_in}]" + ",".join(steps) + f"[{label_out}]"


def _retime(slot) -> str:
    """The `setpts` that plays this slot's source window at its speed ramp.

    Constant speed is the old linear map. A ramp is not: if the rate moves
    linearly in output time from v0 to v1 across a slot of length T, then the
    source position is

        src(t) = v0*t + (v1 - v0) * t^2 / (2T)

    and `setpts` needs the inverse of that — the output moment at which a given
    source frame is due. Inverting a quadratic is the positive root:

        t(s) = T * (sqrt(v0^2 + 2*(v1 - v0)*s/T) - v0) / (v1 - v0)

    The discriminant cannot go negative while both speeds are positive: at the
    far end s = T*(v0+v1)/2 it evaluates to exactly v1^2.

    `T` in the expression is ffmpeg's input timestamp in seconds and `STARTT`
    the first one, so `(T-STARTT)` is the source distance into the window. The
    result is divided by `TB` because setpts answers in timebase units.

    Verified against ffmpeg: a 0.5x-to-1.5x ramp over a 4 s window produced
    exactly 4.000 s and 120 frames, with the per-frame motion rising smoothly
    from 0.214 to 0.465 — a real accelerando, and the frame count still exact.
    """
    v0 = max(1e-3, slot.speed_in)
    v1 = max(1e-3, slot.speed_out)
    duration = slot.duration
    if abs(v1 - v0) < _FLAT_SPEED_EPS or duration <= 0:
        return f"setpts=(PTS-STARTPTS)/{(v0 + v1) / 2:.5f}"
    return (
        f"setpts=({duration:.5f}*(sqrt({v0:.5f}*{v0:.5f}"
        f"+2*({v1:.5f}-{v0:.5f})*(T-STARTT)/{duration:.5f})-{v0:.5f})"
        f"/({v1:.5f}-{v0:.5f}))/TB"
    )


def build_layer_command(
    ffmpeg_path: str,
    plan: LayerPlan,
    sources: list[LayerSource],
    dest: Path,
    width: int,
    height: int,
    fps: int,
    *,
    fade_in: bool = False,
    motion: str = DEFAULT_MOTION,
) -> list[str]:
    """Full ffmpeg argv for `plan`. Pure — no I/O — so it can be unit-tested."""
    if not plan.slots:
        raise ValueError("A layer plan with no slots cannot be rendered")
    if len(sources) < 2:
        raise ValueError("A layer cut needs at least two sources")

    # How many branches each track has to be split into. A transition slot
    # draws on two tracks, a hold on one.
    uses: dict[int, int] = {}
    for slot in plan.slots:
        uses[slot.track] = uses.get(slot.track, 0) + 1
        if slot.is_blend:
            uses[slot.from_track] = uses.get(slot.from_track, 0) + 1

    order = sorted(uses)
    input_of = {track: i for i, track in enumerate(order)}

    cmd: list[str] = [ffmpeg_path, "-y"]
    for track in order:
        cmd += ["-i", str(sources[track].path)]

    filters: list[str] = []
    for track in order:
        n = uses[track]
        labels = "".join(f"[t{track}_{k}]" for k in range(n))
        # The grade goes in front of the split so it runs once however many
        # slots the track appears in.
        head = sources[track].grade
        if n > 1:
            prefix = head + "," if head else ""
            filters.append(f"[{input_of[track]}:v]{prefix}split={n}{labels}")
        else:
            filters.append(f"[{input_of[track]}:v]{head or 'null'}[t{track}_0]")

    taken: dict[int, int] = {}

    def next_label(track: int) -> str:
        k = taken.get(track, 0)
        taken[track] = k + 1
        return f"t{track}_{k}"

    concat_feed = ""
    for i, slot in enumerate(plan.slots):
        frames = segment_frames(slot.start, slot.end, fps)
        if slot.is_blend:
            # Both halves read the SAME source window, so the form is identical
            # on either side of the transition and only the material moves.
            a_in, b_in = next_label(slot.from_track), next_label(slot.track)
            filters.append(_branch(a_in, f"a{i}", slot, frames, fps, width,
                                   height, motion))
            filters.append(_branch(b_in, f"b{i}", slot, frames, fps, width,
                                   height, motion))
            # `variant` is the slot's beat index, which `_reslice` copies onto
            # both halves of a split — so a transition cut in two by a loop
            # boundary keeps wiping the same way across the seam.
            expr = blend_expr(
                slot.blend, slot.duration, slot.blend_p0, slot.blend_p1,
                variant=slot.beat,
            )
            # One expression per plane, never `all_expr`: see the per-plane note
            # in services/video/layers.py. This is the magenta fix.
            filters.append(
                f"[a{i}][b{i}]blend=c0_expr='{expr.luma}'"
                f":c1_expr='{expr.chroma}':c2_expr='{expr.chroma}'[v{i}]"
            )
        else:
            filters.append(
                _branch(next_label(slot.track), f"v{i}", slot, frames, fps,
                        width, height, motion)
            )
        concat_feed += f"[v{i}]"

    filters.append(f"{concat_feed}concat=n={len(plan.slots)}:v=1:a=0[vc]")

    total = plan_duration(plan, fps)
    # Glow goes on the finished stack and before the fades, so a flare near the
    # end still sinks into black with everything else.
    head = "vc"
    if _glows_of(plan, total):
        filters.append(
            f"[vc]sendcmd=f={glow_commands_name(dest)},lutyuv@glow=y=val:u=val:v=val[vg]"
        )
        head = "vg"
    fade_out_at = max(0.0, total - CLOSING_FADE_SECONDS)
    # Opt-in, default off — see services/video/cut_render.py's docstring for
    # why an opening fade-from-black stopped being unconditional.
    tail = [f"fade=t=in:st=0:d={OPENING_FADE_SECONDS:.3f}"] if fade_in else []
    tail.append(f"fade=t=out:st={fade_out_at:.3f}:d={CLOSING_FADE_SECONDS:.3f}")
    filters.append(f"[{head}]{','.join(tail)}[vout]")

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


def _glows_of(plan: LayerPlan, total: float) -> list[list[float]]:
    return [g for g in (plan.glows or []) if 0.0 <= g[0] < total and g[1] > 0]


def plan_duration(plan: LayerPlan, fps: int) -> float:
    """What the rendered file will actually be, to the frame."""
    return sum(segment_frames(s.start, s.end, fps) for s in plan.slots) / fps


async def render_layers(
    plan: LayerPlan,
    sources: list[LayerSource],
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
    cmd = build_layer_command(ffmpeg_path, plan, sources, dest, width, height, fps,
                              fade_in=fade_in, motion=motion)
    cmd, spilled = spill_filtergraph(cmd, dest)
    total = plan_duration(plan, fps)
    glows = _glows_of(plan, total)
    glow_script: Path | None = None
    if glows:
        glow_script = dest.parent / glow_commands_name(dest)
        glow_script.write_text(glow_commands(glows, total, fps), encoding="utf-8")
    blends = sum(1 for s in plan.slots if s.is_blend)
    logger.info(
        "Rendering layer cut: %d segments (%d changes, %d transitions) over "
        "%d tracks → %s (%dx%d @ %d fps, %.2f s, %.1f s of loop consumed, "
        "Bewegung=%s, Schwelle=%.2f, Puls=%.2f, %s, %d glows)",
        len(plan.slots), plan.changes, blends,
        len({s.track for s in plan.slots}), dest.name,
        width, height, fps, plan_duration(plan, fps), plan.source_consumed,
        clamp_motion(motion), plan.swell, plan.pulse, plan.profile, len(plan.glows),
    )
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
            # The glow script is named relative to here; see glow_commands_name.
            cwd=str(dest.parent),
        )
        _, stderr = await communicate(proc)
    finally:
        if spilled:
            spilled.unlink(missing_ok=True)
        if glow_script:
            glow_script.unlink(missing_ok=True)
    if proc.returncode != 0:
        tail = stderr.decode(errors="replace")[-1500:]
        raise RuntimeError(f"ffmpeg layer cut failed (rc={proc.returncode}): {tail}")
    await _verify_length(dest, plan_duration(plan, fps))


# A layer render can come out short WITHOUT ffmpeg saying anything: a branch
# asked for a single frame collapses and exits 0. Measured 2026-09-06 — 200
# one-frame branches produced 0.10 s of a 6.67 s piece. The planner keeps
# segments above that floor now (see MIN_SEGMENT_FRAMES), but a silent
# truncation is exactly the failure that must never be reported as success, so
# the file is measured rather than trusted.
_LENGTH_TOLERANCE = 0.25            # seconds


async def _verify_length(dest: Path, expected: float) -> None:
    actual = await probe_video_duration(dest)
    if actual <= 0:
        return                       # the probe failed; not evidence of a bad file
    if abs(actual - expected) > _LENGTH_TOLERANCE:
        raise RuntimeError(
            f"Layer cut rendered {actual:.2f}s where the plan says "
            f"{expected:.2f}s — the filtergraph dropped part of the piece."
        )
