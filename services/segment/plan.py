"""Colour-region vocabulary and the frame budget of a control track.

Two small, pure pieces that the router, the SAM 3 runner and the frontend all
have to agree on, kept here so they cannot drift apart.

The frame budget exists because a control track is not footage — it is a bill.
AnimateLCM renders one diffusion step per frame and then a second pass over the
same frames, so a 20 s phone clip at 30 fps is 600 frames, which is a render
measured in hours for motion that a tenth of those frames would have carried.
Two dials cut it down, and they cut different things:

  seconds  how much of the take is used at all. Shortens the clip.
  stride   how many of that take's frames are kept. Keeps the whole clip and
           thins it — every 2nd or 3rd frame — so the same movement happens
           over fewer frames, which reads as faster motion. That is not a loss
           here: the render interpolates back up with RIFE afterwards, and
           AnimateDiff's own base rate is 8 fps against a 30 fps phone camera.

Both are applied once, at ingest, and baked into the stored track. Doing it
later — asking the loader for a different rate at render time — is the trap
this avoids: `VHS_LoadVideoPath` with `force_rate=16` against a 30 fps source
silently yields a different frame count than asked for, and the tail of the
render then drifts with nothing guiding it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# The three keys of a mask video. Pure primaries and nothing else: ColorToMask
# measures euclidean RGB distance to an exact triple, so the further apart the
# keys are, the more codec and resampling error a region tolerates before its
# pixels stop matching. Any two of these are 360 units apart, which is the most
# three colours can be.
REGION_COLORS: tuple[tuple[int, int, int], ...] = ((255, 0, 0), (0, 255, 0), (0, 0, 255))
REGION_LABELS: tuple[str, ...] = ("Rot", "Grün", "Blau")

# AnimateLCM chains one IPAdapterAdvanced per region onto the model, all inside
# a single sampler pass. Three is the cap the render path already enforces.
MAX_REGIONS = len(REGION_COLORS)

# Offered in the UI. Not a clamp — an arbitrary length still works — but these
# are the lengths that pair with the render's own cost.
DURATIONS: tuple[float, ...] = (2.0, 3.0, 4.0, 6.0, 8.0, 10.0)
STRIDES: tuple[int, ...] = (1, 2, 3, 4)

# A guard rather than a preference: past this a single render stops being
# something anyone waits for, and it is almost always a mis-set dial.
FRAME_CEILING = 400


@dataclass(frozen=True)
class FrameBudget:
    """What a trim/stride setting actually costs, in frames."""
    source_frames: int
    source_fps: float
    start: float
    seconds: float | None
    stride: int
    taken: int              # frames of the source inside the trim
    kept: int               # what survives the stride — the track's length
    span: float             # seconds of the take covered
    fps: float              # rate the kept frames play back at, in real time
    capped: bool            # the ceiling, not the settings, decided `kept`


def plan_frames(
    source_frames: int,
    source_fps: float | None,
    seconds: float | None = None,
    stride: int = 1,
    start: float = 0.0,
    ceiling: int = FRAME_CEILING,
) -> FrameBudget:
    """Resolve a trim and a stride against a real source into a frame count.

    The count is what the UI shows before anything is uploaded or segmented,
    and what the ingest then produces — one function so the promise and the
    result cannot disagree.
    """
    fps = float(source_fps or 0) or 30.0
    frames = max(0, int(source_frames or 0))
    stride = max(1, int(stride or 1))
    start = max(0.0, float(start or 0.0))

    offset = min(frames, int(round(start * fps)))
    available = frames - offset
    if seconds:
        available = min(available, int(math.floor(float(seconds) * fps)))
    taken = max(0, available)

    # `select='not(mod(n,S))'` keeps frame 0 and every Sth after it, so the
    # count is a ceiling division rather than a floor.
    kept = math.ceil(taken / stride) if taken else 0
    capped = kept > ceiling
    if capped:
        kept = ceiling
        taken = min(taken, kept * stride)

    return FrameBudget(
        source_frames=frames,
        source_fps=fps,
        start=start,
        seconds=float(seconds) if seconds else None,
        stride=stride,
        taken=taken,
        kept=kept,
        span=round(taken / fps, 3) if fps else 0.0,
        fps=round(fps / stride, 4),
        capped=capped,
    )


def budget_dict(budget: FrameBudget) -> dict:
    """The JSON the frontend reads to show "6 s -> 90 Bilder" before uploading."""
    return {
        "frames": budget.kept,
        "taken": budget.taken,
        "stride": budget.stride,
        "seconds": budget.seconds,
        "span": budget.span,
        "fps": budget.fps,
        "source_frames": budget.source_frames,
        "source_fps": budget.source_fps,
        "capped": budget.capped,
        "ceiling": FRAME_CEILING,
    }


def trim_filters(budget: FrameBudget) -> list[str]:
    """The ffmpeg video filters that realise a budget.

    Two filters, and both are load-bearing:

    `select` drops the frames the stride skips. `setpts` then renumbers what
    survives onto an even spacing at the new rate, which is what makes the
    result play back over the seconds it actually covers instead of at
    `stride` times speed.

    The caller pairs these with `-fps_mode passthrough` and **no `-r`**. That
    combination was picked by measurement, not taste — on a 171-frame 30 fps
    source asked for every 2nd frame:

      -r with -fps_mode passthrough   ffmpeg refuses outright ("contradictory")
      setpts, no -r                   86 frames but the container still says
                                      30 fps, so it plays at double speed
      -r alone                        right on this clip, but -r is a CFR
                                      conversion: it is free to *duplicate*
                                      frames, and with -frames:v capping the
                                      count those duplicates would silently
                                      push real frames out of the file
      setpts + passthrough            86 frames, 15 fps, 5.7 s — exact, and
                                      duplication is not representable

    Exactness is not a nicety here: a mask is keyed frame by frame against its
    footage, and a duplicated frame in one of the pair slides every region off
    its object for the rest of the clip.
    """
    filters = []
    if budget.stride > 1:
        filters.append(f"select='not(mod(n\\,{budget.stride}))'")
    filters.append(f"setpts=N/{budget.fps:g}/TB")
    return filters
