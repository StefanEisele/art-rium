"""The "ambient bed" — a video's own generated audio kept quietly under a lead.

MiniMax H3 clips carry sound the model sampled along with the picture: the
room, the material, the machinery. Laying a piano improvisation or a generated
song on top of such a clip used to mean discarding that entirely. Mixing it in
underneath instead keeps the space the clip was rendered in, with the music
still clearly the lead.

Two callers, one calibration:
  services/improv/mux.py       piano improvisation over a source video
  services/video/soundtrack.py a generated song attached to a video

They build different filter graphs — improv loudness-normalises the result,
the soundtrack path fades it out at the end — so only the volumes and their
bounds live here. Those are the part that must not drift: a bed mixed at one
level in one tool and another level in the other would sound like a bug.
"""
from __future__ import annotations

# Measured by ear against real clips: at 0.35 the clip's own sound is present
# as a space rather than as a second voice competing with the music.
BED_VOLUME_DEFAULT = 0.35
LEAD_VOLUME_DEFAULT = 1.0

# Below 0.1 the bed is inaudible and the checkbox may as well be off; above 0.8
# it stops being a bed and starts fighting the lead for attention.
BED_VOLUME_MIN = 0.1
BED_VOLUME_MAX = 0.8


def clamp_bed_volume(vol: float) -> float:
    """Clamp the ambient-bed volume to the supported [MIN, MAX] window."""
    return max(BED_VOLUME_MIN, min(BED_VOLUME_MAX, vol))


def bed_mix_filter(
    bed_stream: str,
    lead_stream: str,
    *,
    bed_volume: float,
    lead_volume: float = LEAD_VOLUME_DEFAULT,
    tail: str = "",
    out_label: str = "aout",
) -> str:
    """ffmpeg -filter_complex that mixes `bed_stream` under `lead_stream`.

    `tail` is appended to the amix output before the label, for whatever the
    caller does to the combined signal (loudness normalisation, a fade). Pass
    it without a leading comma.

    `duration=shortest` matches the `-shortest` both callers already use, so
    the mix ends where the unmixed version would have.

    `normalize=0` matters and was measured: amix's default divides every input
    by their count, so switching the bed on dropped the lead by ~6 dB (song
    tone -21.7 dB without the bed, -27.7 with it). Turning the bed on must
    change what you hear underneath the music, not how loud the music is. With
    normalisation off the volumes above are absolute, so the caller's `tail`
    has to deal with the summed peak — loudness normalisation or a limiter.
    """
    suffix = f",{tail}" if tail else ""
    return (
        f"[{bed_stream}]volume={clamp_bed_volume(bed_volume):.3f}[bed];"
        f"[{lead_stream}]volume={lead_volume:.3f}[lead];"
        f"[bed][lead]amix=inputs=2:duration=shortest:dropout_transition=0:normalize=0"
        f"{suffix}[{out_label}]"
    )
