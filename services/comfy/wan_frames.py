"""Frame arithmetic for control tracks, shared by the Struktur tool's request
path.

Wan-family models sample 4n+1 frames and a track's own frames are the guidance,
so a render may never be longer than the track can fill. The Struktur tool
used to take these from the Wan VACE builder; they outlived it.
"""

# The rate a Struktur job row is stored at. AnimateLCM renders at its own
# base rate and rewrites the row once the file is probed.
FPS = 16


def snap_length(frames: int) -> int:
    """Wan samples 4n+1 frames; anything else is silently wrong."""
    return max(5, ((max(1, frames) - 1) // 4) * 4 + 1)


def loop_length(source_frames: int) -> int:
    """The longest valid length that fits inside a looping control track.

    A 3D turntable render already loops: the last frame is one rotation step
    short of the first, so playing the whole range and wrapping is seamless.
    Nothing needs to be added — the only obstacle is the 4n+1 requirement, and
    the fix is to take the largest 4n+1 length the track can fill rather than
    padding it out. This is `snap_length` under another name; callers reach
    for it to say "as much of this loop as the model takes".
    """
    return snap_length(source_frames)
