"""Formen — generated, seamlessly looping depth tracks for the Struktur tool.

The browser draws the geometry (a WebGL raymarcher, see frontends/tools/forms/);
this package holds the numbers the page and the server have to agree on, and
the encode that turns the page's frames into control tracks.
"""
from services.forms.plan import (
    DEFAULT_FORMAT,
    DEFAULT_SECONDS,
    DURATIONS,
    FORMATS,
    GROUP_LABELS,
    LOOP_FPS,
    MAX_FRAMES,
    MIN_FRAMES,
    TRACK_LONG_EDGE,
    encode_command,
    format_table,
    loop_frames,
    mask_regions,
    track_size,
)

__all__ = [
    "DEFAULT_FORMAT", "DEFAULT_SECONDS", "DURATIONS", "FORMATS", "GROUP_LABELS",
    "LOOP_FPS", "MAX_FRAMES", "MIN_FRAMES", "TRACK_LONG_EDGE",
    "encode_command", "format_table", "loop_frames", "mask_regions", "track_size",
]
