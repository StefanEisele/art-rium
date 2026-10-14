"""The numbers a generated loop has to satisfy before AnimateLCM will treat it
as one.

A depth loop drawn in the browser is only useful if the render downstream
plays it back as the loop it is. Three things decide that, and all three are
easy to get silently wrong:

  rate     AnimateLCM reads a control track frame by frame and writes its base
           pass at 8 fps (services/comfy/animatelcm.py, FPS). A loop meant to
           last six seconds is therefore 48-ish frames, not 6 x 30. The track is
           stored at that same rate, so its duration *is* the render's.
  length   The Struktur request path snaps every length to 4n+1
           (services/comfy/wan_frames.py). A 48-frame loop would be cut to 45
           and the seam would jump three frames' worth of motion. A loop built
           4n+1 long loses nothing.
  shape    "auto" canvas follows the track's own aspect at a fixed pixel
           budget, so the format chosen here is the format that comes back —
           as long as the track carries the exact ratio.

The mask half has one more: ColorToMask keys on an exact RGB triple, so the
mask is written lossless at 4:4:4 (the SAM 3 path measured why) and checked
here for anything that would not key.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image as PILImage

from core.config import settings
from services.comfy.animatelcm import FPS as LCM_FPS
from services.comfy.animatelcm import canvas_size, output_size
from services.segment.plan import REGION_COLORS

LOOP_FPS = LCM_FPS


def loop_frames(seconds: float) -> int:
    """Frames for a loop of about `seconds` at the render's own rate, 4n+1."""
    target = max(1.0, float(seconds)) * LOOP_FPS
    n = max(2, round((target - 1) / 4))
    return 4 * n + 1


# Offered as chips. Six seconds is the length of the Blender loops this tool
# replaces (188 frames at 30 fps, rendered from every 2nd frame); ten is where
# a base pass stops being something anyone waits for.
DURATIONS: tuple[int, ...] = (4, 6, 8, 10)
DEFAULT_SECONDS = 6
MIN_FRAMES = loop_frames(1)
MAX_FRAMES = loop_frames(DURATIONS[-1])

# (key, label, ratio w, ratio h). Every one of them renders: "auto" turns any
# ratio into a canvas at the original's 544x544 pixel budget.
FORMATS: tuple[tuple[str, str, int, int], ...] = (
    ("9:16", "Reel", 9, 16),
    ("2:3", "Hoch", 2, 3),
    ("4:5", "Feed", 4, 5),
    ("1:1", "Quadrat", 1, 1),
    ("3:2", "Quer", 3, 2),
    ("16:9", "Breit", 16, 9),
    ("21:9", "Kino", 21, 9),
)
DEFAULT_FORMAT = "9:16"

# Twice the render canvas or a little more, like the 1080² Blender passes: the
# graph scales the track down with lanczos, and a track drawn at the canvas
# size would have nothing to anti-alias from.
TRACK_LONG_EDGE = 1024


def _snap8(value: float) -> int:
    return max(64, int(round(value / 8)) * 8)


def track_size(ratio_w: int, ratio_h: int) -> tuple[int, int]:
    """The pixel size a loop of this ratio is drawn and stored at."""
    if ratio_w >= ratio_h:
        return TRACK_LONG_EDGE, _snap8(TRACK_LONG_EDGE * ratio_h / ratio_w)
    return _snap8(TRACK_LONG_EDGE * ratio_w / ratio_h), TRACK_LONG_EDGE


def format_table() -> list[dict]:
    """Every format with what it costs downstream, for the page to quote."""
    table = []
    for key, label, rw, rh in FORMATS:
        width, height = track_size(rw, rh)
        table.append({
            "key": key, "label": label, "ratio": [rw, rh],
            "track": [width, height],
            "canvas": list(canvas_size("auto", (width, height))),
            "output": list(output_size("auto", True, source=(width, height))),
        })
    return table


# What the three mask colours mean when the page sends no words of its own.
# Aligned with REGION_COLORS: red, green, blue.
GROUP_LABELS: tuple[str, ...] = ("Motiv A", "Motiv B", "Umgebung")

# RegionSpec's default threshold — the distance at which the render keys.
KEY_THRESHOLD = 20


def _key_of(color: tuple[int, int, int]) -> int | None:
    """Index of the region colour this pixel keys into, -1 for the empty
    background, None for a colour that keys into nothing."""
    def near(key) -> bool:
        return sum((a - b) ** 2 for a, b in zip(color, key)) <= KEY_THRESHOLD ** 2
    if near((0, 0, 0)):
        return -1
    for i, key in enumerate(REGION_COLORS):
        if near(key):
            return i
    return None


def mask_regions(frames: list[Path], labels: list[str] | None = None
                 ) -> tuple[list[dict], float]:
    """What each colour of a mask covers, and how much of it would not key.

    Returns the `regions` list the Struktur page pre-fills its region rows
    from (only the colours that actually occur) and the fraction of pixels
    that are neither a key colour nor black. Anything above zero there means
    the page anti-aliased its mask — the in-between colours at every edge
    would become holes in the render.
    """
    labels = list(labels or []) + list(GROUP_LABELS[len(labels or []):])
    covered = [0] * len(REGION_COLORS)
    present = [0] * len(REGION_COLORS)
    stray = pixels = 0
    for path in frames:
        with PILImage.open(path) as im:
            rgb = im.convert("RGB")
            n = rgb.width * rgb.height
            pixels += n
            colors = rgb.getcolors(maxcolors=n)
        seen = [0] * len(REGION_COLORS)
        for count, color in colors or []:
            key = _key_of(color)
            if key is None:
                stray += count
            elif key >= 0:
                seen[key] += count
        for i, count in enumerate(seen):
            covered[i] += count
            present[i] += 1 if count else 0
    regions = [
        {"color": list(REGION_COLORS[i]), "label": labels[i],
         "frames": present[i], "coverage": round(covered[i] / pixels, 4)}
        for i in range(len(REGION_COLORS)) if present[i]
    ]
    return regions, (round(stray / pixels, 5) if pixels else 0.0)


def encode_command(pattern: Path, dest: Path, frames: int, lossless: bool) -> list[str]:
    """ffmpeg: a numbered PNG sequence into a control track at the loop rate.

    The image demuxer at `-framerate` is exact — one input picture, one output
    frame, no rate conversion that could duplicate or drop one — and
    `-frames:v` makes a stray extra file in the folder harmless.

    Depth at crf 12: it is a control signal, and block artefacts in it would
    be read as structure. The mask lossless at 4:4:4, or every region edge
    grows a ramp of colours ColorToMask cannot key.
    """
    args = [
        settings.ffmpeg_path, "-y", "-v", "error",
        "-framerate", str(LOOP_FPS), "-start_number", "0", "-i", str(pattern),
        "-frames:v", str(int(frames)), "-an",
    ]
    if lossless:
        args += ["-c:v", "libx264", "-preset", "veryfast", "-qp", "0", "-pix_fmt", "yuv444p"]
    else:
        args += ["-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p"]
    args += ["-movflags", "+faststart", str(dest)]
    return args
