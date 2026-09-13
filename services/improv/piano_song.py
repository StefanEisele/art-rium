"""
A piano recording, taken in as a song.

The layer cut, the beat cut and the soundtrack mux all read their music from a
`Song` row: the beat-map sidecar is keyed on its id, the picker lists it, and
every re-mux after an upscale or a look pass finds the track through
`soundtrack_song_id`. Making a recording a Song row with its own workflow is
what lets all of that work unchanged. The one place that has to know the
difference is `routers/cut.py::_song_beatmap`, which reads a recording as a
performance (services/video/piano.py) rather than as a generated track.

Only the sound is kept. The video of the hands stays where it was — on the
phone, or with its improv session.

**The level is corrected with one gain for the whole take, never with a
compressor or `loudnorm`'s dynamic mode.** A piano recording through a Scarlett
typically sits well below streaming level, and the obvious fix — the improv
mux's `loudnorm=I=-14` — works by riding the gain, which flattens exactly the
pianissimo-to-forte arc the layer cut is supposed to follow. So loudness is
*measured* with loudnorm, one gain is derived from it, and the true peak caps
that gain so nothing clips.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import uuid
from datetime import datetime, timezone
from pathlib import Path

from core.config import settings
from core.db import AsyncSessionLocal
from core.models import Song
from core.subproc import communicate
from core.video_thumb import probe_has_audio, probe_video_duration
from services.video import beats, piano

logger = logging.getLogger(__name__)

RECORDING_WORKFLOW = "piano_recording"
RECORDING_TAGS = "Klavier-Aufnahme"

TARGET_LUFS = -14.0          # Instagram's integrated target, same as the improv mux
PEAK_CEILING_DBTP = -1.0     # the gain stops here rather than clip
MAX_GAIN_DB = 24.0

# A layer cut needs at least a couple of changes to be anything.
MIN_SECONDS = 3.0

# A session is imported at most once: its song id is derived from the session
# id, so a second import finds the first one instead of making a copy.
_SESSION_NAMESPACE = uuid.UUID("6f1c3a52-9d0e-4b7a-8c2f-5e4d3b2a1908")


def song_id_for_session(session_id: uuid.UUID) -> uuid.UUID:
    return uuid.uuid5(_SESSION_NAMESPACE, f"improv-session:{session_id}")


def is_recording(song: Song) -> bool:
    return song.workflow == RECORDING_WORKFLOW


# ── Level ────────────────────────────────────────────────────────────────────

def level_gain(integrated_lufs: float, true_peak_dbtp: float) -> float:
    """The one gain, in dB, that brings the take to TARGET_LUFS without
    pushing its true peak past PEAK_CEILING_DBTP. Raises on a silent take."""
    if not math.isfinite(integrated_lufs):
        raise ValueError("Die Aufnahme ist stumm.")
    gain = TARGET_LUFS - integrated_lufs
    if math.isfinite(true_peak_dbtp):
        gain = min(gain, PEAK_CEILING_DBTP - true_peak_dbtp)
    return max(-MAX_GAIN_DB, min(MAX_GAIN_DB, gain))


def parse_loudnorm_json(stderr: str) -> tuple[float, float]:
    """(input_i, input_tp) out of loudnorm's `print_format=json` report, which
    is the last JSON object ffmpeg writes to stderr."""
    start, end = stderr.rfind("{"), stderr.rfind("}")
    if start < 0 or end <= start:
        raise RuntimeError("loudnorm printed no measurement")
    data = json.loads(stderr[start:end + 1])
    return float(data["input_i"]), float(data["input_tp"])


async def measure_loudness(src: Path, ffmpeg_path: str) -> tuple[float, float]:
    proc = await asyncio.create_subprocess_exec(
        ffmpeg_path, "-hide_banner", "-nostats", "-i", str(src),
        "-map", "0:a:0", "-af", f"loudnorm=I={TARGET_LUFS}:TP=-1.5:print_format=json",
        "-f", "null", "-",
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await communicate(proc)
    text = stderr.decode(errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"Loudness measurement failed (rc={proc.returncode}): {text[-500:]}")
    return parse_loudnorm_json(text)


def extract_command(ffmpeg_path: str, src: Path, dest: Path, gain_db: float) -> list[str]:
    """Audio only, one fixed gain, lossless. FLAC because the track is
    re-encoded once more when it is muxed onto the picture, and a lossy
    intermediate would put a second generation of artefacts on a solo piano,
    which hides nothing."""
    return [
        ffmpeg_path, "-y", "-hide_banner", "-i", str(src),
        "-map", "0:a:0", "-vn", "-sn", "-dn",
        "-af", f"volume={gain_db:.2f}dB",
        "-ar", "48000", "-c:a", "flac",
        str(dest),
    ]


# ── Import ───────────────────────────────────────────────────────────────────

async def create_piano_song(
    src: Path,
    *,
    title: str | None,
    origin: str,
    song_id: uuid.UUID | None = None,
) -> Song:
    """Take the sound of `src` in as a finished Song row and return it.

    Raises ValueError for anything the user can fix (no audio, silent, too
    short) and RuntimeError when ffmpeg itself fails.
    """
    if not await probe_has_audio(src):
        raise ValueError("Die Aufnahme hat keine Tonspur.")

    ffmpeg = settings.ffmpeg_path
    integrated, peak = await measure_loudness(src, ffmpeg)
    gain = level_gain(integrated, peak)

    song_id = song_id or uuid.uuid4()
    settings.songs_dir.mkdir(parents=True, exist_ok=True)
    dest = settings.songs_dir / f"piano_{song_id.hex}.flac"

    proc = await asyncio.create_subprocess_exec(
        *extract_command(ffmpeg, src, dest, gain),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await communicate(proc)
    if proc.returncode != 0:
        dest.unlink(missing_ok=True)
        tail = stderr.decode(errors="replace")[-600:]
        raise RuntimeError(f"Audio extraction failed (rc={proc.returncode}): {tail}")

    duration = await probe_video_duration(dest)
    if duration < MIN_SECONDS:
        dest.unlink(missing_ok=True)
        raise ValueError(f"Die Aufnahme ist mit {duration:.1f}s zu kurz für einen Schnitt.")

    song = Song(
        id=song_id,
        filename=dest.name,
        filepath=dest.relative_to(settings.storage_dir).as_posix(),
        tags=RECORDING_TAGS,
        duration_seconds=max(1, round(duration)),
        title=(title or "").strip()[:255] or None,
        notes=(f"{origin} · gemessen {integrated:.1f} LUFS, "
               f"eine Verstärkung von {gain:+.1f} dB für die ganze Aufnahme"),
        status="done",
        workflow=RECORDING_WORKFLOW,
        created_at=datetime.now(timezone.utc),
    )
    async with AsyncSessionLocal() as db:
        db.add(song)
        await db.commit()
        await db.refresh(song)
    logger.info(
        "Piano recording %s taken in as song %s — %.1f s, %.1f LUFS, gain %+.1f dB (%s)",
        src.name, song_id, duration, integrated, gain, origin,
    )
    return song


async def warm_analysis(song_id: uuid.UUID, filepath: str) -> None:
    """Analyse the take once in the background, so the first plan the user
    asks for is a file read rather than a few seconds of waiting."""
    try:
        await piano.load_or_analyze_piano(
            settings.storage_dir / filepath,
            beats.cache_path(settings.songs_dir, song_id),
            ffmpeg_path=settings.ffmpeg_path,
        )
    except Exception as exc:
        # Not fatal: the planner analyses on demand and reports a real failure.
        logger.warning("Warm-up analysis of piano song %s failed: %s", song_id, exc)
