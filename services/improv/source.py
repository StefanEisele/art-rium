"""Which rendition of a source video the improv tool feeds to ffmpeg.

The video tool writes post-processing results as *sibling* files rather than
overwriting the original: a soundtrack muxes into `muxed_filename`, a SEEDVR2
pass upscales that into `upscale_filename`, and the film-grain pass re-encodes
into `grain_filename`. Improv historically always read the clean original
(`filepath`), so a clip the user had deliberately grained came back out of the
mux smooth again.

Unlike the Instagram dispatch paths — which resolve one fixed precedence in
`services/instagram/media.py::resolve_video_path` — improv makes *grain* a
choice: it is a look, and the ungrained rendition stays one tap away for a
clip where the grain fights the piano. Resolution is not a look, so the
upscale is never opted out of — declining it would hand ffmpeg the small
generation canvas of a clip the user had explicitly enlarged.

Note that the grain pass reads the upscaled (or muxed) file when there is one,
so picking the grained rendition also brings along the upscale and any
attached soundtrack. That is inherent to how the file is built and not
separable here; with `include_bed` on the audio lands under the piano as the
ambient bed.
"""
from __future__ import annotations

from pathlib import Path

from core.config import settings
from core.models import Video


def has_grain(video: Video) -> bool:
    """Whether a grained rendition exists for this video."""
    return bool(video.grain_filename)


def source_filename(video: Video, *, use_grain: bool) -> str | None:
    """The filename of the rendition to feed the mux / the loop player.

    All of these live in `videos_dir`, which is what `/share/video/` and
    `/api/video/file/` serve out of, so the caller needs no extra routing.
    """
    if use_grain and video.grain_filename:
        return video.grain_filename
    if video.upscale_filename:
        return video.upscale_filename
    return video.filename


def source_path(video: Video, *, use_grain: bool) -> Path:
    """Absolute path of the rendition to feed ffmpeg.

    Falls back down the chain whenever a rendition was asked for but never
    rendered, so an unchecked box and a missing file behave the same.
    """
    if use_grain and video.grain_filename:
        return settings.videos_dir / video.grain_filename
    if video.upscale_filename:
        return settings.videos_dir / video.upscale_filename
    return settings.storage_dir / video.filepath
