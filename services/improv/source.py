"""Which rendition of a source video the improv tool feeds to ffmpeg.

The video tool writes post-processing results as *sibling* files rather than
overwriting the original: a soundtrack muxes into `muxed_filename`, and the
film-grain pass re-encodes into `grain_filename`. Improv historically always
read the clean original (`filepath`), so a clip the user had deliberately
grained came back out of the mux smooth again.

Unlike the Instagram dispatch paths — which resolve one fixed precedence in
`services/instagram/media.py::resolve_video_path` — improv makes this a
*choice*: the grained rendition is the point of the feature, but the clean
original stays one tap away for a clip where the grain fights the piano.

Note that the grain pass reads the muxed file when there is one, so picking
the grained rendition also brings along any attached soundtrack. That is
inherent to how the file is built and not separable here; with
`include_bed` on it lands under the piano as the ambient bed.
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

    Both live in `videos_dir`, which is what `/share/video/` and
    `/api/video/file/` serve out of, so the caller needs no extra routing.
    """
    if use_grain and video.grain_filename:
        return video.grain_filename
    return video.filename


def source_path(video: Video, *, use_grain: bool) -> Path:
    """Absolute path of the rendition to feed ffmpeg.

    Falls back to the original whenever grain was asked for but never
    rendered, so an unchecked box and a missing grain file behave the same.
    """
    if use_grain and video.grain_filename:
        return settings.videos_dir / video.grain_filename
    return settings.storage_dir / video.filepath
