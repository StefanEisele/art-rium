"""Which rendition of an image the rest of the system should use.

The auto-enhance pass (services/image/enhance.py) writes a *sibling* file
rather than overwriting the original, exactly as the video post-passes do. So
every consumer has to decide which file it means, and the decision must be
made in one place or the answers drift apart.

The rule, chosen deliberately:

- **Publishing uses the enhanced rendition.** Instagram, WordPress and the
  article pipeline are showing the picture to people, and the enhanced file is
  what the user signed off on when they tapped the wand. This mirrors
  `services/instagram/media.py::resolve_video_path`, where the grained render
  wins for the same reason — keep the two in step.

- **Generation reads the original.** i2v / FLF2V / story-frames feed the image
  to a model as a starting point, not to a viewer: handing them a
  contrast-stretched, vibrance-lifted frame means the model animates the
  correction too. The titler VLM likewise judges the raw render. Those paths
  keep reading `Image.filepath` directly and deliberately do not come through
  here.

Both files live under `images_dir` (date-sharded), which is what
`/share/image/` and `/api/image/` resolve by bare filename, so callers need no
extra routing for either.
"""
from __future__ import annotations

from pathlib import Path

from core.config import settings
from core.models import Image


def is_enhanced(image: Image) -> bool:
    """True when an enhanced rendition exists for this image."""
    return bool(image.enhanced_filepath and image.enhanced_filename)


def primary_filename(image: Image) -> str:
    """The bare filename of the rendition viewers should get.

    Used for `/share/image/<name>` and `/api/image/<name>` URLs, both of which
    look files up by name across the date-sharded tree.
    """
    return image.enhanced_filename if is_enhanced(image) else image.filename


def primary_filepath(image: Image) -> str:
    """Storage-relative path of the viewer-facing rendition."""
    return image.enhanced_filepath if is_enhanced(image) else image.filepath


def resolve_image_path(image: Image) -> Path:
    """Absolute path of the file that should actually be published for `image`.

    Every Instagram / WordPress dispatch path must resolve through here, or a
    picture the user enhanced gets published in its unenhanced form.
    """
    return settings.storage_dir / primary_filepath(image)


def original_path(image: Image) -> Path:
    """Absolute path of the untouched original.

    The enhance pass itself reads this (never its own output, so strength
    changes replace rather than stack), and so do the generation paths.
    """
    return settings.storage_dir / image.filepath


def enhanced_path(image: Image) -> Path | None:
    """Absolute path of the enhanced rendition, or None if there isn't one."""
    if not is_enhanced(image):
        return None
    return settings.storage_dir / image.enhanced_filepath


def enhanced_rel_path(image: Image) -> tuple[str, str]:
    """(storage-relative path, bare filename) the enhanced rendition should use.

    Deterministic from the original's name, so re-running the pass overwrites
    the previous rendition instead of littering the directory with one file per
    attempt. The suffix stays inside `_SAFE_FILENAME_RE`'s charset in
    routers/generate.py, without which `/share/image/` would refuse to serve
    it to Instagram.
    """
    original = Path(image.filepath)
    name = f"{original.stem}_enhanced.png"
    return str(original.with_name(name)).replace("\\", "/"), name
