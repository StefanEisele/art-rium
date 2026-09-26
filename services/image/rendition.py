"""Which rendition of an image the rest of the system should use.

The auto-enhance pass (services/image/enhance.py) and the film-grain pass
(services/image/grain.py) each write a *sibling* file rather than overwriting
the original, exactly as the video post-passes do. So every consumer has to
decide which file it means, and the decision must be made in one place or the
answers drift apart.

There are five files at most, in a fixed order:

    original  →  _upscaled  →  _crop  →  _enhanced  →  _grain

Each pass reads the one before it. The order follows from what each pass is
for, and from what it costs:

- **The upscale reads the original, always.** It is a diffusion pass: handing
  it a tone-corrected or grained picture makes the model repaint the correction
  and reinterpret the noise as texture. It wants the rawest pixels there are.
- **The Pillow passes sit on top of it**, because they are cheap enough to be
  re-rendered whenever anything below them moves — milliseconds against the
  upscale's GPU minutes. That asymmetry is the whole reason for this order: the
  expensive pass must never be invalidated by a cheap one. Playing with the
  wand or the grain slider on an upscaled image therefore re-renders *those*
  files at the upscale's resolution and leaves the upscale alone. Only removing
  the upscale itself takes the picture back to its original size.
- **The crop is the first of them**, because it is geometry: the wand should
  measure, and the grain should land on, the framing viewers actually get
  (services/image/crop.py). It is also the only pass that changes the
  picture's *shape* — `delivered_size` is what anything laying the picture
  out has to ask, not `width`/`height`.
- **Grain lands last**, at delivery resolution, and never reads its own output —
  so changing the strength replaces the grain instead of stacking a second
  field onto the first.

The last file that exists is the one viewers get; removing one puts whatever is
under it straight back, because none of them are ever rendered from each
other's output except in that one direction.

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


def is_upscaled(image: Image) -> bool:
    """True when a diffusion-upscaled rendition exists for this image."""
    return bool(image.upscaled_filepath and image.upscaled_filename)


def is_grained(image: Image) -> bool:
    """True when a grained rendition exists for this image."""
    return bool(image.grained_filepath and image.grained_filename)


def is_cropped(image: Image) -> bool:
    """True when a cropped rendition exists for this image."""
    return bool(image.cropped_filepath and image.cropped_filename and image.crop_box)


def primary_filename(image: Image) -> str:
    """The bare filename of the rendition viewers should get.

    Used for `/share/image/<name>` and `/api/image/<name>` URLs, both of which
    look files up by name across the date-sharded tree.
    """
    if is_grained(image):
        return image.grained_filename
    if is_enhanced(image):
        return image.enhanced_filename
    if is_cropped(image):
        return image.cropped_filename
    return image.upscaled_filename if is_upscaled(image) else image.filename


def primary_filepath(image: Image) -> str:
    """Storage-relative path of the viewer-facing rendition."""
    if is_grained(image):
        return image.grained_filepath
    if is_enhanced(image):
        return image.enhanced_filepath
    if is_cropped(image):
        return image.cropped_filepath
    return image.upscaled_filepath if is_upscaled(image) else image.filepath


def delivered_size(image: Image) -> tuple[int | None, int | None]:
    """Pixel size of the rendition viewers get — the shape to lay it out in.

    Not `width`/`height`: those stay the size the picture was *generated* at,
    which is what the recipe and the upscale need. The crop decides the shape
    once there is one; the upscale only the scale. None where a pre-metadata
    row never recorded a size.
    """
    if is_cropped(image) and image.crop_width and image.crop_height:
        return image.crop_width, image.crop_height
    if is_upscaled(image) and image.upscale_width and image.upscale_height:
        return image.upscale_width, image.upscale_height
    return image.width, image.height


def resolve_image_path(image: Image) -> Path:
    """Absolute path of the file that should actually be published for `image`.

    Every Instagram / WordPress dispatch path must resolve through here, or a
    picture the user enhanced gets published in its unenhanced form.
    """
    return settings.storage_dir / primary_filepath(image)


def original_path(image: Image) -> Path:
    """Absolute path of the untouched original.

    The upscale pass reads this unconditionally, and so do the generation
    paths.
    """
    return settings.storage_dir / image.filepath


def enhanced_path(image: Image) -> Path | None:
    """Absolute path of the enhanced rendition, or None if there isn't one."""
    if not is_enhanced(image):
        return None
    return settings.storage_dir / image.enhanced_filepath


def enhance_source_path(image: Image) -> Path:
    """Absolute path of the file the enhance pass must read.

    The crop when there is one, else the upscaled rendition, else the
    original — and never a previous enhancement, so re-running at a new
    strength replaces the correction instead of stacking it. Reading the
    upscale is what keeps the wand from quietly dropping the image back to its
    generated size, and reading the crop is what keeps it from undoing the
    framing.
    """
    return cropped_path(image) or upscaled_path(image) or original_path(image)


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


def upscaled_path(image: Image) -> Path | None:
    """Absolute path of the upscaled rendition, or None if there isn't one."""
    if not is_upscaled(image):
        return None
    return settings.storage_dir / image.upscaled_filepath


def upscale_source_path(image: Image) -> Path:
    """Absolute path of the file the upscale pass must read: the original.

    Always the original, never a derived rendition. Two reasons, and both
    matter: a diffusion model handed a grained picture reinterprets the noise
    as texture and paints it in permanently, and one handed a tone-corrected
    one repaints the correction as if it were the subject. And because this
    pass reads the bottom of the stack, nothing above it can invalidate it —
    which is what lets the wand and the grain be played with freely on an
    image that has already been upscaled.
    """
    return original_path(image)


def grained_path(image: Image) -> Path | None:
    """Absolute path of the grained rendition, or None if there isn't one."""
    if not is_grained(image):
        return None
    return settings.storage_dir / image.grained_filepath


def grain_source_path(image: Image) -> Path:
    """Absolute path of the file the grain pass must read.

    The topmost rendition below it — enhanced, else cropped, else upscaled,
    else the original — but never the grain pass's own output, so changing the
    strength replaces the grain instead of stacking a second field on top of
    the first. Reading the enhanced/upscaled file is what keeps grain at
    delivery resolution rather than leaving a small grained file to be
    stretched later.
    """
    return (enhanced_path(image) or cropped_path(image)
            or upscaled_path(image) or original_path(image))


def cropped_path(image: Image) -> Path | None:
    """Absolute path of the cropped rendition, or None if there isn't one."""
    if not is_cropped(image):
        return None
    return settings.storage_dir / image.cropped_filepath


def crop_source_path(image: Image) -> Path:
    """Absolute path of the file the crop is cut from: the upscale when there
    is one, the original otherwise — never a toned or grained rendition, which
    are rendered from the crop rather than under it."""
    return upscaled_path(image) or original_path(image)


def cropped_rel_path(image: Image) -> tuple[str, str]:
    """(storage-relative path, bare filename) the cropped rendition should use.

    Named off the original like the others, so there is one crop file per
    image however often it is re-framed or the upscale under it changes.
    """
    original = Path(image.filepath)
    name = f"{original.stem}_crop.png"
    return str(original.with_name(name)).replace("\\", "/"), name


def grained_rel_path(image: Image) -> tuple[str, str]:
    """(storage-relative path, bare filename) the grained rendition should use.

    Named off the *original* rather than off the enhanced file it may have been
    rendered from, so an image has exactly one grain file whether or not it is
    also enhanced — otherwise toggling the wand would strand the other one.
    """
    original = Path(image.filepath)
    name = f"{original.stem}_grain.png"
    return str(original.with_name(name)).replace("\\", "/"), name
