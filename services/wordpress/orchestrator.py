"""
WordPress article publisher — Single Source of Truth for generating
EN+DE blog posts and pushing them to WordPress as Polylang language-tagged
posts.

Live flavour (used by the Articles tool):

  generate_modal_article(images, db, *, mode, ...)
    Three modes — essay / work / lab — sharing the EN+DE generation pass
    and per-mode Gutenberg renderers. Voice: prompts/voice-system.md +
    prompts/mode-{mode}.md.

Back-compat flavours (kept for direct callers, no longer routed via the
HTTP API):

  generate_articles_for_images(images, db, *, publish)
    Plain 4-paragraph article-rium voice path. Voice: prompts/voice.md.

  generate_rich_articles_for_series(images, db, *, series_name, ...)
    Older SEO-friendly series article. Voice: prompts/voice-rich.md.

Polylang note: nothing auto-links the EN+DE pair as translations. Polylang
Pro's REST surface only exposes /pll/v1/languages and /settings; linking
is done by hand in the WP admin (one click per post in Polylang's
translation column) or via a future custom mu-plugin endpoint.

Rendering (Gutenberg block builders + per-mode renderers) lives in
services/wordpress/gutenberg.py and services/wordpress/renderers.py; SEO/
taxonomy helpers (featured-alt keyphrase patching, category resolution)
live in services/wordpress/seo.py. This module is the orchestrator: VLM
re-encoding, calling the Ollama article writers, and the WP publish loop.
"""
import html
import logging
import re
import unicodedata
import uuid
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from core.config import settings
from core.imaging import prepare_for_vlm
from core.models import Article, Image, Video
from core.video_thumb import extract_video_frames
from services.ollama.articles import write_article, write_modal_article, write_rich_article
from services.wordpress.client import request_json
from services.wordpress.renderers import (
    _flatten_modal_body_for_search,
    _flatten_rich_body_for_search,
    _render_essay_post,
    _render_lab_post,
    _render_rich_post,
    _render_work_post,
)
from services.wordpress.seo import (
    _MODE_CATEGORY_SLUGS,
    _patch_featured_alt_with_keyphrases,
    _resolve_category_id,
)
from services.wordpress.tags import resolve_tag_ids

logger = logging.getLogger(__name__)


_LANGS = ("en", "de")  # creation order: EN first so DE can reference it later if we add linking
_MODAL_MODES = ("essay", "work", "lab")


def _md_to_html(md: str) -> str:
    """
    Convert paragraph-only Markdown to HTML. The voice guide forbids headings,
    bullets, bold/italic, links — only paragraphs separated by blank lines —
    so this purpose-built converter is enough.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", md) if p.strip()]
    return "\n\n".join(f"<p>{html.escape(p)}</p>" for p in paragraphs)


def _slugify(text: str, lang: str) -> str | None:
    """
    ASCII-fold + kebab-case + lang suffix. Returns None for titles with no
    ASCII content (e.g. Chinese), so the caller can omit the field and let
    WordPress auto-generate a unicode slug from the title.
    """
    folded = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", folded).strip("-").lower()
    if not slug:
        return None
    return f"{slug}-{lang}"


# ── Shared pre-/post-processing for both article writers ────────────────────


async def _encode_for_vlm(images: list[Image]) -> list[bytes]:
    """Re-encode each image's source to a VLM-sized JPG (parallel to *images*).
    Raises FileNotFoundError if any source is missing on disk."""
    jpgs: list[bytes] = []
    for img in images:
        src = settings.storage_dir / img.filepath
        if not src.exists():
            raise FileNotFoundError(f"Source image missing on disk: {src}")
        vlm_jpg, _ = await prepare_for_vlm(src)
        jpgs.append(vlm_jpg)
    return jpgs


def _meta_hints_for(
    images: list[Image],
) -> tuple[list[str | None], list[str | None], list[str | None]]:
    """Pull (title_hint, alt_text, notes) lists from images, parallel to *images*."""
    return (
        [img.title or img.wp_seo_title for img in images],
        [img.wp_alt_text for img in images],
        [img.notes for img in images],
    )


async def _publish_translation(
    db: AsyncSession,
    *,
    lang: str,
    block: dict,
    body_html: str,
    body_md_for_db: str,
    featured_media: int,
    translation_group_id: uuid.UUID,
    image_ids: list[uuid.UUID],
    publish: bool,
    yoast_meta: dict[str, str] | None = None,
    extra_result_fields: dict[str, Any] | None = None,
    category_id: int | None = None,
    tag_ids: list[int] | None = None,
) -> dict:
    """POST one language to WP + queue the Article row + return the result dict.

    The caller owns db.commit() — we only db.add() so the whole translation
    set commits in a single tx. When *category_id* is set, the post is
    assigned to that taxonomy term — required for the new permalink
    structure /%category%/%postname%/. When *tag_ids* is set, the post is
    attached to those tag terms (resolved upstream via tags.resolve_tag_ids).
    """
    slug = _slugify(block["title"], lang)
    target_status = "publish" if publish else "draft"

    post_payload: dict[str, Any] = {
        "title":          block["title"],
        "content":        body_html,
        "excerpt":        block["excerpt"],
        "status":         target_status,
        "featured_media": featured_media,
    }
    if slug:
        post_payload["slug"] = slug
    if yoast_meta:
        post_payload["meta"] = yoast_meta
    if category_id is not None:
        post_payload["categories"] = [category_id]
    if tag_ids:
        post_payload["tags"] = tag_ids

    post = await request_json("POST", "/wp/v2/posts", json=post_payload, params={"lang": lang})
    wp_post_id = post["id"]
    wp_link    = post.get("link", "")

    article = Article(
        title=block["title"],
        body_md=body_md_for_db,
        excerpt=block["excerpt"],
        tags=block["tags"] or None,
        language=lang,
        translation_group_id=translation_group_id,
        wp_post_id=wp_post_id,
        wp_link=wp_link,
        status="published" if publish else "draft",
        image_ids=image_ids,
    )
    db.add(article)

    result: dict[str, Any] = {
        "language":   lang,
        "wp_post_id": wp_post_id,
        "wp_link":    wp_link,
        "status":     article.status,
        "title":      block["title"],
        "excerpt":    block["excerpt"],
        "tags":       block["tags"],
    }
    if extra_result_fields:
        result.update(extra_result_fields)
    return result


async def generate_articles_for_images(
    images: list[Image],
    db: AsyncSession,
    *,
    publish: bool = False,
) -> dict:
    """
    Generate DE/EN/ZH articles for one or more *images* (a series) and push
    them to WordPress.

    Args:
      images:   1+ Image rows, each already uploaded to WP (wp_media_id set).
                The first image's wp_media_id becomes the WP featured_media.
      db:       async session, committed once at the end
      publish:  if True, posts go up as 'publish'; otherwise 'draft'

    Returns:
      {
        "translation_group_id": str,
        "image_ids": [str, ...],
        "articles": [
          {"language": "en", "wp_post_id": int, "wp_link": str, "status": str,
           "title": str, "excerpt": str, "tags": [...]},
          ...
        ],
      }
    """
    if not images:
        raise ValueError("generate_articles_for_images requires at least one image")

    for img in images:
        if not img.wp_media_id:
            raise ValueError(
                f"Image {img.id} has no wp_media_id — upload it via /api/wordpress/media/upload first."
            )

    image_ids = [img.id for img in images]
    series_label = f"series of {len(images)}" if len(images) > 1 else "image"
    logger.info("Article gen %s — re-encoding %d image(s) for VLM", series_label, len(images))

    jpgs = await _encode_for_vlm(images)
    title_hints, alt_texts, notes_list = _meta_hints_for(images)

    logger.info(
        "Article gen %s — calling %s (%d image(s), %dKB total, edge %d)",
        series_label, settings.ollama_llm_model,
        len(images), sum(len(j) for j in jpgs) // 1024, settings.vlm_analysis_max_edge,
    )
    bodies = await write_article(
        jpgs,
        title_hints=title_hints,
        alt_texts=alt_texts,
        notes_list=notes_list,
    )

    translation_group_id = uuid.uuid4()
    target_status = "publish" if publish else "draft"
    featured_media = images[0].wp_media_id
    results: list[dict] = []

    for lang in _LANGS:
        block = bodies[lang]
        logger.info(
            "Article gen %s — POST /wp/v2/posts?lang=%s (status=%s, slug=%s)",
            series_label, lang, target_status, _slugify(block["title"], lang),
        )
        result = await _publish_translation(
            db,
            lang=lang,
            block=block,
            body_html=_md_to_html(block["body_md"]),
            body_md_for_db=block["body_md"],
            featured_media=featured_media,
            translation_group_id=translation_group_id,
            image_ids=image_ids,
            publish=publish,
        )
        results.append(result)

    await db.commit()
    logger.info(
        "Article gen %s done — group=%s, %s post(s) %s",
        series_label, translation_group_id, len(results), target_status,
    )

    return {
        "translation_group_id": str(translation_group_id),
        "image_ids":            [str(i) for i in image_ids],
        "articles":             results,
    }


async def generate_rich_articles_for_series(
    images: list[Image],
    db: AsyncSession,
    *,
    series_name: str | None = None,
    parent_series: dict[str, str] | None = None,  # {"name": str, "url": str}
    singulart_links: list[dict] | None = None,    # [{"title": str, "url": str, "thumbnail_url": str}, ...]
    user_notes: str | None = None,
    artist_mode: str = "third_person",            # "first_person" | "third_person"
    languages: list[str] | None = None,           # subset of ("de","en","zh"); None = all three
    publish: bool = False,
) -> dict:
    """
    Generate DE/EN/ZH rich, SEO-friendly series articles for *images* and push
    them to WordPress as multi-section Gutenberg posts.

    Required: every image has wp_media_id AND wp_source_url set (the gallery
    blocks embed the source URLs). Re-running /media/upload after a fresh DB
    populates these.

    The first image's wp_media_id becomes the WP featured_media. All images
    are embedded as wp:gallery blocks inside the body.

    Returns: {translation_group_id, image_ids, articles[...]} same shape as
    generate_articles_for_images, with a 'kind': 'rich' marker.
    """
    if not images:
        raise ValueError("generate_rich_articles_for_series requires at least one image")

    for img in images:
        if not img.wp_media_id:
            raise ValueError(
                f"Image {img.id} has no wp_media_id — upload via /api/wordpress/media/upload first."
            )
        if not img.wp_source_url:
            raise ValueError(
                f"Image {img.id} has no wp_source_url — gallery rendering needs the WP source URL."
            )

    if parent_series is not None and not (parent_series.get("name") and parent_series.get("url")):
        raise ValueError("parent_series must have both 'name' and 'url' keys")
    if singulart_links:
        for i, link in enumerate(singulart_links):
            if not (link.get("title") and link.get("url") and link.get("thumbnail_url")):
                raise ValueError(
                    f"singulart_links[{i}] must have 'title', 'url', and 'thumbnail_url' keys"
                )

    image_ids = [img.id for img in images]
    logger.info(
        "Rich article gen — %d image(s), series=%s, parent=%s, singulart=%d",
        len(images), series_name or "(none)",
        parent_series["name"] if parent_series else "(none)",
        len(singulart_links or []),
    )

    jpgs = await _encode_for_vlm(images)
    title_hints, alt_texts, notes_list = _meta_hints_for(images)

    logger.info(
        "Rich article gen — calling %s (%d image(s), %dKB total)",
        settings.ollama_llm_model, len(images), sum(len(j) for j in jpgs) // 1024,
    )
    bodies = await write_rich_article(
        jpgs,
        series_name=series_name,
        parent_series=parent_series,
        has_singulart=bool(singulart_links),
        title_hints=title_hints,
        alt_texts=alt_texts,
        notes_list=notes_list,
        user_notes=user_notes,
        artist_mode=artist_mode,
    )

    translation_group_id = uuid.uuid4()
    target_status = "publish" if publish else "draft"
    featured_media = images[0].wp_media_id
    results: list[dict] = []

    selected_langs = tuple(lang for lang in _LANGS if not languages or lang in languages)
    if not selected_langs:
        raise ValueError(f"No supported languages selected — got {languages!r}, supported {_LANGS}")

    for lang in selected_langs:
        block = bodies[lang]
        body_html = _render_rich_post(block, lang, images, singulart_links, parent_series)

        yoast_meta: dict[str, str] = {}
        if block.get("meta_description"):
            yoast_meta["_yoast_wpseo_metadesc"] = block["meta_description"]
        if block.get("focus_keyphrase"):
            yoast_meta["_yoast_wpseo_focuskw"] = block["focus_keyphrase"]

        logger.info(
            "Rich article — POST /wp/v2/posts?lang=%s (status=%s, slug=%s, %d blocks, yoast_meta=%s)",
            lang, target_status, _slugify(block["title"], lang), body_html.count("<!-- wp:"),
            ",".join(sorted(k.replace("_yoast_wpseo_", "") for k in yoast_meta)) or "(none)",
        )
        result = await _publish_translation(
            db,
            lang=lang,
            block=block,
            body_html=body_html,
            body_md_for_db=_flatten_rich_body_for_search(block, parent_series),
            featured_media=featured_media,
            translation_group_id=translation_group_id,
            image_ids=image_ids,
            publish=publish,
            yoast_meta=yoast_meta or None,
            extra_result_fields={
                "meta_description": block.get("meta_description") or "",
                "focus_keyphrase":  block.get("focus_keyphrase") or "",
            },
        )
        results.append(result)

    await db.commit()
    logger.info(
        "Rich article gen done — group=%s, %d post(s) %s",
        translation_group_id, len(results), target_status,
    )

    return {
        "translation_group_id": str(translation_group_id),
        "image_ids":            [str(i) for i in image_ids],
        "kind":                 "rich",
        "articles":             results,
    }


# ── Video-prep helpers for the modal article LLM prompt ─────────────────────


def _video_kind_label(video: Video) -> str:
    """Human-readable kind label for the LLM prompt.

    Animate videos (workflow i2v_multi / flf2v) get described as 'animate clip';
    Improv mixes get described by their mix kind. Anything else falls back to
    a generic label."""
    wf = video.workflow or ""
    if wf in ("i2v_multi", "flf2v"):
        return "animate clip"
    if wf == "improv_synth":
        return "piano improvisation (synth mix)"
    if wf == "improv_hands":
        return "piano improvisation (hands mix)"
    if wf == "improv_pip":
        return "piano improvisation (PiP mix)"
    return "video"


def _video_descriptions_for_prompt(videos: list[Video] | None) -> list[str] | None:
    """One-line description per video, indexed in placeholder order ([VIDEO_1]
    refers to videos[0]). Returns None when no videos are provided."""
    if not videos:
        return None
    out: list[str] = []
    for i, v in enumerate(videos, 1):
        kind = _video_kind_label(v)
        prompt = (v.prompt or "").strip().replace("\n", " ")
        if len(prompt) > 200:
            prompt = prompt[:200].rstrip() + "…"
        desc = f"[VIDEO_{i}] — {kind}"
        if prompt:
            desc += f"; prompt: {prompt}"
        out.append(desc)
    return out


async def _extract_video_frame_samples(
    videos: list[Video] | None,
) -> list[list[bytes]] | None:
    """Pull a small ordered set of JPG frames per video so the VL article
    writer can actually see what each clip looks like (not just the prompt).

    Frame budget: 3 per video for ≤4 videos, otherwise 12//N (so 6 videos
    get 2 each). Keeps the model's vision-token budget bounded under the
    qwen3.6:27b 16k context window.
    """
    if not videos:
        return None
    n = len(videos)
    per_video = 3 if n <= 4 else max(1, 12 // n)

    frames_per_video: list[list[bytes]] = []
    for v in videos:
        if not v.filepath:
            frames_per_video.append([])
            continue
        src = settings.storage_dir / v.filepath
        if not src.exists():
            logger.warning("Video %s source missing on disk: %s — no frames extracted", v.id, src)
            frames_per_video.append([])
            continue
        frames = await extract_video_frames(src, count=per_video, max_edge=384)
        frames_per_video.append(frames)

    total = sum(len(f) for f in frames_per_video)
    logger.info(
        "Extracted %d sample frame(s) across %d video(s) (%d frames/video budget)",
        total, n, per_video,
    )
    return frames_per_video


async def generate_modal_article(
    images: list[Image],
    db: AsyncSession,
    *,
    mode: str,                                                   # "essay" | "work" | "lab"
    series_name: str | None = None,
    parent_series: dict[str, str] | None = None,                  # Work only
    singulart_links: list[dict] | None = None,                    # Work only
    user_notes: str | None = None,
    artist_mode: str = "third_person",                            # Work only
    languages: list[str] | None = None,                           # subset of ("en","de"); None = both
    videos: list[Video] | None = None,                            # uploaded to YouTube, embedded via [VIDEO_K]
    publish: bool = False,
) -> dict:
    """
    Generate an EN+DE modal article in *mode* (essay / work / lab) and push
    each language to WordPress.

    Each image must have wp_media_id and wp_source_url set (the gallery blocks
    embed the source URLs). Re-running /media/upload after a fresh DB populates
    these. The first image's wp_media_id becomes the WP featured_media.

    Returns: {translation_group_id, image_ids, mode, articles[...]} where each
    article dict carries language, wp_post_id, wp_link, status, title, excerpt,
    tags, meta_description, focus_keyphrase, og_image_idea.
    """
    if mode not in _MODAL_MODES:
        raise ValueError(f"generate_modal_article: mode must be one of {_MODAL_MODES}, got {mode!r}")
    if not images:
        raise ValueError("generate_modal_article requires at least one image")

    for img in images:
        if not img.wp_media_id:
            raise ValueError(
                f"Image {img.id} has no wp_media_id — upload via /api/wordpress/media/upload first."
            )
        if not img.wp_source_url:
            raise ValueError(
                f"Image {img.id} has no wp_source_url — gallery rendering needs the WP source URL."
            )

    # Per-mode input validation: Work is the only mode that uses series/parent/singulart.
    if mode != "work":
        if parent_series:
            logger.warning("Mode %s does not use parent_series; ignoring.", mode)
            parent_series = None
        if singulart_links:
            logger.warning("Mode %s does not use singulart_links; ignoring.", mode)
            singulart_links = None

    if parent_series is not None and not (parent_series.get("name") and parent_series.get("url")):
        raise ValueError("parent_series must have both 'name' and 'url' keys")
    if singulart_links:
        for i, link in enumerate(singulart_links):
            if not (link.get("title") and link.get("url") and link.get("thumbnail_url")):
                raise ValueError(
                    f"singulart_links[{i}] must have 'title', 'url', and 'thumbnail_url' keys"
                )

    image_ids = [img.id for img in images]
    logger.info(
        "Modal article gen — mode=%s, %d image(s), series=%s, parent=%s, singulart=%d",
        mode, len(images), series_name or "(none)",
        parent_series["name"] if parent_series else "(none)",
        len(singulart_links or []),
    )

    jpgs = await _encode_for_vlm(images)
    title_hints, alt_texts, notes_list = _meta_hints_for(images)
    video_descriptions = _video_descriptions_for_prompt(videos)
    video_frame_jpgs = await _extract_video_frame_samples(videos)

    bodies = await write_modal_article(
        jpgs,
        mode=mode,
        series_name=series_name,
        parent_series=parent_series,
        has_singulart=bool(singulart_links),
        title_hints=title_hints,
        alt_texts=alt_texts,
        notes_list=notes_list,
        user_notes=user_notes,
        artist_mode=artist_mode,
        video_descriptions=video_descriptions,
        video_frame_jpgs=video_frame_jpgs,
    )

    translation_group_id = uuid.uuid4()
    target_status = "publish" if publish else "draft"
    featured_media = images[0].wp_media_id
    results: list[dict] = []

    selected_langs = tuple(lang for lang in _LANGS if not languages or lang in languages)
    if not selected_langs:
        raise ValueError(f"No supported languages selected — got {languages!r}, supported {_LANGS}")

    # Resolve the WP category ID for each (mode, lang) once. Fail fast (before
    # we POST anything to WP) so a missing category surfaces as a job error
    # rather than a half-published translation pair.
    category_ids: dict[str, int] = {}
    for lang in selected_langs:
        slug = _MODE_CATEGORY_SLUGS.get((mode, lang))
        if slug is None:
            raise RuntimeError(f"No category slug mapped for mode={mode!r} lang={lang!r}")
        category_ids[lang] = await _resolve_category_id(slug)

    # Patch the featured image's alt_text to weave in the focus keyphrases
    # so Yoast's "Keyphrase in image alt attributes" check passes for both
    # languages. Only the featured image is patched — Yoast accepts one
    # qualifying image per post, and patching every gallery item would
    # ripple unwanted alt-text changes through other posts that reuse them.
    await _patch_featured_alt_with_keyphrases(
        images[0], db, bodies=bodies, selected_langs=selected_langs,
    )

    for lang in selected_langs:
        block = bodies[lang]
        if mode == "essay":
            body_html = _render_essay_post(block, lang, images, videos)
        elif mode == "work":
            body_html = _render_work_post(block, lang, images, singulart_links, parent_series, videos)
        else:  # lab
            body_html = _render_lab_post(block, lang, images, videos)

        yoast_meta: dict[str, str] = {}
        if block.get("meta_description"):
            yoast_meta["_yoast_wpseo_metadesc"] = block["meta_description"]
        if block.get("focus_keyphrase"):
            yoast_meta["_yoast_wpseo_focuskw"] = block["focus_keyphrase"]

        tag_ids = await resolve_tag_ids(block.get("tags") or [], lang)

        logger.info(
            "Modal article — POST /wp/v2/posts?lang=%s (mode=%s, status=%s, slug=%s, category=%d, tags=%d, %d blocks)",
            lang, mode, target_status, _slugify(block["title"], lang),
            category_ids[lang], len(tag_ids),
            body_html.count("<!-- wp:"),
        )
        result = await _publish_translation(
            db,
            lang=lang,
            block=block,
            body_html=body_html,
            body_md_for_db=_flatten_modal_body_for_search(block, mode),
            featured_media=featured_media,
            category_id=category_ids[lang],
            tag_ids=tag_ids or None,
            translation_group_id=translation_group_id,
            image_ids=image_ids,
            publish=publish,
            yoast_meta=yoast_meta or None,
            extra_result_fields={
                "mode":             mode,
                "meta_description": block.get("meta_description") or "",
                "focus_keyphrase":  block.get("focus_keyphrase") or "",
                "og_image_idea":    block.get("og_image_idea") or "",
                "wp_tag_ids":       tag_ids,
            },
        )
        results.append(result)

    await db.commit()
    logger.info(
        "Modal article gen done — mode=%s, group=%s, %d post(s) %s",
        mode, translation_group_id, len(results), target_status,
    )

    return {
        "translation_group_id": str(translation_group_id),
        "image_ids":            [str(i) for i in image_ids],
        "mode":                 mode,
        "kind":                 "modal",
        "articles":             results,
    }
