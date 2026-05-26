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
"""
import html
import json
import logging
import re
import unicodedata
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from core.config import settings
from core.imaging import prepare_for_vlm
from core.models import Article, Image, Video
from core.video_thumb import extract_video_frames
from services.ollama.client import write_article, write_modal_article, write_rich_article
from services.wordpress.client import request_json
from services.wordpress.tags import resolve_tag_ids

logger = logging.getLogger(__name__)


_LANGS = ("en", "de")  # creation order: EN first so DE can reference it later if we add linking


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


# ───────────────────────────────────────────────────────────────────────────
# Rich-article rendering — Gutenberg block stitching for SEO series posts
# ───────────────────────────────────────────────────────────────────────────

_HEADINGS = {
    # Rich-article (back-compat) + Work-mode shared
    "concept":            {"en": "The Concept",                "de": "Das Konzept"},
    "visual_language":    {"en": "Visual Language",            "de": "Visuelle Sprache"},
    "technical_approach": {"en": "Technical Approach",         "de": "Technischer Ansatz"},
    "available_works":    {"en": "Available Works",            "de": "Verfügbare Werke"},
    "larger_practice":    {"en": "Part of a Larger Practice",  "de": "Teil einer größeren Praxis"},
    # Lab-mode
    "problem":            {"en": "Problem",                    "de": "Das Problem"},
    "approach":           {"en": "Approach",                   "de": "Ansatz"},
    "steps":              {"en": "Steps",                      "de": "Schritte"},
    "result":             {"en": "Result",                     "de": "Ergebnis"},
    "why":                {"en": "Why this matters",           "de": "Warum das zählt"},
    "tool_stack":         {"en": "Tool stack",                 "de": "Werkzeuge"},
}

_METADATA_LABELS = {
    "year":       {"en": "Year",       "de": "Jahr"},
    "medium":     {"en": "Medium",     "de": "Medium"},
    "dimensions": {"en": "Dimensions", "de": "Format"},
    "edition":    {"en": "Edition",    "de": "Edition"},
    "status":     {"en": "Status",     "de": "Status"},
}

_SINGULART_CAPTION = {
    "en": "Available on Singulart",
    "de": "Erhältlich auf Singulart",
}

# {insta} and {site} are href URLs (already escaped); {site_label} is the displayed text.
_FOOTER_TEMPLATE = {
    "en": 'Follow the development of this and other series on <a href="{insta}">Instagram</a> or explore further work at <a href="{site}">{site_label}</a>',
    "de": 'Folgen Sie der Entwicklung dieser und anderer Serien auf <a href="{insta}">Instagram</a> oder entdecken Sie weitere Arbeiten auf <a href="{site}">{site_label}</a>',
}


def _attr_url(url: str) -> str:
    """Escape a URL for safe inclusion in an HTML attribute (href/src)."""
    return html.escape(url or "", quote=True)


def _attr(text: str) -> str:
    """Escape arbitrary text for inclusion in an HTML attribute."""
    return html.escape(text or "", quote=True)


def _para_block(text_html: str) -> str:
    """Render a wp:paragraph block. *text_html* is already-rendered inline HTML."""
    return f"<!-- wp:paragraph -->\n<p>{text_html}</p>\n<!-- /wp:paragraph -->"


def _h2_block(text: str) -> str:
    return f'<!-- wp:heading -->\n<h2 class="wp-block-heading">{html.escape(text)}</h2>\n<!-- /wp:heading -->'


def _h3_block(text: str) -> str:
    return f'<!-- wp:heading {{"level":3}} -->\n<h3 class="wp-block-heading">{html.escape(text)}</h3>\n<!-- /wp:heading -->'


def _separator_block() -> str:
    return '<!-- wp:separator -->\n<hr class="wp-block-separator has-alpha-channel-opacity"/>\n<!-- /wp:separator -->'


def _bullet_list_block(items: list[str]) -> str:
    list_items = "\n\n".join(
        f"<!-- wp:list-item -->\n<li>{html.escape(it)}</li>\n<!-- /wp:list-item -->"
        for it in items
    )
    return (
        '<!-- wp:list -->\n'
        '<ul class="wp-block-list">'
        f'{list_items}'
        '</ul>\n'
        '<!-- /wp:list -->'
    )


def _gallery_block(images: list[Image]) -> str:
    """Render a wp:gallery containing one wp:image per image. Caller decides which images go in."""
    if not images:
        return ""
    image_blocks: list[str] = []
    for img in images:
        attrs = json.dumps(
            {"lightbox": {"enabled": True}, "id": img.wp_media_id, "sizeSlug": "large", "linkDestination": "none"},
            separators=(",", ":"),
        )
        alt = _attr(img.wp_alt_text or "")
        src = _attr_url(img.wp_source_url or "")
        image_blocks.append(
            f"<!-- wp:image {attrs} -->\n"
            f'<figure class="wp-block-image size-large">'
            f'<img src="{src}" alt="{alt}" class="wp-image-{img.wp_media_id}"/>'
            f"</figure>\n"
            f"<!-- /wp:image -->"
        )
    inner = "\n\n".join(image_blocks)
    return (
        '<!-- wp:gallery {"linkTo":"none"} -->\n'
        '<figure class="wp-block-gallery has-nested-images columns-default is-cropped">'
        f"{inner}"
        '</figure>\n'
        '<!-- /wp:gallery -->'
    )


def _singulart_image_block(link: dict, caption: str) -> str:
    """Render a wp:image block linked to a Singulart product page."""
    title = link.get("title") or ""
    url   = link.get("url") or ""
    thumb = link.get("thumbnail_url") or ""
    attrs = json.dumps({"sizeSlug": "large", "linkDestination": "custom"}, separators=(",", ":"))
    return (
        f"<!-- wp:image {attrs} -->\n"
        f'<figure class="wp-block-image size-large">'
        f'<a href="{_attr_url(url)}"><img src="{_attr_url(thumb)}" alt="{_attr(title)}"/></a>'
        f'<figcaption class="wp-element-caption"><a href="{_attr_url(url)}">{html.escape(caption)}</a></figcaption>'
        f"</figure>\n"
        f"<!-- /wp:image -->"
    )


def _split_for_galleries(images: list[Image]) -> tuple[list[Image], list[Image]]:
    """N=1 → no galleries. N=2–3 → one after concept. N≥4 → split between concept and visual_language."""
    n = len(images)
    if n <= 1:
        return ([], [])
    if n <= 3:
        return (images, [])
    half = (n + 1) // 2
    return (images[:half], images[half:])


_VIDEO_PLACEHOLDER_RE = re.compile(r"^\s*\[VIDEO_(\d+)\]\s*$")


# WordPress core ships CSS for exactly these aspect-ratio embed classes
# (see wp-includes/blocks/embed/style.css). Anything outside this list
# renders as the default 16:9 box, so we snap to the nearest supported
# ratio rather than emit a custom class.
_WP_ASPECT_RATIOS: list[tuple[float, str]] = [
    (21 / 9, "wp-embed-aspect-21-9"),
    (18 / 9, "wp-embed-aspect-18-9"),
    (16 / 9, "wp-embed-aspect-16-9"),
    (4 / 3,  "wp-embed-aspect-4-3"),
    (1.0,    "wp-embed-aspect-1-1"),
    (9 / 16, "wp-embed-aspect-9-16"),
    (1 / 2,  "wp-embed-aspect-1-2"),
]


def _aspect_class(width: int | None, height: int | None) -> str:
    """Pick the closest WordPress-core embed aspect class for the video's
    actual dimensions. Falls back to 16:9 when width/height is unknown."""
    if not width or not height:
        return "wp-embed-aspect-16-9"
    ratio = width / height
    # Compare on log scale so distance between e.g. 4:3 and 1:1 is symmetric.
    import math
    target = math.log(ratio)
    return min(_WP_ASPECT_RATIOS, key=lambda kv: abs(math.log(kv[0]) - target))[1]


def _youtube_embed_block(url: str, width: int | None = None, height: int | None = None) -> str:
    """Render a Gutenberg wp:embed block for a YouTube URL. The block stays
    in sync with what the Block Editor emits when a user pastes a YouTube
    link, so it shows up as a proper Embed block in the WP admin.

    The aspect-ratio class is picked from the video's actual dimensions so
    square Animate clips and vertical Improv recordings don't get squashed
    into a 16:9 letterbox."""
    safe_url = _attr_url(url)
    aspect = _aspect_class(width, height)
    class_attr = f"{aspect} wp-has-aspect-ratio"
    attrs = json.dumps(
        {
            "url":              url,
            "type":             "video",
            "providerNameSlug": "youtube",
            "responsive":       True,
            "className":        class_attr,
        },
        separators=(",", ":"),
    )
    return (
        f"<!-- wp:embed {attrs} -->\n"
        f'<figure class="wp-block-embed is-type-video is-provider-youtube wp-block-embed-youtube {class_attr}">'
        f'<div class="wp-block-embed__wrapper">\n{safe_url}\n</div></figure>\n'
        f"<!-- /wp:embed -->"
    )


def _video_placeholder_index(paragraph: str) -> int | None:
    """If *paragraph* is exactly a [VIDEO_K] token (possibly with whitespace),
    return the 1-based index K. Otherwise None."""
    m = _VIDEO_PLACEHOLDER_RE.match(paragraph or "")
    return int(m.group(1)) if m else None


def _maybe_video_block(paragraph: str, videos: list[Video] | None, consumed: set[int]) -> str | None:
    """If *paragraph* is a [VIDEO_K] placeholder and videos[K-1] has a YouTube
    URL, return the rendered embed block (and mark K as consumed). Otherwise
    return None — caller should render the paragraph normally."""
    if not videos:
        return None
    idx = _video_placeholder_index(paragraph)
    if idx is None or not (1 <= idx <= len(videos)):
        return None
    video = videos[idx - 1]
    if not video.youtube_url:
        logger.warning("Video placeholder [VIDEO_%d] but video %s has no youtube_url", idx, video.id)
        return None
    consumed.add(idx)
    return _youtube_embed_block(video.youtube_url, video.width, video.height)


def _trailing_video_blocks(videos: list[Video] | None, consumed: set[int]) -> list[str]:
    """Return embed blocks for any videos the LLM forgot to reference in
    placeholders. Renders them in input order before the footer so the user
    still sees what they picked."""
    if not videos:
        return []
    out: list[str] = []
    for i, video in enumerate(videos, 1):
        if i in consumed or not video.youtube_url:
            continue
        logger.info("Appending unreferenced video [VIDEO_%d] %s to article tail", i, video.id)
        out.append(_youtube_embed_block(video.youtube_url, video.width, video.height))
    return out


def _render_parent_placeholder(text: str, parent_series: dict[str, str]) -> str:
    """Substitute the literal [PARENT_SERIES] token with an anchor wrapping the parent name.
    *text* must already be HTML-escaped."""
    if "[PARENT_SERIES]" not in text:
        return text
    name = html.escape(parent_series["name"])
    url  = _attr_url(parent_series["url"])
    return text.replace("[PARENT_SERIES]", f'<a href="{url}">{name}</a>')


def _render_rich_post(
    slots: dict,
    lang: str,
    images: list[Image],
    singulart_links: list[dict] | None,
    parent_series: dict[str, str] | None,
) -> str:
    """Stitch the LLM slot prose, heading translations, gallery blocks, conditional
    Singulart product cards, and conditional parent-series link into a Gutenberg post body."""
    blocks: list[str] = []

    blocks.append(_para_block(html.escape(slots["intro"])))

    blocks.append(_h2_block(_HEADINGS["concept"][lang]))
    for p in slots["concept"]:
        blocks.append(_para_block(html.escape(p)))

    gallery_a, gallery_b = _split_for_galleries(images)
    if gallery_a:
        blocks.append(_gallery_block(gallery_a))

    blocks.append(_h2_block(_HEADINGS["visual_language"][lang]))
    for p in slots["visual_language"]:
        blocks.append(_para_block(html.escape(p)))

    if gallery_b:
        blocks.append(_gallery_block(gallery_b))

    blocks.append(_h2_block(_HEADINGS["technical_approach"][lang]))
    ta = slots["technical_approach"]
    blocks.append(_para_block(html.escape(ta["intro"])))
    if ta["steps"]:
        blocks.append(_bullet_list_block(ta["steps"]))
    if ta.get("outro"):
        blocks.append(_para_block(html.escape(ta["outro"])))

    if singulart_links:
        blocks.append(_h2_block(_HEADINGS["available_works"][lang]))
        works_intro = slots.get("available_works_intro") or ""
        if works_intro:
            blocks.append(_para_block(html.escape(works_intro)))
        for link in singulart_links:
            blocks.append(_h3_block(link.get("title") or ""))
            blocks.append(_singulart_image_block(link, _SINGULART_CAPTION[lang]))

    if parent_series and slots.get("larger_practice"):
        blocks.append(_h2_block(_HEADINGS["larger_practice"][lang]))
        for p in slots["larger_practice"]:
            escaped = html.escape(p)
            blocks.append(_para_block(_render_parent_placeholder(escaped, parent_series)))

    blocks.append(_separator_block())
    site_label = (settings.artist_website_url or "")
    site_label = re.sub(r"^https?://", "", site_label).rstrip("/")
    footer_text = _FOOTER_TEMPLATE[lang].format(
        insta=_attr_url(settings.artist_instagram_url),
        site=_attr_url(settings.artist_website_url),
        site_label=html.escape(site_label),
    )
    blocks.append(_para_block(f"<em>{footer_text}</em>"))

    return "\n\n".join(blocks)


def _flatten_rich_body_for_search(slots: dict, parent_series: dict[str, str] | None) -> str:
    """Flatten LLM slot prose into a single paragraph-separated string for Article.body_md.
    Used for full-text search and regeneration debugging — NOT shown to readers (WP holds the HTML)."""
    parts: list[str] = [slots["intro"]]
    parts.extend(slots["concept"])
    parts.extend(slots["visual_language"])
    ta = slots["technical_approach"]
    parts.append(ta["intro"])
    parts.extend(f"- {s}" for s in ta["steps"])
    if ta.get("outro"):
        parts.append(ta["outro"])
    if parent_series and slots.get("larger_practice"):
        parts.extend(slots["larger_practice"])
    return "\n\n".join(p for p in parts if p)


# ───────────────────────────────────────────────────────────────────────────
# Rich-article orchestrator
# ───────────────────────────────────────────────────────────────────────────


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


# ───────────────────────────────────────────────────────────────────────────
# Modal article rendering — Essay / Work / Lab (EN + DE)
# ───────────────────────────────────────────────────────────────────────────


_MODAL_MODES = ("essay", "work", "lab")


# Soft alt-text cap. Yoast doesn't penalise long alts and screen readers
# tolerate 200+ chars, but we keep some headroom so the weaved-in keyphrase
# clauses don't push alts into truncation behaviour on some themes.
_ALT_TEXT_MAX_CHARS = 200


def _alt_covers_keyphrase(alt: str, keyphrase: str) -> bool:
    """Yoast's check: at least half the keyphrase content words must
    appear (case-insensitive) somewhere in the alt text. We mirror that
    threshold so we skip patching when the alt already qualifies."""
    if not keyphrase or not alt:
        return False
    alt_lc = alt.lower()
    words = [w for w in keyphrase.lower().split() if w]
    if not words:
        return False
    hits = sum(1 for w in words if w in alt_lc)
    return hits * 2 >= len(words)


async def _patch_featured_alt_with_keyphrases(
    featured: Image,
    db: AsyncSession,
    *,
    bodies: dict[str, dict],
    selected_langs: tuple[str, ...],
) -> None:
    """Append the per-language focus_keyphrases to the featured image's
    WordPress alt_text so Yoast's image-alt SEO check passes for every
    language being published. No-ops when the alt already covers each
    keyphrase, when the featured image has no wp_media_id, or when no
    keyphrases were emitted by the LLM.
    """
    if not featured.wp_media_id:
        return

    base_alt = (featured.wp_alt_text or "").strip()
    keyphrases = [
        (lang, (bodies.get(lang) or {}).get("focus_keyphrase") or "")
        for lang in selected_langs
    ]
    missing = [
        (lang, kp) for (lang, kp) in keyphrases
        if kp and not _alt_covers_keyphrase(base_alt, kp)
    ]
    if not missing:
        return

    # Build the suffix as " — kp1 — kp2" (em-dash separated). Skip dupes
    # when the same keyphrase ended up in both languages.
    seen: set[str] = set()
    suffix_parts: list[str] = []
    for _lang, kp in missing:
        key = kp.lower()
        if key in seen:
            continue
        seen.add(key)
        suffix_parts.append(kp)

    new_alt = (base_alt + " — " + " — ".join(suffix_parts)).strip(" —")
    if len(new_alt) > _ALT_TEXT_MAX_CHARS:
        new_alt = new_alt[:_ALT_TEXT_MAX_CHARS].rstrip(" —,;:")

    logger.info(
        "Patching featured-media alt_text for SEO: media_id=%s, langs=%s",
        featured.wp_media_id, ",".join(lang for lang, _ in missing),
    )
    try:
        await request_json(
            "POST", f"/wp/v2/media/{featured.wp_media_id}",
            json={"alt_text": new_alt},
        )
    except Exception as exc:
        logger.warning("Featured-media alt PATCH failed (continuing): %s", exc)
        return

    featured.wp_alt_text = new_alt
    db.add(featured)  # ensure the row stays in the unit-of-work for the caller's commit

# WordPress category slug per (mode, language). The permalink structure on
# stefaneisele.com is /%category%/%postname%/, so a missing or wrong slug
# means posts land under /uncategorized/. Keep this in sync with the WP
# Categories admin — slugs are unique site-wide so each row resolves to
# exactly one term ID.
_MODE_CATEGORY_SLUGS: dict[tuple[str, str], str] = {
    ("essay", "en"): "essays",
    ("essay", "de"): "aufsaetze",
    ("work",  "en"): "works",
    ("work",  "de"): "werke",
    ("lab",   "en"): "lab",
    ("lab",   "de"): "labor",
}

# Per-process WP category-id cache (slug → id). WP term IDs don't change
# unless the user renames the slug; safe to cache for the lifetime of the
# server. Reset on server restart.
_CATEGORY_ID_CACHE: dict[str, int] = {}


async def _resolve_category_id(slug: str) -> int:
    """Resolve a WP category slug to its integer term ID. Cached per-process.

    Raises RuntimeError if the slug doesn't exist — the user should create
    the category in WP admin before generating posts in that (mode, lang).
    """
    if slug in _CATEGORY_ID_CACHE:
        return _CATEGORY_ID_CACHE[slug]
    result = await request_json("GET", "/wp/v2/categories", params={"slug": slug})
    if not isinstance(result, list) or not result:
        raise RuntimeError(
            f"WordPress category not found: slug={slug!r}. Create it in WP admin "
            f"(Posts → Categories) so the permalink /%category%/%postname%/ resolves."
        )
    cat_id = int(result[0]["id"])
    _CATEGORY_ID_CACHE[slug] = cat_id
    logger.info("Resolved WP category %r → id=%d", slug, cat_id)
    return cat_id


def _ordered_list_block(items: list[str]) -> str:
    """Render a wp:list ordered block. Items are HTML-escaped here."""
    if not items:
        return ""
    list_items = "\n\n".join(
        f"<!-- wp:list-item -->\n<li>{html.escape(it)}</li>\n<!-- /wp:list-item -->"
        for it in items
    )
    return (
        '<!-- wp:list {"ordered":true} -->\n'
        '<ol class="wp-block-list">'
        f"{list_items}"
        '</ol>\n'
        '<!-- /wp:list -->'
    )


def _code_block(language: str, code: str, caption: str = "") -> str:
    """Render a wp:code block with a language class for syntax highlighting.

    Gutenberg's built-in code block doesn't store a language attribute by default,
    but adding a `language-{lang}` class on the <code> element is the prism/highlight.js
    convention every WP code-highlight plugin (Enlighter, SyntaxHighlighter Evolved,
    Code Syntax Block) understands.
    """
    safe_code = html.escape(code)
    lang_attr = json.dumps({"language": language}, separators=(",", ":"))
    block = (
        f"<!-- wp:code {lang_attr} -->\n"
        f'<pre class="wp-block-code"><code class="language-{html.escape(language)}">'
        f"{safe_code}"
        f"</code></pre>\n"
        f"<!-- /wp:code -->"
    )
    if caption:
        block += "\n\n" + _para_block(f"<em>{html.escape(caption)}</em>")
    return block


def _footer_block(lang: str) -> str:
    """Render the shared site/instagram footer paragraph used by every mode."""
    site_label = (settings.artist_website_url or "")
    site_label = re.sub(r"^https?://", "", site_label).rstrip("/")
    footer_text = _FOOTER_TEMPLATE[lang].format(
        insta=_attr_url(settings.artist_instagram_url),
        site=_attr_url(settings.artist_website_url),
        site_label=html.escape(site_label),
    )
    return _para_block(f"<em>{footer_text}</em>")


def _metadata_block(metadata: dict[str, str], lang: str, singulart_link: str | None) -> str:
    """Render the Work-mode metadata as a clean wp:table block.

    Empty values are skipped. If a singulart_link is provided, it's appended as
    a trailing 'View on Singulart →' row (quiet final line — not a CTA in the body)."""
    rows: list[tuple[str, str]] = []
    for key in ("year", "medium", "dimensions", "edition", "status"):
        val = (metadata.get(key) or "").strip()
        if val:
            rows.append((_METADATA_LABELS[key][lang], val))

    if singulart_link:
        link_label = _SINGULART_CAPTION[lang]
        rows.append((link_label, f'<a href="{_attr_url(singulart_link)}">{html.escape(link_label)} →</a>'))

    if not rows:
        return ""

    body_rows = "".join(
        f"<tr><td><strong>{html.escape(label)}</strong></td><td>{value}</td></tr>"
        for label, value in rows
    )
    return (
        '<!-- wp:table {"className":"is-style-stripes"} -->\n'
        '<figure class="wp-block-table is-style-stripes"><table>'
        f"<tbody>{body_rows}</tbody>"
        '</table></figure>\n'
        '<!-- /wp:table -->'
    )


# ── Essay renderer ──────────────────────────────────────────────────────────


def _render_essay_post(
    slots: dict,
    lang: str,
    images: list[Image],
    videos: list[Video] | None = None,
) -> str:
    """Stitch Essay-mode prose into a Gutenberg post body."""
    blocks: list[str] = []
    consumed: set[int] = set()

    for paragraph in slots["intro"]:
        embed = _maybe_video_block(paragraph, videos, consumed)
        blocks.append(embed if embed is not None else _para_block(html.escape(paragraph)))

    gallery_a, gallery_b = _split_for_galleries(images)
    if gallery_a:
        blocks.append(_gallery_block(gallery_a))

    for idx, movement in enumerate(slots["movements"]):
        blocks.append(_h2_block(movement["heading"]))
        for paragraph in movement["body"]:
            embed = _maybe_video_block(paragraph, videos, consumed)
            blocks.append(embed if embed is not None else _para_block(html.escape(paragraph)))
        # Drop the second gallery (when present) between movements 2 and 3 — keeps
        # the page from being a wall of text in image-heavy essays.
        if gallery_b and idx == max(0, len(slots["movements"]) // 2 - 1):
            blocks.append(_gallery_block(gallery_b))
            gallery_b = []

    if gallery_b:
        blocks.append(_gallery_block(gallery_b))

    if slots.get("closing"):
        closing = slots["closing"]
        embed = _maybe_video_block(closing, videos, consumed)
        blocks.append(embed if embed is not None else _para_block(html.escape(closing)))

    blocks.extend(_trailing_video_blocks(videos, consumed))

    blocks.append(_separator_block())
    blocks.append(_footer_block(lang))
    return "\n\n".join(blocks)


# ── Work renderer ──────────────────────────────────────────────────────────


def _render_work_post(
    slots: dict,
    lang: str,
    images: list[Image],
    singulart_links: list[dict] | None,
    parent_series: dict[str, str] | None,
    videos: list[Video] | None = None,
) -> str:
    """Stitch Work/Series-mode prose into a Gutenberg post body."""
    blocks: list[str] = []
    consumed: set[int] = set()

    # Opening line — styled as a lead paragraph (italic) so it visually carries weight.
    opening = html.escape(slots["opening_line"])
    blocks.append(
        '<!-- wp:paragraph {"className":"has-drop-cap"} -->\n'
        f'<p class="has-drop-cap"><em>{opening}</em></p>\n'
        '<!-- /wp:paragraph -->'
    )

    for paragraph in slots["body"]:
        embed = _maybe_video_block(paragraph, videos, consumed)
        blocks.append(embed if embed is not None else _para_block(html.escape(paragraph)))

    gallery_a, gallery_b = _split_for_galleries(images)
    if gallery_a:
        blocks.append(_gallery_block(gallery_a))

    if singulart_links and slots.get("available_works_intro"):
        blocks.append(_h2_block(_HEADINGS["available_works"][lang]))
        blocks.append(_para_block(html.escape(slots["available_works_intro"])))
        for link in singulart_links:
            blocks.append(_h3_block(link.get("title") or ""))
            blocks.append(_singulart_image_block(link, _SINGULART_CAPTION[lang]))

    if parent_series and slots.get("larger_practice"):
        blocks.append(_h2_block(_HEADINGS["larger_practice"][lang]))
        for paragraph in slots["larger_practice"]:
            embed = _maybe_video_block(paragraph, videos, consumed)
            if embed is not None:
                blocks.append(embed)
            else:
                escaped = html.escape(paragraph)
                blocks.append(_para_block(_render_parent_placeholder(escaped, parent_series)))

    if gallery_b:
        blocks.append(_gallery_block(gallery_b))

    blocks.extend(_trailing_video_blocks(videos, consumed))

    blocks.append(_separator_block())
    # Pick the Singulart URL for the metadata-block trailing line, if any.
    singulart_link = (singulart_links[0]["url"] if singulart_links else None)
    metadata_block = _metadata_block(slots.get("metadata") or {}, lang, singulart_link)
    if metadata_block:
        blocks.append(metadata_block)

    blocks.append(_separator_block())
    blocks.append(_footer_block(lang))
    return "\n\n".join(blocks)


# ── Lab renderer ───────────────────────────────────────────────────────────


def _render_lab_post(
    slots: dict,
    lang: str,
    images: list[Image],
    videos: list[Video] | None = None,
) -> str:
    """Stitch Lab/Tutorial-mode prose into a Gutenberg post body."""
    blocks: list[str] = []
    consumed: set[int] = set()

    # PROBLEM
    blocks.append(_h2_block(_HEADINGS["problem"][lang]))
    problem = slots["problem"]
    embed = _maybe_video_block(problem, videos, consumed)
    blocks.append(embed if embed is not None else _para_block(html.escape(problem)))

    # APPROACH
    blocks.append(_h2_block(_HEADINGS["approach"][lang]))
    # solution_intro can be one paragraph or several joined by blank lines — split & render each.
    for paragraph in re.split(r"\n\s*\n", slots["solution_intro"].strip()):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        embed = _maybe_video_block(paragraph, videos, consumed)
        blocks.append(embed if embed is not None else _para_block(html.escape(paragraph)))

    # Tool stack — inline as a single paragraph for a tight read.
    if slots.get("tool_stack"):
        label = html.escape(_HEADINGS["tool_stack"][lang])
        stack = ", ".join(html.escape(t) for t in slots["tool_stack"])
        blocks.append(_para_block(f"<strong>{label}:</strong> {stack}"))

    # Hardware context — italic callout when present.
    if slots.get("hardware_context"):
        blocks.append(_para_block(f"<em>{html.escape(slots['hardware_context'])}</em>"))

    # Optional gallery — useful when the post documents a visual workflow.
    gallery_a, _gallery_b = _split_for_galleries(images)
    if gallery_a:
        blocks.append(_gallery_block(gallery_a))

    # STEPS
    blocks.append(_h2_block(_HEADINGS["steps"][lang]))
    if slots.get("steps"):
        blocks.append(_ordered_list_block(slots["steps"]))

    # CODE BLOCKS — all rendered immediately after the steps list.
    for cb in slots.get("code_blocks") or []:
        blocks.append(_code_block(cb["language"], cb["code"], cb.get("caption", "")))

    # RESULT
    if slots.get("result"):
        blocks.append(_h2_block(_HEADINGS["result"][lang]))
        result = slots["result"]
        embed = _maybe_video_block(result, videos, consumed)
        blocks.append(embed if embed is not None else _para_block(html.escape(result)))

    # WHY IT MATTERS
    if slots.get("why_it_matters"):
        blocks.append(_h2_block(_HEADINGS["why"][lang]))
        why = slots["why_it_matters"]
        embed = _maybe_video_block(why, videos, consumed)
        blocks.append(embed if embed is not None else _para_block(html.escape(why)))

    blocks.extend(_trailing_video_blocks(videos, consumed))

    blocks.append(_separator_block())
    blocks.append(_footer_block(lang))
    return "\n\n".join(blocks)


# ── body_md flattener (for Article.body_md full-text search / debugging) ────


def _flatten_modal_body_for_search(slots: dict, mode: str) -> str:
    """Flatten a modal-mode language block into a single string for body_md.

    Not shown to readers (WP holds the rendered HTML). Used for FTS + debugging
    + the future regenerate-metadata endpoint."""
    parts: list[str] = []
    if mode == "essay":
        parts.extend(slots.get("intro") or [])
        for mv in slots.get("movements") or []:
            parts.append(f"## {mv['heading']}")
            parts.extend(mv.get("body") or [])
        if slots.get("closing"):
            parts.append(slots["closing"])
    elif mode == "work":
        if slots.get("opening_line"):
            parts.append(slots["opening_line"])
        parts.extend(slots.get("body") or [])
        if slots.get("available_works_intro"):
            parts.append(slots["available_works_intro"])
        parts.extend(slots.get("larger_practice") or [])
    elif mode == "lab":
        if slots.get("problem"):
            parts.append(slots["problem"])
        if slots.get("solution_intro"):
            parts.append(slots["solution_intro"])
        parts.extend(f"- {s}" for s in slots.get("steps") or [])
        for cb in slots.get("code_blocks") or []:
            parts.append(f"```{cb['language']}\n{cb['code']}\n```")
        if slots.get("result"):
            parts.append(slots["result"])
        if slots.get("why_it_matters"):
            parts.append(slots["why_it_matters"])
    return "\n\n".join(p for p in parts if p)


# ───────────────────────────────────────────────────────────────────────────
# Modal article orchestrator
# ───────────────────────────────────────────────────────────────────────────


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
