"""
WordPress article publisher — Single Source of Truth for generating
multilingual blog posts and pushing them to WordPress as Polylang
language-tagged posts.

Two flavours, sharing language-creation order, slugs, and translation grouping:

  generate_articles_for_images(images, db, *, publish)
    Contemplative single-artwork (or small-series) art-rium voice. 4 paragraphs.
    Voice rules: prompts/voice.md.

  generate_rich_articles_for_series(images, db, *, series_name, parent_series,
                                    singulart_links, publish)
    SEO-friendly rich series article. Multi-section structure with H2
    headings, embedded image galleries, optional Singulart product cards,
    optional inline parent-series link. Voice rules: prompts/voice-rich.md.

Polylang note: neither flavour auto-links the three posts as translations.
Polylang Pro's REST surface only exposes /pll/v1/languages and /settings;
linking has to be done via the WP admin (one click per post in Polylang's
translation column) or a future custom mu-plugin endpoint.
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
from core.models import Article, Image
from services.ollama.client import write_article, write_rich_article
from services.wordpress.client import request_json

logger = logging.getLogger(__name__)


_LANGS = ("en", "de", "zh")  # creation order: EN first so DE/ZH can reference it later if we add linking


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
) -> dict:
    """POST one language to WP + queue the Article row + return the result dict.

    The caller owns db.commit() — we only db.add() so the whole translation
    set commits in a single tx.
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
    "concept":            {"de": "Das Konzept",                 "en": "The Concept",                "zh": "创作理念"},
    "visual_language":    {"de": "Visuelle Sprache",            "en": "Visual Language",            "zh": "视觉语言"},
    "technical_approach": {"de": "Technischer Ansatz",          "en": "Technical Approach",         "zh": "技术方法"},
    "available_works":    {"de": "Verfügbare Werke",            "en": "Available Works",            "zh": "可购作品"},
    "larger_practice":    {"de": "Teil einer größeren Praxis",  "en": "Part of a Larger Practice",  "zh": "更大实践的一部分"},
}

_SINGULART_CAPTION = {
    "de": "Erhältlich auf Singulart",
    "en": "Available on Singulart",
    "zh": "在 Singulart 购买限量版",
}

# {insta} and {site} are href URLs (already escaped); {site_label} is the displayed text.
_FOOTER_TEMPLATE = {
    "de": 'Folgen Sie der Entwicklung dieser und anderer Serien auf <a href="{insta}">Instagram</a> oder entdecken Sie weitere Arbeiten auf <a href="{site}">{site_label}</a>',
    "en": 'Follow the development of this and other series on <a href="{insta}">Instagram</a> or explore further work at <a href="{site}">{site_label}</a>',
    "zh": '在 <a href="{insta}">Instagram</a> 关注此系列与其他作品的进展，或访问 <a href="{site}">{site_label}</a> 探索更多作品',
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
