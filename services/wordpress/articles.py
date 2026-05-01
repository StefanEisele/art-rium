"""
WordPress article publisher — Single Source of Truth for generating
multilingual blog posts from a single artwork and pushing them to WordPress
as Polylang language-tagged posts.

Pipeline per image:
  1. Verify the image is already in the WP media library (image.wp_media_id set)
  2. Re-encode source PNG → JPG (vlm_analysis_max_edge for fidelity vs. speed)
  3. Run write_article() — single VLM call returns DE/EN/ZH bodies in one pass
  4. Render each body_md → simple HTML
  5. Create three WP posts via /wp/v2/posts?lang=xx (status=draft|publish)
     featured_media=image.wp_media_id, slug derived from title + lang suffix.
  6. Persist three Article rows sharing one translation_group_id.

Polylang note: this code does NOT auto-link the three posts as translations.
Polylang Pro's REST surface only exposes /pll/v1/languages and /settings;
linking has to be done via the WP admin (one click per post in Polylang's
translation column) or a future custom mu-plugin endpoint.
"""
import html
import logging
import re
import unicodedata
import uuid
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession

from core.config import settings
from core.imaging import prepare_jpg_for_web
from core.models import Article, Image
from services.ollama.client import write_article
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


async def generate_articles_for_image(
    image: Image,
    db: AsyncSession,
    *,
    publish: bool = False,
) -> dict:
    """
    Generate DE/EN/ZH articles for *image* and push them to WordPress.

    Args:
      image:    Image row that has already been uploaded to WP (wp_media_id set)
      db:       async session, committed once at the end
      publish:  if True, posts go up as 'publish'; otherwise 'draft'

    Returns:
      {
        "translation_group_id": str,
        "articles": [
          {"language": "en", "wp_post_id": int, "wp_link": str, "status": str,
           "title": str, "excerpt": str, "tags": [...]},
          ...
        ],
      }
    """
    if not image.wp_media_id:
        raise ValueError(
            f"Image {image.id} has no wp_media_id — upload it via /api/wordpress/media/upload first."
        )

    src = settings.storage_dir / image.filepath
    if not src.exists():
        raise FileNotFoundError(f"Source image missing on disk: {src}")

    logger.info("Article gen %s — re-encoding for VLM", image.id)
    vlm_jpg, _ = await prepare_jpg_for_web(
        src, max_edge=settings.vlm_analysis_max_edge, quality=80
    )

    logger.info(
        "Article gen %s — calling %s (image %dKB, edge %d)",
        image.id, settings.ollama_llm_model,
        len(vlm_jpg) // 1024, settings.vlm_analysis_max_edge,
    )
    bodies = await write_article(
        vlm_jpg,
        title_hint=image.title or image.wp_seo_title,
        alt_text=image.wp_alt_text,
        notes=image.notes,
    )

    translation_group_id = uuid.uuid4()
    target_status = "publish" if publish else "draft"
    results: list[dict] = []

    for lang in _LANGS:
        block = bodies[lang]
        slug = _slugify(block["title"], lang)
        body_html = _md_to_html(block["body_md"])

        post_payload = {
            "title":          block["title"],
            "content":        body_html,
            "excerpt":        block["excerpt"],
            "status":         target_status,
            "featured_media": image.wp_media_id,
        }
        if slug:
            post_payload["slug"] = slug

        logger.info(
            "Article gen %s — POST /wp/v2/posts?lang=%s (status=%s, slug=%s)",
            image.id, lang, target_status, slug,
        )
        post = await request_json(
            "POST",
            "/wp/v2/posts",
            json=post_payload,
            params={"lang": lang},
        )
        wp_post_id = post["id"]
        wp_link    = post.get("link", "")

        article = Article(
            title=block["title"],
            body_md=block["body_md"],
            excerpt=block["excerpt"],
            tags=block["tags"] or None,
            language=lang,
            translation_group_id=translation_group_id,
            wp_post_id=wp_post_id,
            wp_link=wp_link,
            status="published" if publish else "draft",
            image_ids=[image.id],
        )
        db.add(article)

        results.append({
            "language":   lang,
            "wp_post_id": wp_post_id,
            "wp_link":    wp_link,
            "status":     article.status,
            "title":      block["title"],
            "excerpt":    block["excerpt"],
            "tags":       block["tags"],
        })

    await db.commit()
    logger.info(
        "Article gen %s done — group=%s, %s post(s) %s",
        image.id, translation_group_id, len(results), target_status,
    )

    return {
        "translation_group_id": str(translation_group_id),
        "articles":             results,
    }
