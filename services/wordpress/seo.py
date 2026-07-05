"""
SEO/taxonomy helpers that talk to WordPress: featured-image alt-text
keyphrase patching (Yoast's "keyphrase in image alt" check) and category
slug → term-ID resolution for the /%category%/%postname%/ permalink structure.
"""
import logging

from sqlalchemy.ext.asyncio import AsyncSession

from core.models import Image
from services.wordpress.client import request_json

logger = logging.getLogger(__name__)


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
