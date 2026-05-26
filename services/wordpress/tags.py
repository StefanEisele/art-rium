"""
WordPress tag resolver — looks up tags by slug and creates missing ones.

Used by services/wordpress/articles.py to attach LLM-generated tags to
WP posts. Polylang Pro scopes tags per language, so the cache key and
the REST lookup both include `lang`.

Per-process cache: `(slug, lang) → tag_id`. Slugs are stable in WP
unless the user renames them, so this is safe for the server lifetime.
"""
import logging
import re
import unicodedata

from services.wordpress.client import request_json

logger = logging.getLogger(__name__)

_TAG_ID_CACHE: dict[tuple[str, str], int] = {}

# German digraph map — NFKD alone would drop umlauts entirely
# ("übermensch" → "bermensch"), losing distinguishing letters.
_DE_DIGRAPHS = {"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss",
                "Ä": "Ae", "Ö": "Oe", "Ü": "Ue"}


def _slugify_tag(name: str) -> str:
    """ASCII-fold + kebab-case a tag name. Returns '' for input with no
    sluggable content (rare; only Chinese tags would hit this, and we
    don't currently generate Chinese articles in modal mode)."""
    s = name.strip().lower()
    for ch, repl in _DE_DIGRAPHS.items():
        s = s.replace(ch.lower(), repl)
    folded = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "-", folded).strip("-")
    return slug


async def resolve_tag_ids(names: list[str], lang: str) -> list[int]:
    """Resolve a list of tag names to WP tag IDs for the given Polylang
    language. Missing tags are created on the fly.

    - Deduplicates by slug within the input (case-insensitive).
    - Skips names that slugify to empty.
    - Logs and skips individual tags whose WP operation fails — a bad
      tag should not abort the whole article publish.
    """
    ids: list[int] = []
    seen_slugs: set[str] = set()

    for name in names:
        slug = _slugify_tag(name)
        if not slug or slug in seen_slugs:
            continue
        seen_slugs.add(slug)

        cache_key = (slug, lang)
        if cache_key in _TAG_ID_CACHE:
            ids.append(_TAG_ID_CACHE[cache_key])
            continue

        try:
            tag_id = await _lookup_or_create(name, slug, lang)
        except Exception as exc:
            logger.warning("Tag resolve failed for %r (lang=%s): %s", name, lang, exc)
            continue

        _TAG_ID_CACHE[cache_key] = tag_id
        ids.append(tag_id)

    return ids


async def _lookup_or_create(name: str, slug: str, lang: str) -> int:
    """Find a tag by slug in the given language, or POST a new one."""
    found = await request_json(
        "GET", "/wp/v2/tags",
        params={"slug": slug, "lang": lang, "per_page": 1},
    )
    if isinstance(found, list) and found:
        tag = found[0]
        logger.debug("Tag hit: slug=%s lang=%s id=%s", slug, lang, tag["id"])
        return int(tag["id"])

    logger.info("Creating WP tag: name=%r slug=%s lang=%s", name, slug, lang)
    created = await request_json(
        "POST", "/wp/v2/tags",
        json={"name": name, "slug": slug},
        params={"lang": lang},
    )
    return int(created["id"])
