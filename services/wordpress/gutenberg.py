"""
Gutenberg block builders — pure string-in/string-out HTML block renderers
shared by every article mode. No DB or network calls.
"""
import html
import json
import logging
import re

from core.config import settings
from core.models import Image, Video

logger = logging.getLogger(__name__)


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
