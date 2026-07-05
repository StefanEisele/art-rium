"""
Unit tests for the Gutenberg block renderers in
services/wordpress/articles.py (code review P1) — pure string-in/string-out
functions, no DB or network needed.
"""
from types import SimpleNamespace

from services.wordpress.articles import (
    _aspect_class,
    _attr,
    _attr_url,
    _bullet_list_block,
    _code_block,
    _gallery_block,
    _h2_block,
    _h3_block,
    _metadata_block,
    _ordered_list_block,
    _para_block,
    _separator_block,
    _singulart_image_block,
    _split_for_galleries,
)


def _fake_image(**overrides):
    defaults = dict(wp_media_id=42, wp_alt_text="a painting", wp_source_url="https://example.com/a.png")
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class TestEscaping:
    def test_attr_url_escapes_ampersand(self):
        assert _attr_url("https://x.test/?a=1&b=2") == "https://x.test/?a=1&amp;b=2"

    def test_attr_url_none_like_empty(self):
        assert _attr_url("") == ""

    def test_attr_escapes_quotes_and_brackets(self):
        assert _attr('He said "hi" <script>') == 'He said &quot;hi&quot; &lt;script&gt;'

    def test_attr_empty_is_empty(self):
        assert _attr("") == ""


class TestSimpleBlocks:
    def test_para_block_wraps_paragraph_comment(self):
        out = _para_block("hello")
        assert out.startswith("<!-- wp:paragraph -->")
        assert "<p>hello</p>" in out
        assert out.endswith("<!-- /wp:paragraph -->")

    def test_h2_block_escapes_text(self):
        out = _h2_block("A & B")
        assert "<h2 class=\"wp-block-heading\">A &amp; B</h2>" in out

    def test_h3_block_has_level_attr(self):
        out = _h3_block("Sub")
        assert '{"level":3}' in out
        assert "<h3" in out

    def test_separator_block_is_static(self):
        out = _separator_block()
        assert "wp:separator" in out
        assert "<hr" in out

    def test_bullet_list_block_escapes_each_item(self):
        out = _bullet_list_block(["a & b", "<c>"])
        assert "a &amp; b" in out
        assert "&lt;c&gt;" in out
        assert out.count("wp:list-item") == 4  # opening + closing comment per item * 2 items

    def test_ordered_list_block_empty_returns_empty_string(self):
        assert _ordered_list_block([]) == ""

    def test_ordered_list_block_marks_ordered(self):
        out = _ordered_list_block(["first", "second"])
        assert '"ordered":true' in out
        assert "<ol" in out


class TestCodeBlock:
    def test_escapes_code_body(self):
        out = _code_block("python", "print('<hi>')")
        assert "print(&#x27;&lt;hi&gt;&#x27;)" in out
        assert "language-python" in out

    def test_appends_caption_when_given(self):
        out = _code_block("python", "x = 1", caption="Listing 1")
        assert "Listing 1" in out
        assert "wp:paragraph" in out  # caption renders as an emphasized paragraph

    def test_no_caption_omits_paragraph(self):
        out = _code_block("python", "x = 1")
        assert "wp:paragraph" not in out


class TestGalleryBlock:
    def test_empty_images_returns_empty_string(self):
        assert _gallery_block([]) == ""

    def test_renders_one_image_block_per_image(self):
        images = [_fake_image(wp_media_id=1), _fake_image(wp_media_id=2)]
        out = _gallery_block(images)
        assert out.count("wp:image") == 4  # open + close comment per image
        assert "wp-image-1" in out
        assert "wp-image-2" in out

    def test_escapes_alt_text(self):
        images = [_fake_image(wp_alt_text='a "quoted" alt')]
        out = _gallery_block(images)
        assert "&quot;quoted&quot;" in out


class TestSingulartImageBlock:
    def test_renders_link_and_caption(self):
        link = {"title": "My Piece", "url": "https://singulart.com/x", "thumbnail_url": "https://cdn/x.jpg"}
        out = _singulart_image_block(link, "View on Singulart")
        assert "https://singulart.com/x" in out
        assert "View on Singulart" in out

    def test_missing_fields_default_to_empty(self):
        out = _singulart_image_block({}, "caption")
        assert 'href=""' in out


class TestSplitForGalleries:
    def test_single_image_no_galleries(self):
        assert _split_for_galleries([1]) == ([], [])

    def test_zero_images_no_galleries(self):
        assert _split_for_galleries([]) == ([], [])

    def test_two_or_three_images_one_gallery(self):
        assert _split_for_galleries([1, 2]) == ([1, 2], [])
        assert _split_for_galleries([1, 2, 3]) == ([1, 2, 3], [])

    def test_four_or_more_split_evenly(self):
        assert _split_for_galleries([1, 2, 3, 4]) == ([1, 2], [3, 4])
        assert _split_for_galleries([1, 2, 3, 4, 5]) == ([1, 2, 3], [4, 5])


class TestAspectClass:
    def test_missing_dimensions_falls_back_to_16_9(self):
        assert _aspect_class(None, None) == "wp-embed-aspect-16-9"
        assert _aspect_class(0, 100) == "wp-embed-aspect-16-9"

    def test_exact_16_9(self):
        assert _aspect_class(1920, 1080) == "wp-embed-aspect-16-9"

    def test_square(self):
        assert _aspect_class(1000, 1000) == "wp-embed-aspect-1-1"

    def test_vertical_9_16(self):
        assert _aspect_class(1080, 1920) == "wp-embed-aspect-9-16"


class TestMetadataBlock:
    def test_empty_metadata_no_link_returns_empty(self):
        assert _metadata_block({}, "en", None) == ""

    def test_renders_populated_rows_only(self):
        out = _metadata_block({"year": "2026", "medium": "", "dimensions": "30x40cm"}, "en", None)
        assert "2026" in out
        assert "30x40cm" in out
        assert out.count("<tr>") == 2

    def test_appends_singulart_link_row(self):
        out = _metadata_block({"year": "2026"}, "en", "https://singulart.com/x")
        assert "https://singulart.com/x" in out
        assert out.count("<tr>") == 2
