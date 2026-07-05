"""
Unit tests for the hand-rolled adversarial-LLM-output parsing in
services/ollama/client.py (code review P1) — fence stripping, the German
curly-quote JSON salvage pass, and the essay-block validator.
"""
import pytest

from services.ollama.client import (
    _clamp_focus_keyphrase,
    _count_substring,
    _iterative_salvage,
    _salvage_curly_quotes,
    _strip_json_fences,
    _strip_list,
    _strip_str,
    _truncate_at_word,
    _validate_essay_block,
)


class TestStripJsonFences:
    def test_no_fence_is_unchanged(self):
        assert _strip_json_fences('{"a": 1}') == '{"a": 1}'

    def test_strips_json_fence(self):
        raw = '```json\n{"a": 1}\n```'
        assert _strip_json_fences(raw) == '{"a": 1}'

    def test_strips_bare_fence(self):
        raw = '```\n{"a": 1}\n```'
        assert _strip_json_fences(raw) == '{"a": 1}'

    def test_idempotent(self):
        once = _strip_json_fences('```json\n{"a": 1}\n```')
        assert _strip_json_fences(once) == once

    def test_strips_surrounding_whitespace(self):
        assert _strip_json_fences('  \n{"a": 1}\n  ') == '{"a": 1}'

    def test_unclosed_fence_still_strips_opening(self):
        # No closing ``` — only the opening fence + language tag come off.
        assert _strip_json_fences('```json\n{"a": 1}') == '{"a": 1}'


class TestSalvageCurlyQuotes:
    def test_fixes_ascii_close_after_german_open(self):
        raw = 'ein „Zitat" im Satz'
        assert _salvage_curly_quotes(raw) == 'ein „Zitat” im Satz'

    def test_no_german_open_is_unchanged(self):
        raw = 'a "plain" quote'
        assert _salvage_curly_quotes(raw) == raw

    def test_already_curly_is_unchanged(self):
        raw = 'ein „Zitat” im Satz'
        assert _salvage_curly_quotes(raw) == raw

    def test_fixes_inside_json_value(self):
        raw = '{"title": "Der „Klang" der Stille"}'
        fixed = _salvage_curly_quotes(raw)
        assert '„Klang”' in fixed


class TestIterativeSalvage:
    def test_stops_when_no_change(self):
        raw = "no quotes here at all"
        assert _iterative_salvage(raw) == raw

    def test_fixes_multiple_occurrences_across_passes(self):
        raw = '„eins" und „zwei" und „drei"'
        fixed = _iterative_salvage(raw)
        assert fixed == '„eins” und „zwei” und „drei”'

    def test_respects_max_passes_without_infinite_loop(self):
        # Degenerate input that could in principle keep "changing" — must
        # still terminate within max_passes and return a str.
        raw = '„' * 20 + '"' * 20
        result = _iterative_salvage(raw, max_passes=3)
        assert isinstance(result, str)


class TestStripHelpers:
    def test_strip_str_none_becomes_empty(self):
        assert _strip_str(None) == ""

    def test_strip_str_trims_whitespace(self):
        assert _strip_str("  hi  ") == "hi"

    def test_strip_str_truncates(self):
        assert _strip_str("abcdefgh", max_chars=3) == "abc"

    def test_strip_list_non_list_returns_empty(self):
        assert _strip_list("not a list") == []
        assert _strip_list(None) == []

    def test_strip_list_drops_blank_entries(self):
        assert _strip_list(["a", "  ", "", "b"]) == ["a", "b"]

    def test_strip_list_lowercases_and_caps(self):
        assert _strip_list(["Foo", "BAR", "Baz"], max_items=2, lower=True) == ["foo", "bar"]


class TestClampFocusKeyphrase:
    def test_short_phrase_unchanged(self):
        assert _clamp_focus_keyphrase("still life painting") == "still life painting"

    def test_long_phrase_truncated_to_max_words(self):
        assert _clamp_focus_keyphrase("one two three four five six", max_words=4) == "one two three four"

    def test_lowercased(self):
        assert _clamp_focus_keyphrase("Still Life") == "still life"

    def test_empty_input(self):
        assert _clamp_focus_keyphrase("") == ""
        assert _clamp_focus_keyphrase(None) == ""


class TestTruncateAtWord:
    def test_under_budget_unchanged(self):
        assert _truncate_at_word("short text", 50) == "short text"

    def test_cuts_at_word_boundary(self):
        text = "the quick brown fox jumps over the lazy dog"
        result = _truncate_at_word(text, 20)
        # Cuts before "jumps" rather than mid-word ("...fox jum").
        assert result == "the quick brown fox"
        assert len(result) <= 20

    def test_falls_back_to_hard_slice_when_no_good_break(self):
        text = "supercalifragilisticexpialidocious"
        result = _truncate_at_word(text, 10)
        assert result == text[:10]

    def test_strips_trailing_punctuation_after_cut(self):
        text = "alpha beta, gamma delta epsilon zeta"
        result = _truncate_at_word(text, 13)
        assert not result.endswith((",", ";", ":", ".", "-"))


class TestCountSubstring:
    def test_case_insensitive(self):
        assert _count_substring("Der Klang der Stille", "klang") == 1

    def test_counts_multiple_non_overlapping(self):
        assert _count_substring("abab abab", "ab") == 4

    def test_empty_needle_returns_zero(self):
        assert _count_substring("anything", "") == 0

    def test_no_match_returns_zero(self):
        assert _count_substring("hello world", "xyz") == 0


class TestValidateEssayBlock:
    def _valid_block(self, **overrides):
        block = {
            "title": "Still Waters.",
            "intro": ["An opening line about still waters."],
            "movements": [
                {"heading": "First movement", "body": ["Some body text."]},
            ],
            "closing": "A closing line.",
            "excerpt": "A short excerpt.",
            "meta_description": "A meta description.",
            "focus_keyphrase": "still waters",
            "tags": ["a", "b"],
            "og_image_idea": "a lake at dawn",
        }
        block.update(overrides)
        return block

    def test_valid_block_normalizes_and_strips_trailing_period_from_title(self):
        result = _validate_essay_block("en", self._valid_block())
        assert result["title"] == "Still Waters"
        assert result["movements"][0]["heading"] == "First movement"

    def test_missing_title_raises(self):
        with pytest.raises(RuntimeError):
            _validate_essay_block("en", self._valid_block(title=""))

    def test_missing_intro_raises(self):
        with pytest.raises(RuntimeError):
            _validate_essay_block("en", self._valid_block(intro=[]))

    def test_missing_movements_raises(self):
        with pytest.raises(RuntimeError):
            _validate_essay_block("en", self._valid_block(movements=[]))

    def test_movement_missing_heading_raises(self):
        bad = self._valid_block(movements=[{"heading": "", "body": ["text"]}])
        with pytest.raises(RuntimeError):
            _validate_essay_block("en", bad)

    def test_movement_missing_body_raises(self):
        bad = self._valid_block(movements=[{"heading": "H", "body": []}])
        with pytest.raises(RuntimeError):
            _validate_essay_block("en", bad)

    def test_tags_capped_and_lowercased(self):
        block = self._valid_block(tags=["One", "Two", "Three", "Four", "Five", "Six", "Seven"])
        result = _validate_essay_block("en", block)
        assert result["tags"] == ["one", "two", "three", "four", "five", "six"]

    def test_non_dict_movement_entries_are_skipped(self):
        block = self._valid_block(
            movements=["not a dict", {"heading": "Real", "body": ["text"]}]
        )
        result = _validate_essay_block("en", block)
        assert len(result["movements"]) == 1
        assert result["movements"][0]["heading"] == "Real"
