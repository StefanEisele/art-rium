"""
Unit tests for the /share/* filename gate in routers/generate.py (code
review S1 / P1) — the fix for glob-injection via rglob() on a public,
single-static-token endpoint.
"""
import sys

import pytest
from fastapi import HTTPException

from routers.generate import _validate_share_filename


class TestValidateShareFilename:
    def test_plain_filename_passes(self):
        assert _validate_share_filename("abc_123.png") == "abc_123.png"

    def test_dots_underscores_hyphens_allowed(self):
        assert _validate_share_filename("a-file_name.v2.mp4") == "a-file_name.v2.mp4"

    @pytest.mark.parametrize("bad", ["*", "?", "[a-z]*.png", "a*b.png", "**", "a?.png"])
    def test_glob_metacharacters_rejected(self, bad):
        with pytest.raises(HTTPException) as exc:
            _validate_share_filename(bad)
        assert exc.value.status_code == 400

    def test_empty_filename_rejected(self):
        with pytest.raises(HTTPException):
            _validate_share_filename("")

    def test_path_traversal_reduced_to_bare_filename_not_bypassed(self):
        # Path(...).name strips directory components entirely, so this
        # resolves to a plain (safe) filename rather than escaping the
        # search roots.
        assert _validate_share_filename("../../etc/passwd") == "passwd"
        assert _validate_share_filename("a/b.png") == "b.png"

    def test_windows_backslash_traversal_reduced_to_bare_filename(self):
        # Backslash is only a path separator under Windows path semantics —
        # on POSIX it's just a character, which the charset regex rejects
        # outright (still safe, just a different rejection path).
        if sys.platform != "win32":
            pytest.skip("backslash is only a path separator on Windows")
        assert _validate_share_filename("a\\b.png") == "b.png"

    def test_space_and_control_chars_rejected(self):
        with pytest.raises(HTTPException):
            _validate_share_filename("has space.png")
        with pytest.raises(HTTPException):
            _validate_share_filename("has\ttab.png")
