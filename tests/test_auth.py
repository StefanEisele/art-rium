"""
Unit tests for core/auth.py (code review P1 + S2) — local-IP detection,
constant-time key comparison, and the signed session-token scheme that
replaced storing the raw API key in the auth cookie.
"""
import time

import pytest

import core.auth as auth


@pytest.fixture(autouse=True)
def _api_key(monkeypatch):
    """Every test gets a known, deterministic api_key instead of whatever
    is in the real environment's .env."""
    monkeypatch.setattr(auth.settings, "api_key", "test-secret-key")
    yield


class TestIsLocalIp:
    @pytest.mark.parametrize("ip", ["127.0.0.1", "192.168.1.5", "10.0.0.1"])
    def test_recognized_local_ranges(self, ip):
        assert auth.is_local_ip(ip) is True

    @pytest.mark.parametrize("ip", ["172.16.0.1", "172.31.255.255"])
    def test_172_private_range_boundaries(self, ip):
        assert auth.is_local_ip(ip) is True

    @pytest.mark.parametrize("ip", ["172.15.255.255", "172.32.0.1"])
    def test_172_outside_private_range(self, ip):
        assert auth.is_local_ip(ip) is False

    @pytest.mark.parametrize("ip", ["8.8.8.8", "203.0.113.5", ""])
    def test_public_ips_not_local(self, ip):
        assert auth.is_local_ip(ip) is False

    def test_malformed_172_address_does_not_raise(self):
        assert auth.is_local_ip("172.not-a-number.1.1") is False
        assert auth.is_local_ip("172.") is False


class TestVerifyApiKey:
    def test_correct_key(self):
        assert auth.verify_api_key("test-secret-key") is True

    def test_wrong_key(self):
        assert auth.verify_api_key("wrong") is False

    def test_none_or_empty(self):
        assert auth.verify_api_key(None) is False
        assert auth.verify_api_key("") is False

    def test_no_configured_key_means_no_key_verifies(self, monkeypatch):
        monkeypatch.setattr(auth.settings, "api_key", "")
        assert auth.verify_api_key("anything") is False


class TestSessionToken:
    def test_round_trip(self):
        token = auth.create_session_token()
        assert auth.verify_session_token(token) is True

    def test_token_is_not_the_raw_key(self):
        token = auth.create_session_token()
        assert token != auth.settings.api_key
        assert "test-secret-key" not in token

    def test_tampered_signature_rejected(self):
        token = auth.create_session_token()
        assert auth.verify_session_token(token + "x") is False

    def test_garbage_token_rejected(self):
        assert auth.verify_session_token("not-a-token") is False
        assert auth.verify_session_token("") is False
        assert auth.verify_session_token(None) is False

    def test_tampered_timestamp_invalidates_signature(self):
        token = auth.create_session_token()
        issued_at, _, sig = token.partition(".")
        tampered = f"{int(issued_at) + 1000}.{sig}"
        assert auth.verify_session_token(tampered) is False

    def test_expired_token_rejected(self):
        old_issued = str(int(time.time()) - auth.AUTH_COOKIE_MAX_AGE - 10)
        import hashlib
        import hmac
        sig = hmac.new(auth._session_secret(), old_issued.encode(), hashlib.sha256).hexdigest()
        assert auth.verify_session_token(f"{old_issued}.{sig}") is False

    def test_rotating_api_key_invalidates_old_sessions(self, monkeypatch):
        token = auth.create_session_token()
        monkeypatch.setattr(auth.settings, "api_key", "a-different-key")
        assert auth.verify_session_token(token) is False

    def test_no_configured_key_rejects_everything(self, monkeypatch):
        token = auth.create_session_token()
        monkeypatch.setattr(auth.settings, "api_key", "")
        assert auth.verify_session_token(token) is False
