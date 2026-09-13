"""
The AI label on cloud-scheduled posts — the desktop half.

Until 2026-09-13 the desktop sent `is_ai_generated` to the Pi outpost and the
Pi's form parser dropped it without a word: every cloud-scheduled post went out
unlabelled, which Meta's own read of the published media confirmed. The Pi now
stores, sends, echoes and reads the label back. These pin what the desktop has
to do for that to hold: carry an edited label to the Pi, and call out a Pi that
does not echo the label instead of trusting it.
"""
import asyncio
import json
import logging
import uuid

import httpx
import pytest

from core.config import settings
from services.instagram import outpost


@pytest.fixture
def pi(monkeypatch):
    """Every request the desktop sends to the outpost, answered with a 200."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"id": "p1", "status": "queued"})

    real = httpx.AsyncClient
    monkeypatch.setattr(
        httpx, "AsyncClient",
        lambda **kw: real(transport=httpx.MockTransport(handler), **kw),
    )
    monkeypatch.setattr(settings, "outpost_base_url", "https://pi.test")
    monkeypatch.setattr(settings, "outpost_shared_secret", "s")
    return seen


def test_an_edited_label_is_pushed_to_the_pi(pi):
    asyncio.run(outpost.update_on_outpost("p1", ai_label=False))
    (req,) = pi
    assert req.method == "PATCH"
    assert json.loads(req.content) == {"ai_label": False}


def test_the_label_alone_is_worth_a_patch(pi):
    # It used to be neither pushed nor refused — an edit that only changed the
    # label sent nothing at all.
    asyncio.run(outpost.update_on_outpost("p1", ai_label=True))
    assert len(pi) == 1 and json.loads(pi[0].content) == {"ai_label": True}


def test_no_changes_send_nothing(pi):
    asyncio.run(outpost.update_on_outpost("p1"))
    assert pi == []


def test_a_pi_that_does_not_echo_the_label_is_called_out(caplog):
    with caplog.at_level(logging.WARNING, logger=outpost.logger.name):
        outpost._warn_if_label_dropped(uuid.uuid4(), True, {"id": "p1", "status": "queued"})
    assert "predates AI-label support" in caplog.text


def test_an_echoed_label_is_quiet(caplog):
    with caplog.at_level(logging.WARNING, logger=outpost.logger.name):
        outpost._warn_if_label_dropped(uuid.uuid4(), True, {"id": "p1", "ai_label": True})
    assert caplog.text == ""


def test_a_job_queued_before_the_update_gets_its_label():
    assert outpost.label_needs_push(True, {"status": "queued", "ai_label": False})
    assert outpost.label_needs_push(False, {"status": "queued", "ai_label": True})


def test_nothing_is_pushed_that_cannot_or_need_not_change():
    assert not outpost.label_needs_push(True, {"status": "queued", "ai_label": True})
    assert not outpost.label_needs_push(True, {"status": "posted", "ai_label": False})
    # A Pi that does not report the label has no field to correct.
    assert not outpost.label_needs_push(True, {"status": "queued"})


def test_an_unlabelled_post_has_nothing_to_warn_about(caplog):
    with caplog.at_level(logging.WARNING, logger=outpost.logger.name):
        outpost._warn_if_label_dropped(uuid.uuid4(), False, {"id": "p1"})
    assert caplog.text == ""
