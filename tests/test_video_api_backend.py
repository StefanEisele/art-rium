"""The provider adapter: payload shape, validation, status mapping, backoff.

All of it exercised without the network — `parse_task` and `build_payload` are
pure on purpose, so the contract can be tested against recorded payloads.
"""
from pathlib import Path

import pytest

from services.video_api import pricing
from services.video_api.backend import (
    MODEL,
    VideoBackendError,
    VideoRequest,
    build_payload,
    parse_task,
    poll_delay,
    validate,
)


def _req(**kw) -> VideoRequest:
    kw.setdefault("prompt", "a slow pan across a rust-coloured room")
    return VideoRequest(**kw)


# ── Request shape ────────────────────────────────────────────────────────────


def test_payload_matches_the_documented_v2_shape():
    payload = build_payload(_req(duration_s=8, resolution="768P", ratio="16:9"))
    assert payload["model"] == MODEL
    assert payload["resolution"] == "768P"
    assert payload["duration"] == 8
    assert payload["ratio"] == "16:9"
    assert payload["content"][0] == {
        "type": "text", "text": "a slow pan across a rust-coloured room",
    }


def test_exactly_one_text_item_leads_the_content_array():
    content = build_payload(_req())["content"]
    assert len([c for c in content if c["type"] == "text"]) == 1
    assert content[0]["type"] == "text"


def test_mode_is_derived_from_what_is_attached(tmp_path: Path):
    img = tmp_path / "a.png"
    img.write_bytes(b"x")
    assert _req().mode == "text"
    assert _req(first_frame=img).mode == "image"
    assert _req(reference_images=[img]).mode == "reference"


def test_frames_count_towards_the_free_image_allowance(tmp_path: Path):
    """First and last frame are images too — billing them as free would
    under-reserve a job that uses both plus four references."""
    img = tmp_path / "a.png"
    img.write_bytes(b"x")
    request = _req(first_frame=img, last_frame=img, reference_images=[img, img])
    assert request.image_count == 4


# ── Validation ───────────────────────────────────────────────────────────────


def test_empty_prompt_is_refused():
    with pytest.raises(VideoBackendError):
        validate(_req(prompt="   "))


def test_over_long_prompt_is_refused():
    with pytest.raises(VideoBackendError):
        validate(_req(prompt="x" * (pricing.PROMPT_MAX_CHARS + 1)))


@pytest.mark.parametrize("duration", [3, 16, 0, -1])
def test_duration_outside_the_documented_range_is_refused(duration):
    with pytest.raises(VideoBackendError):
        validate(_req(duration_s=duration))


def test_unknown_resolution_and_ratio_are_refused():
    with pytest.raises(VideoBackendError):
        validate(_req(resolution="4K"))
    with pytest.raises(VideoBackendError):
        validate(_req(ratio="17:6"))


def test_too_many_reference_images_are_refused(tmp_path: Path):
    img = tmp_path / "a.png"
    img.write_bytes(b"x")
    validate(_req(reference_images=[img] * pricing.MAX_REFERENCE_IMAGES))
    with pytest.raises(VideoBackendError):
        validate(_req(reference_images=[img] * (pricing.MAX_REFERENCE_IMAGES + 1)))


def test_a_valid_request_passes():
    validate(_req(duration_s=6, resolution="2K", ratio="adaptive"))


# ── Status mapping ───────────────────────────────────────────────────────────


def test_succeeded_task_yields_url_and_billed_usage():
    state = parse_task({"task": {
        "status": "succeeded",
        "content": {"url": "https://cdn.example/clip.mp4"},
        "usage": {"total_seconds": 6, "output_seconds": 6, "input_seconds": 0,
                  "input_image_count": 2},
    }})
    assert state.done and state.status == "succeeded"
    assert state.output_url == "https://cdn.example/clip.mp4"
    assert state.billed_seconds == 6
    assert state.billed_image_count == 2


@pytest.mark.parametrize("status,done", [
    ("queued", False), ("running", False),
    ("succeeded", True), ("failed", True), ("cancelled", True),
])
def test_terminality_of_each_documented_status(status, done):
    assert parse_task({"task": {"status": status}}).done is done


def test_failed_task_carries_a_message():
    state = parse_task({"task": {"status": "failed", "error": "content rejected"}})
    assert state.status == "failed"
    assert "content rejected" in state.error


def test_unknown_status_is_treated_as_running():
    """Inventing a terminal answer here would settle or release money on a
    guess; staying non-terminal costs one more poll."""
    state = parse_task({"task": {"status": "reticulating"}})
    assert state.status == "running"
    assert not state.done


def test_empty_body_does_not_explode():
    assert parse_task({}).status == "queued"


# ── Polling cadence ──────────────────────────────────────────────────────────


def test_poll_backoff_ramps_then_settles_at_the_documented_interval():
    assert [poll_delay(i) for i in range(5)] == [5, 10, 15, 15, 15]
