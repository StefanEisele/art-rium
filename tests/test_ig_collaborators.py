"""
Instagram collaborator tagging — username normalisation and container payload.

The carousel test is the one that matters most in practice: Meta rejects the
whole publish with "param collaborators is not allowed" when the parameter
rides on an `is_carousel_item=true` child instead of the parent container.
"""
import json

import pytest

from services.instagram.collaborators import (
    MAX_COLLABORATORS,
    container_field,
    normalize,
)


# ── normalize ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    (None,                       None),
    ([],                         None),
    (["  "],                     None),
    (["alice"],                  ["alice"]),
    (["@alice"],                 ["alice"]),
    (["  @alice  "],             ["alice"]),
    (["@alice", "bob"],          ["alice", "bob"]),
    (["art.rium_", "b.c"],       ["art.rium_", "b.c"]),
])
def test_normalize_cleans_input(raw, expected):
    assert normalize(raw) == expected


def test_normalize_dedupes_case_insensitively_keeping_order():
    assert normalize(["@Bob", "alice", "bob", "@ALICE"]) == ["Bob", "alice"]


def test_normalize_drops_blanks_between_valid_entries():
    assert normalize(["alice", "", "  ", "@bob"]) == ["alice", "bob"]


@pytest.mark.parametrize("bad", [
    "has space",
    "hyphen-name",
    "email@host",
    "a" * 31,
    "sla/sh",
])
def test_normalize_rejects_invalid_usernames(bad):
    with pytest.raises(ValueError, match="not a valid Instagram username"):
        normalize([bad])


def test_normalize_rejects_more_than_max():
    too_many = [f"user{i}" for i in range(MAX_COLLABORATORS + 1)]
    with pytest.raises(ValueError, match=f"At most {MAX_COLLABORATORS}"):
        normalize(too_many)


def test_normalize_allows_exactly_max():
    names = [f"user{i}" for i in range(MAX_COLLABORATORS)]
    assert normalize(names) == names


def test_normalize_counts_after_dedupe():
    # 4 raw entries, 3 distinct — must pass rather than trip the limit.
    assert normalize(["a", "@A", "b", "c"]) == ["a", "b", "c"]


# ── container_field ──────────────────────────────────────────────────────────

def test_container_field_empty_for_no_collaborators():
    assert container_field(None) == {}
    assert container_field([]) == {}


def test_container_field_encodes_json_array():
    field = container_field(["alice", "bob"])
    assert set(field) == {"collaborators"}
    assert json.loads(field["collaborators"]) == ["alice", "bob"]


def test_container_field_is_form_safe():
    """Values must be str — httpx form-encodes the container payload."""
    for value in container_field(["alice"]).values():
        assert isinstance(value, str)


# ── carousel placement ───────────────────────────────────────────────────────

async def test_collaborators_go_on_carousel_parent_not_children():
    """
    _child_payload must never carry `collaborators`; the parent CAROUSEL
    container must. Sending it on a child fails the whole Graph API call.
    """
    import uuid

    from services.instagram.media import MediaRef
    from services.instagram.publisher import _Snapshot, _child_payload

    ref = MediaRef(kind="image", media_id=uuid.uuid4(),
                   filename="a.png", filepath="images/a.png")
    child = await _child_payload(ref, is_carousel_item=True)
    assert "collaborators" not in child
    assert child["is_carousel_item"] == "true"

    snap = _Snapshot(
        media=[ref, ref], caption="hi", collaborators=["alice"],
        story_delay=None, reel_delay=None, companion_time=None,
        scheduled_at=None, feed_creation_id=None,
    )
    parent = {
        "media_type": "CAROUSEL",
        "children": "1,2",
        "caption": snap.caption,
        **container_field(snap.collaborators),
    }
    assert json.loads(parent["collaborators"]) == ["alice"]
