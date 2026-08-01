"""
Instagram collaborator ("Coop-Partner") tagging — validation + wire format.

Instagram lets you invite up to 3 public accounts as collaborators when the
media container is created. Each invitee gets an invitation and the post only
appears on their profile once they accept it; `GET /{ig-media-id}/collaborators`
reports the per-invitee `invite_status` (Accepted / Pending).

Two rules from the Graph API that this module exists to encode:

  1. Stories cannot have collaborators — only feed images, carousels and reels.
     workers/instagram_companion.py::publish_stories therefore never calls this.
  2. On a carousel the parameter belongs on the *parent* container only. Sent on
     an `is_carousel_item=true` child, Meta rejects the whole call with
     "param collaborators is not allowed".

Wire format is a JSON array of bare usernames: collaborators=["alice","bob"].

Docs: https://developers.facebook.com/docs/instagram-platform/instagram-graph-api/reference/ig-user/media/
"""
from __future__ import annotations

import json
import re

MAX_COLLABORATORS = 3

# Instagram usernames: letters, digits, periods and underscores, up to 30 chars.
_USERNAME_RE = re.compile(r"^[A-Za-z0-9._]{1,30}$")


def normalize(raw: list[str] | None) -> list[str] | None:
    """
    Clean a user-supplied collaborator list into what the Graph API expects.

    Strips a leading "@" and surrounding whitespace, drops blanks, and removes
    case-insensitive duplicates while keeping the given order. Returns None for
    an empty result so callers can store SQL NULL rather than an empty array.

    Raises ValueError on an invalid username or on more than MAX_COLLABORATORS
    entries — both are user input errors the router surfaces as HTTP 400.
    """
    if not raw:
        return None

    cleaned: list[str] = []
    seen: set[str] = set()
    for entry in raw:
        name = (entry or "").strip().lstrip("@").strip()
        if not name:
            continue
        if not _USERNAME_RE.match(name):
            raise ValueError(
                f"{name!r} is not a valid Instagram username "
                "(letters, digits, '.' and '_' only, max 30 characters)"
            )
        if name.casefold() in seen:
            continue
        seen.add(name.casefold())
        cleaned.append(name)

    if not cleaned:
        return None
    if len(cleaned) > MAX_COLLABORATORS:
        raise ValueError(
            f"At most {MAX_COLLABORATORS} collaborators per post "
            f"(got {len(cleaned)})"
        )
    return cleaned


def container_field(names: list[str] | None) -> dict[str, str]:
    """
    The `collaborators` entry for a media-container payload, or `{}` when there
    is nothing to tag. Spread into the container dict:

        data = {"image_url": …, "caption": …, **container_field(names)}

    Never spread this into a carousel *child* payload — see the module docstring.
    """
    if not names:
        return {}
    return {"collaborators": json.dumps(names)}
