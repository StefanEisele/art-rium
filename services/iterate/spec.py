"""Part 2 of prompts/iterate.md, read the way that file says it is read.

Every `## BLOCK: NAME`, `## PROMPT: NAME` or `## TEMPLATE: NAME` section runs
to the next `## ` heading. `{NAME}` inside a prompt is replaced by the block of
that name; `{lowercase}` inside a template is filled per call with
str.format(). The file is the tuning surface, so it is re-read whenever it
changes on disk (services/ollama/chat.py::_read_prompt keys its cache on the
mtime) — an edit takes effect on the next run without a restart.

A broken edit has to fail when a run is *started*, not three rounds into it,
which is why `load_spec()` checks that every section the loop needs exists and
that no `{BLOCK}` placeholder was left unresolved.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from services.ollama.chat import _read_prompt

SPEC_FILE = "iterate.md"

_SECTION_RE = re.compile(r"^## (BLOCK|PROMPT|TEMPLATE): (\w+)\n(.*?)(?=^## |\Z)", re.M | re.S)
_BLOCK_REF_RE = re.compile(r"\{([A-Z_]+)\}")
# "signature (formula mode) — …" starts a preset; its head/tail lines follow.
_PRESET_RE = re.compile(r"^(\w+) \((formula|prose) mode\)", re.M)
_ANCHOR_LINE_RE = re.compile(r'^\s+(head|tail):\s*"(.*?)"', re.M)

ROLES = ("BRIEF", "CRITIC", "DUEL", "EDITOR")
TEMPLATES = ("BRIEF_USER", "CRITIC_USER", "DUEL_USER", "EDITOR_USER")


@dataclass(frozen=True)
class Anchor:
    """The locked text around the editable core. `{triggers}` in the head is
    replaced by the active LoRA stack's trigger words."""
    name: str
    mode: str   # "formula" | "prose"
    head: str
    tail: str


@dataclass(frozen=True)
class Spec:
    prompts: dict[str, str]     # role → system prompt, blocks substituted
    templates: dict[str, str]   # name → user-message template
    anchors: dict[str, Anchor]  # preset name → anchor


def parse_spec(text: str) -> Spec:
    sections = {(kind, name): body.strip() for kind, name, body in _SECTION_RE.findall(text)}
    blocks = {name: body for (kind, name), body in sections.items() if kind == "BLOCK"}

    prompts: dict[str, str] = {}
    for role in ROLES:
        body = sections.get(("PROMPT", role))
        if body is None:
            raise RuntimeError(f"{SPEC_FILE}: section '## PROMPT: {role}' is missing")
        missing = sorted({m for m in _BLOCK_REF_RE.findall(body) if m not in blocks})
        if missing:
            raise RuntimeError(f"{SPEC_FILE}: PROMPT {role} refers to unknown block(s) {missing}")
        prompts[role] = _BLOCK_REF_RE.sub(lambda m: blocks[m.group(1)], body)

    templates: dict[str, str] = {}
    for name in TEMPLATES:
        body = sections.get(("TEMPLATE", name))
        if body is None:
            raise RuntimeError(f"{SPEC_FILE}: section '## TEMPLATE: {name}' is missing")
        templates[name] = body

    anchors = _parse_anchors(blocks.get("ANCHORS", ""))
    if not anchors:
        raise RuntimeError(f"{SPEC_FILE}: BLOCK ANCHORS defines no preset")
    return Spec(prompts=prompts, templates=templates, anchors=anchors)


def _parse_anchors(block: str) -> dict[str, Anchor]:
    starts = list(_PRESET_RE.finditer(block))
    out: dict[str, Anchor] = {}
    for i, m in enumerate(starts):
        end = starts[i + 1].start() if i + 1 < len(starts) else len(block)
        lines = dict(_ANCHOR_LINE_RE.findall(block[m.end():end]))
        out[m.group(1)] = Anchor(
            name=m.group(1), mode=m.group(2),
            head=lines.get("head", ""), tail=lines.get("tail", ""),
        )
    return out


def load_spec() -> Spec:
    return parse_spec(_read_prompt(SPEC_FILE))


def compose_prompt(anchor: Anchor, core: str, triggers: list[str]) -> str:
    """anchor.head ⊕ core ⊕ anchor.tail, comma-joined the way the artist's own
    formula prompts are ("trigger, Psychologically charged …, <core>, in the
    style of …"). An empty trigger list leaves no dangling comma."""
    head = anchor.head.replace("{triggers}", ", ".join(triggers)).strip(" ,")
    parts = [head, core.strip().strip(",.").strip(), anchor.tail.strip(" ,")]
    return ", ".join(p for p in parts if p)
