"""The curated shelf of transition LoRAs for Wan 2.2's high-noise expert.

ComfyUI's loras folder is one flat pile shared by every model family this
project drives: SDXL, SD1.5, Qwen-Image, LTX, z-Image, MiniMax H3. Offering all
of it in the Morph-LoRA slot is worse than useless, because a LoRA built for
another architecture is not an error — ComfyUI loads it, none of its keys match
Wan's, and the clip renders exactly as if no LoRA had been chosen. The mistake
is invisible at the only place it could be noticed. So the slot is an allow
list.

Curated rather than sniffed off the file header for a second reason: what the
slot needs is not "a Wan LoRA" but "a Wan LoRA that turns one picture into
another". A header can tell you the architecture and the rank; it cannot tell
you that. That part is editorial and belongs in a list someone maintains.

Each entry also carries its trigger phrase, which is the half of this that is
easy to get silently wrong in the same way: a trigger-word LoRA that never sees
its trigger behaves precisely like no LoRA at all. So the trigger is put into
the prompt by the builder rather than left to whoever is writing it — see
`with_lora_trigger`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class TransitionLora:
    """One shelf entry.

    `match` is a tuple of alternative token sets: a file qualifies if every
    token of any one set appears in its normalised name. Names are matched
    rather than pinned because these files arrive from Civitai under whatever
    the uploader called that revision — "Spatial Magic_V2.safetensors" today,
    a "_V3" tomorrow — and an entry that only knows one exact spelling
    disappears from the shelf on the next download.
    """

    label: str
    match: tuple[tuple[str, ...], ...]
    trigger: str = ""
    strength: float = 1.0
    note: str = ""


# Ordered: this is the order the dropdown offers them in.
CATALOG: tuple[TransitionLora, ...] = (
    TransitionLora(
        label="Spatial Magic",
        match=(("spatial", "magic"),),
        trigger="kjmf magic",
        strength=1.0,
        note="Splits landscape and architecture open to reveal what is behind them.",
    ),
    TransitionLora(
        label="Claymation Transformation",
        match=(("claymation",),),
        strength=1.0,
        note="Trained on metamorphosis — objects knead themselves into other objects.",
    ),
    TransitionLora(
        label="LuisaP Transition V2",
        match=(("luisap",), ("deepfake", "transition")),
        trigger="DEEPFAKE, scene transition.",
        strength=1.0,
        note="Scene-to-scene transitions, high noise.",
    ),
    TransitionLora(
        label="Screen Flood",
        match=(("screen", "flood"),),
        strength=1.0,
        note="The frame floods over into what comes next.",
    ),
)

# The escape hatch. A Wan transition LoRA that is not on the shelf yet still
# reaches the slot if its own name says what it is, so downloading one does not
# require editing this file first. It arrives without a trigger phrase, which is
# the honest state of affairs: nobody here knows what it wants.
_GENERIC = ("transition", "morph", "metamorph", "transform")

_NON_WORD = re.compile(r"[^a-z0-9]+")


def _normalise(lora_name: str) -> str:
    """Lower-cased, punctuation-flattened basename, without the extension.

    ComfyUI reports a LoRA by its path relative to the loras folder, so a file
    in a subfolder arrives as "wan/Spatial Magic_V2.safetensors".
    """
    base = lora_name.replace("\\", "/").rsplit("/", 1)[-1]
    base = re.sub(r"\.(safetensors|sft|ckpt|pt)$", "", base, flags=re.I)
    return _NON_WORD.sub(" ", base.lower()).strip()


def _entry_for(lora_name: str) -> TransitionLora | None:
    norm = _normalise(lora_name)
    for entry in CATALOG:
        if any(all(tok in norm for tok in tokens) for tokens in entry.match):
            return entry
    return None


def offered(names: set[str]) -> list[dict]:
    """The shelf, restricted to what ComfyUI actually has on disk.

    Shelf order first, then any generic matches alphabetically. An entry whose
    file is missing is simply absent — the dropdown can never point at a LoRA
    that is not installed, which is the failure mode worth designing out here:
    ComfyUI rejects the whole prompt when a lora_name is unknown, so one stale
    option would kill the job rather than the LoRA.
    """
    out: list[dict] = []
    claimed: set[str] = set()
    for entry in CATALOG:
        for name in sorted(names):
            if _entry_for(name) is entry:
                out.append({
                    "file": name,
                    "label": entry.label,
                    "trigger": entry.trigger,
                    "strength": entry.strength,
                    "note": entry.note,
                })
                claimed.add(name)
                break
    for name in sorted(names - claimed):
        norm = _normalise(name)
        if _entry_for(name) is None and any(word in norm for word in _GENERIC):
            out.append({
                "file": name,
                "label": name.rsplit(".", 1)[0],
                "trigger": "",
                "strength": 1.0,
                "note": "",
            })
    return out


def trigger_for(lora_name: str | None) -> str:
    """The trigger phrase for a LoRA file, or "" if it has none on the shelf."""
    if not lora_name:
        return ""
    entry = _entry_for(lora_name)
    return entry.trigger if entry else ""


def with_lora_trigger(prompt: str, lora_name: str | None) -> str:
    """`prompt` with the LoRA's trigger phrase in front of it.

    Prepended rather than appended because that is where these LoRAs are
    trained to see it, and done here rather than in the prompt writer because
    it is mechanical: the trigger is a property of the file that was selected,
    not a creative decision about this particular transition. Left alone if the
    phrase is already in the prompt, so a user who types it themselves does not
    get it twice.
    """
    trigger = trigger_for(lora_name)
    if not trigger:
        return prompt
    if trigger.lower().strip(" .,") in prompt.lower():
        return prompt
    return f"{trigger} {prompt}".strip()
