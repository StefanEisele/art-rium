"""
Story key-frames — turn one source image + a short story into N Z-Image
prompts that stay visually consistent with the source.

Two-step pipeline, each model doing what it is good at:
  1. describe_image_for_story()      — titler VLM (small, warm) extracts a
                                       concrete visual description of the
                                       source artwork.
  2. generate_story_frame_prompts()  — the prompt model (same one as the
                                       z-image enhancer) writes N sequential
                                       frame prompts from description +
                                       original prompt + story, each beat
                                       ~1-2 seconds of action apart.

System prompt lives in prompts/story-frames.md — tunable without a code
change, same convention as prompts/video-transitions.md.
"""
import logging
import re

from core.config import settings
from services.ollama.analysis import _TITLER_KEEP_ALIVE
from services.ollama.chat import _chat_json, _read_prompt

logger = logging.getLogger(__name__)

_FRAMES_ARRAY_START = re.compile(r'"frames"\s*:\s*\[')


def _salvage_truncated_frames(s: str) -> str:
    """Recover whichever leading frame prompts are complete when the model
    runs on inside one frame's string and gets cut off by num_predict before
    ever writing the closing quote/bracket — the JSON parser then reports an
    unterminated string. Drops the trailing (incomplete) entry; the caller's
    existing count-mismatch padding fills in whatever ends up missing.

    Returns `s` unchanged when nothing usable can be recovered (e.g. the very
    first frame is the one that ran away), so the normal retry-on-failure
    path in `_chat_json` kicks in instead of "succeeding" with zero frames.
    """
    m = _FRAMES_ARRAY_START.search(s)
    if not m:
        return s
    i, n = m.end(), len(s)
    frames: list[str] = []
    while i < n:
        while i < n and s[i] in " \t\r\n,":
            i += 1
        if i >= n or s[i] != '"':
            break
        j, buf, closed = i + 1, [], False
        while j < n:
            c = s[j]
            if c == "\\" and j + 1 < n:
                buf.append(c)
                buf.append(s[j + 1])
                j += 2
                continue
            if c == '"':
                closed = True
                j += 1
                break
            buf.append(c)
            j += 1
        if not closed:
            break
        frames.append('"' + "".join(buf) + '"')
        i = j
    if not frames:
        return s
    return '{"frames": [' + ", ".join(frames) + "]}"

_DESCRIBE_SYSTEM = """You are an art analyst preparing a reference sheet for an image-generation pipeline. Describe the artwork image precisely and concretely in 3-5 sentences: subject(s) with appearance/clothing/materials, setting, composition and camera angle, color palette (name the actual colors), lighting, and artistic style/medium/texture. No interpretation, no story, no marketing language — only what is visible.

Return STRICT JSON: {"description": "..."} — no prose outside the JSON, no code fences."""


async def describe_image_for_story(jpg_bytes: bytes, *, timeout: float = 180.0) -> str:
    """Concrete visual description of the source artwork via the titler VLM."""
    parsed = await _chat_json(
        model=settings.ollama_titler_model,
        system=_DESCRIBE_SYSTEM,
        user_text="Describe this artwork for the reference sheet.",
        jpgs=[jpg_bytes],
        options={"temperature": 0.3},
        keep_alive=_TITLER_KEEP_ALIVE,
        timeout=timeout,
        label="describe_image_for_story",
    )
    description = str(parsed.get("description") or "").strip()
    if not description:
        raise RuntimeError("describe_image_for_story: VLM returned no description")
    return description


async def generate_story_frame_prompts(
    *,
    story: str,
    n: int,
    description: str,
    source_prompt: str | None = None,
    trigger: str | None = None,
    beat_seconds: int = 10,
    style_block: str | None = None,
    timeout: float = 300.0,
) -> list[str]:
    """
    Write *n* sequential Z-Image frame prompts advancing *story* from the
    source image, keeping subject/setting/style consistent.

    *beat_seconds* is the story time between consecutive frames — larger
    values make each frame a bigger narrative jump (film cut) instead of a
    near-identical micro-step.

    *style_block* is an optional z-Image enhancer style block (from
    prompts/zimage-styles.md) injected into the system prompt's style
    layer; when None the style is derived from the source image alone.

    Returns exactly n non-empty strings. When the model returns the wrong
    count it is padded (by repeating the last prompt) or truncated, with a
    warning — a missing frame prompt would otherwise abort the whole job.

    Raises RuntimeError on Ollama error or when no usable prompts come back.
    """
    story = story.strip()
    if not story:
        raise ValueError("generate_story_frame_prompts: story is empty")
    if n < 1:
        raise ValueError("generate_story_frame_prompts: n must be >= 1")

    user_parts = [
        f"Source image description:\n{description}",
    ]
    if source_prompt and source_prompt.strip():
        user_parts.append(f"Original generation prompt of the source image:\n{source_prompt.strip()}")
    if trigger:
        user_parts.append(f"Trigger word (start every frame prompt with it, verbatim): {trigger}")
    user_parts.append(f"Story to advance across the frames:\n{story}")
    user_parts.append(
        f"Beat interval: roughly {max(1, beat_seconds)} seconds of story time "
        f"pass between consecutive frames. Make each frame advance the story "
        f"by that much — a visibly different moment, not a micro-variation of "
        f"the previous frame."
    )
    user_parts.append(
        f"Write exactly {n} frame prompt(s). "
        f'Return STRICT JSON: {{"frames": ["prompt 1", ...]}} with exactly {n} entries.'
    )

    system = _read_prompt("story-frames.md").replace(
        "{STYLE_BLOCK}",
        style_block.strip() if style_block and style_block.strip()
        else "none — use the source image's own style.",
    )

    parsed = await _chat_json(
        model=settings.ollama_prompt_model,
        system=system,
        user_text="\n\n".join(user_parts),
        options={
            "temperature":    0.7,
            "top_p":          0.9,
            "repeat_penalty": 1.05,
            "num_ctx":        8192,
            "num_predict":    350 * n,
        },
        think=False,
        timeout=timeout,
        label="generate_story_frame_prompts",
        salvage=_salvage_truncated_frames,
        max_attempts=3,
    )
    raw = parsed.get("frames") or []
    if not isinstance(raw, list):
        raise RuntimeError(f"Story LLM 'frames' field is not a list: {type(raw).__name__}")

    cleaned = [str(p).strip() for p in raw if str(p).strip()]
    if not cleaned:
        raise RuntimeError("Story LLM returned no usable frame prompts")
    if len(cleaned) != n:
        logger.warning(
            "generate_story_frame_prompts: expected %d prompts, got %d — padding/truncating",
            n, len(cleaned),
        )
        while len(cleaned) < n:
            cleaned.append(cleaned[-1])
        cleaned = cleaned[:n]

    return [ensure_trigger(p, trigger) for p in cleaned]


def ensure_trigger(prompt: str, trigger: str | None) -> str:
    """Prepend the LoRA trigger word when the model dropped it (belt and
    braces — the system prompt already demands it)."""
    if not trigger or trigger.lower() in prompt.lower():
        return prompt
    return f"{trigger}, {prompt}"
