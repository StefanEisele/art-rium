"""
Ollama VLM analysis — per-image alt-text/SEO metadata, title suggestions
(image + video), and the titler warm-up call.
"""
import base64
import logging

import httpx

from core.config import settings
from services.ollama.chat import _chat_json, _read_prompt

logger = logging.getLogger(__name__)


# ── Per-image metadata (alt-text + SEO) ──────────────────────────────────────


_SYSTEM_PROMPT = """You are an art-blog image analyst. Analyze the artwork image and return STRICT JSON with exactly four fields:
  seo_title: 3 to 7 words, max 60 characters. An evocative title for the artwork — concrete (subject, palette, mood), in the art-rium voice. Title Case. No clickbait, no ALL CAPS, no colons cramming two ideas, no trailing period.
  alt_text: one sentence describing what is visibly in the image, suitable for screen readers. Maximum 125 characters. Concrete (subject, composition, palette). No interpretation, no marketing language.
  seo_description: one sentence describing the artwork's mood and subject for use as a meta description on a blog post. Maximum 155 characters. Concrete imagery, no buzzwords, no superlatives.
  caption: 1-2 sentences for the WordPress media library, slightly more descriptive than alt_text, may include subject and atmosphere. Maximum 300 characters.

Use the provided title and notes as context but do not quote them verbatim. Write in the requested language. Return ONLY the JSON object — no prose, no code fences, no commentary."""


async def analyze_image(
    jpg_bytes: bytes,
    *,
    title: str | None = None,
    notes: str | None = None,
    language: str = "en",
    timeout: float = 600.0,
) -> dict[str, str]:
    """
    Run the local VLM on *jpg_bytes* and return alt_text / seo_description / caption.

    Raises:
      RuntimeError if Ollama returns an error or the model output is not valid JSON.
      httpx.HTTPError on network/timeout failures.
    """
    user_text = (
        f"Title: {title or '(none)'}\n"
        f"Notes: {notes or '(none)'}\n\n"
        f"Language for output: {language}"
    )
    parsed = await _chat_json(
        model=settings.ollama_vlm_model,
        system=_SYSTEM_PROMPT,
        user_text=user_text,
        jpgs=[jpg_bytes],
        options={"temperature": 0.4},
        timeout=timeout,
        label="analyze_image",
    )
    return {
        "seo_title":       (parsed.get("seo_title") or "").strip().rstrip(".").strip()[:60],
        "alt_text":        (parsed.get("alt_text") or "").strip()[:125],
        "seo_description": (parsed.get("seo_description") or "").strip()[:155],
        "caption":         (parsed.get("caption") or "").strip()[:300],
    }


# ── Titler — title suggestions + warm-up ─────────────────────────────────────


_TITLER_SYSTEM = "You are an art curator specialising in contemporary media art."
_TITLER_KEEP_ALIVE = "30m"  # keep VLM resident in VRAM after each call — cold-load is ~2.5 min


async def warm_titler_model(timeout: float = 600.0) -> None:
    """
    Fire-and-forget Ollama call that loads the titler model into VRAM.

    Called from the FastAPI lifespan as a background task so the first real
    request from the frontend doesn't pay the ~150s cold-load and hit
    upstream timeouts (Cloudflare tunnel caps at ~100s).

    Uses a tiny synthetic JPG so the vision tower warms up too, with
    num_predict=1 to keep wall time near the pure load cost.
    """
    from io import BytesIO
    from PIL import Image as PILImage

    buf = BytesIO()
    PILImage.new("RGB", (32, 32), (128, 128, 128)).save(buf, "JPEG", quality=50)
    jpg = buf.getvalue()

    payload = {
        "model":      settings.ollama_titler_model,
        "messages":   [{"role": "user", "content": "ok", "images": [base64.b64encode(jpg).decode("ascii")]}],
        "stream":     False,
        "keep_alive": _TITLER_KEEP_ALIVE,
        "options":    {"num_predict": 1, "temperature": 0.0},
    }
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(f"{settings.ollama_host}/api/chat", json=payload)
        if r.status_code != 200:
            logger.warning("Titler warm-up: ollama returned %s — %s", r.status_code, r.text[:200])
            return
        data = r.json()
        load_ms = (data.get("load_duration") or 0) / 1_000_000
        total_ms = (data.get("total_duration") or 0) / 1_000_000
        logger.info(
            "Titler warm-up complete: model=%s, load=%.1fs, total=%.1fs",
            settings.ollama_titler_model, load_ms / 1000, total_ms / 1000,
        )
    except Exception as exc:
        logger.warning("Titler warm-up failed (will retry on first request): %s", exc)


async def generate_titles(
    jpg_bytes: bytes,
    *,
    n: int = 5,
    timeout: float = 120.0,
) -> list[str]:
    """
    Generate *n* short title suggestions for the artwork in *jpg_bytes*.

    Uses OLLAMA_TITLER_MODEL (must be vision-capable). Returns a deduplicated
    list of cleaned title strings, capped at *n*.

    Raises RuntimeError on Ollama error / non-JSON output, httpx.HTTPError
    on network failures.
    """
    user_text = (
        f"Suggest {n} short, evocative titles for this artwork. "
        f"Each title: 2 to 6 words, Title Case, no trailing punctuation, "
        f"no surrounding quotes, no numbering, no commentary.\n\n"
        f'Return STRICT JSON: {{"titles": ["title one", "title two", ...]}}'
    )
    parsed = await _chat_json(
        model=settings.ollama_titler_model,
        system=_TITLER_SYSTEM,
        user_text=user_text,
        jpgs=[jpg_bytes],
        options={"temperature": 0.8},
        keep_alive=_TITLER_KEEP_ALIVE,
        timeout=timeout,
        label="generate_titles",
    )
    return _clean_titles(parsed, n=n)


async def generate_video_titles(
    jpgs: list[bytes],
    *,
    n: int = 5,
    timeout: float = 180.0,
) -> list[str]:
    """
    Generate *n* short title suggestions for a short video, given a handful of
    evenly-spaced sample frames in playback order.

    Reuses OLLAMA_TITLER_MODEL with a video-aware user prompt that tells the
    model the frames belong to one work (so titles describe the piece as a
    whole, not each frame separately). Returns a deduplicated, cleaned list
    capped at *n*.
    """
    if not jpgs:
        raise RuntimeError("generate_video_titles requires at least one frame")

    user_text = (
        f"The {len(jpgs)} images below are evenly-spaced frames from one short "
        f"video artwork, in playback order. Suggest {n} short, evocative titles "
        f"for the video as a whole — not the individual frames. "
        f"Each title: 2 to 6 words, Title Case, no trailing punctuation, "
        f"no surrounding quotes, no numbering, no commentary.\n\n"
        f'Return STRICT JSON: {{"titles": ["title one", "title two", ...]}}'
    )
    parsed = await _chat_json(
        model=settings.ollama_titler_model,
        system=_TITLER_SYSTEM,
        user_text=user_text,
        jpgs=jpgs,
        options={"temperature": 0.8},
        keep_alive=_TITLER_KEEP_ALIVE,
        timeout=timeout,
        label="generate_video_titles",
    )
    return _clean_titles(parsed, n=n)


# ── Transition prompts — Wan2.2 FLF2V key-frame sequence ─────────────────────


async def generate_transition_prompts(
    jpgs: list[bytes],
    *,
    context: str = "",
    timeout: float = 300.0,
) -> list[str]:
    """
    Given N key-frame images in playback order, ask the titler VLM to suggest
    one Wan2.2 FLF2V transition prompt per adjacent pair (N-1 prompts total)
    in a single vision call — cheaper than N-1 separate calls (one model
    load) and gives the model whole-sequence context for coherent motion.

    *context* is optional free text describing what the sequence is about
    (the story-frames flow passes the user's story here) so the suggested
    motion follows the intended narrative instead of being guessed from the
    images alone.

    System prompt lives in prompts/video-transitions.md — tunable without a
    code change, same convention as prompts/zimage-styles.md.

    Returns exactly len(jpgs)-1 strings, padding with "" or truncating if the
    model returns the wrong count (logged as a warning either way — the
    client renders blank textareas for any padded slots).

    Raises RuntimeError if jpgs has fewer than 2 images, on Ollama error, or
    if the parsed response has no usable "transitions" list.
    """
    if len(jpgs) < 2:
        raise RuntimeError("generate_transition_prompts requires at least 2 images")

    n_trans = len(jpgs) - 1
    context_block = (
        f"Story context for the whole sequence (the key frames were generated "
        f"to tell this story, in order):\n{context.strip()}\n\n"
        if context.strip() else ""
    )
    user_text = (
        f"The {len(jpgs)} images below are key frames for one video, in "
        f"playback order. Write exactly {n_trans} transition prompt(s), one "
        f"per adjacent pair (image 1→2, image 2→3, …), following the system "
        f"instructions.\n\n"
        f"{context_block}"
        f'Return STRICT JSON: {{"transitions": ["prompt 1", "prompt 2", ...]}} '
        f"with exactly {n_trans} entries, in order."
    )
    parsed = await _chat_json(
        model=settings.ollama_titler_model,
        system=_read_prompt("video-transitions.md"),
        user_text=user_text,
        jpgs=jpgs,
        options={"temperature": 0.6},
        keep_alive=_TITLER_KEEP_ALIVE,
        timeout=timeout,
        label="generate_transition_prompts",
    )
    raw = parsed.get("transitions") or []
    if not isinstance(raw, list):
        raise RuntimeError(f"Transition VLM 'transitions' field is not a list: {type(raw).__name__}")

    cleaned = [str(t).strip() for t in raw]
    if len(cleaned) != n_trans:
        logger.warning(
            "generate_transition_prompts: expected %d prompts, got %d — padding/truncating",
            n_trans, len(cleaned),
        )
        cleaned = (cleaned + [""] * n_trans)[:n_trans]
    return cleaned


# ── i2v animation prompts — surreal per-image motion (Wan2.2 / LTX) ──────────


def _extract_animation(parsed: dict) -> str:
    """Pull the single prompt out of a per-image response, tolerating the
    model echoing the plural/array shape instead of {"animation": "..."}."""
    value = parsed.get("animation")
    if not value:
        alt = parsed.get("animations")
        if isinstance(alt, list) and alt:
            value = alt[0]
    if isinstance(value, list):
        value = value[0] if value else ""
    return str(value or "").strip()


async def generate_i2v_motion_prompts(
    jpgs: list[bytes],
    *,
    context: str = "",
    timeout: float = 300.0,
) -> list[str]:
    """
    Given N images (each becomes its own independent i2v clip, in playback
    order), ask the titler VLM for one surreal Wan2.2-style animation prompt
    per image — one vision call PER image, unlike generate_transition_prompts.

    Rationale for per-image calls: the small titler VLM (qwen2.5vl:3b)
    effectively only looks at the first image of a multi-image message and
    repeats one prompt N times. Serialized single-image calls fix that; the
    model stays resident between calls (keep_alive), so each extra call costs
    inference only, no reload. *timeout* applies per call.

    *context* is optional free text about the intended video, folded into
    every per-image message when present.

    System prompt lives in prompts/video-i2v-motion.md. Returns exactly
    len(jpgs) strings; an image whose call fails yields "" (logged), so the
    client just shows an empty textarea for that slot.

    Raises RuntimeError if jpgs is empty or if EVERY per-image call failed.
    """
    if not jpgs:
        raise RuntimeError("generate_i2v_motion_prompts requires at least 1 image")

    n = len(jpgs)
    context_block = (
        f"Context for the whole video (what it is about / intended mood):\n"
        f"{context.strip()}\n\n"
        if context.strip() else ""
    )

    prompts: list[str] = []
    failures = 0
    for i, jpg in enumerate(jpgs):
        user_text = (
            f"This is image {i + 1} of {n} for one video — each image becomes "
            f"its own independent clip, played in order. Write ONE surreal "
            f"animation prompt for THIS image, following the system "
            f"instructions.\n\n"
            f"{context_block}"
            f'Return STRICT JSON: {{"animation": "<the prompt>"}}.'
        )
        try:
            parsed = await _chat_json(
                model=settings.ollama_titler_model,
                system=_read_prompt("video-i2v-motion.md"),
                user_text=user_text,
                jpgs=[jpg],
                options={"temperature": 0.7},
                keep_alive=_TITLER_KEEP_ALIVE,
                timeout=timeout,
                label=f"generate_i2v_motion_prompts[{i + 1}/{n}]",
            )
        except Exception as exc:
            logger.warning(
                "generate_i2v_motion_prompts: image %d/%d failed (%s: %s) — leaving slot empty",
                i + 1, n, type(exc).__name__, exc,
            )
            prompts.append("")
            failures += 1
            continue
        prompt = _extract_animation(parsed)
        if not prompt:
            logger.warning(
                "generate_i2v_motion_prompts: image %d/%d returned no usable prompt", i + 1, n,
            )
        prompts.append(prompt)

    if failures == n:
        raise RuntimeError("generate_i2v_motion_prompts: every per-image call failed")
    return prompts


def _clean_titles(parsed: dict, *, n: int) -> list[str]:
    raw = parsed.get("titles") or []
    if not isinstance(raw, list):
        raise RuntimeError(f"Titler VLM 'titles' field is not a list: {type(raw).__name__}")

    seen: set[str] = set()
    out: list[str] = []
    for t in raw:
        s = str(t).strip().strip('"').strip("'").rstrip(".").strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out[:n]
