"""
Ollama VLM analysis — per-image alt-text/SEO metadata, title suggestions
(image + video), and the titler warm-up call.
"""
import base64
import json
import logging
import re
from collections.abc import Callable

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


_ANIMATION_KEY = re.compile(r'"animation"\s*:\s*"')


def _trim_dangling(text: str) -> str:
    """Cut a truncated prompt back to its last complete sentence, or failing
    that drop the final (half-written) word and close it off."""
    text = text.strip()
    cut = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
    if cut > 0:
        return text[: cut + 1]
    head = text.rsplit(" ", 1)[0].rstrip(" ,;:—-") if " " in text else ""
    return f"{head}." if head else ""


def _salvage_truncated_animation(s: str) -> str:
    """Recover the sentence when num_predict cuts the model off mid-string,
    leaving `{"animation": "…` with no closing quote or brace — the JSON
    parser reports an unterminated string and the image's slot would
    otherwise come back empty.

    A token cap cannot shorten a JSON response, only break it; this is what
    makes capping safe. Returns *s* unchanged when the string is properly
    closed (so the parse failed for some other reason) or when nothing usable
    was written, leaving `_chat_json`'s normal failure path in charge.
    """
    m = _ANIMATION_KEY.search(s)
    if not m:
        return s
    i, n = m.end(), len(s)
    buf: list[str] = []
    while i < n:
        c = s[i]
        if c == "\\" and i + 1 < n:
            buf.append(s[i:i + 2])
            i += 2
            continue
        if c == '"':
            return s  # properly terminated — not the truncation case
        buf.append(c)
        i += 1

    try:
        text = json.loads('"' + "".join(buf).rstrip("\\") + '"')
    except json.JSONDecodeError:
        return s
    trimmed = _trim_dangling(text)
    return json.dumps({"animation": trimmed}) if trimmed else s


# A legitimate prompt describes the image in front of the model; it never
# shares this many consecutive words with the instructions it was given, nor
# with the prompt written for a different image.
_LEAK_NGRAM = 8
_WORD_RE = re.compile(r"[a-z0-9']+")


def _word_ngrams(text: str, n: int = _LEAK_NGRAM) -> set[tuple[str, ...]]:
    words = _WORD_RE.findall(text.lower())
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def first_sentence(prompt: str) -> str:
    """Keep only the opening sentence.

    A one-beat prompt is what stops the video model cutting, and sentence two
    is where a second beat lands. The system prompt asks for one sentence and
    the model mostly complies, but "mostly" is not a guarantee the generator
    can rely on — this makes it one deterministically, at no extra model call.
    Anything lost is a further beat we did not want.
    """
    parts = [p for p in _SENTENCE_SPLIT.split(prompt.strip()) if p.strip()]
    return parts[0].strip() if parts else ""


# Language that announces a second shot. "Revealing"/"reveals" matter most:
# a camera move that brings something new into frame is, to the video model,
# a cut to a different view — the exact artefact being chased here.
_SHOT_BREAK_RE = re.compile(
    r"\b(?:revealing|reveals?|then|suddenly|meanwhile|afterwards?|"
    r"transform(?:s|ing) into|cut(?:s|ting) to|scene (?:shifts?|changes?))\b",
    re.IGNORECASE,
)

_REASON_COPIED = "copied wording from the system prompt"
_REASON_REPEAT = "repeats an earlier image's prompt"
_REASON_EMPTY = "empty response"
_REASON_SHOT_BREAK = "shot-break language"


def _reject_reason(
    prompt: str,
    system_ngrams: set[tuple[str, ...]],
    accepted: list[str],
    *,
    ban_shot_breaks: bool,
) -> str | None:
    """Why *prompt* should be re-asked, or None if it is usable.

    Small vision models (qwen2.5vl:3b) under a long system prompt tend to
    return one of the worked examples verbatim instead of looking at the
    image, and to repeat one prompt across every image of a batch. Both
    produce text that does not describe the picture it is attached to —
    which is exactly what makes the video model drift away from the source
    image and cut mid-clip.
    """
    if not prompt:
        return _REASON_EMPTY
    grams = _word_ngrams(prompt)
    if grams & system_ngrams:
        return _REASON_COPIED
    if any(grams & _word_ngrams(earlier) for earlier in accepted):
        return _REASON_REPEAT
    if ban_shot_breaks and _SHOT_BREAK_RE.search(prompt):
        return _REASON_SHOT_BREAK
    return None


# One corrective retry per image. Beyond that the model is unlikely to find a
# different answer, and N images × M attempts is real wall-clock time in a
# job the user is watching.
_MOTION_ATTEMPTS = 2

_NUDGE_FRESH = (
    "\n\nYour previous answer did not describe THIS image — it reused wording "
    "from the instructions or from another image. Look at the picture again "
    "and write a new sentence about what is actually in it, in your own words."
)
_NUDGE_ONE_SHOT = (
    "\n\nYour previous answer had the camera reveal something new, or a second "
    "thing happening after the first. Both read as a cut. Write one sentence "
    "about a single thing already visible in the image moving, frame static."
)
_RETRY_NUDGES = {
    _REASON_EMPTY: _NUDGE_FRESH,
    _REASON_COPIED: _NUDGE_FRESH,
    _REASON_REPEAT: _NUDGE_FRESH,
    _REASON_SHOT_BREAK: _NUDGE_ONE_SHOT,
}


async def _generate_per_image_motion_prompts(
    jpgs: list[bytes],
    *,
    system_file: str,
    instruction: str,
    label: str,
    context: str = "",
    timeout: float = 300.0,
    on_progress: Callable[[int, int], None] | None = None,
    llm_options: dict | None = None,
    postprocess: Callable[[str], str] | None = None,
    ban_shot_breaks: bool = False,
) -> list[str]:
    """
    Shared per-image loop behind generate_i2v_motion_prompts (Wan2.2) and
    generate_ltx_motion_prompts (LTX-2.3) — same VLM, same fan-out rationale,
    different system prompt / instruction text per model family.

    Given N images (each becomes its own independent i2v clip, in playback
    order), ask the titler VLM for one surreal animation prompt per image —
    one vision call PER image. Rationale: the small titler VLM (qwen2.5vl:3b)
    effectively only looks at the first image of a multi-image message and
    repeats one prompt N times. Serialized single-image calls fix that; the
    model stays resident between calls (keep_alive), so each extra call costs
    inference only, no reload. *timeout* applies per call.

    *context* is optional free text about the intended video, folded into
    every per-image message when present.

    N serialized vision calls easily add up past a minute — *on_progress*,
    when given, is called (completed, total) after each image so a caller
    running this as a background job (see routers/video.py's suggest-i2v
    job) can report incremental status instead of leaving the client to
    guess for the whole duration.

    *llm_options* overrides the default `{"temperature": 0.7}` Ollama options
    (e.g. to add a num_predict cap) without touching the other caller.

    Each image gets up to _MOTION_ATTEMPTS tries: a response that copied the
    system prompt's examples or repeated an earlier image's prompt is re-asked
    once (see _reject_reason) rather than handed to the user, since such a
    prompt describes a different picture than the one it is attached to.
    *ban_shot_breaks* additionally re-asks on language that announces a second
    shot; *postprocess* (e.g. first_sentence) tightens each response before it
    is judged, so a fixable answer is fixed rather than spending a retry.

    Returns exactly len(jpgs) strings; an image whose call fails yields ""
    (logged), so the client just shows an empty textarea for that slot.
    Raises RuntimeError if jpgs is empty or if EVERY per-image call failed.
    """
    if not jpgs:
        raise RuntimeError(f"{label} requires at least 1 image")

    options = llm_options if llm_options is not None else {"temperature": 0.7}
    system = _read_prompt(system_file)
    system_ngrams = _word_ngrams(system)
    n = len(jpgs)
    context_block = (
        f"Context for the whole video (what it is about / intended mood):\n"
        f"{context.strip()}\n\n"
        if context.strip() else ""
    )

    prompts: list[str] = []
    failures = 0
    for i, jpg in enumerate(jpgs):
        base_user_text = (
            f"This is image {i + 1} of {n} for one video — each image becomes "
            f"its own independent clip, played in order. {instruction}\n\n"
            f"{context_block}"
            f'Return STRICT JSON: {{"animation": "<the prompt>"}}.'
        )

        prompt = ""
        nudge = ""
        for attempt in range(1, _MOTION_ATTEMPTS + 1):
            try:
                parsed = await _chat_json(
                    model=settings.ollama_titler_model,
                    system=system,
                    user_text=base_user_text + nudge,
                    jpgs=[jpg],
                    # Retries climb in temperature — repeating the same call at
                    # the same settings mostly reproduces the same answer.
                    options=options if attempt == 1 else {**options, "temperature": 0.95},
                    keep_alive=_TITLER_KEEP_ALIVE,
                    timeout=timeout,
                    salvage=_salvage_truncated_animation,
                    label=f"{label}[{i + 1}/{n}]",
                )
            except Exception as exc:
                # Transport/parse failures don't get better from a reword.
                logger.warning(
                    "%s: image %d/%d failed (%s: %s) — leaving slot empty",
                    label, i + 1, n, type(exc).__name__, exc,
                )
                prompt = ""
                failures += 1
                break

            prompt = _extract_animation(parsed)
            if postprocess is not None:
                prompt = postprocess(prompt)
            reason = _reject_reason(
                prompt, system_ngrams, prompts, ban_shot_breaks=ban_shot_breaks,
            )
            if reason is None:
                break
            logger.warning(
                "%s: image %d/%d attempt %d rejected (%s)%s",
                label, i + 1, n, attempt, reason,
                "" if attempt < _MOTION_ATTEMPTS else " — keeping it anyway, out of attempts",
            )
            nudge = _RETRY_NUDGES[reason]

        prompts.append(prompt)
        if on_progress is not None:
            on_progress(i + 1, n)

    if failures == n:
        raise RuntimeError(f"{label}: every per-image call failed")
    return prompts


async def generate_i2v_motion_prompts(
    jpgs: list[bytes],
    *,
    context: str = "",
    timeout: float = 300.0,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[str]:
    """One surreal Wan2.2-style animation prompt per image (no audio — Wan2.2
    is silent). System prompt lives in prompts/video-i2v-motion.md."""
    return await _generate_per_image_motion_prompts(
        jpgs,
        system_file="video-i2v-motion.md",
        instruction=(
            "Write ONE surreal animation prompt for THIS image, following "
            "the system instructions."
        ),
        label="generate_i2v_motion_prompts",
        context=context,
        timeout=timeout,
        on_progress=on_progress,
    )


async def generate_ltx_motion_prompts(
    jpgs: list[bytes],
    *,
    context: str = "",
    timeout: float = 300.0,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[str]:
    """One surreal LTX-2.3-style animation+audio prompt per image — LTX-2.3
    generates native audio synced to the prompt, so unlike the Wan variant
    this also describes the clip's soundscape. System prompt lives in
    prompts/video-ltx-motion.md.

    LTX-2.3 reads a multi-beat prompt as multiple shots and cuts between
    them, so the system prompt asks for one 12-25 word sentence. num_predict
    is the backstop against the model ignoring that: ~90 tokens leaves room
    for that sentence plus its JSON wrapper (roughly 45 tokens) while making
    a multi-sentence ramble impossible. The earlier value of 220 never bound
    — measured output tops out near 75 tokens — so it capped nothing.

    A cap on JSON output truncates mid-string rather than shortening the
    answer, which is why _salvage_truncated_animation exists; without it a
    cap that ever bit would blank the slot instead of trimming it."""
    return await _generate_per_image_motion_prompts(
        jpgs,
        system_file="video-ltx-motion.md",
        instruction=(
            "Write ONE sentence of 12-25 words animating THIS image — one "
            "moving thing, its sound, one camera treatment. No second action."
        ),
        label="generate_ltx_motion_prompts",
        context=context,
        timeout=timeout,
        on_progress=on_progress,
        llm_options={"temperature": 0.7, "num_predict": 90},
        postprocess=first_sentence,
        ban_shot_breaks=True,
    )


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
