"""
Vision probe for a candidate article writer (e.g. Gemma 3 12B).

Mirrors the REAL article call path: loads the real system prompt
(voice-system.md + mode-essay.md), attaches 3 real gallery thumbnails,
and requests STRICT JSON with think=false, format=json, num_ctx=16384 —
exactly what services/ollama/client.py:_chat_json sends.

Validates:
  1. Model loads GPU-only and runs at num_ctx=16384 without crashing.
  2. Ollama accepts `think=false` for this model (Gemma is NOT a thinking
     model — if Ollama rejects the field, the real pipeline breaks on swap).
  3. The model follows OUR Qwen-tuned prompt: valid JSON, Essay schema,
     EN+DE blocks. Prints samples so voice/German can be eyeballed.

Usage:
  python scripts/probe_vision_article_llm.py [model]
"""
import base64
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

from core.config import settings

_PROMPTS = Path(__file__).resolve().parent.parent / "prompts"
_THUMBS = settings.storage_dir / "thumbnails"

_model = sys.argv[1] if len(sys.argv) > 1 else settings.ollama_llm_model
_num_ctx = int(sys.argv[2]) if len(sys.argv) > 2 else 16384

# Pick 3 real thumbnails as stand-in gallery images.
thumbs = sorted(_THUMBS.glob("*.jpg"))[:3]
if not thumbs:
    print(f"No thumbnails under {_THUMBS}")
    sys.exit(4)
images_b64 = [base64.b64encode(p.read_bytes()).decode("ascii") for p in thumbs]
print(f"Using {len(images_b64)} thumbnails: {[p.name for p in thumbs]}")

system = (
    (_PROMPTS / "voice-system.md").read_text(encoding="utf-8")
    + "\n\n---\n\n"
    + (_PROMPTS / "mode-essay.md").read_text(encoding="utf-8")
)

user_text = """MODE: essay

Perspective: first person — the artist's own argument. Cite real sources with date + venue only.

Author's intent / context for this article (anchor the prose in these specifics — do not quote verbatim):
The series confronts AI Slop as a symptom of human disengagement, not a failure of the models. The thesis should land hard: the slop is the visible tip of an iceberg of abdicated authorship.

Number of images attached: 3

Per-image metadata (use as context, do not quote verbatim):
  Image 1: rust orange and dim blue corroded surface, oxide bloom from a weld seam
  Image 2: vertical fold of weathered metal, patina layered like sediment
  Image 3: flaking lacquer over orange substrate, cracks in a dry river-delta pattern

Write the article now. JSON only."""

payload = {
    "model": _model,
    "messages": [
        {"role": "system", "content": system},
        {"role": "user", "content": user_text, "images": images_b64},
    ],
    "format": "json",
    "stream": False,
    "think": False,
    "options": {"temperature": 0.65, "num_ctx": _num_ctx, "num_predict": 6000},
}

print(f"Model: {_model}  (VISION, 3 imgs, num_ctx={_num_ctx}, think=false)")
t0 = time.time()
try:
    with httpx.Client(timeout=900.0) as client:
        r = client.post(f"{settings.ollama_host}/api/chat", json=payload)
    dt = time.time() - t0
    print(f"HTTP {r.status_code}  in {dt:.1f}s")
    if r.status_code != 200:
        print("FAILED — body:")
        print(r.text[:1000])
        sys.exit(1)

    data = r.json()
    content = (data.get("message", {}) or {}).get("content", "") or ""
    load_ms = (data.get("load_duration") or 0) / 1e6
    eval_count = data.get("eval_count") or 0
    eval_ms = (data.get("eval_duration") or 0) / 1e6
    tok_s = (eval_count / (eval_ms / 1000)) if eval_ms else 0
    print(f"load={load_ms/1000:.1f}s  output_tokens={eval_count}  speed={tok_s:.1f} tok/s")

    print("\n--- JSON validity ---")
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        print(f"INVALID JSON: {exc}\ncontent head:\n{content[:800]}")
        sys.exit(2)
    print("Valid JSON parsed.")

    expected = {"title", "intro", "movements", "closing", "excerpt",
                "meta_description", "focus_keyphrase", "tags"}
    for lang in ("en", "de"):
        block = parsed.get(lang)
        if not isinstance(block, dict):
            print(f"  [{lang}] MISSING or not an object")
            continue
        missing = expected - set(block.keys())
        mv = block.get("movements") or []
        n_mv = len(mv) if isinstance(mv, list) else 0
        headings = [m.get("heading") for m in mv if isinstance(m, dict)][:3] if isinstance(mv, list) else []
        tag = "OK" if not missing else f"MISSING {sorted(missing)}"
        print(f"  [{lang}] {tag}  | movements={n_mv}  fkp={str(block.get('focus_keyphrase'))!r}")
        print(f"        title: {str(block.get('title'))[:70]!r}")
        print(f"        headings: {headings}")

    for lang in ("en", "de"):
        block = parsed.get(lang) or {}
        intro = block.get("intro") or []
        print(f"\n--- {lang.upper()} intro[0] ---")
        if isinstance(intro, list) and intro:
            print(str(intro[0])[:500])
        elif isinstance(intro, str):
            print(intro[:500])

except SystemExit:
    raise
except Exception as exc:
    dt = time.time() - t0
    print(f"EXCEPTION after {dt:.1f}s: {type(exc).__name__}: {exc}")
    sys.exit(3)
