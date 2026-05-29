"""
Text-only probe for the decoupled article writer.

Mirrors the planned text-only call path: loads the REAL system prompt
(voice-system.md + mode-essay.md), sends a representative text-only user
message (image descriptions instead of JPEGs — the decoupled-vision design),
and requests STRICT JSON with think=false, format=json, num_ctx=16384.

Validates two risks for a freshly-pulled community quant:
  1. Does it load GPU-only and run at num_ctx=16384 without crashing?
     (check `ollama ps` PROCESSOR column separately — should be 100% GPU)
  2. Does the model's chat template follow our prompt and emit valid JSON
     with the Essay-mode schema keys?

Usage:
  python scripts/probe_text_article_llm.py [model]
  (defaults to settings.ollama_llm_model)
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx

from core.config import settings

_PROMPTS = Path(__file__).resolve().parent.parent / "prompts"

_model = sys.argv[1] if len(sys.argv) > 1 else settings.ollama_llm_model

system = (
    (_PROMPTS / "voice-system.md").read_text(encoding="utf-8")
    + "\n\n---\n\n"
    + (_PROMPTS / "mode-essay.md").read_text(encoding="utf-8")
)

# Representative text-only user message: image descriptions stand in for the
# JPEGs the model used to see. This is what the decoupled caption pass feeds.
user_text = """MODE: essay

Perspective: first person — the artist's own argument. Cite real sources with date + venue only.

Author's intent / context for this article (anchor the prose in these specifics — do not quote verbatim):
The series confronts AI Slop as a symptom of human disengagement, not a failure of the models. I want the thesis to land hard: the slop is the visible tip of an iceberg of abdicated authorship.

Number of images attached: 3

Per-image visual descriptions (these replace the images — anchor concrete observations in them):
  Image 1: A corroded steel surface in rust orange and dim blue, oxide blooms spreading from a central weld seam; the edges erode into granular noise where the diffusion model lost coherence.
  Image 2: A vertical fold of weathered metal, patina layered like sediment; dim-blue shadow pooling in the crease, the texture too regular to be photographic yet too damaged to be clean render.
  Image 3: A close crop of flaking lacquer over orange substrate, cracks forming a dry river-delta pattern; the algorithmic origin shows in the unnaturally even crack spacing.

Write the article now. JSON only."""

payload = {
    "model": _model,
    "messages": [
        {"role": "system", "content": system},
        {"role": "user", "content": user_text},
    ],
    "format": "json",
    "stream": False,
    "think": False,
    "options": {"temperature": 0.65, "num_ctx": 16384, "num_predict": 6000},
}

print(f"Model: {_model}  (TEXT-ONLY, num_ctx=16384, num_predict=6000)")
print("Posting to /api/chat ... (cold load can take a few minutes)")
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
        print(f"INVALID JSON: {exc}")
        print("content head:")
        print(content[:800])
        sys.exit(2)
    print("Valid JSON parsed.")

    # Essay schema check, per language.
    expected = {"title", "intro", "movements", "closing", "excerpt",
                "meta_description", "focus_keyphrase", "tags"}
    for lang in ("en", "de"):
        block = parsed.get(lang)
        if not isinstance(block, dict):
            print(f"  [{lang}] MISSING or not an object")
            continue
        present = set(block.keys())
        missing = expected - present
        n_mv = len(block.get("movements") or []) if isinstance(block.get("movements"), list) else 0
        tag = "OK" if not missing else f"MISSING {sorted(missing)}"
        print(f"  [{lang}] {tag}  | movements={n_mv}  title={str(block.get('title'))[:60]!r}")

    print("\n--- EN intro[0] sample ---")
    en = parsed.get("en") or {}
    intro = en.get("intro") or []
    if isinstance(intro, list) and intro:
        print(str(intro[0])[:400])
    print("\n--- DE intro[0] sample ---")
    de = parsed.get("de") or {}
    dintro = de.get("intro") or []
    if isinstance(dintro, list) and dintro:
        print(str(dintro[0])[:400])

except Exception as exc:
    dt = time.time() - t0
    print(f"EXCEPTION after {dt:.1f}s: {type(exc).__name__}: {exc}")
    sys.exit(3)
