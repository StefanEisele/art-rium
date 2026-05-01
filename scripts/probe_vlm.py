"""
Stand-alone probe: re-encode the newest gallery image and call Ollama
with the SAME payload services/ollama/client.py uses, but with verbose
streaming so we can see whether the model is generating tokens or
truly hung.
"""
import asyncio
import base64
import json
import sys
import time
from pathlib import Path

# Project root on path so we can reuse core.imaging
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import httpx

from core.imaging import prepare_jpg_for_web

OLLAMA_HOST = "http://localhost:11434"
SRC = ROOT / "storage/images/2026/05/fb07ecae-c66f-4fc0-ab99-b947582dbc6d_z-image_01870_.png"

SYSTEM_PROMPT = """You are an art-blog image analyst. Analyze the artwork image and return STRICT JSON with exactly three fields:
  alt_text: one sentence describing what is visibly in the image, suitable for screen readers. Maximum 125 characters. Concrete (subject, composition, palette). No interpretation, no marketing language.
  seo_description: one sentence describing the artwork's mood and subject for use as a meta description on a blog post. Maximum 155 characters. Concrete imagery, no buzzwords, no superlatives.
  caption: 1-2 sentences for the WordPress media library, slightly more descriptive than alt_text, may include subject and atmosphere. Maximum 300 characters.

Use the provided title and notes as context but do not quote them verbatim. Write in the requested language. Return ONLY the JSON object — no prose, no code fences, no commentary."""


async def main(model: str, max_edge: int):
    print(f"[probe] src    = {SRC.name}", flush=True)
    print(f"[probe] model  = {model}", flush=True)
    print(f"[probe] edge   = {max_edge}", flush=True)

    t0 = time.monotonic()
    jpg_bytes, _ = await prepare_jpg_for_web(SRC, max_edge=max_edge, quality=80)
    print(f"[probe] encoded {len(jpg_bytes)//1024} KB in {time.monotonic()-t0:.2f}s", flush=True)

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "Title: (none)\nNotes: (none)\n\nLanguage for output: en",
                "images": [base64.b64encode(jpg_bytes).decode("ascii")],
            },
        ],
        "format": "json",
        "stream": True,           # stream so we see tokens as they arrive
        "options": {"temperature": 0.4},
    }

    t0 = time.monotonic()
    print("[probe] POST /api/chat (streaming)...", flush=True)
    full = ""
    last_tick = t0
    async with httpx.AsyncClient(timeout=600) as client:
        async with client.stream("POST", f"{OLLAMA_HOST}/api/chat", json=payload) as r:
            print(f"[probe] connect {r.status_code} after {time.monotonic()-t0:.2f}s", flush=True)
            async for line in r.aiter_lines():
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    print(f"[probe] non-json line: {line[:200]}", flush=True)
                    continue
                if "error" in obj:
                    print(f"[probe] ERROR: {obj['error']}", flush=True)
                    return
                chunk = obj.get("message", {}).get("content", "")
                full += chunk
                now = time.monotonic()
                if chunk and (now - last_tick) > 1.0:
                    print(f"[probe] +{now-t0:5.1f}s  {len(full):4d} chars  tail={full[-60:]!r}", flush=True)
                    last_tick = now
                if obj.get("done"):
                    print(f"[probe] DONE in {time.monotonic()-t0:.1f}s, {len(full)} chars", flush=True)
                    pe_cnt = obj.get("prompt_eval_count")
                    pe_dur = obj.get("prompt_eval_duration", 0) / 1e9
                    ev_cnt = obj.get("eval_count")
                    ev_dur = obj.get("eval_duration", 0) / 1e9
                    ld_dur = obj.get("load_duration", 0) / 1e9
                    print(f"[probe] load_duration       = {ld_dur:.2f}s")
                    print(f"[probe] prompt_eval_count   = {pe_cnt}  ({pe_cnt/pe_dur:.1f} tok/s)" if pe_cnt and pe_dur else f"[probe] prompt_eval_count   = {pe_cnt}")
                    print(f"[probe] prompt_eval_duration= {pe_dur:.2f}s")
                    print(f"[probe] eval_count          = {ev_cnt}  ({ev_cnt/ev_dur:.1f} tok/s)" if ev_cnt and ev_dur else f"[probe] eval_count          = {ev_cnt}")
                    print(f"[probe] eval_duration       = {ev_dur:.2f}s")
                    print("[probe] full output:", full)
                    try:
                        print("[probe] parsed:", json.loads(full))
                    except Exception as exc:
                        print(f"[probe] PARSE FAIL: {exc}")
                    return


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5vl:latest"
    edge = int(sys.argv[2]) if len(sys.argv) > 2 else 512
    asyncio.run(main(model, edge))
