"""
Stand-alone probe: stream qwen3.5 article generation so we can see where
it stalls (thinking-mode tokens? prompt eval? actual generation?).

Usage:  probe_article.py [model] [edge]
"""
import asyncio
import base64
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import httpx

from core.imaging import prepare_jpg_for_web

OLLAMA_HOST = "http://localhost:11434"
SRC = ROOT / "storage/images/2026/05/e22adc86-8c9e-4e55-9f61-1f1f69169686_z-image_01869_.png"

# Tiny prompt — strip voice guide, just ask for one short article triple
SYSTEM_PROMPT = """You are an art-blog writer. Look at the image and return STRICT JSON:
{
  "de": {"title": "...", "body_md": "..."},
  "en": {"title": "...", "body_md": "..."},
  "zh": {"title": "...", "body_md": "..."}
}
title: 1-5 words. body_md: 2-3 short paragraphs about the artwork. JSON only, no commentary."""


async def main(model: str, edge: int, think: bool):
    print(f"[probe] model={model} edge={edge} think={think}", flush=True)
    jpg, _ = await prepare_jpg_for_web(SRC, max_edge=edge, quality=80)
    print(f"[probe] {len(jpg)//1024}KB encoded", flush=True)

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Write the article triple now.",
             "images": [base64.b64encode(jpg).decode("ascii")]},
        ],
        "format": "json",
        "stream": True,
        "think": think,                    # disable thinking explicitly
        "options": {"temperature": 0.5, "num_predict": 600},
    }

    t0 = time.monotonic()
    full = ""
    last = t0
    print("[probe] POST...", flush=True)
    async with httpx.AsyncClient(timeout=900) as client:
        async with client.stream("POST", f"{OLLAMA_HOST}/api/chat", json=payload) as r:
            print(f"[probe] connect {r.status_code} after {time.monotonic()-t0:.1f}s", flush=True)
            async for line in r.aiter_lines():
                if not line:
                    continue
                obj = json.loads(line)
                if "error" in obj:
                    print(f"[probe] ERROR: {obj['error']}", flush=True)
                    return
                msg = obj.get("message", {})
                chunk = msg.get("content", "")
                think_chunk = msg.get("thinking", "")
                if think_chunk:
                    full_think_indicator = "[THINK] " + think_chunk[:80]
                    print(f"[probe] +{time.monotonic()-t0:5.1f}s {full_think_indicator}", flush=True)
                full += chunk
                now = time.monotonic()
                if chunk and (now - last) > 1.0:
                    print(f"[probe] +{now-t0:5.1f}s {len(full):4d} chars  tail={full[-60:]!r}", flush=True)
                    last = now
                if obj.get("done"):
                    print(f"[probe] DONE in {time.monotonic()-t0:.1f}s, {len(full)} chars", flush=True)
                    pe = obj.get("prompt_eval_count")
                    pd = obj.get("prompt_eval_duration", 0) / 1e9
                    ec = obj.get("eval_count")
                    ed = obj.get("eval_duration", 0) / 1e9
                    print(f"[probe] prompt_eval = {pe} tok / {pd:.1f}s ({pe/pd:.0f} tok/s)" if pe and pd else f"[probe] prompt_eval = {pe} / {pd}")
                    print(f"[probe] eval        = {ec} tok / {ed:.1f}s ({ec/ed:.0f} tok/s)" if ec and ed else f"[probe] eval = {ec} / {ed}")
                    print(f"[probe] full output (first 500 chars): {full[:500]}", flush=True)
                    try:
                        parsed = json.loads(full)
                        print(f"[probe] PARSED keys: {sorted(parsed.keys())}")
                    except Exception as exc:
                        print(f"[probe] PARSE FAIL: {exc}")
                    return


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "qwen3.5:latest"
    edge = int(sys.argv[2]) if len(sys.argv) > 2 else 384
    think = sys.argv[3].lower() == "true" if len(sys.argv) > 3 else False
    asyncio.run(main(model, edge, think))
