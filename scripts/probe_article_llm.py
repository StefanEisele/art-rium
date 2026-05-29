"""
Smoke-test the article LLM (qwen3.6:27b) against the live Ollama, mirroring
the real write_modal_article call path: think=false, format=json,
num_ctx=24576, several small images. Reproduces (or clears) the
ggml_cuda_cpy CPU-offload crash in ~15s if it still fails, or ~4-5 min
(cold load + generate) if it now works.
"""
import base64
import sys
import time
from io import BytesIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
from PIL import Image

from core.config import settings


def _img(seed: int, size: int = 512) -> str:
    img = Image.new("RGB", (size, size), (40, 80, 200))
    px = img.load()
    for y in range(size):
        for x in range(0, size, 4):
            px[x, y] = ((x * 255) // size, (y * 255) // size, (seed * 40) % 255)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode("ascii")


_model = sys.argv[1] if len(sys.argv) > 1 else settings.ollama_llm_model

payload = {
    "model": _model,
    "messages": [
        {
            "role": "user",
            "content": 'Look at these images and return STRICT JSON: {"summary": "<one sentence>"}.',
            "images": [_img(1), _img(2), _img(3), _img(4), _img(5)],
        }
    ],
    "format": "json",
    "stream": False,
    "think": False,
    "options": {"temperature": 0.65, "num_ctx": 24576, "num_predict": 128},
}

print(f"Model: {_model}  (5 images, num_ctx=24576)")
print("Posting to /api/chat ... (cold load can take several minutes)")
t0 = time.time()
try:
    with httpx.Client(timeout=600.0) as client:
        r = client.post(f"{settings.ollama_host}/api/chat", json=payload)
    dt = time.time() - t0
    print(f"HTTP {r.status_code}  in {dt:.1f}s")
    if r.status_code == 200:
        data = r.json()
        content = (data.get("message", {}) or {}).get("content", "")
        print("SUCCESS — model produced output:")
        print(content[:500])
    else:
        print("FAILED — body:")
        print(r.text[:800])
except Exception as exc:
    dt = time.time() - t0
    print(f"EXCEPTION after {dt:.1f}s: {type(exc).__name__}: {exc}")
