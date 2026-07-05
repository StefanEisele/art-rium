import copy
import json
import logging
import random
import re
import secrets
import uuid
from pathlib import Path

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.auth import require_auth, ws_auth_ok
from core.comfy import post_prompt
from core.config import settings
from core.db import get_db
from core.loras import ALLOWED_LORAS, DEFAULT_LORA, LORAS
from core.models import Image
from core.thumbnail import make_thumbnail, thumb_rel_path

logger = logging.getLogger(__name__)
router = APIRouter()

# Workflow template loaded once at import time
_TEMPLATE = json.loads(
    (Path(__file__).parent.parent / "workflows" / "z-image_turbo.json").read_text()
)


def _build_workflow(
    prompt: str, seed: int, width: int, height: int,
    lora_name: str, lora_strength: float,
) -> dict:
    wf = copy.deepcopy(_TEMPLATE)
    wf["45"]["inputs"]["text"] = prompt
    wf["44"]["inputs"]["seed"] = seed if seed >= 0 else random.randint(0, 2**32 - 1)
    wf["41"]["inputs"]["width"] = width
    wf["41"]["inputs"]["height"] = height
    wf["51"]["inputs"]["lora_name"] = lora_name
    wf["51"]["inputs"]["strength_model"] = round(max(0.0, min(1.0, lora_strength)), 3)
    return wf


class GenerateRequest(BaseModel):
    prompt: str = ""
    prompts: list[str] | None = None  # when set, overrides prompt + batch_count: one submission per item
    seed: int = -1
    width: int = 1024
    height: int = 1024
    client_id: str
    batch_count: int = 1
    lora_name: str = DEFAULT_LORA
    lora_strength: float = 0.5


class EnhancePromptsRequest(BaseModel):
    idea: str
    n: int = 6


@router.get("/api/loras", dependencies=[Depends(require_auth)])
async def list_loras():
    """Return the LoRA catalogue used by the Z-Image picker (SSOT for the frontend)."""
    return {"loras": LORAS, "default": DEFAULT_LORA}


@router.post("/api/generate", dependencies=[Depends(require_auth)])
async def generate(req: GenerateRequest, request: Request):
    # Resolve the prompt list — either one-prompt × batch_count (classic mode)
    # or an explicit list of N distinct prompts (enhancer mode).
    if req.prompts:
        prompt_list = [p.strip() for p in req.prompts if p.strip()]
        if not prompt_list:
            raise HTTPException(status_code=400, detail="prompts list is empty")
        if len(prompt_list) > 10:
            raise HTTPException(status_code=400, detail="At most 10 prompts per batch")
    else:
        if not req.prompt.strip():
            raise HTTPException(status_code=400, detail="Prompt is required")
        batch_count = max(1, min(10, req.batch_count))
        prompt_list = [req.prompt] * batch_count

    if req.lora_name not in ALLOWED_LORAS:
        raise HTTPException(status_code=400, detail=f"Unknown LoRA: {req.lora_name}")

    listener = request.app.state.comfy_listener
    total = len(prompt_list)
    batch_id = str(uuid.uuid4())
    prompt_ids = []

    for i, prompt_text in enumerate(prompt_list):
        seed = (req.seed + i) if req.seed >= 0 else random.randint(0, 2**32 - 1)
        workflow = _build_workflow(prompt_text, seed, req.width, req.height, req.lora_name, req.lora_strength)

        result = await post_prompt(workflow, listener.client_id)

        prompt_id = result.get("prompt_id")
        if not prompt_id:
            raise HTTPException(status_code=500, detail="ComfyUI did not return prompt_id")

        listener.register_prompt(
            prompt_id=prompt_id,
            client_id=req.client_id,
            index=i + 1,
            total=total,
            batch_id=batch_id,
            prompt_text=prompt_text,
            seed=seed,
            width=req.width,
            height=req.height,
        )
        prompt_ids.append(prompt_id)
        logger.info(f"Queued [{i+1}/{total}] prompt={prompt_id}")

    return {"batch_id": batch_id, "prompt_ids": prompt_ids, "batch_count": total}


@router.post("/api/prompts/enhance", dependencies=[Depends(require_auth)])
async def enhance_prompts(body: EnhancePromptsRequest):
    """Run the local Qwen prompt-enhancer over a short idea and return one
    prompt per style family (A..F default, +G if n=7). Styles that fail
    are silently skipped — the response may contain fewer than n entries."""
    from services.ollama.zimage_enhance import enhance_zimage_prompts

    if not body.idea.strip():
        raise HTTPException(status_code=400, detail="idea is required")
    if not (1 <= body.n <= 7):
        raise HTTPException(status_code=400, detail="n must be between 1 and 7")

    try:
        prompts = await enhance_zimage_prompts(body.idea, n=body.n)
    except Exception as exc:
        logger.error("enhance_prompts failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=502, detail=f"{type(exc).__name__}: {exc}")

    if not prompts:
        raise HTTPException(
            status_code=502,
            detail="Prompt enhancer returned no usable variants (all styles failed).",
        )
    return {"requested": body.n, "produced": len(prompts), "prompts": prompts}


def _verify_share_token(token: str = "") -> None:
    """Public-share gate: when IMAGE_SHARE_TOKEN is set, the request must
    carry it as ?token=…. When unset (dev mode), the endpoints are open."""
    if settings.image_share_token and not secrets.compare_digest(token, settings.image_share_token):
        raise HTTPException(status_code=403, detail="Invalid share token")


_SAFE_FILENAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def _validate_share_filename(filename: str) -> str:
    """Reduce the path param to a bare filename and reject anything outside
    a strict charset. `Path(...).name` alone strips directory components but
    lets glob metacharacters (`*`, `?`, `[...]`) through, which turns
    `_find_image_on_disk`'s `rglob()` into a file-enumeration primitive on a
    public, single-static-token endpoint."""
    name = Path(filename).name
    if not name or not _SAFE_FILENAME_RE.match(name):
        raise HTTPException(status_code=400, detail="Invalid filename")
    return name


def _find_image_on_disk(safe_name: str) -> Path | None:
    """Locate an image by its bare filename. Looks in managed storage first,
    then the raw ComfyUI output directory. Returns None when missing.
    `safe_name` must come from `_validate_share_filename` — it is used as an
    `rglob` pattern for recursive lookup across date-sharded subdirectories,
    and an unvalidated name could contain glob metacharacters."""
    for search_dir in (settings.images_dir, settings.comfyui_output_dir):
        for candidate in search_dir.rglob(safe_name):
            if candidate.is_file():
                return candidate
    return None


@router.get("/share/image/{filename}", dependencies=[Depends(_verify_share_token)])
async def get_shared_image(filename: str):
    """Public image endpoint for external services (e.g. Instagram)."""
    candidate = _find_image_on_disk(_validate_share_filename(filename))
    if candidate is None:
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(candidate, media_type="image/png")


@router.get("/share/reel/{filename}", dependencies=[Depends(_verify_share_token)])
async def get_shared_reel(filename: str):
    """Public video endpoint for Reel uploads to Instagram Graph API."""
    safe_name = _validate_share_filename(filename)
    candidate = settings.reels_dir / safe_name
    if candidate.is_file():
        return FileResponse(candidate, media_type="video/mp4")
    raise HTTPException(status_code=404, detail="Reel not found")


@router.get("/share/video/{filename}", dependencies=[Depends(_verify_share_token)])
async def get_shared_video(filename: str):
    """Public endpoint for serving generated videos to Instagram Graph API."""
    safe_name = _validate_share_filename(filename)
    candidate = settings.videos_dir / safe_name
    if candidate.is_file():
        return FileResponse(candidate, media_type="video/mp4")
    raise HTTPException(status_code=404, detail="Video not found")


_LOOP_PLAYER_HTML = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="theme-color" content="#000">
<title>art-rium</title>
<style>
  html,body{margin:0;padding:0;height:100%;background:#000;overflow:hidden;-webkit-tap-highlight-color:transparent;font-family:ui-sans-serif,system-ui,-apple-system,sans-serif}
  video{position:fixed;inset:0;width:100%;height:100%;object-fit:contain;background:#000}

  #cd{position:fixed;inset:0;display:none;align-items:center;justify-content:center;
      font-size:32vh;font-weight:700;color:#fff;background:rgba(0,0,0,0.55);
      z-index:5;user-select:none;pointer-events:none;letter-spacing:-2px}
  #cd.show{display:flex}

  #tick{position:fixed;right:max(18px,env(safe-area-inset-right));bottom:max(22px,env(safe-area-inset-bottom));
        width:20px;height:20px;border-radius:50%;background:rgba(255,255,255,0.9);
        opacity:0;transition:opacity 220ms linear,transform 220ms ease-out;z-index:4;pointer-events:none}
  #tick.on{opacity:1;transition:none;transform:scale(1)}
  #tick.accent{width:34px;height:34px;background:#f5a623;box-shadow:0 0 28px rgba(245,166,35,0.55)}

  #bar{position:fixed;left:0;right:0;bottom:0;height:3px;background:rgba(255,255,255,0.06);z-index:3;pointer-events:none}
  #bar>i{display:block;height:100%;width:0%;background:rgba(255,255,255,0.55)}

  #cog{position:fixed;top:max(12px,env(safe-area-inset-top));right:max(12px,env(safe-area-inset-right));
       width:38px;height:38px;border-radius:50%;display:flex;align-items:center;justify-content:center;
       background:rgba(0,0,0,0.45);color:#fff;font-size:18px;cursor:pointer;z-index:6;
       opacity:0;transition:opacity 240ms;border:1px solid rgba(255,255,255,0.15)}
  #cog.show{opacity:0.85}

  #panel{position:fixed;top:max(58px,calc(env(safe-area-inset-top) + 58px));left:50%;transform:translateX(-50%);
         display:none;flex-direction:column;gap:10px;padding:16px 18px;min-width:300px;max-width:90vw;
         background:rgba(20,20,22,0.92);color:#fff;border-radius:14px;border:1px solid rgba(255,255,255,0.12);
         font-size:13px;z-index:7;backdrop-filter:blur(10px)}
  #panel.show{display:flex}
  #panel h3{margin:0 0 4px;font-size:13px;font-weight:700;letter-spacing:0.4px;text-transform:uppercase;color:#bbb}
  #panel label{display:flex;justify-content:space-between;align-items:center;gap:14px}
  #panel input[type=number]{width:78px;padding:7px 9px;border-radius:7px;border:1px solid rgba(255,255,255,0.18);
                            background:rgba(255,255,255,0.06);color:#fff;font:inherit;text-align:right}
  #panel .row{display:flex;gap:8px}
  #panel button{flex:1;padding:9px 12px;border-radius:7px;border:1px solid rgba(255,255,255,0.18);
                background:rgba(255,255,255,0.08);color:#fff;font:inherit;cursor:pointer}
  #panel button.primary{background:#f5a623;border-color:#f5a623;color:#000;font-weight:700}
  #panel .hint{font-size:11px;opacity:0.7;line-height:1.4}

  #starttap{position:fixed;inset:0;display:none;align-items:center;justify-content:center;
            background:rgba(0,0,0,0.7);color:#fff;font-size:18px;font-weight:600;z-index:8;cursor:pointer}
  #starttap.show{display:flex}

  /* One-shot hint that AudioContext needs a user gesture on iOS. Auto-fades. */
  #sndhint{position:fixed;left:50%;bottom:max(28px,calc(env(safe-area-inset-bottom) + 22px));
           transform:translateX(-50%);padding:9px 16px;border-radius:999px;
           background:rgba(20,20,22,0.78);color:#fff;font-size:12px;font-weight:600;
           border:1px solid rgba(255,255,255,0.14);backdrop-filter:blur(8px);
           opacity:0;transition:opacity 320ms;z-index:7;pointer-events:none}
  #sndhint.show{opacity:0.95}
</style>
</head><body>
<video id="v" src="__SRC__" muted playsinline preload="auto" disablepictureinpicture></video>
<div id="cd"></div>
<div id="bar"><i id="bar-fill"></i></div>
<div id="tick"></div>
<div id="cog">⚙</div>
<div id="panel">
  <h3>Loop Settings</h3>
  <label>Countdown (s) <input id="cfg-countdown" type="number" min="0" max="10" step="1"></label>
  <label>Tick every (frames) <input id="cfg-tick" type="number" min="1" step="1"></label>
  <label>Accent every (ticks, 0=off) <input id="cfg-accent" type="number" min="0" step="1"></label>
  <label>Loop bars (0=auto) <input id="cfg-bars" type="number" min="0" step="1"></label>
  <label>Click sound <input id="cfg-sound" type="checkbox"></label>
  <div class="hint" id="hint">—</div>
  <div class="row">
    <button id="btn-restart" class="primary">Restart</button>
    <button id="btn-close">Close</button>
  </div>
</div>
<div id="starttap">Tap to start</div>
<div id="sndhint">🔊 Tap anywhere to enable sound</div>
<script>
  const qs = new URLSearchParams(location.search);
  const STORE_KEY = 'improv_loop_settings_v1';
  function readStored(){ try { return JSON.parse(localStorage.getItem(STORE_KEY)||'{}'); } catch { return {}; } }
  function writeStored(o){ try { localStorage.setItem(STORE_KEY, JSON.stringify(o)); } catch {} }

  // URL params are server-side defaults; localStorage overrides them per device.
  const stored = readStored();
  const pick = (key, fallback) => {
    if (stored[key] !== undefined) return Number(stored[key]);
    const q = qs.get(key);
    return q !== null ? Number(q) : fallback;
  };
  const cfg = {
    fps:          Math.max(1, Number(qs.get('fps')) || 24),
    countdown:    Math.max(0, Math.min(10, pick('countdown', 4))),
    tick_every:   Math.max(1, pick('tick_every', 24)),
    accent_every: Math.max(0, pick('accent_every', 4)),
    loop_bars:    Math.max(0, pick('loop_bars', 0)),
    sound:        pick('sound', 1) ? 1 : 0,
  };

  const v = document.getElementById('v');
  const cd = document.getElementById('cd');
  const tickEl = document.getElementById('tick');
  const barFill = document.getElementById('bar-fill');
  const cog = document.getElementById('cog');
  const panel = document.getElementById('panel');
  const hint = document.getElementById('hint');
  const startTap = document.getElementById('starttap');
  const sndHint = document.getElementById('sndhint');

  // ── Audio click (Web Audio API, no asset) ────────────────────────────────
  // iOS Safari keeps the AudioContext suspended until a user gesture, even
  // though the muted video can autoplay. We create+resume on any first tap
  // and show a one-shot hint pill in the meantime so the user knows why
  // the metronome is silent on iPad.
  let audioCtx = null;
  let audioUnlocked = false;
  let sndHintTimer = null;

  function ensureAudio() {
    if (!audioCtx) {
      try {
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      } catch { return; }
    }
    if (audioCtx.state === 'suspended') {
      audioCtx.resume().catch(() => {});
    }
    if (audioCtx.state === 'running' && !audioUnlocked) {
      audioUnlocked = true;
      hideSoundHint();
    }
  }

  function showSoundHint() {
    if (!cfg.sound || audioUnlocked) return;
    sndHint.classList.add('show');
    clearTimeout(sndHintTimer);
    sndHintTimer = setTimeout(hideSoundHint, 6000);
  }
  function hideSoundHint() {
    sndHint.classList.remove('show');
    clearTimeout(sndHintTimer);
  }

  function playClick(isAccent) {
    if (!cfg.sound || !audioCtx || audioCtx.state !== 'running') return;
    try {
      const t0 = audioCtx.currentTime;
      const dur = isAccent ? 0.075 : 0.045;
      const peak = isAccent ? 0.32 : 0.20;
      const freq = isAccent ? 1600 : 1000;
      const osc = audioCtx.createOscillator();
      const gain = audioCtx.createGain();
      osc.type = 'sine';
      osc.frequency.setValueAtTime(freq, t0);
      gain.gain.setValueAtTime(0, t0);
      gain.gain.linearRampToValueAtTime(peak, t0 + 0.003);
      gain.gain.exponentialRampToValueAtTime(0.0001, t0 + dur);
      osc.connect(gain).connect(audioCtx.destination);
      osc.start(t0);
      osc.stop(t0 + dur + 0.02);
    } catch {}
  }

  // Unlock on any user gesture (capture so it fires before other handlers).
  ['pointerdown', 'touchstart', 'keydown'].forEach(evt => {
    document.addEventListener(evt, ensureAudio, { capture: true, passive: true });
  });

  let totalFrames = 0;
  let loopEndFrame = 0;
  let running = false;
  let lastTick = -1;
  let starting = false;

  function recomputeLoop() {
    const barFrames = cfg.tick_every * Math.max(1, cfg.accent_every || 1);
    let bars = cfg.loop_bars > 0 ? cfg.loop_bars : Math.floor(totalFrames / barFrames);
    if (bars < 1) bars = 1;
    loopEndFrame = Math.min(totalFrames || barFrames, bars * barFrames);
    const secs = (loopEndFrame / cfg.fps).toFixed(2);
    const dur = v.duration ? v.duration.toFixed(2) : '?';
    hint.textContent = `${bars} bar${bars===1?'':'s'} · loop @ ${secs}s · video ${dur}s @ ${cfg.fps}fps`;
  }

  v.addEventListener('loadedmetadata', () => {
    totalFrames = Math.floor((v.duration || 0) * cfg.fps);
    recomputeLoop();
    beginSession();
  });

  function runCountdown(n) {
    return new Promise(resolve => {
      cd.classList.add('show');
      let i = n;
      cd.textContent = String(i);
      const tickDown = () => {
        i -= 1;
        if (i <= 0) { cd.classList.remove('show'); resolve(); return; }
        cd.textContent = String(i);
        setTimeout(tickDown, 1000);
      };
      setTimeout(tickDown, 1000);
    });
  }

  async function beginSession() {
    if (starting) return;
    starting = true;
    running = false;
    lastTick = -1;
    barFill.style.width = '0%';
    v.pause();
    try { v.currentTime = 0; } catch {}
    if (cfg.countdown > 0) await runCountdown(cfg.countdown);
    try {
      await v.play();
      startTap.classList.remove('show');
      running = true;
      if (cfg.sound) showSoundHint();
      requestAnimationFrame(loop);
    } catch {
      // Autoplay blocked — show tap-to-start overlay, retry on tap.
      startTap.classList.add('show');
    } finally {
      starting = false;
    }
  }

  function loop() {
    if (!running) return;
    const frame = Math.floor(v.currentTime * cfg.fps);
    barFill.style.width = (Math.min(1, frame / loopEndFrame) * 100).toFixed(2) + '%';
    if (frame >= loopEndFrame) {
      try { v.currentTime = 0; } catch {}
      lastTick = -1;
      barFill.style.width = '0%';
    }
    const tickIndex = Math.floor(frame / cfg.tick_every);
    if (tickIndex !== lastTick && frame >= 0) {
      lastTick = tickIndex;
      const isAccent = cfg.accent_every > 0 && (tickIndex % cfg.accent_every === 0);
      tickEl.classList.remove('on', 'accent');
      void tickEl.offsetWidth;
      tickEl.classList.add('on');
      if (isAccent) tickEl.classList.add('accent');
      setTimeout(() => tickEl.classList.remove('on'), 110);
      playClick(isAccent);
    }
    requestAnimationFrame(loop);
  }

  // Tap-to-start overlay (autoplay fallback)
  startTap.addEventListener('click', () => beginSession());

  // Reveal cog briefly on any tap that isn't on the panel or cog itself.
  let cogTimer = null;
  document.addEventListener('click', e => {
    if (panel.contains(e.target) || cog.contains(e.target) || startTap.contains(e.target)) return;
    cog.classList.add('show');
    clearTimeout(cogTimer);
    cogTimer = setTimeout(() => cog.classList.remove('show'), 2500);
  }, true);

  function syncInputs() {
    document.getElementById('cfg-countdown').value  = cfg.countdown;
    document.getElementById('cfg-tick').value       = cfg.tick_every;
    document.getElementById('cfg-accent').value     = cfg.accent_every;
    document.getElementById('cfg-bars').value       = cfg.loop_bars;
    document.getElementById('cfg-sound').checked    = !!cfg.sound;
  }
  cog.addEventListener('click', () => { syncInputs(); recomputeLoop(); panel.classList.add('show'); });

  function applyPanel() {
    cfg.countdown    = Math.max(0, Math.min(10, parseInt(document.getElementById('cfg-countdown').value || '0', 10)));
    cfg.tick_every   = Math.max(1, parseInt(document.getElementById('cfg-tick').value || '24', 10));
    cfg.accent_every = Math.max(0, parseInt(document.getElementById('cfg-accent').value || '0', 10));
    cfg.loop_bars    = Math.max(0, parseInt(document.getElementById('cfg-bars').value || '0', 10));
    cfg.sound        = document.getElementById('cfg-sound').checked ? 1 : 0;
    writeStored({
      countdown: cfg.countdown, tick_every: cfg.tick_every,
      accent_every: cfg.accent_every, loop_bars: cfg.loop_bars,
      sound: cfg.sound,
    });
    recomputeLoop();
    if (cfg.sound) { ensureAudio(); if (!audioUnlocked) showSoundHint(); }
    else { hideSoundHint(); }
  }
  ['cfg-countdown','cfg-tick','cfg-accent','cfg-bars'].forEach(id => {
    document.getElementById(id).addEventListener('input', applyPanel);
  });
  document.getElementById('cfg-sound').addEventListener('change', applyPanel);
  document.getElementById('btn-restart').addEventListener('click', () => {
    panel.classList.remove('show');
    beginSession();
  });
  document.getElementById('btn-close').addEventListener('click', () => panel.classList.remove('show'));
</script>
</body></html>"""


@router.get("/share/video-loop/{filename}", dependencies=[Depends(_verify_share_token)])
async def get_shared_video_loop(filename: str):
    """Public HTML player that loops a generated video — used by the Improv
    tool's share-URL/QR-code so a second device (iPad next to the piano) can
    play the source over and over without manual restarts.

    Query params (all optional, with safe defaults baked into the JS):
      fps           — source frame rate, drives the metronome timebase
      countdown     — pre-roll seconds (0 = off)
      tick_every    — tick once every N source frames
      accent_every  — every N-th tick is a downbeat accent (0 = off)
      loop_bars     — number of bars per loop iteration (0 = auto-fit)

    The loop player reads these from `location.search` directly, so this
    endpoint doesn't need to parse them — it just serves the static HTML.
    """
    safe_name = _validate_share_filename(filename)
    if not (settings.videos_dir / safe_name).is_file():
        raise HTTPException(status_code=404, detail="Video not found")
    src = f"/share/video/{safe_name}"
    if settings.image_share_token:
        src += f"?token={settings.image_share_token}"
    return HTMLResponse(_LOOP_PLAYER_HTML.replace("__SRC__", src))


@router.get("/api/image/{filename}", dependencies=[Depends(require_auth)])
async def get_image(filename: str):
    candidate = _find_image_on_disk(Path(filename).name)
    if candidate is None:
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(candidate, media_type="image/png")


@router.get("/api/image/{filename}/thumb", dependencies=[Depends(require_auth)])
async def get_image_thumb(filename: str, db: AsyncSession = Depends(get_db)):
    """Serve the JPEG thumbnail; fall back to the full image if no thumbnail exists."""
    safe_name = Path(filename).name

    # Look up DB record to find stored thumbnail_path
    result = await db.execute(select(Image).where(Image.filename == safe_name))
    img = result.scalar_one_or_none()

    if img and img.thumbnail_path:
        thumb = settings.storage_dir / img.thumbnail_path
        if thumb.exists():
            return FileResponse(thumb, media_type="image/jpeg")

    # Fall back: serve the full image.
    candidate = _find_image_on_disk(safe_name)
    if candidate is None:
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(candidate, media_type="image/png")


@router.post("/api/images/backfill-thumbnails", dependencies=[Depends(require_auth)])
async def backfill_thumbnails(db: AsyncSession = Depends(get_db)):
    """
    Generate missing thumbnails for all images that don't have one yet.
    Safe to call multiple times — skips images that already have a thumbnail.
    """
    result = await db.execute(select(Image).where(Image.thumbnail_path.is_(None)))
    images = result.scalars().all()

    done, failed = 0, 0
    for img in images:
        src = settings.storage_dir / img.filepath
        if not src.exists():
            failed += 1
            continue
        rel = thumb_rel_path(img.filename)
        dest = settings.storage_dir / rel
        ok = await make_thumbnail(src, dest)
        if ok:
            img.thumbnail_path = rel
            done += 1
        else:
            failed += 1

    await db.commit()
    return {"backfilled": done, "failed": failed, "total": len(images)}


@router.post("/api/clear_pending_images/{client_id}", dependencies=[Depends(require_auth)])
async def clear_pending_images(client_id: str, request: Request):
    listener = request.app.state.comfy_listener
    count = len(listener._pending.get(client_id, []))
    listener._pending.pop(client_id, None)
    return {"cleared": count}


@router.get("/api/health")
async def health():
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(f"http://{settings.comfyui_host}/system_stats")
            comfy_ok = r.status_code == 200
    except Exception:
        comfy_ok = False
    return {
        "status": "ok",
        "comfyui": "connected" if comfy_ok else "unreachable",
        "auth_required": bool(settings.api_key),
    }


@router.websocket("/ws/{client_id}")
async def ws_endpoint(websocket: WebSocket, client_id: str):
    if not ws_auth_ok(websocket):
        logger.warning(f"Rejected WS connection from {websocket.client.host}")
        await websocket.close(code=4001)
        return

    await websocket.accept()
    listener = websocket.app.state.comfy_listener
    listener.add_ws(client_id, websocket)
    logger.info(f"Frontend connected: {client_id}")

    # Replay any images that arrived while client was disconnected
    await listener.replay_pending(client_id, websocket)

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        listener.remove_ws(client_id)
        logger.info(f"Frontend disconnected: {client_id}")
