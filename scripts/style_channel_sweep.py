"""
Which channel carries a trained style best through the AnimateLCM graph?

One seed, one control track, one prompt per scene; only the style channel
moves. The four candidates are the ones the Struktur tool offers:

    lora       a picture-style LoRA on the checkpoint (+ its trigger word)
    emb        a textual-inversion embedding, inline in the prompt
    ip         an IP-Adapter picture from the same dataset
    inject     that picture laid into the base pass (ADE image injection)

plus a prompt-only baseline and the pairings worth having. Base pass only by
default — the look is decided there; `--hires` re-runs chosen variants through
the whole graph. Resumable: a variant whose video exists is not rendered again.

    python scripts/style_channel_sweep.py --out E:/00_comfy/output/style_sweep ^
        --lora artvis3_sd15.safetensors --trigger artvis3 ^
        --embedding artrium/artvis3emb ^
        --ref-a storage/lora_datasets/artvision_v3/24_xxx.jpg ...

Metrics are a separate step (scripts/style_channel_metrics.py) because they
need CLIP, which lives in ComfyUI's venv rather than art-rium's.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402
from PIL import Image  # noqa: E402

from core.config import settings  # noqa: E402
from services.comfy.animatelcm import (  # noqa: E402
    AnimateLcmRequest,
    Embedding,
    Injection,
    build_animatelcm_base_workflow,
    build_animatelcm_workflow,
)
from services.comfy.client import free_memory, poll_history, post_workflow  # noqa: E402

TIMEOUT = 60 * 60
SEED = 1234


def variants(a: argparse.Namespace) -> dict[str, dict]:
    """name -> request overrides. `_trigger` prefixes the LoRA's trigger word."""
    lora = {"style_lora": a.lora, "_trigger": True}
    emb = lambda w: {"embeddings": [Embedding(a.embedding, w)]}  # noqa: E731
    ip = lambda ref, w: {"reference_image": ref, "ip_weight": w}  # noqa: E731
    inj = lambda vid, s: {"_inject": (vid, s)}  # noqa: E731
    return {
        "base":            {},
        "lora05":          {**lora, "style_lora_strength": 0.5},
        "lora08":          {**lora, "style_lora_strength": 0.8},
        "lora10":          {**lora, "style_lora_strength": 1.0},
        "lora08_notrig":   {"style_lora": a.lora, "style_lora_strength": 0.8},
        "ipA06":           ip("A", 0.6),
        "ipA10":           ip("A", 1.0),
        "ipB10":           ip("B", 1.0),
        "injA03":          inj("A", 0.3),
        "injA05":          inj("A", 0.5),
        "injB03":          inj("B", 0.3),
        "lora08+ipA06":    {**lora, "style_lora_strength": 0.8, **ip("A", 0.6)},
        "lora08+injA03":   {**lora, "style_lora_strength": 0.8, **inj("A", 0.3)},
        "emb10":           emb(1.0),
        "emb12":           emb(1.2),
        "emb10+ipA06":     {**emb(1.0), **ip("A", 0.6)},
        "lora08+emb10":    {**lora, "style_lora_strength": 0.8, **emb(1.0)},
    }


SCENES = {
    "cube": {
        "control": "E:/00_comfy/input/01_mask_depth_96f0001-0094.mp4",
        "derive": False,
        "prompt": "a monolithic cube standing on a vast empty plain, a lone small "
                  "figure beside it, wide shot, cinematic",
    },
    "hand": {
        "control": "E:/00_comfy/input/0001-0151.mp4",
        "derive": True,
        "every": 3,          # 30 fps footage thinned the way the tool does
        "prompt": "a human hand slowly opening, fingers spread, in a quiet empty "
                  "room, cinematic",
    },
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--lora", required=True)
    ap.add_argument("--trigger", default="")
    ap.add_argument("--embedding", required=True)
    ap.add_argument("--ref-a", required=True, type=Path)
    ap.add_argument("--ref-b", required=True, type=Path)
    ap.add_argument("--scenes", nargs="*", default=list(SCENES))
    ap.add_argument("--only", nargs="*", help="variant names (default: all)")
    ap.add_argument("--skip", nargs="*", default=[])
    ap.add_argument("--frames", type=int, default=48)
    ap.add_argument("--hires", action="store_true", help="whole graph, not base only")
    ap.add_argument("--checkpoint", default=None,
                    help="SD 1.5 checkpoint file name, e.g. photon_v1.safetensors; "
                         "omit for the Juggernaut default")
    ap.add_argument("--tag", default="", help="suffix for this run's manifest and file names")
    return ap.parse_args()


def ffmpeg(*args: str) -> None:
    subprocess.run([settings.ffmpeg_path, "-y", "-v", "error", *args],
                   check=True, capture_output=True)


def square_crop(src: Path, dest: Path) -> Path:
    """Both scenes are square. A pad-fitted portrait would inject black bars."""
    with Image.open(src) as im:
        im = im.convert("RGB")
        side = min(im.size)
        left, top = (im.width - side) // 2, (im.height - side) // 2
        im.crop((left, top, left + side, top + side)).resize((768, 768), Image.LANCZOS).save(dest)
    return dest


def prepare(a: argparse.Namespace) -> dict:
    """Stage references into ComfyUI's input folder, make the injection stills
    and the thinned footage. Idempotent."""
    work = a.out / "_inputs"
    work.mkdir(parents=True, exist_ok=True)
    comfy_in = settings.comfyui_output_dir.parent / "input"
    staged = {}
    for key, src in (("A", a.ref_a), ("B", a.ref_b)):
        sq = square_crop(src, work / f"ref{key}.png")
        name = f"artrium_style_ref{key}.png"
        (comfy_in / name).write_bytes(sq.read_bytes())
        vid = work / f"inject{key}.mp4"
        if not vid.exists():
            ffmpeg("-loop", "1", "-i", str(sq), "-frames:v", str(a.frames), "-r", "8",
                   "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", str(vid))
        staged[key] = {"image": name, "video": str(vid)}
    controls = {}
    for scene in a.scenes:
        spec = SCENES[scene]
        src = Path(spec["control"])
        if spec.get("every"):
            dest = work / f"{scene}_control.mp4"
            if not dest.exists():
                ffmpeg("-i", str(src), "-vf", f"select='not(mod(n\\,{spec['every']}))'",
                       "-fps_mode", "passthrough", "-frames:v", str(a.frames), "-an",
                       "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p", str(dest))
            src = dest
        controls[scene] = str(src)
    return {"refs": staged, "controls": controls}


def request_for(a, scene: str, name: str, over: dict, inputs: dict) -> AnimateLcmRequest:
    spec = SCENES[scene]
    over = dict(over)
    prompt = spec["prompt"]
    if over.pop("_trigger", False) and a.trigger:
        prompt = f"{a.trigger}, {prompt}"
    if "reference_image" in over:
        over["reference_image"] = inputs["refs"][over["reference_image"]]["image"]
    inject = over.pop("_inject", None)
    if inject:
        key, strength = inject
        over["injection"] = Injection(video=inputs["refs"][key]["video"], strength=strength,
                                      source_frames=a.frames)
    extra = {"checkpoint": a.checkpoint} if a.checkpoint else {}
    return AnimateLcmRequest(
        control_video=inputs["controls"][scene], prompt=prompt,
        derive_depth=spec["derive"], invert_depth=not spec["derive"],
        source_width=1080, source_height=1080,
        length=a.frames, source_frames=a.frames, seed=SEED, hires=a.hires,
        filename_prefix=f"style_sweep/{scene}_{name.replace('+', '_')}"
                        + (f"_{a.tag}" if a.tag else "")
                        + ("_hires" if a.hires else ""),
        **extra, **over,
    )


async def render(wf: dict, node: str) -> Path:
    async with httpx.AsyncClient(timeout=240) as client:
        prompt_id = await post_workflow(client, wf)
        outputs = await poll_history(client, prompt_id, timeout=TIMEOUT, interval=4)
    entry = (outputs.get(node) or {}).get("gifs") or []
    if not entry:
        raise RuntimeError(f"ComfyUI returned no video ({list(outputs)})")
    return settings.comfyui_output_dir / entry[0].get("subfolder", "") / entry[0]["filename"]


async def main() -> None:
    a = parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    inputs = prepare(a)
    stem = "manifest_hires" if a.hires else "manifest"
    manifest_path = a.out / (f"{stem}_{a.tag}.json" if a.tag else f"{stem}.json")
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    table = variants(a)
    names = [n for n in (a.only or table) if n not in a.skip]
    for scene in a.scenes:
        for name in names:
            key = f"{scene}/{name}"
            if key in manifest and Path(manifest[key]["video"]).exists():
                continue
            req = request_for(a, scene, name, table[name], inputs)
            if a.hires:
                wf, node = build_animatelcm_workflow(req)
            else:
                wf, _preview, node = build_animatelcm_base_workflow(req)
            print(f"-> {key}", flush=True)
            started = time.monotonic()
            try:
                video = await render(wf, node)
            except Exception as exc:                    # noqa: BLE001
                print(f"   FAILED: {exc}", flush=True)
                continue
            seconds = round(time.monotonic() - started)
            manifest[key] = {"video": str(video), "seconds": seconds,
                             "prompt": req.prompt, "scene": scene, "variant": name}
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            print(f"   {video.name} ({seconds} s)", flush=True)
        async with httpx.AsyncClient(timeout=60) as client:
            await free_memory(client)


if __name__ == "__main__":
    asyncio.run(main())
