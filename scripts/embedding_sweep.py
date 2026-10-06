"""
Phase 0 of the embedding work: does kentskooking's way of joining textual-
inversion embeddings do anything on *our* AnimateLCM graph?

One seed, one control track, one canvas, and only the join moves:

    prompt     the prompt alone — the baseline every other row is read against
    inline     prompt and embeddings in one CLIPTextEncode, the A1111 habit
    concat     each encoded on its own, concatenated along the token axis
    sandwich   concat with an empty encode between every pair (Kent's trick)
    bare       sandwich without the prompt — Kent's own renders have no text

    python scripts/embedding_sweep.py ^
        --control E:/00_comfy/input/01_mask_depth_96f0001-0094.mp4 ^
        --prompt "a monolith in a flooded hall" ^
        --embeddings style-rustmagic style-swirlmagic:0.8

`--recipe kent` repeats the sheet the way Kent samples: LCM LoRA off, euler /
normal, 20 steps, cfg 8 (the ComfyUI default graph he starts from). That is the
second question — whether embeddings survive our LCM-at-cfg-1 recipe at all.

The base pass only by default: the look is decided there, and the hires pass
would triple the cost of a sheet that exists to compare looks.
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
from PIL import Image, ImageChops, ImageDraw, ImageFont, ImageStat  # noqa: E402

from core.config import settings  # noqa: E402
from core.video_thumb import probe_video_dimensions  # noqa: E402
from services.comfy.animatelcm import (  # noqa: E402
    FPS,
    AnimateLcmRequest,
    Embedding,
    build_animatelcm_base_workflow,
    build_animatelcm_workflow,
)
from services.comfy.client import free_memory, poll_history, post_workflow  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from animatelcm_sweep import stage_reference  # noqa: E402

TIMEOUT = 60 * 60
VARIANTS = ("prompt", "inline", "concat", "sandwich", "bare")
RECIPES = {
    "lcm": {},
    "kent": {"lcm_lora_strength": 0.0, "sampler": "euler", "scheduler": "normal",
             "steps": 20, "cfg": 8.0},
}


def parse_embedding(spec: str) -> Embedding:
    name, _, weight = spec.partition(":")
    return Embedding(name, float(weight) if weight else 1.0)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--control", required=True, help="absolute path to the control video")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--embeddings", nargs="+", required=True, type=parse_embedding,
                    help="stems in ComfyUI's embeddings folder, optionally name:weight")
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS), choices=VARIANTS)
    ap.add_argument("--singles", action="store_true",
                    help="add one row per embedding on its own (sandwich join), so a "
                         "combined row can be read back into who contributed what")
    ap.add_argument("--recipe", default="lcm", choices=list(RECIPES))
    ap.add_argument("--cfg", type=float, help="override the recipe's cfg")
    ap.add_argument("--steps", type=int, help="override the recipe's steps")
    ap.add_argument("--reference", help="optional IP-Adapter picture; off by default "
                                        "so it cannot mask what the embeddings do")
    ap.add_argument("--ip", type=float, default=0.55)
    ap.add_argument("--hires", action="store_true", help="also run the hires pass")
    ap.add_argument("--frames", type=int, default=48)
    ap.add_argument("--seed", type=int, default=1234, help="fixed on purpose")
    ap.add_argument("--no-invert-depth", action="store_true")
    return ap.parse_args()


async def known_embeddings() -> set[str]:
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(f"http://{settings.comfyui_host}/embeddings")
        r.raise_for_status()
        return {name.replace("\\", "/") for name in r.json()}


def variant_request(base: AnimateLcmRequest, variant: str,
                    embeddings: list[Embedding]) -> AnimateLcmRequest:
    fields = dict(base.__dict__)
    fields["filename_prefix"] = f"{base.filename_prefix}_{variant.replace(':', '-')}"
    if variant.startswith("only:"):
        fields["embeddings"] = [e for e in embeddings if e.name == variant[5:]]
        fields["embedding_join"] = "sandwich"
    elif variant == "prompt":
        fields["embeddings"] = []
    else:
        fields["embeddings"] = embeddings
        fields["embedding_join"] = "sandwich" if variant == "bare" else variant
        if variant == "bare":
            fields["prompt"] = ""
    return AnimateLcmRequest(**fields)


async def render(label: str, wf: dict, save_node: str) -> Path:
    async with httpx.AsyncClient(timeout=240) as client:
        prompt_id = await post_workflow(client, wf)
        outputs = await poll_history(client, prompt_id, timeout=TIMEOUT, interval=4)
    entry = (outputs.get(save_node) or {}).get("gifs") or []
    if not entry:
        raise RuntimeError(f"{label}: ComfyUI returned no video ({list(outputs)})")
    return settings.comfyui_output_dir / entry[0].get("subfolder", "") / entry[0]["filename"]


def grab(video: Path, at: float, dest: Path) -> Image.Image:
    subprocess.run([settings.ffmpeg_path, "-y", "-v", "error", "-ss", f"{at:.3f}",
                    "-i", str(video), "-frames:v", "1", str(dest)],
                   check=True, capture_output=True)
    return Image.open(dest).convert("RGB")


def mean_abs_diff(a: Image.Image, b: Image.Image) -> float:
    """0 = pixel-identical. A variant that reads 0 against `prompt` means the
    embedding never loaded — ComfyUI skips a missing one with a log line only."""
    return round(sum(ImageStat.Stat(ImageChops.difference(a, b)).mean) / 3, 2)


def contact_sheet(rows: list[tuple[str, list[Image.Image]]], dest: Path) -> None:
    """One row per variant, three moments per row, the label in the margin."""
    cell_w, cell_h = rows[0][1][0].size
    scale = min(1.0, 360 / cell_w)
    cw, ch = int(cell_w * scale), int(cell_h * scale)
    margin = 260
    sheet = Image.new("RGB", (margin + 3 * cw, len(rows) * ch), (18, 18, 18))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=22)
    for r, (label, frames) in enumerate(rows):
        draw.text((12, r * ch + ch // 2 - 12), label, fill=(235, 235, 235), font=font)
        for c, frame in enumerate(frames):
            sheet.paste(frame.resize((cw, ch)), (margin + c * cw, r * ch))
    sheet.save(dest)


async def main() -> None:
    args = parse_args()
    control = Path(args.control)
    if not control.is_file():
        sys.exit(f"Control track not found: {control}")

    # A missing embedding does not fail a render, it just is not there. Refuse
    # up front rather than produce a sheet that says "no effect" for the wrong
    # reason.
    known = await known_embeddings()
    missing = [e.name for e in args.embeddings if e.name not in known]
    if missing:
        sys.exit(f"ComfyUI does not know {missing}. Known: {sorted(known) or 'none'}")

    recipe = dict(RECIPES[args.recipe])
    if args.cfg is not None:
        recipe["cfg"] = args.cfg
    if args.steps is not None:
        recipe["steps"] = args.steps

    src_w, src_h = await probe_video_dimensions(control)
    stamp = int(time.time())
    out_dir = settings.comfyui_output_dir / f"embedding_sweep_{stamp}_{args.recipe}"
    out_dir.mkdir(parents=True, exist_ok=True)
    base = AnimateLcmRequest(
        control_video=str(control), source_width=src_w, source_height=src_h,
        prompt=args.prompt,
        reference_image=stage_reference(args.reference) if args.reference else None,
        ip_weight=args.ip, length=args.frames, source_frames=args.frames,
        hires=args.hires, seed=args.seed, invert_depth=not args.no_invert_depth,
        filename_prefix=f"artrium_embsweep_{stamp}", **recipe,
    )

    print(f"{len(args.variants)} variants, {args.frames} frames, seed {args.seed}, "
          f"recipe {args.recipe} {recipe or '(defaults)'}, "
          f"embeddings {[(e.name, e.weight) for e in args.embeddings]}\n")

    variants = list(args.variants)
    if args.singles:
        variants += [f"only:{e.name}" for e in args.embeddings]

    results: dict[str, dict] = {}
    rows: list[tuple[str, list[Image.Image]]] = []
    moments = [f / FPS for f in (2, args.frames // 2, args.frames - 3)]
    for variant in variants:
        req = variant_request(base, variant, args.embeddings)
        if args.hires:
            wf, node = build_animatelcm_workflow(req)
        else:
            wf, _preview, node = build_animatelcm_base_workflow(req)
        started = time.monotonic()
        print(f"-> {variant}", flush=True)
        produced = await render(variant, wf, node)
        seconds = round(time.monotonic() - started)
        frames = [grab(produced, t, out_dir / f"{variant.replace(':', '-')}_{i}.png")
                  for i, t in enumerate(moments)]
        rows.append((variant, frames))
        results[variant] = {"video": str(produced), "seconds": seconds}
        print(f"   {produced}  ({seconds} s)\n", flush=True)
        async with httpx.AsyncClient(timeout=60) as client:
            await free_memory(client)

    if "prompt" in results:
        reference_mid = rows[[r[0] for r in rows].index("prompt")][1][1]
        for label, frames in rows:
            results[label]["diff_vs_prompt"] = mean_abs_diff(frames[1], reference_mid)
    # Every row against every other, on the middle frame: "does the blank do
    # anything" is sandwich-vs-concat, not either of them against the prompt.
    labels = [label for label, _ in rows]
    matrix = {a: {b: mean_abs_diff(fa[1], fb[1]) for b, (_, fb) in zip(labels, rows)}
              for a, (_, fa) in zip(labels, rows)}
    sheet = out_dir / "sheet.png"
    contact_sheet(rows, sheet)
    (out_dir / "summary.json").write_text(json.dumps({
        "args": {k: (v if not isinstance(v, list) else [str(x) for x in v])
                 for k, v in vars(args).items()},
        "recipe": recipe, "results": results, "pairwise_mid_frame_diff": matrix,
    }, indent=2, default=str), encoding="utf-8")
    for label, info in results.items():
        print(f"{label:22s} {info['seconds']:5d} s   diff vs prompt {info.get('diff_vs_prompt')}")
    width = max(len(label) for label in labels)
    print("\n" + " " * (width + 1) + " ".join(f"{i:>6d}" for i in range(len(labels))))
    for i, a in enumerate(labels):
        print(f"{a:>{width}s} " + " ".join(f"{matrix[a][b]:6.1f}" for b in labels) + f"  [{i}]")
    print(f"\nContact sheet: {sheet}")


if __name__ == "__main__":
    asyncio.run(main())
