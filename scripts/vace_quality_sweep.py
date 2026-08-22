"""
Render one VACE scene under several sampler recipes and put the results side by
side.

The question this exists to answer: the old AnimateDiff/LCM/IP-Adapter graph
produced a hand-made, film-like picture, and VACE so far has not. Every VACE
render to date used the `fast` recipe — a cfg-step-distill LoRA at 6 steps and
cfg 1.0 — which is exactly the configuration that trades micro-texture and
prompt nuance for speed. So before rebuilding the old stack, hold everything
still and change only the sampler.

Everything except the sampler *is* held still: `quality_sweep_workflows` copies
the request, so the control track, the reference, the prompt, the strength, the
canvas and the seed are shared. That matters more than it sounds — a seed means
something different at every latent shape, so a comparison across canvases would
be two different scenes, not two samplers.

    python scripts/vace_quality_sweep.py ^
        --control E:/00_comfy/input/01_mask_depth_96f0001-0094.mp4 ^
        --reference my_picture.png ^
        --prompt "..." ^
        --recipes fast rich full

`--reference` is a filename inside ComfyUI's own input folder (LoadImage
resolves by bare name), not a path into art-rium storage.

Costs, on the 4060 Ti at 832x480 and 93 frames: fast ~9 min, rich ~17 min,
full ~57 min. The whole sheet is a bit over an hour, and it is meant to be
started and left alone.
"""
from __future__ import annotations

import argparse
import asyncio
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402

from core.config import settings  # noqa: E402
from services.comfy.client import free_memory, poll_history, post_workflow  # noqa: E402
from services.comfy.vace import (  # noqa: E402
    DEFAULT_ASPECT,
    DEFAULT_CANVAS,
    FPS,
    STRENGTH_DEFAULT,
    VaceRequest,
    estimate_seconds,
    loop_length,
    quality_sweep_workflows,
)

TIMEOUT = 4 * 60 * 60      # a `full` pass at 720p runs past two hours


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--control", required=True, help="absolute path to the control video")
    ap.add_argument("--reference", required=True,
                    help="a picture: any path, or a bare filename already in ComfyUI's input")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--recipes", nargs="+", default=["fast", "rich", "full"])
    ap.add_argument("--canvas", default=DEFAULT_CANVAS, choices=["sketch", "final"])
    ap.add_argument("--aspect", default=DEFAULT_ASPECT, choices=["wide", "tall", "square"])
    ap.add_argument("--strength", type=float, default=STRENGTH_DEFAULT)
    ap.add_argument("--frames", type=int, default=81)
    ap.add_argument("--seed", type=int, default=1234, help="fixed on purpose")
    ap.add_argument("--no-invert-depth", action="store_true",
                    help="the control track is already near-bright / far-dark")
    return ap.parse_args()


def stage_reference(name_or_path: str) -> str:
    """LoadImage resolves by bare filename inside ComfyUI's input folder, so a
    picture living anywhere else has to be put where the node can see it."""
    src = Path(name_or_path)
    if not src.is_absolute() and not src.exists():
        return name_or_path          # already an input-folder filename
    if not src.is_file():
        sys.exit(f"Reference not found: {src}")
    inp = settings.comfyui_output_dir.parent / "input"
    inp.mkdir(parents=True, exist_ok=True)
    dest = inp / f"artrium_sweep_ref{src.suffix.lower()}"
    shutil.copy2(src, dest)
    return dest.name


async def render(recipe: str, wf: dict, save_node: str) -> Path:
    async with httpx.AsyncClient(timeout=240) as client:
        prompt_id = await post_workflow(client, wf)
        outputs = await poll_history(client, prompt_id, timeout=TIMEOUT, interval=5)
    entry = (outputs.get(save_node) or {}).get("gifs") or []
    if not entry:
        raise RuntimeError(f"{recipe}: ComfyUI returned no video ({list(outputs)})")
    return settings.comfyui_output_dir / entry[0].get("subfolder", "") / entry[0]["filename"]


def contact_sheet(renders: list[tuple[str, Path]], dest: Path, at: float) -> Path | None:
    """One frame from the middle of each clip, stacked left to right.

    The middle rather than the first: frame 0 of a VACE render is the most
    control-faithful frame in the clip and so the least informative about what
    the sampler invented on top of it.
    """
    stills = []
    for recipe, path in renders:
        still = dest.parent / f"{dest.stem}_{recipe}.png"
        cmd = [settings.ffmpeg_path, "-y", "-v", "error",
               "-i", str(path), "-ss", f"{at:.2f}", "-frames:v", "1", str(still)]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            stills.append(still)
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            print(f"  ! no still for {recipe}: {exc}")
    if len(stills) < 2:
        return None
    cmd = [settings.ffmpeg_path, "-y", "-v", "error"]
    for still in stills:
        cmd += ["-i", str(still)]
    cmd += ["-filter_complex", f"hstack=inputs={len(stills)}", str(dest)]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"  ! no contact sheet: {exc}")
        return None
    return dest


async def main() -> None:
    args = parse_args()
    control = Path(args.control)
    if not control.is_file():
        sys.exit(f"Control track not found: {control}")

    length = loop_length(args.frames)
    base = VaceRequest(
        control_video=str(control),
        prompt=args.prompt,
        reference_image=stage_reference(args.reference),
        canvas=args.canvas,
        aspect=args.aspect,
        length=length,
        source_frames=length,
        strength=args.strength,
        seed=args.seed,
        invert_depth=not args.no_invert_depth,
        filename_prefix=f"artrium_vace_sweep_{int(time.time())}",
    )

    sheet = quality_sweep_workflows(base, recipes=tuple(args.recipes))
    total = sum(estimate_seconds(args.canvas, args.aspect, r, length) for r in args.recipes)
    print(f"{len(sheet)} renders, {length} frames, seed {args.seed}, "
          f"strength {args.strength} — estimated {total // 60} min total\n")

    renders: list[tuple[str, Path]] = []
    for recipe, wf, save_node in sheet:
        started = time.monotonic()
        print(f"-> {recipe}: {wf['vc_ks']['inputs']['steps']} steps, "
              f"cfg {wf['vc_ks']['inputs']['cfg']}", flush=True)
        produced = await render(recipe, wf, save_node)
        renders.append((recipe, produced))
        print(f"  {produced}  ({time.monotonic() - started:.0f} s)\n", flush=True)
        async with httpx.AsyncClient(timeout=60) as client:
            await free_memory(client)

    dest = settings.comfyui_output_dir / f"{base.filename_prefix}_sheet.png"
    if contact_sheet(renders, dest, at=length / 2 / FPS):
        print(f"Contact sheet: {dest}")


if __name__ == "__main__":
    asyncio.run(main())
