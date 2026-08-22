"""
Stage B: repaint a finished render with VACE, and find the strength band for it.

Stage A (AnimateLCM + IP-Adapter) decides composition, palette and material but
renders flat, low and flickering. This pass hands that render to VACE as its
`control_video` — not as a depth pass — so the big model repaints it with
temporal coherence and invented detail while the composition survives.

The path is not new: it is what `plan_region_passes` already does from pass 1
onward, where each region pass takes the previous pass's *render* as its
control. What is new is the strength band. The measured 0.50-0.70 window
describes a **depth** control track, where "preserve the control video" means
preserving flat grey. Here the control video is a picture, so preserving it
means something entirely different and the old numbers do not transfer.

Deliberately **no reference image**. The material is already in the control
video, and adding the reference back would make the result unreadable: a
surviving swirl could then have come either from the control track or from a
fresh injection, and telling those apart is the entire question.

    python scripts/vace_repaint.py ^
        --control E:/00_comfy/output/artrium_alcm_..._ip095_00001.mp4 ^
        --prompt "..." --strengths 0.45 0.65 0.85 --frames 45

Cost note: probe short and small. At 832x480 with the `fast` recipe a 45-frame
pass is about 4 minutes, so a three-point band costs ~13 min. Re-run the winner
long and at 720p once the band is known.
"""
from __future__ import annotations

import argparse
import asyncio
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402

from core.config import settings  # noqa: E402
from services.comfy.client import free_memory, poll_history, post_workflow  # noqa: E402
from services.comfy.vace import (  # noqa: E402
    FPS,
    VaceRequest,
    build_vace_workflow,
    clamp_strength,
    estimate_seconds,
    snap_length,
)

TIMEOUT = 4 * 60 * 60


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--control", required=True,
                    help="a finished render to repaint, not a depth pass")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--strengths", nargs="+", type=float, default=[0.45, 0.65, 0.85])
    ap.add_argument("--canvas", default="sketch", choices=["sketch", "final"])
    ap.add_argument("--aspect", default="wide", choices=["wide", "tall", "square"])
    ap.add_argument("--recipe", default="fast", choices=["fast", "rich", "full", "small"])
    ap.add_argument("--frames", type=int, default=45)
    ap.add_argument("--seed", type=int, default=1234)
    # A render is neither a depth pass to invert nor footage to derive depth
    # from, so both switches stay off and are not exposed.
    ap.add_argument("--fit", default="crop", choices=["crop", "pad", "stretch"],
                    help="crop by default: black pad bars would become content")
    ap.add_argument("--color-match", type=float, default=0.0,
                    help="0..1, grade the repaint back onto the control track's palette")
    return ap.parse_args()


async def render(label: str, wf: dict, save_node: str) -> Path:
    async with httpx.AsyncClient(timeout=240) as client:
        prompt_id = await post_workflow(client, wf)
        outputs = await poll_history(client, prompt_id, timeout=TIMEOUT, interval=4)
    entry = (outputs.get(save_node) or {}).get("gifs") or []
    if not entry:
        raise RuntimeError(f"{label}: ComfyUI returned no video ({list(outputs)})")
    return settings.comfyui_output_dir / entry[0].get("subfolder", "") / entry[0]["filename"]


def still(path: Path, dest: Path, at: float) -> Path | None:
    cmd = [settings.ffmpeg_path, "-y", "-v", "error",
           "-i", str(path), "-ss", f"{at:.2f}", "-frames:v", "1", str(dest)]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return dest
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"  ! no still: {exc}")
        return None


def contact_sheet(stills: list[Path], dest: Path) -> Path | None:
    """Source frame first, then the repaints — the comparison only means
    anything next to what went in."""
    if len(stills) < 2:
        return None
    cmd = [settings.ffmpeg_path, "-y", "-v", "error"]
    for s in stills:
        cmd += ["-i", str(s)]
    # Scale everything to the first still's height before stacking: the source
    # render and the repaints are different sizes.
    filters = "".join(f"[{i}:v]scale=-2:480[s{i}];" for i in range(len(stills)))
    filters += "".join(f"[s{i}]" for i in range(len(stills)))
    filters += f"hstack=inputs={len(stills)}"
    cmd += ["-filter_complex", filters, str(dest)]
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
        sys.exit(f"Control render not found: {control}")

    length = snap_length(args.frames)
    prefix = f"artrium_repaint_{int(time.time())}"
    at = length / 2 / FPS

    per_pass = estimate_seconds(args.canvas, args.aspect, args.recipe, length)
    print(f"{len(args.strengths)} repaints of {control.name}, {length} frames, "
          f"seed {args.seed}, no reference image — "
          f"~{per_pass * len(args.strengths) // 60} min total\n", flush=True)

    scratch = control.parent
    stills = []
    source_still = still(control, scratch / f"{prefix}_source.png", at)
    if source_still:
        stills.append(source_still)

    for strength in args.strengths:
        value = clamp_strength(strength)
        label = f"s{int(value * 100):03d}"
        req = VaceRequest(
            control_video=str(control),
            prompt=args.prompt,
            reference_image=None,
            canvas=args.canvas,
            aspect=args.aspect,
            recipe=args.recipe,
            fit=args.fit,
            length=length,
            source_frames=length,
            strength=value,
            seed=args.seed,
            invert_depth=False,
            derive_depth=False,
            color_match=args.color_match,
            filename_prefix=f"{prefix}_{label}",
        )
        wf, save_node = build_vace_workflow(req)
        started = time.monotonic()
        print(f"-> {label}", flush=True)
        produced = await render(label, wf, save_node)
        print(f"   {produced}  ({time.monotonic() - started:.0f} s)\n", flush=True)
        shot = still(produced, scratch / f"{prefix}_{label}.png", at)
        if shot:
            stills.append(shot)
        async with httpx.AsyncClient(timeout=60) as client:
            await free_memory(client)

    dest = settings.comfyui_output_dir / f"{prefix}_sheet.png"
    if contact_sheet(stills, dest):
        print(f"Contact sheet (source first): {dest}")


if __name__ == "__main__":
    asyncio.run(main())
