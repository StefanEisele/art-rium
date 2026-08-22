"""
Render one AnimateLCM scene across one dial and put the results side by side.

This is the user's own `AnimateDiff LCM Workflow.json`, rebuilt in
`services/comfy/animatelcm.py` — base pass, hires refine, optional RIFE — with
the three settings that actually change the picture exposed as dials:

    --dial depth   how literally the depth pass dictates the picture (base CN)
    --dial ip      how hard the reference picture imposes its material
    --dial hires   the second pass's denoise: how much detail is invented on top

`hires` is the one to start with. It is the original's detail engine, and the
original file had no handle for it at all.

    python scripts/animatelcm_sweep.py ^
        --control E:/00_comfy/input/01_mask_depth_96f0001-0094.mp4 ^
        --reference 260131_RecursiveIdentities_III.jpg ^
        --prompt "..." --dial hires

Costs: the base pass at 544x544 with 9 LCM steps is minutes; the hires pass runs
a second sampler over 1088x1088, so budget roughly three times the base.
`--no-hires` sweeps the base dials cheaply and leaves the refine for later.
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
from core.video_thumb import probe_video_dimensions  # noqa: E402
from services.comfy.animatelcm import (  # noqa: E402
    DEPTH_DEFAULT,
    DEPTH_SWEEP,
    FPS,
    HIRES_DENOISE_DEFAULT,
    HIRES_SWEEP,
    IP_DEFAULT,
    IP_SWEEP,
    AnimateLcmRequest,
    build_animatelcm_workflow,
    output_size,
    sweep_workflows,
)
from services.comfy.client import free_memory, poll_history, post_workflow  # noqa: E402

TIMEOUT = 60 * 60


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--control", required=True, help="absolute path to the control video")
    ap.add_argument("--reference", required=True,
                    help="a picture: any path, or a bare filename already in ComfyUI's input")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--dial", default="hires", choices=["depth", "ip", "hires", "none"],
                    help="'none' renders a single clip at the defaults")
    ap.add_argument("--values", nargs="*", type=float,
                    help="override the swept values")
    ap.add_argument("--depth", type=float, default=DEPTH_DEFAULT,
                    help="held fixed while sweeping the other dial")
    ap.add_argument("--ip", type=float, default=IP_DEFAULT)
    ap.add_argument("--hires-denoise", type=float, default=HIRES_DENOISE_DEFAULT)
    ap.add_argument("--no-hires", action="store_true",
                    help="base pass only — cheap, and about a quarter the detail")
    ap.add_argument("--rife", type=int, default=1, help="interpolation factor, 1 = off")
    ap.add_argument("--aspect", default="auto",
                    choices=["auto", "square", "wide", "tall"],
                    help="auto follows the control track's own shape")
    ap.add_argument("--frames", type=int, default=94)
    ap.add_argument("--steps", type=int, default=9)
    ap.add_argument("--cfg", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1234, help="fixed on purpose")
    ap.add_argument("--no-invert-depth", action="store_true")
    return ap.parse_args()


def stage_reference(name_or_path: str) -> str:
    """LoadImage resolves by bare filename inside ComfyUI's input folder."""
    src = Path(name_or_path)
    if not src.is_absolute() and not src.exists():
        return name_or_path
    if not src.is_file():
        sys.exit(f"Reference not found: {src}")
    inp = settings.comfyui_output_dir.parent / "input"
    inp.mkdir(parents=True, exist_ok=True)
    dest = inp / f"artrium_alcm_ref{src.suffix.lower()}"
    shutil.copy2(src, dest)
    return dest.name


async def render(label: str, wf: dict, save_node: str) -> Path:
    async with httpx.AsyncClient(timeout=240) as client:
        prompt_id = await post_workflow(client, wf)
        outputs = await poll_history(client, prompt_id, timeout=TIMEOUT, interval=4)
    entry = (outputs.get(save_node) or {}).get("gifs") or []
    if not entry:
        raise RuntimeError(f"{label}: ComfyUI returned no video ({list(outputs)})")
    return settings.comfyui_output_dir / entry[0].get("subfolder", "") / entry[0]["filename"]


def contact_sheet(renders: list[tuple[str, Path]], dest: Path, at: float) -> Path | None:
    """One frame from the middle of each clip, stacked left to right."""
    stills = []
    for label, path in renders:
        still = dest.parent / f"{dest.stem}_{label}.png"
        cmd = [settings.ffmpeg_path, "-y", "-v", "error",
               "-i", str(path), "-ss", f"{at:.2f}", "-frames:v", "1", str(still)]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            stills.append(still)
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            print(f"  ! no still for {label}: {exc}")
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

    src_w, src_h = await probe_video_dimensions(control)
    base = AnimateLcmRequest(
        control_video=str(control),
        source_width=src_w,
        source_height=src_h,
        prompt=args.prompt,
        reference_image=stage_reference(args.reference),
        aspect=args.aspect,
        length=args.frames,
        source_frames=args.frames,
        depth_strength=args.depth,
        ip_weight=args.ip,
        hires=not args.no_hires,
        hires_denoise=args.hires_denoise,
        rife=args.rife,
        steps=args.steps,
        cfg=args.cfg,
        seed=args.seed,
        invert_depth=not args.no_invert_depth,
        filename_prefix=f"artrium_alcm_{int(time.time())}",
    )

    if args.dial == "none":
        wf, node = build_animatelcm_workflow(base)
        sheet = [("single", wf, node)]
    else:
        values = tuple(args.values) if args.values else {
            "depth": DEPTH_SWEEP, "ip": IP_SWEEP, "hires": HIRES_SWEEP,
        }[args.dial]
        sheet = [(f"{args.dial}{int(v * 100):03d}", wf, node)
                 for v, wf, node in sweep_workflows(base, args.dial, values)]

    out_w, out_h = output_size(args.aspect, hires=not args.no_hires,
                               source=(src_w, src_h))
    print(f"{len(sheet)} renders, {args.frames} frames, seed {args.seed}, "
          f"depth {args.depth}, ip {args.ip} -> {out_w}x{out_h}\n")

    renders: list[tuple[str, Path]] = []
    for label, wf, save_node in sheet:
        started = time.monotonic()
        print(f"-> {label}", flush=True)
        produced = await render(label, wf, save_node)
        renders.append((label, produced))
        print(f"   {produced}  ({time.monotonic() - started:.0f} s)\n", flush=True)
        async with httpx.AsyncClient(timeout=60) as client:
            await free_memory(client)

    dest = settings.comfyui_output_dir / f"{base.filename_prefix}_sheet.png"
    if contact_sheet(renders, dest, at=args.frames / 2 / FPS):
        print(f"Contact sheet: {dest}")


if __name__ == "__main__":
    asyncio.run(main())
