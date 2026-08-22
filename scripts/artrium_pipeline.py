"""
The whole two-stage video pass in one command: art direction, repaint, upscale,
grain.

    stage A   AnimateLCM + Juggernaut + depth ControlNet + IP-Adapter, 768x432
              composition, palette and material — the half VACE cannot do,
              because its reference is latent context and this one is
              cross-attention on every step
    stage B   Wan VACE repaint at 1280x720, the stage-A render as control video
              coherence, light and invented detail — the half AnimateLCM cannot
              do
    upscale   SEEDVR2 restoration to a 1080 short edge
    grain     the thing that makes "35mm" read as film rather than as a clean
              render

Each stage writes a file and prints its path, so a failed run can be resumed by
hand from the last good one with `--from`.

Two facts worth carrying into any judgement of the output:

**Detail comes from the gap.** VACE invents where pixels have to be added. A
768x432 source repainted at 832x480 has almost no gap and comes back looking
like the source, only softer — measured. At 1280x720 the gap is 2.7x, and that
is where the detail lives.

**The scene changes with the canvas.** A seed means something different at every
latent shape, so the same seed at 720p is not the 480p clip in more pixels. It
is a different picture.

    python scripts/artrium_pipeline.py ^
        --control E:/00_comfy/input/01_mask_depth_96f0001-0094.mp4 ^
        --reference 260131_RecursiveIdentities_III.jpg ^
        --prompt "..."
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
from core.video_thumb import probe_video_dimensions, probe_video_duration  # noqa: E402
from services.comfy.animatelcm import (  # noqa: E402
    DEPTH_DEFAULT,
    AnimateLcmRequest,
    build_animatelcm_workflow,
)
from services.comfy.client import free_memory, poll_history, post_workflow  # noqa: E402
from services.comfy.vace import VaceRequest, build_vace_workflow, snap_length  # noqa: E402
from services.video.grain import render_grain  # noqa: E402
from services.video.upscale import (  # noqa: E402
    build_upscale_workflow,
    clamp_resolution,
    output_dimensions,
)
from services.video.upscale import estimate_seconds as upscale_seconds  # noqa: E402

TIMEOUT = 4 * 60 * 60
STAGES = ("a", "b", "upscale", "grain")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--control", help="depth track for stage A (required unless --from)")
    ap.add_argument("--reference", help="the material picture for stage A")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--from", dest="start", default="a", choices=STAGES,
                    help="resume at this stage; needs --input")
    ap.add_argument("--input", help="the file to resume from")
    # Stage A dials — ip 0.95 measured as the strongest material transfer.
    ap.add_argument("--ip", type=float, default=0.95)
    ap.add_argument("--depth", type=float, default=DEPTH_DEFAULT)
    # Stage B — 0.65..0.85 measured as the window where stage A survives.
    ap.add_argument("--repaint", type=float, default=0.75)
    ap.add_argument("--canvas", default="final", choices=["sketch", "final"])
    ap.add_argument("--aspect", default="wide", choices=["wide", "tall", "square"])
    ap.add_argument("--recipe", default="fast", choices=["fast", "rich", "full", "small"])
    ap.add_argument("--frames", type=int, default=94)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--resolution", type=int, default=1080, help="0 skips the upscale")
    ap.add_argument("--grain", type=int, default=35, help="0 skips the grain")
    # Measured: the repaint drifts to saturated cyan/orange whatever the prompt
    # says, so this is on by default rather than opt-in.
    ap.add_argument("--color-match", type=float, default=0.8,
                    help="0..1, grade stage B back onto stage A's palette")
    ap.add_argument("--no-invert-depth", action="store_true")
    return ap.parse_args()


def stage_reference(name_or_path: str) -> str:
    src = Path(name_or_path)
    if not src.is_absolute() and not src.exists():
        return name_or_path
    if not src.is_file():
        sys.exit(f"Reference not found: {src}")
    inp = settings.comfyui_output_dir.parent / "input"
    inp.mkdir(parents=True, exist_ok=True)
    dest = inp / f"artrium_pipe_ref{src.suffix.lower()}"
    shutil.copy2(src, dest)
    return dest.name


async def submit(label: str, wf: dict, save_node: str) -> Path:
    started = time.monotonic()
    async with httpx.AsyncClient(timeout=240) as client:
        prompt_id = await post_workflow(client, wf)
        outputs = await poll_history(client, prompt_id, timeout=TIMEOUT, interval=4)
    entry = (outputs.get(save_node) or {}).get("gifs") or []
    if not entry:
        raise RuntimeError(f"{label}: ComfyUI returned no video ({list(outputs)})")
    produced = settings.comfyui_output_dir / entry[0].get("subfolder", "") / entry[0]["filename"]
    if not produced.exists():
        raise FileNotFoundError(f"{label}: rendered file missing at {produced}")
    print(f"   {produced}  ({time.monotonic() - started:.0f} s)\n", flush=True)
    async with httpx.AsyncClient(timeout=60) as client:
        await free_memory(client)
    return produced


def still(path: Path, dest: Path, at: float) -> None:
    cmd = [settings.ffmpeg_path, "-y", "-v", "error",
           "-i", str(path), "-ss", f"{at:.2f}", "-frames:v", "1", str(dest)]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass


async def main() -> None:
    args = parse_args()
    order = list(STAGES)
    start_at = order.index(args.start)
    if start_at and not args.input:
        sys.exit("--from needs --input")
    if not start_at and not (args.control and args.reference):
        sys.exit("stage A needs --control and --reference")

    prefix = f"artrium_pipe_{int(time.time())}"
    frames = args.frames
    current = Path(args.input) if args.input else None
    if current and not current.is_file():
        sys.exit(f"--input not found: {current}")

    # ── Stage A ──────────────────────────────────────────────────────────────
    if start_at <= 0:
        control = Path(args.control)
        if not control.is_file():
            sys.exit(f"Control track not found: {control}")
        print(f"[A] AnimateLCM 768x432, depth {args.depth}, ip {args.ip}", flush=True)
        wf, node = build_animatelcm_workflow(AnimateLcmRequest(
            control_video=str(control),
            prompt=args.prompt,
            reference_image=stage_reference(args.reference),
            length=frames,
            source_frames=frames,
            depth_strength=args.depth,
            ip_weight=args.ip,
            seed=args.seed,
            invert_depth=not args.no_invert_depth,
            filename_prefix=f"{prefix}_a",
        ))
        current = await submit("stage A", wf, node)

    # ── Stage B ──────────────────────────────────────────────────────────────
    if start_at <= 1:
        length = snap_length(frames)
        print(f"[B] VACE repaint {args.canvas}/{args.aspect}/{args.recipe}, "
              f"strength {args.repaint}", flush=True)
        wf, node = build_vace_workflow(VaceRequest(
            control_video=str(current),
            prompt=args.prompt,
            reference_image=None,      # the material is already in the control track
            canvas=args.canvas,
            aspect=args.aspect,
            recipe=args.recipe,
            fit="crop",                # pad bars would become content here
            length=length,
            source_frames=length,
            strength=args.repaint,
            seed=args.seed,
            invert_depth=False,
            derive_depth=False,
            color_match=args.color_match,
            filename_prefix=f"{prefix}_b",
        ))
        current = await submit("stage B", wf, node)

    # ── SEEDVR2 ──────────────────────────────────────────────────────────────
    if start_at <= 2 and args.resolution:
        resolution = clamp_resolution(args.resolution)
        width, height = await probe_video_dimensions(current)
        if not width or not height:
            # ffprobe can come back empty; guessing a size here would hand
            # SEEDVR2 a wrong target rather than fail visibly.
            sys.exit(f"could not probe {current} — rerun with --from upscale --input <file>")
        out_w, out_h = output_dimensions(width, height, resolution)
        duration = await probe_video_duration(current)
        eta = upscale_seconds(duration, out_w, out_h)
        print(f"[U] SEEDVR2 {width}x{height} -> {out_w}x{out_h}, ~{eta // 60} min",
              flush=True)
        if (out_w, out_h) == (width, height):
            print("   already at or above the target, skipping\n")
        else:
            wf, node = build_upscale_workflow(
                current, resolution=resolution,
                filename_prefix=f"{prefix}_u", has_audio=False,
            )
            current = await submit("upscale", wf, node)

    # ── Grain ────────────────────────────────────────────────────────────────
    if start_at <= 3 and args.grain:
        print(f"[G] film grain, strength {args.grain}", flush=True)
        grained = settings.comfyui_output_dir / f"{prefix}_final.mp4"
        started = time.monotonic()
        await render_grain(current, grained, args.grain,
                           ffmpeg_path=settings.ffmpeg_path)
        print(f"   {grained}  ({time.monotonic() - started:.0f} s)\n", flush=True)
        current = grained

    duration = await probe_video_duration(current)
    width, height = await probe_video_dimensions(current)
    still(current, current.with_suffix(".png"), duration / 2)
    print(f"FINAL  {current}  ({width}x{height}, {duration:.1f} s)")


if __name__ == "__main__":
    asyncio.run(main())
