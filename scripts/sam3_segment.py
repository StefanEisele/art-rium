"""Turn filmed footage into a flat colour-ID mask video, with SAM 3.

This is the missing front end of the AnimateLCM region path. That path has
always been able to render a different reference picture into each colour of a
mask video (`ColorToMask` -> `IPAdapterAdvanced.attn_mask`, one adapter per
region, all in one sampler pass) — but the mask video had to be authored by
hand in Blender, so it only ever existed for scenes that were built in Blender
to begin with. A filmed clip had no way in. This script writes that mask from
the footage itself: name the things ("tomato", "hand", "wooden board"), and
each one comes back as a flat pure-colour layer that the render keys on.

**It runs in ComfyUI's venv, not art-rium's, and that is the point.**
art-rium has no torch and should not grow one — a second CUDA stack on the same
machine is 2.5 GB of wheels that then compete for the same 16 GB card. The
ComfyUI venv already carries torch 2.11+cu128 and transformers 5.9, and
transformers 5.9 ships SAM 3 natively (`Sam3VideoModel`), so nothing has to be
installed and no ComfyUI custom node has to be added to a stack whose render
behaviour is measured and calibrated. art-rium calls this file as a subprocess
and reads JSON lines off stdout.

Why SAM 3 and not the alternatives
──────────────────────────────────
  SAM 3          text prompt -> every instance of the concept, tracked across
                 the clip with one identity per object. No clicking, no
                 per-frame work, and the temporal consistency is the model's
                 own rather than something bolted on. ~4 GB in fp16.
  SAM 2          tracks well but has no concept prompting: something has to
                 click the first frame, which is exactly the manual step this
                 is meant to remove.
  Grounding DINO
   + SAM 2       text prompting bolted onto SAM 2 with a second model and a
                 box hand-off. Two models, two failure modes, worse masks.
  Mask2Former /
   OneFormer     per-frame panoptic labels with no tracking, so the region a
                 pixel belongs to flickers between frames — and a flickering
                 attn_mask makes the IP-Adapter swap materials mid-clip.

Two things that will silently ruin the output if changed
────────────────────────────────────────────────────────
1. **The mask video is written 4:4:4 and lossless.** `ColorToMask` keys on
   euclidean RGB distance to an exact colour (KJNodes, `mask_nodes.py`), and
   the default tolerance is small. yuv420p halves the chroma resolution, so
   every region boundary gets a ramp of invented in-between colours that
   belongs to no region at all — the outline of every object would drop out of
   its own mask. `-qp 0 -pix_fmt yuv444p` keeps each pixel within a rounding
   error of the colour it was given.
2. **Overlaps are resolved winner-take-all, in list order.** A pixel claimed by
   both "tomato" and "hand" cannot be painted (255,255,0): that colour is 180
   units from red and 180 from green, so it would key into neither region and
   the overlap would become a hole. The concepts are painted back-to-front, so
   the first concept in the list wins every pixel it claims.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import Sam3VideoModel, Sam3VideoProcessor

# The long edge the footage is reduced to before segmentation. SAM 3 resizes to
# its own working resolution internally, so feeding it 1080p buys nothing and
# costs the decode, the transfer and the mask interpolation. The mask is only
# ever consumed after being fitted to a 544-768px render canvas with
# `nearest-exact`, so detail beyond this is thrown away twice over.
SEGMENT_LONG_EDGE = 1024

# Below this the detector's box is treated as noise rather than an object.
DEFAULT_SCORE = 0.5


def _emit(**payload) -> None:
    """One JSON object per line on stdout. The parent reads these as progress."""
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def _probe(src: Path) -> tuple[int, int, float, int]:
    """(width, height, fps, frame_count) of the first video stream."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-of", "json",
         "-show_entries", "stream=width,height,nb_frames,r_frame_rate,duration",
         str(src)],
        capture_output=True, text=True, check=True,
    ).stdout
    s = (json.loads(out).get("streams") or [{}])[0]
    num, _, den = (s.get("r_frame_rate") or "0/1").partition("/")
    fps = float(num) / float(den) if float(den or 0) else 0.0
    frames = int(s.get("nb_frames") or 0)
    if not frames and s.get("duration"):
        frames = int(float(s["duration"]) * fps)
    return int(s.get("width") or 0), int(s.get("height") or 0), fps, frames


def _fit(width: int, height: int, long_edge: int) -> tuple[int, int]:
    """Scale to fit `long_edge`, on even pixels — H.264 will not take odd ones."""
    if max(width, height) <= long_edge:
        scale = 1.0
    else:
        scale = long_edge / max(width, height)
    w = max(2, int(round(width * scale / 2)) * 2)
    h = max(2, int(round(height * scale / 2)) * 2)
    return w, h


def decode(src: Path, spec: dict) -> tuple[np.ndarray, int, int]:
    """Decode the frames this job will actually use, and only those.

    Trimming and decimating here rather than after segmentation is most of the
    speed of the whole feature: SAM 3 is the expensive part and it never sees a
    frame that the render would have thrown away. `-fps_mode passthrough` is
    load-bearing — without it ffmpeg re-times the decimated stream back up to
    the source rate by duplicating frames, and `select` becomes a no-op.
    """
    src_w, src_h, _, _ = _probe(src)
    w, h = _fit(src_w, src_h, spec.get("long_edge") or SEGMENT_LONG_EDGE)

    stride = max(1, int(spec.get("stride") or 1))
    filters = []
    if stride > 1:
        filters.append(f"select='not(mod(n\\,{stride}))'")
    filters.append(f"scale={w}:{h}:flags=bilinear")

    cmd = ["ffmpeg", "-v", "error"]
    if spec.get("start"):
        cmd += ["-ss", str(float(spec["start"]))]
    cmd += ["-i", str(src)]
    if spec.get("seconds"):
        # Applied after -i so it counts source seconds, before decimation.
        cmd += ["-t", str(float(spec["seconds"]))]
    cmd += ["-vf", ",".join(filters), "-fps_mode", "passthrough",
            "-an", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]

    max_frames = int(spec.get("max_frames") or 0)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    frame_bytes = w * h * 3
    frames: list[np.ndarray] = []
    try:
        while not max_frames or len(frames) < max_frames:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            frames.append(np.frombuffer(buf, np.uint8).reshape(h, w, 3))
    finally:
        proc.stdout.close()
        proc.wait(timeout=30)

    if not frames:
        raise RuntimeError("ffmpeg decoded no frames from that video")
    return np.stack(frames), w, h


def segment(video: np.ndarray, spec: dict) -> tuple[np.ndarray, dict]:
    """Run SAM 3 over the clip and paint one flat colour per concept.

    Returns the RGB mask video and a per-concept report of what was found —
    "no object matched this word" is the single most likely way this feature
    disappoints, and it has to reach the user as a message rather than as a
    black layer they are left to interpret.
    """
    model_dir = spec["model_dir"]
    device = spec.get("device") or "cuda"
    dtype = torch.float16 if device.startswith("cuda") else torch.float32

    _emit(event="stage", stage="load", message="SAM 3 wird geladen")
    processor = Sam3VideoProcessor.from_pretrained(model_dir)
    model = Sam3VideoModel.from_pretrained(model_dir, dtype=dtype).to(device).eval()

    # The decoded clip stays on the CPU and frames are pulled across as they
    # come up. On a 16 GB card the model is only ~4 GB, but a 120-frame session
    # held at the model's working resolution is another half gigabyte for no
    # reason — and this may well run while ComfyUI still holds a render.
    session = processor.init_video_session(
        video=video,
        inference_device=device,
        video_storage_device="cpu",
        dtype=dtype,
    )
    concepts = spec["concepts"]
    processor.add_text_prompt(session, [c["text"] for c in concepts])

    total = len(video)
    height, width = video.shape[1], video.shape[2]
    out = np.zeros((total, height, width, 3), np.uint8)
    score_min = float(spec.get("score_threshold") or DEFAULT_SCORE)
    found = {c["text"]: 0 for c in concepts}
    coverage = np.zeros(len(concepts), np.float64)

    started = time.time()
    with torch.inference_mode():
        for step, raw in enumerate(model.propagate_in_video_iterator(session)):
            res = processor.postprocess_outputs(session, raw)
            idx = raw.frame_idx if raw.frame_idx is not None else step
            if not 0 <= idx < total:
                continue

            masks = res["masks"]
            scores = res["scores"]
            by_prompt = res.get("prompt_to_obj_ids") or {}
            ids = [int(i) for i in res["object_ids"].tolist()]
            position = {obj: n for n, obj in enumerate(ids)}

            # Painted back-to-front so that concept 0 keeps every pixel it
            # claims; see the module docstring on why an overlap cannot simply
            # be blended.
            frame = out[idx]
            for c_index in range(len(concepts) - 1, -1, -1):
                concept = concepts[c_index]
                obj_ids = by_prompt.get(concept["text"]) or []
                union = None
                for obj in obj_ids:
                    n = position.get(int(obj))
                    if n is None or float(scores[n]) < score_min:
                        continue
                    m = masks[n].cpu().numpy().astype(bool)
                    union = m if union is None else (union | m)
                if union is None or not union.any():
                    continue
                found[concept["text"]] += 1
                coverage[c_index] += float(union.mean())
                frame[union] = np.array(concept["color"], np.uint8)

            # The first frame already knows whether a word found anything, and
            # saying so after ~50 s beats saying it after four minutes of
            # tracking. The usual cause is a non-English term: SAM 3's text
            # encoder is English-trained, and "Tomate" scores exactly zero on
            # the same clip where "tomato" covers half the frame (measured).
            if step == 0:
                blind = [c["text"] for c in concepts if not found[c["text"]]]
                if blind:
                    _emit(event="unmatched", concepts=blind)

            if step % 4 == 0 or step == total - 1:
                _emit(event="progress", done=step + 1, total=total,
                      elapsed=round(time.time() - started, 1))

    report = {
        c["text"]: {
            "frames": found[c["text"]],
            "coverage": round(coverage[i] / max(1, total), 4),
            "color": c["color"],
        }
        for i, c in enumerate(concepts)
    }
    return out, report


def encode(masks: np.ndarray, dest: Path, fps: float) -> None:
    """Write the mask video losslessly at 4:4:4.

    See the module docstring: subsampled chroma would put a ramp of unkeyable
    in-between colours around every region, and `ColorToMask` measures distance
    to an exact RGB triple.
    """
    total, height, width = masks.shape[:3]
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}", "-r", f"{fps:g}", "-i", "-",
        "-an", "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv444p",
        "-preset", "veryfast", str(dest),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for i in range(total):
            proc.stdin.write(masks[i].tobytes())
    finally:
        proc.stdin.close()
        err = proc.stderr.read().decode(errors="replace")
        proc.wait(timeout=120)
    if proc.returncode != 0 or not dest.is_file():
        raise RuntimeError(err[:400] or "ffmpeg failed to write the mask video")


def encode_preview(video: np.ndarray, masks: np.ndarray, dest: Path, fps: float) -> None:
    """The footage with its regions tinted over it.

    A flat colour-ID video is close to unreadable on its own — three silhouettes
    on black say nothing about whether the right things were caught. This is the
    frame the user actually judges the segmentation on, so it is written even
    though nothing in the render pipeline reads it.
    """
    lit = masks.any(axis=3, keepdims=True)
    blend = np.where(lit, (video * 0.45 + masks * 0.55), video).astype(np.uint8)
    total, height, width = blend.shape[:3]
    dest.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{width}x{height}", "-r", f"{fps:g}", "-i", "-", "-an",
         "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
         "-pix_fmt", "yuv420p", str(dest)],
        stdin=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    try:
        for i in range(total):
            proc.stdin.write(blend[i].tobytes())
    finally:
        proc.stdin.close()
        proc.wait(timeout=120)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", help="path to the job's JSON spec")
    args = parser.parse_args()
    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))

    try:
        _emit(event="stage", stage="decode", message="Video wird gelesen")
        video, width, height = decode(Path(spec["source"]), spec)

        masks, report = segment(video, spec)

        _emit(event="stage", stage="encode", message="Maske wird geschrieben")
        fps = float(spec.get("fps") or 16)
        encode(masks, Path(spec["dest"]), fps)
        if spec.get("preview"):
            encode_preview(video, masks, Path(spec["preview"]), fps)

        _emit(event="done", frames=len(video), width=width, height=height,
              fps=fps, report=report)
        return 0
    except Exception as exc:                       # noqa: BLE001 — reported, not swallowed
        _emit(event="error", message=f"{type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
