"""
Scores for scripts/style_channel_sweep.py. Runs in ComfyUI's venv (CLIP via
transformers, numpy, PIL), not art-rium's:

    E:/00_comfy/venv/Scripts/python.exe scripts/style_channel_metrics.py ^
        --out E:/00_comfy/output/style_sweep --dataset storage/lora_datasets/artvision_v3 ^
        --refs 24_6b2b2c73 12_6c74f926

Per video, all numbers on the frames the sampler produced:

    style      CLIP ViT-L/14 cosine of the clip's mean frame embedding to the
               dataset centroid — the dataset WITHOUT the two reference pictures,
               so a channel that just copies its reference earns nothing for it
    style_nn   mean over frames of the best match to any one dataset picture
    copyA/B    cosine to each reference picture: how much was copied
    prompt     CLIPScore-style text/image cosine x100 against the scene prompt
               (without the trigger word): does the scene survive the style
    palette    histogram intersection of the LAB a*/b* distribution with the
               dataset's (1 = same colour distribution)
    structure  Pearson correlation of edge maps, output vs control track: does
               the geometry survive
    sharp      variance of the Laplacian at 512 px (detail / mush)
    flicker    mean / max consecutive-frame mean-abs-diff at 384 px; spike =
               max / median
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from transformers import CLIPModel, CLIPProcessor

FFMPEG = "ffmpeg"
CLIP_ID = "openai/clip-vit-large-patch14"


def read_frames(video: Path, size: int) -> np.ndarray:
    """All frames as uint8 (n, size, size, 3)."""
    raw = subprocess.run(
        [FFMPEG, "-v", "error", "-i", str(video), "-vf", f"scale={size}:{size}:flags=area",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        check=True, capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, size, size, 3)


def rgb2lab(rgb: np.ndarray) -> np.ndarray:
    c = rgb.astype(np.float32) / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    m = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722],
                  [0.0193, 0.1192, 0.9505]], np.float32)
    xyz = c @ m.T / np.array([0.95047, 1.0, 1.08883], np.float32)
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])], -1)


def ab_hist(rgb: np.ndarray) -> np.ndarray:
    lab = rgb2lab(rgb.reshape(-1, 3))
    h, _, _ = np.histogram2d(lab[:, 1], lab[:, 2], bins=24, range=[[-60, 60], [-60, 60]])
    return h / h.sum()


def edges(gray: np.ndarray) -> np.ndarray:
    img = Image.fromarray(gray).filter(ImageFilter.GaussianBlur(1.5))
    g = np.asarray(img, np.float32)
    gx = np.zeros_like(g)
    gy = np.zeros_like(g)
    gx[:, 1:-1] = g[:, 2:] - g[:, :-2]
    gy[1:-1] = g[2:] - g[:-2]
    return np.hypot(gx, gy)


def structure(out: np.ndarray, ctrl: np.ndarray) -> float:
    n = min(len(out), len(ctrl))
    vals = []
    for i in np.linspace(0, n - 1, 6).astype(int):
        a = edges(np.asarray(Image.fromarray(out[i]).convert("L"))).ravel()
        b = edges(np.asarray(Image.fromarray(ctrl[i]).convert("L"))).ravel()
        vals.append(np.corrcoef(a, b)[0, 1])
    return float(np.mean(vals))


def laplacian_var(rgb: np.ndarray) -> float:
    g = np.asarray(Image.fromarray(rgb).convert("L"), np.float32)
    lap = (-4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:])
    return float(lap.var())


def flicker(frames: np.ndarray) -> tuple[float, float, float]:
    f = frames.astype(np.float32)
    d = np.abs(f[1:] - f[:-1]).mean(axis=(1, 2, 3))
    return float(d.mean()), float(d.max()), float(d.max() / max(np.median(d), 1e-6))


class Clip:
    def __init__(self):
        self.model = CLIPModel.from_pretrained(CLIP_ID).eval()
        self.proc = CLIPProcessor.from_pretrained(CLIP_ID)

    @torch.no_grad()
    def images(self, ims: list[Image.Image]) -> torch.Tensor:
        out = []
        for i in range(0, len(ims), 16):
            px = self.proc(images=ims[i:i + 16], return_tensors="pt")
            e = self.model.get_image_features(**px)
            e = getattr(e, "pooler_output", e)
            out.append(torch.nn.functional.normalize(e, dim=-1))
        return torch.cat(out)

    @torch.no_grad()
    def text(self, t: str) -> torch.Tensor:
        tok = self.proc(text=[t], return_tensors="pt", padding=True)
        e = self.model.get_text_features(**tok)
        e = getattr(e, "pooler_output", e)
        return torch.nn.functional.normalize(e, dim=-1)[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--dataset", required=True, type=Path)
    ap.add_argument("--refs", nargs=2, required=True, help="dataset stems of ref A and B")
    ap.add_argument("--manifest", default="manifest.json")
    ap.add_argument("--prompts", type=Path, help="json {scene: prompt}")
    ap.add_argument("--trigger", default="artvis3")
    ap.add_argument("--control", nargs="*", default=[],
                    help="scene=path for scenes whose control track was used as-is")
    args = ap.parse_args()
    controls = dict(c.split("=", 1) for c in args.control)

    manifest = json.loads((args.out / args.manifest).read_text())
    clip = Clip()
    cache = args.out / "_dataset_clip.pt"
    paths = sorted(args.dataset.glob("*.jpg"))
    if cache.exists():
        ds = torch.load(cache)
    else:
        ds = {"stems": [p.stem for p in paths],
              "emb": clip.images([Image.open(p).convert("RGB") for p in paths])}
        hist = np.zeros((24, 24))
        for p in paths:
            hist += ab_hist(np.asarray(Image.open(p).convert("RGB").resize((256, 256))))
        ds["hist"] = torch.from_numpy(hist / len(paths))
        torch.save(ds, cache)
    keep = [i for i, s in enumerate(ds["stems"]) if s not in args.refs]
    pool = ds["emb"][keep]
    centroid = torch.nn.functional.normalize(pool.mean(0), dim=0)
    ref_emb = {k: ds["emb"][ds["stems"].index(s)] for k, s in zip("AB", args.refs)}

    prompts = json.loads(args.prompts.read_text()) if args.prompts else {}
    text_cache: dict[str, torch.Tensor] = {}
    ctrl_cache: dict[str, np.ndarray] = {}
    inputs = args.out / "_inputs"
    scores = {}
    for key, info in manifest.items():
        scene = info["scene"]
        frames = read_frames(Path(info["video"]), 384)
        pick = np.linspace(0, len(frames) - 1, 8).astype(int)
        big = read_frames(Path(info["video"]), 512)
        emb = clip.images([Image.fromarray(big[i]) for i in pick])
        mean = torch.nn.functional.normalize(emb.mean(0), dim=0)
        prompt = prompts.get(scene) or info["prompt"]
        if args.trigger and prompt.startswith(args.trigger + ", "):
            prompt = prompt[len(args.trigger) + 2:]
        if scene not in text_cache:
            text_cache[scene] = clip.text(prompt)
        if scene not in ctrl_cache:
            ctrl = inputs / f"{scene}_control.mp4"
            src = ctrl if ctrl.exists() else Path(controls.get(scene, ""))
            ctrl_cache[scene] = read_frames(src, 384) if src.exists() else None
        fm, fx, spike = flicker(frames)
        scores[key] = {
            "scene": scene, "variant": info["variant"], "seconds": info["seconds"],
            "style": round(float(mean @ centroid), 4),
            "style_nn": round(float((emb @ pool.T).max(1).values.mean()), 4),
            "copyA": round(float(mean @ ref_emb["A"]), 4),
            "copyB": round(float(mean @ ref_emb["B"]), 4),
            "prompt": round(float((emb @ text_cache[scene]).mean()) * 100, 2),
            "palette": round(float(np.minimum(ab_hist(big[pick]), ds["hist"].numpy()).sum()), 4),
            "structure": (round(structure(frames, ctrl_cache[scene]), 4)
                          if ctrl_cache[scene] is not None else None),
            "sharp": round(float(np.mean([laplacian_var(big[i]) for i in pick])), 1),
            "flicker_mean": round(fm, 2), "flicker_max": round(fx, 2),
            "spike": round(spike, 2),
        }
        print(key, scores[key], flush=True)
    name = args.manifest.replace("manifest", "scores")
    (args.out / name).write_text(json.dumps(scores, indent=2), encoding="utf-8")

    # One contact sheet per scene: rows = variants, 4 moments.
    for scene in sorted({s["scene"] for s in scores.values()}):
        rows = [(k, manifest[k]) for k in manifest if manifest[k]["scene"] == scene]
        cw = 200
        margin = 230
        sheet = Image.new("RGB", (margin + 4 * cw, len(rows) * cw), (16, 16, 16))
        draw = ImageDraw.Draw(sheet)
        font = ImageFont.load_default(size=17)
        for r, (k, info) in enumerate(rows):
            fr = read_frames(Path(info["video"]), cw)
            s = scores[k]
            label = (f"{info['variant']}\nstyle {s['style']:.3f}\nprompt {s['prompt']:.1f}"
                     f"\nstruct {s['structure']}\nflick {s['flicker_mean']}")
            draw.multiline_text((8, r * cw + 10), label, fill=(235, 235, 235), font=font)
            for c, i in enumerate(np.linspace(0, len(fr) - 1, 4).astype(int)):
                sheet.paste(Image.fromarray(fr[i]), (margin + c * cw, r * cw))
        sheet.save(args.out / f"sheet_{scene}{name[6:-5]}.jpg", quality=88)


if __name__ == "__main__":
    main()
