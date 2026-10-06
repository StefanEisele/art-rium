"""Paint candidate icons for the art-rium hub, and install the ones you pick.

The hub's tool cards (frontends/dashboard/index.html) used emoji for icons.
This paints proper ones: Z-Image renders one object on a plain grey backdrop,
BRIA RMBG cuts it out in the same ComfyUI job, and the cut-out is trimmed to a
square so every icon fills its tile alike. Ported from the sticker pipeline of
17_kids_app (core/render.py + core/wardrobe_stickers.py), which paid for the
hard lessons — prompts/ui-icon-styles.md lists the ones that carried over.

Every tool has a couple of motifs (MOTIFS below), and each motif is painted in
every style of prompts/ui-icon-styles.md, so one run is a grid to choose from:

    python scripts/ui_icons.py paint                    # every tool, every style
    python scripts/ui_icons.py paint video music -n 2   # two seeds per motif × style
    python scripts/ui_icons.py paint --style keramik    # one style only

Candidates accumulate across runs (a good one from an earlier run is never
painted over); delete storage/ui_icon_candidates/ to start from scratch. Look
at the contact sheet the run writes there, then install the winners:

    python scripts/ui_icons.py pick video=keramik-2 music=objekt-1
    python scripts/ui_icons.py sheet                    # redraw the sheet only

`pick` writes frontends/dashboard/icons/<tool>.webp. The dashboard's service
worker serves icons cache-first, so bump its cache name in
frontends/dashboard/sw.js after installing new ones.

Needs ComfyUI reachable (with ComfyUI-BRIA_AI-RMBG), not art-rium itself. The
renders are submitted under their own client id, so a running art-rium's
listener never ingests them into the gallery.
"""

import argparse
import asyncio
import io
import json
import random
import sys
from pathlib import Path

# Allow running as `python scripts/ui_icons.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx                                                    # noqa: E402
from PIL import Image, ImageDraw, ImageFont                     # noqa: E402

from core.config import settings                                # noqa: E402
from services.comfy import zimage                               # noqa: E402
from services.comfy.client import poll_history, post_workflow   # noqa: E402
from services.comfy.vram import evict_ollama                    # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
STYLE_FILE = ROOT / "prompts" / "ui-icon-styles.md"
ICON_DIR = ROOT / "frontends" / "dashboard" / "icons"
DEFAULT_OUT = ROOT / "storage" / "ui_icon_candidates"

# What each hub tool shows, keyed by its URL segment (/tools/<id>/; "shop" is
# the not-yet-built Shop Prep card). Things you could hold in your hand only —
# an abstract idea ("speed", "publishing") makes the model paint a scene.
#
# The first full run (66 renders) taught two more things:
# - The shape has to carry the motif, because the style decides the material:
#   "an optical glass prism" came out as a porcelain pyramid, "a camera lens"
#   as a cup.
# - Long thin things lie diagonally. Upright, a pen or a brush is trimmed to a
#   tall sliver and is a single line in a 46 px tile.
MOTIFS: dict[str, list[str]] = {
    "z-image":   ["a camera aperture with fanned iris blades, wide open",
                  "a painter's flat brush lying diagonally"],
    "video":     ["a cinema film reel", "a film clapperboard"],
    "video-api": ["a sculpted cloud", "a satellite dish"],
    "vace":      ["a wireframe icosahedron", "a faceted low-poly polyhedron"],
    "embeddings": ["a painter's colour swatch fan deck, spread open in a fan",
                   "a small glass jar of coloured pigment powder with a cork"],
    "improv":    ["a grand piano with its lid open", "a short row of piano keys"],
    "music":     ["a vinyl record with fine grooves, tilted at an angle",
                  "a pair of over-ear studio headphones"],
    "gallery":   ["an ornate baroque picture frame with carved corners",
                  "a painter's easel with a blank canvas"],
    "titler":    ["a fountain pen lying diagonally", "a fountain pen nib seen up close"],
    "instagram": ["a vintage rangefinder camera", "an instant film camera"],
    "articles":  ["a typewriter", "a folded newspaper"],
    "shop":      ["a paper hang tag on a string", "a shopping bag"],
}

RENDER_SIZE = 768       # square; the icon shows at 46 CSS px, this is plenty
CANDIDATE_SIZE = 512    # kept per candidate, trimmed — room for a PWA icon later
ICON_SIZE = 256         # installed: the hub shows 76 px × 3 on a phone, 88 px × 2 on a desktop
TRIM_PAD = 0.05         # of the object's longer side, all round
CLIENT_ID = "art-rium-ui-icons"

_MASK_NODE = "save_mask"
# BRIA leaves a faint haze over the backdrop; below this it is backdrop.
_MASK_FLOOR = 16
# A cut-out that keeps less than this of the picture found no object at all.
_MIN_COVER = 0.02


class EmptyCutout(Exception):
    pass


def load_styles() -> dict[str, str]:
    """{name: text} from the `## name` sections of STYLE_FILE."""
    sections = STYLE_FILE.read_text(encoding="utf-8").split("\n## ")[1:]
    return {name.strip(): body.strip()
            for name, _, body in (s.partition("\n") for s in sections)}


def build_workflow(prompt: str, seed: int, prefix: str) -> dict:
    """Z-Image exactly as the z-Image tool renders it (no LoRA, Detail Daemon
    at 0), with BRIA RMBG on the decoded picture.

    We keep BRIA's *mask*, not its RGBA output: that one is pasted onto
    transparent black, which darkens every soft edge into a dark fringe. The
    picture's own colours with the mask as alpha stay clean.
    """
    wf = zimage.build_zimage_workflow(prompt, seed, RENDER_SIZE, RENDER_SIZE, loras=[])
    save = wf[zimage.ZIMAGE_SAVE_NODE]["inputs"]
    save["filename_prefix"] = prefix
    wf["rmbg_model"] = {"class_type": "BRIA_RMBG_ModelLoader_Zho", "inputs": {}}
    wf["rmbg"] = {"class_type": "BRIA_RMBG_Zho", "inputs": {
        "rmbgmodel": ["rmbg_model", 0], "image": save["images"],
    }}
    wf["mask_image"] = {"class_type": "MaskToImage", "inputs": {"mask": ["rmbg", 1]}}
    wf[_MASK_NODE] = {"class_type": "SaveImage", "inputs": {
        "images": ["mask_image", 0], "filename_prefix": f"{prefix}_mask",
    }}
    return wf


def take_output(outputs: dict, node: str) -> bytes:
    """A SaveImage output, read straight from ComfyUI's output folder (same
    machine) and then deleted there — the trimmed candidate is what we keep."""
    entry = outputs[node]["images"][0]
    path = settings.comfyui_output_dir / entry.get("subfolder", "") / entry["filename"]
    data = path.read_bytes()
    path.unlink(missing_ok=True)
    return data


def cutout(picture: bytes, mask: bytes) -> Image.Image:
    """Picture + BRIA mask → the object on transparency, cropped to it and
    centred on a square with TRIM_PAD all round. Renders leave 25–40 % empty
    margin, different for every object; trimmed, every icon fills its tile
    alike."""
    with Image.open(io.BytesIO(picture)) as im, Image.open(io.BytesIO(mask)) as m:
        im = im.convert("RGB")
        alpha = m.convert("L").resize(im.size, Image.BILINEAR)
    alpha = alpha.point(lambda v: 0 if v < _MASK_FLOOR else v)
    cover = sum(alpha.histogram()[128:]) / (im.width * im.height)
    if cover < _MIN_COVER:
        raise EmptyCutout(f"cut-out kept only {cover:.1%} of the picture")
    im.putalpha(alpha)
    obj = im.crop(alpha.point(lambda v: 255 if v >= 24 else 0).getbbox())
    side = round(max(obj.size) * (1 + 2 * TRIM_PAD))
    square = Image.new("RGBA", (side, side))
    square.paste(obj, ((side - obj.width) // 2, (side - obj.height) // 2))
    return square.resize((CANDIDATE_SIZE, CANDIDATE_SIZE), Image.LANCZOS)


async def render(client: httpx.AsyncClient, prompt: str, seed: int, prefix: str) -> Image.Image:
    wf = build_workflow(prompt, seed, prefix)
    prompt_id = await post_workflow(client, wf, client_id=CLIENT_ID)
    # Generous: the first job of a run also loads Z-Image and BRIA.
    outputs = await poll_history(client, prompt_id, timeout=600, interval=1)
    return cutout(take_output(outputs, zimage.ZIMAGE_SAVE_NODE), take_output(outputs, _MASK_NODE))


def read_index(out: Path) -> dict:
    path = out / "index.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def next_number(index: dict, tool: str, style: str) -> int:
    taken = [int(k.rsplit("-", 1)[1]) for k in index if k.startswith(f"{tool}__{style}-")]
    return max(taken, default=0) + 1


async def paint(tools: list[str], styles: dict[str, str], per_motif: int, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    index = read_index(out)
    try:
        await evict_ollama()    # a resident titler VLM holds gigabytes of the card
    except Exception as e:
        print(f"(Ollama not evicted: {e})")
    total = len(styles) * per_motif * sum(len(MOTIFS[t]) for t in tools)
    done = 0
    async with httpx.AsyncClient(timeout=60) as client:
        for tool in tools:                          # one at a time: the GPU is a queue of one
            for style, text in styles.items():
                n = next_number(index, tool, style)
                for motif in MOTIFS[tool]:
                    for _ in range(per_motif):
                        stem = f"{tool}__{style}-{n}"
                        n += 1
                        done += 1
                        seed = random.randint(0, 2**32 - 1)
                        prompt = f"{motif[0].upper()}{motif[1:]}, {text}"
                        try:
                            icon = await render(client, prompt, seed, f"art-rium/ui_icons/{stem}")
                        except EmptyCutout as e:
                            print(f"  [{done}/{total}] {stem}: {e} — skipped", flush=True)
                            continue
                        icon.save(out / f"{stem}.png")
                        index[stem] = {"tool": tool, "style": style, "motif": motif,
                                       "seed": seed, "prompt": prompt}
                        (out / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
                        print(f"  [{done}/{total}] {stem}  ({motif})", flush=True)


# The hub's own colours (frontends/shared/shared.css), so the sheet shows each
# candidate where it will actually sit. The small tile is .tool-card.active
# .tool-icon: --accent-soft and --accent-border composited over --surface.
_BG, _CARD = (10, 10, 12), (20, 20, 23)
_TILE, _TILE_BORDER = (49, 34, 28), (112, 66, 39)
_TEXT, _DIM = (240, 240, 243), (138, 138, 147)
_CELL_W, _CELL_H, _LABEL_W = 180, 240, 130


def contact_sheet(out: Path) -> Path:
    """Every candidate on the hub's card surface, one row per tool — large, and
    at its real 46 px tile size next to its key, because an icon that only
    reads large is no icon."""
    index = read_index(out)
    rows = [(tool, sorted((s for s, e in index.items() if e["tool"] == tool and (out / f"{s}.png").exists()),
                          key=lambda s: (index[s]["style"], int(s.rsplit("-", 1)[1]))))
            for tool in MOTIFS]
    rows = [(tool, stems) for tool, stems in rows if stems]
    if not rows:
        sys.exit(f"no candidates in {out}")
    width = _LABEL_W + _CELL_W * max(len(stems) for _, stems in rows)
    sheet = Image.new("RGB", (width, _CELL_H * len(rows)), _BG)
    draw = ImageDraw.Draw(sheet)
    font, small = ImageFont.load_default(size=15), ImageFont.load_default(size=12)
    for r, (tool, stems) in enumerate(rows):
        y = r * _CELL_H
        draw.text((12, y + 80), tool, fill=_TEXT, font=font)
        for c, stem in enumerate(stems):
            x = _LABEL_W + c * _CELL_W
            draw.rounded_rectangle((x + 8, y + 8, x + 172, y + 172), radius=14, fill=_CARD)
            draw.rounded_rectangle((x + 8, y + 182, x + 54, y + 228), radius=12,
                                   fill=_TILE, outline=_TILE_BORDER)
            with Image.open(out / f"{stem}.png") as im:
                big = im.resize((148, 148), Image.LANCZOS)
                tiny = im.resize((36, 36), Image.LANCZOS)
            sheet.paste(big, (x + 16, y + 16), big)
            sheet.paste(tiny, (x + 13, y + 187), tiny)
            motif = index[stem]["motif"].removeprefix("a ").removeprefix("an ")
            draw.text((x + 64, y + 186), stem.split("__", 1)[1], fill=_TEXT, font=font)
            draw.text((x + 64, y + 208), motif if len(motif) <= 17 else motif[:16] + "…",
                      fill=_DIM, font=small)
    path = out / "sheet.png"
    sheet.save(path)
    return path


def pick(choices: list[str], out: Path) -> None:
    ICON_DIR.mkdir(exist_ok=True)
    for choice in choices:
        tool, _, key = choice.partition("=")
        src = out / f"{tool}__{key}.png"
        if tool not in MOTIFS or not key or not src.exists():
            sys.exit(f"no candidate {tool}={key} (expected e.g. video=keramik-2, see {out / 'sheet.png'})")
        with Image.open(src) as im:
            im.resize((ICON_SIZE, ICON_SIZE), Image.LANCZOS).save(
                ICON_DIR / f"{tool}.webp", quality=90, method=6)
        print(f"  {ICON_DIR / f'{tool}.webp'}  <- {src.name}")
    print("Bump the cache name in frontends/dashboard/sw.js so installed apps pick them up.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="candidate folder")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("paint", help="render candidates and write the contact sheet")
    p.add_argument("tools", nargs="*", help=f"tool ids (default: all) — {', '.join(MOTIFS)}")
    p.add_argument("--style", action="append", help="only this style (repeatable)")
    p.add_argument("-n", "--per-motif", type=int, default=1, help="seeds per motif × style")
    k = sub.add_parser("pick", help="install candidates as the hub's icons")
    k.add_argument("choices", nargs="+", help="tool=candidate, e.g. video=keramik-2")
    sub.add_parser("sheet", help="redraw the contact sheet only")
    args = parser.parse_args()
    sys.stdout.reconfigure(errors="replace")    # a German-Windows console chokes on "…"/"←"

    if args.cmd == "pick":
        pick(args.choices, args.out)
        return
    if args.cmd == "paint":
        styles = load_styles()
        unknown = [t for t in args.tools if t not in MOTIFS] + \
                  [s for s in args.style or [] if s not in styles]
        if unknown:
            sys.exit(f"unknown: {', '.join(unknown)} — tools: {', '.join(MOTIFS)}; "
                     f"styles: {', '.join(styles)}")
        if args.style:
            styles = {s: styles[s] for s in args.style}
        asyncio.run(paint(args.tools or list(MOTIFS), styles, args.per_motif, args.out))
    print(f"contact sheet: {contact_sheet(args.out)}")


if __name__ == "__main__":
    main()
