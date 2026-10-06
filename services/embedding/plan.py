"""Everything about an embedding training that can be decided without running it.

The trainer is kohya's `train_textual_inversion.py` in its own venv
(settings.sd_scripts_python), so — like the SAM 3 worker — this is a process
boundary. What crosses it is a command line, a dataset TOML and a prompt file;
what comes back is a tqdm bar on stderr and files on disk. Both directions are
pinned here, as pure functions, because a renamed flag or a changed file name
fails minutes into a job rather than at import.

Defaults, and where they came from
──────────────────────────────────
Measured 2026-09-28 on the 4060 Ti, "Existential Echoes" (10 pictures), SD 1.5
Juggernaut Reborn, batch 2, fp16, SDPA: **3.5 steps/s**, so 2000 steps is ~10
minutes plus a few seconds per preview image.

  learning rate  5e-3, the A1111 default. sd-scripts' own docs say 1e-6, which
                 is a LoRA value: AdamW moves a parameter by about the learning
                 rate per step, so 1600 steps at 1e-6 shift a vector by ~0.0016
                 against CLIP token entries of ~0.014 — nothing is learned. At
                 5e-3 the smoke run had the user's rust-orange-on-steel-blue
                 staging after 100 steps.
  vectors        8. kentskooking uses 70 ("fill it up and overtrain"); 75 is
                 the ceiling, because an embedding has to fit one 77-token
                 chunk with its BOS/EOS. More vectors = more capacity, and a
                 bigger share of the prompt in an inline render.
  template       "style" — kohya's "a painting in the style of {}" set, so no
                 captions are needed. "object" is for a thing rather than a look.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from pathlib import Path

# ── Dials ────────────────────────────────────────────────────────────────────
VECTORS_MIN, VECTORS_MAX, VECTORS_DEFAULT = 1, 75, 8
STEPS_MIN, STEPS_MAX, STEPS_DEFAULT = 200, 10000, 2000
LR_MIN, LR_MAX, LR_DEFAULT = 1e-4, 2e-2, 5e-3
TEMPLATES = ("style", "object")
DEFAULT_INIT_WORD = {"style": "painting", "object": "object"}
MIN_IMAGES, MAX_IMAGES = 3, 200
SAVE_EVERY = 250
BATCH_SIZE = 2
RESOLUTION = 512
# Prepared pictures are downsized to this long edge. Bucketing brings them to
# ~512² anyway; this only keeps the dataset folder from holding 8K upscales.
DATASET_MAX_EDGE = 1024

# Where the files land, relative to settings.embeddings_dir. The active file
# sits one level up from the training folder so ComfyUI's list stays readable:
# `artrium/<name>` is the embedding, `artrium/_train/...` is its workshop.
NAMESPACE = "artrium"
TRAIN_SUBDIR = "_train"

# ~2.5 GB of fp16 weights, the UNet activations of a batch of two at 512², and
# the optimiser state of a few thousand floats. Measured headroom, not a guess
# at the peak: the smoke run never came near it.
TRAIN_VRAM = 7.0e9

# One fixed seed per prompt, so a row of snapshots differs only by the
# embedding. The monolith is the scene the Phase-0 sweeps used; the other two
# ask the embedding for a figure and for an empty space, which is where a style
# either shows or does not.
SAMPLE_PROMPTS = (
    ("a monolith in a flooded hall", 1234),
    ("a lone figure standing in a vast room", 777),
    ("an empty landscape at dusk", 4242),
)

_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{1,39}$")


class PlanError(ValueError):
    """A training request that cannot run as asked — message is for the UI."""


# ── Names ────────────────────────────────────────────────────────────────────
def normalize_name(raw: str) -> str:
    """The file stem, and so the word a render uses. Lower-case ASCII, digits,
    `-` and `_`; umlauts are spelled out rather than dropped, so "Rostblüte"
    stays readable as `rostbluete`."""
    text = (raw or "").strip().lower()
    for a, b in (("ä", "ae"), ("ö", "oe"), ("ü", "ue"), ("ß", "ss")):
        text = text.replace(a, b)
    text = re.sub(r"[\s.]+", "-", text)
    text = re.sub(r"[^a-z0-9_-]", "", text).strip("-_")
    if not _NAME.match(text):
        raise PlanError(
            "Name: 2–40 Zeichen, Buchstaben, Ziffern, - und _ (beginnt mit Buchstabe oder Ziffer)"
        )
    return text


def token_for(training_id: uuid.UUID) -> str:
    """The token the trainer adds to CLIP's vocabulary.

    It must not already exist there — kohya asserts that every vector got a
    fresh token — so it is derived from the row id rather than from the name,
    which is an English word often enough to collide.
    """
    return f"ar{training_id.hex[:10]}"


def active_name(name: str) -> str:
    """What a render writes after `embedding:`."""
    return f"{NAMESPACE}/{name}"


def active_file(embeddings_dir: Path, name: str) -> Path:
    return embeddings_dir / NAMESPACE / f"{name}.safetensors"


def train_dir(embeddings_dir: Path, name: str) -> Path:
    return embeddings_dir / NAMESPACE / TRAIN_SUBDIR / name


def snapshot_name(name: str, step: int | None) -> str:
    """The ComfyUI name of one snapshot; None is the final save."""
    stem = name if step is None else f"{name}-step{step:08d}"
    return f"{NAMESPACE}/{TRAIN_SUBDIR}/{name}/{stem}"


def is_workshop_name(comfy_name: str) -> bool:
    """A snapshot or final save inside a training folder — hidden from the
    picker, which offers each trained embedding once, by its active name."""
    return comfy_name.replace("\\", "/").startswith(f"{NAMESPACE}/{TRAIN_SUBDIR}/")


# ── The request ──────────────────────────────────────────────────────────────
@dataclass
class TrainingPlan:
    name: str
    token: str
    template: str
    init_word: str
    vectors: int
    steps: int
    learning_rate: float
    image_count: int


def _clamp(value, low, high, fallback):
    if value is None:
        return fallback
    return min(max(value, low), high)


def plan_training(
    training_id: uuid.UUID,
    name: str,
    image_count: int,
    *,
    template: str = "style",
    init_word: str | None = None,
    vectors: int | None = None,
    steps: int | None = None,
    learning_rate: float | None = None,
) -> TrainingPlan:
    """Validate and complete a request. Raises PlanError with a UI message."""
    if template not in TEMPLATES:
        raise PlanError(f"Vorlage muss eine von {TEMPLATES} sein")
    if image_count < MIN_IMAGES:
        raise PlanError(f"Mindestens {MIN_IMAGES} Bilder — kentskooking nimmt etwa 20")
    if image_count > MAX_IMAGES:
        raise PlanError(f"Höchstens {MAX_IMAGES} Bilder pro Training")
    word = (init_word or DEFAULT_INIT_WORD[template]).strip().lower()
    # kohya initialises every vector from this word's token(s); a phrase would
    # silently be cycled over the vectors, which is not what anyone typing one
    # means.
    if not re.fullmatch(r"[a-z]{2,32}", word):
        raise PlanError("Startwort: ein einzelnes englisches Wort, z. B. painting")
    return TrainingPlan(
        name=normalize_name(name),
        token=token_for(training_id),
        template=template,
        init_word=word,
        vectors=int(_clamp(vectors, VECTORS_MIN, VECTORS_MAX, VECTORS_DEFAULT)),
        steps=int(_clamp(steps, STEPS_MIN, STEPS_MAX, STEPS_DEFAULT)),
        learning_rate=float(_clamp(learning_rate, LR_MIN, LR_MAX, LR_DEFAULT)),
        image_count=image_count,
    )


def estimate_seconds(steps: int) -> int:
    """Measured, not modelled: 3.5 steps/s at batch 2 on the 4060 Ti, plus
    ~15 s to load and ~1.5 s per preview image."""
    previews = (steps // SAVE_EVERY + 1) * len(SAMPLE_PROMPTS)
    return int(15 + steps / 3.5 + previews * 1.5)


# ── Files handed to the trainer ──────────────────────────────────────────────
def _toml_path(path: Path) -> str:
    # TOML basic strings treat backslashes as escapes; forward slashes are
    # valid Windows paths and need none.
    return str(path).replace("\\", "/")


def dataset_toml(dataset_dir: Path) -> str:
    """Bucketed, not cropped: kohya sorts every picture into the aspect bucket
    nearest its own shape at ~512², so a 9:16 portrait trains as a portrait.
    kentskooking outpaints to square with Flux for the same reason — a centre
    crop throws away the composition the style lives in."""
    return (
        "[general]\n"
        'caption_extension = ".txt"\n'
        "\n[[datasets]]\n"
        f"resolution = {RESOLUTION}\n"
        f"batch_size = {BATCH_SIZE}\n"
        "enable_bucket = true\n"
        "min_bucket_reso = 256\n"
        "max_bucket_reso = 1024\n"
        "\n  [[datasets.subsets]]\n"
        f'  image_dir = "{_toml_path(dataset_dir)}"\n'
        "  num_repeats = 1\n"
    )


def sample_prompts(token: str) -> str:
    """kohya's prompt-file syntax, one preview per line and snapshot. The token
    is expanded to all of its vectors by the trainer itself."""
    return "".join(
        f"{text}, {token} --n blurry, lowres --w 512 --h 512 --d {seed} --l 7 --s 20\n"
        for text, seed in SAMPLE_PROMPTS
    )


def build_command(
    python: Path, script: Path, checkpoint: Path, plan: TrainingPlan,
    dataset_config: Path, prompts_file: Path, output_dir: Path,
) -> list[str]:
    template_flag = "--use_style_template" if plan.template == "style" else "--use_object_template"
    return [
        str(python), str(script),
        "--pretrained_model_name_or_path", str(checkpoint),
        "--dataset_config", str(dataset_config),
        "--output_dir", str(output_dir),
        "--output_name", plan.name,
        "--save_model_as", "safetensors",
        "--token_string", plan.token,
        "--init_word", plan.init_word,
        "--num_vectors_per_token", str(plan.vectors),
        template_flag,
        "--max_train_steps", str(plan.steps),
        "--save_every_n_steps", str(SAVE_EVERY),
        "--learning_rate", f"{plan.learning_rate:g}",
        "--optimizer_type", "AdamW",
        "--mixed_precision", "fp16",
        "--cache_latents",
        "--sdpa",
        # Workers would re-import torch per process on Windows for a dataset
        # that fits in memory ten times over.
        "--max_data_loader_n_workers", "0",
        "--sample_prompts", str(prompts_file),
        "--sample_every_n_steps", str(SAVE_EVERY),
        # The "before" row: the init word's own picture, which is what every
        # snapshot is read against.
        "--sample_at_first",
        "--sample_sampler", "euler_a",
        "--seed", "1234",
    ]


def trainer_env(gpu: str) -> dict[str, str]:
    """Environment overrides for the trainer process.

    PYTHONUTF8 is not optional: sd-scripts prints Japanese log lines, and with
    stdout redirected on a German Windows the console codec is cp1252 — the
    first `accelerator.print` raises UnicodeEncodeError and the run dies before
    step one (it did, on the first attempt).
    """
    return {
        "PYTHONUTF8": "1",
        "PYTHONIOENCODING": "utf-8",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": gpu,
    }


# ── What comes back ──────────────────────────────────────────────────────────
# The outer bar only: "steps:  12%|█▏   | 240/2000 [00:40<04:52, 6.02it/s, loss=0.123]".
# The preview sampler draws its own inner bars without the "steps:" prefix.
_PROGRESS = re.compile(
    r"steps:\s+\d+%\|[^|]*\|\s*(\d+)/(\d+)\s*\[[^\]]*?(?:loss=([\d.]+))?\]"
)


def parse_progress(text: str) -> tuple[int, int, float | None] | None:
    """The last training-step reading in a chunk of trainer output."""
    found = None
    for m in _PROGRESS.finditer(text):
        found = (int(m.group(1)), int(m.group(2)),
                 float(m.group(3)) if m.group(3) else None)
    return found


# <name>_000250_01_20260928203350_1234.png — and the "before" row, which
# sample_at_first writes with an epoch-style suffix: <name>_e000000_00_…png.
_SAMPLE = re.compile(r"^(?P<name>.+)_(?:e?)(?P<step>\d{6})_(?P<idx>\d{2})_\d{14}(?:_\d+)?\.png$")
_SNAPSHOT = re.compile(r"^(?P<name>.+)-step(?P<step>\d{8})\.safetensors$")


def list_snapshots(folder: Path, name: str) -> list[dict]:
    """Every snapshot and the preview images drawn at its step, oldest first.

    Step 0 has previews and no file (it is the untrained init word); the final
    save has a file and shares its previews with the last step.
    """
    by_step: dict[int, dict] = {}
    if folder.is_dir():
        for path in folder.glob("*.safetensors"):
            m = _SNAPSHOT.match(path.name)
            if m and m.group("name") == name:
                step = int(m.group("step"))
                by_step.setdefault(step, {"step": step, "samples": []})["file"] = path.name
        for path in sorted((folder / "sample").glob("*.png")):
            m = _SAMPLE.match(path.name)
            if m and m.group("name") == name:
                step = int(m.group("step"))
                entry = by_step.setdefault(step, {"step": step, "samples": []})
                entry["samples"].append(path.name)
    out = []
    for step in sorted(by_step):
        entry = by_step[step]
        entry.setdefault("file", None)
        entry["samples"].sort()
        out.append(entry)
    return out


def is_sample_filename(filename: str) -> bool:
    """Guards the preview endpoint: only names this trainer writes, no paths."""
    return bool(_SAMPLE.match(filename)) and "/" not in filename and "\\" not in filename
