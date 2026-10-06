"""Run one embedding training: prepare the pictures, drive sd-scripts, report.

Same shape as services/segment/runner.py — a worker in another venv, the card
freed first, the child killed if the job is cancelled — with one difference in
how progress arrives. kohya reports through a tqdm bar, which redraws with
carriage returns and never ends a line until the run is over, so stderr is read
in chunks and scanned rather than iterated by line.
"""
from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Awaitable, Callable
from pathlib import Path

from PIL import Image as PILImage

from core.config import settings
from core.subproc import kill
from services.embedding.plan import (
    DATASET_MAX_EDGE,
    TRAIN_VRAM,
    TrainingPlan,
    build_command,
    dataset_toml,
    parse_progress,
    sample_prompts,
    trainer_env,
)

logger = logging.getLogger(__name__)

ProgressFn = Callable[[int, int, float | None], Awaitable[None] | None]
# The whole log is kept for the error message; the tail is what gets shown.
_LOG_KEEP = 200_000


class TrainingError(RuntimeError):
    """The trainer failed, with the reason it gave."""


def prepare_dataset(sources: list[Path], dest: Path) -> int:
    """Copy the pictures into a flat folder, downsized, as RGB PNG.

    Blocking — call through `asyncio.to_thread`. Downsizing is only about disk
    and load time; the trainer buckets everything to ~512² itself.
    """
    dest.mkdir(parents=True, exist_ok=True)
    for old in dest.glob("*"):
        old.unlink()
    count = 0
    for i, src in enumerate(sources):
        with PILImage.open(src) as im:
            im = im.convert("RGB")
            scale = min(1.0, DATASET_MAX_EDGE / max(im.size))
            if scale < 1.0:
                im = im.resize((round(im.width * scale), round(im.height * scale)),
                               PILImage.LANCZOS)
            im.save(dest / f"{i:03d}.png")
            count += 1
    return count


def write_configs(work_dir: Path, dataset_dir: Path, plan: TrainingPlan) -> tuple[Path, Path]:
    dataset_config = work_dir / "dataset.toml"
    dataset_config.write_text(dataset_toml(dataset_dir), encoding="utf-8")
    prompts = work_dir / "prompts.txt"
    prompts.write_text(sample_prompts(plan.token), encoding="utf-8")
    return dataset_config, prompts


async def _free_the_card() -> None:
    """Best-effort, as for segmentation: training needs no ComfyUI, so ComfyUI
    being down is no reason to refuse."""
    try:
        from services.comfy.vram import free_vram_for
        await free_vram_for(TRAIN_VRAM, "Embedding-Training")
    except Exception as exc:                        # noqa: BLE001
        logger.info("VRAM pre-flight skipped for embedding training: %s", exc)


async def run_training(
    plan: TrainingPlan,
    work_dir: Path,
    output_dir: Path,
    on_progress: ProgressFn | None = None,
    timeout: float = 4 * 3600.0,
) -> None:
    """Train one embedding into `output_dir`. Raises TrainingError on failure."""
    dataset_config, prompts = write_configs(work_dir, work_dir / "dataset", plan)
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_command(
        settings.sd_scripts_python,
        settings.sd_scripts_dir / "train_textual_inversion.py",
        settings.embedding_base_checkpoint,
        plan, dataset_config, prompts, output_dir,
    )
    await _free_the_card()
    logger.info("Training embedding %s: %d pictures, %d vectors, %d steps",
                plan.name, plan.image_count, plan.vectors, plan.steps)

    log_path = work_dir / "train.log"
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        # One stream: kohya logs to stdout and draws its bar on stderr, and the
        # interleaving is what makes the log readable after a failure.
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(settings.sd_scripts_dir),
        env={**os.environ, **trainer_env(settings.embedding_train_gpu)},
    )

    log = bytearray()
    last_step = -1

    async def pump() -> None:
        nonlocal last_step
        assert proc.stdout is not None
        tail = ""
        while True:
            chunk = await proc.stdout.read(4096)
            if not chunk:
                break
            log.extend(chunk)
            if len(log) > _LOG_KEEP:
                del log[: len(log) - _LOG_KEEP]
            # Keep a little of the previous chunk: a bar split across two reads
            # would otherwise never match.
            text = tail + chunk.decode("utf-8", errors="replace")
            tail = text[-300:]
            reading = parse_progress(text)
            if reading and reading[0] != last_step and on_progress:
                last_step = reading[0]
                out = on_progress(*reading)
                if asyncio.iscoroutine(out):
                    await out

    try:
        await asyncio.wait_for(asyncio.gather(pump(), proc.wait()), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        raise TrainingError(
            f"Training hat das Zeitlimit von {timeout / 3600:.0f} h überschritten"
        ) from None
    except asyncio.CancelledError:
        # The trainer holds the 4060 Ti for the whole run; a cancelled job
        # must not leave it training into a folder about to be deleted.
        kill(proc)
        raise
    finally:
        log_path.write_bytes(bytes(log))

    if proc.returncode != 0:
        text = bytes(log).decode("utf-8", errors="replace").replace("\r", "\n")
        lines = [ln for ln in text.splitlines() if ln.strip()]
        # The last exception line is nearly always the one that says why.
        reason = next((ln for ln in reversed(lines) if "Error" in ln), None)
        raise TrainingError(reason or "\n".join(lines[-4:])
                            or f"Training endete mit Code {proc.returncode}")
