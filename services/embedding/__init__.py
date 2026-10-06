"""Textual-inversion embeddings trained from gallery pictures.

`plan` decides everything that can be decided without running (names, bounds,
the trainer's command line and files, reading its output back); `runner`
prepares the pictures and drives kohya sd-scripts in its own venv.
"""
from services.embedding.plan import (
    LR_DEFAULT,
    LR_MAX,
    LR_MIN,
    MAX_IMAGES,
    MIN_IMAGES,
    SAMPLE_PROMPTS,
    SAVE_EVERY,
    STEPS_DEFAULT,
    STEPS_MAX,
    STEPS_MIN,
    TEMPLATES,
    VECTORS_DEFAULT,
    VECTORS_MAX,
    VECTORS_MIN,
    PlanError,
    TrainingPlan,
    active_file,
    active_name,
    estimate_seconds,
    is_sample_filename,
    is_workshop_name,
    list_snapshots,
    normalize_name,
    plan_training,
    snapshot_name,
    train_dir,
)
from services.embedding.runner import TrainingError, prepare_dataset, run_training

__all__ = [
    "LR_DEFAULT", "LR_MAX", "LR_MIN", "MAX_IMAGES", "MIN_IMAGES", "SAMPLE_PROMPTS",
    "SAVE_EVERY", "STEPS_DEFAULT", "STEPS_MAX", "STEPS_MIN", "TEMPLATES",
    "VECTORS_DEFAULT", "VECTORS_MAX", "VECTORS_MIN", "PlanError", "TrainingPlan",
    "active_file", "active_name", "estimate_seconds", "is_sample_filename",
    "is_workshop_name", "list_snapshots", "normalize_name", "plan_training",
    "snapshot_name", "train_dir", "TrainingError", "prepare_dataset", "run_training",
]
