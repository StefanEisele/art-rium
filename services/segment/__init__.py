"""Automatic colour-ID masks for the AnimateLCM region path.

`plan` holds the region palette and the frame budget; `runner` drives the
SAM 3 worker that lives in `scripts/sam3_segment.py`.
"""
from services.segment.plan import (
    DURATIONS,
    FRAME_CEILING,
    MAX_REGIONS,
    REGION_COLORS,
    REGION_LABELS,
    STRIDES,
    FrameBudget,
    budget_dict,
    plan_frames,
    trim_filters,
)
from services.segment.runner import SegmentError, build_spec, run_segmentation

__all__ = [
    "DURATIONS", "FRAME_CEILING", "MAX_REGIONS", "REGION_COLORS", "REGION_LABELS",
    "STRIDES", "FrameBudget", "budget_dict", "plan_frames", "trim_filters",
    "SegmentError", "build_spec", "run_segmentation",
]
