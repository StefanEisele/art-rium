"""The spec handed across the interpreter boundary to the SAM 3 worker.

The worker runs in ComfyUI's venv, so nothing type-checks between here and
there and a renamed key fails at runtime, minutes into a job. These pin the
shape.
"""
import json
from pathlib import Path

from core.config import settings
from services.segment import build_spec, plan_frames


def _concepts():
    return [
        {"text": "Tomate", "color": [255, 0, 0]},
        {"text": "Hand", "color": [0, 255, 0]},
    ]


def test_spec_carries_the_paths_and_the_model():
    spec = build_spec(
        Path("in.mp4"), Path("out.mp4"), Path("prev.mp4"), _concepts(),
    )
    assert spec["source"] == "in.mp4"
    assert spec["dest"] == "out.mp4"
    assert spec["preview"] == "prev.mp4"
    assert spec["model_dir"] == str(settings.sam3_model_dir)
    assert spec["device"] == settings.sam3_device
    assert spec["concepts"] == _concepts()


def test_a_spec_without_a_preview_omits_the_key_entirely():
    """The worker treats a missing `preview` as "do not write one", so it must
    be absent rather than null."""
    spec = build_spec(Path("in.mp4"), Path("out.mp4"), None, _concepts())
    assert "preview" not in spec


def test_a_budget_becomes_the_workers_trim_settings():
    budget = plan_frames(600, 30.0, seconds=6, stride=2)
    spec = build_spec(Path("in.mp4"), Path("out.mp4"), None, _concepts(), budget)
    assert spec["stride"] == 2
    assert spec["seconds"] == 6.0
    assert spec["max_frames"] == 90
    assert spec["fps"] == 15.0
    assert spec["start"] == 0.0


def test_no_budget_means_the_worker_reads_the_whole_clip():
    """The router trims before segmenting when it trims at all, so passing a
    budget here as well would compound the stride and thin the clip twice."""
    spec = build_spec(Path("in.mp4"), Path("out.mp4"), None, _concepts())
    for key in ("stride", "seconds", "max_frames", "start"):
        assert key not in spec


def test_the_spec_survives_a_json_round_trip():
    """It is written to a file and read by another interpreter, so everything
    in it has to be plain JSON — a Path or a dataclass would raise at dump."""
    budget = plan_frames(300, 30.0, seconds=3, stride=3)
    spec = build_spec(
        Path("a/in.mp4"), Path("b/out.mp4"), Path("c/p.mp4"), _concepts(),
        budget, score_threshold=0.4,
    )
    assert json.loads(json.dumps(spec)) == spec
    assert spec["score_threshold"] == 0.4
