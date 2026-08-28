"""
The video tool shows the expert split next to the Steps chips, and it reads it
from a small lookup table baked into the page rather than from an endpoint.
That is a deliberate trade — a hint does not deserve a round trip — but a stale
hint is worse than no hint, so the table is pinned here against the function
that actually builds the graph.

Also checks the chip sets themselves stay inside what the builder accepts, so
the UI can never offer a step count the server will silently clamp away.
"""
import ast
import re
from pathlib import Path

import pytest

from core.config import settings
from routers.video import (
    _WAN_SEC_PER_STEP_PF_SAGE,
    _WAN_SEC_POST_PF,
    _WAN_STEPS_MAX,
    _WAN_STEPS_MIN,
    clamp_lora_high,
    clamp_wan_steps,
    estimate_wan_seconds,
)
from services.comfy.wan_moe import I2V_BOUNDARY, moe_split_step

_PAGE = Path(__file__).resolve().parents[1] / "frontends" / "tools" / "video" / "index.html"
_WAN_SHIFT = 5.0  # what routers/video.py feeds ModelSamplingSD3


def _js_literal(name: str) -> str:
    """The right-hand side of `const <name> = …;` in the video tool's script."""
    page = _PAGE.read_text(encoding="utf-8")
    m = re.search(rf"^const {re.escape(name)}\s*=\s*(.+?);$", page, re.M | re.S)
    assert m, f"{name} not found in {_PAGE.name}"
    return m.group(1)


@pytest.fixture(scope="module")
def split_table() -> dict[int, int]:
    # JS object keys are bare numbers; quoting them makes the literal parseable.
    raw = re.sub(r"(\d+)\s*:", r"'\1':", _js_literal("WAN_MOE_SPLIT"))
    return {int(k): v for k, v in ast.literal_eval(raw).items()}


@pytest.fixture(scope="module")
def step_options() -> list[int]:
    return ast.literal_eval(_js_literal("WAN_STEP_OPTIONS"))


@pytest.fixture(scope="module")
def motion_options() -> list[list]:
    return ast.literal_eval(_js_literal("WAN_MOTION_OPTIONS"))


def test_split_table_matches_the_builder(split_table):
    for steps, high in split_table.items():
        assert high == moe_split_step(steps, _WAN_SHIFT, I2V_BOUNDARY), (
            f"the page claims {high} high-noise steps at {steps}, "
            f"the graph builds {moe_split_step(steps, _WAN_SHIFT, I2V_BOUNDARY)}"
        )


def test_every_offered_step_count_has_a_split(split_table, step_options):
    assert set(step_options) == set(split_table)


def test_offered_step_counts_survive_the_clamp(step_options):
    for steps in step_options:
        assert clamp_wan_steps(steps) == steps
        assert _WAN_STEPS_MIN <= steps <= _WAN_STEPS_MAX


def test_offered_motion_values_survive_the_clamp(motion_options):
    for strength, label in motion_options:
        assert clamp_lora_high(strength) == strength
        assert label


def test_motion_chips_read_from_most_movement_to_least(motion_options):
    strengths = [s for s, _ in motion_options]
    assert strengths == sorted(strengths)
    assert strengths[0] == 0.0 and strengths[-1] == 1.0


# The page fetches /api/video/sampler-info on load and prefers the server's
# numbers. These constants are what it renders with until that lands, and if it
# never does — so they still have to be the shipped configuration's numbers,
# not some earlier fit nobody updated.
@pytest.mark.parametrize("js_name,py_value", [
    ("WAN_SEC_PER_STEP_PF", _WAN_SEC_PER_STEP_PF_SAGE),
    ("WAN_SEC_POST_PF", _WAN_SEC_POST_PF),
])
def test_fallback_coefficients_match_the_shipped_default(js_name, py_value):
    assert float(_js_literal(js_name)) == py_value


def test_the_fallback_prices_a_worked_example_the_way_the_server_does():
    page_seconds = (
        960 * 960 * 49
        * (float(_js_literal("WAN_SEC_PER_STEP_PF")) * 6
           + float(_js_literal("WAN_SEC_POST_PF")))
    )
    assert round(page_seconds) == estimate_wan_seconds(960, 960, 49, 6, sage=True)


def test_the_shipped_default_is_the_one_the_fallback_assumes():
    # The fallback hard-codes the SageAttention figure. If the project ever
    # ships with SDPA as the default, that line has to move with it.
    assert settings.wan_sage_attention is True
