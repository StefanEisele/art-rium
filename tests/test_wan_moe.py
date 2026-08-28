"""
Wan 2.2's expert handover — the sigma arithmetic behind services/comfy/wan_moe.py.

Two things are being pinned here. The first is agreement with ComfyUI: the sigma
schedule this module predicts has to be the one comfy/samplers.py::simple_scheduler
will actually build, or the split is computed for a curve the sampler never walks.
The second is agreement with the reference implementation of the technique
(stduhpf/ComfyUI-WanMoeKSampler), whose published splits for euler/simple at
shift 5 are the fixtures below.
"""
import pytest

from services.comfy.wan_moe import (
    I2V_BOUNDARY,
    T2V_BOUNDARY,
    flow_sigmas,
    moe_split_step,
    time_snr_shift,
)


class TestTimeSnrShift:
    def test_shift_one_is_the_identity(self):
        for t in (0.0, 0.25, 0.5, 1.0):
            assert time_snr_shift(1.0, t) == t

    def test_endpoints_are_fixed_under_any_shift(self):
        # The schedule always starts fully noised and ends fully clean; shift
        # only bends what happens between.
        for shift in (1.0, 5.0, 8.0, 12.0):
            assert time_snr_shift(shift, 0.0) == 0.0
            assert time_snr_shift(shift, 1.0) == 1.0

    def test_shift_pushes_the_middle_toward_noise(self):
        # This is why a higher shift moves the boundary crossing later: every
        # interior timestep is mapped to a *higher* sigma.
        assert time_snr_shift(8.0, 0.5) > time_snr_shift(5.0, 0.5) > 0.5


class TestFlowSigmas:
    def test_starts_at_one_and_ends_at_zero(self):
        sigmas = flow_sigmas(6, 5.0)
        assert len(sigmas) == 7
        assert sigmas[0] == pytest.approx(1.0)
        assert sigmas[-1] == 0.0

    def test_monotonically_descending(self):
        for steps in (4, 6, 8, 10, 16, 20):
            sigmas = flow_sigmas(steps, 5.0)
            assert all(a > b for a, b in zip(sigmas, sigmas[1:]))

    def test_matches_comfyui_simple_scheduler_at_shift_five(self):
        # Hand-computed from comfy/samplers.py::simple_scheduler over
        # ModelSamplingDiscreteFlow's 1000-entry table, integer truncation and
        # all: t = (1000 - int(x * 1000/steps)) / 1000, sigma = 5t/(1+4t).
        assert flow_sigmas(4, 5.0) == pytest.approx(
            [1.0, 0.9375, 0.833333, 0.625, 0.0], abs=1e-5
        )

    def test_truncation_is_reproduced_not_rounded(self):
        # 6 steps lands on t=0.834, not 1/1.2 — int() floors the stride index.
        # Rounding instead would move sigma by ~4e-4, which is enough to flip a
        # split that sits close to the boundary.
        assert flow_sigmas(6, 5.0)[1] == pytest.approx(time_snr_shift(5.0, 0.834))


class TestMoeSplitStep:
    # The reference splits for euler/simple at shift 5.0, i2v boundary 0.900.
    @pytest.mark.parametrize("steps,expected_high", [
        (4, 1), (6, 2), (8, 2), (10, 3), (12, 4), (16, 5), (20, 7),
    ])
    def test_reference_splits_at_shift_five(self, steps, expected_high):
        assert moe_split_step(steps, 5.0, I2V_BOUNDARY) == expected_high

    def test_never_the_naive_half_above_eight_steps(self):
        # The bug this module exists to fix: steps//2 hands the high-noise
        # expert far more of the schedule than the model was trained to give
        # it, and the high-noise expert is the one that decides motion.
        for steps in (10, 12, 16, 20):
            assert moe_split_step(steps, 5.0, I2V_BOUNDARY) < steps // 2

    def test_both_experts_always_get_work(self):
        for steps in range(1, 33):
            split = moe_split_step(steps, 5.0, I2V_BOUNDARY)
            assert 1 <= split <= max(1, steps - 1)

    def test_higher_shift_gives_the_high_noise_expert_more(self):
        # Shift raises every interior sigma, so the boundary is crossed later.
        assert moe_split_step(20, 8.0, I2V_BOUNDARY) > moe_split_step(20, 5.0, I2V_BOUNDARY)

    def test_t2v_boundary_switches_no_earlier_than_i2v(self):
        # The schedule descends, so it passes 0.900 before it reaches 0.875:
        # the lower (t2v) boundary keeps the high-noise expert around longer.
        for steps in (4, 6, 8, 10, 16, 20):
            assert (moe_split_step(steps, 5.0, T2V_BOUNDARY)
                    >= moe_split_step(steps, 5.0, I2V_BOUNDARY))

    def test_split_grows_with_steps(self):
        # Non-strictly: the boundary sits at a fixed timestep, so its step index
        # can only move later as the schedule is sampled more finely.
        splits = [moe_split_step(s, 5.0, I2V_BOUNDARY) for s in range(4, 25)]
        assert all(a <= b for a, b in zip(splits, splits[1:]))
