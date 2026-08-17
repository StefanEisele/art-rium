"""
Unit tests for the Z-Image + Ultimate SD Upscale still-image pass
(services/image/upscale.py) — pure builder and math, no ComfyUI.
"""
import pytest

from services.image.upscale import (
    DENOISE_DEFAULT,
    DENOISE_MAX,
    DENOISE_MIN,
    MAX_OUTPUT_PIXELS,
    SCALE_DEFAULT,
    TILE_SIZE,
    UPSCALE_MODELS,
    build_image_upscale_workflow,
    clamp_denoise,
    clamp_scale,
    estimate_seconds,
    output_size,
    tile_count,
    upscaled_rel_path,
)


class TestClampDenoise:
    """The range exists to keep tiles agreeing with each other: past ~0.5 each
    one reimagines its own contents and the result is a collage."""

    def test_default_when_unset(self):
        assert clamp_denoise(None) == DENOISE_DEFAULT

    def test_default_sits_inside_the_range(self):
        assert DENOISE_MIN <= DENOISE_DEFAULT <= DENOISE_MAX

    def test_creative_end_is_capped_short_of_hallucination(self):
        assert clamp_denoise(0.95) == DENOISE_MAX
        assert DENOISE_MAX < 0.5

    def test_consistent_end_is_floored_above_a_no_op(self):
        assert clamp_denoise(0.0) == DENOISE_MIN

    def test_a_value_in_range_is_kept(self):
        assert clamp_denoise(0.33) == 0.33


class TestClampScale:
    def test_default_when_unset(self):
        assert clamp_scale(None, 1024, 1024) == SCALE_DEFAULT

    def test_a_normal_request_is_honoured(self):
        assert clamp_scale(2.0, 1080, 1920) == 2.0

    def test_oversized_output_is_reduced_not_refused(self):
        # 4x on 1080x1920 would be 33 MP; whatever comes back must fit the cap.
        scale = clamp_scale(4.0, 1080, 1920)
        w, h = output_size(1080, 1920, scale)
        assert w * h <= MAX_OUTPUT_PIXELS

    def test_a_huge_source_is_clamped_hard(self):
        scale = clamp_scale(4.0, 4000, 4000)
        w, h = output_size(4000, 4000, scale)
        assert w * h <= MAX_OUTPUT_PIXELS
        assert scale < 2.0

    def test_unknown_source_size_falls_back_to_the_request(self):
        assert clamp_scale(3.0, 0, 0) == 3.0


class TestTiling:
    def test_a_2x_portrait_render_tiles_as_expected(self):
        # 1080x1920 at 2x = 2160x3840 → 3 cols x 4 rows at 1024².
        w, h = output_size(1080, 1920, 2.0)
        assert (w, h) == (2160, 3840)
        assert tile_count(w, h) == 12

    def test_a_source_smaller_than_one_tile_is_still_one_tile(self):
        assert tile_count(400, 400) == 1

    def test_estimate_grows_with_area_not_with_the_factor(self):
        # Twice the linear scale is four times the tiles. Not four times the
        # wait, because the cold model load is paid once either way — but the
        # estimate has to move with the tile count, not with the factor.
        one = estimate_seconds(1024, 1024, 2.0)   # 4 tiles
        two = estimate_seconds(1024, 1024, 4.0)   # 16 tiles
        assert two >= one * 2.5

    def test_estimate_matches_the_measured_run(self):
        # 1024² at 1.5x → 1536², 4 tiles, measured 101 s end to end on the
        # 4060 Ti. Within 20% is close enough to set an expectation with.
        assert abs(estimate_seconds(1024, 1024, 1.5) - 101) <= 20


class TestWorkflowBuilder:
    def _wf(self, **kw):
        wf, save = build_image_upscale_workflow(
            "src.png", prompt="a rusted blue shoreline",
            denoise=kw.pop("denoise", 0.25), scale=kw.pop("scale", 2.0), **kw,
        )
        return wf, save, wf["iu_usdu"]["inputs"]

    def test_save_node_is_returned_and_wired(self):
        wf, save, _ = self._wf()
        assert save == "iu_save"
        assert wf[save]["inputs"]["images"] == ["iu_usdu", 0]

    def test_the_sampler_matches_the_generation_workflow(self):
        # A distilled model steered differently is a different model.
        _, _, u = self._wf()
        assert u["cfg"] == 1.0
        assert u["sampler_name"] == "res_multistep"
        assert u["scheduler"] == "simple"
        assert u["steps"] == 9

    def test_steps_are_not_scaled_down_by_denoise(self):
        # ComfyUI samples the full `steps` at any denoise (KSampler.set_steps
        # truncates the schedule, it does not shorten the run) — so a low
        # denoise must NOT be compensated for here.
        _, _, low = self._wf(denoise=0.10)
        _, _, high = self._wf(denoise=0.45)
        assert low["steps"] == high["steps"] == 9

    def test_denoise_is_clamped_inside_the_builder(self):
        _, _, u = self._wf(denoise=9.0)
        assert u["denoise"] == DENOISE_MAX

    def test_negative_is_a_zeroed_copy_of_the_positive(self):
        wf, _, u = self._wf()
        assert wf["iu_neg"]["class_type"] == "ConditioningZeroOut"
        assert wf["iu_neg"]["inputs"]["conditioning"] == ["iu_pos", 0]
        assert u["negative"] == ["iu_neg", 0]

    def test_prompt_conditions_the_tiles(self):
        wf, _, _ = self._wf()
        assert wf["iu_pos"]["inputs"]["text"] == "a rusted blue shoreline"

    def test_tiles_use_the_models_native_canvas(self):
        _, _, u = self._wf()
        assert u["tile_width"] == u["tile_height"] == TILE_SIZE == 1024

    def test_seam_fix_is_off_so_its_full_denoise_cannot_apply(self):
        _, _, u = self._wf()
        assert u["seam_fix_mode"] == "None"

    @pytest.mark.parametrize("key,filename", sorted(UPSCALE_MODELS.items()))
    def test_both_upscale_models_can_be_selected(self, key, filename):
        wf, _, _ = self._wf(upscale_model=key)
        assert wf["iu_upmod"]["inputs"]["model_name"] == filename

    def test_an_unknown_model_falls_back_to_the_default(self):
        wf, _, _ = self._wf(upscale_model="nope")
        assert wf["iu_upmod"]["inputs"]["model_name"] == UPSCALE_MODELS["realesrgan"]

    def test_seed_is_honoured_and_otherwise_random(self):
        _, _, fixed = self._wf(seed=1234)
        assert fixed["seed"] == 1234
        _, _, a = self._wf(seed=None)
        assert 0 <= a["seed"] < 2**32

    def test_the_source_image_is_the_uploaded_name(self):
        wf, _, u = self._wf()
        assert wf["iu_load"]["inputs"]["image"] == "src.png"
        assert u["image"] == ["iu_load", 0]

    def test_every_node_reference_points_at_a_node_that_exists(self):
        wf, _, _ = self._wf()
        for node_id, node in wf.items():
            for value in node["inputs"].values():
                if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                    assert value[0] in wf, f"{node_id} references missing node {value[0]}"


class TestUpscaledRelPath:
    def test_it_is_a_sibling_of_the_original(self):
        rel, name = upscaled_rel_path("images/2026/08/art_0042.png")
        assert rel == "images/2026/08/art_0042_upscaled.png"
        assert name == "art_0042_upscaled.png"

    def test_rerunning_replaces_rather_than_litters(self):
        assert upscaled_rel_path("a/b.png") == upscaled_rel_path("a/b.png")

    def test_name_stays_in_the_share_endpoint_charset(self):
        import re
        _, name = upscaled_rel_path("images/2026/08/art_0042.png")
        assert re.match(r"^[A-Za-z0-9._-]+$", name)
