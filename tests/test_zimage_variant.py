"""
Unit tests for the Z-Image Turbo variant workflow (services/comfy/zimage.py)
— pure builder and range logic, no ComfyUI.

The interesting property is that the variant graph must stay the *same* graph
as generation apart from where the latent comes from: a variant sampled unlike
its source would not read as a variant of it.
"""
import pytest

from routers.generate import _prompt_was_changed
from services.comfy.zimage import (
    DENOISE_DEFAULT,
    DENOISE_DEFAULT_EDITED,
    DENOISE_MAX,
    DENOISE_MIN,
    EDITED_SWEET_MAX,
    EDITED_SWEET_MIN,
    KEPT_SWEET_MAX,
    KEPT_SWEET_MIN,
    build_zimage_variant_workflow,
    build_zimage_workflow,
    clamp_denoise,
    denoise_bands,
    describe_denoise,
    latent_size,
    recommended_range,
)

LORA = [{"name": "a.safetensors", "strength": 0.4}]


def variant(**kw):
    args = {"image_name": "src.png", "prompt": "a cat", "seed": 7,
            "denoise": 0.4, "loras": []}
    args.update(kw)
    return build_zimage_variant_workflow(
        args["image_name"], args["prompt"], args["seed"],
        args["denoise"], args["loras"],
    )


class TestGraphShape:
    def test_the_latent_comes_from_the_source_image(self):
        wf = variant()
        assert wf["44"]["inputs"]["latent_image"] == ["48", 0]
        assert wf["48"]["class_type"] == "VAEEncode"
        assert wf["48"]["inputs"]["pixels"] == ["49", 0]
        assert wf["49"]["inputs"]["image"] == "src.png"

    def test_no_empty_latent_survives_from_the_generation_template(self):
        # Node 41 is EmptySD3LatentImage in the generation graph. If it were
        # still here the source would be silently ignored.
        assert "41" not in variant()

    def test_sampler_matches_generation_except_for_denoise(self):
        gen = build_zimage_workflow("a cat", 7, 1024, 1024, [])["44"]["inputs"]
        var = variant()["44"]["inputs"]
        for key in ("steps", "cfg", "sampler_name", "scheduler"):
            assert var[key] == gen[key], key
        assert gen["denoise"] == 1
        assert var["denoise"] == 0.4

    def test_model_stack_matches_generation(self):
        gen = build_zimage_workflow("a cat", 7, 1024, 1024, [])
        var = variant()
        for node in ("39", "40", "46", "47"):
            assert var[node]["class_type"] == gen[node]["class_type"]
            assert var[node]["inputs"] == gen[node]["inputs"]

    def test_negative_is_the_zeroed_positive(self):
        wf = variant()
        assert wf["42"]["class_type"] == "ConditioningZeroOut"
        assert wf["42"]["inputs"]["conditioning"] == ["45", 0]
        assert wf["44"]["inputs"]["negative"] == ["42", 0]

    def test_prompt_and_seed_land_where_they_belong(self):
        wf = variant(prompt="a dog on a roof", seed=99)
        assert wf["45"]["inputs"]["text"] == "a dog on a roof"
        assert wf["44"]["inputs"]["seed"] == 99

    def test_negative_seed_is_randomised(self):
        seeds = {variant(seed=-1)["44"]["inputs"]["seed"] for _ in range(5)}
        assert len(seeds) > 1
        assert all(0 <= s < 2**32 for s in seeds)

    def test_denoise_is_clamped_on_the_way_into_the_graph(self):
        assert variant(denoise=5.0)["44"]["inputs"]["denoise"] == DENOISE_MAX
        assert variant(denoise=0.0)["44"]["inputs"]["denoise"] == DENOISE_MIN


class TestLoraChain:
    """Both builders share `_apply_lora_chain`; the variant must wire it the
    same way or a variant of a LoRA'd picture comes back without the LoRA."""

    def test_no_lora_wires_the_unet_straight_through(self):
        wf = variant(loras=[])
        assert wf["47"]["inputs"]["model"] == ["46", 0]
        assert not [k for k in wf if k.startswith("lora_")]

    def test_one_lora_sits_between_unet_and_sampling(self):
        wf = variant(loras=LORA)
        assert wf["lora_0"]["inputs"]["model"] == ["46", 0]
        assert wf["lora_0"]["inputs"]["lora_name"] == "a.safetensors"
        assert wf["lora_0"]["inputs"]["strength_model"] == 0.4
        assert wf["47"]["inputs"]["model"] == ["lora_0", 0]

    def test_several_loras_chain_in_list_order(self):
        loras = [
            {"name": "a.safetensors", "strength": 0.3},
            {"name": "b.safetensors", "strength": 0.5},
            {"name": "c.safetensors", "strength": 0.7},
        ]
        wf = variant(loras=loras)
        assert wf["lora_0"]["inputs"]["model"] == ["46", 0]
        assert wf["lora_1"]["inputs"]["model"] == ["lora_0", 0]
        assert wf["lora_2"]["inputs"]["model"] == ["lora_1", 0]
        assert wf["47"]["inputs"]["model"] == ["lora_2", 0]

    def test_strength_is_clamped(self):
        assert variant(loras=[{"name": "a", "strength": 9}])["lora_0"]["inputs"]["strength_model"] == 1.0
        assert variant(loras=[{"name": "a", "strength": -3}])["lora_0"]["inputs"]["strength_model"] == 0.0

    def test_the_two_builders_produce_the_same_chain(self):
        var = variant(loras=LORA)
        gen = build_zimage_workflow("a cat", 7, 1024, 1024, LORA)
        assert var["lora_0"] == gen["lora_0"]

    def test_the_template_is_not_mutated_between_builds(self):
        variant(loras=LORA)
        assert variant(loras=[])["47"]["inputs"]["model"] == ["46", 0]


class TestDenoiseRange:
    def test_defaults_sit_inside_their_own_recommended_band(self):
        assert KEPT_SWEET_MIN <= DENOISE_DEFAULT <= KEPT_SWEET_MAX
        assert EDITED_SWEET_MIN <= DENOISE_DEFAULT_EDITED <= EDITED_SWEET_MAX

    def test_an_edited_prompt_needs_more_denoise_than_a_kept_one(self):
        # Measured: at 0.35 an edited prompt does not appear at all, at 0.75 it
        # lands. The two bands must not overlap or the advice means nothing.
        assert KEPT_SWEET_MAX < EDITED_SWEET_MIN
        assert DENOISE_DEFAULT < DENOISE_DEFAULT_EDITED

    def test_both_bands_fit_inside_the_slider(self):
        for low, high in (recommended_range(), recommended_range(prompt_changed=True)):
            assert DENOISE_MIN <= low < high <= DENOISE_MAX

    def test_default_follows_what_happened_to_the_prompt(self):
        assert clamp_denoise(None) == DENOISE_DEFAULT
        assert clamp_denoise(None, prompt_changed=True) == DENOISE_DEFAULT_EDITED

    def test_a_value_in_range_is_kept(self):
        assert clamp_denoise(0.42) == 0.42

    def test_out_of_range_is_pulled_in_rather_than_refused(self):
        assert clamp_denoise(-1) == DENOISE_MIN
        assert clamp_denoise(2) == DENOISE_MAX


class TestDescribeDenoise:
    def test_every_slider_position_gets_a_band(self):
        v = DENOISE_MIN
        while v <= DENOISE_MAX + 1e-9:
            assert describe_denoise(round(v, 2))["band"]
            v += 0.01

    def test_the_same_value_can_be_recommended_or_not_by_context(self):
        assert describe_denoise(0.40)["recommended"] is True
        assert describe_denoise(0.40, prompt_changed=True)["recommended"] is False
        assert describe_denoise(0.72, prompt_changed=True)["recommended"] is True

    def test_source_retained_falls_as_denoise_rises(self):
        kept = [describe_denoise(d)["source_retained"] for d in (0.15, 0.40, 0.70, 0.90)]
        assert kept == sorted(kept, reverse=True)

    def test_source_retained_matches_the_auraflow_shift(self):
        # sigma = 3d/(1+2d) with shift 3; what is left of the source is 1-sigma.
        for d in (0.15, 0.40, 0.70):
            assert describe_denoise(d)["source_retained"] == pytest.approx(
                1 - (3 * d / (1 + 2 * d)), abs=0.005,
            )

    def test_bands_tile_the_range_without_gaps(self):
        bands = denoise_bands()
        assert bands[0]["from"] == DENOISE_MIN
        assert bands[-1]["to"] == DENOISE_MAX
        for a, b in zip(bands, bands[1:]):
            assert a["to"] == b["from"]


class TestLatentSize:
    """VAEEncode center-crops to a multiple of 8 (VAE.vae_encode_crop_pixels),
    so the DB row has to record what comes back, not what went in."""

    def test_a_clean_size_is_untouched(self):
        assert latent_size(1024, 1024) == (1024, 1024)
        assert latent_size(1080, 1920) == (1080, 1920)

    def test_an_odd_size_is_rounded_down(self):
        assert latent_size(1620, 2880) == (1616, 2880)
        assert latent_size(1085, 1921) == (1080, 1920)

    def test_a_tiny_size_never_collapses_to_zero(self):
        assert latent_size(3, 3) == (8, 8)


class TestPromptChangeDetection:
    """Which band to recommend hinges on this, so it must not be fooled by the
    whitespace a textarea round-trip adds or eats."""

    def test_identical_prompt_is_not_a_change(self):
        assert _prompt_was_changed(["a cat"], "a cat") is False

    def test_whitespace_only_difference_is_not_a_change(self):
        assert _prompt_was_changed(["  a   cat\n"], "a cat") is False

    def test_real_edit_is_a_change(self):
        assert _prompt_was_changed(["a dog"], "a cat") is True

    def test_any_changed_entry_in_a_batch_counts(self):
        assert _prompt_was_changed(["a cat", "a dog"], "a cat") is True

    def test_a_source_without_a_prompt_makes_any_text_a_change(self):
        assert _prompt_was_changed(["a cat"], None) is True
        assert _prompt_was_changed([""], None) is False
