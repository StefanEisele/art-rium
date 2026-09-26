"""Detail Daemon on the Z-Image Turbo graphs (services/comfy/zimage.py).

The dial itself is a judgement call, but two things about it are not, and both
are cheap to pin here rather than to rediscover on the GPU:

  * **The sampler chain has to be wired the way Detail Daemon needs it.** The
    node takes a SAMPLER and returns one, which a plain KSampler never
    produces — so the graph is the decomposition KSampler performs internally,
    and if any of those five nodes loses its wire the submission is rejected
    by ComfyUI rather than quietly sampled differently.

  * **Nothing about the sampling may change when the dial is not touched.**
    The decomposition was measured to be pixel-identical to the KSampler at
    detail 0 (see the module note), which only holds while steps, cfg,
    sampler_name, scheduler and denoise are carried across untouched. These
    tests are what keeps that true.
"""
import pytest

from services.comfy.zimage import (
    DETAIL_DEFAULT,
    DETAIL_MAX,
    DETAIL_MIN,
    DETAIL_SWEET_MAX,
    DETAIL_SWEET_MIN,
    _DETAIL_SCHEDULE,
    build_zimage_variant_workflow,
    build_zimage_workflow,
    clamp_detail,
    describe_detail,
    detail_bands,
)


def gen(detail=None, **kw):
    args = {"prompt": "a cat", "seed": 7, "width": 1024, "height": 1024, "loras": []}
    args.update(kw)
    return build_zimage_workflow(
        args["prompt"], args["seed"], args["width"], args["height"],
        args["loras"], detail,
    )


def var(detail=None, denoise=0.4):
    return build_zimage_variant_workflow("src.png", "a cat", 7, denoise, [], detail)


class TestSamplerChain:
    def test_the_ksampler_is_gone(self):
        # Left behind, it would render a second picture nobody asked for and
        # burn a full sampling pass doing it.
        assert "44" not in gen()
        assert "44" not in var()
        assert not [n for n in gen().values() if n["class_type"] == "KSampler"]

    @pytest.mark.parametrize("builder", [gen, var])
    def test_every_chain_node_is_present_and_wired(self, builder):
        wf = builder()
        assert wf["50"]["class_type"] == "RandomNoise"
        assert wf["51"]["class_type"] == "CFGGuider"
        assert wf["52"]["class_type"] == "BasicScheduler"
        assert wf["53"]["class_type"] == "KSamplerSelect"
        assert wf["54"]["class_type"] == "DetailDaemonSamplerNode"
        assert wf["55"]["class_type"] == "SamplerCustomAdvanced"

        assert wf["54"]["inputs"]["sampler"] == ["53", 0]
        s = wf["55"]["inputs"]
        assert s["noise"] == ["50", 0]
        assert s["guider"] == ["51", 0]
        assert s["sampler"] == ["54", 0]
        assert s["sigmas"] == ["52", 0]

    @pytest.mark.parametrize("builder", [gen, var])
    def test_the_decode_reads_the_plain_output(self, builder):
        # Slot 1 is `denoised_output` and is a different picture; the KSampler
        # this replaced returned the equivalent of slot 0.
        assert builder()["43"]["inputs"]["samples"] == ["55", 0]

    def test_the_guider_and_scheduler_see_the_model_the_loras_end_at(self):
        wf = gen(loras=[{"name": "a.safetensors", "strength": 0.4}])
        assert wf["47"]["inputs"]["model"] == ["lora_0", 0]
        # Both read ModelSamplingAuraFlow, which is where the LoRA chain ends.
        assert wf["51"]["inputs"]["model"] == ["47", 0]
        assert wf["52"]["inputs"]["model"] == ["47", 0]

    def test_sampling_settings_survive_the_decomposition(self):
        # The measured pixel-identity at detail 0 depends on every one of
        # these coming across from the template untouched.
        wf = gen()
        assert wf["52"]["inputs"]["steps"] == 9
        assert wf["52"]["inputs"]["scheduler"] == "simple"
        assert wf["52"]["inputs"]["denoise"] == 1
        assert wf["51"]["inputs"]["cfg"] == 1
        assert wf["53"]["inputs"]["sampler_name"] == "res_multistep"

    def test_the_variant_keeps_its_denoise_and_its_source_latent(self):
        wf = var(denoise=0.55)
        assert wf["52"]["inputs"]["denoise"] == 0.55
        assert wf["55"]["inputs"]["latent_image"] == ["48", 0]
        assert wf["48"]["class_type"] == "VAEEncode"


class TestDial:
    def test_untouched_means_zero(self):
        # Not merely "small": 0 is the value the chain was measured to be
        # pixel-identical to the old KSampler at.
        assert gen(None)["54"]["inputs"]["detail_amount"] == 0.0
        assert gen()["54"]["inputs"]["detail_amount"] == DETAIL_DEFAULT

    def test_the_value_reaches_the_node(self):
        assert gen(1.25)["54"]["inputs"]["detail_amount"] == 1.25
        assert var(-0.5)["54"]["inputs"]["detail_amount"] == -0.5

    def test_out_of_range_is_clamped_rather_than_rejected(self):
        assert gen(99)["54"]["inputs"]["detail_amount"] == DETAIL_MAX
        assert gen(-99)["54"]["inputs"]["detail_amount"] == DETAIL_MIN

    def test_the_schedule_is_the_z_image_tuned_one(self):
        # The node's own defaults (start 0.2, end 0.8, exponent 1) are for
        # many-step SDXL/Flux runs; this model gets nine steps.
        inputs = gen(1.0)["54"]["inputs"]
        for key, value in _DETAIL_SCHEDULE.items():
            assert inputs[key] == value, key
        assert inputs["exponent"] == 3.0
        assert inputs["end"] == 1.0

    def test_cfg_override_stays_off(self):
        # 0 lets the wrapper read the CFG off the guider. At cfg 1 that gives
        # a scale factor of 1.0, so the adjustment applies undiminished —
        # the dial is not inert on this distilled model.
        assert gen(1.0)["54"]["inputs"]["cfg_scale_override"] == 0.0


class TestClampAndDescribe:
    def test_clamp_rounds_and_bounds(self):
        assert clamp_detail(None) == DETAIL_DEFAULT
        assert clamp_detail(0.123456) == 0.12
        assert clamp_detail(DETAIL_MAX + 10) == DETAIL_MAX
        assert clamp_detail(DETAIL_MIN - 10) == DETAIL_MIN

    def test_describe_names_a_band_everywhere_in_range(self):
        v = DETAIL_MIN
        while v <= DETAIL_MAX + 1e-9:
            d = describe_detail(v)
            assert d["band"], v
            assert d["effect"], v
            v = round(v + 0.05, 2)

    def test_recommended_matches_the_sweet_band(self):
        assert describe_detail(DETAIL_SWEET_MIN)["recommended"] is True
        assert describe_detail(DETAIL_SWEET_MAX)["recommended"] is True
        assert describe_detail(DETAIL_MIN)["recommended"] is False
        assert describe_detail(DETAIL_MAX)["recommended"] is False

    def test_zero_is_described_as_unchanged_rather_than_as_an_effect(self):
        assert describe_detail(0)["band"] == "Wie gehabt"

    def test_bands_tile_the_whole_range_without_gaps(self):
        bands = detail_bands()
        assert bands[0]["from"] == DETAIL_MIN
        assert bands[-1]["to"] == DETAIL_MAX
        for a, b in zip(bands, bands[1:]):
            assert a["to"] == b["from"], (a, b)


class TestTemplateIsolation:
    def test_the_template_is_not_consumed_by_the_first_build(self):
        # _apply_detail_sampler pops node 44; on a shared template that would
        # work exactly once and then raise KeyError for the rest of the
        # process's life.
        first = gen(1.0)
        second = gen(0.0)
        assert first["54"]["inputs"]["detail_amount"] == 1.0
        assert second["54"]["inputs"]["detail_amount"] == 0.0
        assert second["52"]["inputs"]["steps"] == 9

    def test_variant_and_generation_templates_stay_independent(self):
        gen(2.0)
        assert var()["55"]["inputs"]["latent_image"] == ["48", 0]
        assert "41" not in var()
