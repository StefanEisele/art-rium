"""
Unit tests for the still-image upscale (services/image/upscale.py) — pure
builder and math, no ComfyUI.

The settings under test were measured, not chosen (see the module docstring
for the numbers). The tests pin the decisions those measurements forced, so
that a later tidy-up cannot quietly undo one:

  * euler, not res_multistep, for the redraw — res_multistep blotched flat
    areas (mottle 1.34 → 0.99 with euler).
  * the Kreativität slider is the redraw's START NOISE from a shift-1
    schedule, not a denoise fraction under AuraFlow shift 3 — under shift 3
    "0.25" was 50 % noise and "0.35" repainted cloth folds into a padlock.
  * SeedVR2 is the default enlarger and is not an ESRGAN file, so anything
    that validates models against the ESRGAN list silently drops it.
  * SeedVR2 in one pass up to ~10 MP out, tiled above — one pass OOMed at
    33 MP on the 16 GB card.
"""
import pytest

from services.image.upscale import (
    DEFAULT_UPSCALE_MODEL,
    DENOISE_DEFAULT,
    DENOISE_MAX,
    DENOISE_MIN,
    ENLARGERS,
    MAX_OUTPUT_PIXELS,
    SCALE_DEFAULT,
    TILE_SIZE,
    UPSCALE_MODELS,
    _SVR_SINGLE_PASS_MAX_PIXELS,
    _SVR_TILE_OUT,
    build_image_upscale_workflow,
    clamp_denoise,
    clamp_scale,
    denoise_bands,
    describe_denoise,
    estimate_seconds,
    output_size,
    resolve_enlarger,
    tile_count,
    timing,
    upscaled_rel_path,
)

SRC = {"src_width": 1080, "src_height": 1920}


def build(**kw):
    kw.setdefault("denoise", 0.25)
    kw.setdefault("scale", 2.0)
    kw.setdefault("upscale_model", "realesrgan")
    for k, v in SRC.items():
        kw.setdefault(k, v)
    return build_image_upscale_workflow("src.png", prompt="a rusted blue shoreline", **kw)


def classes(wf):
    return {n["class_type"] for n in wf.values()}


class TestEnlargers:
    def test_seedvr2_is_the_default(self):
        assert DEFAULT_UPSCALE_MODEL == "seedvr2"

    def test_seedvr2_is_not_an_esrgan_file(self):
        # The trap: validating a request against UPSCALE_MODELS would quietly
        # swap every SeedVR2 request for the fallback.
        assert "seedvr2" not in UPSCALE_MODELS
        assert resolve_enlarger("seedvr2") == "seedvr2"

    def test_every_esrgan_entry_names_its_file(self):
        for key, spec in ENLARGERS.items():
            if key != "seedvr2":
                assert UPSCALE_MODELS[key] == spec["file"]

    def test_unknown_keys_fall_back_to_the_default(self):
        assert resolve_enlarger("nope") == DEFAULT_UPSCALE_MODEL
        assert resolve_enlarger(None) == DEFAULT_UPSCALE_MODEL

    def test_seedvr2_defaults_to_no_redraw_and_esrgan_to_a_light_one(self):
        # SeedVR2 alone measured best at 2x; an ESRGAN file alone has painted
        # nothing and is either plastic or grainy.
        assert ENLARGERS["seedvr2"]["default_denoise"] == 0.0
        assert ENLARGERS["realesrgan"]["default_denoise"] > 0
        assert ENLARGERS["nomos8ksc"]["default_denoise"] > 0

    def test_every_label_and_hint_is_present(self):
        for spec in ENLARGERS.values():
            assert spec["label"] and spec["hint"]


class TestClampDenoise:
    def test_none_takes_the_enlargers_own_default(self):
        assert clamp_denoise(None, "seedvr2") == 0.0
        assert clamp_denoise(None, "realesrgan") == ENLARGERS["realesrgan"]["default_denoise"]
        assert clamp_denoise(None) == DENOISE_DEFAULT

    def test_zero_is_a_real_choice(self):
        # 0 means "enlarge only", not "use the minimum redraw".
        assert DENOISE_MIN == 0.0
        assert clamp_denoise(0.0, "realesrgan") == 0.0

    def test_the_creative_end_is_capped(self):
        assert clamp_denoise(0.95) == DENOISE_MAX
        # Start noise, not denoise: 0.6 already rereads shapes.
        assert DENOISE_MAX <= 0.6

    def test_negative_values_are_floored(self):
        assert clamp_denoise(-1.0) == 0.0

    def test_a_value_in_range_is_kept(self):
        assert clamp_denoise(0.33) == 0.33


class TestBands:
    def test_bands_tile_the_range(self):
        b = denoise_bands()
        assert b[0]["from"] == DENOISE_MIN
        assert b[-1]["to"] == DENOISE_MAX
        for x, y in zip(b, b[1:]):
            assert x["to"] == y["from"]

    def test_zero_is_named_as_enlarge_only(self):
        assert describe_denoise(0.0)["band"] == "Nur vergrößern"
        assert describe_denoise(0.2)["band"] != "Nur vergrößern"


class TestClampScale:
    def test_default_when_unset(self):
        assert clamp_scale(None, 1024, 1024) == SCALE_DEFAULT

    def test_a_normal_request_is_honoured(self):
        assert clamp_scale(2.0, 1080, 1920) == 2.0

    def test_oversized_output_is_reduced_not_refused(self):
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


class TestTimingAndTiles:
    def test_a_2x_portrait_render_tiles_as_expected(self):
        w, h = output_size(1080, 1920, 2.0)
        assert (w, h) == (2160, 3840)
        assert tile_count(w, h) == 12

    def test_a_source_smaller_than_one_tile_is_still_one_tile(self):
        assert tile_count(400, 400) == 1

    def test_the_redraw_matches_the_measured_runs(self):
        # 1024² → 1536² (4 tiles) 101 s; 2160x3840 (12 tiles) ~214 s.
        assert abs(timing(1024, 1024, 1.5)["redraw"] - 101) <= 20
        assert abs(timing(1080, 1920, 2.0)["redraw"] - 214) <= 30

    def test_seedvr2_alone_matches_the_measured_runs(self):
        # 8.3 MP one pass: 40 s. 33 MP tiled: 312 s.
        assert abs(estimate_seconds(1080, 1920, 2.0, model="seedvr2", denoise=0) - 40) <= 12
        assert abs(estimate_seconds(1080, 1920, 4.0, model="seedvr2", denoise=0) - 312) <= 60

    def test_the_slider_is_what_moves_the_cost(self):
        fast = estimate_seconds(1080, 1920, 2.0, model="seedvr2", denoise=0)
        slow = estimate_seconds(1080, 1920, 2.0, model="seedvr2", denoise=0.3)
        assert slow > 4 * fast

    def test_esrgan_is_not_paid_twice(self):
        # With a redraw, USDU runs the ESRGAN pass itself.
        t = timing(1080, 1920, 2.0)
        assert estimate_seconds(1080, 1920, 2.0, model="realesrgan", denoise=0.3) == t["redraw"]


class TestRedraw:
    """The Z-Image stage — present whenever Kreativität > 0."""

    def test_it_is_the_custom_sample_node_with_explicit_sampler_and_sigmas(self):
        wf, _ = build()
        u = wf["iu_usdu"]
        assert u["class_type"] == "UltimateSDUpscaleCustomSample"
        assert u["inputs"]["custom_sampler"] == ["iu_sampler", 0]
        assert u["inputs"]["custom_sigmas"] == ["iu_sigmas", 0]

    def test_euler_not_res_multistep(self):
        wf, _ = build()
        assert wf["iu_sampler"]["inputs"]["sampler_name"] == "euler"

    def test_the_slider_is_the_start_noise_of_a_shift_1_schedule(self):
        wf, _ = build(denoise=0.4)
        sig = wf["iu_sigmas"]["inputs"]
        assert sig["denoise"] == 0.4
        assert sig["model"] == ["iu_sshift", 0]
        assert wf["iu_sshift"]["inputs"]["shift"] == 1.0

    def test_the_sampling_model_stays_as_generated(self):
        # Shift 3 on the model the tiles are sampled with, exactly as the
        # generation workflow — the shift-1 node only feeds the scheduler.
        wf, _ = build()
        assert wf["iu_usdu"]["inputs"]["model"] == ["iu_shift", 0]
        assert wf["iu_shift"]["inputs"]["shift"] == 3

    def test_full_steps_at_any_creativity(self):
        low, _ = build(denoise=0.1)
        high, _ = build(denoise=0.6)
        assert low["iu_sigmas"]["inputs"]["steps"] == high["iu_sigmas"]["inputs"]["steps"] == 9

    def test_denoise_is_clamped_inside_the_builder(self):
        wf, _ = build(denoise=9.0)
        assert wf["iu_sigmas"]["inputs"]["denoise"] == DENOISE_MAX

    def test_negative_is_a_zeroed_copy_of_the_positive(self):
        wf, _ = build()
        assert wf["iu_neg"]["class_type"] == "ConditioningZeroOut"
        assert wf["iu_neg"]["inputs"]["conditioning"] == ["iu_pos", 0]
        assert wf["iu_usdu"]["inputs"]["negative"] == ["iu_neg", 0]

    def test_prompt_conditions_the_tiles(self):
        wf, _ = build()
        assert wf["iu_pos"]["inputs"]["text"] == "a rusted blue shoreline"

    def test_tiles_use_the_models_native_canvas(self):
        wf, _ = build()
        u = wf["iu_usdu"]["inputs"]
        assert u["tile_width"] == u["tile_height"] == TILE_SIZE == 1024

    def test_seam_fix_is_off_so_its_full_denoise_cannot_apply(self):
        wf, _ = build()
        assert wf["iu_usdu"]["inputs"]["seam_fix_mode"] == "None"

    def test_seed_is_honoured_and_otherwise_random(self):
        fixed, _ = build(seed=1234)
        assert fixed["iu_usdu"]["inputs"]["seed"] == 1234
        rnd, _ = build(seed=None)
        assert 0 <= rnd["iu_usdu"]["inputs"]["seed"] < 2**32

    @pytest.mark.parametrize("key", ["realesrgan", "nomos8ksc"])
    def test_esrgan_enlarges_inside_usdu(self, key):
        wf, _ = build(upscale_model=key)
        u = wf["iu_usdu"]["inputs"]
        assert wf["iu_upmod"]["inputs"]["model_name"] == UPSCALE_MODELS[key]
        assert u["upscale_model"] == ["iu_upmod", 0]
        assert u["image"] == ["iu_load", 0]
        assert u["upscale_by"] == 2.0

    def test_seedvr2_enlarges_first_and_usdu_only_redraws(self):
        wf, _ = build(upscale_model="seedvr2", denoise=0.3)
        u = wf["iu_usdu"]["inputs"]
        assert u["image"] == ["iu_svr", 0]
        assert u["upscale_by"] == 1.0
        assert "upscale_model" not in u
        assert "iu_upmod" not in wf


class TestEnlargeOnly:
    """Kreativität 0: nothing is redrawn, so Z-Image must not even load."""

    @pytest.mark.parametrize("key", ["seedvr2", "realesrgan", "nomos8ksc"])
    def test_no_z_image_at_zero(self, key):
        wf, save = build(upscale_model=key, denoise=0.0)
        assert not classes(wf) & {"UNETLoader", "UltimateSDUpscaleCustomSample",
                                  "CLIPLoader", "BasicScheduler"}
        assert wf[save]["class_type"] == "SaveImage"

    def test_esrgan_alone_lands_on_the_asked_for_factor(self):
        # The files are 4x; a 2x request is the 4x output halved.
        wf, save = build(upscale_model="realesrgan", denoise=0.0, scale=2.0)
        assert wf["iu_fit"]["inputs"]["scale_by"] == 0.5
        assert wf[save]["inputs"]["images"] == ["iu_fit", 0]

    def test_seedvr2_alone_saves_its_own_output(self):
        wf, save = build(upscale_model="seedvr2", denoise=0.0)
        assert wf[save]["inputs"]["images"] == ["iu_svr", 0]


class TestSeedVR2Sizing:
    def test_one_pass_while_it_fits(self):
        wf, _ = build(upscale_model="seedvr2", denoise=0.0, scale=2.0)
        n = wf["iu_svr"]
        assert n["class_type"] == "SeedVR2VideoUpscaler"
        # Sized by the SHORT side.
        assert n["inputs"]["resolution"] == 2160
        assert n["inputs"]["batch_size"] == 1

    def test_tiled_above_the_measured_budget(self):
        wf, _ = build(upscale_model="seedvr2", denoise=0.0, scale=4.0)
        n = wf["iu_svr"]
        assert n["class_type"] == "SeedVR2TilingUpscaler"
        assert n["inputs"]["new_resolution"] == 4320
        assert n["inputs"]["resolution_target"] == "shortest"

    def test_a_tile_is_never_upscaled_short_and_stretched(self):
        # The node only CAPS each tile at tile_upscale_resolution; a cap below
        # tile × factor would soften every tile.
        for scale in (3.0, 4.0):
            wf, _ = build(upscale_model="seedvr2", denoise=0.0, scale=scale)
            i = wf["iu_svr"]["inputs"]
            assert i["tile_width"] * scale <= i["tile_upscale_resolution"] == _SVR_TILE_OUT

    def test_the_switch_point_is_the_pixel_budget(self):
        w, h = output_size(1080, 1920, 2.0)
        assert w * h <= _SVR_SINGLE_PASS_MAX_PIXELS
        w, h = output_size(1080, 1920, 3.0)
        assert w * h > _SVR_SINGLE_PASS_MAX_PIXELS

    def test_seedvr2_needs_the_source_size(self):
        with pytest.raises(ValueError):
            build_image_upscale_workflow("src.png", prompt="", denoise=0,
                                         scale=2.0, upscale_model="seedvr2")

    def test_the_card_is_freed_for_the_redraw(self):
        wf, _ = build(upscale_model="seedvr2", denoise=0.3)
        assert wf["iu_svr_dit"]["inputs"]["cache_model"] is False
        assert wf["iu_svr_vae"]["inputs"]["cache_model"] is False


class TestGraphIntegrity:
    @pytest.mark.parametrize("model", ["seedvr2", "realesrgan", "nomos8ksc"])
    @pytest.mark.parametrize("denoise", [0.0, 0.3])
    @pytest.mark.parametrize("scale", [2.0, 4.0])
    def test_every_node_reference_points_at_a_node_that_exists(self, model, denoise, scale):
        wf, save = build(upscale_model=model, denoise=denoise, scale=scale)
        assert save in wf
        for node_id, node in wf.items():
            for value in node["inputs"].values():
                if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                    assert value[0] in wf, f"{node_id} references missing node {value[0]}"

    def test_the_source_image_is_the_uploaded_name(self):
        wf, _ = build()
        assert wf["iu_load"]["inputs"]["image"] == "src.png"


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
