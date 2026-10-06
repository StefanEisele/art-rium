"""
Unit tests for the AnimateLCM builder (services/comfy/animatelcm.py) — pure
graph construction, no ComfyUI.

What is worth pinning here is everything that makes this a *rebuild of the
user's own workflow* rather than a fresh attempt at one: the hires pass and its
two ControlNets taken from the base frames, the motion module still being in the
chain when that pass samples, and the handful of settings (beta schedule, noise
type, IP-Adapter scaling) that look like arbitrary values and are in fact the
reference file's.
"""
import pytest

from services.comfy.animatelcm import (
    AD_MOTION_LORA,
    BETA_SCHEDULE,
    CONTEXT_LENGTH,
    DEPTH_DEFAULT,
    HIRES_DENOISE_DEFAULT,
    IP_DEFAULT,
    IP_MAX,
    NOISE_TYPE,
    AnimateLcmRequest,
    Embedding,
    Injection,
    Region,
    build_animatelcm_base_workflow,
    base_curve,
    build_animatelcm_workflow,
    canvas_size,
    clamp_depth,
    clamp_hires,
    clamp_ip,
    curve_values,
    fade_points,
    injection_start,
    output_size,
    positive_chunks,
    snap_size,
    sweep_workflows,
)

DEPTH = "C:/fake/depth.mp4"
MASKS = "C:/fake/masks.mp4"


def simple(**kw) -> dict:
    args = {"control_video": DEPTH, "prompt": "a world", "reference_image": "ref.png"}
    args.update(kw)
    return build_animatelcm_workflow(AnimateLcmRequest(**args))[0]


def regional(n: int, **kw) -> dict:
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    args = {
        "control_video": DEPTH, "mask_video": MASKS, "prompt": "a world",
        "regions": [Region(colors[i], f"ref{i}.png") for i in range(n)],
    }
    args.update(kw)
    return build_animatelcm_workflow(AnimateLcmRequest(**args))[0]


def nodes_of(wf: dict, class_type: str) -> list[str]:
    return sorted(k for k, v in wf.items() if v["class_type"] == class_type)


class TestHiresPass:
    """The detail engine, and the reason the original workflow looks the way it
    does. A second diffusion pass at denoise 0.4 over 2x-upscaled frames,
    steered by ControlNets derived from the base render."""

    def test_it_is_on_by_default(self):
        wf = simple()
        assert len(nodes_of(wf, "KSampler")) == 2
        assert wf["al_ks2"]["inputs"]["denoise"] == HIRES_DENOISE_DEFAULT

    def test_it_refines_rather_than_repaints(self):
        # The whole difference to the VACE detour: the base picture is encoded
        # and partially renoised, not reconstructed from a control signal.
        wf = simple()
        assert wf["al_ks2"]["inputs"]["latent_image"] == ["al_enc", 0]
        assert wf["al_enc"]["inputs"]["pixels"] == ["al_down", 0]
        assert 0 < wf["al_ks2"]["inputs"]["denoise"] < 1.0

    def test_the_motion_module_is_still_in_the_chain(self):
        # This is what keeps the second pass from boiling: the model the hires
        # sampler gets is the AnimateDiff-wrapped one, so the motion module sees
        # the whole batch again rather than 94 independent frames.
        wf = simple()
        assert wf["al_ks2"]["inputs"]["model"] == wf["al_ks"]["inputs"]["model"]
        assert wf["al_ipa0"]["inputs"]["model"] == ["al_ipaload", 0]
        assert wf["al_ipaload"]["inputs"]["model"] == ["al_evolved", 0]

    def test_net_scale_is_two_after_a_four_times_model(self):
        wf = simple()
        assert wf["al_down"]["inputs"]["scale_by"] == 0.5
        assert wf["al_up"]["inputs"]["image"] == ["al_dec", 0]
        assert output_size("square", hires=True) == (1088, 1088)
        assert output_size("square", hires=False) == (544, 544)

    def test_both_guides_come_from_the_base_render(self):
        # Not from the control track: the hires pass must follow the picture it
        # is refining, not the geometry that produced it.
        wf = simple()
        assert wf["al_lineart"]["inputs"]["image"] == ["al_dec", 0]
        assert wf["al_midas"]["inputs"]["image"] == ["al_dec", 0]
        assert wf["al_cn2"]["inputs"]["image"] == ["al_lineart", 0]
        assert wf["al_cn3"]["inputs"]["image"] == ["al_midas", 0]

    def test_the_two_hires_controlnets_are_chained(self):
        wf = simple()
        assert wf["al_cn3"]["inputs"]["positive"] == ["al_cn2", 0]
        assert wf["al_cn3"]["inputs"]["negative"] == ["al_cn2", 1]
        assert wf["al_ks2"]["inputs"]["positive"] == ["al_cn3", 0]

    def test_hires_conditioning_bypasses_the_base_depth_controlnet(self):
        # The base pass's depth ControlNet described the Blender geometry; in
        # the hires pass that job belongs to MiDaS on the render itself.
        wf = simple()
        assert wf["al_cn2"]["inputs"]["positive"] == ["al_pos", 0]

    def test_it_can_be_switched_off(self):
        wf = simple(hires=False)
        assert len(nodes_of(wf, "KSampler")) == 1
        assert wf["al_out"]["inputs"]["images"] == ["al_dec", 0]

    def test_the_denoise_is_clamped_to_the_useful_window(self):
        assert clamp_hires(0.95) <= 0.75
        assert clamp_hires(0.01) >= 0.2
        assert clamp_hires(None) == HIRES_DENOISE_DEFAULT

    def test_the_default_is_the_measured_value_not_the_originals(self):
        # 0.60 beat the original's 0.40 on detail *and* on temporal stability
        # over 94 frames. Measured, not preferred.
        assert HIRES_DENOISE_DEFAULT == 0.60


class TestFaithfulToTheOriginal:
    """Values that look arbitrary and are not — they are the reference file's,
    and the reference file is the best result the user has had."""

    def test_the_original_beta_schedule_not_the_documented_one(self):
        # AnimateLCM's authors suggest lcm[100_ots]; the workflow that actually
        # produced the good results used sqrt_linear.
        assert simple()["al_evolved"]["inputs"]["beta_schedule"] == BETA_SCHEDULE
        assert "sqrt_linear" in BETA_SCHEDULE

    def test_freenoise_across_context_windows(self):
        assert simple()["al_settings"]["inputs"]["noise_type"] == NOISE_TYPE

    def test_lcm_karras_at_cfg_one(self):
        ks = simple()["al_ks"]["inputs"]
        assert (ks["sampler_name"], ks["scheduler"]) == ("lcm", "karras")
        assert ks["cfg"] == 1.0
        assert ks["steps"] == 9

    def test_the_motion_lora_and_scale(self):
        wf = simple()
        # `name`, not `lora_name`: this loader breaks the convention, and
        # getting it wrong fails at submit time rather than at build time.
        assert wf["al_mlora"]["inputs"]["name"] == AD_MOTION_LORA
        assert wf["al_mlora"]["inputs"]["strength"] == 0.45
        assert wf["al_mscale"]["inputs"]["float_val"] == 1.2
        assert wf["al_adapply"]["inputs"]["motion_lora"] == ["al_mlora", 0]

    def test_the_motion_lora_is_optional(self):
        wf = simple(motion_lora=None)
        assert "al_mlora" not in wf
        assert "motion_lora" not in wf["al_adapply"]["inputs"]

    def test_ipadapter_carries_composition_not_only_colour(self):
        # K+V rather than V only, and it lets go at 0.8 so the sampler can
        # settle the last fifth on its own.
        ipa = simple()["al_ipa0"]["inputs"]
        assert ipa["embeds_scaling"] == "K+V w/ C penalty"
        assert ipa["end_at"] == 0.8
        assert ipa["weight"] == 1.0

    def test_the_base_controlnet_lets_go_early(self):
        cn = simple()["al_cn"]["inputs"]
        assert cn["strength"] == DEPTH_DEFAULT == 0.45
        assert cn["end_percent"] == 0.7

    def test_the_negative_is_empty_because_cfg_one_ignores_it(self):
        wf = simple()
        assert wf["al_neg"]["inputs"]["text"] == ""
        assert wf["al_ks"]["inputs"]["cfg"] == 1.0

    def test_the_original_canvas_is_square(self):
        assert canvas_size("square") == (544, 544)
        # No probed source: "auto" falls back to the original's own canvas.
        assert simple()["al_latent"]["inputs"]["width"] == 544


class TestOnePass:
    """N regions still cost one base render — `attn_mask` on the adapter, which
    is what the three muted channel groups in the original file were doing the
    hard way."""

    def test_three_regions_one_base_sampler(self):
        wf = regional(3)
        assert len(nodes_of(wf, "IPAdapterAdvanced")) == 3
        assert nodes_of(wf, "KSampler") == ["al_ks", "al_ks2"]

    def test_the_adapters_chain(self):
        wf = regional(3)
        assert wf["al_ipa1"]["inputs"]["model"] == ["al_ipa0", 0]
        assert wf["al_ipa2"]["inputs"]["model"] == ["al_ipa1", 0]
        assert wf["al_ks"]["inputs"]["model"] == ["al_ipa2", 0]

    def test_each_region_has_its_own_mask_and_picture(self):
        wf = regional(3)
        for i in range(3):
            assert wf[f"al_ipa{i}"]["inputs"]["attn_mask"] == [f"al_mask{i}", 0]
            assert wf[f"al_ref{i}"]["inputs"]["image"] == f"ref{i}.png"

    def test_without_regions_the_adapter_is_unmasked(self):
        wf = simple()
        assert nodes_of(wf, "IPAdapterAdvanced") == ["al_ipa0"]
        assert "attn_mask" not in wf["al_ipa0"]["inputs"]


class TestInterpolation:
    def test_off_by_default(self):
        wf = simple()
        assert "al_rife" not in wf
        assert wf["al_out"]["inputs"]["frame_rate"] == 8

    def test_it_runs_after_the_hires_pass(self):
        # Before it, RIFE would multiply the frames the expensive sampler has
        # to redraw.
        wf = simple(rife=4)
        assert wf["al_rife"]["inputs"]["frames"] == ["al_dec2", 0]
        assert wf["al_out"]["inputs"]["images"] == ["al_rife", 0]

    def test_the_frame_rate_follows_the_multiplier(self):
        assert simple(rife=4)["al_out"]["inputs"]["frame_rate"] == 32
        assert simple(rife=4, fps=6)["al_out"]["inputs"]["frame_rate"] == 24


class TestControlTrack:
    def test_a_blender_depth_pass_is_inverted_by_default(self):
        assert "al_inv" in simple()

    def test_derived_depth_is_never_also_inverted(self):
        wf = simple(derive_depth=True, invert_depth=True)
        assert not any(v["class_type"] == "ImageInvert" for v in wf.values())

    def test_the_control_track_is_padded_rather_than_stretched(self):
        assert simple()["al_fit"]["inputs"]["method"] == "pad"
        assert simple(fit="stretch")["al_fit"]["inputs"]["method"] == "stretch"

    def test_the_mask_track_is_fitted_exactly_like_the_control_track(self):
        wf = regional(1, fit="crop")
        assert wf["al_maskfit"]["inputs"]["method"] == wf["al_fit"]["inputs"]["method"]
        assert wf["al_maskfit"]["inputs"]["interpolation"] == "nearest-exact"

    def test_the_source_frame_count_caps_the_length(self):
        assert simple(length=96, source_frames=50)["al_latent"]["inputs"]["batch_size"] == 50

    def test_the_track_is_not_resampled_by_default(self):
        assert simple()["al_ctrl"]["inputs"]["force_rate"] == 0.0

    def test_the_whole_clip_is_one_latent_batch(self):
        assert simple(length=64)["al_latent"]["inputs"]["batch_size"] == 64

    def test_context_windows_are_wired_to_the_motion_stack(self):
        wf = simple()
        assert wf["al_evolved"]["inputs"]["context_options"] == ["al_ctx", 0]
        assert wf["al_ctx"]["inputs"]["context_length"] == CONTEXT_LENGTH
        assert wf["al_ctx"]["inputs"]["closed_loop"] is True


class TestGeometry:
    def test_sizes_snap_to_the_latent_grid(self):
        assert snap_size(433) == 432
        assert snap_size(1) == 64

    def test_the_wide_preset_stays_under_sd15s_duplication_range(self):
        assert canvas_size("wide") == (768, 432)

    def test_an_unknown_aspect_falls_back_rather_than_raising(self):
        assert canvas_size("nope") == canvas_size("square")


class TestAutoCanvas:
    """The control track already carries the aspect the scene was framed in, so
    the canvas follows it. No flag, no padding, no crop."""

    def test_it_follows_the_source(self):
        assert canvas_size("auto", (1080, 1080)) == (544, 544)
        assert canvas_size("auto", (1920, 1080)) == (720, 408)
        assert canvas_size("auto", (1080, 1920)) == (408, 720)

    def test_every_aspect_costs_the_same(self):
        # Area is held, not an edge: SD 1.5 duplicates on pixel count, so a
        # constant budget is what makes portrait as safe as square.
        areas = [w * h for w, h in
                 (canvas_size("auto", s) for s in
                  ((1080, 1080), (1920, 1080), (1080, 1920), (1600, 1200)))]
        assert max(areas) - min(areas) < 0.05 * max(areas)

    def test_an_extreme_ratio_gives_up_area_rather_than_stretching(self):
        width, height = canvas_size("auto", (4096, 1300))
        assert max(width, height) <= 768
        assert width * height < 544 * 544

    def test_it_falls_back_to_the_original_square_without_a_source(self):
        # Guessing an aspect is worse than admitting there is none.
        assert canvas_size("auto", None) == (544, 544)
        assert canvas_size("auto", (None, None)) == (544, 544)

    def test_a_named_aspect_still_overrides_the_source(self):
        assert canvas_size("tall", (1920, 1080)) == (432, 768)

    def test_the_builder_uses_the_probed_source(self):
        wf = simple(aspect="auto", source_width=1920, source_height=1080)
        latent = wf["al_latent"]["inputs"]
        assert (latent["width"], latent["height"]) == (720, 408)

    def test_output_size_accounts_for_the_hires_pass(self):
        assert output_size("auto", True, source=(1920, 1080)) == (1440, 816)
        assert output_size("auto", False, source=(1920, 1080)) == (720, 408)

    def test_the_dials_are_clamped(self):
        assert clamp_ip(99) == IP_MAX
        assert clamp_depth(None) == DEPTH_DEFAULT
        assert clamp_ip(None) == IP_DEFAULT


class TestValidation:
    def test_regions_need_a_mask_video(self):
        with pytest.raises(ValueError, match="mask_video"):
            build_animatelcm_workflow(AnimateLcmRequest(
                control_video=DEPTH, prompt="x",
                regions=[Region((255, 0, 0), "r.png")],
            ))

    def test_a_region_needs_a_picture(self):
        with pytest.raises(ValueError, match="reference"):
            build_animatelcm_workflow(AnimateLcmRequest(
                control_video=DEPTH, mask_video=MASKS, prompt="x",
                regions=[Region((255, 0, 0), "")],
            ))

    def test_something_has_to_supply_the_material(self):
        with pytest.raises(ValueError, match="reference"):
            build_animatelcm_workflow(
                AnimateLcmRequest(control_video=DEPTH, prompt="  ")
            )

    def test_a_prompt_alone_is_a_valid_baseline(self):
        # The sweep's control row. The tool itself still demands a picture.
        wf = build_animatelcm_workflow(AnimateLcmRequest(control_video=DEPTH, prompt="x"))[0]
        assert "al_ipaload" not in wf


A, B = Embedding("style-a"), Embedding("style-b", 0.8)


def stacked(join: str, **kw) -> dict:
    return simple(embeddings=[A, B], embedding_join=join, **kw)


def walk_concat(wf: dict, node: str = "al_pos") -> list[str]:
    """The positive's texts in token order, read back off the concat chain."""
    entry = wf[node]
    if entry["class_type"] == "CLIPTextEncode":
        return [entry["inputs"]["text"]]
    inputs = entry["inputs"]
    return (walk_concat(wf, inputs["conditioning_to"][0])
            + walk_concat(wf, inputs["conditioning_from"][0]))


class TestEmbeddings:
    """kentskooking's joins, and the guarantee that a request without
    embeddings builds exactly what it always did."""

    def test_no_embeddings_is_the_plain_prompt(self):
        wf = simple()
        assert wf["al_pos"] == {"class_type": "CLIPTextEncode",
                                "inputs": {"text": "a world", "clip": ["al_ckpt", 1]}}
        assert not nodes_of(wf, "ConditioningConcat")

    def test_inline_writes_one_prompt(self):
        wf = stacked("inline")
        assert wf["al_pos"]["inputs"]["text"] == \
            "a world, embedding:style-a, (embedding:style-b:0.8)"
        assert not nodes_of(wf, "ConditioningConcat")

    def test_concat_encodes_each_on_its_own(self):
        wf = stacked("concat")
        assert walk_concat(wf) == ["a world", "embedding:style-a", "(embedding:style-b:0.8)"]
        assert len(nodes_of(wf, "ConditioningConcat")) == 2

    def test_the_sandwich_puts_a_blank_between_every_pair(self):
        wf = stacked("sandwich")
        assert walk_concat(wf) == ["a world", "", "embedding:style-a", "",
                                   "(embedding:style-b:0.8)"]

    def test_every_blank_is_one_node(self):
        wf = stacked("sandwich")
        blanks = [k for k, v in wf.items()
                  if v["class_type"] == "CLIPTextEncode" and v["inputs"]["text"] == ""
                  and k != "al_neg"]
        assert blanks == ["al_pos_blank"]

    def test_inline_is_the_default(self):
        # Measured: the only join that keeps the prompt's scene.
        assert positive_chunks(AnimateLcmRequest(
            control_video=DEPTH, prompt="x", embeddings=[A])) == ["x, embedding:style-a"]

    def test_an_empty_prompt_leaves_only_the_embeddings(self):
        # Kent's own renders: no text at all, the embeddings are the prompt.
        wf = stacked("sandwich", prompt="  ")
        assert walk_concat(wf) == ["embedding:style-a", "", "(embedding:style-b:0.8)"]

    def test_both_passes_read_the_stack(self):
        wf = stacked("sandwich")
        assert wf["al_cn"]["inputs"]["positive"] == ["al_pos", 0]
        assert wf["al_cn2"]["inputs"]["positive"] == ["al_pos", 0]
        assert wf["al_pos"]["class_type"] == "ConditioningConcat"

    def test_the_weight_is_clamped(self):
        req = AnimateLcmRequest(control_video=DEPTH, prompt="", embedding_join="inline",
                                embeddings=[Embedding("x", 9.0)])
        assert positive_chunks(req) == ["(embedding:x:2)"]

    def test_embeddings_alone_need_no_picture(self):
        wf = build_animatelcm_workflow(AnimateLcmRequest(
            control_video=DEPTH, prompt="a world", embeddings=[A]))[0]
        assert not nodes_of(wf, "IPAdapterAdvanced")
        assert "al_ipaload" not in wf
        assert wf["al_ks"]["inputs"]["model"] == ["al_evolved", 0]
        assert wf["al_ks2"]["inputs"]["model"] == ["al_evolved", 0]

    def test_the_base_only_graph_carries_the_stack_too(self):
        wf = build_animatelcm_base_workflow(AnimateLcmRequest(
            control_video=DEPTH, prompt="a world", embeddings=[A, B],
            embedding_join="concat"))[0]
        assert walk_concat(wf)[-1] == "(embedding:style-b:0.8)"

    @pytest.mark.parametrize("name", ["has space", "a:b", "(x)", "", "../up"])
    def test_names_comfy_would_silently_skip_are_refused(self, name):
        with pytest.raises(ValueError, match="embedding name"):
            simple(embeddings=[Embedding(name)])

    def test_a_subfolder_is_a_usable_name(self):
        assert "embedding:civitai/style-a" in \
            simple(embeddings=[Embedding("civitai/style-a")], embedding_join="inline")[
                "al_pos"]["inputs"]["text"]

    def test_an_unknown_join_is_refused(self):
        with pytest.raises(ValueError, match="embedding_join"):
            stacked("blend")


class TestCurves:
    """Embeddings that move through the clip: one masked layer per moving
    embedding, the prompt layer taking what the curves leave."""

    def moving(self, **kw):
        args = {"embeddings": [Embedding("held"), Embedding("mover", 0.9, curve="fade_in")],
                "length": 10, "reference_image": None}
        args.update(kw)
        return simple(**args)

    def test_without_curves_nothing_changes(self):
        wf = simple(embeddings=[A, B])
        assert not nodes_of(wf, "ConditioningSetMask")
        assert wf["al_pos"]["class_type"] == "CLIPTextEncode"

    def test_one_curve_is_two_layers(self):
        wf = self.moving()
        assert nodes_of(wf, "ConditioningSetMask") == ["al_pos_base_m", "al_pos_l0_m"]
        assert wf["al_pos"]["class_type"] == "ConditioningCombine"
        assert wf["al_pos"]["inputs"] == {"conditioning_1": ["al_pos_base_m", 0],
                                          "conditioning_2": ["al_pos_l0_m", 0]}

    def test_held_embeddings_are_in_every_layer(self):
        wf = self.moving()
        assert wf["al_pos_base"]["inputs"]["text"] == "a world, embedding:held"
        assert wf["al_pos_l0"]["inputs"]["text"] == \
            "a world, embedding:held, (embedding:mover:0.9)"

    def test_the_masks_are_the_curves_frame_by_frame(self):
        wf = self.moving()
        mask = wf["al_pos_l0_m_mask"]["inputs"]
        assert mask["frames"] == 10 and mask["interpolation"] == "linear"
        values = [float(p.split("(")[1].rstrip(")")) for p in mask["points_string"].split(",")]
        assert values == curve_values("fade_in", 10)
        base = [float(p.split("(")[1].rstrip(")"))
                for p in wf["al_pos_base_m_mask"]["inputs"]["points_string"].split(",")]
        assert base[0] == 1.0 and base[-1] == 0.001, "the base never reaches zero"

    def test_both_passes_read_the_layered_positive(self):
        wf = self.moving()
        assert wf["al_cn"]["inputs"]["positive"] == ["al_pos", 0]
        assert wf["al_cn2"]["inputs"]["positive"] == ["al_pos", 0]

    def test_three_moving_embeddings_chain(self):
        wf = simple(embeddings=[Embedding("a", curve="fade_out"), Embedding("b", curve="swell"),
                                Embedding("c", curve="pulse", cycles=3)], length=12)
        assert len(nodes_of(wf, "ConditioningSetMask")) == 4
        assert len(nodes_of(wf, "ConditioningCombine")) == 3

    def test_an_unknown_curve_is_refused(self):
        with pytest.raises(ValueError, match="curve"):
            simple(embeddings=[Embedding("a", curve="wobble")])

    def test_the_mask_follows_the_clamped_length(self):
        wf = self.moving(length=96, source_frames=40)
        assert wf["al_pos_l0_m_mask"]["inputs"]["frames"] == 40


class TestRegionEmbeddings:
    """kentskooking's RGB masks: each colour its own embedding stack, the
    global layers masked out of it."""

    def build(self, **kw):
        args = {
            "control_video": DEPTH, "mask_video": MASKS, "prompt": "a world",
            "regions": [Region((255, 0, 0), None, embeddings=[Embedding("rust", 1.1)]),
                        Region((0, 255, 0), "g.png")],
            "length": 16,
        }
        args.update(kw)
        return build_animatelcm_workflow(AnimateLcmRequest(**args))[0]

    def test_a_region_of_embeddings_needs_no_picture(self):
        wf = self.build()
        assert nodes_of(wf, "IPAdapterAdvanced") == ["al_ipa0"], "only the green region adapts"
        assert wf["al_ipa0"]["inputs"]["attn_mask"] == ["al_mask1", 0]
        assert wf["al_mask0"]["class_type"] == "ColorToMask", "the mask exists regardless"

    def test_the_region_layer_keys_on_its_mask(self):
        wf = self.build()
        assert wf["al_pos_r0"]["inputs"]["text"] == "a world, (embedding:rust:1.1)"
        assert wf["al_pos_r0_m"]["inputs"]["mask"] == ["al_mask0", 0]
        assert "al_pos_r1" not in wf, "a picture-only region adds no text layer"

    def test_the_global_layer_is_masked_out_of_the_region(self):
        wf = self.build()
        assert wf["al_pos_outside"] == {"class_type": "InvertMask",
                                        "inputs": {"mask": ["al_mask0", 0]}}
        assert wf["al_pos_base_m"]["inputs"]["mask"] == ["al_pos_outside", 0]
        assert wf["al_pos"]["inputs"] == {"conditioning_1": ["al_pos_base_m", 0],
                                          "conditioning_2": ["al_pos_r0_m", 0]}

    def test_two_embedding_regions_are_one_union(self):
        wf = self.build(regions=[Region((255, 0, 0), None, embeddings=[Embedding("a")]),
                                 Region((0, 0, 255), None, embeddings=[Embedding("b")])])
        assert wf["al_pos_union1"]["inputs"]["operation"] == "add"
        assert wf["al_pos_outside"]["inputs"]["mask"] == ["al_pos_union1", 0]
        assert not nodes_of(wf, "IPAdapterAdvanced")

    def test_curves_and_regions_compose(self):
        wf = self.build(embeddings=[Embedding("mover", curve="swell")], width=544, height=544)
        # The curve's mask is drawn at canvas size so it can be multiplied with
        # the region masks, which come off the fitted mask video.
        mask = wf["al_pos_l0_m_mask"]["inputs"]
        assert (mask["width"], mask["height"]) == (544, 544)
        out = wf["al_pos_l0_m_out"]["inputs"]
        assert out["operation"] == "multiply" and out["source"] == ["al_pos_outside", 0]
        assert len(nodes_of(wf, "ConditioningSetMask")) == 3

    def test_a_region_needs_something(self):
        with pytest.raises(ValueError, match="reference image or an embedding"):
            self.build(regions=[Region((255, 0, 0), None)])


class TestInjection:
    def build(self, **kw):
        inj = kw.pop("injection", Injection("C:/fake/static.mp4"))
        return simple(injection=inj, length=48, **kw)

    def test_it_hangs_off_the_sample_settings(self):
        wf = self.build()
        assert wf["al_settings"]["inputs"]["image_inject"] == ["al_inj", 0]
        inj = wf["al_inj"]["inputs"]
        assert inj["image"] == ["al_injfit", 0] and inj["vae"] == ["al_ckpt", 2]
        assert inj["strength_multival"] == ["al_injstr", 0]
        assert wf["al_injstr"]["inputs"]["float_val"] == 0.3

    def test_the_source_is_fitted_like_the_control_track(self):
        wf = self.build()
        assert wf["al_injfit"]["inputs"]["method"] == wf["al_fit"]["inputs"]["method"]
        assert wf["al_injvid"]["inputs"]["frame_load_cap"] == 48

    def test_a_short_source_is_looped_to_the_clip(self):
        wf = self.build(injection=Injection("C:/fake/static.mp4", source_frames=20))
        assert wf["al_injrep"]["inputs"]["amount"] == 3
        assert wf["al_injcut"]["inputs"] == {"image": ["al_injrep", 0], "batch_index": 0,
                                            "length": 48}
        assert wf["al_injfit"]["inputs"]["image"] == ["al_injcut", 0]

    def test_it_stays_out_of_the_hires_pass(self):
        # ADE skips an injection whose start lies before a sampler's range; the
        # refine starts at 1 - denoise.
        wf = self.build(injection=Injection("v.mp4", start=0.5), hires_denoise=0.6)
        assert wf["al_inj"]["inputs"]["start_percent"] == 0.35
        wf = self.build(injection=Injection("v.mp4", start=0.5), hires=False)
        assert wf["al_inj"]["inputs"]["start_percent"] == 0.5
        assert injection_start(Injection("v", start=0.3), True, 0.75) == 0.2

    def test_it_can_be_limited_to_a_region(self):
        wf = simple(mask_video=MASKS, length=48,
                    injection=Injection("v.mp4", color=(0, 0, 255)))
        assert wf["al_inj"]["inputs"]["mask_opt"] == ["al_injmask", 0]
        assert wf["al_injmask"]["inputs"]["blue"] == 255

    def test_a_region_injection_needs_the_mask_video(self):
        with pytest.raises(ValueError, match="mask_video"):
            simple(injection=Injection("v.mp4", color=(255, 0, 0)))

    def test_without_it_the_settings_are_untouched(self):
        assert "image_inject" not in simple()["al_settings"]["inputs"]


class TestCurveValues:
    def test_fades_run_end_to_end(self):
        assert curve_values("fade_in", 5) == [0.0, 0.146, 0.5, 0.854, 1.0]
        assert curve_values("fade_out", 5) == [1.0, 0.854, 0.5, 0.146, 0.0]

    def test_swell_and_pulse_close_the_loop(self):
        # closed_loop renders wrap: the frame after the last is the first.
        swell = curve_values("swell", 9)
        assert swell[0] == swell[-1] == 0.0 and max(swell) == 1.0
        pulse = curve_values("pulse", 16, cycles=2)
        assert pulse[0] == 0.0 and pulse[4] == 1.0 and pulse[8] == 0.0

    def test_cycles_are_clamped(self):
        assert curve_values("pulse", 32, cycles=99) == curve_values("pulse", 32, cycles=8)

    def test_the_base_takes_the_rest_and_never_zero(self):
        assert base_curve([[0.0, 0.5, 1.0], [0.0, 0.8, 0.5]], 3) == [1.0, 0.001, 0.001]
        assert base_curve([[0.25, 0.5]], 2) == [0.75, 0.5]

    def test_points_parse_the_way_kjnodes_reads_them(self):
        # mask_nodes.py::createfademask — split on ",", strip, frame:(value).
        text = fade_points([0.0, 0.5, 1.0])
        parsed = [(int(f.strip()), float(v.strip()[1:-1]))
                  for f, v in (pt.split(":") for pt in text.rstrip(",\n").split(","))]
        assert parsed == [(0, 0.0), (1, 0.5), (2, 1.0)]


class TestRecipe:
    """Kent keeps the AnimateLCM motion module but samples plainly; the LCM
    distillation LoRA and the sampler have to be separable for that."""

    def test_the_lcm_lora_can_be_dropped(self):
        wf = simple(lcm_lora_strength=0)
        assert "al_lcmlora" not in wf
        assert wf["al_evolved"]["inputs"]["model"] == ["al_ckpt", 0]

    def test_the_lcm_lora_is_on_by_default(self):
        wf = simple()
        assert wf["al_evolved"]["inputs"]["model"] == ["al_lcmlora", 0]
        assert wf["al_lcmlora"]["inputs"]["strength_model"] == 1.0

    def test_the_sampler_reaches_both_passes(self):
        wf = simple(sampler="euler", scheduler="normal")
        for ks in ("al_ks", "al_ks2"):
            assert (wf[ks]["inputs"]["sampler_name"], wf[ks]["inputs"]["scheduler"]) \
                == ("euler", "normal")


class TestSweep:
    def base(self):
        return AnimateLcmRequest(control_video=DEPTH, prompt="x",
                                 reference_image="r.png", filename_prefix="s")

    def test_the_hires_sweep_moves_only_the_second_denoise(self):
        sheet = sweep_workflows(self.base(), "hires")
        assert len({wf["al_ks2"]["inputs"]["denoise"] for _, wf, _ in sheet}) == len(sheet)
        assert len({wf["al_cn"]["inputs"]["strength"] for _, wf, _ in sheet}) == 1

    def test_the_depth_sweep_moves_only_the_base_controlnet(self):
        sheet = sweep_workflows(self.base(), "depth")
        assert len({wf["al_cn"]["inputs"]["strength"] for _, wf, _ in sheet}) == len(sheet)
        assert len({wf["al_ipa0"]["inputs"]["weight"] for _, wf, _ in sheet}) == 1

    def test_the_ip_sweep_moves_only_the_adapter(self):
        sheet = sweep_workflows(self.base(), "ip")
        assert len({wf["al_ipa0"]["inputs"]["weight"] for _, wf, _ in sheet}) == len(sheet)
        assert len({wf["al_cn"]["inputs"]["strength"] for _, wf, _ in sheet}) == 1

    def test_the_seed_is_shared(self):
        sheet = sweep_workflows(self.base(), "hires")
        assert len({wf["al_ks"]["inputs"]["seed"] for _, wf, _ in sheet}) == 1

    def test_each_render_writes_its_own_file(self):
        sheet = sweep_workflows(self.base(), "hires", values=(0.3, 0.5))
        assert {wf["al_out"]["inputs"]["filename_prefix"]
                for _, wf, _ in sheet} == {"s_hires030", "s_hires050"}

    def test_an_unknown_dial_is_refused(self):
        with pytest.raises(ValueError, match="dial"):
            sweep_workflows(self.base(), "colour")
