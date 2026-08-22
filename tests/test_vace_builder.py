"""
Unit tests for the Wan VACE structure-video builder (services/comfy/vace.py)
— pure graph construction, no ComfyUI.

The properties worth pinning down are the ones that came out of reading
comfy_extras/nodes_wan.py and then being contradicted by the GPU: that a mask
makes VACE *preserve* the unmasked control video rather than select a
reference for it, so regions are a sequence of passes and never several masked
blocks in one; that a later pass must not treat its own control track as a
depth map; and that the reference frames have to be trimmed back off after
sampling.
"""
import pytest

from services.comfy.vace import (
    ARTISTIC_NEGATIVE,
    DEFAULT_LENGTH,
    STRENGTH_DEFAULT,
    STRENGTH_MAX,
    STOCK_NEGATIVE,
    STRENGTH_MIN,
    SWEEP_VALUES,
    Region,
    VaceRequest,
    build_vace_workflow,
    canvas_size,
    clamp_strength,
    estimate_seconds,
    plan_region_passes,
    quality_sweep_workflows,
    snap_length,
    snap_size,
    sweep_workflows,
)

DEPTH = "C:/fake/depth.mp4"
MASKS = "C:/fake/masks.mp4"


def simple(**kw) -> dict:
    args = {"control_video": DEPTH, "prompt": "a world", "reference_image": "ref.png"}
    args.update(kw)
    return build_vace_workflow(VaceRequest(**args))[0]


def regional(n: int, **kw) -> dict:
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    args = {
        "control_video": DEPTH, "mask_video": MASKS, "prompt": "a world",
        "regions": [Region(colors[i], f"ref{i}.png") for i in range(n)],
    }
    args.update(kw)
    return build_vace_workflow(VaceRequest(**args))[0]


def vace_blocks(wf: dict) -> list[str]:
    return sorted(k for k, v in wf.items() if v["class_type"] == "WanVaceToVideo")


class TestSimpleShape:
    """Depth video + one reference image, no mask — the user's plain case."""

    def test_one_vace_block(self):
        assert len(vace_blocks(simple())) == 1

    def test_no_mask_input_is_wired(self):
        wf = simple()
        assert "control_masks" not in wf["vc_vace0"]["inputs"]

    def test_reference_is_wired_through_a_loadimage(self):
        wf = simple()
        ref = wf["vc_vace0"]["inputs"]["reference_image"]
        assert wf[ref[0]]["class_type"] == "LoadImage"
        assert wf[ref[0]]["inputs"]["image"] == "ref.png"

    def test_reference_is_optional(self):
        wf = simple(reference_image=None)
        assert "reference_image" not in wf["vc_vace0"]["inputs"]

    def test_control_video_reaches_the_block(self):
        wf = simple()
        assert "control_video" in wf["vc_vace0"]["inputs"]


class TestDepthOrientation:
    """A Blender depth pass arrives inverted; the old graph fixed it with an
    ImageInvert and this one keeps the switch rather than assuming."""

    def test_inverted_by_default(self):
        wf = simple()
        assert any(v["class_type"] == "ImageInvert" for v in wf.values())

    def test_can_be_turned_off(self):
        wf = simple(invert_depth=False)
        assert not any(v["class_type"] == "ImageInvert" for v in wf.values())

    def test_fitting_pads_rather_than_crops_or_stretches(self):
        # VACE's own fallback centre-crops and the old builder stretched; a
        # square turntable on a wide canvas needs neither. Padding keeps the
        # geometry undistorted and hands the model empty background instead.
        assert simple()["vc_fit"]["inputs"]["method"] == "pad"
        assert simple(fit="crop")["vc_fit"]["inputs"]["method"] == "fill / crop"
        assert simple(fit="stretch")["vc_fit"]["inputs"]["method"] == "stretch"


class TestDerivedDepth:
    """Ordinary footage becomes a depth pass in-graph — the user's 'Realvideo
    durch einen Depth-Map-Node' case."""

    def test_preprocessor_is_inserted(self):
        wf = simple(derive_depth=True)
        assert wf["vc_depth"]["class_type"] == "DepthAnythingV2Preprocessor"
        assert wf["vc_depth"]["inputs"]["image"] == ["vc_ctrl", 0]
        assert wf["vc_fit"]["inputs"]["image"] == ["vc_depth", 0]

    def test_derived_depth_is_never_also_inverted(self):
        # DepthAnything already outputs near-bright/far-dark; inverting on top
        # would hand VACE an upside-down depth pass.
        wf = simple(derive_depth=True, invert_depth=True)
        assert not any(v["class_type"] == "ImageInvert" for v in wf.values())

    def test_off_by_default(self):
        assert not any(
            v["class_type"] == "DepthAnythingV2Preprocessor" for v in simple().values()
        )

    def test_resolution_follows_the_canvas(self):
        wf = simple(derive_depth=True, width=720, height=480)
        assert wf["vc_depth"]["inputs"]["resolution"] == 720


class TestRegions:
    """One workflow is one pass, and one pass repaints one region — measured:
    chaining several masked blocks into a single pass renders the depth map
    instead, because a mask asks VACE to *preserve* the unmasked control video
    and a depth pass is grey."""

    def test_one_region_builds_one_block(self):
        assert len(vace_blocks(regional(1))) == 1

    def test_several_regions_in_one_pass_are_refused(self):
        with pytest.raises(ValueError, match="one region per pass"):
            regional(3)

    def test_region_colour_reaches_colortomask(self):
        wf = regional(1)
        key = next(v for v in wf.values() if v["class_type"] == "ColorToMask")["inputs"]
        assert (key["red"], key["green"], key["blue"]) == (255, 0, 0)

    def test_mask_is_wired_into_the_block(self):
        wf = regional(1)
        assert wf["vc_vace0"]["inputs"]["control_masks"] == ["vc_mask0", 0]

    def test_mask_track_is_scaled_without_smoothing(self):
        # A keyed colour must survive resampling; bilinear would blend the
        # region edges into colours ColorToMask no longer recognises.
        wf = regional(1)
        assert wf["vc_maskfit"]["inputs"]["interpolation"] == "nearest-exact"

    def test_mask_track_is_fitted_exactly_like_the_control_track(self):
        # Anything else moves the regions off the thing they were keyed from.
        wf = regional(1, fit="crop")
        assert wf["vc_maskfit"]["inputs"]["method"] == wf["vc_fit"]["inputs"]["method"]
        assert wf["vc_maskfit"]["inputs"]["width"] == wf["vc_fit"]["inputs"]["width"]
        assert wf["vc_maskfit"]["inputs"]["height"] == wf["vc_fit"]["inputs"]["height"]

    def test_regions_need_a_mask_video(self):
        with pytest.raises(ValueError, match="mask_video"):
            build_vace_workflow(VaceRequest(
                control_video=DEPTH, prompt="x",
                regions=[Region((255, 0, 0), "a.png")],
            ))

    def test_a_region_needs_a_reference(self):
        with pytest.raises(ValueError, match="needs a reference"):
            build_vace_workflow(VaceRequest(
                control_video=DEPTH, mask_video=MASKS, prompt="x",
                regions=[Region((255, 0, 0), "")],
            ))


class TestRegionPlan:
    """The multi-region job is a sequence: pass 0 paints everything, each later
    pass repaints one region on top of the previous pass's render."""

    def plan(self, n=3):
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        return plan_region_passes(VaceRequest(
            control_video=DEPTH, mask_video=MASKS, prompt="x",
            regions=[Region(colors[i], f"ref{i}.png") for i in range(n)],
        ))

    def test_one_pass_per_region(self):
        assert len(self.plan(3)) == 3

    def test_first_pass_has_no_mask_and_paints_everything(self):
        first = self.plan()[0]
        assert first.regions == []
        assert first.reference_image == "ref0.png"

    def test_first_pass_keeps_the_depth_track(self):
        first = self.plan()[0]
        assert first.control_video == DEPTH
        assert first.invert_depth is True

    def test_later_passes_await_the_previous_render(self):
        for later in self.plan()[1:]:
            assert later.control_video == "", "the caller fills this in"
            assert len(later.regions) == 1

    def test_later_passes_never_treat_a_render_as_depth(self):
        # The control track is a finished picture from pass 1 on: inverting it
        # or running DepthAnything over it would destroy what it preserves.
        for later in self.plan()[1:]:
            assert later.invert_depth is False
            assert later.derive_depth is False

    def test_each_pass_writes_its_own_file(self):
        prefixes = [p.filename_prefix for p in self.plan()]
        assert len(set(prefixes)) == 3

    def test_no_regions_is_a_single_pass(self):
        plan = plan_region_passes(VaceRequest(
            control_video=DEPTH, prompt="x", reference_image="r.png"))
        assert len(plan) == 1 and plan[0].regions == []

    def test_the_request_is_not_mutated(self):
        req = VaceRequest(control_video=DEPTH, mask_video=MASKS, prompt="x",
                          regions=[Region((255, 0, 0), "a.png"), Region((0, 255, 0), "b.png")])
        plan_region_passes(req)
        assert len(req.regions) == 2 and req.reference_image is None


class TestSamplingTail:
    def test_sampler_reads_the_vace_block(self):
        wf = regional(1)
        assert wf["vc_ks"]["inputs"]["latent_image"] == ["vc_vace0", 2]
        assert wf["vc_ks"]["inputs"]["positive"] == ["vc_vace0", 0]

    def test_reference_frames_are_trimmed_before_the_decode(self):
        wf = simple()
        assert wf["vc_trim"]["inputs"]["samples"] == ["vc_ks", 0]
        assert wf["vc_trim"]["inputs"]["trim_amount"] == ["vc_vace0", 3]
        assert wf["vc_dec"]["inputs"]["samples"] == ["vc_trim", 0]

    def test_trim_amount_comes_from_the_block(self):
        wf = regional(1)
        assert wf["vc_trim"]["inputs"]["trim_amount"] == ["vc_vace0", 3]


class TestCanvasAndRecipe:
    """Size and sampler are separate axes on purpose. They used to be one
    `tier`, and that made the only interesting comparison — the same scene under
    two samplers — impossible, because changing the canvas changes the scene."""

    def test_both_canvases_load_the_same_gguf_model(self):
        for canvas in ("sketch", "final"):
            wf = simple(canvas=canvas)
            assert wf["vc_unet"]["class_type"] == "UnetLoaderGGUF"
            assert wf["vc_lora"]["inputs"]["model"] == ["vc_unet", 0]
            assert wf["vc_shift"]["inputs"]["model"] == ["vc_lora", 0]

    def test_the_canvas_changes_the_size_and_nothing_else(self):
        sketch = simple(canvas="sketch")["vc_ks"]["inputs"]
        final = simple(canvas="final")["vc_ks"]["inputs"]
        assert (sketch["steps"], sketch["cfg"]) == (final["steps"], final["cfg"])
        assert (simple(canvas="sketch")["vc_vace0"]["inputs"]["width"]
                < simple(canvas="final")["vc_vace0"]["inputs"]["width"])

    def test_the_recipe_changes_the_sampler_and_nothing_else(self):
        fast = simple(recipe="fast")
        full = simple(recipe="full")
        for wf in (fast, full):
            block = wf["vc_vace0"]["inputs"]
            assert (block["width"], block["height"]) == (832, 480)
        assert fast["vc_ks"]["inputs"]["steps"] != full["vc_ks"]["inputs"]["steps"]

    def test_the_default_recipe_is_distilled_at_cfg_one(self):
        wf = simple()
        assert wf["vc_ks"]["inputs"]["cfg"] == 1.0
        assert wf["vc_lora"]["inputs"]["strength_model"] == 1.0

    def test_the_full_recipe_drops_the_distill_lora_and_guides(self):
        wf = simple(recipe="full")
        assert "vc_lora" not in wf
        assert wf["vc_shift"]["inputs"]["model"] == ["vc_unet", 0]
        assert wf["vc_ks"]["inputs"]["cfg"] > 1.0
        assert wf["vc_ks"]["inputs"]["steps"] == 20

    def test_the_middle_recipe_weakens_the_lora_rather_than_removing_it(self):
        wf = simple(recipe="rich")
        assert wf["vc_lora"]["inputs"]["strength_model"] == 0.5
        assert wf["vc_ks"]["inputs"]["cfg"] == 1.0

    def test_small_is_the_other_model(self):
        wf = simple(recipe="small")
        assert wf["vc_unet"]["class_type"] == "UNETLoader"
        assert "vc_lora" not in wf

    def test_explicit_overrides_beat_the_recipe(self):
        wf = simple(recipe="fast", steps=12, cfg=3.5)
        assert wf["vc_ks"]["inputs"]["steps"] == 12
        assert wf["vc_ks"]["inputs"]["cfg"] == 3.5

    def test_wide_is_wans_own_480p_shape(self):
        assert canvas_size("sketch", "wide") == (832, 480)
        assert canvas_size("sketch", "tall") == (480, 832)
        assert canvas_size("final", "wide") == (1280, 720)

    def test_an_unknown_axis_value_falls_back_rather_than_raising(self):
        # The size table is not user input, but a stale client should not take
        # a background job down ten minutes in.
        assert canvas_size("nope", "wide") == canvas_size("sketch", "wide")
        assert canvas_size("sketch", "nope") == canvas_size("sketch", "wide")


class TestQualitySweep:
    """The contact sheet that answers "is it the sampler?" — and it only
    answers it if everything except the sampler is held still."""

    def sheet(self, **kw):
        args = {"control_video": DEPTH, "prompt": "a world", "reference_image": "r.png"}
        args.update(kw)
        return quality_sweep_workflows(VaceRequest(**args))

    def test_one_workflow_per_recipe(self):
        assert [name for name, _, _ in self.sheet()] == ["fast", "rich", "full"]

    def test_the_seed_and_the_canvas_are_shared(self):
        sheet = self.sheet()
        seeds = {wf["vc_ks"]["inputs"]["seed"] for _, wf, _ in sheet}
        sizes = {(wf["vc_vace0"]["inputs"]["width"],
                  wf["vc_vace0"]["inputs"]["height"]) for _, wf, _ in sheet}
        assert len(seeds) == 1 and len(sizes) == 1

    def test_the_samplers_actually_differ(self):
        sheet = self.sheet()
        samplers = {(wf["vc_ks"]["inputs"]["steps"], wf["vc_ks"]["inputs"]["cfg"])
                    for _, wf, _ in sheet}
        assert len(samplers) == 3

    def test_each_pass_writes_its_own_file(self):
        prefixes = {wf["vc_out"]["inputs"]["filename_prefix"]
                    for _, wf, _ in self.sheet(filename_prefix="q")}
        assert prefixes == {"q_fast", "q_rich", "q_full"}


class TestNegativePrompt:
    """The stock Wan negative forbids stylisation and painting outright, which
    is the opposite of what this tool is for."""

    def test_the_default_no_longer_forbids_stylisation(self):
        for term in ("风格化", "作品", "画作", "整体发灰"):
            assert term in STOCK_NEGATIVE
            assert term not in ARTISTIC_NEGATIVE

    def test_the_anatomy_and_motion_terms_survive(self):
        for term in ("多余的手指", "静态", "低质量"):
            assert term in ARTISTIC_NEGATIVE

    def test_the_builder_uses_the_artistic_one(self):
        assert simple()["vc_neg"]["inputs"]["text"] == ARTISTIC_NEGATIVE


class TestColorMatch:
    """Measured: repainting a warm stage-A render comes back in saturated cyan,
    and rewording the prompt does nothing about it — at cfg 1.0 there is no
    unconditional pass for the negative to work through. So the palette is
    graded back after the decode instead."""

    def test_off_by_default(self):
        wf = simple()
        assert "vc_match" not in wf
        assert wf["vc_out"]["inputs"]["images"] == ["vc_dec", 0]

    def test_it_sits_between_the_decode_and_the_muxer(self):
        wf = simple(color_match=0.8)
        assert wf["vc_match"]["inputs"]["image"] == ["vc_dec", 0]
        assert wf["vc_out"]["inputs"]["images"] == ["vc_match", 0]

    def test_it_grades_against_the_fitted_control_track(self):
        # Against the *fitted* track, not the raw one: the reference has to be
        # the same shape as the thing being graded.
        wf = simple(color_match=0.8)
        assert wf["vc_match"]["inputs"]["reference"] == ["vc_fit", 0]

    def test_lab_so_the_repaints_light_survives(self):
        assert simple(color_match=1.0)["vc_match"]["inputs"]["color_space"] == "LAB"

    def test_it_grades_the_whole_clip_at_once(self):
        # Not a preference — measured the hard way. The node takes per-frame
        # reference statistics over the whole reference batch but chunks only
        # the image, so any batch_size below the frame count raises
        # "size of tensor a (8) must match the size of tensor b (45)".
        assert simple(color_match=1.0)["vc_match"]["inputs"]["batch_size"] == 0

    def test_it_grades_on_the_cpu(self):
        # A per-pixel affine transform, run immediately after sampling — no
        # reason to spike VRAM for it.
        assert simple(color_match=1.0)["vc_match"]["inputs"]["device"] == "cpu"


class TestEstimate:
    """An ETA, anchored on the two measured renders. Its job is to stop a
    50-minute job from starting by accident, not to be exact."""

    def test_the_measured_sketch_render_lands_within_a_fifth(self):
        # 480x480, 81 frames, 6 distilled steps took 229 s.
        assert 183 <= estimate_seconds("sketch", "square", "fast", 81) <= 275

    def test_the_measured_final_render_lands_within_a_fifth(self):
        # 720x720, same clip and sampler, 575 s.
        assert 460 <= estimate_seconds("final", "square", "fast", 81) <= 690

    def test_guidance_doubles_the_work_per_step(self):
        guided = estimate_seconds("sketch", "wide", "full", 81)
        unguided = estimate_seconds("sketch", "wide", "rich", 81)
        # 20 steps at cfg against 12 at cfg 1.0: 40 evaluations against 12.
        assert round(guided / unguided) == 3


class TestGeometryConstraints:
    """Wan samples 4n+1 frames and VACE declares step=16 on the edges; getting
    either wrong fails quietly rather than loudly."""

    @pytest.mark.parametrize("raw,snapped", [(81, 81), (94, 93), (80, 77), (1, 5), (0, 5)])
    def test_length_snaps_to_4n_plus_1(self, raw, snapped):
        assert snap_length(raw) == snapped
        assert (snap_length(raw) - 1) % 4 == 0

    @pytest.mark.parametrize("raw,snapped", [(480, 480), (1080, 1072), (17, 16), (0, 16)])
    def test_size_snaps_to_multiples_of_16(self, raw, snapped):
        assert snap_size(raw) == snapped

    def test_default_length_is_valid(self):
        assert snap_length(DEFAULT_LENGTH) == DEFAULT_LENGTH

    def test_snapped_values_reach_the_block(self):
        wf = simple(width=1080, height=1080, length=94)
        block = wf["vc_vace0"]["inputs"]
        assert (block["width"], block["height"], block["length"]) == (1072, 1072, 93)


class TestStrength:
    def test_default_when_unset(self):
        assert clamp_strength(None) == STRENGTH_DEFAULT

    def test_clamped_into_range(self):
        assert clamp_strength(-1) == STRENGTH_MIN
        assert clamp_strength(99) == STRENGTH_MAX

    def test_value_in_range_survives(self):
        assert clamp_strength(0.55) == 0.55

    def test_a_region_carries_its_own_strength(self):
        wf = build_vace_workflow(VaceRequest(
            control_video=DEPTH, mask_video=MASKS, prompt="x",
            regions=[Region((255, 0, 0), "a.png", strength=0.3)],
        ))[0]
        assert wf["vc_vace0"]["inputs"]["strength"] == 0.3


class TestSweep:
    def test_one_workflow_per_value(self):
        req = VaceRequest(control_video=DEPTH, prompt="x", reference_image="r.png")
        out = sweep_workflows(req)
        assert [v for v, _, _ in out] == list(SWEEP_VALUES)

    def test_seed_is_fixed_across_the_sweep(self):
        req = VaceRequest(control_video=DEPTH, prompt="x", reference_image="r.png", seed=-1)
        seeds = {wf["vc_ks"]["inputs"]["seed"] for _, wf, _ in sweep_workflows(req)}
        assert len(seeds) == 1, "a sweep that moves the seed measures two things at once"

    def test_only_strength_moves(self):
        req = VaceRequest(control_video=DEPTH, prompt="x", reference_image="r.png", seed=7)
        found = {wf["vc_vace0"]["inputs"]["strength"] for _, wf, _ in sweep_workflows(req)}
        assert found == set(SWEEP_VALUES)

    def test_each_variant_writes_its_own_file(self):
        req = VaceRequest(control_video=DEPTH, prompt="x", reference_image="r.png")
        names = {wf["vc_out"]["inputs"]["filename_prefix"] for _, wf, _ in sweep_workflows(req)}
        assert len(names) == len(SWEEP_VALUES)

    def test_region_strength_follows_the_sweep(self):
        req = VaceRequest(
            control_video=DEPTH, mask_video=MASKS, prompt="x",
            regions=[Region((255, 0, 0), "a.png")],
        )
        for value, wf, _ in sweep_workflows(req):
            assert wf["vc_vace0"]["inputs"]["strength"] == value

    def test_the_request_is_not_mutated(self):
        req = VaceRequest(control_video=DEPTH, prompt="x", reference_image="r.png",
                          strength=0.75)
        sweep_workflows(req)
        assert req.strength == 0.75
