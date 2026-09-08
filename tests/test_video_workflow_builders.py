"""
Unit tests for the flf2v per-transition workflow builder (pure node-graph
wiring, no ComfyUI) and the transition-prompt VLM wrapper's pad/truncate
logic (mocked _chat_json, no Ollama).
"""
import asyncio
import uuid

import pytest

from routers import video as video_module
from services.comfy import vram as vram_module
from routers.video import (
    _build_flf2v_single_workflow,
    _build_i2v_single_workflow,
    _build_minimax_single_workflow,
    _look_source,
    _is_graining,
    _is_upscaling,
    _render_version,
    _upscale_source,
    _validate_upscale_target,
    adapt_minimax_canvas,
    align_minimax_length,
    clamp_lora_high,
    clamp_wan_steps,
    align_wan_length,
    ensure_sound_only_audio,
    wan_cfg_high,
    wan_model_evals,
    estimate_wan_seconds,
    wan_output_fps,
    wan_poll_timeout,
)
from routers.video import _LORA_HIGH as _LORA_HIGH_NAME
from routers.video import _LORA_LOW as _LORA_LOW_NAME
from routers.video import clamp_style_strength
from routers.video import (
    _UNET_FUN_HIGH,
    _UNET_FUN_LOW,
    _UNET_HIGH,
    _UNET_LOW,
    WAN_NATIVE_FPS,
)
from routers.video import settings as video_settings
from services.comfy.node_labels import label_for_class
from services.comfy.wan_transition_loras import (
    offered as offered_transition_loras,
)
from services.comfy.wan_transition_loras import trigger_for, with_lora_trigger
from services.video.upscale import (
    RESOLUTION_KEEP,
    build_upscale_workflow,
    clamp_resolution,
    clamp_rife,
    estimate_seconds,
    needs_comfy,
    output_dimensions,
    plan_frame_rate,
)
from services.ollama import analysis as analysis_module
from services.ollama.analysis import (
    generate_i2v_motion_prompts,
    generate_minimax_motion_prompts,
    generate_minimax_transition_prompts,
    generate_transition_prompts,
)


class TestBuildFlf2vSingleWorkflow:
    def test_returns_dict_and_matching_save_node(self):
        wf, save_id = _build_flf2v_single_workflow(
            "start.png", "end.png", "camera pushes in", 25, 960, 960, "prefix", 3,
        )
        assert save_id in wf
        assert wf[save_id]["class_type"] == "VHS_VideoCombine"

    def test_load_image_nodes_for_start_and_end(self):
        wf, _ = _build_flf2v_single_workflow(
            "start.png", "end.png", "prompt", 25, 960, 960, "prefix", 3,
        )
        assert wf["img_start"] == {"class_type": "LoadImage", "inputs": {"image": "start.png", "upload": "image"}}
        assert wf["img_end"] == {"class_type": "LoadImage", "inputs": {"image": "end.png", "upload": "image"}}

    def test_end_frame_append_is_off_by_default(self):
        # Default: RIFE reads the diffused frames directly — no raw end photo
        # appended (that append IS the hard cut when diffusion undershoots).
        wf, _ = _build_flf2v_single_workflow(
            "start.png", "end.png", "prompt", 25, 960, 960, "prefix", 3,
        )
        assert "batch_final" not in wf
        assert wf["rife"]["inputs"]["frames"] == ["t0_decode", 0]

    def test_batch_final_appends_raw_end_frame_when_opted_in(self):
        wf, _ = _build_flf2v_single_workflow(
            "start.png", "end.png", "prompt", 25, 960, 960, "prefix", 3,
            append_end_frame=True,
        )
        batch = wf["batch_final"]
        assert batch["class_type"] == "ImageBatch"
        assert batch["inputs"]["image1"] == ["t0_decode", 0]
        assert batch["inputs"]["image2"] == ["img_end", 0]
        assert wf["rife"]["inputs"]["frames"] == ["batch_final", 0]

    def test_sampler_lightning_fast_path(self):
        # Lightning path: cfg=1 on both experts, full-strength distill on the
        # LOW-noise one. The step count and the high/low split are no longer
        # fixed at 4 and 2 — they default to 6 steps split by the sigma
        # schedule (2/4 at shift 5), and the high-noise distill defaults to
        # half strength so the transition actually travels.
        wf, _ = _build_flf2v_single_workflow(
            "start.png", "end.png", "prompt", 25, 960, 960, "prefix", 3,
        )
        ks_h, ks_l = wf["t0_ks_h"]["inputs"], wf["t0_ks_l"]["inputs"]
        assert ks_h["steps"] == 6 and ks_l["steps"] == 6
        assert ks_h["end_at_step"] == 2 and ks_l["start_at_step"] == 2
        assert ks_h["cfg"] == 1
        assert ks_l["cfg"] == 1
        assert wf["t0_lora_h"]["inputs"]["strength_model"] == 0.5
        assert wf["t0_lora_l"]["inputs"]["strength_model"] == 1.0

    def test_rife_multiplier_is_configurable(self):
        wf, _ = _build_flf2v_single_workflow(
            "start.png", "end.png", "prompt", 25, 960, 960, "prefix", 4,
        )
        assert wf["rife"]["inputs"]["multiplier"] == 4

    def test_save_node_reads_from_rife(self):
        wf, save_id = _build_flf2v_single_workflow(
            "start.png", "end.png", "prompt", 25, 960, 960, "prefix", 3,
        )
        assert wf[save_id]["inputs"]["images"] == ["rife", 0]

    def test_prompt_lands_in_positive_clip_encode(self):
        wf, _ = _build_flf2v_single_workflow(
            "start.png", "end.png", "a specific transition prompt", 25, 960, 960, "prefix", 3,
        )
        assert wf["t0_pos"]["inputs"]["text"] == "a specific transition prompt"


def _model_chain(wf: dict, node_id: str) -> list[str]:
    """The MODEL inputs feeding `node_id`, nearest first, skipping pure
    pass-through patches (an attention backend, say) that neither load nor
    modify weights. What is left is what the sampler is actually running."""
    passthrough = {"PathchSageAttentionKJ"}
    chain, seen = [], set()
    cur = wf[node_id]["inputs"].get("model")
    while isinstance(cur, list) and cur[0] not in seen:
        seen.add(cur[0])
        node = wf[cur[0]]
        if node["class_type"] not in passthrough:
            chain.append(cur[0])
        cur = node["inputs"].get("model")
    return chain


class TestAttentionBackend:
    """SageAttention is an approximation, so it is scoped to the two Wan
    builders and switchable without a code change."""

    def _i2v(self, **kw):
        return _build_i2v_single_workflow(
            "img.png", "prompt", 49, 960, 960, "prefix", 3, False, **kw
        )[0]

    def _flf2v(self, **kw):
        return _build_flf2v_single_workflow(
            "start.png", "end.png", "prompt", 49, 960, 960, "prefix", 3, **kw
        )[0]

    def test_both_experts_get_the_patch_when_enabled(self, monkeypatch):
        monkeypatch.setattr(video_settings, "wan_sage_attention", True)
        wf = self._i2v()
        assert wf["s0_sage_h"]["inputs"]["sage_attention"] == "auto"
        assert wf["s0_sage_l"]["inputs"]["sage_attention"] == "auto"
        assert wf["s0_samp_h"]["inputs"]["model"] == ["s0_sage_h", 0]
        assert wf["s0_samp_l"]["inputs"]["model"] == ["s0_sage_l", 0]

    def test_turning_it_off_yields_the_graph_from_before_it_existed(self, monkeypatch):
        monkeypatch.setattr(video_settings, "wan_sage_attention", False)
        wf = self._i2v()
        assert not any("sage" in n for n in wf)
        assert wf["s0_samp_h"]["inputs"]["model"] == ["s0_lora_h", 0]
        assert wf["s0_samp_l"]["inputs"]["model"] == ["s0_lora_l", 0]

    def test_it_patches_after_the_lora_not_before(self, monkeypatch):
        # Patching the raw UNET would leave the LoRA-merged model running
        # unpatched attention; the order has to be loader → LoRA → patch.
        monkeypatch.setattr(video_settings, "wan_sage_attention", True)
        wf = self._i2v(lora_high=0.5)
        assert wf["s0_sage_h"]["inputs"]["model"] == ["s0_lora_h", 0]
        assert wf["s0_sage_l"]["inputs"]["model"] == ["s0_lora_l", 0]

    def test_it_patches_the_bare_unet_when_the_high_lora_is_dropped(self, monkeypatch):
        monkeypatch.setattr(video_settings, "wan_sage_attention", True)
        wf = self._i2v(lora_high=0.0)
        assert wf["s0_sage_h"]["inputs"]["model"] == ["s0_unet_h", 0]

    @pytest.mark.parametrize("enabled", [True, False])
    def test_the_expert_chain_is_the_same_either_way(self, monkeypatch, enabled):
        monkeypatch.setattr(video_settings, "wan_sage_attention", enabled)
        wf = self._i2v(lora_high=0.5)
        assert _model_chain(wf, "s0_samp_h") == ["s0_lora_h", "s0_unet_h"]
        assert _model_chain(wf, "s0_samp_l") == ["s0_lora_l", "s0_unet_l"]

    @pytest.mark.parametrize("enabled", [True, False])
    def test_flf2v_is_patched_the_same_way(self, monkeypatch, enabled):
        monkeypatch.setattr(video_settings, "wan_sage_attention", enabled)
        wf = self._flf2v()
        assert ("t0_sage_h" in wf) is enabled
        assert _model_chain(wf, "t0_samp_h") == ["t0_lora_h", "t0_unet_h"]

    @pytest.mark.parametrize("enabled", [True, False])
    def test_no_dangling_links_either_way(self, monkeypatch, enabled):
        monkeypatch.setattr(video_settings, "wan_sage_attention", enabled)
        for wf in (self._i2v(lora_high=0.0), self._i2v(lora_high=0.5),
                   self._flf2v(lora_high=0.0), self._flf2v(lora_high=0.5)):
            for node in wf.values():
                for value in node["inputs"].values():
                    if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                        assert value[0] in wf, f"dangling link to {value[0]}"

    def test_the_patch_has_a_progress_label(self):
        # Otherwise the progress line reads "Pathch Sage Attention KJ…".
        assert label_for_class("PathchSageAttentionKJ") == "Switching to fast attention…"


class TestWanSamplerControls:
    """The two dials both Wan builders now share: total steps, and how much
    distill LoRA the high-noise expert carries (the motion dial)."""

    def _i2v(self, **kw):
        return _build_i2v_single_workflow(
            "img.png", "prompt", 49, 960, 960, "prefix", 3, False, **kw
        )[0]

    def _flf2v(self, **kw):
        return _build_flf2v_single_workflow(
            "start.png", "end.png", "prompt", 49, 960, 960, "prefix", 3, **kw
        )[0]

    @pytest.mark.parametrize("steps,split", [(4, 1), (6, 2), (8, 2), (10, 3), (16, 5)])
    def test_split_follows_the_sigma_schedule_not_half_the_steps(self, steps, split):
        wf = self._i2v(steps=steps)
        assert wf["s0_ks_h"]["inputs"]["end_at_step"] == split
        assert wf["s0_ks_l"]["inputs"]["start_at_step"] == split
        assert wf["s0_ks_h"]["inputs"]["steps"] == steps
        assert wf["s0_ks_l"]["inputs"]["steps"] == steps

    def test_both_wan_builders_agree_on_the_split(self):
        # They used to hold separate copies of these numbers and had already
        # drifted on shift; _wan_expert_nodes is the one source now.
        for steps in (4, 6, 8, 10, 16):
            i2v = self._i2v(steps=steps)["s0_ks_h"]["inputs"]
            flf = self._flf2v(steps=steps)["t0_ks_h"]["inputs"]
            assert i2v["end_at_step"] == flf["end_at_step"]
            assert i2v["scheduler"] == flf["scheduler"] == "simple"

    def test_high_noise_lora_strength_is_the_motion_dial(self):
        assert self._i2v(lora_high=0.4)["s0_lora_h"]["inputs"]["strength_model"] == 0.4
        assert self._flf2v(lora_high=0.7)["t0_lora_h"]["inputs"]["strength_model"] == 0.7

    def test_low_noise_lora_stays_at_full_strength_whatever_the_dial_says(self):
        # Detail comes from the low-noise expert; weakening it would cost
        # quality without buying any motion.
        for lh in (0.0, 0.4, 1.0):
            assert self._i2v(lora_high=lh)["s0_lora_l"]["inputs"]["strength_model"] == 1.0

    def test_zero_strength_drops_the_lora_node_entirely(self):
        wf = self._i2v(lora_high=0.0)
        assert "s0_lora_h" not in wf
        # Walk the chain rather than naming the immediate neighbour: an
        # attention-backend patch may or may not sit between the loader and
        # the sampler, and that is not what this test is about.
        assert _model_chain(wf, "s0_samp_h") == ["s0_unet_h"]
        # The low-noise chain is untouched by that.
        assert _model_chain(wf, "s0_samp_l") == ["s0_lora_l", "s0_unet_l"]

    def test_high_noise_expert_is_still_loaded_at_zero_strength(self):
        # Dropping the LoRA must not drop the expert — it is the one that
        # plans the motion in the first place.
        wf = self._i2v(lora_high=0.0)
        assert wf["s0_unet_h"]["inputs"]["unet_name"].startswith("wan2.2_i2v_high_noise")

    def test_omitted_controls_fall_back_to_the_defaults(self):
        # An older service-worker-cached frontend sends neither field.
        wf = self._i2v()
        assert wf["s0_ks_h"]["inputs"]["steps"] == 6
        assert wf["s0_lora_h"]["inputs"]["strength_model"] == 0.5

    @pytest.mark.parametrize("given,expected", [
        (None, 6), (1, 4), (4, 4), (6, 6), (16, 16), (40, 16), (-3, 4),
    ])
    def test_steps_are_clamped_into_a_range_the_card_can_finish(self, given, expected):
        assert clamp_wan_steps(given) == expected

    @pytest.mark.parametrize("given,expected", [
        (None, 0.5), (0.0, 0.0), (0.4, 0.4), (1.0, 1.0), (2.5, 1.0), (-1.0, 0.0),
    ])
    def test_lora_strength_is_clamped_to_zero_one(self, given, expected):
        assert clamp_lora_high(given) == expected

    def test_out_of_range_values_still_build_a_valid_graph(self):
        wf = self._i2v(steps=999, lora_high=99.0)
        ks_h = wf["s0_ks_h"]["inputs"]
        assert ks_h["steps"] == 16 and 1 <= ks_h["end_at_step"] < 16
        assert wf["s0_lora_h"]["inputs"]["strength_model"] == 1.0

    def test_every_node_reference_resolves(self):
        # Cheap structural check that dropping/adding the LoRA node never
        # leaves a dangling link in either builder.
        for wf in (self._i2v(lora_high=0.0), self._i2v(lora_high=0.5),
                   self._flf2v(lora_high=0.0), self._flf2v(lora_high=0.5)):
            for node in wf.values():
                for value in node["inputs"].values():
                    if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                        assert value[0] in wf, f"dangling link to {value[0]}"


class TestWanRenderBudget:
    """A flat 30-minute deadline was safe while every render was 4 steps on one
    canvas. Steps, canvas and frame count are all settings now, and they
    multiply — the budget has to move with them or a long job is killed with
    its file already half-written."""

    # 960x960 x 49 frames, RIFE 3, 16GB 4060 Ti, measured 2026-08-26 on
    # PyTorch SDPA — the four points the line was fitted to.
    @pytest.mark.parametrize("steps,measured", [
        (4, 355.9), (6, 496.2), (8, 646.6), (10, 797.2),
    ])
    def test_stays_within_a_percent_of_what_was_measured(self, steps, measured):
        assert (estimate_wan_seconds(960, 960, 49, steps, sage=False)
                == pytest.approx(measured, rel=0.01))

    # Same canvas, same seed, SageAttention on. Two points only, and the slope
    # was fitted holding the SDPA line's fixed cost rather than given its own
    # intercept — hence the looser tolerance than the four-point SDPA line.
    @pytest.mark.parametrize("steps,measured", [(6, 325.9), (10, 516.5)])
    def test_the_sage_line_matches_its_own_measurements(self, steps, measured):
        assert (estimate_wan_seconds(960, 960, 49, steps, sage=True)
                == pytest.approx(measured, rel=0.02))

    def test_sage_is_cheaper_at_every_step_count(self):
        for steps in (4, 6, 8, 10, 16):
            assert (estimate_wan_seconds(960, 960, 49, steps, sage=True)
                    < estimate_wan_seconds(960, 960, 49, steps, sage=False))

    def test_sage_changes_sampling_only(self):
        # The gap between the two backends must be pure per-step cost: the
        # model loads, VAE, RIFE and encode are identical either way, so the
        # difference has to scale exactly with the step count.
        def gap(steps):
            return (estimate_wan_seconds(960, 960, 49, steps, sage=False)
                    - estimate_wan_seconds(960, 960, 49, steps, sage=True))
        assert gap(10) == pytest.approx(gap(5) * 2, rel=0.01)

    def test_six_sage_steps_undercut_the_old_four_sdpa_steps(self):
        # The headline result, pinned: the new default is both a better picture
        # and a shorter wait than what it replaced.
        assert (estimate_wan_seconds(960, 960, 49, 6, sage=True)
                < estimate_wan_seconds(960, 960, 49, 4, sage=False))

    def test_the_default_backend_is_what_the_setting_says(self):
        for flag in (True, False):
            with pytest.MonkeyPatch.context() as mp:
                mp.setattr(video_settings, "wan_sage_attention", flag)
                assert (estimate_wan_seconds(960, 960, 49, 6)
                        == estimate_wan_seconds(960, 960, 49, 6, sage=flag))

    def test_cost_rises_with_every_factor(self):
        base = estimate_wan_seconds(960, 960, 49, 6)
        assert estimate_wan_seconds(960, 960, 49, 8) > base    # steps
        assert estimate_wan_seconds(1920, 1088, 49, 6) > base  # canvas
        assert estimate_wan_seconds(960, 960, 81, 6) > base    # frames

    def test_steps_are_the_dominant_term(self):
        # Sampling was measured at ~72 s/step against ~65 s for the whole rest
        # of the render, so doubling the steps must nearly double the estimate.
        ratio = estimate_wan_seconds(960, 960, 49, 8) / estimate_wan_seconds(960, 960, 49, 4)
        assert 1.6 < ratio < 1.9

    def test_short_renders_keep_the_old_flat_floor(self):
        assert wan_poll_timeout(960, 960, 49, 4) == 1800
        assert wan_poll_timeout(960, 960, 49, 6) == 1800

    def test_the_expensive_corner_gets_a_budget_it_can_finish_in(self):
        # 16 steps at 1920x1088 x 81 frames is the worst the UI can ask for.
        # The old flat 1800 s would have killed it around a third of the way in.
        worst = wan_poll_timeout(1920, 1088, 81, 16)
        assert worst > estimate_wan_seconds(1920, 1088, 81, 16)
        assert worst > 3 * 1800

    def test_budget_is_always_at_least_three_times_the_estimate(self):
        for w, h, f, s in [(960, 960, 49, 16), (1920, 1088, 49, 8), (704, 1280, 81, 12)]:
            assert wan_poll_timeout(w, h, f, s) >= 3 * estimate_wan_seconds(w, h, f, s)

    def test_degenerate_inputs_do_not_produce_a_zero_deadline(self):
        assert estimate_wan_seconds(0, 0, 0, 0) >= 1
        assert wan_poll_timeout(0, 0, 0, 0) == 1800


class TestAlignMinimaxLength:
    """MiniMax H3 only accepts frame counts on a 17k+5 grid (mirrors
    align_frame_count in comfy_extras/nodes_minimax_h3.py)."""

    @pytest.mark.parametrize("requested,expected", [
        (124, 124),   # already on the grid (the node's own default, ~5s)
        (100, 107),   # snaps up
        (5, 5),       # the grid's floor
        (1, 5),       # below the floor is lifted to it
        (363, 379),   # past the trained range still lands on the grid
    ])
    def test_snaps_up_onto_the_grid(self, requested, expected):
        assert align_minimax_length(requested) == expected

    def test_every_result_satisfies_the_model_constraint(self):
        assert all(align_minimax_length(n) % 17 == 5 for n in range(1, 400))


class TestAdaptMinimaxCanvas:
    """The conditioning node does not clamp its own width/height, so an
    off-canvas request would really be sampled off-canvas."""

    @pytest.mark.parametrize("size", [(768, 768), (1344, 768), (768, 1344), (1024, 768)])
    def test_native_canvases_pass_through_untouched(self, size):
        assert adapt_minimax_canvas(*size) == size

    @pytest.mark.parametrize("size", [(960, 544), (544, 960), (864, 480), (480, 864)])
    def test_small_canvases_are_not_enlarged(self, size):
        # The UI ships these four deliberately: below the model's native 768px
        # short edge, to buy back sampling time. An adapter that "helpfully"
        # scaled them up to native would make that choice impossible to express.
        # They must also survive untouched — a canvas the adapter would round
        # is a canvas the UI is lying about in its chip label.
        assert adapt_minimax_canvas(*size) == size

    def test_oversized_landscape_is_pulled_onto_the_canvas(self):
        # 1920x1088 is ~2x the model's pixel budget; the aspect ratio survives.
        w, h = adapt_minimax_canvas(1920, 1088)
        assert (w, h) == (1344, 768)

    def test_oversized_portrait_is_pulled_onto_the_canvas(self):
        assert adapt_minimax_canvas(1088, 1920) == (768, 1344)

    def test_result_is_always_a_multiple_of_32_within_the_pixel_cap(self):
        for w, h in [(960, 960), (1280, 704), (704, 1280), (1500, 500), (500, 1500)]:
            aw, ah = adapt_minimax_canvas(w, h)
            assert aw % 32 == 0 and ah % 32 == 0
            # Rounding to 32 can nudge one axis a step past the nominal cap.
            assert aw * ah <= 768 * 1344 * 1.05


class TestEnsureSoundOnlyAudio:
    """MiniMax samples audio from the same prompt and has no negative prompt,
    so "noise, not music" has to be in the positive text of every submission —
    not only the ones whose prompt the suggester wrote."""

    def test_prompt_without_an_audio_line_gets_one(self):
        out = ensure_sound_only_audio("Smoke curls off the surface.")
        assert "Audio:" in out
        assert "No background music or score" in out
        assert out.startswith("Smoke curls off the surface.")

    def test_existing_audio_line_survives_verbatim(self):
        out = ensure_sound_only_audio("Rain falls.\nAudio: heavy drops on tin, close and dry.")
        assert "Audio: heavy drops on tin, close and dry." in out
        assert out.count("Audio:") == 1          # no second, competing line
        assert "No background music or score" in out

    def test_prompt_that_already_rules_music_out_is_untouched(self):
        original = "Ash drifts.\nAudio: a low room tone. No music, no score."
        assert ensure_sound_only_audio(original) == original

    def test_empty_prompt_still_asks_for_sound(self):
        out = ensure_sound_only_audio("   ")
        assert out.startswith("Audio:")
        assert "No background music or score" in out

    def test_the_directive_reaches_the_workflow(self):
        wf, _, _ = _build_minimax_single_workflow(
            "img.png", "Smoke curls.", 56, 864, 480, "vid",
        )
        node = next(n for n in wf.values() if n["class_type"] == "MiniMaxH3ImageToVideo")
        assert "No background music or score" in node["inputs"]["prompt"]

    def test_case_and_spacing_variants_count_as_an_audio_line(self):
        for text in ["Wind moves.\naudio: hiss.", "Wind moves.\n  Audio : hiss."]:
            out = ensure_sound_only_audio(text)
            # "Audio :" with a space is not the documented shape, so a fallback
            # line is acceptable there — what must never happen is losing the
            # user's own words.
            assert "hiss." in out


class TestBuildMinimaxSingleWorkflow:
    def test_default_no_rife_node_wired_straight_from_decode(self):
        wf, save_id, _ = _build_minimax_single_workflow(
            "img.png", "a prompt", 124, 768, 768, "prefix",
        )
        assert "mmx_rife" not in wf
        assert wf[save_id]["inputs"]["images"] == ["mmx_vdec", 0]

    def test_rife_multiplier_inserts_rife_node(self):
        wf, save_id, _ = _build_minimax_single_workflow(
            "img.png", "a prompt", 124, 768, 768, "prefix", 3,
        )
        assert wf["mmx_rife"]["class_type"] == "RIFE VFI"
        assert wf["mmx_rife"]["inputs"]["multiplier"] == 3
        assert wf["mmx_rife"]["inputs"]["frames"] == ["mmx_vdec", 0]
        assert wf[save_id]["inputs"]["images"] == ["mmx_rife", 0]

    def test_audio_always_wired_from_native_decode_regardless_of_rife(self):
        # The generated audio is intentionally left at its pre-RIFE length —
        # the caller (services.video.audio_stretch) re-syncs it afterward.
        wf, save_id, _ = _build_minimax_single_workflow(
            "img.png", "a prompt", 124, 768, 768, "prefix", 4,
        )
        assert wf[save_id]["inputs"]["audio"] == ["mmx_adec", 0]

    def test_muxer_never_trims_the_video_back_to_the_audio(self):
        # RIFE lengthens the video while the generated audio stays short (we
        # stretch it afterwards). If VHS trimmed to the audio here, the whole
        # interpolation pass would be silently discarded.
        wf, save_id, _ = _build_minimax_single_workflow(
            "img.png", "a prompt", 124, 768, 768, "prefix", 3,
        )
        assert wf[save_id]["inputs"]["trim_to_audio"] is False

    def test_video_and_audio_decode_the_same_av_latent(self):
        # H3's defining property: one sampler pass yields both streams, so they
        # are in sync without any alignment step.
        wf, _, _ = _build_minimax_single_workflow(
            "img.png", "a prompt", 124, 768, 768, "prefix",
        )
        assert wf["mmx_vdec"]["inputs"]["samples"] == ["mmx_ks", 0]
        assert wf["mmx_adec"]["inputs"]["samples"] == ["mmx_ks", 0]
        assert wf["mmx_vdec"]["inputs"]["vae"] == ["mmx_vae", 0]
        assert wf["mmx_adec"]["inputs"]["vae"] == ["mmx_avae", 0]

    def test_frame_count_is_snapped_and_returned(self):
        wf, _, length = _build_minimax_single_workflow(
            "img.png", "prompt", 100, 768, 768, "prefix",
        )
        assert length == 107
        # The node must be asked for the same length the caller was told about,
        # or the audio-stretch maths is computed against the wrong duration.
        assert wf["mmx_i2v"]["inputs"]["length"] == 107

    def test_canvas_is_adapted_consistently_across_the_graph(self):
        wf, _, _ = _build_minimax_single_workflow(
            "img.png", "prompt", 124, 1920, 1088, "prefix",
        )
        assert wf["mmx_i2v"]["inputs"]["width"] == 1344
        assert wf["mmx_i2v"]["inputs"]["height"] == 768
        # The pre-scale must target the same canvas, otherwise the node's own
        # aspect-ignoring stretch kicks in and distorts the frame.
        assert wf["mmx_scale"]["inputs"]["width"] == 1344
        assert wf["mmx_scale"]["inputs"]["height"] == 768
        assert wf["mmx_scale"]["inputs"]["crop"] == "center"

    def test_source_image_reaches_the_node_via_the_center_crop(self):
        wf, _, _ = _build_minimax_single_workflow(
            "img.png", "prompt", 124, 768, 768, "prefix",
        )
        assert wf["mmx_load"]["inputs"]["image"] == "img.png"
        assert wf["mmx_scale"]["inputs"]["image"] == ["mmx_load", 0]
        assert wf["mmx_i2v"]["inputs"]["first_frame"] == ["mmx_scale", 0]

    def test_prompt_lands_in_the_conditioning_node(self):
        # Verbatim, then the audio directive — the model samples sound from
        # this same text and has no negative prompt to hear "not a score" from.
        wf, _, _ = _build_minimax_single_workflow(
            "img.png", "a specific minimax prompt", 124, 768, 768, "prefix",
        )
        prompt = wf["mmx_i2v"]["inputs"]["prompt"]
        assert prompt.startswith("a specific minimax prompt")
        assert "No background music or score" in prompt

    def test_sampling_is_guidance_free_off_the_conditioning_output(self):
        # BasicGuider takes positive only — H3 has no negative prompt / cfg.
        wf, _, _ = _build_minimax_single_workflow(
            "img.png", "prompt", 124, 768, 768, "prefix",
        )
        assert wf["mmx_guider"]["class_type"] == "BasicGuider"
        assert wf["mmx_guider"]["inputs"]["conditioning"] == ["mmx_i2v", 0]
        assert "negative" not in wf["mmx_guider"]["inputs"]
        assert wf["mmx_ks"]["inputs"]["latent_image"] == ["mmx_i2v", 1]

    def test_save_runs_at_the_models_fixed_frame_rate(self):
        # The builder takes no fps at all — the muxer must run at the rate the
        # audio was sampled against, or the two drift apart.
        wf, save_id, _ = _build_minimax_single_workflow(
            "img.png", "prompt", 124, 768, 768, "prefix",
        )
        assert wf[save_id]["inputs"]["frame_rate"] == 24


class TestGenerateTransitionPrompts:
    async def test_returns_n_minus_one_prompts_on_exact_match(self, monkeypatch):
        async def fake_chat_json(**kwargs):
            return {"transitions": ["a to b", "b to c"]}
        monkeypatch.setattr(analysis_module, "_chat_json", fake_chat_json)

        result = await generate_transition_prompts([b"1", b"2", b"3"])
        assert result == ["a to b", "b to c"]

    async def test_pads_short_response(self, monkeypatch):
        async def fake_chat_json(**kwargs):
            return {"transitions": ["only one"]}
        monkeypatch.setattr(analysis_module, "_chat_json", fake_chat_json)

        result = await generate_transition_prompts([b"1", b"2", b"3"])
        assert result == ["only one", ""]

    async def test_truncates_long_response(self, monkeypatch):
        async def fake_chat_json(**kwargs):
            return {"transitions": ["a", "b", "c", "d"]}
        monkeypatch.setattr(analysis_module, "_chat_json", fake_chat_json)

        result = await generate_transition_prompts([b"1", b"2"])
        assert result == ["a"]

    async def test_non_list_transitions_raises(self, monkeypatch):
        async def fake_chat_json(**kwargs):
            return {"transitions": "not a list"}
        monkeypatch.setattr(analysis_module, "_chat_json", fake_chat_json)

        with pytest.raises(RuntimeError):
            await generate_transition_prompts([b"1", b"2"])

    async def test_fewer_than_two_images_raises(self):
        with pytest.raises(RuntimeError):
            await generate_transition_prompts([b"1"])

    async def test_reads_prompt_file_without_raising(self):
        # Sanity check the prompts/video-transitions.md path/filename is correct
        # before it ever hits a live Ollama call.
        from services.ollama.chat import _read_prompt
        text = _read_prompt("video-transitions.md")
        assert text.strip()


class TestGenerateI2vMotionPrompts:
    async def test_one_call_per_image_with_single_jpg_each(self, monkeypatch):
        # The 3B titler VLM only really looks at the first image of a
        # multi-image message — the wrapper must fan out to one call per image.
        calls = []
        async def fake_chat_json(**kwargs):
            calls.append(kwargs)
            return {"animation": f"prompt {len(calls)}"}
        monkeypatch.setattr(analysis_module, "_chat_json", fake_chat_json)

        result = await generate_i2v_motion_prompts([b"1", b"2", b"3"])
        assert result == ["prompt 1", "prompt 2", "prompt 3"]
        assert len(calls) == 3
        assert all(kw["jpgs"] == [img] for kw, img in zip(calls, [b"1", b"2", b"3"]))
        assert "image 2 of 3" in calls[1]["user_text"]

    async def test_tolerates_plural_array_response_shape(self, monkeypatch):
        async def fake_chat_json(**kwargs):
            return {"animations": ["from the array shape"]}
        monkeypatch.setattr(analysis_module, "_chat_json", fake_chat_json)

        result = await generate_i2v_motion_prompts([b"1"])
        assert result == ["from the array shape"]

    async def test_failed_image_yields_empty_slot(self, monkeypatch):
        calls = []
        async def fake_chat_json(**kwargs):
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError("boom")
            return {"animation": f"prompt {len(calls)}"}
        monkeypatch.setattr(analysis_module, "_chat_json", fake_chat_json)

        result = await generate_i2v_motion_prompts([b"1", b"2", b"3"])
        assert result == ["prompt 1", "", "prompt 3"]

    async def test_all_calls_failing_raises(self, monkeypatch):
        async def fake_chat_json(**kwargs):
            raise RuntimeError("ollama down")
        monkeypatch.setattr(analysis_module, "_chat_json", fake_chat_json)

        with pytest.raises(RuntimeError):
            await generate_i2v_motion_prompts([b"1", b"2"])

    async def test_empty_image_list_raises(self):
        with pytest.raises(RuntimeError):
            await generate_i2v_motion_prompts([])

    async def test_reads_prompt_file_without_raising(self):
        from services.ollama.chat import _read_prompt
        text = _read_prompt("video-i2v-motion.md")
        assert text.strip()

    async def test_on_progress_fires_once_per_image_including_failures(self, monkeypatch):
        # The suggest-i2v background job (routers/video.py) reports status via
        # this callback — it must fire for every image, success or failure,
        # so a job never gets stuck mid-progress.
        calls = []
        async def fake_chat_json(**kwargs):
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError("boom")
            return {"animation": f"prompt {len(calls)}"}
        monkeypatch.setattr(analysis_module, "_chat_json", fake_chat_json)

        progress = []
        await generate_i2v_motion_prompts(
            [b"1", b"2", b"3"], on_progress=lambda done, total: progress.append((done, total)),
        )
        assert progress == [(1, 3), (2, 3), (3, 3)]


class TestGenerateMinimaxMotionPrompts:
    """Mirrors TestGenerateI2vMotionPrompts — same per-image fan-out helper,
    different system prompt file (MiniMax H3's audio-aware variant)."""

    async def test_one_call_per_image_with_single_jpg_each(self, monkeypatch):
        calls = []
        async def fake_chat_json(**kwargs):
            calls.append(kwargs)
            return {"animation": f"prompt {len(calls)}"}
        monkeypatch.setattr(analysis_module, "_chat_json", fake_chat_json)

        result = await generate_minimax_motion_prompts([b"1", b"2", b"3"])
        assert result == ["prompt 1", "prompt 2", "prompt 3"]
        assert len(calls) == 3
        assert all(kw["jpgs"] == [img] for kw, img in zip(calls, [b"1", b"2", b"3"]))
        assert "image 2 of 3" in calls[1]["user_text"]

    async def test_uses_minimax_system_prompt_not_wan(self, monkeypatch):
        systems = []
        async def fake_chat_json(**kwargs):
            systems.append(kwargs["system"])
            return {"animation": "ok"}
        monkeypatch.setattr(analysis_module, "_chat_json", fake_chat_json)

        await generate_minimax_motion_prompts([b"1"])
        from services.ollama.chat import _read_prompt
        assert systems[0] == _read_prompt("video-minimax-motion.md")
        assert systems[0] != _read_prompt("video-i2v-motion.md")

    async def test_failed_image_yields_empty_slot(self, monkeypatch):
        calls = []
        async def fake_chat_json(**kwargs):
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError("boom")
            return {"animation": f"prompt {len(calls)}"}
        monkeypatch.setattr(analysis_module, "_chat_json", fake_chat_json)

        result = await generate_minimax_motion_prompts([b"1", b"2", b"3"])
        assert result == ["prompt 1", "", "prompt 3"]

    async def test_all_calls_failing_raises(self, monkeypatch):
        async def fake_chat_json(**kwargs):
            raise RuntimeError("ollama down")
        monkeypatch.setattr(analysis_module, "_chat_json", fake_chat_json)

        with pytest.raises(RuntimeError):
            await generate_minimax_motion_prompts([b"1", b"2"])

    async def test_empty_image_list_raises(self):
        with pytest.raises(RuntimeError):
            await generate_minimax_motion_prompts([])

    async def test_reads_prompt_file_without_raising(self):
        from services.ollama.chat import _read_prompt
        text = _read_prompt("video-minimax-motion.md")
        assert text.strip()

    async def test_multi_sentence_answers_survive_intact(self, monkeypatch):
        # H3 clips run 5-15s, so the prompt is deliberately several sentences
        # plus an Audio: line — truncating to the first sentence (as the
        # retired LTX path did) would drop the whole soundtrack instruction.
        answer = "Rust bleeds down the panel. The drip never stops. Audio: slow ticking in a wide hall."
        async def fake_chat_json(**kwargs):
            return {"animation": answer}
        monkeypatch.setattr(analysis_module, "_chat_json", fake_chat_json)

        result = await generate_minimax_motion_prompts([b"1"])
        assert result == [answer]

    async def test_num_predict_leaves_room_for_the_audio_line(self, monkeypatch):
        seen = []
        async def fake_chat_json(**kwargs):
            seen.append(kwargs["options"])
            return {"animation": "ok"}
        monkeypatch.setattr(analysis_module, "_chat_json", fake_chat_json)

        await generate_minimax_motion_prompts([b"1"])
        assert seen[0]["num_predict"] >= 180


class TestLookSource:
    """The grain pass must never read its own output — otherwise moving the
    strength slider bakes a second pass on top of the first and the picture
    silts up a little more with every adjustment."""

    def _video(self, **kw):
        from core.models import Video
        return Video(filepath="videos/clean.mp4", **kw)

    def test_ignores_its_own_output(self):
        v = self._video(grain_filename="x_grain.mp4", grain_strength=40)
        assert _look_source(v).name == "clean.mp4"

    def test_prefers_the_muxed_variant_so_a_soundtrack_survives(self):
        v = self._video(muxed_filename="x_muxed.mp4")
        assert _look_source(v).name == "x_muxed.mp4"

    def test_regrading_still_reads_the_muxed_variant_not_the_grain(self):
        v = self._video(muxed_filename="x_muxed.mp4", grain_filename="x_grain.mp4")
        assert _look_source(v).name == "x_muxed.mp4"

    def test_falls_back_to_the_original(self):
        assert _look_source(self._video()).name == "clean.mp4"


class TestRenderVersion:
    """Derived renders reuse one filename per video, so the URL has to carry
    a token that changes when the bytes do — /api/video/file sends an ETag but
    no Cache-Control, and browsers then cache heuristically (roughly 10% of the
    file's age) and replay the first render for days without revalidating."""

    def _video(self, tmp_path, monkeypatch, **kw):
        # videos_dir is a derived property; storage_dir is the settable field.
        from core.config import settings as cfg
        from core.models import Video
        (tmp_path / "videos").mkdir(exist_ok=True)
        monkeypatch.setattr(cfg, "storage_dir", tmp_path)
        return Video(filename="clip.mp4", filepath="videos/clip.mp4", **kw)

    def test_original_needs_no_token(self, tmp_path, monkeypatch):
        # Written once at generation and never rewritten.
        v = self._video(tmp_path, monkeypatch)
        assert _render_version(v, "clip.mp4") is None

    def test_token_changes_when_the_render_is_replaced(self, tmp_path, monkeypatch):
        import os
        v = self._video(tmp_path, monkeypatch, grain_filename="g.mp4", grain_strength=30)
        f = tmp_path / "videos" / "g.mp4"
        f.write_bytes(b"first render")
        before = _render_version(v, "g.mp4")

        f.write_bytes(b"second render at a different strength")
        st = f.stat()
        os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000))
        assert _render_version(v, "g.mp4") != before

    def test_token_is_stable_for_an_untouched_render(self, tmp_path, monkeypatch):
        v = self._video(tmp_path, monkeypatch, grain_filename="g.mp4", grain_strength=30)
        (tmp_path / "videos" / "g.mp4").write_bytes(b"render")
        assert _render_version(v, "g.mp4") == _render_version(v, "g.mp4")

    def test_missing_file_degrades_to_no_token(self, tmp_path, monkeypatch):
        v = self._video(tmp_path, monkeypatch, grain_filename="gone.mp4")
        assert _render_version(v, "gone.mp4") is None


class TestGrainRenderingFlag:
    """Re-applying the same strength leaves grain_strength unchanged, so the
    client cannot use it to tell a finished render from one still being
    written — it needs an in-flight signal of its own."""

    def _video(self):
        from core.models import Video
        return Video(id=uuid.UUID("11111111-2222-3333-4444-555555555555"))

    def test_false_when_nothing_is_running(self):
        v = self._video()
        video_module._progress.pop(str(v.id), None)
        assert _is_graining(v) is False

    def test_true_while_a_grain_render_is_in_flight(self):
        v = self._video()
        video_module._progress[str(v.id)] = {"phase": "graining", "message": "…", "pct": 50}
        try:
            assert _is_graining(v) is True
        finally:
            video_module._progress.pop(str(v.id), None)

    def test_other_phases_do_not_count(self):
        v = self._video()
        video_module._progress[str(v.id)] = {"phase": "muxing", "message": "…", "pct": 50}
        try:
            assert _is_graining(v) is False
        finally:
            video_module._progress.pop(str(v.id), None)


class TestUpscaleSource:
    """The upscale pass reads the *cleanest* audio-bearing rendition. Reading
    its own output would restore a reconstruction, and reading the grained file
    would have a restorer treat film grain as detail to sharpen."""

    def _video(self, **kw):
        from core.models import Video
        return Video(filepath="videos/clean.mp4", **kw)

    def test_ignores_its_own_output(self):
        v = self._video(upscale_filename="x_upscale.mp4", upscale_resolution=1080)
        assert _upscale_source(v).name == "clean.mp4"

    def test_never_reads_the_grained_file(self):
        v = self._video(grain_filename="x_grain.mp4", grain_strength=40)
        assert _upscale_source(v).name == "clean.mp4"

    def test_prefers_the_muxed_variant_so_a_soundtrack_survives(self):
        v = self._video(muxed_filename="x_muxed.mp4")
        assert _upscale_source(v).name == "x_muxed.mp4"


class TestGrainSourcePrefersTheUpscale:
    """Grain belongs at the delivery resolution — grading the small render and
    leaving the 1080p one ungrained would also leave _serialize preferring a
    grained file built from the wrong picture."""

    def _video(self, **kw):
        from core.models import Video
        return Video(filepath="videos/clean.mp4", **kw)

    def test_upscale_wins_over_the_muxed_variant(self):
        v = self._video(muxed_filename="x_muxed.mp4", upscale_filename="x_upscale.mp4")
        assert _look_source(v).name == "x_upscale.mp4"

    def test_upscale_wins_over_the_original(self):
        v = self._video(upscale_filename="x_upscale.mp4")
        assert _look_source(v).name == "x_upscale.mp4"


class TestUpscaleRenderingFlag:
    """Re-running at the same resolution leaves upscale_resolution unchanged,
    so the client needs an in-flight signal of its own — see _is_graining."""

    def _video(self):
        from core.models import Video
        return Video(id=uuid.UUID("66666666-7777-8888-9999-000000000000"))

    def test_false_when_nothing_is_running(self):
        v = self._video()
        video_module._progress.pop(str(v.id), None)
        assert _is_upscaling(v) is False

    def test_true_while_an_upscale_is_in_flight(self):
        v = self._video()
        video_module._progress[str(v.id)] = {"phase": "upscaling", "message": "…", "pct": 30}
        try:
            assert _is_upscaling(v) is True
            assert _is_graining(v) is False
        finally:
            video_module._progress.pop(str(v.id), None)


class TestPreflightVram:
    """Evicting Ollama is not proof the card came free: a cold load already
    under way cannot be aborted, allocates VRAM progressively, and is not even
    listed by /api/ps until it finishes. So the submission gate checks the one
    fact that matters — how much is actually free.

    `preflight_vram` itself (services/comfy/vram.py, shared by video and music
    generation) only knows a required byte count and a label; resolving a
    workflow name to that many bytes is each caller's own policy, tested
    separately below for video's `_free_ollama_vram`."""

    GB = 1024 ** 3

    def _devices(self, free_gb, total_gb=16.0):
        return [{
            "name": "cuda:0 NVIDIA GeForce RTX 4060 Ti",
            "vram_free": free_gb * self.GB,
            "vram_total": total_gb * self.GB,
        }]

    def _no_waiting(self, monkeypatch, released: list | None = None):
        """Stub both holders and collapse the wait, so a shortfall fails fast."""
        monkeypatch.setattr(vram_module, "evict_ollama", lambda: _async_value(None))
        monkeypatch.setattr(
            vram_module, "release_comfy_models",
            lambda: _async_value(released.append(1) if released is not None else None),
        )
        monkeypatch.setattr(vram_module, "_VRAM_WAIT_TIMEOUT", 0.0)
        monkeypatch.setattr(vram_module, "_VRAM_WAIT_POLL", 0.0)
        monkeypatch.setattr(vram_module, "_COMFY_FREE_SETTLE", 0.0)

    async def test_passes_when_the_card_is_free(self, monkeypatch):
        monkeypatch.setattr(vram_module, "comfy_devices",
                            lambda: _async_value(self._devices(15.5)))
        await vram_module.preflight_vram(13.5 * self.GB, "minimax_i2v")  # must not raise

    async def test_a_free_card_is_not_disturbed(self, monkeypatch):
        """Unloading costs a cold reload, so it must only happen on a shortfall."""
        released = []
        monkeypatch.setattr(vram_module, "comfy_devices",
                            lambda: _async_value(self._devices(15.5)))
        self._no_waiting(monkeypatch, released)
        await vram_module.preflight_vram(13.5 * self.GB, "minimax_i2v")
        assert released == []

    async def test_a_shortfall_unloads_comfyui_before_waiting(self, monkeypatch):
        """The regression this exists for: an upscale queued right after a
        render found ~3 GB free, and ComfyUI — not Ollama — was holding it, so
        the old loop re-evicted Ollama for 210 s and then gave up."""
        released = []
        monkeypatch.setattr(vram_module, "comfy_devices",
                            lambda: _async_value(self._devices(3.4)))
        self._no_waiting(monkeypatch, released)
        with pytest.raises(RuntimeError, match="still busy"):
            await vram_module.preflight_vram(9.0 * self.GB, "upscale")
        assert released, "ComfyUI was never asked to unload"

    async def test_refuses_a_minimax_run_on_a_half_full_card(self, monkeypatch):
        # 14956 MB staged for the text encoder alone — half a card is not a
        # slow run, it is a CUDA OOM that kills ComfyUI's worker thread.
        monkeypatch.setattr(vram_module, "comfy_devices",
                            lambda: _async_value(self._devices(8.0)))
        self._no_waiting(monkeypatch)
        with pytest.raises(RuntimeError, match="still busy"):
            await vram_module.preflight_vram(13.5 * self.GB, "minimax_i2v")

    async def test_the_same_card_is_fine_for_a_lighter_workflow(self, monkeypatch):
        # Wan and the 3B upscaler do not need the whole card.
        monkeypatch.setattr(vram_module, "comfy_devices",
                            lambda: _async_value(self._devices(10.0)))
        await vram_module.preflight_vram(9.0 * self.GB, "i2v_multi")

    async def test_the_error_names_the_numbers(self, monkeypatch):
        monkeypatch.setattr(vram_module, "comfy_devices",
                            lambda: _async_value(self._devices(2.0)))
        self._no_waiting(monkeypatch)
        with pytest.raises(RuntimeError) as exc:
            await vram_module.preflight_vram(13.5 * self.GB, "minimax_i2v")
        msg = str(exc.value)
        assert "2.1 GB free" in msg and "ollama ps" in msg

    async def test_no_devices_does_not_block_the_job(self, monkeypatch):
        # A ComfyUI build that reports no devices should not make video
        # generation impossible — the gate is a safety net, not a gatekeeper.
        monkeypatch.setattr(vram_module, "comfy_devices", lambda: _async_value([]))
        await vram_module.preflight_vram(13.5 * self.GB, "minimax_i2v")

    async def test_free_ollama_vram_resolves_minimax_requirement(self, monkeypatch):
        """Video's own wrapper picks MiniMax's higher floor by workflow name."""
        calls = []
        monkeypatch.setattr(
            video_module, "free_vram_for",
            lambda required, label: _async_value(calls.append((required, label))),
        )
        await video_module._free_ollama_vram("minimax_i2v")
        assert calls == [(13.5 * self.GB, "minimax_i2v")]

    async def test_free_ollama_vram_falls_back_to_default(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            video_module, "free_vram_for",
            lambda required, label: _async_value(calls.append((required, label))),
        )
        await video_module._free_ollama_vram("upscale")
        assert calls == [(9.0 * self.GB, "upscale")]


async def _async_value(value):
    return value


class TestCancelTitlerWarmup:
    """The startup warm-up is the reason the first MiniMax run after every
    reboot OOMed: during its ~2.5 min cold load the model is not yet resident,
    so evicting Ollama finds an empty card and reports success — and then the
    warm-up loads 5.3 GB straight into ComfyUI's render. Evicting is not enough;
    the loader itself has to be called off first."""

    async def test_no_warmup_running_is_a_no_op(self):
        analysis_module._warm_task = None
        await analysis_module.cancel_titler_warmup()  # must not raise

    async def test_a_finished_warmup_is_left_alone(self):
        async def done():
            return None
        task = asyncio.create_task(done())
        await task
        analysis_module._warm_task = task
        await analysis_module.cancel_titler_warmup()
        assert not task.cancelled()

    async def test_an_in_flight_warmup_is_cancelled_and_awaited(self):
        started = asyncio.Event()

        async def slow_warmup():
            started.set()
            await asyncio.sleep(30)          # stands in for the cold load

        task = asyncio.create_task(slow_warmup())
        await started.wait()
        analysis_module._warm_task = task

        await analysis_module.cancel_titler_warmup()
        # Awaited, not merely signalled: returning while the load is still in
        # flight would let it finish during the render, which is the bug.
        assert task.done()
        assert task.cancelled()

    async def test_a_failing_warmup_does_not_propagate(self):
        # The caller is about to render; a broken warm-up must not take the
        # video job down with it.
        async def boom():
            raise RuntimeError("ollama exploded")

        task = asyncio.create_task(boom())
        await asyncio.sleep(0)
        analysis_module._warm_task = task
        await analysis_module.cancel_titler_warmup()


class TestSongAndInterpolationAreMutuallyExclusive:
    """Interpolation stretches whatever audio the file carries — right for a
    model's own generated track, destructive for music. The guard has to hold
    in BOTH directions, because attaching a song rewrites the upscale's source
    and triggers a re-render that would run the stretch over the song."""

    def _video(self, **kw):
        from core.models import Video
        return Video(
            id=uuid.uuid4(), status="done", filename="clip.mp4",
            filepath="videos/clip.mp4", **kw,
        )

    def test_interpolation_is_refused_when_a_song_is_attached(self):
        from fastapi import HTTPException
        v = self._video(soundtrack_song_id=uuid.uuid4())
        with pytest.raises(HTTPException) as exc:
            _validate_upscale_target(v, 1080, 3)
        assert exc.value.status_code == 409

    def test_a_target_rate_makes_interpolation_safe_for_a_song(self):
        # What the guard is really about is the *stretch*, not the
        # interpolation: a target rate holds the clip's length, so the music
        # stays exactly where it was.
        v = self._video(soundtrack_song_id=uuid.uuid4())
        _validate_upscale_target(v, 1080, 3, 24)  # must not raise

    def test_a_plain_upscale_is_still_fine_with_a_song(self):
        v = self._video(soundtrack_song_id=uuid.uuid4())
        _validate_upscale_target(v, 1080, 1)  # must not raise

    def test_interpolation_is_fine_without_a_song(self):
        _validate_upscale_target(self._video(), 1080, 3)  # must not raise

    def test_unsupported_multipliers_are_rejected_outright(self):
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc:
            _validate_upscale_target(self._video(), 1080, 5)
        assert exc.value.status_code == 422


class TestUpscaleOutputDimensions:
    def test_minimax_landscape_reaches_1080p(self):
        assert output_dimensions(864, 480, 1080) == (1944, 1080)

    def test_minimax_portrait_reaches_1080_on_the_short_edge(self):
        assert output_dimensions(480, 864, 1080) == (1080, 1944)

    def test_aspect_ratio_is_preserved(self):
        w, h = output_dimensions(864, 480, 720)
        assert abs(w / h - 864 / 480) < 0.01

    def test_a_source_already_at_the_target_is_left_alone(self):
        # SEEDVR2 would happily downscale; this pass exists to add pixels.
        assert output_dimensions(1920, 1080, 720) == (1920, 1080)

    def test_dimensions_stay_even_for_yuv420p(self):
        w, h = output_dimensions(853, 481, 1080)
        assert w % 2 == 0 and h % 2 == 0


class TestClampResolution:
    def test_garbage_reads_as_the_default(self):
        assert clamp_resolution(None) == 1080
        assert clamp_resolution("nonsense") == 1080

    def test_out_of_range_values_are_clamped(self):
        assert clamp_resolution(99) == 480
        assert clamp_resolution(4000) == 1440


class TestEstimateSeconds:
    def test_matches_the_measured_reference_run(self):
        # 56 frames at 24 fps = 2.33 s of source → 1944×1080 took 438 s.
        est = estimate_seconds(2.333, 1944, 1080)
        assert 330 <= est <= 550

    def test_scales_with_duration(self):
        short = estimate_seconds(2.0, 1944, 1080)
        long = estimate_seconds(8.0, 1944, 1080)
        assert 3.5 < long / short < 4.5

    def test_unknown_duration_yields_no_estimate(self):
        # probe_video_duration returns 0.0 when ffprobe fails; a fabricated
        # number would be worse than none, since the UI shows it as a promise.
        assert estimate_seconds(0.0, 1944, 1080) == 0


class TestBuildUpscaleWorkflow:
    def _build(self, **kw):
        from pathlib import Path
        kw.setdefault("resolution", 1080)
        kw.setdefault("filename_prefix", "artrium_up_test")
        kw.setdefault("has_audio", True)
        return build_upscale_workflow(Path("D:/storage/videos/clip.mp4"), **kw)

    def test_reads_the_source_by_absolute_path(self):
        wf, _ = self._build()
        assert wf["sv_load"]["inputs"]["video"].endswith("clip.mp4")

    def test_frame_rate_comes_from_the_file_not_a_constant(self):
        # A RIFE'd clip's stored fps describes the model's rate, not the
        # file's — taking it from the row would desync every interpolated clip.
        wf, save = self._build()
        assert wf[save]["inputs"]["frame_rate"] == ["sv_info", 0]
        assert wf["sv_info"]["inputs"]["video_info"] == ["sv_load", 3]

    def test_audio_bypasses_the_model_entirely(self):
        wf, save = self._build()
        assert wf[save]["inputs"]["audio"] == ["sv_load", 2]

    def test_a_silent_source_leaves_the_audio_slot_unconnected(self):
        # VHS raises when asked to extract audio from a stream-less file, so a
        # silent clip must not be wired up at all.
        wf, save = self._build(has_audio=False)
        assert "audio" not in wf[save]["inputs"]

    def test_batch_size_follows_the_nodes_4n_plus_1_rule(self):
        wf, _ = self._build()
        assert (wf["sv_up"]["inputs"]["batch_size"] - 1) % 4 == 0

    def test_vae_tiling_is_on_because_batch_5_ooms_without_it(self):
        wf, _ = self._build()
        vae = wf["sv_vae"]["inputs"]
        assert vae["encode_tiled"] is True and vae["decode_tiled"] is True

    def test_runs_on_the_first_gpu_only(self):
        # The 2070 (sm_75) has no bf16 conv3d kernel, which the VAE needs
        # whatever the checkpoint dtype — cuda:1 fails outright.
        wf, _ = self._build()
        assert wf["sv_dit"]["inputs"]["device"] == "cuda:0"
        assert wf["sv_vae"]["inputs"]["device"] == "cuda:0"

    def test_resolution_is_clamped_on_the_way_in(self):
        wf, _ = self._build(resolution=99999)
        assert wf["sv_up"]["inputs"]["resolution"] == 1440

    def test_muxer_never_trims_the_video_back_to_the_audio(self):
        # A RIFE'd source is legitimately longer than its native audio track.
        wf, save = self._build()
        assert wf[save]["inputs"]["trim_to_audio"] is False

    def test_every_link_points_at_a_node_that_exists(self):
        wf, _ = self._build()
        for node in wf.values():
            for value in node["inputs"].values():
                if isinstance(value, list) and value and isinstance(value[0], str):
                    assert value[0] in wf, f"dangling link to {value[0]}"

    def test_no_rife_node_at_all_when_interpolation_is_off(self):
        wf, save = self._build(rife_multiplier=1)
        assert "sv_rife" not in wf
        assert wf[save]["inputs"]["images"] == ["sv_up", 0]

    def test_rife_sits_after_the_restore_not_before_it(self):
        # The entire point of the ordering: SEEDVR2 restores only the source's
        # real frames, so its cost stays independent of the multiplier.
        wf, save = self._build(rife_multiplier=3)
        assert wf["sv_rife"]["inputs"]["frames"] == ["sv_up", 0]
        assert wf["sv_up"]["inputs"]["image"] == ["sv_load", 0]
        assert wf[save]["inputs"]["images"] == ["sv_rife", 0]

    def test_interpolated_output_still_links_every_node(self):
        wf, _ = self._build(rife_multiplier=4)
        for node in wf.values():
            for value in node["inputs"].values():
                if isinstance(value, list) and value and isinstance(value[0], str):
                    assert value[0] in wf, f"dangling link to {value[0]}"

    def test_an_unsupported_multiplier_falls_back_to_off(self):
        wf, _ = self._build(rife_multiplier=7)
        assert "sv_rife" not in wf

    def test_frame_rate_is_untouched_by_interpolation(self):
        # RIFE lengthens real time by multiplying frames at a fixed rate;
        # raising frame_rate too would keep the duration and only add
        # smoothness, which is not what the generation path does either.
        wf, save = self._build(rife_multiplier=3)
        assert wf[save]["inputs"]["frame_rate"] == ["sv_info", 0]


class TestUpscaleRifeClamp:
    def test_garbage_reads_as_off(self):
        assert clamp_rife(None) == 1
        assert clamp_rife("nonsense") == 1

    def test_unsupported_factors_read_as_off(self):
        # Off is the safe fallback: it never silently changes the clip length.
        assert clamp_rife(0) == 1
        assert clamp_rife(5) == 1

    def test_supported_factors_pass_through(self):
        assert [clamp_rife(m) for m in (1, 2, 3, 4)] == [1, 2, 3, 4]


class TestEstimateWithInterpolation:
    """The estimate has to show that interpolation is nearly free here — that
    is the user-visible consequence of running RIFE after the restore."""

    def test_interpolation_barely_moves_the_estimate(self):
        base = estimate_seconds(5.0, 1944, 1080, 1)
        triple = estimate_seconds(5.0, 1944, 1080, 3)
        assert triple > base                      # not free
        assert triple < base * 1.2                # but nowhere near 3x

    def test_matches_the_measured_rife_run(self):
        # 124 frames at 24 fps = 5.17 s, 2.09 MPx, RIFE 3x measured at 50 s.
        rife_only = (estimate_seconds(5.17, 1944, 1080, 3)
                     - estimate_seconds(5.17, 1944, 1080, 1))
        assert 35 <= rife_only <= 70


class TestMotionPromptRejection:
    """The small titler VLM (qwen2.5vl:3b) under a long system prompt returns
    one of the worked examples verbatim, or repeats one answer across every
    image. Either way the prompt describes a different picture than the one
    it is attached to, which is what makes the video model drift and cut mid-clip."""

    async def test_example_copied_from_system_prompt_is_retried(self, monkeypatch):
        from services.ollama.chat import _read_prompt

        # An 8-word run lifted straight out of the shipped system prompt.
        leaked = " ".join(_read_prompt("video-minimax-motion.md").split()[20:40])
        answers = iter([leaked, "The rope frays slowly, fibres ticking, the frame static."])
        async def fake_chat_json(**kwargs):
            return {"animation": next(answers)}
        monkeypatch.setattr(analysis_module, "_chat_json", fake_chat_json)

        result = await generate_minimax_motion_prompts([b"1"])
        assert result == ["The rope frays slowly, fibres ticking, the frame static."]

    async def test_prompt_repeated_across_images_is_retried(self, monkeypatch):
        dupe = ("The rust-red paint ridge lifts off the canvas in fine curling "
                "threads, rising slowly, the camera orbiting gently.")
        answers = iter([dupe, dupe, "The glass sags inward with a low groan, frame static."])
        async def fake_chat_json(**kwargs):
            return {"animation": next(answers)}
        monkeypatch.setattr(analysis_module, "_chat_json", fake_chat_json)

        result = await generate_minimax_motion_prompts([b"1", b"2"])
        assert result[0] == dupe
        assert result[1] == "The glass sags inward with a low groan, frame static."

    async def test_gives_up_after_max_attempts_rather_than_blanking(self, monkeypatch):
        # A rejected prompt the user can edit beats an empty textarea.
        dupe = ("The rust-red paint ridge lifts off the canvas in fine curling "
                "threads, rising slowly, the camera orbiting gently.")
        async def fake_chat_json(**kwargs):
            return {"animation": dupe}
        monkeypatch.setattr(analysis_module, "_chat_json", fake_chat_json)

        result = await generate_minimax_motion_prompts([b"1", b"2"])
        assert result == [dupe, dupe]

    async def test_distinct_prompts_are_not_retried(self, monkeypatch):
        calls = []
        async def fake_chat_json(**kwargs):
            calls.append(1)
            return {"animation": f"The number {len(calls)} object drifts upward with a faint hum."}
        monkeypatch.setattr(analysis_module, "_chat_json", fake_chat_json)

        await generate_minimax_motion_prompts([b"1", b"2", b"3"])
        assert len(calls) == 3  # no wasted retries on good output


class TestOneShotEnforcement:
    """One unbroken shot is what stops the video model cutting mid-clip, so
    shot-break language is rejected in code rather than trusted to the system
    prompt. Sentence *count* is deliberately not policed: an H3 clip runs
    5-15s and needs more than one sentence to fill."""

    async def test_camera_revealing_new_content_is_retried(self, monkeypatch):
        # A camera move that brings something new into frame reads to the
        # video model as a cut to a different view.
        answers = iter([
            "The nail drips, the camera tilting up to reveal the rusted wall behind it.",
            "The nail drips slowly, a faint metallic tick, the frame static.",
        ])
        async def fake_chat_json(**kwargs):
            return {"animation": next(answers)}
        monkeypatch.setattr(analysis_module, "_chat_json", fake_chat_json)

        result = await generate_minimax_motion_prompts([b"1"])
        assert result == ["The nail drips slowly, a faint metallic tick, the frame static."]

    async def test_shot_break_retry_gets_a_targeted_nudge(self, monkeypatch):
        texts = []
        answers = iter([
            "The rope frays, then suddenly the light changes.",
            "The rope frays slowly, fibres ticking, the frame static.",
        ])
        async def fake_chat_json(**kwargs):
            texts.append(kwargs["user_text"])
            return {"animation": next(answers)}
        monkeypatch.setattr(analysis_module, "_chat_json", fake_chat_json)

        await generate_minimax_motion_prompts([b"1"])
        assert "reveal something new" in texts[1]
        assert "reused wording" not in texts[1]

    async def test_wan_variant_keeps_its_own_language_rules(self, monkeypatch):
        # Wan2.2's prompt file is tuned separately; the shot-break ban is opt-in
        # per model family and must not silently apply to it ("then" here would
        # be rejected under the MiniMax rules).
        multi = ("The paint drips, then the camera pans. "
                 "A second sentence survives here.")
        calls = []
        async def fake_chat_json(**kwargs):
            calls.append(1)
            return {"animation": multi}
        monkeypatch.setattr(analysis_module, "_chat_json", fake_chat_json)

        result = await generate_i2v_motion_prompts([b"1"])
        assert result == [multi]
        assert len(calls) == 1  # not retried


class TestSalvageTruncatedAnimation:
    """num_predict truncates JSON mid-string rather than shortening the
    answer, so the cap is only safe with this repair in place."""

    def test_recovers_sentence_cut_off_mid_string(self):
        raw = '{"animation": "The clock face keeps sagging, a low metallic gro'
        repaired = analysis_module._salvage_truncated_animation(raw)
        import json as _json
        assert _json.loads(repaired)["animation"] == "The clock face keeps sagging, a low metallic."

    def test_keeps_complete_sentence_when_one_exists(self):
        raw = '{"animation": "The rope frays slowly. A second half-written'
        repaired = analysis_module._salvage_truncated_animation(raw)
        import json as _json
        assert _json.loads(repaired)["animation"] == "The rope frays slowly."

    def test_leaves_properly_closed_json_alone(self):
        raw = '{"animation": "The rope frays slowly.", "extra": bad}'
        assert analysis_module._salvage_truncated_animation(raw) == raw

    def test_leaves_unrelated_content_alone(self):
        raw = "not json at all"
        assert analysis_module._salvage_truncated_animation(raw) == raw

    def test_handles_escaped_quote_inside_the_string(self):
        raw = '{"animation": "The figure says \\"go\\" as the fabric ripp'
        repaired = analysis_module._salvage_truncated_animation(raw)
        import json as _json
        assert _json.loads(repaired)["animation"] == 'The figure says "go" as the fabric.'


class TestWanFrameRate:
    """Wan 2.2 animates at 16 fps. RIFE multiplies the frames, so the container
    rate has to rise with it — otherwise the extra frames stretch the clip
    instead of smoothing it, and every render comes out in slow motion. This is
    the same rule services/video/upscale.py::plan_frame_rate states for the
    upscale path; the generation path used to write a flat 24 and play 16 fps
    content at 8."""

    @pytest.mark.parametrize("rife,expected", [(1, 16), (2, 32), (3, 48), (4, 64)])
    def test_rate_is_native_times_rife(self, rife, expected):
        assert wan_output_fps(rife) == expected

    def test_out_of_range_multipliers_are_clamped_not_multiplied_out(self):
        assert wan_output_fps(0) == WAN_NATIVE_FPS
        assert wan_output_fps(None) == WAN_NATIVE_FPS
        assert wan_output_fps(9) == WAN_NATIVE_FPS * 4

    @pytest.mark.parametrize("rife", [2, 3, 4])
    def test_i2v_save_node_uses_it(self, rife):
        wf, save_id = _build_i2v_single_workflow(
            "img.png", "prompt", 49, 960, 960, "prefix", rife, False,
        )
        assert wf[save_id]["inputs"]["frame_rate"] == WAN_NATIVE_FPS * rife

    @pytest.mark.parametrize("rife", [2, 3, 4])
    def test_flf2v_save_node_uses_it(self, rife):
        wf, save_id = _build_flf2v_single_workflow(
            "start.png", "end.png", "prompt", 49, 960, 960, "prefix", rife,
        )
        assert wf[save_id]["inputs"]["frame_rate"] == WAN_NATIVE_FPS * rife

    def test_the_clip_keeps_its_duration(self):
        # 49 frames is 3.06 s of sampled motion at any RIFE setting. The bug
        # this asserts against made it 3.06 x rife / 24 seconds instead.
        frames = 49
        for rife in (2, 3, 4):
            wf, save_id = _build_flf2v_single_workflow(
                "a.png", "b.png", "p", frames, 960, 960, "prefix", rife,
            )
            rate = wf[save_id]["inputs"]["frame_rate"]
            assert abs((frames * rife) / rate - frames / WAN_NATIVE_FPS) < 1e-9


class TestFlf2vExpertPair:
    """Wan 2.2 shipped no first-last-frame checkpoint. The i2v pair treats a
    pinned end frame as out-of-distribution and cross-fades to satisfy it;
    Wan2.2-Fun-InP was trained on start+end and is the same graph with two
    different files in the loaders."""

    def _unets(self, **kw):
        wf, _ = _build_flf2v_single_workflow(
            "start.png", "end.png", "prompt", 49, 960, 960, "prefix", 3, **kw
        )
        return wf["t0_unet_h"]["inputs"]["unet_name"], wf["t0_unet_l"]["inputs"]["unet_name"]

    def test_defaults_to_the_i2v_pair(self):
        assert self._unets() == (_UNET_HIGH, _UNET_LOW)

    def test_fun_inp_swaps_both_experts(self):
        assert self._unets(fun_inp=True) == (_UNET_FUN_HIGH, _UNET_FUN_LOW)

    def test_nothing_else_about_the_graph_changes(self):
        plain, _ = _build_flf2v_single_workflow(
            "start.png", "end.png", "prompt", 49, 960, 960, "prefix", 3,
        )
        fun, _ = _build_flf2v_single_workflow(
            "start.png", "end.png", "prompt", 49, 960, 960, "prefix", 3, fun_inp=True,
        )
        assert set(plain) == set(fun)
        differing = {k for k in plain if plain[k] != fun[k]}
        # Seeds are random per build, so the sampler nodes differ; the point is
        # that no *structural* node moved.
        assert differing <= {"t0_unet_h", "t0_unet_l", "t0_ks_h"}

    def test_i2v_workflow_is_untouched_by_the_transition_swap(self):
        wf, _ = _build_i2v_single_workflow(
            "img.png", "prompt", 49, 960, 960, "prefix", 3, False,
        )
        assert wf["s0_unet_h"]["inputs"]["unet_name"] == _UNET_HIGH
        assert wf["s0_unet_l"]["inputs"]["unet_name"] == _UNET_LOW


class TestMinimaxFirstLastFrame:
    """MiniMax H3's checkpoint is the fl2va one — first-last-to-video-audio.
    MiniMaxH3ImageToVideo carries the `last_frame` input for it; this builder
    simply never filled it in."""

    def _cond(self, wf):
        return next(n for n in wf.values() if n["class_type"] == "MiniMaxH3ImageToVideo")

    def test_no_end_frame_by_default(self):
        wf, _, _ = _build_minimax_single_workflow(
            "img.png", "a prompt", 124, 768, 768, "prefix",
        )
        assert "last_frame" not in self._cond(wf)["inputs"]
        assert "mmx_load_end" not in wf

    def test_end_frame_is_loaded_scaled_and_wired(self):
        wf, _, _ = _build_minimax_single_workflow(
            "start.png", "a prompt", 124, 768, 768, "prefix",
            end_comfy_filename="end.png",
        )
        assert wf["mmx_load_end"]["inputs"]["image"] == "end.png"
        assert self._cond(wf)["inputs"]["last_frame"] == ["mmx_scale_end", 0]

    def test_both_key_frames_reach_the_canvas_the_same_way(self):
        # The node stretches first_frame and centre-crops last_frame. Two
        # pictures arriving by different rules would not line up, and a
        # transition is entirely about them lining up.
        wf, _, _ = _build_minimax_single_workflow(
            "start.png", "a prompt", 124, 1344, 768, "prefix",
            end_comfy_filename="end.png",
        )
        assert wf["mmx_scale"]["inputs"] == wf["mmx_scale_end"]["inputs"] | {
            "image": ["mmx_load", 0]
        }

    def test_first_frame_still_points_at_the_start_picture(self):
        wf, _, _ = _build_minimax_single_workflow(
            "start.png", "p", 124, 768, 768, "prefix", end_comfy_filename="end.png",
        )
        assert self._cond(wf)["inputs"]["first_frame"] == ["mmx_scale", 0]
        assert wf["mmx_load"]["inputs"]["image"] == "start.png"

    def test_rife_and_the_end_frame_coexist(self):
        wf, save_id, _ = _build_minimax_single_workflow(
            "start.png", "p", 124, 768, 768, "prefix", 3, end_comfy_filename="end.png",
        )
        assert wf["mmx_rife"]["inputs"]["multiplier"] == 3
        assert wf[save_id]["inputs"]["images"] == ["mmx_rife", 0]
        assert self._cond(wf)["inputs"]["last_frame"] == ["mmx_scale_end", 0]


class TestWanTemporalGrid:
    """Wan's VAE compresses time by 4 with the first frame standing alone, so a
    clip is 1 + 4n frames — every Wan node in ComfyUI declares step 4 for
    `length` and its own UI cannot express anything else. This tool's number
    field used step 1 and let 48 through on a real job."""

    @pytest.mark.parametrize("given,snapped", [
        (5, 5), (9, 9), (33, 33), (49, 49), (81, 81),      # already on the grid
        (46, 49), (47, 49), (48, 49),                       # snapped up, never down
        (50, 53), (80, 81),
    ])
    def test_snaps_up_onto_the_grid(self, given, snapped):
        assert align_wan_length(given) == snapped

    @pytest.mark.parametrize("n", list(range(5, 82)))
    def test_every_result_is_4n_plus_1_and_never_shorter(self, n):
        out = align_wan_length(n)
        assert (out - 1) % 4 == 0
        assert out >= n

    def test_floor_protects_against_zero_and_none(self):
        assert align_wan_length(0) == 5
        assert align_wan_length(None) == 5

    def test_it_is_idempotent(self):
        for n in range(5, 82):
            assert align_wan_length(align_wan_length(n)) == align_wan_length(n)

    def test_flf2v_graph_renders_the_snapped_length(self):
        wf, _ = _build_flf2v_single_workflow(
            "a.png", "b.png", "p", 48, 960, 960, "prefix", 3,
        )
        assert wf["t0_flf2v"]["inputs"]["length"] == 49

    def test_i2v_graph_renders_the_snapped_length(self):
        wf, _ = _build_i2v_single_workflow(
            "img.png", "p", 48, 960, 960, "prefix", 3, False,
        )
        assert wf["s0_i2v"]["inputs"]["length"] == 49


class TestGuidanceFollowsTheMotionDial:
    """cfg 1 is what a distill LoRA requires, not a house style. The motion dial
    can take the distill off the high-noise expert, and at that point cfg 1
    leaves a plain Wan expert running guidance-free: the prompt barely steers
    and the negative prompt — which ends in 慢动作 (slow motion) and
    静止不动的画面 (static image) — is never scored at all. On a transition that
    is the difference between a transformation and a cross-fade."""

    def _flf(self, lora_high, steps=10):
        return _build_flf2v_single_workflow(
            "a.png", "b.png", "p", 49, 960, 960, "x", 3,
            steps=steps, lora_high=lora_high,
        )[0]

    @pytest.mark.parametrize("lora_high", [0.1, 0.3, 0.5, 1.0])
    def test_any_distill_keeps_the_guidance_free_path(self, lora_high):
        assert wan_cfg_high(lora_high) == 1
        assert self._flf(lora_high)["t0_ks_h"]["inputs"]["cfg"] == 1

    def test_no_distill_gets_real_guidance(self):
        assert wan_cfg_high(0.0) > 1
        assert self._flf(0.0)["t0_ks_h"]["inputs"]["cfg"] == 3.0

    @pytest.mark.parametrize("lora_high", [0.0, 0.5, 1.0])
    def test_the_low_noise_expert_never_gets_guidance(self, lora_high):
        # It keeps its distill at full strength in every case, so cfg 1 is
        # right for it in every case.
        assert self._flf(lora_high)["t0_ks_l"]["inputs"]["cfg"] == 1

    def test_the_i2v_builder_follows_the_same_rule(self):
        wf = _build_i2v_single_workflow(
            "img.png", "p", 49, 960, 960, "x", 3, False, steps=10, lora_high=0.0,
        )[0]
        assert wf["s0_ks_h"]["inputs"]["cfg"] == 3.0
        assert wf["s0_ks_l"]["inputs"]["cfg"] == 1

    def test_guided_steps_are_counted_as_two_evaluations(self):
        # Only the high-noise portion is guided, so the surcharge is the split.
        assert wan_model_evals(10, 0.5) == 10
        assert wan_model_evals(10, 0.0) == 10 + 3

    def test_the_estimate_moves_with_it(self):
        cheap = estimate_wan_seconds(960, 960, 49, 10, lora_high=0.5)
        guided = estimate_wan_seconds(960, 960, 49, 10, lora_high=0.0)
        assert guided > cheap
        # Bounded: a guided high-noise pass is far from doubling the clip.
        assert guided < cheap * 1.5

    def test_the_deadline_covers_the_guided_clip(self):
        assert wan_poll_timeout(960, 960, 49, 10, 0.0) >= wan_poll_timeout(960, 960, 49, 10, 0.5)


class TestHighNoiseStyleLoraSlot:
    """FLF2V infers the path between two pinned frames, so with two unrelated
    pictures it infers a blend — there is no trajectory in its prior connecting
    them. The community's answer is not a setting but a LoRA, and every one of
    them is trained for the *high-noise* expert, because that is the expert that
    decides what the clip becomes."""

    def _chain(self, wf, prefix):
        """LoRA file names feeding the high-noise sampler, nearest expert first."""
        cur, out = wf[f"{prefix}samp_h"]["inputs"]["model"][0], []
        while cur in wf and "model" in wf[cur]["inputs"]:
            node = wf[cur]
            if node["class_type"] == "LoraLoaderModelOnly":
                out.append((node["inputs"]["lora_name"], node["inputs"]["strength_model"]))
            cur = node["inputs"]["model"][0]
        return out

    def _flf(self, **kw):
        return _build_flf2v_single_workflow(
            "a.png", "b.png", "p", 49, 960, 960, "x", 3, steps=10, **kw
        )[0]

    def test_absent_by_default(self):
        wf = self._flf(lora_high=0.5)
        assert "t0_lora_s" not in wf
        assert self._chain(wf, "t0_") == [(_LORA_HIGH_NAME, 0.5)]

    def test_chains_after_the_distill_not_instead_of_it(self):
        wf = self._flf(lora_high=0.5, style_lora="morph.safetensors")
        # Nearest the sampler is the style LoRA; the distill is still under it.
        assert self._chain(wf, "t0_") == [("morph.safetensors", 1.0), (_LORA_HIGH_NAME, 0.5)]

    def test_works_with_the_distill_off(self):
        wf = self._flf(lora_high=0.0, style_lora="morph.safetensors")
        assert self._chain(wf, "t0_") == [("morph.safetensors", 1.0)]
        # And the expert is still guided, because the distill is what cfg 1 was for.
        assert wf["t0_ks_h"]["inputs"]["cfg"] == 3.0

    def test_never_touches_the_low_noise_expert(self):
        wf = self._flf(lora_high=0.0, style_lora="morph.safetensors")
        cur, names = wf["t0_samp_l"]["inputs"]["model"][0], []
        while cur in wf and "model" in wf[cur]["inputs"]:
            if wf[cur]["class_type"] == "LoraLoaderModelOnly":
                names.append(wf[cur]["inputs"]["lora_name"])
            cur = wf[cur]["inputs"]["model"][0]
        assert names == [_LORA_LOW_NAME]

    @pytest.mark.parametrize("given,expected", [
        (None, 1.0), (0.0, 0.0), (0.75, 0.75), (1.0, 1.0), (2.0, 2.0),
        (3.5, 2.0), (-1.0, 0.0),
    ])
    def test_strength_is_clamped_to_a_range_wan_survives(self, given, expected):
        assert clamp_style_strength(given) == expected

    def test_the_i2v_builder_has_the_same_slot(self):
        wf = _build_i2v_single_workflow(
            "img.png", "p", 49, 960, 960, "x", 3, False,
            steps=10, lora_high=0.0, style_lora="morph.safetensors",
        )[0]
        assert self._chain(wf, "s0_") == [("morph.safetensors", 1.0)]


class TestTransitionLoraShelf:
    """Which of ComfyUI's LoRAs the slot is allowed to offer.

    The folder is shared with every other model family in the project, and a
    LoRA built for one of them is not an error: ComfyUI loads it, no key
    matches Wan's, and the clip comes out exactly as if nothing had been
    selected. The mistake is invisible at the one place it could be noticed,
    which is why the slot is an allow list rather than a directory listing."""

    # A real listing of the loras folder on this machine, trimmed.
    INSTALLED = {
        "Spatial Magic_V2.safetensors",
        "high_screen_flood.safetensors",
        "mh3-lys.safetensors",                 # MiniMax H3, not Wan
        "lcm-lora-sdxl.safetensors",
        "Qwen-Image-Lightning-8steps-V1.0.safetensors",
        "ltx-2.3-22b-distilled-lora-384.safetensors",
        "zImageT_zidiusArt_melancholy.safetensors",
        "Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors",
    }

    def _labels(self, names):
        return [e["label"] for e in offered_transition_loras(names)]

    def test_only_the_transition_loras_are_offered(self):
        assert self._labels(self.INSTALLED) == ["Spatial Magic", "Screen Flood"]

    def test_a_wan_lora_that_is_not_a_transition_lora_stays_off(self):
        # It is a Wan file and it would load — it just does something else.
        assert "Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32" not in str(
            offered_transition_loras(self.INSTALLED)
        )

    def test_an_entry_whose_file_is_missing_is_simply_absent(self):
        # ComfyUI rejects the whole prompt over an unknown lora_name, so a
        # shelf entry with no file behind it would kill the job, not the LoRA.
        assert "Claymation Transformation" not in self._labels(self.INSTALLED)

    def test_it_reappears_when_the_file_lands(self):
        names = self.INSTALLED | {"Claymation_Transformation_v1.safetensors"}
        assert "Claymation Transformation" in self._labels(names)

    @pytest.mark.parametrize("name", [
        "Spatial Magic_V3.safetensors",
        "wan/Spatial Magic_V2.safetensors",
        "spatial-magic.safetensors",
    ])
    def test_a_renamed_revision_still_matches(self, name):
        # These arrive under whatever the uploader called that revision.
        assert self._labels({name}) == ["Spatial Magic"]

    def test_an_unknown_transition_lora_names_itself_on(self):
        # The escape hatch: downloading one should not require a code change.
        offers = offered_transition_loras({"wan22_metamorph_hi.safetensors"})
        assert [o["label"] for o in offers] == ["wan22_metamorph_hi"]
        assert offers[0]["trigger"] == ""       # nobody here knows what it wants

    def test_the_file_is_reported_verbatim(self):
        # It goes straight into the graph's lora_name, so it has to be the name
        # ComfyUI gave, not the label or a normalised form of it.
        assert offered_transition_loras(self.INSTALLED)[0]["file"] == "Spatial Magic_V2.safetensors"


class TestLoraTriggerReachesThePrompt:
    """A trigger-word LoRA that never sees its trigger behaves precisely like
    no LoRA at all — the same silent nothing as loading one for the wrong
    architecture. The trigger is a property of the selected file, not a
    creative decision, so the builder puts it in."""

    SPATIAL = "Spatial Magic_V2.safetensors"

    def test_the_trigger_leads_the_prompt(self):
        out = with_lora_trigger("The cliff cracks open.", self.SPATIAL)
        assert out == "kjmf magic The cliff cracks open."

    def test_it_is_not_added_twice(self):
        typed = "kjmf magic the cliff cracks open"
        assert with_lora_trigger(typed, self.SPATIAL) == typed

    def test_a_lora_without_a_known_trigger_leaves_the_prompt_alone(self):
        assert with_lora_trigger("p", "high_screen_flood.safetensors") == "p"
        assert trigger_for("high_screen_flood.safetensors") == ""

    def test_no_lora_leaves_the_prompt_alone(self):
        assert with_lora_trigger("p", None) == "p"

    def _pos(self, wf, prefix):
        return wf[f"{prefix}pos"]["inputs"]["text"]

    def test_the_transition_builder_encodes_it(self):
        wf = _build_flf2v_single_workflow(
            "a.png", "b.png", "the cliff opens", 49, 960, 960, "x", 3,
            steps=10, lora_high=0.5, style_lora=self.SPATIAL,
        )[0]
        assert self._pos(wf, "t0_") == "kjmf magic the cliff opens"

    def test_the_i2v_builder_encodes_it_too(self):
        wf = _build_i2v_single_workflow(
            "img.png", "the cliff opens", 49, 960, 960, "x", 3, False,
            steps=10, lora_high=0.5, style_lora=self.SPATIAL,
        )[0]
        assert self._pos(wf, "s0_") == "kjmf magic the cliff opens"

    def test_the_negative_prompt_never_gets_it(self):
        wf = _build_flf2v_single_workflow(
            "a.png", "b.png", "p", 49, 960, 960, "x", 3,
            steps=10, lora_high=0.5, style_lora=self.SPATIAL,
        )[0]
        assert "kjmf" not in wf["t0_neg"]["inputs"]["text"]

    def test_an_unselected_slot_leaves_the_prompt_verbatim(self):
        wf = _build_flf2v_single_workflow(
            "a.png", "b.png", "the cliff opens", 49, 960, 960, "x", 3,
            steps=10, lora_high=0.5,
        )[0]
        assert self._pos(wf, "t0_") == "the cliff opens"


@pytest.mark.asyncio
class TestMinimaxTransitionWriter:
    """Two stages, because one was measurably not enough. The single vision
    call the Wan writer uses asks a 3B VLM to look at N images AND invent a
    constrained, audio-carrying prompt; it produced one Audio: line in four,
    banned cross-fade verbs in half, and once a destination that was not in
    the picture at all. So the VLM now only DESCRIBES, and the instruct model
    WRITES from those descriptions."""

    def _spy(self, monkeypatch, *, n_prompts=1):
        """Record every _chat_json call and answer each stage in its own shape."""
        calls = []
        async def fake_chat_json(**kwargs):
            calls.append(kwargs)
            if kwargs["label"] == "describe_key_frame":
                return {"description": "a medium portrait, blue background"}
            return {"transitions": ["x"] * n_prompts}
        monkeypatch.setattr(analysis_module, "_chat_json", fake_chat_json)
        return calls

    async def test_it_describes_every_key_frame_then_writes_once(self, monkeypatch):
        calls = self._spy(monkeypatch, n_prompts=3)
        await generate_minimax_transition_prompts([b"1", b"2", b"3", b"4"])
        labels = [c["label"] for c in calls]
        assert labels == ["describe_key_frame"] * 4 + ["generate_minimax_transition_prompts"]

    async def test_the_describing_is_done_by_the_vision_model(self, monkeypatch):
        calls = self._spy(monkeypatch)
        await generate_minimax_transition_prompts([b"1", b"2"])
        describes = [c for c in calls if c["label"] == "describe_key_frame"]
        assert all(c["model"] == video_settings.ollama_titler_model for c in describes)
        assert all(c["jpgs"] for c in describes)      # it needs to see them

    async def test_the_writing_is_done_by_the_instruct_model_without_images(self, monkeypatch):
        calls = self._spy(monkeypatch)
        await generate_minimax_transition_prompts([b"1", b"2"])
        write = calls[-1]
        assert write["model"] == video_settings.ollama_prompt_model
        assert not write["jpgs"]                      # it works from the descriptions

    async def test_the_descriptions_reach_the_writer(self, monkeypatch):
        calls = self._spy(monkeypatch)
        await generate_minimax_transition_prompts([b"1", b"2"])
        assert "a medium portrait, blue background" in calls[-1]["user_text"]

    async def test_it_uses_its_own_system_prompt(self, monkeypatch):
        calls = self._spy(monkeypatch)
        await generate_minimax_transition_prompts([b"1", b"2"])
        system = calls[-1]["system"]
        # The two properties that define this writer, rather than a heading
        # that may be reworded: it asks for sound, and it asks the prompt to
        # name where it arrives.
        assert "Audio:" in system
        assert "destination" in system

    async def test_the_wan_writer_is_unchanged_and_silent(self, monkeypatch):
        seen = {}
        async def fake_chat_json(**kwargs):
            seen.update(kwargs)
            return {"transitions": ["x"]}
        monkeypatch.setattr(analysis_module, "_chat_json", fake_chat_json)
        await generate_transition_prompts([b"1", b"2"])
        assert "Audio:" not in seen["system"]
        assert seen["jpgs"]                           # still one vision call

    async def test_an_unreadable_frame_does_not_lose_the_sequence(self, monkeypatch):
        async def fake_chat_json(**kwargs):
            if kwargs["label"] == "describe_key_frame":
                raise RuntimeError("VLM returned no description")
            return {"transitions": ["still written"]}
        monkeypatch.setattr(analysis_module, "_chat_json", fake_chat_json)
        assert await generate_minimax_transition_prompts([b"1", b"2"]) == ["still written"]

    async def test_the_token_budget_scales_with_the_number_of_pairs(self, monkeypatch):
        calls = self._spy(monkeypatch, n_prompts=1)
        await generate_minimax_transition_prompts([b"1", b"2"])
        two = calls[-1]["options"]["num_predict"]
        calls = self._spy(monkeypatch, n_prompts=5)
        await generate_minimax_transition_prompts([b"1"] * 6)
        six = calls[-1]["options"]["num_predict"]
        # One call answers for every pair, so a long sequence must not be
        # truncated into blank slots.
        assert six > two

    async def test_it_still_returns_one_prompt_per_pair(self, monkeypatch):
        self._spy(monkeypatch, n_prompts=1)
        assert await generate_minimax_transition_prompts([b"1", b"2", b"3"]) == ["x", ""]

    async def test_two_images_are_the_minimum(self, monkeypatch):
        self._spy(monkeypatch)
        with pytest.raises(RuntimeError):
            await generate_minimax_transition_prompts([b"1"])


class TestSuggestedMinimaxPromptCarriesAudio:
    """A small VLM does not reliably produce the Audio: line the system prompt
    asks for. The builder would add a fallback at submit time, but then the
    user never sees it and cannot edit it — so the endpoint completes it while
    the answer is still on its way to a textarea."""

    def test_a_prompt_without_audio_gains_one(self):
        out = ensure_sound_only_audio("The skin opens and the man is drawn into the powder.")
        assert "Audio:" in out
        assert "The skin opens" in out

    def test_a_prompt_that_has_one_keeps_its_own_words(self):
        written = "The hall shatters into sand.\nAudio: grains hissing in a wide echo."
        out = ensure_sound_only_audio(written)
        assert "grains hissing in a wide echo" in out
        assert out.count("Audio:") == 1


class TestResolutionKeep:
    """The restoration became optional, so "no upscale" had to become a value
    the whole path can carry rather than the absence of a request."""

    def test_zero_is_an_explicit_keep(self):
        assert clamp_resolution(0) == RESOLUTION_KEEP

    def test_a_missing_value_still_reads_as_the_default(self):
        # An absent choice is not the same as choosing to keep the size, and
        # reading a typo as "skip the expensive stage" would silently produce a
        # differently-shaped render.
        assert clamp_resolution(None) == 1080
        assert clamp_resolution("nonsense") == 1080

    def test_keeping_the_resolution_leaves_the_dimensions_alone(self):
        assert output_dimensions(864, 480, RESOLUTION_KEEP) == (864, 480)

    def test_a_real_target_still_resizes(self):
        assert output_dimensions(864, 480, 1080) == (1944, 1080)


class TestNeedsComfy:
    """Three routes out of the pass, and this is what picks between them."""

    def test_a_restoration_needs_the_gpu(self):
        assert needs_comfy(1080, 1) is True

    def test_interpolation_alone_still_needs_the_gpu(self):
        # RIFE is a ComfyUI node too — cheap, but not ffmpeg.
        assert needs_comfy(RESOLUTION_KEEP, 3) is True

    def test_a_plain_retime_never_reaches_comfyui(self):
        # Keep the size, no interpolation: whatever is left is a frame-rate
        # conform, which is ffmpeg's job and takes seconds.
        assert needs_comfy(RESOLUTION_KEEP, 1) is False
        assert needs_comfy(RESOLUTION_KEEP, None) is False


class TestBuildRetimeOnlyWorkflow:
    """A RESOLUTION_KEEP pass must not load a diffusion model it has no use
    for — that is the entire saving."""

    def _build(self, **kw):
        from pathlib import Path
        kw.setdefault("resolution", RESOLUTION_KEEP)
        kw.setdefault("filename_prefix", "artrium_up_test")
        kw.setdefault("has_audio", True)
        return build_upscale_workflow(Path("D:/storage/videos/clip.mp4"), **kw)

    def test_no_restoration_nodes_at_all(self):
        wf, _ = self._build(rife_multiplier=3)
        assert "sv_dit" not in wf
        assert "sv_vae" not in wf
        assert "sv_up" not in wf

    def test_rife_reads_the_loader_directly(self):
        wf, save = self._build(rife_multiplier=3)
        assert wf["sv_rife"]["inputs"]["frames"] == ["sv_load", 0]
        assert wf[save]["inputs"]["images"] == ["sv_rife", 0]

    def test_without_interpolation_the_muxer_reads_the_loader(self):
        wf, save = self._build(rife_multiplier=1)
        assert wf[save]["inputs"]["images"] == ["sv_load", 0]

    def test_every_link_still_points_at_a_node_that_exists(self):
        for mult in (1, 2, 3, 4):
            wf, _ = self._build(rife_multiplier=mult)
            for node in wf.values():
                for value in node["inputs"].values():
                    if isinstance(value, list) and value and isinstance(value[0], str):
                        assert value[0] in wf, f"dangling link to {value[0]}"

    def test_the_audio_rule_is_unchanged(self):
        wf, save = self._build(has_audio=False, rife_multiplier=2)
        assert "audio" not in wf[save]["inputs"]

    def test_a_target_rate_still_reaches_the_muxer(self):
        wf, save = self._build(rife_multiplier=2, render_fps=32.0)
        assert wf[save]["inputs"]["frame_rate"] == 32.0


class TestEstimateWithoutRestoration:
    def test_dropping_the_restoration_dominates_the_estimate(self):
        full = estimate_seconds(5.0, 1944, 1080, 3, True)
        retime = estimate_seconds(5.0, 1944, 1080, 3, False)
        # The restoration is ~40x the interpolation per pixel-frame, so this is
        # the difference between minutes and under a minute.
        assert retime < full / 5

    def test_a_pure_conform_falls_to_the_floor(self):
        assert estimate_seconds(5.0, 864, 480, 1, False) == 10

    def test_unknown_duration_still_yields_no_estimate(self):
        assert estimate_seconds(0.0, 864, 480, 1, False) == 0


class TestPassSettingsValidation:
    """Every dial is optional on its own, which makes exactly one new way to be
    wrong: asking for a render with nothing in it."""

    def _video(self, **kw):
        from core.models import Video
        return Video(
            id=uuid.uuid4(), status="done", filename="clip.mp4",
            filepath="videos/clip.mp4", **kw,
        )

    def test_all_three_dials_off_is_refused(self):
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc:
            _validate_upscale_target(self._video(), RESOLUTION_KEEP, 1, None)
        assert exc.value.status_code == 422

    def test_interpolation_alone_is_a_valid_pass(self):
        _validate_upscale_target(self._video(), RESOLUTION_KEEP, 3, None)

    def test_a_frame_rate_alone_is_a_valid_pass(self):
        _validate_upscale_target(self._video(), RESOLUTION_KEEP, 1, 30)

    def test_an_auto_multiplier_is_accepted(self):
        _validate_upscale_target(self._video(), RESOLUTION_KEEP, None, 24)

    def test_a_resolution_between_zero_and_the_minimum_is_still_rejected(self):
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc:
            _validate_upscale_target(self._video(), 200, 1, None)
        assert exc.value.status_code == 422


class TestExplicitMultiplierBeatsDerivation:
    """`None` asks the planner to work the factor out from the target rate; a
    number says what to do. The router used to force the first, which made the
    interpolation chips decorative the moment a rate was picked."""

    def test_none_derives_from_the_target(self):
        plan = plan_frame_rate(16.0, 60, None)
        assert plan["rife_multiplier"] == 4       # ceil(60/16)
        assert plan["needs_conform"] is True      # 64 -> 60

    def test_a_number_is_honoured(self):
        plan = plan_frame_rate(16.0, 60, 2)
        assert plan["rife_multiplier"] == 2
        assert plan["render_fps"] == 32.0
        assert plan["needs_conform"] is True

    def test_no_interpolation_with_a_target_is_a_plain_conform(self):
        plan = plan_frame_rate(48.0, 24, 1)
        assert plan["rife_multiplier"] == 1
        assert plan["needs_conform"] is True
        assert plan["duration_factor"] == 1.0


class TestRefineRenderRoutes:
    """`_refine_render` now has three ways out, and the cheapest one is the
    point of the change: a pass with nothing for the GPU to do must not submit
    an empty graph, evict VRAM, or queue behind somebody's restoration."""

    async def _run(self, monkeypatch, plan, tmp_path):
        submitted = []
        conformed = []
        copied = []

        monkeypatch.setattr(
            video_module, "probe_has_audio", lambda src: _async_value(True))
        monkeypatch.setattr(
            video_module, "build_upscale_workflow",
            lambda *a, **kw: submitted.append(kw) or ({}, "sv_save"))

        async def _conform(src, dest, fps, *, has_audio):
            conformed.append(fps)
            dest.write_bytes(b"conformed")

        monkeypatch.setattr(video_module, "_conform_frame_rate", _conform)
        monkeypatch.setattr(
            video_module.shutil, "copy2",
            lambda src, dest: copied.append(dest))

        src = tmp_path / "clip.mp4"
        src.write_bytes(b"source")
        await video_module._refine_render(
            src, tmp_path / "out.mp4",
            resolution=plan["resolution"], plan=plan,
            progress_key="test", prefix="artrium_test", log_subject="test",
        )
        return submitted, conformed, copied

    def _plan(self, **kw):
        base = {
            "resolution": 0, "restore": False, "needs_comfy": False,
            "rife_multiplier": 1, "render_fps": None, "target_fps": None,
            "needs_conform": False, "duration_factor": 1.0, "duration": 4.0,
            "width": 864, "height": 480, "seconds": 10,
        }
        base.update(kw)
        return base

    async def test_a_plain_retime_never_builds_a_workflow(self, monkeypatch, tmp_path):
        submitted, conformed, _ = await self._run(
            monkeypatch,
            self._plan(target_fps=30, needs_conform=True),
            tmp_path,
        )
        assert submitted == []          # ComfyUI never heard about it
        assert conformed == [30]

    async def test_nothing_to_change_just_places_the_rendition(self, monkeypatch, tmp_path):
        # The target rate is the one the file already has: no conform, no
        # render, but the row still expects a file where it points.
        submitted, conformed, copied = await self._run(
            monkeypatch, self._plan(target_fps=24), tmp_path)
        assert submitted == [] and conformed == []
        assert len(copied) == 1

    async def test_interpolation_alone_still_goes_through_comfyui(
        self, monkeypatch, tmp_path,
    ):
        # RIFE is a ComfyUI node, so this route is taken — it just carries no
        # diffusion model, which build_upscale_workflow decides, not this.
        # A sentinel at the VRAM gate proves the route without a live server.
        async def _boom(_workflow):
            raise RuntimeError("reached comfy")

        monkeypatch.setattr(video_module, "_free_ollama_vram", _boom)
        plan = self._plan(needs_comfy=True, rife_multiplier=3)
        with pytest.raises(RuntimeError, match="reached comfy"):
            await self._run(monkeypatch, plan, tmp_path)
