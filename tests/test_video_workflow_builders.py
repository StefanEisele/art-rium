"""
Unit tests for the flf2v per-transition workflow builder (pure node-graph
wiring, no ComfyUI) and the transition-prompt VLM wrapper's pad/truncate
logic (mocked _chat_json, no Ollama).
"""
import asyncio
import uuid

import pytest

from routers import video as video_module
from routers.video import (
    _build_flf2v_single_workflow,
    _build_minimax_single_workflow,
    _grain_source,
    _is_graining,
    _is_upscaling,
    _render_version,
    _upscale_source,
    _validate_upscale_target,
    adapt_minimax_canvas,
    align_minimax_length,
    ensure_sound_only_audio,
)
from services.video.upscale import (
    build_upscale_workflow,
    clamp_resolution,
    clamp_rife,
    estimate_seconds,
    output_dimensions,
)
from services.ollama import analysis as analysis_module
from services.ollama.analysis import (
    generate_i2v_motion_prompts,
    generate_minimax_motion_prompts,
    generate_transition_prompts,
)


class TestBuildFlf2vSingleWorkflow:
    def test_returns_dict_and_matching_save_node(self):
        wf, save_id = _build_flf2v_single_workflow(
            "start.png", "end.png", "camera pushes in", 25, 960, 960, 24, "prefix", 3,
        )
        assert save_id in wf
        assert wf[save_id]["class_type"] == "VHS_VideoCombine"

    def test_load_image_nodes_for_start_and_end(self):
        wf, _ = _build_flf2v_single_workflow(
            "start.png", "end.png", "prompt", 25, 960, 960, 24, "prefix", 3,
        )
        assert wf["img_start"] == {"class_type": "LoadImage", "inputs": {"image": "start.png", "upload": "image"}}
        assert wf["img_end"] == {"class_type": "LoadImage", "inputs": {"image": "end.png", "upload": "image"}}

    def test_end_frame_append_is_off_by_default(self):
        # Default: RIFE reads the diffused frames directly — no raw end photo
        # appended (that append IS the hard cut when diffusion undershoots).
        wf, _ = _build_flf2v_single_workflow(
            "start.png", "end.png", "prompt", 25, 960, 960, 24, "prefix", 3,
        )
        assert "batch_final" not in wf
        assert wf["rife"]["inputs"]["frames"] == ["t0_decode", 0]

    def test_batch_final_appends_raw_end_frame_when_opted_in(self):
        wf, _ = _build_flf2v_single_workflow(
            "start.png", "end.png", "prompt", 25, 960, 960, 24, "prefix", 3,
            append_end_frame=True,
        )
        batch = wf["batch_final"]
        assert batch["class_type"] == "ImageBatch"
        assert batch["inputs"]["image1"] == ["t0_decode", 0]
        assert batch["inputs"]["image2"] == ["img_end", 0]
        assert wf["rife"]["inputs"]["frames"] == ["batch_final", 0]

    def test_sampler_lightning_fast_path(self):
        # Plain Lightning fast path: 4 steps split 2/2, cfg=1 and full-strength
        # distill LoRA on both experts (the 8-step anti-hard-cut variant was
        # reverted — too slow in practice).
        wf, _ = _build_flf2v_single_workflow(
            "start.png", "end.png", "prompt", 25, 960, 960, 24, "prefix", 3,
        )
        ks_h, ks_l = wf["t0_ks_h"]["inputs"], wf["t0_ks_l"]["inputs"]
        assert ks_h["steps"] == 4 and ks_l["steps"] == 4
        assert ks_h["end_at_step"] == 2 and ks_l["start_at_step"] == 2
        assert ks_h["cfg"] == 1
        assert ks_l["cfg"] == 1
        assert wf["t0_lora_h"]["inputs"]["strength_model"] == 1.0
        assert wf["t0_lora_l"]["inputs"]["strength_model"] == 1

    def test_rife_multiplier_is_configurable(self):
        wf, _ = _build_flf2v_single_workflow(
            "start.png", "end.png", "prompt", 25, 960, 960, 24, "prefix", 4,
        )
        assert wf["rife"]["inputs"]["multiplier"] == 4

    def test_save_node_reads_from_rife(self):
        wf, save_id = _build_flf2v_single_workflow(
            "start.png", "end.png", "prompt", 25, 960, 960, 24, "prefix", 3,
        )
        assert wf[save_id]["inputs"]["images"] == ["rife", 0]

    def test_prompt_lands_in_positive_clip_encode(self):
        wf, _ = _build_flf2v_single_workflow(
            "start.png", "end.png", "a specific transition prompt", 25, 960, 960, 24, "prefix", 3,
        )
        assert wf["t0_pos"]["inputs"]["text"] == "a specific transition prompt"


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


class TestGrainSource:
    """The grain pass must never read its own output — otherwise moving the
    strength slider bakes a second pass on top of the first and the picture
    silts up a little more with every adjustment."""

    def _video(self, **kw):
        from core.models import Video
        return Video(filepath="videos/clean.mp4", **kw)

    def test_ignores_its_own_output(self):
        v = self._video(grain_filename="x_grain.mp4", grain_strength=40)
        assert _grain_source(v).name == "clean.mp4"

    def test_prefers_the_muxed_variant_so_a_soundtrack_survives(self):
        v = self._video(muxed_filename="x_muxed.mp4")
        assert _grain_source(v).name == "x_muxed.mp4"

    def test_regrading_still_reads_the_muxed_variant_not_the_grain(self):
        v = self._video(muxed_filename="x_muxed.mp4", grain_filename="x_grain.mp4")
        assert _grain_source(v).name == "x_muxed.mp4"

    def test_falls_back_to_the_original(self):
        assert _grain_source(self._video()).name == "clean.mp4"


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
        assert _grain_source(v).name == "x_upscale.mp4"

    def test_upscale_wins_over_the_original(self):
        v = self._video(upscale_filename="x_upscale.mp4")
        assert _grain_source(v).name == "x_upscale.mp4"


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
    fact that matters — how much is actually free."""

    GB = 1024 ** 3

    def _devices(self, free_gb, total_gb=16.0):
        return [{
            "name": "cuda:0 NVIDIA GeForce RTX 4060 Ti",
            "vram_free": free_gb * self.GB,
            "vram_total": total_gb * self.GB,
        }]

    def _no_waiting(self, monkeypatch, released: list | None = None):
        """Stub both holders and collapse the wait, so a shortfall fails fast."""
        monkeypatch.setattr(video_module, "_evict_ollama", lambda: _async_value(None))
        monkeypatch.setattr(
            video_module, "_release_comfy_models",
            lambda: _async_value(released.append(1) if released is not None else None),
        )
        monkeypatch.setattr(video_module, "_VRAM_WAIT_TIMEOUT", 0.0)
        monkeypatch.setattr(video_module, "_VRAM_WAIT_POLL", 0.0)
        monkeypatch.setattr(video_module, "_COMFY_FREE_SETTLE", 0.0)

    async def test_passes_when_the_card_is_free(self, monkeypatch):
        monkeypatch.setattr(video_module, "_comfy_devices",
                            lambda: _async_value(self._devices(15.5)))
        await video_module._preflight_comfyui("minimax_i2v")  # must not raise

    async def test_a_free_card_is_not_disturbed(self, monkeypatch):
        """Unloading costs a cold reload, so it must only happen on a shortfall."""
        released = []
        monkeypatch.setattr(video_module, "_comfy_devices",
                            lambda: _async_value(self._devices(15.5)))
        self._no_waiting(monkeypatch, released)
        await video_module._preflight_comfyui("minimax_i2v")
        assert released == []

    async def test_a_shortfall_unloads_comfyui_before_waiting(self, monkeypatch):
        """The regression this exists for: an upscale queued right after a
        render found ~3 GB free, and ComfyUI — not Ollama — was holding it, so
        the old loop re-evicted Ollama for 210 s and then gave up."""
        released = []
        monkeypatch.setattr(video_module, "_comfy_devices",
                            lambda: _async_value(self._devices(3.4)))
        self._no_waiting(monkeypatch, released)
        with pytest.raises(RuntimeError, match="still busy"):
            await video_module._preflight_comfyui("upscale")
        assert released, "ComfyUI was never asked to unload"

    async def test_refuses_a_minimax_run_on_a_half_full_card(self, monkeypatch):
        # 14956 MB staged for the text encoder alone — half a card is not a
        # slow run, it is a CUDA OOM that kills ComfyUI's worker thread.
        monkeypatch.setattr(video_module, "_comfy_devices",
                            lambda: _async_value(self._devices(8.0)))
        self._no_waiting(monkeypatch)
        with pytest.raises(RuntimeError, match="still busy"):
            await video_module._preflight_comfyui("minimax_i2v")

    async def test_the_same_card_is_fine_for_a_lighter_workflow(self, monkeypatch):
        # Wan and the 3B upscaler do not need the whole card.
        monkeypatch.setattr(video_module, "_comfy_devices",
                            lambda: _async_value(self._devices(10.0)))
        await video_module._preflight_comfyui("i2v_multi")

    async def test_the_error_names_the_numbers(self, monkeypatch):
        monkeypatch.setattr(video_module, "_comfy_devices",
                            lambda: _async_value(self._devices(2.0)))
        self._no_waiting(monkeypatch)
        with pytest.raises(RuntimeError) as exc:
            await video_module._preflight_comfyui("minimax_i2v")
        msg = str(exc.value)
        assert "2.1 GB free" in msg and "ollama ps" in msg

    async def test_no_devices_does_not_block_the_job(self, monkeypatch):
        # A ComfyUI build that reports no devices should not make video
        # generation impossible — the gate is a safety net, not a gatekeeper.
        monkeypatch.setattr(video_module, "_comfy_devices", lambda: _async_value([]))
        await video_module._preflight_comfyui("minimax_i2v")


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
