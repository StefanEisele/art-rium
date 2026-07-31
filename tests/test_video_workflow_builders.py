"""
Unit tests for the flf2v per-transition workflow builder (pure node-graph
wiring, no ComfyUI) and the transition-prompt VLM wrapper's pad/truncate
logic (mocked _chat_json, no Ollama).
"""
import uuid

import pytest

from routers import video as video_module
from routers.video import (
    _build_flf2v_single_workflow,
    _build_ltx_single_workflow,
    _grain_source,
    _is_graining,
    _render_version,
)
from services.ollama import analysis as analysis_module
from services.ollama.analysis import (
    generate_i2v_motion_prompts,
    generate_ltx_motion_prompts,
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


class TestBuildLtxSingleWorkflow:
    def test_default_no_rife_node_wired_straight_from_decode(self):
        wf, save_id, _ = _build_ltx_single_workflow(
            "img.png", "a prompt", 25, 960, 960, 24, "prefix",
        )
        assert "ltx_rife" not in wf
        assert wf[save_id]["inputs"]["images"] == ["ltx_vdec", 0]

    def test_rife_multiplier_inserts_rife_node(self):
        wf, save_id, _ = _build_ltx_single_workflow(
            "img.png", "a prompt", 25, 960, 960, 24, "prefix", 3,
        )
        assert wf["ltx_rife"]["class_type"] == "RIFE VFI"
        assert wf["ltx_rife"]["inputs"]["multiplier"] == 3
        assert wf["ltx_rife"]["inputs"]["frames"] == ["ltx_vdec", 0]
        assert wf[save_id]["inputs"]["images"] == ["ltx_rife", 0]

    def test_audio_always_wired_from_native_decode_regardless_of_rife(self):
        # Native LTX audio is intentionally left at its pre-RIFE length — the
        # caller (services.video.audio_stretch) re-syncs it afterward.
        wf, save_id, _ = _build_ltx_single_workflow(
            "img.png", "a prompt", 25, 960, 960, 24, "prefix", 4,
        )
        assert wf[save_id]["inputs"]["audio"] == ["ltx_adec", 0]

    def test_frame_count_snapped_to_8n_plus_1_and_returned(self):
        _, _, length = _build_ltx_single_workflow(
            "img.png", "prompt", 30, 960, 960, 24, "prefix",
        )
        assert length == 33  # (30-1+4)//8*8+1

    def test_frame_count_already_valid_is_unchanged(self):
        _, _, length = _build_ltx_single_workflow(
            "img.png", "prompt", 49, 960, 960, 24, "prefix",
        )
        assert length == 49

    def test_prompt_lands_in_positive_clip_encode(self):
        wf, _, _ = _build_ltx_single_workflow(
            "img.png", "a specific ltx prompt", 25, 960, 960, 24, "prefix",
        )
        assert wf["ltx_pos"]["inputs"]["text"] == "a specific ltx prompt"


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


class TestGenerateLtxMotionPrompts:
    """Mirrors TestGenerateI2vMotionPrompts — same per-image fan-out helper,
    different system prompt file (LTX-2.3's audio-aware variant)."""

    async def test_one_call_per_image_with_single_jpg_each(self, monkeypatch):
        calls = []
        async def fake_chat_json(**kwargs):
            calls.append(kwargs)
            return {"animation": f"prompt {len(calls)}"}
        monkeypatch.setattr(analysis_module, "_chat_json", fake_chat_json)

        result = await generate_ltx_motion_prompts([b"1", b"2", b"3"])
        assert result == ["prompt 1", "prompt 2", "prompt 3"]
        assert len(calls) == 3
        assert all(kw["jpgs"] == [img] for kw, img in zip(calls, [b"1", b"2", b"3"]))
        assert "image 2 of 3" in calls[1]["user_text"]

    async def test_uses_ltx_system_prompt_not_wan(self, monkeypatch):
        systems = []
        async def fake_chat_json(**kwargs):
            systems.append(kwargs["system"])
            return {"animation": "ok"}
        monkeypatch.setattr(analysis_module, "_chat_json", fake_chat_json)

        await generate_ltx_motion_prompts([b"1"])
        from services.ollama.chat import _read_prompt
        assert systems[0] == _read_prompt("video-ltx-motion.md")
        assert systems[0] != _read_prompt("video-i2v-motion.md")

    async def test_failed_image_yields_empty_slot(self, monkeypatch):
        calls = []
        async def fake_chat_json(**kwargs):
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError("boom")
            return {"animation": f"prompt {len(calls)}"}
        monkeypatch.setattr(analysis_module, "_chat_json", fake_chat_json)

        result = await generate_ltx_motion_prompts([b"1", b"2", b"3"])
        assert result == ["prompt 1", "", "prompt 3"]

    async def test_all_calls_failing_raises(self, monkeypatch):
        async def fake_chat_json(**kwargs):
            raise RuntimeError("ollama down")
        monkeypatch.setattr(analysis_module, "_chat_json", fake_chat_json)

        with pytest.raises(RuntimeError):
            await generate_ltx_motion_prompts([b"1", b"2"])

    async def test_empty_image_list_raises(self):
        with pytest.raises(RuntimeError):
            await generate_ltx_motion_prompts([])

    async def test_reads_prompt_file_without_raising(self):
        from services.ollama.chat import _read_prompt
        text = _read_prompt("video-ltx-motion.md")
        assert text.strip()

    async def test_caps_num_predict_low_enough_to_block_a_ramble(self, monkeypatch):
        # The cap is the backstop behind the system prompt's length rule. It
        # only works if it is anywhere near the 12-25 word target; the old
        # value of 220 allowed ~165 words and so capped nothing.
        seen = []
        async def fake_chat_json(**kwargs):
            seen.append(kwargs["options"])
            return {"animation": "ok"}
        monkeypatch.setattr(analysis_module, "_chat_json", fake_chat_json)

        await generate_ltx_motion_prompts([b"1"])
        assert seen[0]["num_predict"] <= 100


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


class TestMotionPromptRejection:
    """The small titler VLM (qwen2.5vl:3b) under a long system prompt returns
    one of the worked examples verbatim, or repeats one answer across every
    image. Either way the prompt describes a different picture than the one
    it is attached to, which is what makes LTX drift and cut mid-clip."""

    async def test_example_copied_from_system_prompt_is_retried(self, monkeypatch):
        from services.ollama.chat import _read_prompt

        # An 8-word run lifted straight out of the shipped system prompt.
        leaked = " ".join(_read_prompt("video-ltx-motion.md").split()[20:40])
        answers = iter([leaked, "The rope frays slowly, fibres ticking, the frame static."])
        async def fake_chat_json(**kwargs):
            return {"animation": next(answers)}
        monkeypatch.setattr(analysis_module, "_chat_json", fake_chat_json)

        result = await generate_ltx_motion_prompts([b"1"])
        assert result == ["The rope frays slowly, fibres ticking, the frame static."]

    async def test_prompt_repeated_across_images_is_retried(self, monkeypatch):
        dupe = ("The rust-red paint ridge lifts off the canvas in fine curling "
                "threads, rising slowly, the camera orbiting gently.")
        answers = iter([dupe, dupe, "The glass sags inward with a low groan, frame static."])
        async def fake_chat_json(**kwargs):
            return {"animation": next(answers)}
        monkeypatch.setattr(analysis_module, "_chat_json", fake_chat_json)

        result = await generate_ltx_motion_prompts([b"1", b"2"])
        assert result[0] == dupe
        assert result[1] == "The glass sags inward with a low groan, frame static."

    async def test_gives_up_after_max_attempts_rather_than_blanking(self, monkeypatch):
        # A rejected prompt the user can edit beats an empty textarea.
        dupe = ("The rust-red paint ridge lifts off the canvas in fine curling "
                "threads, rising slowly, the camera orbiting gently.")
        async def fake_chat_json(**kwargs):
            return {"animation": dupe}
        monkeypatch.setattr(analysis_module, "_chat_json", fake_chat_json)

        result = await generate_ltx_motion_prompts([b"1", b"2"])
        assert result == [dupe, dupe]

    async def test_distinct_prompts_are_not_retried(self, monkeypatch):
        calls = []
        async def fake_chat_json(**kwargs):
            calls.append(1)
            return {"animation": f"The number {len(calls)} object drifts upward with a faint hum."}
        monkeypatch.setattr(analysis_module, "_chat_json", fake_chat_json)

        await generate_ltx_motion_prompts([b"1", b"2", b"3"])
        assert len(calls) == 3  # no wasted retries on good output


class TestOneBeatEnforcement:
    """A one-beat prompt is what stops LTX cutting mid-clip, so the sentence
    limit is enforced in code rather than trusted to the system prompt."""

    async def test_extra_sentences_are_trimmed_away(self, monkeypatch):
        async def fake_chat_json(**kwargs):
            return {"animation": (
                "The woman walks through the corridor, her footsteps echoing. "
                "The sound of her boots claps against the floor. "
                "The camera remains static."
            )}
        monkeypatch.setattr(analysis_module, "_chat_json", fake_chat_json)

        result = await generate_ltx_motion_prompts([b"1"])
        assert result == ["The woman walks through the corridor, her footsteps echoing."]

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

        result = await generate_ltx_motion_prompts([b"1"])
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

        await generate_ltx_motion_prompts([b"1"])
        assert "reveal something new" in texts[1]
        assert "reused wording" not in texts[1]

    async def test_wan_variant_keeps_its_own_length_and_language_rules(self, monkeypatch):
        # Wan2.2's prompt file is tuned separately; the LTX-only one-sentence
        # trim and shot-break ban must not silently apply to it.
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


class TestFirstSentence:
    def test_keeps_single_sentence_unchanged(self):
        assert analysis_module.first_sentence("The rope frays slowly.") == "The rope frays slowly."

    def test_drops_trailing_sentences(self):
        assert analysis_module.first_sentence("One thing. Two thing.") == "One thing."

    def test_handles_missing_terminator(self):
        assert analysis_module.first_sentence("no full stop here") == "no full stop here"

    def test_empty_stays_empty(self):
        assert analysis_module.first_sentence("   ") == ""


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
