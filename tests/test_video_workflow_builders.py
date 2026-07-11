"""
Unit tests for the flf2v per-transition workflow builder (pure node-graph
wiring, no ComfyUI) and the transition-prompt VLM wrapper's pad/truncate
logic (mocked _chat_json, no Ollama).
"""
import pytest

from routers.video import _build_flf2v_single_workflow
from services.ollama import analysis as analysis_module
from services.ollama.analysis import (
    generate_i2v_motion_prompts,
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
