"""
Unit tests for the story-frames pipeline's pure parts: the shared z-Image
workflow builder (no ComfyUI) and the frame-prompt LLM wrapper's
normalization/trigger logic (mocked _chat_json, no Ollama).
"""
import json

import pytest

from routers.video import _seed_for_frame
from services.comfy.zimage import ZIMAGE_SAVE_NODE, build_zimage_workflow
from services.ollama import story_frames as story_module
from services.ollama.story_frames import (
    _salvage_truncated_frames,
    ensure_trigger,
    generate_story_frame_prompts,
)


class TestBuildZimageWorkflow:
    def test_prompt_seed_size_and_lora_land_in_the_right_nodes(self):
        loras = [{"name": "some_lora.safetensors", "strength": 0.7}]
        wf = build_zimage_workflow("a rusty gate", 1234, 960, 704, loras)
        assert wf["45"]["inputs"]["text"] == "a rusty gate"
        assert wf["44"]["inputs"]["seed"] == 1234
        assert wf["41"]["inputs"]["width"] == 960
        assert wf["41"]["inputs"]["height"] == 704
        assert wf["lora_0"]["inputs"]["lora_name"] == "some_lora.safetensors"
        assert wf["lora_0"]["inputs"]["strength_model"] == 0.7
        assert wf["47"]["inputs"]["model"] == ["lora_0", 0]

    def test_negative_seed_is_randomized(self):
        wf = build_zimage_workflow("p", -1, 512, 512, [{"name": "l.safetensors", "strength": 0.5}])
        assert wf["44"]["inputs"]["seed"] >= 0

    def test_lora_strength_is_clamped(self):
        wf = build_zimage_workflow("p", 1, 512, 512, [{"name": "l.safetensors", "strength": 1.7}])
        assert wf["lora_0"]["inputs"]["strength_model"] == 1.0

    def test_multiple_loras_are_chained_in_order(self):
        loras = [
            {"name": "a.safetensors", "strength": 0.3},
            {"name": "b.safetensors", "strength": 0.5},
        ]
        wf = build_zimage_workflow("p", 1, 512, 512, loras)
        assert wf["lora_0"]["inputs"]["model"] == ["46", 0]
        assert wf["lora_0"]["inputs"]["lora_name"] == "a.safetensors"
        assert wf["lora_1"]["inputs"]["model"] == ["lora_0", 0]
        assert wf["lora_1"]["inputs"]["lora_name"] == "b.safetensors"
        assert wf["47"]["inputs"]["model"] == ["lora_1", 0]

    def test_no_loras_wires_unet_straight_into_model_sampling(self):
        wf = build_zimage_workflow("p", 1, 512, 512, [])
        assert wf["47"]["inputs"]["model"] == ["46", 0]
        assert not any(k.startswith("lora_") for k in wf)

    def test_template_is_not_mutated_between_calls(self):
        build_zimage_workflow("first", 1, 512, 512, [{"name": "l.safetensors", "strength": 0.5}])
        wf2 = build_zimage_workflow("second", 2, 512, 512, [{"name": "l.safetensors", "strength": 0.5}])
        assert wf2["45"]["inputs"]["text"] == "second"

    def test_save_node_exists_in_template(self):
        wf = build_zimage_workflow("p", 1, 512, 512, [{"name": "l.safetensors", "strength": 0.5}])
        assert wf[ZIMAGE_SAVE_NODE]["class_type"] == "SaveImage"


class TestEnsureTrigger:
    def test_prepends_missing_trigger(self):
        assert ensure_trigger("a quiet harbor", "art_vision") == "art_vision, a quiet harbor"

    def test_keeps_prompt_when_trigger_present_case_insensitive(self):
        assert ensure_trigger("Art_Vision, a quiet harbor", "art_vision") == "Art_Vision, a quiet harbor"

    def test_no_trigger_is_a_noop(self):
        assert ensure_trigger("a quiet harbor", None) == "a quiet harbor"


class TestGenerateStoryFramePrompts:
    async def test_returns_n_prompts_on_exact_match(self, monkeypatch):
        async def fake_chat_json(**kwargs):
            return {"frames": ["frame one", "frame two", "frame three"]}
        monkeypatch.setattr(story_module, "_chat_json", fake_chat_json)

        result = await generate_story_frame_prompts(
            story="a door opens", n=3, description="desc",
        )
        assert result == ["frame one", "frame two", "frame three"]

    async def test_pads_short_response_by_repeating_last(self, monkeypatch):
        async def fake_chat_json(**kwargs):
            return {"frames": ["only one"]}
        monkeypatch.setattr(story_module, "_chat_json", fake_chat_json)

        result = await generate_story_frame_prompts(
            story="a door opens", n=3, description="desc",
        )
        assert result == ["only one", "only one", "only one"]

    async def test_truncates_long_response(self, monkeypatch):
        async def fake_chat_json(**kwargs):
            return {"frames": ["a", "b", "c", "d"]}
        monkeypatch.setattr(story_module, "_chat_json", fake_chat_json)

        result = await generate_story_frame_prompts(
            story="a door opens", n=2, description="desc",
        )
        assert result == ["a", "b"]

    async def test_trigger_is_enforced_on_every_frame(self, monkeypatch):
        async def fake_chat_json(**kwargs):
            return {"frames": ["zidiusArt, with trigger", "forgot the trigger"]}
        monkeypatch.setattr(story_module, "_chat_json", fake_chat_json)

        result = await generate_story_frame_prompts(
            story="s", n=2, description="d", trigger="zidiusArt",
        )
        assert result[0] == "zidiusArt, with trigger"
        assert result[1] == "zidiusArt, forgot the trigger"

    async def test_beat_seconds_lands_in_user_text(self, monkeypatch):
        captured = {}
        async def fake_chat_json(**kwargs):
            captured.update(kwargs)
            return {"frames": ["a", "b"]}
        monkeypatch.setattr(story_module, "_chat_json", fake_chat_json)

        await generate_story_frame_prompts(
            story="s", n=2, description="d", beat_seconds=10,
        )
        assert "10 seconds" in captured["user_text"]

    async def test_style_block_lands_in_system_prompt(self, monkeypatch):
        captured = {}
        async def fake_chat_json(**kwargs):
            captured.update(kwargs)
            return {"frames": ["a", "b"]}
        monkeypatch.setattr(story_module, "_chat_json", fake_chat_json)

        await generate_story_frame_prompts(
            story="s", n=2, description="d",
            style_block="## Style A — Test Style\nsome descriptors",
        )
        assert "some descriptors" in captured["system"]
        assert "{STYLE_BLOCK}" not in captured["system"]

    async def test_no_style_block_falls_back_to_source_style(self, monkeypatch):
        captured = {}
        async def fake_chat_json(**kwargs):
            captured.update(kwargs)
            return {"frames": ["a", "b"]}
        monkeypatch.setattr(story_module, "_chat_json", fake_chat_json)

        await generate_story_frame_prompts(story="s", n=2, description="d")
        assert "none — use the source image's own style." in captured["system"]
        assert "{STYLE_BLOCK}" not in captured["system"]

    async def test_non_list_frames_raises(self, monkeypatch):
        async def fake_chat_json(**kwargs):
            return {"frames": "not a list"}
        monkeypatch.setattr(story_module, "_chat_json", fake_chat_json)

        with pytest.raises(RuntimeError):
            await generate_story_frame_prompts(story="s", n=2, description="d")

    async def test_all_empty_frames_raises(self, monkeypatch):
        async def fake_chat_json(**kwargs):
            return {"frames": ["", "   "]}
        monkeypatch.setattr(story_module, "_chat_json", fake_chat_json)

        with pytest.raises(RuntimeError):
            await generate_story_frame_prompts(story="s", n=2, description="d")

    async def test_empty_story_raises(self):
        with pytest.raises(ValueError):
            await generate_story_frame_prompts(story="   ", n=2, description="d")

    async def test_reads_prompt_file_without_raising(self):
        # Sanity check the prompts/story-frames.md path/filename is correct
        # before it ever hits a live Ollama call.
        from services.ollama.chat import _read_prompt
        text = _read_prompt("story-frames.md")
        assert text.strip()
        assert "{STYLE_BLOCK}" in text


class TestSalvageTruncatedFrames:
    def test_recovers_complete_frames_before_truncation(self):
        raw = '{"frames": ["frame one", "frame two", "frame three that got cut off mid'
        salvaged = _salvage_truncated_frames(raw)
        assert salvaged == '{"frames": ["frame one", "frame two"]}'

    def test_no_frames_key_returns_unchanged(self):
        raw = '{"other": "stuff"'
        assert _salvage_truncated_frames(raw) == raw

    def test_first_frame_truncated_returns_unchanged(self):
        # Nothing usable to recover — let the caller retry instead of
        # "succeeding" with zero frames.
        raw = '{"frames": ["this one never closes and runs on forever'
        assert _salvage_truncated_frames(raw) == raw

    def test_handles_escaped_quotes_inside_a_frame(self):
        raw = r'{"frames": ["a \"quoted\" phrase", "second frame", "third cut off'
        salvaged = _salvage_truncated_frames(raw)
        assert salvaged == r'{"frames": ["a \"quoted\" phrase", "second frame"]}'
        json.loads(salvaged)

    def test_fully_complete_array_recovers_all_entries(self):
        raw = '{"frames": ["a", "b", "c"]}'
        assert _salvage_truncated_frames(raw) == '{"frames": ["a", "b", "c"]}'

    async def test_salvage_feeds_into_chat_json_retry_flow(self, monkeypatch):
        # End-to-end through _chat_json: first response is truncated
        # mid-string (real-world failure mode), salvage recovers the
        # complete leading frames instead of raising.
        import services.ollama.chat as chat_module

        class FakeResponse:
            status_code = 200
            def json(self):
                return {"message": {"content":
                    '{"frames": ["art_vision, first frame text", '
                    '"art_vision, second frame text", '
                    '"art_vision, third frame that runs on and on and never closes'
                }}

        class FakeClient:
            async def __aenter__(self):
                return self
            async def __aexit__(self, *a):
                return False
            async def post(self, *a, **kw):
                return FakeResponse()

        monkeypatch.setattr(chat_module.httpx, "AsyncClient", lambda **kw: FakeClient())

        result = await generate_story_frame_prompts(
            story="a door opens", n=3, description="desc", trigger="art_vision",
        )
        assert result == [
            "art_vision, first frame text",
            "art_vision, second frame text",
            "art_vision, second frame text",
        ]


class TestSeedForFrame:
    def test_first_occurrence_keeps_base_seed(self):
        seen: dict[str, int] = {}
        assert _seed_for_frame("a prompt", 1000, seen) == 1000

    def test_repeated_prompt_bumps_seed(self):
        seen: dict[str, int] = {}
        _seed_for_frame("a prompt", 1000, seen)
        assert _seed_for_frame("a prompt", 1000, seen) == 1001
        assert _seed_for_frame("a prompt", 1000, seen) == 1002

    def test_different_prompts_all_keep_base_seed(self):
        seen: dict[str, int] = {}
        assert _seed_for_frame("frame one", 1000, seen) == 1000
        assert _seed_for_frame("frame two", 1000, seen) == 1000
        assert _seed_for_frame("frame three", 1000, seen) == 1000

    def test_mutates_seen_prompts_across_calls(self):
        seen: dict[str, int] = {}
        _seed_for_frame("dup", 5, seen)
        _seed_for_frame("dup", 5, seen)
        assert seen["dup"] == 2


class TestZimageStyleCatalogue:
    def test_lists_styles_with_letter_and_name(self):
        from services.ollama.zimage_enhance import list_zimage_styles
        styles = list_zimage_styles()
        assert styles, "no styles parsed from prompts/zimage-styles.md"
        for s in styles:
            assert s["style"] in ("A", "B", "C", "D")
            assert s["name"].strip()

    def test_get_block_roundtrip_and_unknown_letter(self):
        from services.ollama.zimage_enhance import (
            get_zimage_style_block,
            list_zimage_styles,
        )
        first = list_zimage_styles()[0]["style"]
        assert get_zimage_style_block(first).startswith("## Style")
        assert get_zimage_style_block("Z") is None
