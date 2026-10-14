"""The style LoRA in the Struktur tool's request path (routers/vace.py).

The page sends the measured default (artvis3 at 0.8 plus the artvis3
embedding, see core/loras.py); the API itself stays opt-in so an old caller
gets the graph it always got. Both stages and the ladder rebuild their builder
request through `_lcm_request`, so that is where the field has to arrive.
"""
import uuid

import pytest
from fastapi import HTTPException

from core.loras import SD15_DEFAULT_STYLE_LORA, SD15_STYLE_LORAS
from routers import vace
from routers.vace import (
    EmbeddingSpec,
    GenerateRequest,
    _check_style_lora,
    _lcm_request,
    vace_options,
)


def form(**kw) -> GenerateRequest:
    args = {"control_track_id": uuid.uuid4(), "prompt": "a monolith",
            "embeddings": [EmbeddingSpec(name="artrium/artvis3emb")]}
    args.update(kw)
    return GenerateRequest(**args)


def builder_request(req: GenerateRequest):
    return _lcm_request(req, uuid.uuid4(), "control/c.mp4", None, [], 32,
                        (1080, 1080), "depth", 1234)


class TestRequest:
    def test_off_unless_asked_for(self):
        assert builder_request(form()).style_lora is None

    def test_it_reaches_the_builder(self):
        lcm = builder_request(form(style_lora="artvis3_sd15.safetensors",
                                   style_lora_strength=0.6))
        assert (lcm.style_lora, lcm.style_lora_strength) == ("artvis3_sd15.safetensors", 0.6)

    def test_it_survives_the_plan_round_trip(self):
        # The hires stage rebuilds the request from the plan's JSON.
        req = form(style_lora="artvis3_sd15.safetensors")
        again = GenerateRequest(**req.model_dump(mode="json"))
        assert builder_request(again).style_lora == "artvis3_sd15.safetensors"

    def test_a_registered_lora_passes(self):
        _check_style_lora(form(style_lora=SD15_DEFAULT_STYLE_LORA))
        _check_style_lora(form())

    @pytest.mark.parametrize("name", [
        "AnimateLCM_sd15_t2v_lora.safetensors",      # a distillation LoRA
        "xl_more_art-full_v1.safetensors",           # SDXL
        "../x.safetensors",
    ])
    def test_anything_else_is_refused(self, name):
        with pytest.raises(HTTPException, match="Stil-LoRA"):
            _check_style_lora(form(style_lora=name))


class TestOptions:
    async def test_the_page_gets_the_measured_defaults(self, monkeypatch):
        async def known():
            return ["artrium/artvis3emb", "style-rustmagic"]
        monkeypatch.setattr(vace, "embedding_names", known)
        lcm = (await vace_options())["lcm"]
        assert lcm["style_lora"]["default"] == "artvis3_sd15.safetensors"
        assert lcm["style_lora"]["loras"] == SD15_STYLE_LORAS
        assert SD15_STYLE_LORAS[0]["strength"] == 0.8
        assert lcm["embeddings"] == [
            {"name": "artrium/artvis3emb", "weight": 1.0, "label": "artvis3emb"}]

    async def test_a_default_comfy_cannot_load_is_not_offered(self, monkeypatch):
        async def known():
            return ["style-rustmagic"]
        monkeypatch.setattr(vace, "embedding_names", known)
        assert (await vace_options())["lcm"]["embeddings"] == []

    async def test_comfy_offline_still_offers_it(self, monkeypatch):
        async def offline():
            return None
        monkeypatch.setattr(vace, "embedding_names", offline)
        assert [e["name"] for e in (await vace_options())["lcm"]["embeddings"]] == [
            "artrium/artvis3emb"]
