"""Embeddings in the Struktur tool's request path (routers/vace.py).

Two failures this guards against are silent ones. ComfyUI skips an embedding it
cannot find with a log line and renders without it, so a typo must be refused
before anything is queued. And a ladder is only a comparison if exactly one
thing moves between its rungs — the seed, the pass, every other embedding held.
"""
import uuid

import pytest
from fastapi import HTTPException

from routers import vace
from routers.vace import (
    EmbeddingSpec,
    GenerateRequest,
    LadderRequest,
    _check_embeddings,
    _ladder_variant,
    _layer_factor,
    _lcm_request,
    plan_ladder,
)


def form(**kw) -> GenerateRequest:
    args = {
        "control_track_id": uuid.uuid4(), "prompt": "a monolith",
        "embeddings": [EmbeddingSpec(name="artrium/echos"),
                       EmbeddingSpec(name="style-swirlmagic", weight=0.8)],
    }
    args.update(kw)
    return GenerateRequest(**args)


def ladder(**kw) -> LadderRequest:
    args = {"request": form(), "target": "style-swirlmagic", "weights": [0.8, 1.1, 1.4]}
    args.update(kw)
    return LadderRequest(**args)


class TestPlanLadder:
    def test_a_weight_ladder(self):
        assert plan_ladder(ladder()) == [
            {"label": "0.80", "weight": 0.8}, {"label": "1.10", "weight": 1.1},
            {"label": "1.40", "weight": 1.4},
        ]

    def test_weights_are_clamped(self):
        assert [r["weight"] for r in plan_ladder(ladder(weights=[-1, 9]))] == [0.0, 2.0]

    def test_a_snapshot_ladder_carries_its_labels(self):
        rungs = plan_ladder(ladder(
            target="artrium/echos", weights=[],
            names=["artrium/_train/echos/echos-step00000250",
                   "artrium/_train/echos/echos-step00000500"],
            labels=["250", "500"]))
        assert rungs[0] == {"label": "250", "name": "artrium/_train/echos/echos-step00000250"}

    def test_the_target_must_be_in_the_render(self):
        with pytest.raises(HTTPException, match="ausgewählt"):
            plan_ladder(ladder(target="style-rustmagic"))

    @pytest.mark.parametrize("kw", [
        {"weights": [], "names": []},
        {"names": ["a", "b"]},                       # both at once
        {"weights": [1.0]},                          # one rung compares nothing
        {"weights": [1.0] * 9},
    ])
    def test_malformed_ladders_are_refused(self, kw):
        with pytest.raises(HTTPException):
            plan_ladder(ladder(**kw))

    def test_vace_has_no_ladder(self):
        with pytest.raises(HTTPException, match="AnimateLCM"):
            plan_ladder(ladder(request=form(engine="vace")))


class TestVariant:
    def test_only_the_target_moves(self):
        req = form()
        variant = _ladder_variant(req, "style-swirlmagic", {"weight": 1.3})
        assert [(e.name, e.weight) for e in variant.embeddings] == [
            ("artrium/echos", 1.0), ("style-swirlmagic", 1.3)]
        assert req.embeddings[1].weight == 0.8, "the form itself is untouched"

    def test_a_rung_is_the_base_pass_only(self):
        variant = _ladder_variant(form(hires=True, rife=4), "style-swirlmagic", {"weight": 1})
        assert (variant.hires, variant.rife, variant.preview_first) == (False, 1, False)

    def test_a_snapshot_rung_swaps_the_name(self):
        variant = _ladder_variant(form(), "artrium/echos", {"name": "artrium/_train/echos/x"})
        assert variant.embeddings[0].name == "artrium/_train/echos/x"


class TestBuilderRequest:
    def test_embeddings_reach_the_builder(self):
        lcm = _lcm_request(form(embedding_join="sandwich"), uuid.uuid4(), "control/c.mp4",
                           None, [], 32, (1080, 1080), "depth", 1234)
        assert [(e.name, e.weight) for e in lcm.embeddings] == [
            ("artrium/echos", 1.0), ("style-swirlmagic", 0.8)]
        assert lcm.embedding_join == "sandwich"
        assert lcm.reference_image is None, "embeddings alone need no picture"

    def test_inline_is_the_tools_default(self):
        assert form().embedding_join == "inline"

    def test_a_curve_reaches_the_builder(self):
        req = form(embeddings=[EmbeddingSpec(name="artrium/echos", curve="pulse", cycles=4)])
        lcm = _lcm_request(req, uuid.uuid4(), "control/c.mp4", None, [], 32, (1080, 1080),
                           "depth", 1)
        assert (lcm.embeddings[0].curve, lcm.embeddings[0].cycles) == ("pulse", 4)

    def test_every_curve_costs_a_pass(self):
        assert _layer_factor(form()) == 1
        assert _layer_factor(form(embeddings=[
            EmbeddingSpec(name="a", curve="fade_in"), EmbeddingSpec(name="b", curve="swell"),
            EmbeddingSpec(name="c")])) == 3
        assert _layer_factor(form(engine="vace")) == 1


@pytest.mark.asyncio
class TestCheckEmbeddings:
    @pytest.fixture
    def comfy(self, monkeypatch):
        names = ["artrium/echos", "style-swirlmagic"]

        async def known():
            return names
        monkeypatch.setattr(vace, "embedding_names", known)
        return names

    async def test_known_names_pass(self, comfy):
        await _check_embeddings(form())

    async def test_a_name_comfy_would_silently_skip_is_refused(self, comfy):
        with pytest.raises(HTTPException, match="kennt dieses Embedding nicht"):
            await _check_embeddings(form(embeddings=[EmbeddingSpec(name="artrium/typo")]))

    async def test_comfy_offline_is_not_a_refusal(self, monkeypatch):
        async def offline():
            return None
        monkeypatch.setattr(vace, "embedding_names", offline)
        await _check_embeddings(form())

    async def test_only_animatelcm_reads_embeddings(self, comfy):
        with pytest.raises(HTTPException, match="AnimateLCM"):
            await _check_embeddings(form(engine="vace"))

    async def test_an_unknown_curve(self, comfy):
        with pytest.raises(HTTPException, match="Verlauf"):
            await _check_embeddings(form(embeddings=[
                EmbeddingSpec(name="artrium/echos", curve="wobble")]))

    async def test_an_unknown_join(self, comfy):
        with pytest.raises(HTTPException, match="embedding_join"):
            await _check_embeddings(form(embedding_join="blend"))

    async def test_nothing_to_check_without_embeddings(self, monkeypatch):
        async def boom():
            raise AssertionError("must not ask ComfyUI")
        monkeypatch.setattr(vace, "embedding_names", boom)
        await _check_embeddings(form(embeddings=[]))
