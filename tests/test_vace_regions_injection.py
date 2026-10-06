"""Region embeddings and image injection in the Struktur request path.

The trap these guard: the reference list is aligned with the regions by index
and travels through the two-stage plan on disk. With regions that carry only
embeddings it has gaps, and a gap must stay a gap — not shift the next
region's picture onto the wrong colour, and not come back as the string
"None" from JSON.
"""
import json
import uuid

import pytest
from fastapi import HTTPException

from core.config import settings
from routers import vace
from routers.vace import (
    NOISE_PATTERNS,
    EmbeddingSpec,
    GenerateRequest,
    InjectionSpec,
    RegionSpec,
    _check_embeddings,
    _layer_factor,
    _lcm_request,
    _lcm_plan_path,
    _write_lcm_plan,
    noise_command,
)


def form(**kw) -> GenerateRequest:
    args = {
        "control_track_id": uuid.uuid4(), "prompt": "a monolith", "mask_track_id": uuid.uuid4(),
        "regions": [
            RegionSpec(color=(255, 0, 0), embeddings=[EmbeddingSpec(name="artrium/rust", weight=1.1)]),
            RegionSpec(color=(0, 255, 0), image_id=uuid.uuid4()),
        ],
    }
    args.update(kw)
    return GenerateRequest(**args)


class TestBuilderRequest:
    def test_a_gap_stays_on_its_region(self):
        lcm = _lcm_request(form(), uuid.uuid4(), "control/c.mp4", "control/m.mp4",
                           [None, "green.png"], 32, (1080, 1080), "depth", 1)
        assert lcm.regions[0].reference is None
        assert [(e.name, e.weight) for e in lcm.regions[0].embeddings] == [("artrium/rust", 1.1)]
        assert lcm.regions[1].reference == "green.png"
        assert lcm.regions[1].embeddings == []
        assert lcm.base_reference is None

    def test_the_injection_resolves_to_the_track_file(self):
        req = form(injection=InjectionSpec(track_id=uuid.uuid4(), strength=0.4, start=0.25,
                                           color=(0, 255, 0)))
        lcm = _lcm_request(req, uuid.uuid4(), "control/c.mp4", "control/m.mp4",
                           [None, "green.png"], 32, (1080, 1080), "depth", 1,
                           ("control/static.mp4", 96))
        inj = lcm.injection
        assert inj.video == str(settings.storage_dir / "control/static.mp4")
        assert (inj.strength, inj.start, inj.color, inj.source_frames) == (0.4, 0.25, (0, 255, 0), 96)

    def test_no_injection_without_its_track(self):
        req = form(injection=InjectionSpec(track_id=uuid.uuid4()))
        lcm = _lcm_request(req, uuid.uuid4(), "c.mp4", "m.mp4", [None, "g.png"], 32,
                           (1080, 1080), "depth", 1, None)
        assert lcm.injection is None


class TestPlan:
    def test_gaps_are_null_and_the_injection_rides_along(self, tmp_path, monkeypatch):
        monkeypatch.setattr(settings, "storage_dir", tmp_path)
        vid = uuid.uuid4()
        green = uuid.uuid4()
        _write_lcm_plan(vid, form(), "control/c.mp4", "control/m.mp4", [None, green],
                        32, (1080, 1080), "depth", 7, ("control/static.mp4", 48))
        plan = json.loads(_lcm_plan_path(vid).read_text(encoding="utf-8"))
        assert plan["image_ids"] == [None, str(green)]
        assert plan["injection"] == ["control/static.mp4", 48]
        restored = GenerateRequest(**plan["request"])
        assert restored.regions[0].embeddings[0].name == "artrium/rust"
        assert restored.regions[0].image_id is None


class TestCost:
    def test_each_embedding_region_is_a_layer(self):
        assert _layer_factor(form()) == 2
        assert _layer_factor(form(embeddings=[EmbeddingSpec(name="a", curve="swell")])) == 3


@pytest.mark.asyncio
class TestCheckEmbeddings:
    async def test_region_names_are_checked_too(self, monkeypatch):
        async def known():
            return ["artrium/other"]
        monkeypatch.setattr(vace, "embedding_names", known)
        with pytest.raises(HTTPException, match="artrium/rust"):
            await _check_embeddings(form())

    async def test_vace_refuses_region_embeddings(self, monkeypatch):
        async def known():
            return ["artrium/rust"]
        monkeypatch.setattr(vace, "embedding_names", known)
        with pytest.raises(HTTPException, match="AnimateLCM"):
            await _check_embeddings(form(engine="vace"))


def graph(pattern, seconds, dest):
    cmd = noise_command(pattern, seconds, dest)
    return cmd[cmd.index("-i") + 1]


class TestNoise:
    def test_static_is_luma_only(self, tmp_path):
        # noise does not take gray; with `alls` ffmpeg's silent YUV conversion
        # put colour into the chroma planes.
        g = graph("static", 6, tmp_path / "s.mp4")
        assert "c0s=" in g and "alls" not in g
        assert "d=6" in g

    def test_pixel_blocks_are_rgb_and_held(self, tmp_path):
        g = graph("pixel", 6, tmp_path / "p.mp4")
        assert "format=rgb24" in g and "flags=neighbor" in g and "fps=16" in g

    def test_length_is_clamped(self, tmp_path):
        assert "d=20," in graph("static", 999, tmp_path / "s.mp4")
        assert "d=2," in graph("static", 0, tmp_path / "s.mp4")

    def test_the_patterns_are_labelled(self):
        assert set(NOISE_PATTERNS) == {"static", "pixel"}
