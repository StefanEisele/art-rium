"""
Unit tests for the ComfyUI stage-label map and the progress payloads built on
top of it — pure dict work, no ComfyUI and no listener socket.
"""
import pytest

from routers import video as video_module
from routers.video import _attach_live_stage, _set_progress
from services.comfy.node_labels import build_label_map, label_for_class
from workers.comfy_listener import ComfyListener


# ── Label map ─────────────────────────────────────────────────────────────────

def test_curated_class_labels():
    assert label_for_class("KSamplerAdvanced") == "Sampling…"
    assert label_for_class("VAEDecodeAudio") == "Decoding audio…"
    assert label_for_class("VHS_VideoCombine") == "Encoding video…"


def test_unknown_class_falls_back_to_a_humanised_name():
    # Never "Node 47" — an unrecognised node still reads as something.
    assert label_for_class("SomeFutureSamplerNode") == "Some Future Sampler Node…"
    assert label_for_class("XYZ_LoadVideoThing") == "Load Video Thing…"


def test_build_label_map_covers_every_node():
    wf = {
        "1": {"class_type": "UNETLoader", "inputs": {}},
        "sampler": {"class_type": "KSampler", "inputs": {}},
        "9": {"class_type": "SaveImage", "inputs": {}},
    }
    assert build_label_map(wf) == {
        "1": "Loading model…",
        "sampler": "Sampling…",
        "9": "Saving image…",
    }


# ── Listener stage tracking ───────────────────────────────────────────────────

def _listener() -> ComfyListener:
    return ComfyListener(app_state=None)


@pytest.mark.asyncio
async def test_progress_event_carries_the_registered_label():
    lis = _listener()
    lis.register_node_labels("p1", {"s": {"class_type": "KSampler", "inputs": {}}})
    await lis._route({"type": "progress", "data": {
        "prompt_id": "p1", "node": "s", "value": 7, "max": 20,
    }})
    assert lis.get_step_progress("p1") == {
        "value": 7, "max": 20, "node": "s", "label": "Sampling…"
    }


@pytest.mark.asyncio
async def test_entering_a_node_clears_the_previous_step_counter():
    """A stale '20/20' from the sampler must not read as progress through the
    decode that follows it."""
    lis = _listener()
    lis.register_node_labels("p1", {
        "s": {"class_type": "KSampler", "inputs": {}},
        "d": {"class_type": "VAEDecode", "inputs": {}},
    })
    await lis._route({"type": "progress", "data": {
        "prompt_id": "p1", "node": "s", "value": 20, "max": 20,
    }})
    await lis._route({"type": "executing", "data": {"prompt_id": "p1", "node": "d"}})

    step = lis.get_step_progress("p1")
    assert step["label"] == "Decoding picture…"
    assert step["value"] is None and step["max"] is None


@pytest.mark.asyncio
async def test_finished_prompt_drops_its_stage_and_labels():
    lis = _listener()
    lis.register_node_labels("p1", {"s": {"class_type": "KSampler", "inputs": {}}})
    await lis._route({"type": "progress", "data": {
        "prompt_id": "p1", "node": "s", "value": 3, "max": 20,
    }})
    await lis._route({"type": "execution_success", "data": {"prompt_id": "p1"}})

    assert lis.get_step_progress("p1") is None
    assert "p1" not in lis._node_labels


@pytest.mark.asyncio
async def test_label_maps_are_capped():
    lis = _listener()
    for i in range(200):
        lis.register_node_labels(f"p{i}", {"s": {"class_type": "KSampler", "inputs": {}}})
    assert len(lis._node_labels) <= 64
    assert "p199" in lis._node_labels   # newest survives


# ── Progress payload ──────────────────────────────────────────────────────────

@pytest.fixture
def live_listener(monkeypatch):
    """A listener the video router's _attach_live_stage will read from."""
    lis = _listener()
    monkeypatch.setattr(video_module, "get_listener", lambda: lis)
    return lis


@pytest.mark.asyncio
async def test_sampler_step_scales_into_the_segment_band(live_listener):
    live_listener.register_node_labels("p1", {"s": {"class_type": "KSampler", "inputs": {}}})
    await live_listener._route({"type": "progress", "data": {
        "prompt_id": "p1", "node": "s", "value": 10, "max": 20,
    }})

    _set_progress("v1", "running", "Clip 1/2 — generating frames…", 20,
                  prompt_id="p1", band=(20, 60))
    out = _attach_live_stage(dict(video_module._progress["v1"]))

    assert out["pct"] == 40                       # halfway through the 20-60 band
    assert out["step"] == {"value": 10, "max": 20}
    assert out["detail"] == "Sampling… step 10/20"
    assert "_prompt_id" not in out and "_band" not in out


@pytest.mark.asyncio
async def test_counterless_stage_reports_its_name_without_moving_the_bar(live_listener):
    live_listener.register_node_labels("p1", {"u": {"class_type": "UNETLoader", "inputs": {}}})
    await live_listener._route({"type": "executing", "data": {"prompt_id": "p1", "node": "u"}})

    _set_progress("v2", "running", "Clip 1/2 — generating frames…", 20,
                  prompt_id="p1", band=(20, 60))
    out = _attach_live_stage(dict(video_module._progress["v2"]))

    assert out["detail"] == "Loading model…"
    assert out["pct"] == 20
    assert "step" not in out


def test_no_live_prompt_leaves_the_payload_alone(live_listener):
    _set_progress("v3", "finalizing", "Saving video…", 94)
    out = _attach_live_stage(dict(video_module._progress["v3"]))
    assert out == {"phase": "finalizing", "message": "Saving video…", "pct": 94}


# ── Band ownership ────────────────────────────────────────────────────────────
# Every ComfyUI node reports a step counter, and most of them are not a
# fraction of the job. A VHS loader reading a 93-frame control track announces
# "93/93" a second in; mapped into the band that reads as almost-finished, and
# the bar then falls back once the sampler starts. `band_node` names the node
# whose counter is allowed to move the percentage.

class _FixedListener:
    """Stands in for the live listener with one canned step event."""

    def __init__(self, step):
        self._step = step

    def get_step_progress(self, prompt_id):
        return self._step


def _staged(monkeypatch, step, **kw):
    import services.comfy.progress as progress_module

    monkeypatch.setattr(progress_module, "get_listener", lambda: _FixedListener(step))
    _set_progress("job", "generating", "…", 5, prompt_id="p1", band=(5, 95), **kw)
    return progress_module.attach_live_stage(dict(video_module._progress["job"]))


def test_a_loader_counter_does_not_drive_a_named_band(monkeypatch):
    loader = {"value": 93, "max": 93, "node": "vc_ctrl", "label": "Reading source video…"}
    out = _staged(monkeypatch, loader, band_node="vc_ks")
    assert out["pct"] == 5, "the loader finished, the job did not"
    # The stage text is still worth showing — only the percentage is withheld.
    assert "Reading source video…" in out["detail"]


def test_the_named_node_does_drive_the_band(monkeypatch):
    sampler = {"value": 3, "max": 6, "node": "vc_ks", "label": "Sampling…"}
    out = _staged(monkeypatch, sampler, band_node="vc_ks")
    assert out["pct"] == 50
    assert out["step"] == {"value": 3, "max": 6}


def test_without_a_named_node_any_counter_still_drives_the_band(monkeypatch):
    # The older callers (video generation, image upscale) pass no band_node and
    # must keep behaving exactly as before.
    loader = {"value": 93, "max": 93, "node": "vc_ctrl", "label": "Reading source video…"}
    assert _staged(monkeypatch, loader)["pct"] == 95


def test_private_keys_never_reach_the_client(monkeypatch):
    sampler = {"value": 1, "max": 6, "node": "vc_ks", "label": "Sampling…"}
    out = _staged(monkeypatch, sampler, band_node="vc_ks")
    assert not [k for k in out if k.startswith("_")]


# ── Submissions must be addressed to the listener ─────────────────────────────
# ComfyUI routes execution events to the session that submitted the prompt
# (execution.py: send_sync("executing", …, server.client_id)). A prompt posted
# without a client_id therefore produces no `executing` or `progress` events
# for the listener, and every live stage readout in the app stays empty for the
# whole render.

@pytest.mark.asyncio
async def test_post_workflow_addresses_the_listener(monkeypatch):
    import services.comfy.client as client_module

    sent = {}

    class _Resp:
        status_code = 200

        @staticmethod
        def json():
            return {"prompt_id": "p-1"}

    class _Client:
        async def post(self, url, json=None, timeout=None):
            sent.update(json or {})
            return _Resp()

    class _Listener:
        client_id = "listener-sid"

    monkeypatch.setattr(client_module, "get_listener", lambda: _Listener(), raising=False)
    import workers.comfy_listener as listener_module
    monkeypatch.setattr(listener_module, "get_listener", lambda: _Listener())

    pid = await client_module.post_workflow(_Client(), {"1": {"class_type": "KSampler"}})
    assert pid == "p-1"
    assert sent.get("client_id") == "listener-sid", "events would go nowhere without this"


@pytest.mark.asyncio
async def test_an_explicit_client_id_wins(monkeypatch):
    import services.comfy.client as client_module

    sent = {}

    class _Resp:
        status_code = 200

        @staticmethod
        def json():
            return {"prompt_id": "p-2"}

    class _Client:
        async def post(self, url, json=None, timeout=None):
            sent.update(json or {})
            return _Resp()

    await client_module.post_workflow(_Client(), {}, client_id="explicit")
    assert sent.get("client_id") == "explicit"
