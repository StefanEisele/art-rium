"""
Unit tests for the two-stage AnimateLCM split (services/comfy/animatelcm.py).

The split only works if the second stage is the same graph as the second half
of the one-shot graph — same model, same adapters, same seed, same denoise —
with a video file where the base sampler's decode used to be. Anything that
drifts between them turns "finish it" into "render something else".

The properties pinned here are exactly the ones that would break silently on
the GPU rather than raise: a base pass that still contains the hires sampler, a
finishing pass built on a bare checkpoint (which would refine the picture into
different material), a preview node that is really the un-interpolated file, or
a second stage seeded differently from the first.
"""
import pytest

from services.comfy.animatelcm import (
    HIRES_STEPS,
    AnimateLcmRequest,
    Region,
    build_animatelcm_base_workflow,
    build_animatelcm_hires_workflow,
    build_animatelcm_workflow,
    clamp_hires,
    resolve_seed,
)


def request(**kw):
    base = dict(
        control_video="C:/tracks/turntable.mp4",
        prompt="rust and blue paint pour",
        reference_image="ref.png",
        length=32,
        source_frames=32,
        source_width=1080,
        source_height=1080,
        seed=4242,
    )
    base.update(kw)
    return AnimateLcmRequest(**base)


def classes(wf):
    return [node["class_type"] for node in wf.values()]


def node_of(wf, class_type):
    hits = [n for n in wf.values() if n["class_type"] == class_type]
    assert hits, f"no {class_type} in graph"
    return hits


# ── Seeds ────────────────────────────────────────────────────────────────────

def test_resolve_seed_passes_a_real_seed_through():
    assert resolve_seed(7) == 7
    assert resolve_seed(0) == 0


def test_resolve_seed_invents_one_for_minus_one():
    seed = resolve_seed(-1)
    assert 0 <= seed < 2 ** 32


def test_both_stages_sample_with_the_same_seed():
    """The seed is resolved by the caller and stored, because a second pass
    seeded differently from the first is refining someone else's frames."""
    req = request(seed=98765)
    base, _, _ = build_animatelcm_base_workflow(req)
    hires, _ = build_animatelcm_hires_workflow(req, "C:/out/base.mp4")
    assert {n["inputs"]["seed"] for n in node_of(base, "KSampler")} == {98765}
    assert {n["inputs"]["seed"] for n in node_of(hires, "KSampler")} == {98765}


# ── Stage one ────────────────────────────────────────────────────────────────

def test_base_workflow_stops_before_the_hires_pass():
    wf, _, _ = build_animatelcm_base_workflow(request(hires=True))
    assert len(node_of(wf, "KSampler")) == 1
    for absent in ("UpscaleModelLoader", "VAEEncode", "LineArtPreprocessor",
                   "MiDaS-DepthMapPreprocessor"):
        assert absent not in classes(wf), f"{absent} belongs to the second stage"


def test_base_workflow_keeps_the_control_track():
    """The base pass is where the geometry enters; without it the preview would
    not be a preview of anything."""
    wf, _, _ = build_animatelcm_base_workflow(request())
    assert "ControlNetApplyAdvanced" in classes(wf)
    assert "EmptyLatentImage" in classes(wf)


def test_base_workflow_writes_two_files_when_interpolating():
    """The preview is what the user watches; the base is what stage two reads.
    Interpolating the latter would hand the expensive pass four times the
    frames."""
    wf, preview, base = build_animatelcm_base_workflow(request(rife=4))
    assert preview != base
    assert wf[base]["inputs"]["frame_rate"] == 8
    assert wf[preview]["inputs"]["frame_rate"] == 32
    # Only the preview branch goes through RIFE.
    rife = node_of(wf, "RIFE VFI")
    assert len(rife) == 1
    assert wf[preview]["inputs"]["images"][0] != wf[base]["inputs"]["images"][0]
    assert wf[base]["inputs"]["filename_prefix"].endswith("_base")


def test_base_workflow_writes_one_file_without_interpolation():
    """Nothing to interpolate: both names refer to the same node, and the
    runner must not delete it twice."""
    wf, preview, base = build_animatelcm_base_workflow(request(rife=1))
    assert preview == base
    assert "RIFE VFI" not in classes(wf)


def test_base_and_preview_are_saved_outputs():
    wf, preview, base = build_animatelcm_base_workflow(request(rife=2))
    for node in (preview, base):
        assert wf[node]["class_type"] == "VHS_VideoCombine"
        assert wf[node]["inputs"]["save_output"] is True


# ── Stage two ────────────────────────────────────────────────────────────────

def test_hires_workflow_reads_the_base_render():
    wf, _ = build_animatelcm_hires_workflow(request(), "C:/out/base.mp4")
    loaders = [n for n in node_of(wf, "VHS_LoadVideoPath")
               if n["inputs"]["video"] == "C:/out/base.mp4"]
    assert loaders, "the finishing pass never opens the file it is finishing"
    assert "EmptyLatentImage" not in classes(wf), "stage two refines, it does not sample from noise"


def test_hires_workflow_keeps_the_whole_model_stack():
    """The reason this graph does not boil is that the hires sampler runs on
    the AnimateDiff-wrapped, IP-Adapter-loaded model. A bare checkpoint here
    would refine the base into different material, frame by frame."""
    wf, _ = build_animatelcm_hires_workflow(request(), "base.mp4")
    for required in ("ADE_UseEvolvedSampling", "ADE_ApplyAnimateDiffModelSimple",
                     "IPAdapterUnifiedLoader", "IPAdapterAdvanced",
                     "ADE_LoopedUniformContextOptions"):
        assert required in classes(wf), required

    sampler = node_of(wf, "KSampler")[0]
    model_node = wf[sampler["inputs"]["model"][0]]
    assert model_node["class_type"] == "IPAdapterAdvanced"


def test_hires_workflow_derives_its_controlnets_from_the_base_frames():
    wf, _ = build_animatelcm_hires_workflow(request(), "base.mp4")
    loader_id = next(nid for nid, n in wf.items()
                     if n["class_type"] == "VHS_LoadVideoPath")
    for pre in ("LineArtPreprocessor", "MiDaS-DepthMapPreprocessor"):
        assert node_of(wf, pre)[0]["inputs"]["image"] == [loader_id, 0]
    # ...and so does the upscale that feeds the sampler.
    assert node_of(wf, "ImageUpscaleWithModel")[0]["inputs"]["image"] == [loader_id, 0]


def test_hires_workflow_samples_at_the_hires_denoise():
    req = request(hires_denoise=0.45)
    wf, _ = build_animatelcm_hires_workflow(req, "base.mp4")
    sampler = node_of(wf, "KSampler")[0]
    assert sampler["inputs"]["denoise"] == pytest.approx(clamp_hires(0.45))
    assert sampler["inputs"]["steps"] == HIRES_STEPS


def test_hires_workflow_interpolates_after_refining():
    wf, out = build_animatelcm_hires_workflow(request(rife=3), "base.mp4")
    assert wf[out]["inputs"]["frame_rate"] == 24
    rife = node_of(wf, "RIFE VFI")[0]
    assert wf[rife["inputs"]["frames"][0]]["class_type"] == "VAEDecode"


def test_hires_workflow_does_not_reapply_the_control_track():
    """The base render already carries the geometry. Loading the depth pass
    again would put a second ControlNet on top of the two derived ones."""
    wf, _ = build_animatelcm_hires_workflow(request(), "base.mp4")
    videos = [n["inputs"]["video"] for n in node_of(wf, "VHS_LoadVideoPath")]
    assert "C:/tracks/turntable.mp4" not in videos
    assert "DepthAnythingV2Preprocessor" not in classes(wf)


# ── The two stages against the one-shot graph ────────────────────────────────

def test_the_split_reproduces_the_one_shot_hires_pass():
    """Same sampler settings, same ControlNet strengths, same model — the only
    difference the split is allowed to make is where the frames came from."""
    req = request(rife=2)
    whole, _ = build_animatelcm_workflow(req)
    part, _ = build_animatelcm_hires_workflow(req, "base.mp4")

    whole_hires = sorted(
        (n["inputs"]["denoise"], n["inputs"]["steps"], n["inputs"]["seed"])
        for n in node_of(whole, "KSampler")
    )[0]           # the hires sampler is the one below denoise 1.0
    part_hires = (
        node_of(part, "KSampler")[0]["inputs"]["denoise"],
        node_of(part, "KSampler")[0]["inputs"]["steps"],
        node_of(part, "KSampler")[0]["inputs"]["seed"],
    )
    assert whole_hires == part_hires

    def cn_settings(wf):
        return sorted(
            (n["inputs"]["strength"], n["inputs"]["end_percent"])
            for n in node_of(wf, "ControlNetApplyAdvanced")
        )
    # The one-shot graph also holds the base pass's depth ControlNet; the two
    # the hires pass uses must appear identically in both.
    assert set(cn_settings(part)).issubset(set(cn_settings(whole)))


def test_regions_survive_into_both_stages():
    req = request(
        reference_image=None,
        mask_video="C:/tracks/ids.mp4",
        regions=[Region(color=(255, 0, 0), reference="a.png"),
                 Region(color=(0, 255, 0), reference="b.png")],
    )
    base, _, _ = build_animatelcm_base_workflow(req)
    hires, _ = build_animatelcm_hires_workflow(req, "base.mp4")
    for wf in (base, hires):
        assert len(node_of(wf, "ColorToMask")) == 2
        assert len(node_of(wf, "IPAdapterAdvanced")) == 2
        assert all("attn_mask" in n["inputs"] for n in node_of(wf, "IPAdapterAdvanced"))


def test_both_stage_builders_validate_their_input():
    with pytest.raises(ValueError):
        build_animatelcm_base_workflow(request(reference_image=None))
    with pytest.raises(ValueError):
        build_animatelcm_hires_workflow(request(reference_image=None), "base.mp4")
    with pytest.raises(ValueError):
        build_animatelcm_base_workflow(
            request(regions=[Region(color=(255, 0, 0), reference="a.png")])
        )
