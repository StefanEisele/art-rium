"""
Unit tests for the pure ffmpeg argv builders (code review P1) — these
assemble a `list[str]` command and never touch a subprocess, so they're
testable without ffmpeg installed.
"""
import pytest

from pathlib import Path

from services.improv.mux import (
    BED_VOLUME_MAX,
    BED_VOLUME_MIN,
    PIP_WIDTH_PCT_MAX,
    PIP_WIDTH_PCT_MIN,
    _hands_cmd,
    _pip_cmd,
    _synth_cmd,
    _synth_cmd_with_bed,
    clamp_bed_volume,
    clamp_pip_width,
)
from services.video.audio_stretch import build_rubberband_filter, build_stretch_cmd
from services.video.grain import (
    NOISE_CEILING,
    _encode_args,
    _grain_cmd,
    _grain_preview_cmd,
    clamp_strength,
    grain_filter,
    preview_window,
)
from services.video.soundtrack import _mux_cmd
from workers.video_generator import _scale_pad, _single_cmd, _slideshow_cmd


class TestClampPipWidth:
    def test_within_range_unchanged(self):
        assert clamp_pip_width(0.24) == 0.24

    def test_below_min_clamped(self):
        assert clamp_pip_width(0.01) == PIP_WIDTH_PCT_MIN

    def test_above_max_clamped(self):
        assert clamp_pip_width(0.99) == PIP_WIDTH_PCT_MAX


class TestClampBedVolume:
    def test_within_range_unchanged(self):
        assert clamp_bed_volume(0.35) == 0.35

    def test_below_min_clamped(self):
        assert clamp_bed_volume(0.0) == BED_VOLUME_MIN

    def test_above_max_clamped(self):
        assert clamp_bed_volume(5.0) == BED_VOLUME_MAX


class TestImprovMuxCmds:
    def test_synth_cmd_maps_video_from_source_audio_from_recording(self):
        cmd = _synth_cmd("ffmpeg", Path("source.mp4"), Path("rec.mp4"), Path("out.mp4"))
        assert cmd[0] == "ffmpeg"
        assert "-i" in cmd and str(Path("source.mp4")) in cmd
        assert str(Path("rec.mp4")) in cmd
        assert cmd[-1] == str(Path("out.mp4"))
        assert "0:v:0" in cmd  # video from first input (source)
        assert "1:a:0" in cmd  # audio from second input (recording)
        assert "-shortest" in cmd

    def test_synth_cmd_with_bed_mixes_both_audio_streams(self):
        cmd = _synth_cmd_with_bed(
            "ffmpeg", Path("source.mp4"), Path("rec.mp4"), Path("out.mp4"),
            bed_volume=0.35, piano_volume=1.0,
        )
        assert cmd.count("-i") == 2
        assert "0:v:0" in cmd  # video still comes from the source
        assert "[aout]" in cmd  # mixed audio output, not a raw stream index
        filter_complex = cmd[cmd.index("-filter_complex") + 1]
        assert "volume=0.350" in filter_complex
        assert "volume=1.000" in filter_complex
        assert "amix=inputs=2" in filter_complex
        assert cmd[-1] == str(Path("out.mp4"))

    def test_hands_cmd_copies_video_only_recording_input(self):
        cmd = _hands_cmd("ffmpeg", Path("rec.mp4"), Path("out.mp4"))
        assert cmd.count("-i") == 1
        assert str(Path("rec.mp4")) in cmd
        assert "copy" in cmd

    def test_pip_cmd_uses_both_inputs_and_overlay_filter(self):
        cmd = _pip_cmd("ffmpeg", Path("bg.mp4"), Path("inset.mp4"), Path("out.mp4"), corner="tr", width_pct=0.24)
        assert cmd.count("-i") == 2
        assert any("overlay=" in c for c in cmd)
        assert cmd[-1] == str(Path("out.mp4"))

    def test_pip_cmd_corner_changes_overlay_expression(self):
        tr = _pip_cmd("ffmpeg", Path("bg.mp4"), Path("inset.mp4"), Path("out.mp4"), corner="tr")
        bl = _pip_cmd("ffmpeg", Path("bg.mp4"), Path("inset.mp4"), Path("out.mp4"), corner="bl")
        filt_tr = next(c for c in tr if "overlay=" in c)
        filt_bl = next(c for c in bl if "overlay=" in c)
        assert filt_tr != filt_bl

    def test_pip_cmd_unknown_corner_falls_back_to_default(self):
        default = _pip_cmd("ffmpeg", Path("bg.mp4"), Path("inset.mp4"), Path("out.mp4"), corner="tr")
        unknown = _pip_cmd("ffmpeg", Path("bg.mp4"), Path("inset.mp4"), Path("out.mp4"), corner="nonsense")
        assert default == unknown


class TestSoundtrackMuxCmd:
    def test_maps_video_from_first_audio_from_second(self):
        cmd = _mux_cmd(
            "ffmpeg", Path("video.mp4"), Path("song.mp3"), Path("out.mp4"),
            fade_start=10.0, fade_duration=1.0,
        )
        assert "0:v:0" in cmd
        assert "1:a:0" in cmd
        assert cmd[-1] == str(Path("out.mp4"))

    def test_fade_expression_uses_given_start_and_duration(self):
        cmd = _mux_cmd(
            "ffmpeg", Path("video.mp4"), Path("song.mp3"), Path("out.mp4"),
            fade_start=12.5, fade_duration=2.0,
        )
        afade = next(c for c in cmd if c.startswith("afade="))
        assert "st=12.500" in afade
        assert "d=2.000" in afade


class TestGrainFilter:
    def test_maps_ui_scale_onto_the_noise_ceiling(self):
        assert grain_filter(100) == f"noise=c0s={NOISE_CEILING}:c0f=t"
        assert grain_filter(50) == f"noise=c0s={round(NOISE_CEILING / 2)}:c0f=t"

    def test_luma_plane_only(self):
        # Noising chroma produces coloured speckle that reads as a compression
        # fault, not as grain.
        f = grain_filter(30)
        assert "c0s=" in f
        assert "c1s=" not in f and "c2s=" not in f and "alls=" not in f

    def test_grain_is_temporal(self):
        # Without the t flag the pattern freezes into a static dirt overlay.
        assert grain_filter(30).endswith("c0f=t")

    def test_strength_is_clamped_to_the_ui_range(self):
        assert clamp_strength(-5) == 0
        assert clamp_strength(500) == 100
        assert clamp_strength(None) == 0
        assert clamp_strength("nonsense") == 0
        assert clamp_strength(42.4) == 42


class TestGrainPreviewWindow:
    def test_takes_the_window_from_the_middle(self):
        # The opening frames of an i2v clip are the source still barely
        # moving — the least representative place to judge grain on motion.
        start, length = preview_window(20.0, 4.0)
        assert (start, length) == (8.0, 4.0)

    def test_short_clip_is_graded_whole(self):
        assert preview_window(2.5, 4.0) == (0.0, 2.5)

    def test_failed_probe_falls_back_to_the_start(self):
        # probe_video_duration returns 0.0 rather than raising.
        assert preview_window(0.0, 4.0) == (0.0, 4.0)


class TestGrainCmds:
    def test_full_render_copies_audio_through(self):
        # A soundtrack mux or LTX's native track must survive the re-encode.
        cmd = _grain_cmd("ffmpeg", Path("in.mp4"), Path("out.mp4"), 30)
        assert cmd[cmd.index("-c:a") + 1] == "copy"
        assert cmd[-1] == str(Path("out.mp4"))

    def test_preview_seeks_before_input(self):
        # -ss after -i decodes up to the mark instead of seeking by index,
        # which is the difference between a 2s preview and a 20s one.
        cmd = _grain_preview_cmd(
            "ffmpeg", Path("in.mp4"), Path("out.mp4"), 30, start=8.0, length=4.0,
        )
        assert cmd.index("-ss") < cmd.index("-i")
        assert cmd[cmd.index("-ss") + 1] == "8.000"
        assert cmd[cmd.index("-t") + 1] == "4.000"
        assert "-an" in cmd

    def test_preview_and_full_render_encode_identically(self):
        # If they diverged, the strength dialled in on the preview would not
        # be the strength the full render produces.
        full = _grain_cmd("ffmpeg", Path("in.mp4"), Path("out.mp4"), 44)
        prev = _grain_preview_cmd(
            "ffmpeg", Path("in.mp4"), Path("p.mp4"), 44, start=1.0, length=4.0,
        )
        shared = _encode_args(44)
        assert all(a in full for a in shared)
        assert all(a in prev for a in shared)

    def test_tunes_the_encoder_for_grain(self):
        # Without -tune grain the encoder spends its bits smoothing the noise
        # straight back out.
        cmd = _grain_cmd("ffmpeg", Path("in.mp4"), Path("out.mp4"), 30)
        assert cmd[cmd.index("-tune") + 1] == "grain"

    def test_hevc_is_tagged_hvc1_for_browser_playback(self):
        cmd = _grain_cmd("ffmpeg", Path("in.mp4"), Path("out.mp4"), 30)
        assert cmd[cmd.index("-tag:v") + 1] == "hvc1"


class TestAudioStretch:
    def test_default_detector_is_percussive(self):
        assert build_rubberband_filter(1.5) == "rubberband=tempo=1.500000:detector=percussive"

    def test_ratio_below_one_needs_no_chaining(self):
        # rife_multiplier=4 -> ratio 1/4; rubberband takes any ratio in one stage.
        filt = build_rubberband_filter(0.25)
        assert filt == "rubberband=tempo=0.250000:detector=percussive"

    def test_detector_is_overridable(self):
        filt = build_rubberband_filter(0.5, detector="compound")
        assert filt == "rubberband=tempo=0.500000:detector=compound"

    def test_non_positive_ratio_raises(self):
        with pytest.raises(ValueError):
            build_rubberband_filter(0)

    def test_stretch_cmd_trims_padding_before_stretching(self):
        cmd = build_stretch_cmd(
            "ffmpeg", Path("in.mp4"), Path("out.mp4"),
            ratio=0.5, native_audio_duration=2.042, target_duration=9.0,
        )
        assert cmd[0] == "ffmpeg"
        assert "0:v:0" in cmd
        assert "0:a:0" in cmd
        assert "copy" in cmd
        af = next(c for c in cmd if c.startswith("atrim="))
        assert "atrim=end=2.042000" in af
        assert "asetpts=PTS-STARTPTS" in af
        assert "rubberband=tempo=0.500000:detector=percussive" in af
        assert "apad" in af
        assert "9.000" in cmd
        assert cmd[-1] == str(Path("out.mp4"))


class TestSlideshowCmds:
    def test_scale_pad_forces_target_dimensions(self):
        expr = _scale_pad()
        assert "scale=1080:1920" in expr
        assert "pad=1080:1920" in expr

    def test_single_cmd_loops_one_image(self):
        cmd = _single_cmd("ffmpeg", Path("img.png"), Path("out.mp4"))
        assert "-loop" in cmd
        assert str(Path("img.png")) in cmd
        assert cmd[-1] == str(Path("out.mp4"))

    def test_slideshow_cmd_has_one_input_per_image(self):
        imgs = [Path("a.png"), Path("b.png"), Path("c.png")]
        cmd = _slideshow_cmd("ffmpeg", imgs, Path("out.mp4"))
        assert cmd.count("-loop") == len(imgs)
        for img in imgs:
            assert str(img) in cmd

    def test_slideshow_cmd_chains_xfade_for_each_transition(self):
        imgs = [Path("a.png"), Path("b.png"), Path("c.png")]
        cmd = _slideshow_cmd("ffmpeg", imgs, Path("out.mp4"))
        filter_complex = cmd[cmd.index("-filter_complex") + 1]
        # n images -> n-1 transitions
        assert filter_complex.count("xfade=") == len(imgs) - 1
        assert filter_complex.endswith("[out]")

    def test_slideshow_cmd_two_images_single_transition(self):
        imgs = [Path("a.png"), Path("b.png")]
        cmd = _slideshow_cmd("ffmpeg", imgs, Path("out.mp4"))
        filter_complex = cmd[cmd.index("-filter_complex") + 1]
        assert filter_complex.count("xfade=") == 1
