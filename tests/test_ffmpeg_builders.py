"""
Unit tests for the pure ffmpeg argv builders (code review P1) — these
assemble a `list[str]` command and never touch a subprocess, so they're
testable without ffmpeg installed.
"""
from pathlib import Path

from services.improv.mux import (
    PIP_WIDTH_PCT_MAX,
    PIP_WIDTH_PCT_MIN,
    _hands_cmd,
    _pip_cmd,
    _synth_cmd,
    clamp_pip_width,
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
