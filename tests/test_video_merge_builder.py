"""
Unit tests for the cross-job clip-merge ffmpeg argv builder (pure function,
no subprocess) — services/video/merge.py — plus the canvas the merge picks.
"""
from pathlib import Path
from types import SimpleNamespace

import pytest

from routers.video import _clip_dimensions, _merge_canvas
from services.video.merge import MergeInput, build_merge_command


def _cmd_str(cmd: list[str]) -> str:
    return " ".join(cmd)


def _clip(width=864, height=480, **upscale):
    """A stand-in for a VideoClip row — only the size fields matter here."""
    return SimpleNamespace(
        width=width, height=height,
        upscale_filename=upscale.get("filename"),
        upscale_width=upscale.get("up_w"),
        upscale_height=upscale.get("up_h"),
    )


class TestMergeCanvas:
    """Per-clip upscaling exists so the merge can consume it — which only works
    if the merge sizes its canvas from the rendition it actually feeds in."""

    def test_uniform_selection_uses_the_shared_size(self):
        assert _merge_canvas([_clip(), _clip(), _clip()]) == (864, 480)

    def test_upscaled_clip_reports_its_upscaled_size(self):
        c = _clip(filename="seg_0_up.mp4", up_w=1920, up_h=1080)
        assert _clip_dimensions(c) == (1920, 1080)

    def test_row_size_wins_when_there_is_no_upscale_file(self):
        # Columns without the file mean a removed upscale; the row's own
        # canvas is the truth then.
        c = _clip(up_w=1920, up_h=1080)
        assert _clip_dimensions(c) == (864, 480)

    def test_partly_upscaled_selection_keeps_the_larger_canvas(self):
        # "First clip wins" would scale the restored clip back down to 864x480
        # and throw the whole pass away.
        clips = [_clip(), _clip(filename="seg_1_up.mp4", up_w=1920, up_h=1080)]
        assert _merge_canvas(clips) == (1920, 1080)

    def test_missing_dimensions_fall_back_to_a_square_default(self):
        assert _merge_canvas([_clip(width=None, height=None)]) == (960, 960)

    def test_empty_selection_does_not_raise(self):
        assert _merge_canvas([]) == (960, 960)


class TestBuildMergeCommand:
    def test_silent_only_merge_has_no_audio_graph(self):
        cmd = build_merge_command(
            "ffmpeg",
            [
                MergeInput(path=Path("a.mp4"), has_audio=False),
                MergeInput(path=Path("b.mp4"), has_audio=False),
            ],
            Path("out.mp4"), 960, 960, 24,
        )
        s = _cmd_str(cmd)
        assert cmd[0] == "ffmpeg"
        assert cmd.count("-i") == 2                    # no lavfi silence inputs
        assert "anullsrc" not in s
        assert "concat=n=2:v=1:a=0[v]" in s
        assert "-c:a" not in cmd
        assert cmd[-1] == "out.mp4"

    def test_every_input_is_normalized_to_target(self):
        cmd = build_merge_command(
            "ffmpeg",
            [
                MergeInput(path=Path("a.mp4"), has_audio=False),
                MergeInput(path=Path("b.mp4"), has_audio=False),
            ],
            Path("out.mp4"), 1280, 704, 30,
        )
        graph = cmd[cmd.index("-filter_complex") + 1]
        for i in range(2):
            assert f"[{i}:v]scale=1280:704:force_original_aspect_ratio=decrease" in graph
            assert "pad=1280:704:(ow-iw)/2:(oh-ih)/2" in graph
            assert "fps=30" in graph
        assert cmd[cmd.index("-r") + 1] == "30"

    def test_mixed_audio_pads_silent_clips_with_anullsrc(self):
        cmd = build_merge_command(
            "ffmpeg",
            [
                MergeInput(path=Path("wan.mp4"), has_audio=False, duration=3.25),
                MergeInput(path=Path("minimax.mp4"), has_audio=True),
            ],
            Path("out.mp4"), 960, 960, 24,
        )
        s = _cmd_str(cmd)
        # Silence input trimmed to the silent clip's duration, feeding [a0]
        assert "-t 3.250 -i anullsrc=channel_layout=stereo:sample_rate=44100" in s
        graph = cmd[cmd.index("-filter_complex") + 1]
        assert "[2:a]anull[a0]" in graph               # lavfi input index 2 → clip 0
        assert "[1:a]aresample=44100" in graph         # real audio normalized
        assert "concat=n=2:v=1:a=1[v][a]" in graph
        assert "-c:a" in cmd and "aac" in cmd

    def test_silent_clip_without_duration_raises_in_mixed_merge(self):
        with pytest.raises(ValueError):
            build_merge_command(
                "ffmpeg",
                [
                    MergeInput(path=Path("wan.mp4"), has_audio=False),  # no duration
                    MergeInput(path=Path("minimax.mp4"), has_audio=True),
                ],
                Path("out.mp4"), 960, 960, 24,
            )

    def test_fewer_than_two_inputs_raises(self):
        with pytest.raises(ValueError):
            build_merge_command(
                "ffmpeg",
                [MergeInput(path=Path("a.mp4"), has_audio=False)],
                Path("out.mp4"), 960, 960, 24,
            )

    def test_video_encode_matches_tool_convention(self):
        cmd = build_merge_command(
            "ffmpeg",
            [
                MergeInput(path=Path("a.mp4"), has_audio=False),
                MergeInput(path=Path("b.mp4"), has_audio=False),
            ],
            Path("out.mp4"), 960, 960, 24,
        )
        assert cmd[cmd.index("-c:v") + 1] == "libx265"
        assert cmd[cmd.index("-crf") + 1] == "22"
        assert cmd[cmd.index("-pix_fmt") + 1] == "yuv420p10le"
        assert cmd[cmd.index("-tag:v") + 1] == "hvc1"
