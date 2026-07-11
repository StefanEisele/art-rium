"""
Unit tests for the cross-job clip-merge ffmpeg argv builder (pure function,
no subprocess) — services/video/merge.py.
"""
from pathlib import Path

import pytest

from services.video.merge import MergeInput, build_merge_command


def _cmd_str(cmd: list[str]) -> str:
    return " ".join(cmd)


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
                MergeInput(path=Path("ltx.mp4"), has_audio=True),
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
                    MergeInput(path=Path("ltx.mp4"), has_audio=True),
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
