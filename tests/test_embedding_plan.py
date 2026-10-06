"""The embedding trainer's side of the process boundary (services/embedding/plan.py).

sd-scripts runs in its own venv, so nothing type-checks between the command
line built here and the argparse that reads it, or between the files it writes
and the listing that reads them back. The progress lines below are copied from
the real smoke run of 2026-09-28, not written to fit the regex.
"""
import tomllib
import uuid
from pathlib import Path

import pytest

from services.embedding.plan import (
    LR_DEFAULT,
    SAMPLE_PROMPTS,
    STEPS_MAX,
    VECTORS_DEFAULT,
    VECTORS_MAX,
    PlanError,
    active_file,
    active_name,
    build_command,
    dataset_toml,
    estimate_seconds,
    is_sample_filename,
    is_workshop_name,
    list_snapshots,
    normalize_name,
    parse_progress,
    plan_training,
    sample_prompts,
    snapshot_name,
    token_for,
    train_dir,
    trainer_env,
)

TID = uuid.UUID("12345678-1234-5678-1234-567812345678")


def plan(**kw):
    args = {"training_id": TID, "name": "Rostblüte", "image_count": 10}
    args.update(kw)
    return plan_training(**args)


class TestNames:
    def test_umlauts_are_spelled_out_not_dropped(self):
        assert normalize_name("Rostblüte") == "rostbluete"
        assert normalize_name("Große Echos") == "grosse-echos"

    @pytest.mark.parametrize("raw", ["", "a", "   ", "!!!", "x" * 41])
    def test_unusable_names_are_refused(self, raw):
        with pytest.raises(PlanError):
            normalize_name(raw)

    def test_the_token_comes_from_the_id_not_the_name(self):
        # A name is an English word often enough to already be a CLIP token,
        # and kohya refuses a token that exists.
        assert token_for(TID) == "ar1234567812"
        assert plan().token != plan().name

    def test_the_active_file_sits_above_its_workshop(self, tmp_path):
        assert active_name("echo") == "artrium/echo"
        assert active_file(tmp_path, "echo") == tmp_path / "artrium" / "echo.safetensors"
        assert train_dir(tmp_path, "echo") == tmp_path / "artrium" / "_train" / "echo"

    def test_snapshots_are_hidden_from_the_picker_by_their_path(self):
        assert snapshot_name("echo", 500) == "artrium/_train/echo/echo-step00000500"
        assert snapshot_name("echo", None) == "artrium/_train/echo/echo"
        assert is_workshop_name("artrium/_train/echo/echo-step00000500")
        assert is_workshop_name("artrium\\_train\\echo\\echo")
        assert not is_workshop_name("artrium/echo")
        assert not is_workshop_name("style-rustmagic")


class TestPlan:
    def test_defaults_are_the_measured_ones(self):
        p = plan()
        assert (p.vectors, p.learning_rate, p.template, p.init_word) == \
            (VECTORS_DEFAULT, LR_DEFAULT, "style", "painting")

    def test_the_learning_rate_is_not_the_docs_lora_value(self):
        # sd-scripts' TI docs say 1e-6; at that rate nothing is learnt.
        assert LR_DEFAULT == 5e-3
        assert plan(learning_rate=1e-6).learning_rate >= 1e-4

    def test_dials_are_clamped_not_refused(self):
        p = plan(vectors=500, steps=10**6)
        assert (p.vectors, p.steps) == (VECTORS_MAX, STEPS_MAX)

    def test_seventy_vectors_is_allowed(self):
        # kentskooking's own setting.
        assert plan(vectors=70).vectors == 70

    def test_too_few_pictures(self):
        with pytest.raises(PlanError, match="Mindestens"):
            plan(image_count=2)

    def test_an_object_template_starts_from_object(self):
        assert plan(template="object").init_word == "object"

    @pytest.mark.parametrize("word", ["two words", "ölbild", "x", "painting1"])
    def test_the_init_word_is_one_english_word(self, word):
        with pytest.raises(PlanError, match="Startwort"):
            plan(init_word=word)

    def test_an_unknown_template(self):
        with pytest.raises(PlanError):
            plan(template="lora")

    def test_the_estimate_follows_the_measured_rate(self):
        # 3.5 steps/s: 2000 steps is ten minutes, give or take the previews.
        assert 9 * 60 < estimate_seconds(2000) < 12 * 60


class TestCommand:
    def cmd(self, **kw):
        return build_command(Path("py.exe"), Path("train_textual_inversion.py"),
                             Path("ckpt.safetensors"), plan(**kw), Path("ds.toml"),
                             Path("prompts.txt"), Path("out"))

    def value(self, cmd, flag):
        return cmd[cmd.index(flag) + 1]

    def test_it_names_what_the_trainer_needs(self):
        cmd = self.cmd()
        assert cmd[:2] == ["py.exe", "train_textual_inversion.py"]
        assert self.value(cmd, "--output_name") == "rostbluete"
        assert self.value(cmd, "--token_string") == "ar1234567812"
        assert self.value(cmd, "--save_model_as") == "safetensors"
        assert self.value(cmd, "--learning_rate") == "0.005"
        assert "--use_style_template" in cmd and "--use_object_template" not in cmd

    def test_the_before_row_is_drawn(self):
        assert "--sample_at_first" in self.cmd()

    def test_snapshots_and_previews_share_a_cadence(self):
        cmd = self.cmd()
        assert self.value(cmd, "--save_every_n_steps") == self.value(cmd, "--sample_every_n_steps")

    def test_object_template(self):
        assert "--use_object_template" in self.cmd(template="object")

    def test_the_trainer_speaks_utf8_on_this_windows(self):
        # Without it the first Japanese log line raises UnicodeEncodeError.
        env = trainer_env("1")
        assert env["PYTHONUTF8"] == "1" and env["PYTHONIOENCODING"] == "utf-8"
        assert env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
        assert env["CUDA_VISIBLE_DEVICES"] == "1"


class TestFiles:
    def test_the_dataset_is_bucketed_toml(self):
        parsed = tomllib.loads(dataset_toml(Path("C:\\data\\set")))
        ds = parsed["datasets"][0]
        assert ds["enable_bucket"] is True and ds["resolution"] == 512
        # Backslashes would be TOML escapes.
        assert ds["subsets"][0]["image_dir"] == "C:/data/set"

    def test_one_prompt_line_per_preview_with_its_own_seed(self):
        lines = sample_prompts("artok").strip().splitlines()
        assert len(lines) == len(SAMPLE_PROMPTS)
        for line, (_, seed) in zip(lines, SAMPLE_PROMPTS):
            assert "artok" in line and f"--d {seed}" in line


class TestProgress:
    def test_real_lines_from_the_smoke_run(self):
        assert parse_progress("steps:   1%|          | 1/100 [00:02<04:06,  2.49s/it]") \
            == (1, 100, None)
        assert parse_progress("steps:   1%|          | 1/100 [00:02<04:06,  2.49s/it, loss=0.116]") \
            == (1, 100, 0.116)
        line = ("steps: 100%|\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588\u2588| 100/100 "
                "[00:28<00:00,  3.51it/s, loss=0.168]2026-09-28 20:34:04 INFO     model saved.")
        assert parse_progress(line) == (100, 100, 0.168)

    def test_the_preview_samplers_inner_bars_are_ignored(self):
        assert parse_progress("100%|\u2588\u2588\u2588\u2588| 20/20 [00:01<00:00, 13.35it/s]") is None

    def test_the_last_reading_in_a_chunk_wins(self):
        chunk = ("steps:  10%|#  | 10/100 [00:03<00:27, 3.3it/s, loss=0.2]\r"
                 "steps:  11%|#  | 11/100 [00:03<00:27, 3.3it/s, loss=0.19]\r")
        assert parse_progress(chunk) == (11, 100, 0.19)


class TestSnapshots:
    def test_files_and_previews_are_grouped_by_step(self, tmp_path):
        # The names the smoke run actually wrote, including the epoch-style
        # suffix sample_at_first uses for the "before" row.
        (tmp_path / "sample").mkdir()
        for name in ("echo-step00000250.safetensors", "echo-step00000500.safetensors",
                     "echo.safetensors", "other-step00000250.safetensors"):
            (tmp_path / name).write_bytes(b"x")
        for name in ("echo_e000000_00_20260928203337_1234.png",
                     "echo_000250_01_20260928203350_777.png",
                     "echo_000250_00_20260928203350_1234.png",
                     "other_000250_00_20260928203350_1234.png"):
            (tmp_path / "sample" / name).write_bytes(b"x")
        snaps = list_snapshots(tmp_path, "echo")
        assert [s["step"] for s in snaps] == [0, 250, 500]
        assert snaps[0]["file"] is None
        assert snaps[1]["file"] == "echo-step00000250.safetensors"
        assert snaps[1]["samples"] == ["echo_000250_00_20260928203350_1234.png",
                                       "echo_000250_01_20260928203350_777.png"]
        assert snaps[2]["samples"] == []

    def test_a_missing_folder_is_no_snapshots(self, tmp_path):
        assert list_snapshots(tmp_path / "nope", "echo") == []

    @pytest.mark.parametrize("name,ok", [
        ("echo_000250_00_20260928203350_1234.png", True),
        ("echo_e000000_00_20260928203337_1234.png", True),
        ("../echo_000250_00_20260928203350_1234.png", False),
        ("echo.safetensors", False),
        ("x.png", False),
    ])
    def test_the_preview_endpoint_only_serves_preview_names(self, name, ok):
        assert is_sample_filename(name) is ok
