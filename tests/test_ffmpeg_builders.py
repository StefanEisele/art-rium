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
from services.improv.source import has_grain, source_filename, source_path
from services.video.audio_stretch import build_rubberband_filter, build_stretch_cmd
from services.video.look import (
    NOISE_CEILING,
    PRESET_BY_KEY,
    PRESETS,
    Look,
    clamp_strength,
    encode_args,
    preset_options,
    preview_command,
    preview_window,
    render_command,
)
from services.video.soundtrack import _mux_cmd, _mux_cmd_with_bed
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


class TestImprovSourceRendition:
    """Improv used to read `filepath` unconditionally, so a clip the user had
    deliberately grained came back out of the mux smooth again. Unlike the
    Instagram dispatch paths this is a choice, not a fixed precedence — but an
    unchecked box and a missing grain file have to behave identically."""

    def _video(self, **kw):
        from core.models import Video
        return Video(filename="clean.mp4", filepath="videos/clean.mp4", **kw)

    def test_grained_rendition_wins_when_asked_for(self):
        v = self._video(grain_filename="x_grain.mp4", grain_strength=30)
        assert source_path(v, use_grain=True).name == "x_grain.mp4"
        assert source_filename(v, use_grain=True) == "x_grain.mp4"

    def test_original_when_grain_is_declined(self):
        v = self._video(grain_filename="x_grain.mp4", grain_strength=30)
        assert source_path(v, use_grain=False).name == "clean.mp4"
        assert source_filename(v, use_grain=False) == "clean.mp4"

    def test_falls_back_to_the_original_when_nothing_was_grained(self):
        v = self._video()
        assert source_path(v, use_grain=True).name == "clean.mp4"
        assert source_filename(v, use_grain=True) == "clean.mp4"
        assert not has_grain(v)

    def test_grain_ignores_the_muxed_variant_it_was_built_from(self):
        # The grain pass reads the muxed file, so the grained rendition
        # already carries the soundtrack — resolving to muxed here would
        # hand ffmpeg the ungrained picture.
        v = self._video(muxed_filename="x_muxed.mp4", grain_filename="x_grain.mp4")
        assert source_path(v, use_grain=True).name == "x_grain.mp4"

    def test_declining_grain_keeps_todays_behaviour_of_reading_the_original(self):
        # Improv has never picked up an attached soundtrack on its own, and
        # this change is about grain — turning it off must not quietly start
        # muxing the song in as the bed.
        v = self._video(muxed_filename="x_muxed.mp4", grain_filename="x_grain.mp4")
        assert source_path(v, use_grain=False).name == "clean.mp4"

    def test_the_upscale_is_not_opt_out_the_way_grain_is(self):
        # Resolution is not a look. Declining grain must not also hand ffmpeg
        # the small generation canvas of a clip the user chose to enlarge.
        v = self._video(upscale_filename="x_upscale.mp4", upscale_resolution=1080)
        assert source_path(v, use_grain=False).name == "x_upscale.mp4"
        assert source_filename(v, use_grain=False) == "x_upscale.mp4"

    def test_grain_still_wins_over_the_upscale_it_was_built_from(self):
        # The grain pass reads the upscaled file, so the grained rendition is
        # already at the upscaled size.
        v = self._video(upscale_filename="x_upscale.mp4", grain_filename="x_grain.mp4")
        assert source_path(v, use_grain=True).name == "x_grain.mp4"


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


class TestSoundtrackMuxWithBed:
    """The song may keep the clip's own generated audio underneath it instead
    of replacing it — the same ambient bed the improv tool offers."""

    def _cmd(self, bed_volume=0.35, fade_start=10.0, fade_duration=1.0):
        return _mux_cmd_with_bed(
            "ffmpeg", Path("video.mp4"), Path("song.mp3"), Path("out.mp4"),
            fade_start=fade_start, fade_duration=fade_duration, bed_volume=bed_volume,
        )

    def test_mixes_both_streams_instead_of_mapping_one(self):
        cmd = self._cmd()
        fc = cmd[cmd.index("-filter_complex") + 1]
        assert "[0:a]volume=0.350[bed]" in fc     # the clip's own audio, quiet
        assert "[1:a]volume=1.000[lead]" in fc    # the song, at full level
        assert "amix=inputs=2" in fc
        assert "1:a:0" not in cmd                 # not the replace-audio mapping
        assert "[aout]" in cmd

    def test_video_stream_is_still_copied(self):
        cmd = self._cmd()
        assert "0:v:0" in cmd
        assert cmd[cmd.index("-c:v") + 1] == "copy"

    def test_fade_applies_to_the_mix_not_the_song_alone(self):
        # Fading only the music would leave the clip's own noise running on
        # by itself after the song had gone.
        cmd = self._cmd(fade_start=12.5, fade_duration=2.0)
        fc = cmd[cmd.index("-filter_complex") + 1]
        assert "amix=" in fc and fc.index("amix=") < fc.index("afade=")
        assert "st=12.500" in fc and "d=2.000" in fc

    def test_mix_does_not_rescale_the_song(self):
        # Measured: amix's default divides each input by their count, so
        # switching the bed on dropped the song by ~6 dB (-21.7 → -27.7).
        # Turning the bed on must change what is underneath the music, not
        # how loud the music is.
        cmd = self._cmd()
        fc = cmd[cmd.index("-filter_complex") + 1]
        assert "normalize=0" in fc

    def test_the_summed_peak_is_limited(self):
        # The price of normalize=0: song and bed now add up, and unlike the
        # improv path there is no loudness pass here to catch the peak.
        cmd = self._cmd()
        fc = cmd[cmd.index("-filter_complex") + 1]
        assert "alimiter=" in fc and fc.index("amix=") < fc.index("alimiter=")

    @pytest.mark.parametrize("asked,expected", [(0.0, BED_VOLUME_MIN), (5.0, BED_VOLUME_MAX)])
    def test_bed_volume_is_clamped_inside_the_filter(self, asked, expected):
        fc = self._cmd(bed_volume=asked)[self._cmd(bed_volume=asked).index("-filter_complex") + 1]
        assert f"[0:a]volume={expected:.3f}[bed]" in fc

    def test_shares_its_levels_with_the_improv_bed(self):
        # Two tools laying down "the same" bed at different levels would sound
        # like a bug, so both read one calibration.
        song = self._cmd(bed_volume=0.35)
        song_fc = song[song.index("-filter_complex") + 1]
        improv = _synth_cmd_with_bed(
            "ffmpeg", Path("src.mp4"), Path("rec.mp4"), Path("out.mp4"), bed_volume=0.35,
        )
        improv_fc = improv[improv.index("-filter_complex") + 1]
        assert "[0:a]volume=0.350[bed]" in song_fc
        assert "[0:a]volume=0.350[bed]" in improv_fc


class TestGrainDial:
    def test_maps_ui_scale_onto_the_noise_ceiling(self):
        assert Look(grain=100).filter_chain() == f"noise=c0s={NOISE_CEILING}:c0f=t"
        assert Look(grain=50).filter_chain() == (
            f"noise=c0s={round(NOISE_CEILING / 2)}:c0f=t")

    def test_luma_plane_only(self):
        # Noising chroma produces coloured speckle that reads as a compression
        # fault, not as grain.
        f = Look(grain=30).filter_chain()
        assert "c0s=" in f
        assert "c1s=" not in f and "c2s=" not in f and "alls=" not in f

    def test_grain_is_temporal(self):
        # Without the t flag the pattern freezes into a static dirt overlay.
        assert Look(grain=30).filter_chain().endswith("c0f=t")

    def test_strength_is_clamped_to_the_ui_range(self):
        assert clamp_strength(-5) == 0
        assert clamp_strength(500) == 100
        assert clamp_strength(None) == 0
        assert clamp_strength("nonsense") == 0
        assert clamp_strength(42.4) == 42


class TestLookChain:
    """Every dial is a no-op at 0 — that is what lets the chain be assembled by
    concatenation, so it is worth testing rather than assuming."""

    def test_an_empty_look_produces_no_filter_at_all(self):
        # Not "null": the callers read an empty chain as "there is no look",
        # and a null filter would have them encode a copy of the source.
        assert Look().filter_chain() == ""
        assert Look().is_empty

    def test_each_dial_alone_contributes_exactly_one_stage(self):
        for field, expect in [
            ("sharpen", "unsharp="),
            ("contrast", "eq=contrast="),
            ("saturation", "eq=saturation="),
            ("temperature", "colortemperature="),
            ("vignette", "vignette="),
            ("aberration", "rgbashift="),
            ("grain", "noise="),
        ]:
            chain = Look(**{field: 50}).filter_chain()
            assert chain.startswith(expect), (field, chain)
            assert chain.count(",") == 0, (field, chain)

    def test_contrast_and_saturation_share_one_eq(self):
        # Two eq passes would cost two filter invocations for one correction.
        chain = Look(contrast=20, saturation=-20).filter_chain()
        assert chain.count("eq=") == 1
        assert "contrast=" in chain and "saturation=" in chain

    def test_the_order_is_correct_grade_optics_grain(self):
        chain = Look(sharpen=40, contrast=20, temperature=30,
                     vignette=40, aberration=30, grain=30).filter_chain()
        order = [chain.index(x) for x in
                 ("eq=", "colortemperature=", "unsharp=", "vignette=",
                  "rgbashift=", "noise=")]
        assert order == sorted(order), chain

    def test_a_positive_temperature_dial_warms_the_picture(self):
        # colortemperature corrects *for* a temperature, so warming means
        # telling it a lower number than neutral. Easy to get backwards.
        warm = Look(temperature=100).filter_chain()
        cool = Look(temperature=-100).filter_chain()
        kelvin = lambda c: float(c.split("temperature=")[2].split(":")[0])   # noqa: E731
        assert kelvin(warm) < 6500 < kelvin(cool)

    def test_sharpening_leaves_chroma_alone(self):
        # On 4:2:0 a chroma sharpen amplifies the subsampling, not detail.
        assert Look(sharpen=60).filter_chain().endswith(":5:5:0.0")

    def test_saturation_bottoms_out_near_monochrome_not_at_zero(self):
        chain = Look(saturation=-100).filter_chain()
        assert "saturation=0.1500" in chain


class TestHalationSplice:
    """Halation is the one stage that needs the picture twice, so it is spliced
    into the chain rather than appended to it."""

    def test_it_stays_a_simple_graph(self):
        # One open input and one open output, or -vf will not take it.
        chain = Look(halation=50).filter_chain()
        assert chain.count("split") == 1
        assert "[hl_base][hl_glow]blend=" in chain
        assert not chain.endswith(";")

    def test_the_linear_runs_are_spliced_on_either_side(self):
        chain = Look(contrast=20, halation=50, grain=30).filter_chain()
        assert chain.index("eq=") < chain.index("split")
        assert chain.index("blend=") < chain.index("noise=")

    def test_only_highlights_bloom(self):
        # A bloom lifted off the midtones is a veil over the whole frame.
        assert "if(gt(val," in Look(halation=50).filter_chain()

    def test_screen_not_add(self):
        # add clips the very highlights the bloom is made of.
        assert "all_mode=screen" in Look(halation=50).filter_chain()


class TestLookPresets:
    def test_every_preset_produces_a_chain(self):
        for key, _label, _hint, look in PRESETS:
            assert look.filter_chain(), key

    def test_presets_ship_their_values_to_the_client(self):
        # So picking a chip fills the sliders without a second round trip.
        opts = preset_options()
        assert {o["key"] for o in opts} == set(PRESET_BY_KEY)
        assert all(isinstance(o["look"], dict) and o["look"] for o in opts)

    def test_korn_is_the_old_grain_pass_and_nothing_else(self):
        assert PRESET_BY_KEY["korn"].filter_chain().startswith("noise=")

    def test_klar_does_not_grain(self):
        # It exists to answer "just undo the softening", which grain undoes
        # nothing of.
        assert not PRESET_BY_KEY["klar"].has_grain


class TestLookPreviewWindow:
    def test_takes_the_window_from_the_middle(self):
        # The opening frames of an i2v clip are the source still barely
        # moving — the least representative place to judge a look on motion.
        start, length = preview_window(20.0, 4.0)
        assert (start, length) == (8.0, 4.0)

    def test_short_clip_is_graded_whole(self):
        assert preview_window(2.5, 4.0) == (0.0, 2.5)

    def test_failed_probe_falls_back_to_the_start(self):
        # probe_video_duration returns 0.0 rather than raising.
        assert preview_window(0.0, 4.0) == (0.0, 4.0)


class TestLookCmds:
    LOOK = Look(sharpen=30, contrast=15, grain=30)

    def test_full_render_copies_audio_through(self):
        # A soundtrack mux or a model's native track must survive the re-encode.
        cmd = render_command("ffmpeg", Path("in.mp4"), Path("out.mp4"), self.LOOK)
        assert cmd[cmd.index("-c:a") + 1] == "copy"
        assert cmd[-1] == str(Path("out.mp4"))

    def test_preview_seeks_before_input(self):
        # -ss after -i decodes up to the mark instead of seeking by index,
        # which is the difference between a 2s preview and a 20s one.
        cmd = preview_command(
            "ffmpeg", Path("in.mp4"), Path("out.mp4"), self.LOOK,
            start=8.0, length=4.0,
        )
        assert cmd.index("-ss") < cmd.index("-i")
        assert cmd[cmd.index("-ss") + 1] == "8.000"
        assert cmd[cmd.index("-t") + 1] == "4.000"
        assert "-an" in cmd

    def test_preview_and_full_render_encode_identically(self):
        # If they diverged, the look dialled in on the preview would not be the
        # look the full render produces.
        full = render_command("ffmpeg", Path("in.mp4"), Path("out.mp4"), self.LOOK)
        prev = preview_command(
            "ffmpeg", Path("in.mp4"), Path("p.mp4"), self.LOOK,
            start=1.0, length=4.0,
        )
        shared = encode_args(self.LOOK)
        assert all(a in full for a in shared)
        assert all(a in prev for a in shared)

    def test_tunes_the_encoder_for_grain_only_when_there_is_grain(self):
        # -tune grain tells the encoder to protect noise. With no noise to
        # protect it just costs bits the picture could have had — measured at
        # 0.9710 picture SSIM without the tune against 0.9585 with it.
        assert "-tune" in encode_args(Look(grain=30))
        assert "-tune" not in encode_args(Look(sharpen=40))

    def test_a_clean_look_is_encoded_at_a_higher_quality(self):
        # It can afford to be: there is no incompressible noise in it.
        grainy = encode_args(Look(grain=30))
        clean = encode_args(Look(sharpen=40))
        assert int(clean[clean.index("-crf") + 1]) < int(grainy[grainy.index("-crf") + 1])

    def test_the_crf_is_well_clear_of_the_old_default(self):
        # 30 was measured softening the picture; the whole point of the change.
        assert int(encode_args(Look(grain=30))[encode_args(Look(grain=30)).index("-crf") + 1]) <= 26

    def test_hevc_is_tagged_hvc1_for_browser_playback(self):
        cmd = render_command("ffmpeg", Path("in.mp4"), Path("out.mp4"), self.LOOK)
        assert cmd[cmd.index("-tag:v") + 1] == "hvc1"

    def test_an_empty_look_still_produces_a_runnable_command(self):
        # The endpoints refuse an empty look, but the builder must not emit
        # `-vf ` with nothing after it if one ever reaches it.
        cmd = render_command("ffmpeg", Path("in.mp4"), Path("out.mp4"), Look())
        assert cmd[cmd.index("-vf") + 1] == "null"


class TestLookFromDict:
    def test_unknown_keys_are_dropped_and_missing_ones_default(self):
        look = Look.from_dict({"grain": 40, "nonsense": 9})
        assert look.grain == 40 and look.sharpen == 0

    def test_out_of_range_values_clamp_rather_than_raise(self):
        look = Look.from_dict({"grain": 500, "contrast": -900})
        assert look.grain == 100 and look.contrast == -100

    def test_none_reads_as_an_empty_look(self):
        assert Look.from_dict(None).is_empty


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
