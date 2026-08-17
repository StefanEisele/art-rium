"""Film grain over a still image — strength mapping and pixel behaviour.

The promises worth pinning are the ones that separate "film grain" from
"broken image": it is monochrome, it is bounded, it fades out at the ends of
the tonal range, and a preview rendered small predicts what the full-size
render will look like once both are fitted into the same box on screen.
"""
import statistics

from PIL import Image as PILImage
from PIL import ImageStat

from services.image.grain import (
    SIGMA_CEILING,
    STRENGTH_DEFAULT,
    STRENGTH_MAX,
    apply_grain,
    clamp_strength,
    preview_sigma,
    sigma_for,
)


def _flat(value=128, size=(160, 120), mode="RGB") -> PILImage.Image:
    fill = (value, value, value) if mode == "RGB" else value
    return PILImage.new(mode, size, fill)


def _channel_stdev(img: PILImage.Image) -> list[float]:
    return ImageStat.Stat(img.convert("RGB")).stddev[:3]


# ── Strength clamping ────────────────────────────────────────────────────────


def test_clamp_strength_bounds():
    assert clamp_strength(-10) == 0
    assert clamp_strength(999) == STRENGTH_MAX
    assert clamp_strength(30) == 30


def test_clamp_strength_garbage_reads_as_off():
    # Opposite default to enhance.clamp_strength, and deliberately so: an
    # unreadable grain value must not silently texture the picture.
    assert clamp_strength(None) == 0
    assert clamp_strength("loud") == 0


def test_sigma_scales_linearly_to_the_ceiling():
    assert sigma_for(0) == 0
    assert sigma_for(100) == SIGMA_CEILING
    assert sigma_for(50) == SIGMA_CEILING / 2


def test_default_strength_is_modest():
    # The one-click amount is a look, not the maximum the slider allows.
    assert 0 < STRENGTH_DEFAULT < STRENGTH_MAX


# ── Pixels ───────────────────────────────────────────────────────────────────


def test_zero_sigma_is_a_faithful_copy():
    src = _flat(100)
    out = apply_grain(src, 0)
    assert out.tobytes() == src.tobytes()


def test_grain_adds_variance_where_there_was_none():
    out = apply_grain(_flat(128), sigma_for(60))
    assert min(_channel_stdev(out)) > 2.0


def test_more_strength_means_more_grain():
    light = _channel_stdev(apply_grain(_flat(128), sigma_for(20)))
    heavy = _channel_stdev(apply_grain(_flat(128), sigma_for(80)))
    assert statistics.mean(heavy) > statistics.mean(light) * 1.5


def test_grain_is_monochrome():
    """R, G and B must move together — independent noise reads as colour
    speckle, i.e. as a compression fault rather than as film."""
    out = apply_grain(_flat(128), sigma_for(70)).convert("RGB")
    r, g, b = out.split()
    assert ImageStat.Stat(out).stddev[0] > 2.0        # the channels really are noisy
    assert r.tobytes() == g.tobytes() == b.tobytes()  # …with the same noise


def test_grain_does_not_shift_overall_brightness():
    """Noise is symmetric around zero, so the picture must not get lighter or
    darker — only textured."""
    src = _flat(128)
    out = apply_grain(src, sigma_for(60))
    assert abs(ImageStat.Stat(out.convert("L")).mean[0] - 128) < 2.0


def test_extremes_keep_less_grain_than_midtones():
    """Film grain fades towards pure black and pure white; without that, deep
    shadows — which this library is full of — turn into noise mush."""
    mid = statistics.mean(_channel_stdev(apply_grain(_flat(128), sigma_for(60))))
    dark = statistics.mean(_channel_stdev(apply_grain(_flat(6), sigma_for(60))))
    assert dark < mid


def test_alpha_survives_the_pass():
    src = PILImage.new("RGBA", (48, 48), (120, 120, 120, 77))
    out = apply_grain(src, sigma_for(50))
    assert out.mode == "RGBA"
    assert set(out.getchannel("A").tobytes()) == {77}


def test_extremes_clip_rather_than_wrap():
    """A near-white field must not sprout black pixels (and vice versa) — the
    signature of an 8-bit add that wrapped around instead of clipping."""
    assert min(apply_grain(_flat(253), sigma_for(100)).convert("L").tobytes()) > 128
    assert max(apply_grain(_flat(2), sigma_for(100)).convert("L").tobytes()) < 128


# ── Preview fidelity ─────────────────────────────────────────────────────────


def test_preview_sigma_shrinks_with_the_downscale():
    """A 1200 px preview of a 4800 px original carries a quarter of the sigma,
    because the browser fitting the full render into the same box averages
    away three quarters of it."""
    s = preview_sigma(100, source_edge=4800, preview_edge=1200)
    assert abs(s - SIGMA_CEILING / 4) < 0.01


def test_preview_sigma_never_amplifies_a_small_original():
    """An original smaller than the preview edge is not upscaled, so its grain
    must not be exaggerated either."""
    assert preview_sigma(100, source_edge=600, preview_edge=1200) == sigma_for(100)


def test_preview_sigma_keeps_a_visible_floor():
    """A huge original would otherwise scale grain down to invisibility and
    make the slider look broken."""
    assert preview_sigma(10, source_edge=20000, preview_edge=1200) > 0


def test_preview_sigma_off_is_off():
    assert preview_sigma(0, source_edge=4800, preview_edge=1200) == 0.0
