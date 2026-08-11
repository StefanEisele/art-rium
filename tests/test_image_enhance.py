"""Auto-enhance ("Zauberstab") — calibration and pipeline tests.

`plan()` is where every judgement call lives, and it is pure, so most of these
assert on amounts rather than on pixels. The art-safe calibration is the thing
most likely to be broken by a well-meaning tweak, so its promises are pinned
here: nothing fires on an image that needs nothing, and no single lever can
run away even on a pathological input.
"""
from PIL import Image as PILImage
from PIL import ImageDraw, ImageStat

from services.image.enhance import (
    STRENGTH_DEFAULT,
    STRENGTH_MAX,
    Analysis,
    analyze,
    apply,
    build_tone_lut,
    clamp_strength,
    enhance_image,
    plan,
)


def _analysis(**overrides) -> Analysis:
    """A neutral, already-good image's measurements, with targeted overrides."""
    base = dict(
        black_point=0,
        white_point=255,
        mean=122.0,
        shadow_mass=0.1,
        highlight_mass=0.02,
        saturation=90.0,
        acutance=12.0,
        channel_means=(100.0, 100.0, 100.0),
    )
    base.update(overrides)
    return Analysis(**base)


def _gradient(size=(96, 64), low=0, high=255) -> PILImage.Image:
    img = PILImage.new("RGB", size)
    d = ImageDraw.Draw(img)
    for x in range(size[0]):
        v = round(low + (high - low) * x / max(1, size[0] - 1))
        d.line([(x, 0), (x, size[1])], fill=(v, v, v))
    return img


# ── Strength clamping ────────────────────────────────────────────────────────


def test_clamp_strength_bounds():
    assert clamp_strength(-40) == 0
    assert clamp_strength(1000) == STRENGTH_MAX
    assert clamp_strength(100) == 100


def test_clamp_strength_garbage_falls_back_to_default():
    # Deliberate: this value arrives from a query string, and a typo should
    # still enhance the image rather than silently return the original.
    assert clamp_strength(None) == STRENGTH_DEFAULT
    assert clamp_strength("banana") == STRENGTH_DEFAULT


# ── The art-safe promises ────────────────────────────────────────────────────


def test_already_good_image_is_left_alone():
    """An image that measures fine must come out untouched — the whole point
    of the art-safe calibration."""
    adj = plan(_analysis())
    assert adj.is_noop


def test_strength_zero_is_identity():
    adj = plan(_analysis(mean=30.0, shadow_mass=0.9, saturation=5.0), strength=0)
    assert adj.is_noop


def test_deliberately_dim_image_keeps_its_exposure():
    """Mean luma ~92 is the everyday case in this library, not underexposure.

    This is the regression that the measured `_MEAN_BAND` exists to prevent:
    a photographic target of ~122 pulled every image in the library up by
    20-50 %, which overrides exactly the muted look the user wants kept.
    """
    assert plan(_analysis(mean=92.0)).gamma == 1.0


def test_genuinely_underexposed_image_is_rescued_but_capped():
    adj = plan(_analysis(mean=40.0))
    assert adj.gamma < 1.0            # brightened
    assert adj.gamma >= 0.85          # but not hauled all the way to a photo norm


def test_no_lever_runs_away_on_a_pathological_image():
    """Every amount stays inside its cap even at max strength on a worst case."""
    adj = plan(
        _analysis(
            black_point=90, white_point=140, mean=8.0,
            shadow_mass=1.0, highlight_mass=1.0,
            saturation=0.0, acutance=0.0, channel_means=(200.0, 40.0, 40.0),
        ),
        strength=STRENGTH_MAX,
    )
    assert adj.black_point <= 12
    assert adj.white_point >= 255 - 18
    assert 0.85 <= adj.gamma <= 1.10
    assert adj.contrast <= 0.22
    assert adj.shadows <= 0.28
    assert adj.highlights <= 0.25
    assert adj.vibrance <= 0.16
    assert adj.definition <= 0.5
    assert all(abs(g - 1.0) <= 0.0401 for g in adj.wb_gains)   # 0.04 + float slack


def test_saturated_image_gets_almost_no_vibrance_or_white_balance():
    """Vibrance spares what is already colourful, and a strongly toned frame's
    "cast" is its palette — correcting it would be vandalism."""
    muted = plan(_analysis(saturation=20.0))
    vivid = plan(_analysis(saturation=125.0, channel_means=(130.0, 90.0, 70.0)))
    assert vivid.vibrance == 0.0            # already colourful enough
    assert muted.vibrance > 0.0
    # A strong orange cast damps grey-world trust to ~4 %, leaving under 2 %
    # of channel shift — a few levels at most, on a deliberately toned frame.
    assert all(abs(g - 1.0) < 0.02 for g in vivid.wb_gains)


def test_crisp_image_is_not_sharpened():
    """Sharpening an already-crisp render only buys halos."""
    assert plan(_analysis(acutance=20.0)).definition == 0.0
    assert plan(_analysis(acutance=1.0)).definition > 0.0


def test_strength_scales_amounts_monotonically():
    # Mild enough that neither end saturates its cap — otherwise this asserts
    # on the clamp rather than on the scaling.
    soft = plan(_analysis(shadow_mass=0.45, saturation=30.0), strength=50)
    hard = plan(_analysis(shadow_mass=0.45, saturation=30.0), strength=150)
    assert hard.shadows > soft.shadows
    assert hard.vibrance > soft.vibrance


# ── Tone LUT ─────────────────────────────────────────────────────────────────


def test_tone_lut_is_monotone_and_in_range():
    """A non-monotone curve would collapse two input tones onto one output —
    visible as posterisation in a gradient."""
    adj = plan(_analysis(black_point=40, white_point=190, mean=70.0, saturation=20.0))
    lut = build_tone_lut(adj)
    assert len(lut) == 768
    for band in range(3):
        values = lut[band * 256:(band + 1) * 256]
        assert all(0 <= v <= 255 for v in values)
        assert all(b >= a for a, b in zip(values, values[1:]))


def test_tone_lut_identity_when_nothing_to_do():
    lut = build_tone_lut(plan(_analysis()))
    assert lut[:256] == list(range(256))


# ── End to end ───────────────────────────────────────────────────────────────


def test_flat_low_contrast_image_gains_range():
    """A hazy image (tones squeezed into the middle) should come out using
    more of the scale than it went in with."""
    flat = _gradient(low=90, high=150)
    out, adj = enhance_image(flat)

    before = flat.convert("L").getextrema()
    after = out.convert("L").getextrema()
    assert (after[1] - after[0]) > (before[1] - before[0])
    assert adj.black_point > 0 or adj.white_point < 255


def test_enhance_preserves_size_and_mode():
    src = _gradient()
    out, _ = enhance_image(src)
    assert out.size == src.size
    assert out.mode == "RGB"


def test_enhance_preserves_alpha():
    """Alpha must survive: the gallery serves PNGs and a dropped alpha channel
    would silently flatten transparency to black on the next re-encode."""
    src = _gradient().convert("RGBA")
    src.putalpha(128)
    out, _ = enhance_image(src)
    assert out.mode == "RGBA"
    assert out.getchannel("A").getextrema() == (128, 128)


def test_full_scale_image_survives_round_trip_without_clipping():
    """A picture already spanning black to white must not have its ends
    crushed by the level stretch."""
    src = _gradient(low=0, high=255)
    out, _ = enhance_image(src)
    lo, hi = out.convert("L").getextrema()
    assert lo <= 4 and hi >= 251


def test_apply_is_deterministic():
    src = _gradient(low=40, high=200)
    adj = plan(analyze(src))
    assert apply(src, adj).tobytes() == apply(src, adj).tobytes()


def test_dark_image_gets_brighter_but_not_washed_out():
    dark = _gradient(low=0, high=70)
    out, _ = enhance_image(dark)
    assert ImageStat.Stat(out.convert("L")).mean[0] > ImageStat.Stat(dark.convert("L")).mean[0]
    # Still a dark picture — a rescue, not a reinterpretation.
    assert ImageStat.Stat(out.convert("L")).mean[0] < 140
