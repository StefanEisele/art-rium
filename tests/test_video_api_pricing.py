"""MiniMax H3 cost arithmetic.

This is the module that decides what something costs, so its promises are
pinned hard: the rate card is applied as documented, the free-image allowance
is respected, reservations round *up* and displayed prices round normally, and
nothing silently prices an unknown mode.
"""
from decimal import Decimal

import pytest

from services.video_api import pricing


def _price(**kw):
    kw.setdefault("usd_eur_rate", Decimal("1.08"))
    kw.setdefault("safety_factor", Decimal("1.10"))
    return pricing.price(**kw)


# ── Rate card ────────────────────────────────────────────────────────────────


def test_768p_and_2k_rates_match_the_published_card():
    assert pricing.OUTPUT_RATE_USD["768P"] == Decimal("0.08")
    assert pricing.OUTPUT_RATE_USD["2K"] == Decimal("0.13")
    assert pricing.REGENERATION_RATE_USD == Decimal("0.05")


def test_video_cost_is_seconds_times_rate():
    b = _price(kind="generate_2k", duration_s=10)
    assert b.video_usd == Decimal("1.30")
    assert b.cost_usd == Decimal("1.30")


def test_768p_is_the_cheap_tier():
    cheap = _price(kind="generate_768p", duration_s=10).cost_usd
    dear = _price(kind="generate_2k", duration_s=10).cost_usd
    assert cheap < dear


def test_regeneration_is_cheaper_than_generating_at_2k():
    regen = _price(kind="regenerate_2k", duration_s=10).cost_usd
    fresh = _price(kind="generate_2k", duration_s=10).cost_usd
    assert regen == Decimal("0.50")
    assert regen < fresh


def test_unknown_kind_and_resolution_raise():
    with pytest.raises(ValueError):
        _price(kind="generate_4k", duration_s=5)
    with pytest.raises(ValueError):
        pricing.output_rate("1080p")


# ── Reference images ─────────────────────────────────────────────────────────


def test_first_five_images_are_free():
    assert pricing.billable_images(5) == 0
    assert pricing.billable_images(0) == 0
    assert pricing.billable_images(8) == 3


def test_extra_images_are_charged():
    b = _price(kind="generate_2k", duration_s=4, image_count=8)
    assert b.image_usd == Decimal("0.12")          # 3 × $0.04
    assert b.cost_usd == Decimal("0.64")           # 4 × 0.13 + 0.12


def test_regeneration_charges_images_at_the_reduced_rate():
    b = _price(kind="regenerate_2k", duration_s=4, image_count=7)
    assert b.image_usd == Decimal("0.05")          # 2 × $0.025


# ── Input video (absent from the original plan, present in the rate card) ────


def test_input_video_is_billed_per_second_at_the_output_rate():
    """The plan's formula omits this. Leaving it out would under-reserve every
    reference-to-video job — the exact case the budget exists to contain."""
    without = _price(kind="generate_2k", duration_s=6).cost_usd
    with_ref = _price(kind="generate_2k", duration_s=6, input_video_seconds=4).cost_usd
    assert with_ref - without == Decimal("0.52")   # 4 × $0.13


def test_regeneration_bills_input_video_at_the_regeneration_rate():
    b = _price(kind="regenerate_2k", duration_s=6, input_video_seconds=4)
    assert b.input_video_usd == Decimal("0.20")    # 4 × $0.05


# ── Currency, safety factor, rounding ────────────────────────────────────────


def test_display_price_converts_at_the_given_rate():
    b = _price(kind="generate_2k", duration_s=10, usd_eur_rate=Decimal("1.00"))
    assert b.cost_eur == Decimal("1.30")


def test_reserved_amount_carries_the_safety_factor():
    b = _price(kind="generate_2k", duration_s=10, usd_eur_rate=Decimal("1.00"),
               safety_factor=Decimal("1.10"))
    assert b.cost_eur == Decimal("1.30")
    assert b.reserve_eur == Decimal("1.43")


def test_reservation_always_rounds_up():
    """A reservation a cent short of the real charge is the one way this guard
    fails open, so the padding is rounded towards more, never less."""
    b = _price(kind="generate_768p", duration_s=4, usd_eur_rate=Decimal("1.07"),
               safety_factor=Decimal("1.10"))
    exact = Decimal("0.32") / Decimal("1.07") * Decimal("1.10")
    assert b.reserve_eur >= exact
    assert b.reserve_eur - exact < Decimal("0.01")


def test_safety_factor_of_one_reserves_the_plain_price():
    b = _price(kind="generate_2k", duration_s=8, usd_eur_rate=Decimal("1.00"),
               safety_factor=1)
    assert b.reserve_eur == b.cost_eur


def test_zero_or_negative_exchange_rate_is_rejected():
    with pytest.raises(ValueError):
        _price(kind="generate_2k", duration_s=5, usd_eur_rate=Decimal("0"))


def test_settled_price_drops_the_safety_factor():
    usd, eur = pricing.settled_eur(
        kind="generate_2k", duration_s=10, image_count=0,
        input_video_seconds=0, usd_eur_rate=Decimal("1.00"),
    )
    assert (usd, eur) == (Decimal("1.3000"), Decimal("1.30"))


def test_kind_follows_resolution():
    assert pricing.kind_for("768P") == "generate_768p"
    assert pricing.kind_for("2K") == "generate_2k"


# ── Documented API bounds ────────────────────────────────────────────────────


def test_duration_bounds_match_the_api():
    assert (pricing.DURATION_MIN, pricing.DURATION_MAX) == (4, 15)


def test_ratio_list_includes_adaptive_and_the_named_ratios():
    assert "adaptive" in pricing.RATIOS
    for ratio in ("21:9", "16:9", "4:3", "1:1", "3:4", "9:16"):
        assert ratio in pricing.RATIOS
