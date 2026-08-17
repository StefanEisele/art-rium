"""MiniMax H3 rate card and cost arithmetic.

MiniMax bills **per second of video**, which is the property this whole budget
feature rests on: the price of a call is known exactly *before* it is made, so
the guard can be a hard pre-check rather than a running estimate that discovers
an overrun after the money is gone.

Everything here is pure. No DB, no network — it is the part that decides what
something costs, and it is the part that must not be wrong.

────────────────────────────────────────────────────────────────────────────
RATE CARD — verified 2026-08-15 against
https://platform.minimax.io/docs/guides/pricing-paygo.md

Two of these lines are additionally confirmed by real charges on this account
(2026-08-16): a 4 s 768P generation settled at exactly $0.32, and its 2 K
regeneration at exactly $0.20. Both tiers are available on plain pay-as-you-go
— no whitelist, despite what several resellers write about 768P.

    768P output                     $0.08 / s
    2K   output                     $0.13 / s
    regeneration 768P → 2K          $0.05 / s
    input images                    first 5 free, then $0.04 each
    input images (on regeneration)  first 5 free, then $0.025 each
    input video                     billed per second at the output rate
    input video (on regeneration)   $0.05 / s
    input audio                     free

Re-check this before trusting a bill. Third-party resellers quote different
numbers and several of them are wrong about the 768P tier.
────────────────────────────────────────────────────────────────────────────

Two of these lines are **not** in the original implementation plan and were
found in the official rate card: input video is charged per second, and a
regeneration re-bills the original task's input material. A plan-faithful
formula would silently under-reserve every reference-to-video job — precisely
the case the budget is there to contain — so they are priced here.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_CEILING, ROUND_HALF_UP, Decimal

# ── The rate card itself ─────────────────────────────────────────────────────
RATE_CHECKED_ON = "2026-08-15"

RESOLUTIONS = ("768P", "2K")

OUTPUT_RATE_USD = {
    "768P": Decimal("0.08"),
    "2K":   Decimal("0.13"),
}
REGENERATION_RATE_USD = Decimal("0.05")

FREE_IMAGES = 5
IMAGE_RATE_USD = Decimal("0.04")
IMAGE_RATE_REGEN_USD = Decimal("0.025")

# Audio input is free, so it has no rate and never enters the arithmetic.

# ── Bounds the API itself enforces (docs/api-reference/video-generation-v2-create) ──
DURATION_MIN = 4
DURATION_MAX = 15
PROMPT_MAX_CHARS = 7000
MAX_REFERENCE_IMAGES = 9
MAX_REFERENCE_VIDEOS = 3
MAX_REFERENCE_AUDIO = 3
MAX_REQUEST_BYTES = 64 * 1024 * 1024
RATIOS = ("adaptive", "21:9", "16:9", "4:3", "1:1", "3:4", "9:16")

_CENT = Decimal("0.01")
_USD_PRECISION = Decimal("0.0001")


@dataclass(frozen=True, slots=True)
class CostBreakdown:
    """What a call costs, itemised.

    `cost_eur` is the honest price to show a human. `reserve_eur` is what the
    ledger blocks — the same number with the safety factor on top, rounded up.
    Keeping them apart is deliberate: padding the reservation protects the
    limit, padding the *displayed* price would just lie about the bill.
    """
    video_usd: Decimal
    image_usd: Decimal
    input_video_usd: Decimal
    cost_usd: Decimal
    cost_eur: Decimal
    reserve_eur: Decimal

    def as_dict(self) -> dict:
        return {
            "video_usd":       str(self.video_usd),
            "image_usd":       str(self.image_usd),
            "input_video_usd": str(self.input_video_usd),
            "cost_usd":        str(self.cost_usd),
            "cost_eur":        str(self.cost_eur),
            "reserve_eur":     str(self.reserve_eur),
        }


def _dec(value) -> Decimal:
    """Decimal from anything numeric, via str so floats do not leak binary dust."""
    return value if isinstance(value, Decimal) else Decimal(str(value))


def output_rate(resolution: str) -> Decimal:
    try:
        return OUTPUT_RATE_USD[resolution]
    except KeyError:
        raise ValueError(f"Unknown resolution {resolution!r}; expected one of {RESOLUTIONS}")


def billable_images(image_count: int) -> int:
    """Images beyond the free allowance. First-frame and last-frame count too."""
    return max(0, int(image_count) - FREE_IMAGES)


def generation_cost_usd(
    *,
    duration_s: int,
    resolution: str,
    image_count: int = 0,
    input_video_seconds: float | Decimal = 0,
) -> tuple[Decimal, Decimal, Decimal]:
    """(video, images, input video) in USD for one generation task."""
    rate = output_rate(resolution)
    video = rate * _dec(duration_s)
    images = IMAGE_RATE_USD * billable_images(image_count)
    input_video = rate * _dec(input_video_seconds)
    return video, images, input_video


def regeneration_cost_usd(
    *,
    duration_s: int,
    image_count: int = 0,
    input_video_seconds: float | Decimal = 0,
) -> tuple[Decimal, Decimal, Decimal]:
    """(video, images, input video) in USD for one 768P → 2K regeneration.

    The original task's input material is billed again, at reduced rates — the
    rate card is explicit about this and it is easy to miss.
    """
    video = REGENERATION_RATE_USD * _dec(duration_s)
    images = IMAGE_RATE_REGEN_USD * billable_images(image_count)
    input_video = REGENERATION_RATE_USD * _dec(input_video_seconds)
    return video, images, input_video


def price(
    *,
    kind: str,
    duration_s: int,
    resolution: str = "2K",
    image_count: int = 0,
    input_video_seconds: float | Decimal = 0,
    usd_eur_rate: float | Decimal,
    safety_factor: float | Decimal,
) -> CostBreakdown:
    """Full cost of one task.

    `kind` is the ledger kind: 'generate_768p', 'generate_2k' or
    'regenerate_2k'. Resolution is derived from it for generations so the
    booked kind and the charged rate can never disagree.
    """
    if kind == "regenerate_2k":
        video, images, input_video = regeneration_cost_usd(
            duration_s=duration_s,
            image_count=image_count,
            input_video_seconds=input_video_seconds,
        )
    elif kind in ("generate_768p", "generate_2k"):
        resolution = "768P" if kind == "generate_768p" else "2K"
        video, images, input_video = generation_cost_usd(
            duration_s=duration_s,
            resolution=resolution,
            image_count=image_count,
            input_video_seconds=input_video_seconds,
        )
    else:
        raise ValueError(f"Unknown ledger kind {kind!r}")

    usd = (video + images + input_video).quantize(_USD_PRECISION, ROUND_HALF_UP)

    rate = _dec(usd_eur_rate)
    if rate <= 0:
        raise ValueError("usd_eur_rate must be positive")
    eur_exact = usd / rate

    return CostBreakdown(
        video_usd=video,
        image_usd=images,
        input_video_usd=input_video,
        cost_usd=usd,
        cost_eur=eur_exact.quantize(_CENT, ROUND_HALF_UP),
        # Rounded *up*: a reservation that is a cent short of the real charge
        # is the one way this guard fails open.
        reserve_eur=(eur_exact * _dec(safety_factor)).quantize(_CENT, ROUND_CEILING),
    )


def kind_for(resolution: str) -> str:
    """Ledger kind for a generation at `resolution`."""
    return "generate_768p" if resolution == "768P" else "generate_2k"


def settled_eur(
    *,
    kind: str,
    duration_s: int,
    image_count: int,
    input_video_seconds: float | Decimal,
    usd_eur_rate: float | Decimal,
) -> tuple[Decimal, Decimal]:
    """(usd, eur) actually owed, with no safety factor.

    Used when a task finishes and MiniMax reports what it really billed, so the
    ledger settles on the true figure instead of keeping the padded estimate.
    """
    b = price(
        kind=kind,
        duration_s=duration_s,
        image_count=image_count,
        input_video_seconds=input_video_seconds,
        usd_eur_rate=usd_eur_rate,
        safety_factor=1,
    )
    return b.cost_usd, b.cost_eur
