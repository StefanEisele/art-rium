"""The budget ledger — two-phase booking against a monthly limit.

The naive version of this feature adds up what jobs cost after they finish and
checks the total before starting the next one. That is correct exactly until
two jobs run at once: both pass the check, both run, and the limit is gone.
MiniMax allows two concurrent tasks, so the naive version is wrong on day one.

So money moves in two phases:

    RESERVE   price the call → block it in the ledger → only then submit
    SETTLE    task succeeded → 'settled', at what was really billed
    RELEASE   task failed / cancelled / never submitted → 'reserved' frees up

The reserve step holds a row lock on the period (`SELECT … FOR UPDATE`) while
it checks availability and writes the entry, so two requests arriving together
are serialised by the database rather than racing each other. That lock is the
whole guarantee — without it this module is decoration.

`released` never counts. Rows are never deleted: the ledger is the audit trail
and "why is my budget gone?" has to be answerable.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Iterable

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import settings
from core.models import COMMITTED_STATES, BudgetPeriod, LedgerEntry
from services.video_api import pricing

logger = logging.getLogger(__name__)

ZERO = Decimal("0.00")

# A reservation whose submit never landed is stranded: no task id will ever
# arrive for it, and nothing else would ever free it.
STRANDED_AFTER = timedelta(minutes=5)


class BudgetExceeded(Exception):
    """Raised instead of submitting when a call would break the limit."""

    def __init__(self, *, cost_eur: Decimal, available_eur: Decimal):
        self.cost_eur = cost_eur
        self.available_eur = available_eur
        self.shortfall_eur = (cost_eur - available_eur).quantize(Decimal("0.01"))
        super().__init__(
            f"Would exceed the monthly limit: {cost_eur} € needed, "
            f"{available_eur} € available"
        )

    def as_detail(self) -> dict:
        return {
            "reason": "WOULD_EXCEED",
            "cost_eur": str(self.cost_eur),
            "available_eur": str(self.available_eur),
            "shortfall_eur": str(self.shortfall_eur),
        }


# ── Pure helpers ─────────────────────────────────────────────────────────────
# Kept free of the session so the arithmetic can be tested without a database.


def current_month(now: datetime | None = None) -> str:
    return (now or datetime.now(timezone.utc)).strftime("%Y-%m")


def committed_eur(entries: Iterable[LedgerEntry]) -> Decimal:
    """What this period has spoken for: reserved + settled, never released."""
    return sum(
        (e.amount_eur for e in entries if e.state in COMMITTED_STATES), ZERO,
    ).quantize(Decimal("0.01"))


def available_eur(limit: Decimal, committed: Decimal) -> Decimal:
    """Never negative: lowering the limit below what is already committed is
    allowed and must read as "nothing free", not as a negative balance."""
    return max(ZERO, (limit - committed).quantize(Decimal("0.01")))


def fits(available: Decimal, cost: Decimal) -> bool:
    """A job costing exactly what is left is allowed; one cent more is not."""
    return cost <= available


def is_stranded(entry: LedgerEntry, now: datetime | None = None) -> bool:
    """A reservation that was never submitted and is old enough to give up on."""
    if entry.state != "reserved" or entry.task_id:
        return False
    created = entry.created_at
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    return (now or datetime.now(timezone.utc)) - created > STRANDED_AFTER


# ── Period ───────────────────────────────────────────────────────────────────


async def get_or_create_period(
    db: AsyncSession, *, now: datetime | None = None,
) -> BudgetPeriod:
    """This month's period, carrying the previous month's settings forward.

    Carrying forward rather than resetting to the config defaults is the point
    of a monthly period: the limit is a standing decision, the spend is what
    starts over.
    """
    month = current_month(now)
    found = (await db.execute(
        select(BudgetPeriod).where(BudgetPeriod.month == month)
    )).scalar_one_or_none()
    if found:
        return found

    previous = (await db.execute(
        select(BudgetPeriod).order_by(BudgetPeriod.month.desc()).limit(1)
    )).scalar_one_or_none()

    period = BudgetPeriod(
        month=month,
        limit_eur=previous.limit_eur if previous else Decimal(str(settings.video_api_default_limit_eur)),
        warn_threshold_pct=previous.warn_threshold_pct if previous else settings.video_api_warn_threshold_pct,
        usd_eur_rate=previous.usd_eur_rate if previous else Decimal(str(settings.video_api_usd_eur_rate)),
    )
    db.add(period)
    try:
        await db.commit()
    except IntegrityError:
        # Two requests created the month at the same time; the other one won.
        await db.rollback()
        period = (await db.execute(
            select(BudgetPeriod).where(BudgetPeriod.month == month)
        )).scalar_one()
    await db.refresh(period)
    logger.info("Budget period %s ready (limit %s €)", period.month, period.limit_eur)
    return period


async def _entries_of(db: AsyncSession, period_id: uuid.UUID) -> list[LedgerEntry]:
    return list((await db.execute(
        select(LedgerEntry).where(LedgerEntry.period_id == period_id)
    )).scalars().all())


async def summary(db: AsyncSession, *, now: datetime | None = None) -> dict:
    """Everything the budget bar needs, in one read."""
    period = await get_or_create_period(db, now=now)
    entries = await _entries_of(db, period.id)

    settled = sum((e.amount_eur for e in entries if e.state == "settled"), ZERO)
    reserved = sum((e.amount_eur for e in entries if e.state == "reserved"), ZERO)
    committed = (settled + reserved).quantize(Decimal("0.01"))
    limit = Decimal(period.limit_eur)
    available = available_eur(limit, committed)
    pct = int(committed / limit * 100) if limit > 0 else 100

    return {
        "month": period.month,
        "limit_eur": str(limit),
        "settled_eur": str(settled.quantize(Decimal("0.01"))),
        "reserved_eur": str(reserved.quantize(Decimal("0.01"))),
        "committed_eur": str(committed),
        "available_eur": str(available),
        "used_pct": min(100, pct),
        "warn_threshold_pct": period.warn_threshold_pct,
        "warning": pct >= period.warn_threshold_pct,
        "exhausted": available <= ZERO,
        "usd_eur_rate": str(period.usd_eur_rate),
        "rate_updated_at": period.rate_updated_at.isoformat(),
        "rate_checked_on": pricing.RATE_CHECKED_ON,
        "safety_factor": str(settings.video_api_safety_factor),
        "entry_count": len(entries),
    }


# ── Estimate ─────────────────────────────────────────────────────────────────


async def estimate(
    db: AsyncSession,
    *,
    kind: str,
    duration_s: int,
    image_count: int = 0,
    input_video_seconds: float | Decimal = 0,
    now: datetime | None = None,
) -> dict:
    """Price a call and say whether it would fit — without booking anything.

    Backs the live cost preview, so it must use the *same* pricing path the
    reservation uses; a preview computed differently from the booking is worse
    than no preview.
    """
    period = await get_or_create_period(db, now=now)
    breakdown = pricing.price(
        kind=kind,
        duration_s=duration_s,
        image_count=image_count,
        input_video_seconds=input_video_seconds,
        usd_eur_rate=period.usd_eur_rate,
        safety_factor=settings.video_api_safety_factor,
    )
    entries = await _entries_of(db, period.id)
    available = available_eur(Decimal(period.limit_eur), committed_eur(entries))
    ok = fits(available, breakdown.reserve_eur)

    return {
        **breakdown.as_dict(),
        "kind": kind,
        "duration_s": duration_s,
        "ok": ok,
        "available_eur": str(available),
        "available_after_eur": str(
            available - breakdown.reserve_eur if ok else available
        ),
        "shortfall_eur": str(
            ZERO if ok else (breakdown.reserve_eur - available).quantize(Decimal("0.01"))
        ),
    }


# ── Two-phase booking ────────────────────────────────────────────────────────


async def reserve(
    db: AsyncSession,
    *,
    kind: str,
    duration_s: int,
    idempotency_key: str,
    video_id: uuid.UUID | None = None,
    image_count: int = 0,
    input_video_seconds: float | Decimal = 0,
    now: datetime | None = None,
) -> LedgerEntry:
    """Block the cost of one call, or raise BudgetExceeded.

    Availability check and insert happen under a row lock on the period, so
    two callers that would each fit individually cannot both be admitted.
    Commits on success — the reservation must be durable before anything is
    submitted to the provider.
    """
    period = await get_or_create_period(db, now=now)

    # Re-read the period FOR UPDATE. Everything between here and the commit is
    # serialised against any other reservation in the same month.
    locked = (await db.execute(
        select(BudgetPeriod).where(BudgetPeriod.id == period.id).with_for_update()
    )).scalar_one()

    existing = (await db.execute(
        select(LedgerEntry).where(LedgerEntry.idempotency_key == idempotency_key)
    )).scalar_one_or_none()
    if existing:
        # A retried submit. Hand back the original booking rather than a second.
        await db.commit()
        return existing

    breakdown = pricing.price(
        kind=kind,
        duration_s=duration_s,
        image_count=image_count,
        input_video_seconds=input_video_seconds,
        usd_eur_rate=locked.usd_eur_rate,
        safety_factor=settings.video_api_safety_factor,
    )
    entries = await _entries_of(db, locked.id)
    available = available_eur(Decimal(locked.limit_eur), committed_eur(entries))
    if not fits(available, breakdown.reserve_eur):
        await db.rollback()
        raise BudgetExceeded(cost_eur=breakdown.reserve_eur, available_eur=available)

    entry = LedgerEntry(
        period_id=locked.id,
        video_id=video_id,
        idempotency_key=idempotency_key,
        kind=kind,
        duration_s=duration_s,
        ref_image_count=image_count,
        input_video_seconds=Decimal(str(input_video_seconds)),
        amount_usd=breakdown.cost_usd,
        amount_eur=breakdown.reserve_eur,
        usd_eur_rate=locked.usd_eur_rate,
        state="reserved",
    )
    db.add(entry)
    await db.commit()
    await db.refresh(entry)
    logger.info(
        "Reserved %s € for %s (%ds, %s) — %s € was free",
        entry.amount_eur, kind, duration_s, entry.id, available,
    )
    return entry


async def mark_submitted(db: AsyncSession, entry_id: uuid.UUID, task_id: str) -> None:
    """Record the provider's task id, which is what makes the entry traceable."""
    entry = await db.get(LedgerEntry, entry_id)
    if entry:
        entry.task_id = task_id
        await db.commit()


async def settle(
    db: AsyncSession,
    entry_id: uuid.UUID,
    *,
    billed_seconds: float | None = None,
    billed_image_count: int | None = None,
    billed_input_video_seconds: float | None = None,
) -> LedgerEntry | None:
    """Turn a reservation into a real charge.

    When the provider reports what it actually billed (`task.usage`), the entry
    is re-priced from those numbers *without* the safety factor, so the ledger
    ends up holding the true cost rather than the padded estimate. Without a
    usage report the reservation stands as booked.
    """
    entry = await db.get(LedgerEntry, entry_id)
    if not entry or entry.state != "reserved":
        return entry

    if billed_seconds:
        usd, eur = pricing.settled_eur(
            kind=entry.kind,
            duration_s=billed_seconds,
            image_count=(
                entry.ref_image_count if billed_image_count is None else billed_image_count
            ),
            input_video_seconds=(
                entry.input_video_seconds
                if billed_input_video_seconds is None
                else billed_input_video_seconds
            ),
            usd_eur_rate=entry.usd_eur_rate,
        )
        entry.amount_usd = usd
        entry.amount_eur = eur
        entry.duration_s = int(round(billed_seconds))

    entry.state = "settled"
    entry.settled_at = datetime.now(timezone.utc)
    await db.commit()
    await db.refresh(entry)
    logger.info("Settled %s € for entry %s", entry.amount_eur, entry.id)
    return entry


async def release(
    db: AsyncSession, entry_id: uuid.UUID, *, note: str,
) -> LedgerEntry | None:
    """Give a reservation back. Idempotent — releasing twice is a no-op."""
    entry = await db.get(LedgerEntry, entry_id)
    if not entry or entry.state != "reserved":
        return entry
    entry.state = "released"
    entry.note = note[:500]
    entry.settled_at = datetime.now(timezone.utc)
    await db.commit()
    await db.refresh(entry)
    logger.info("Released %s € for entry %s (%s)", entry.amount_eur, entry.id, note)
    return entry


async def release_stranded(db: AsyncSession, *, now: datetime | None = None) -> int:
    """Free reservations whose submit never went through.

    Without this, a crash between reserving and submitting eats budget forever:
    no task id means no poller will ever pick the entry up and finish it.
    """
    reserved = (await db.execute(
        select(LedgerEntry).where(
            LedgerEntry.state == "reserved", LedgerEntry.task_id.is_(None)
        )
    )).scalars().all()
    freed = [e for e in reserved if is_stranded(e, now)]
    for entry in freed:
        entry.state = "released"
        entry.note = "Submit never completed"
        entry.settled_at = datetime.now(timezone.utc)
    if freed:
        await db.commit()
        logger.warning("Released %d stranded reservation(s)", len(freed))
    return len(freed)


# ── Settings ─────────────────────────────────────────────────────────────────


async def update_period(
    db: AsyncSession,
    *,
    limit_eur: Decimal | None = None,
    warn_threshold_pct: int | None = None,
    usd_eur_rate: Decimal | None = None,
    now: datetime | None = None,
) -> BudgetPeriod:
    """Edit this month's settings.

    Lowering the limit below what is already committed is allowed on purpose:
    it blocks new jobs without cancelling running ones. Cancelling work the
    user already paid for would be the more surprising behaviour.
    """
    period = await get_or_create_period(db, now=now)
    if limit_eur is not None:
        period.limit_eur = limit_eur
    if warn_threshold_pct is not None:
        period.warn_threshold_pct = warn_threshold_pct
    if usd_eur_rate is not None and Decimal(usd_eur_rate) != Decimal(period.usd_eur_rate):
        period.usd_eur_rate = usd_eur_rate
        period.rate_updated_at = datetime.now(timezone.utc)
    await db.commit()
    await db.refresh(period)
    return period


async def ledger(db: AsyncSession, *, limit: int = 100, month: str | None = None) -> list[dict]:
    """The ledger view: what was booked, for what, and how it ended."""
    stmt = select(LedgerEntry).order_by(LedgerEntry.created_at.desc()).limit(limit)
    if month:
        period = (await db.execute(
            select(BudgetPeriod).where(BudgetPeriod.month == month)
        )).scalar_one_or_none()
        if not period:
            return []
        stmt = stmt.where(LedgerEntry.period_id == period.id)
    entries = (await db.execute(stmt)).scalars().all()
    return [serialize_entry(e) for e in entries]


def serialize_entry(e: LedgerEntry) -> dict:
    return {
        "id": str(e.id),
        "video_id": str(e.video_id) if e.video_id else None,
        "task_id": e.task_id,
        "kind": e.kind,
        "duration_s": e.duration_s,
        "ref_image_count": e.ref_image_count,
        "input_video_seconds": str(e.input_video_seconds),
        "amount_usd": str(e.amount_usd),
        "amount_eur": str(e.amount_eur),
        "state": e.state,
        "note": e.note,
        "created_at": e.created_at.isoformat(),
        "settled_at": e.settled_at.isoformat() if e.settled_at else None,
    }
