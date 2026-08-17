"""The six failure cases the implementation plan calls out, against Postgres.

These need a real database because they are about things SQLite and mocks
cannot show: `SELECT … FOR UPDATE` serialising two concurrent reservations,
and rows surviving a simulated restart.

Everything runs in a throwaway schema (`artrium_test_budget`) that is created
and dropped per session, so the live tables are never touched. If no database
is reachable — CI, or a laptop without Postgres — the whole module skips.
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from core.config import settings
from core.db import Base
from core.models import API_WORKFLOW, BudgetPeriod, LedgerEntry, Video
from services.video_api import budget

SCHEMA = "artrium_test_budget"

# One event loop for the whole module: the engine and its asyncpg connections
# are created once at module scope, and a per-test loop would leave them bound
# to a loop that is already closed.
pytestmark = pytest.mark.asyncio(loop_scope="module")


def _engine():
    # The test schema *only*: with `public` on the path as well, create_all
    # finds the live tables, decides everything already exists, and the tests
    # would quietly run against real data.
    return create_async_engine(
        settings.database_url,
        connect_args={"server_settings": {"search_path": SCHEMA}},
    )


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def sessionmaker_():
    engine = _engine()
    try:
        async with engine.begin() as conn:
            await conn.execute(text(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE"))
            await conn.execute(text(f"CREATE SCHEMA {SCHEMA}"))
    except Exception as exc:                              # pragma: no cover
        await engine.dispose()
        pytest.skip(f"No Postgres for the ledger integration tests: {exc}")

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield async_sessionmaker(engine, expire_on_commit=False)

    async with engine.begin() as conn:
        await conn.execute(text(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE"))
    await engine.dispose()


@pytest_asyncio.fixture(loop_scope="module")
async def db(sessionmaker_):
    """A clean ledger for every test."""
    async with sessionmaker_() as session:
        await session.execute(text(f"TRUNCATE {SCHEMA}.ledger_entries, {SCHEMA}.budget_periods CASCADE"))
        await session.execute(text(f"TRUNCATE {SCHEMA}.videos CASCADE"))
        await session.commit()
        yield session


async def _period(db, limit: str = "20.00", rate: str = "1.00") -> BudgetPeriod:
    period = await budget.get_or_create_period(db)
    period.limit_eur = Decimal(limit)
    period.usd_eur_rate = Decimal(rate)
    await db.commit()
    return period


async def _reserve(db, *, duration_s=10, kind="generate_2k", key=None, **kw):
    return await budget.reserve(
        db, kind=kind, duration_s=duration_s,
        idempotency_key=key or f"test:{uuid.uuid4()}", **kw,
    )


# ── 1. Parallelism ───────────────────────────────────────────────────────────


async def test_two_concurrent_reservations_cannot_both_pass(sessionmaker_, db):
    """Each job fits on its own; together they break the limit. Exactly one
    must be admitted — this is the case a check-then-write without a lock gets
    wrong, and the reason MiniMax's 2-way concurrency matters.
    """
    await _period(db, limit="2.00", rate="1.00")     # 10 s @ 2K = $1.30 → 1.43 € reserved

    async def attempt():
        async with sessionmaker_() as session:
            try:
                await budget.reserve(
                    session, kind="generate_2k", duration_s=10,
                    idempotency_key=f"race:{uuid.uuid4()}",
                )
                return "ok"
            except budget.BudgetExceeded:
                return "refused"

    results = await asyncio.gather(attempt(), attempt())
    assert sorted(results) == ["ok", "refused"]

    summary = await budget.summary(db)
    assert Decimal(summary["committed_eur"]) <= Decimal(summary["limit_eur"])


async def test_many_concurrent_reservations_stop_at_the_limit(sessionmaker_, db):
    await _period(db, limit="5.00", rate="1.00")     # room for exactly 3 × 1.43 €

    async def attempt():
        async with sessionmaker_() as session:
            try:
                await budget.reserve(
                    session, kind="generate_2k", duration_s=10,
                    idempotency_key=f"burst:{uuid.uuid4()}",
                )
                return True
            except budget.BudgetExceeded:
                return False

    results = await asyncio.gather(*[attempt() for _ in range(8)])
    assert sum(results) == 3
    summary = await budget.summary(db)
    assert Decimal(summary["committed_eur"]) == Decimal("4.29")


# ── 2. Release ───────────────────────────────────────────────────────────────


async def test_a_failed_job_gives_its_reservation_back(db):
    await _period(db, limit="20.00", rate="1.00")
    before = (await budget.summary(db))["available_eur"]

    entry = await _reserve(db)
    assert (await budget.summary(db))["available_eur"] != before

    await budget.release(db, entry.id, note="provider said no")
    after = await budget.summary(db)
    assert after["available_eur"] == before
    assert Decimal(after["reserved_eur"]) == Decimal("0.00")


def _assert_two_places(value: str):
    assert Decimal(value) == Decimal(value).quantize(Decimal("0.01"))


async def test_settling_uses_what_was_really_billed(db):
    """The reservation carries a safety factor; the settled entry must not —
    otherwise the month slowly fills up with padding that was never charged."""
    await _period(db, limit="20.00", rate="1.00")
    entry = await _reserve(db, duration_s=10)
    assert entry.amount_eur == Decimal("1.43")       # 1.30 × 1.10

    await budget.settle(db, entry.id, billed_seconds=10)
    summary = await budget.summary(db)
    assert Decimal(summary["settled_eur"]) == Decimal("1.30")
    assert Decimal(summary["reserved_eur"]) == Decimal("0.00")
    _assert_two_places(summary["available_eur"])


async def test_settling_a_shorter_clip_refunds_the_difference(db):
    await _period(db, limit="20.00", rate="1.00")
    entry = await _reserve(db, duration_s=15)
    await budget.settle(db, entry.id, billed_seconds=6)
    assert Decimal((await budget.summary(db))["settled_eur"]) == Decimal("0.78")


async def test_release_is_idempotent(db):
    await _period(db)
    entry = await _reserve(db)
    await budget.release(db, entry.id, note="once")
    await budget.release(db, entry.id, note="twice")
    assert Decimal((await budget.summary(db))["committed_eur"]) == Decimal("0.00")


async def test_a_settled_entry_cannot_be_released(db):
    """Money already spent must not become available again."""
    await _period(db)
    entry = await _reserve(db)
    await budget.settle(db, entry.id, billed_seconds=10)
    await budget.release(db, entry.id, note="too late")
    refreshed = await db.get(LedgerEntry, entry.id)
    assert refreshed.state == "settled"


# ── 3. Restart ───────────────────────────────────────────────────────────────


async def test_a_reservation_that_was_never_submitted_is_released(db):
    """The restart bug: a crash between reserving and submitting leaves money
    blocked that no poller will ever pick up. After a few restarts the budget
    reads full without a cent spent."""
    await _period(db)
    entry = await _reserve(db)
    entry.created_at = datetime.now(timezone.utc) - timedelta(minutes=10)
    await db.commit()

    freed = await budget.release_stranded(db)
    assert freed == 1
    assert Decimal((await budget.summary(db))["committed_eur"]) == Decimal("0.00")


async def test_an_in_flight_reservation_survives_a_restart(db):
    """A submitted task keeps running while art-rium is down, so its money
    must stay blocked — releasing it would let the same budget be spent twice."""
    await _period(db)
    entry = await _reserve(db)
    await budget.mark_submitted(db, entry.id, "task-abc")
    entry.created_at = datetime.now(timezone.utc) - timedelta(hours=2)
    await db.commit()

    assert await budget.release_stranded(db) == 0
    assert Decimal((await budget.summary(db))["reserved_eur"]) == entry.amount_eur


async def test_reconcile_adopts_in_flight_jobs(db, sessionmaker_, monkeypatch):
    """Restart with a job in flight: it is picked back up for polling rather
    than left hanging or marked failed."""
    from services.video_api import queue as queue_module

    await _period(db)
    video = Video(
        workflow=API_WORKFLOW, status="generating", api_task_id="task-xyz",
        api_resolution="2K", duration_s=6, prompt="x",
    )
    db.add(video)
    await db.commit()
    entry = await _reserve(db, video_id=video.id)
    await budget.mark_submitted(db, entry.id, "task-xyz")

    monkeypatch.setattr(queue_module, "AsyncSessionLocal", sessionmaker_)
    report = await queue_module.CloudVideoQueue().reconcile()

    assert report["adopted"] == 1
    assert report["released"] == 0
    assert video.id in queue_module._poll_state


# ── 4. Month rollover ────────────────────────────────────────────────────────


async def test_a_new_month_starts_a_new_period(db):
    august = await budget.get_or_create_period(
        db, now=datetime(2026, 8, 15, tzinfo=timezone.utc)
    )
    september = await budget.get_or_create_period(
        db, now=datetime(2026, 9, 1, tzinfo=timezone.utc)
    )
    assert august.id != september.id
    assert (august.month, september.month) == ("2026-08", "2026-09")


async def test_last_months_spend_does_not_count_against_the_new_month(db):
    august = await budget.get_or_create_period(
        db, now=datetime(2026, 8, 15, tzinfo=timezone.utc)
    )
    august.limit_eur = Decimal("20.00")
    august.usd_eur_rate = Decimal("1.00")
    await db.commit()
    await budget.reserve(
        db, kind="generate_2k", duration_s=15, idempotency_key="aug:1",
        now=datetime(2026, 8, 15, tzinfo=timezone.utc),
    )

    september = await budget.summary(db, now=datetime(2026, 9, 2, tzinfo=timezone.utc))
    assert september["month"] == "2026-09"
    assert Decimal(september["committed_eur"]) == Decimal("0.00")
    assert Decimal(september["available_eur"]) == Decimal(september["limit_eur"])


async def test_a_new_period_carries_the_previous_settings_forward(db):
    august = await budget.get_or_create_period(
        db, now=datetime(2026, 8, 15, tzinfo=timezone.utc)
    )
    august.limit_eur = Decimal("42.00")
    august.warn_threshold_pct = 65
    august.usd_eur_rate = Decimal("1.1234")
    await db.commit()

    september = await budget.get_or_create_period(
        db, now=datetime(2026, 9, 1, tzinfo=timezone.utc)
    )
    assert september.limit_eur == Decimal("42.00")
    assert september.warn_threshold_pct == 65
    assert september.usd_eur_rate == Decimal("1.1234")


# ── 5. The exact boundary ────────────────────────────────────────────────────


async def test_a_job_costing_exactly_the_remainder_is_accepted(db):
    # 4 s @ 768P = $0.32; at rate 1.00 with factor 1.10 → 0.36 € reserved.
    await _period(db, limit="0.36", rate="1.00")
    entry = await _reserve(db, duration_s=4, kind="generate_768p")
    assert entry.amount_eur == Decimal("0.36")
    assert Decimal((await budget.summary(db))["available_eur"]) == Decimal("0.00")


async def test_one_cent_short_is_refused(db):
    await _period(db, limit="0.35", rate="1.00")
    with pytest.raises(budget.BudgetExceeded) as excinfo:
        await _reserve(db, duration_s=4, kind="generate_768p")
    assert excinfo.value.shortfall_eur == Decimal("0.01")


async def test_the_refusal_carries_the_numbers_the_ui_needs(db):
    await _period(db, limit="0.10", rate="1.00")
    with pytest.raises(budget.BudgetExceeded) as excinfo:
        await _reserve(db, duration_s=10)
    detail = excinfo.value.as_detail()
    assert detail["reason"] == "WOULD_EXCEED"
    assert Decimal(detail["cost_eur"]) > Decimal(detail["available_eur"])
    assert Decimal(detail["shortfall_eur"]) > 0


# ── 6. Lowering the limit ────────────────────────────────────────────────────


async def test_lowering_the_limit_below_committed_is_allowed(db):
    """It blocks new jobs without cancelling paid-for work — and must not throw
    or produce a negative balance."""
    await _period(db, limit="20.00", rate="1.00")
    await _reserve(db, duration_s=10)                       # 1.43 € committed

    await budget.update_period(db, limit_eur=Decimal("0.50"))
    summary = await budget.summary(db)

    assert Decimal(summary["available_eur"]) == Decimal("0.00")
    assert Decimal(summary["committed_eur"]) == Decimal("1.43")
    assert summary["exhausted"] is True
    assert summary["used_pct"] == 100


async def test_no_new_job_is_admitted_after_the_limit_is_lowered(db):
    await _period(db, limit="20.00", rate="1.00")
    await _reserve(db, duration_s=10)
    await budget.update_period(db, limit_eur=Decimal("0.50"))
    with pytest.raises(budget.BudgetExceeded):
        await _reserve(db, duration_s=4, kind="generate_768p")


async def test_running_work_keeps_its_reservation_when_the_limit_drops(db):
    await _period(db, limit="20.00", rate="1.00")
    entry = await _reserve(db, duration_s=10)
    await budget.update_period(db, limit_eur=Decimal("0.50"))
    assert (await db.get(LedgerEntry, entry.id)).state == "reserved"


# ── Idempotency ──────────────────────────────────────────────────────────────


async def test_the_same_idempotency_key_books_once(db):
    """A retried submit must not book twice — the plan's rule, and the reason
    the column is unique."""
    await _period(db, limit="20.00", rate="1.00")
    first = await _reserve(db, key="stable-key")
    second = await _reserve(db, key="stable-key")
    assert first.id == second.id
    assert Decimal((await budget.summary(db))["committed_eur"]) == first.amount_eur


# ── Warning threshold ────────────────────────────────────────────────────────


async def test_the_warning_flips_at_the_configured_threshold(db):
    await _period(db, limit="10.00", rate="1.00")
    await budget.update_period(db, warn_threshold_pct=80)

    await _reserve(db, duration_s=5)                 # 0.72 € → 7 %
    assert (await budget.summary(db))["warning"] is False

    await _reserve(db, duration_s=15)                # +2.15 €
    await _reserve(db, duration_s=15)
    await _reserve(db, duration_s=15)
    await _reserve(db, duration_s=10)                # ~8.30 € total → 83 %
    summary = await budget.summary(db)
    assert summary["used_pct"] >= 80
    assert summary["warning"] is True
