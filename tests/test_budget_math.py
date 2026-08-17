"""Ledger arithmetic, without a database.

The DB-backed behaviour (locking, reconciliation, month rollover) is covered in
test_budget_ledger_db.py, which needs Postgres. What lives here is the part
that must be right regardless of where the rows come from.
"""
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace

from services.video_api.budget import (
    STRANDED_AFTER,
    available_eur,
    committed_eur,
    current_month,
    fits,
    is_stranded,
)


def _entry(amount: str, state: str = "reserved", *, task_id=None, age_minutes=0):
    return SimpleNamespace(
        amount_eur=Decimal(amount),
        state=state,
        task_id=task_id,
        created_at=datetime.now(timezone.utc) - timedelta(minutes=age_minutes),
    )


# ── What counts against the limit ────────────────────────────────────────────


def test_reserved_and_settled_both_count():
    entries = [_entry("1.50", "reserved"), _entry("2.25", "settled")]
    assert committed_eur(entries) == Decimal("3.75")


def test_released_never_counts():
    """A refunded reservation must free its money completely, or a few failed
    jobs would quietly eat the month."""
    entries = [_entry("1.50", "settled"), _entry("9.99", "released")]
    assert committed_eur(entries) == Decimal("1.50")


def test_empty_period_has_nothing_committed():
    assert committed_eur([]) == Decimal("0.00")


def test_available_is_limit_minus_committed():
    assert available_eur(Decimal("20.00"), Decimal("12.40")) == Decimal("7.60")


def test_available_never_goes_negative():
    """Lowering the limit under what is already committed is allowed; it has to
    read as "nothing free", not as a negative balance the UI would render as a
    bar pointing the wrong way."""
    assert available_eur(Decimal("5.00"), Decimal("12.40")) == Decimal("0.00")


# ── The boundary ─────────────────────────────────────────────────────────────


def test_a_job_costing_exactly_the_remainder_is_admitted():
    assert fits(Decimal("1.32"), Decimal("1.32"))


def test_one_cent_more_than_the_remainder_is_refused():
    assert not fits(Decimal("1.32"), Decimal("1.33"))


def test_nothing_fits_into_nothing():
    assert not fits(Decimal("0.00"), Decimal("0.01"))
    assert fits(Decimal("0.00"), Decimal("0.00"))


# ── Stranded reservations ────────────────────────────────────────────────────


def test_a_fresh_reservation_without_a_task_id_is_not_stranded_yet():
    """The submit may still be in flight — releasing here would let the same
    budget be spent twice."""
    assert not is_stranded(_entry("1.00", "reserved", age_minutes=1))


def test_an_old_reservation_without_a_task_id_is_stranded():
    age = STRANDED_AFTER.total_seconds() / 60 + 1
    assert is_stranded(_entry("1.00", "reserved", age_minutes=age))


def test_a_submitted_reservation_is_never_stranded():
    age = STRANDED_AFTER.total_seconds() / 60 + 60
    assert not is_stranded(_entry("1.00", "reserved", task_id="abc", age_minutes=age))


def test_settled_and_released_entries_are_never_stranded():
    for state in ("settled", "released"):
        assert not is_stranded(_entry("1.00", state, age_minutes=999))


def test_naive_timestamps_are_treated_as_utc():
    """Postgres hands back tz-aware values, but a row built in a test or by an
    older migration may not be — that must not raise."""
    entry = SimpleNamespace(
        amount_eur=Decimal("1.00"), state="reserved", task_id=None,
        created_at=datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=1),
    )
    assert is_stranded(entry)


# ── Month keys ───────────────────────────────────────────────────────────────


def test_month_key_format():
    assert current_month(datetime(2026, 8, 15, tzinfo=timezone.utc)) == "2026-08"


def test_month_key_rolls_over_at_the_first():
    last = current_month(datetime(2026, 8, 31, 23, 59, tzinfo=timezone.utc))
    first = current_month(datetime(2026, 9, 1, 0, 0, tzinfo=timezone.utc))
    assert last == "2026-08" and first == "2026-09"
