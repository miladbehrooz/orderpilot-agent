"""Policy rules at their exact boundaries, on hand-built orders (pure functions, no IO)."""

from datetime import UTC, datetime, timedelta

import pytest
from pytest_mock import MockerFixture

from orderpilot import policy
from orderpilot.domain import Order, OrderStatus
from orderpilot.settings import settings


def _order(
    status: OrderStatus,
    created_at: datetime,
    *,
    shipped_at: datetime | None = None,
    delivered_at: datetime | None = None,
    returned_at: datetime | None = None,
) -> Order:
    """Build an order without items.

    Args:
        status: Order status.
        created_at: Creation time.
        shipped_at: Ship time, if any.
        delivered_at: Delivery time, if any.
        returned_at: Return time, if any.

    Returns:
        The order.
    """
    return Order(
        order_id=1,
        status=status,
        created_at=created_at,
        shipped_at=shipped_at,
        delivered_at=delivered_at,
        returned_at=returned_at,
        items=(),
    )


def _days_ago(today: datetime, days: int) -> datetime:
    """Return a time ``days`` calendar days before ``today``, at 10:00.

    Args:
        today: Reference time (midnight).
        days: Days to go back.

    Returns:
        The earlier time.
    """
    return today - timedelta(days=days) + timedelta(hours=10)


@pytest.mark.parametrize(
    ("start", "now", "expected"),
    [
        pytest.param(
            datetime(2026, 11, 11, 23, 59, tzinfo=UTC),
            datetime(2026, 11, 12, 0, 1, tzinfo=UTC),
            1,
            id="across-midnight-is-one-day",
        ),
        pytest.param(
            datetime(2026, 11, 12, 0, 1, tzinfo=UTC),
            datetime(2026, 11, 12, 23, 59, tzinfo=UTC),
            0,
            id="same-day-is-zero",
        ),
        pytest.param(
            datetime(2026, 10, 13, 12, 0, tzinfo=UTC),
            datetime(2026, 11, 12, 0, 0, tzinfo=UTC),
            30,
            id="thirty-calendar-days",
        ),
    ],
)
def test_days_between_counts_calendar_days(start: datetime, now: datetime, expected: int) -> None:
    """Days are calendar days, not elapsed 24-hour periods.

    Args:
        start: Earlier time.
        now: Reference time.
        expected: Expected day count.
    """
    assert policy.days_between(start, now) == expected


@pytest.mark.parametrize(
    ("status", "dates", "expected_days"),
    [
        pytest.param(OrderStatus.PROCESSING, {}, 10, id="processing-counts-from-created"),
        pytest.param(OrderStatus.CANCELLED, {}, 10, id="cancelled-counts-from-created"),
        pytest.param(OrderStatus.SHIPPED, {"shipped_at": 8}, 8, id="shipped-from-shipped"),
        pytest.param(
            OrderStatus.COMPLETE,
            {"shipped_at": 8, "delivered_at": 5},
            5,
            id="complete-from-delivered",
        ),
        pytest.param(
            OrderStatus.RETURNED,
            {"shipped_at": 8, "delivered_at": 5, "returned_at": 2},
            2,
            id="returned-from-returned",
        ),
        pytest.param(OrderStatus.SHIPPED, {}, 10, id="missing-marker-falls-back-to-created"),
    ],
)
def test_days_in_status(
    today: datetime, status: OrderStatus, dates: dict[str, int], expected_days: int
) -> None:
    """Days in status count from the timestamp that marks the current status.

    Args:
        today: Reference time.
        status: Order status.
        dates: Days ago of each optional timestamp.
        expected_days: Expected days in status.
    """
    order = _order(
        status,
        _days_ago(today, 10),
        **{name: _days_ago(today, days) for name, days in dates.items()},
    )
    assert policy.days_in_status(order, today) == expected_days


@pytest.mark.parametrize(
    ("days", "expected"),
    [
        pytest.param(settings.STUCK_DAYS, False, id="at-limit-not-stuck"),
        pytest.param(settings.STUCK_DAYS + 1, True, id="over-limit-stuck"),
    ],
)
def test_q1_stuck_boundary(today: datetime, days: int, expected: bool) -> None:
    """Q1: an order is stuck only after more than STUCK_DAYS days in Processing.

    Args:
        today: Reference time.
        days: Days in Processing.
        expected: Whether it counts as stuck.
    """
    order = _order(OrderStatus.PROCESSING, _days_ago(today, days))
    assert policy.is_stuck(order, today) is expected


@pytest.mark.parametrize(
    "status", [s for s in OrderStatus if s is not OrderStatus.PROCESSING], ids=str
)
def test_q1_only_processing_orders_can_be_stuck(today: datetime, status: OrderStatus) -> None:
    """Q1: an old order in any other status is never stuck.

    Args:
        today: Reference time.
        status: A non-Processing status.
    """
    old = _days_ago(today, 400)
    order = _order(status, old, shipped_at=old, delivered_at=old, returned_at=old)
    assert policy.is_stuck(order, today) is False


def test_q1_stuck_threshold_comes_from_settings(mocker: MockerFixture, today: datetime) -> None:
    """Q1: the stuck threshold is read from settings, not hard-coded.

    Args:
        mocker: Pytest-mock fixture.
        today: Reference time.
    """
    mocker.patch.object(settings, "STUCK_DAYS", 3)
    order = _order(OrderStatus.PROCESSING, _days_ago(today, 4))
    assert policy.is_stuck(order, today) is True


@pytest.mark.parametrize(
    ("days", "expected"),
    [
        pytest.param(settings.LOST_DAYS, False, id="at-limit-not-lost"),
        pytest.param(settings.LOST_DAYS + 1, True, id="over-limit-lost"),
    ],
)
def test_q2_lost_boundary(today: datetime, days: int, expected: bool) -> None:
    """Q2: a shipped order is possibly lost only after more than LOST_DAYS days.

    Args:
        today: Reference time.
        days: Days since shipping.
        expected: Whether it counts as possibly lost.
    """
    order = _order(
        OrderStatus.SHIPPED, _days_ago(today, days + 2), shipped_at=_days_ago(today, days)
    )
    assert policy.is_possibly_lost(order, today) is expected


def test_q2_delivered_order_is_never_lost(today: datetime) -> None:
    """Q2: an order that was delivered is not lost, however long ago it shipped.

    Args:
        today: Reference time.
    """
    old = _days_ago(today, 60)
    order = _order(OrderStatus.COMPLETE, old, shipped_at=old, delivered_at=old)
    assert policy.is_possibly_lost(order, today) is False
