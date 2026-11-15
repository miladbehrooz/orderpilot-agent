"""Policy rules at their exact boundaries, on hand-built orders (pure functions, no IO)."""

from datetime import UTC, datetime, timedelta

import pytest
from pytest_mock import MockerFixture

from orderpilot import policy
from orderpilot.domain import Order, OrderItem, OrderStatus
from orderpilot.services.results import ReasonCode
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


def _item(
    status: OrderStatus,
    delivered_at: datetime | None = None,
    *,
    return_requested: bool = False,
) -> OrderItem:
    """Build an order item priced 20.00.

    Args:
        status: Item status.
        delivered_at: Delivery time, if any.
        return_requested: Whether a return request exists.

    Returns:
        The item.
    """
    created = datetime(2025, 1, 1, tzinfo=UTC)
    return OrderItem(
        order_item_id=7,
        product_name="Shirt",
        brand="Acme",
        category="Tops & Tees",
        sale_price=20.0,
        status=status,
        created_at=created,
        shipped_at=None if status is OrderStatus.PROCESSING else created,
        delivered_at=delivered_at,
        returned_at=None,
        return_requested=return_requested,
    )


@pytest.mark.parametrize(
    ("status", "eligible", "code"),
    [
        pytest.param(OrderStatus.PROCESSING, True, ReasonCode.ELIGIBLE, id="p1-processing"),
        pytest.param(OrderStatus.SHIPPED, False, ReasonCode.ALREADY_SHIPPED, id="p2-shipped"),
        pytest.param(OrderStatus.COMPLETE, False, ReasonCode.ALREADY_SHIPPED, id="complete"),
        pytest.param(OrderStatus.RETURNED, False, ReasonCode.ALREADY_RETURNED, id="returned"),
        pytest.param(OrderStatus.CANCELLED, False, ReasonCode.ORDER_CANCELLED, id="cancelled"),
    ],
)
def test_p1_p2_cancel_decision_by_status(
    today: datetime, status: OrderStatus, eligible: bool, code: ReasonCode
) -> None:
    """P1, P2: only a Processing order can be cancelled; every status has its own code.

    Args:
        today: Reference time.
        status: Order status.
        eligible: Expected eligibility.
        code: Expected reason code.
    """
    decision = policy.cancel_decision(_order(status, _days_ago(today, 3)))
    assert (decision.eligible, decision.reason_code) == (eligible, code)


def test_p1_stuck_processing_order_can_still_be_cancelled(today: datetime) -> None:
    """P1: how long an order has been Processing doesn't matter for cancelling.

    Args:
        today: Reference time.
    """
    stuck = _order(OrderStatus.PROCESSING, _days_ago(today, 400))
    assert policy.is_stuck(stuck, today)
    assert policy.cancel_decision(stuck).eligible is True


@pytest.mark.parametrize(
    ("days", "code"),
    [
        pytest.param(settings.RETURN_WINDOW_DAYS, ReasonCode.ELIGIBLE, id="last-day-eligible"),
        pytest.param(settings.RETURN_WINDOW_DAYS + 1, ReasonCode.WINDOW_EXPIRED, id="day-after"),
    ],
)
def test_p3_p4_return_window_boundary(today: datetime, days: int, code: ReasonCode) -> None:
    """P3, P4: day 30 after delivery is still in the window, day 31 is not.

    Args:
        today: Reference time.
        days: Days since delivery.
        code: Expected reason code.
    """
    decision = policy.return_decision(_item(OrderStatus.COMPLETE, _days_ago(today, days)), today)
    assert decision.reason_code is code
    assert decision.days_since_delivery == days


def test_p3_eligible_return_has_refund_and_deadline(today: datetime) -> None:
    """P3: an eligible item reports its sale price as refund and the last return day.

    Args:
        today: Reference time.
    """
    delivered = _days_ago(today, 5)
    decision = policy.return_decision(_item(OrderStatus.COMPLETE, delivered), today)
    assert decision.eligible is True
    assert decision.refund_amount == 20.0
    assert decision.delivered_on == delivered.date()
    assert decision.return_deadline == delivered.date() + timedelta(days=30)


def test_p4_expired_return_keeps_delivery_facts(today: datetime) -> None:
    """P4: an expired item has no refund but still reports delivery date and deadline.

    Args:
        today: Reference time.
    """
    delivered = _days_ago(today, 45)
    decision = policy.return_decision(_item(OrderStatus.COMPLETE, delivered), today)
    assert (decision.eligible, decision.refund_amount) == (False, None)
    assert decision.days_since_delivery == 45
    assert decision.return_deadline == delivered.date() + timedelta(days=30)


@pytest.mark.parametrize(
    ("status", "code"),
    [
        pytest.param(OrderStatus.PROCESSING, ReasonCode.NOT_DELIVERED, id="p5-processing"),
        pytest.param(OrderStatus.SHIPPED, ReasonCode.NOT_DELIVERED, id="p5-shipped"),
        pytest.param(OrderStatus.CANCELLED, ReasonCode.ORDER_CANCELLED, id="p6-cancelled"),
    ],
)
def test_p5_p6_undelivered_items_cannot_be_returned(
    today: datetime, status: OrderStatus, code: ReasonCode
) -> None:
    """P5, P6: an item that was never delivered can't be returned.

    Args:
        today: Reference time.
        status: Item status.
        code: Expected reason code.
    """
    decision = policy.return_decision(_item(status), today)
    assert (decision.eligible, decision.reason_code) == (False, code)
    assert decision.delivered_on is None


def test_p5_complete_without_delivery_date_is_not_delivered(today: datetime) -> None:
    """P5: a Complete item with no delivery date is treated as not delivered.

    Args:
        today: Reference time.
    """
    decision = policy.return_decision(_item(OrderStatus.COMPLETE), today)
    assert decision.reason_code is ReasonCode.NOT_DELIVERED


@pytest.mark.parametrize(
    ("status", "return_requested"),
    [
        pytest.param(OrderStatus.RETURNED, False, id="p7-returned-status"),
        pytest.param(OrderStatus.COMPLETE, True, id="w5-return-request-row"),
    ],
)
def test_p7_already_returned(today: datetime, status: OrderStatus, return_requested: bool) -> None:
    """P7 (and W5 later): a Returned item, or one with a return request, is already returned.

    Args:
        today: Reference time.
        status: Item status.
        return_requested: Whether a return request exists.
    """
    item = _item(status, _days_ago(today, 3), return_requested=return_requested)
    decision = policy.return_decision(item, today)
    assert (decision.eligible, decision.reason_code) == (False, ReasonCode.ALREADY_RETURNED)


def test_return_window_comes_from_settings(mocker: MockerFixture, today: datetime) -> None:
    """The return window is read from settings, not hard-coded.

    Args:
        mocker: Pytest-mock fixture.
        today: Reference time.
    """
    mocker.patch.object(settings, "RETURN_WINDOW_DAYS", 10)
    decision = policy.return_decision(_item(OrderStatus.COMPLETE, _days_ago(today, 11)), today)
    assert decision.reason_code is ReasonCode.WINDOW_EXPIRED


@pytest.mark.parametrize(
    ("codes", "expected"),
    [
        pytest.param(
            [ReasonCode.WINDOW_EXPIRED, ReasonCode.ELIGIBLE], ReasonCode.ELIGIBLE, id="any"
        ),
        pytest.param(
            [ReasonCode.NOT_DELIVERED, ReasonCode.NOT_DELIVERED],
            ReasonCode.NOT_DELIVERED,
            id="shared",
        ),
        pytest.param(
            [ReasonCode.ALREADY_RETURNED, ReasonCode.WINDOW_EXPIRED],
            ReasonCode.ALREADY_RETURNED,
            id="mixed-first-wins",
        ),
    ],
)
def test_overall_return_code(codes: list[ReasonCode], expected: ReasonCode) -> None:
    """The combined code is ELIGIBLE if any item is, else the shared or first item's code.

    Args:
        codes: Item reason codes, in order.
        expected: Expected combined code.
    """
    decisions = [
        policy.ReturnDecision(order_item_id=i, eligible=c is ReasonCode.ELIGIBLE, reason_code=c)
        for i, c in enumerate(codes)
    ]
    assert policy.overall_return_code(decisions) is expected


def test_overall_return_code_needs_a_decision() -> None:
    """An empty list is a programming error, not a decision."""
    with pytest.raises(ValueError, match="at least one"):
        policy.overall_return_code([])
