"""Business rules as pure functions, shared by the read tools and (later) the write tools.

Every function takes domain objects and ``today`` (``meta.reference_now``, never the
clock) and returns a decision. Nothing here does IO, so the rules can be tested at their
exact boundaries, and a write tool can re-run the same check its read tool ran (rule 7).
Thresholds come from :mod:`orderpilot.settings`.

Days are counted in calendar days: something that happened at 23:59 is one day old at
00:01 the next day, the way a customer reads "30 days".
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from orderpilot.domain import Order, OrderItem, OrderStatus
from orderpilot.services.results import ReasonCode
from orderpilot.settings import settings


@dataclass(frozen=True, slots=True)
class CancelDecision:
    """Whether an order can be cancelled.

    Attributes:
        eligible: ``True`` if it can be cancelled.
        reason_code: ``ELIGIBLE`` or why it can't be cancelled.
    """

    eligible: bool
    reason_code: ReasonCode


@dataclass(frozen=True, slots=True)
class ReturnDecision:
    """Whether one item can be returned, with the numbers behind the decision.

    Attributes:
        order_item_id: The item.
        eligible: ``True`` if it can be returned.
        reason_code: ``ELIGIBLE`` or why it can't be returned.
        refund_amount: The refund (its sale price); only set when eligible.
        delivered_on: Delivery date, if delivered.
        days_since_delivery: Calendar days since delivery, if delivered.
        return_deadline: Last day a return is accepted, if delivered.
    """

    order_item_id: int
    eligible: bool
    reason_code: ReasonCode
    refund_amount: float | None = None
    delivered_on: date | None = None
    days_since_delivery: int | None = None
    return_deadline: date | None = None


def days_between(start: datetime, today: datetime) -> int:
    """Count calendar days from ``start`` to ``today``.

    Args:
        start: The earlier moment (UTC).
        today: The reference time (UTC).

    Returns:
        ``today``'s date minus ``start``'s date, in days.
    """
    return (today.date() - start.date()).days


def status_since(order: Order) -> datetime:
    """Return when the order entered its current status.

    Args:
        order: The order.

    Returns:
        The timestamp that marks the current status: created for Processing and
        Cancelled, shipped for Shipped, delivered for Complete, returned for Returned.
        Falls back to the creation time if that timestamp is missing.
    """
    marker = {
        OrderStatus.SHIPPED: order.shipped_at,
        OrderStatus.COMPLETE: order.delivered_at,
        OrderStatus.RETURNED: order.returned_at,
    }.get(order.status)
    return marker or order.created_at


def days_in_status(order: Order, today: datetime) -> int:
    """Count calendar days the order has been in its current status.

    Args:
        order: The order.
        today: The reference time (``meta.reference_now``).

    Returns:
        Days since :func:`status_since`.
    """
    return days_between(status_since(order), today)


def is_stuck(order: Order, today: datetime) -> bool:
    """Return whether the order has been Processing for too long (scenario Q1).

    Args:
        order: The order.
        today: The reference time (``meta.reference_now``).

    Returns:
        ``True`` if it is Processing for more than ``settings.STUCK_DAYS`` days.
    """
    return (
        order.status is OrderStatus.PROCESSING
        and days_in_status(order, today) > settings.STUCK_DAYS
    )


def is_possibly_lost(order: Order, today: datetime) -> bool:
    """Return whether a shipped order may be lost in transit (scenario Q2).

    Args:
        order: The order.
        today: The reference time (``meta.reference_now``).

    Returns:
        ``True`` if it is Shipped, not delivered, and was shipped more than
        ``settings.LOST_DAYS`` days ago.
    """
    return (
        order.status is OrderStatus.SHIPPED
        and order.delivered_at is None
        and days_in_status(order, today) > settings.LOST_DAYS
    )


def cancel_decision(order: Order) -> CancelDecision:
    """Decide whether an order can be cancelled: only while it is Processing (P1, P2).

    A stuck Processing order can still be cancelled; how long it has waited doesn't matter.

    Args:
        order: The order.

    Returns:
        ``ELIGIBLE`` for Processing; ``ALREADY_SHIPPED`` once it has shipped (Shipped or
        Complete); ``ALREADY_RETURNED`` or ``ORDER_CANCELLED`` otherwise.
    """
    match order.status:
        case OrderStatus.PROCESSING:
            return CancelDecision(eligible=True, reason_code=ReasonCode.ELIGIBLE)
        case OrderStatus.SHIPPED | OrderStatus.COMPLETE:
            return CancelDecision(eligible=False, reason_code=ReasonCode.ALREADY_SHIPPED)
        case OrderStatus.RETURNED:
            return CancelDecision(eligible=False, reason_code=ReasonCode.ALREADY_RETURNED)
        case OrderStatus.CANCELLED:
            return CancelDecision(eligible=False, reason_code=ReasonCode.ORDER_CANCELLED)


def return_decision(item: OrderItem, today: datetime) -> ReturnDecision:
    """Decide whether one item can be returned (P3-P8).

    An item can be returned if it is delivered (Complete) and at most
    ``settings.RETURN_WINDOW_DAYS`` calendar days have passed since delivery, so day 30
    is still in the window and day 31 is not. The refund is its sale price.

    Args:
        item: The item.
        today: The reference time (``meta.reference_now``).

    Returns:
        The decision. ``ALREADY_RETURNED`` if it was returned or has a return request,
        ``ORDER_CANCELLED`` if cancelled, ``NOT_DELIVERED`` if not delivered yet,
        ``WINDOW_EXPIRED`` if the window has passed, otherwise ``ELIGIBLE``. Delivery
        facts are included whenever the item was delivered.
    """
    window = settings.RETURN_WINDOW_DAYS
    delivered = item.delivered_at
    days = None if delivered is None else days_between(delivered, today)
    code = _return_code(item, days, window)
    eligible = code is ReasonCode.ELIGIBLE
    return ReturnDecision(
        order_item_id=item.order_item_id,
        eligible=eligible,
        reason_code=code,
        refund_amount=item.sale_price if eligible else None,
        delivered_on=None if delivered is None else delivered.date(),
        days_since_delivery=days,
        return_deadline=None if delivered is None else delivered.date() + timedelta(days=window),
    )


def _return_code(item: OrderItem, days_since_delivery: int | None, window: int) -> ReasonCode:
    """Return the reason code of a return decision; checks run in priority order.

    Args:
        item: The item.
        days_since_delivery: Calendar days since delivery, or ``None`` if not delivered.
        window: Return window in days.

    Returns:
        The first reason that applies, or ``ELIGIBLE``.
    """
    if item.return_requested or item.status is OrderStatus.RETURNED:
        return ReasonCode.ALREADY_RETURNED
    if item.status is OrderStatus.CANCELLED:
        return ReasonCode.ORDER_CANCELLED
    if item.status is not OrderStatus.COMPLETE or days_since_delivery is None:
        return ReasonCode.NOT_DELIVERED
    if days_since_delivery > window:
        return ReasonCode.WINDOW_EXPIRED
    return ReasonCode.ELIGIBLE


def overall_return_code(decisions: Sequence[ReturnDecision]) -> ReasonCode:
    """Combine item decisions into one code for the whole request.

    Args:
        decisions: One decision per checked item (at least one).

    Returns:
        ``ELIGIBLE`` if any item is eligible; otherwise the first item's code, which is
        also the shared code when all items fail for the same reason.

    Raises:
        ValueError: If ``decisions`` is empty.
    """
    if not decisions:
        raise ValueError("overall_return_code needs at least one decision")
    if any(d.eligible for d in decisions):
        return ReasonCode.ELIGIBLE
    return decisions[0].reason_code


__all__ = [
    "CancelDecision",
    "ReturnDecision",
    "cancel_decision",
    "days_between",
    "days_in_status",
    "is_possibly_lost",
    "is_stuck",
    "overall_return_code",
    "return_decision",
    "status_since",
]
