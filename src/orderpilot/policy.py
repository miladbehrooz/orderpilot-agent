"""Business rules as pure functions, shared by the read tools and (later) the write tools.

Every function takes domain objects and ``today`` (``meta.reference_now``, never the
clock) and returns a decision. Nothing here does IO, so the rules can be tested at their
exact boundaries, and a write tool can re-run the same check its read tool ran (rule 7).
Thresholds come from :mod:`orderpilot.settings`.

Days are counted in calendar days: something that happened at 23:59 is one day old at
00:01 the next day, the way a customer reads "30 days".
"""

from datetime import datetime

from orderpilot.domain import Order, OrderStatus
from orderpilot.settings import settings


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


__all__ = ["days_between", "days_in_status", "is_possibly_lost", "is_stuck", "status_since"]
