"""Order lookups (scenarios L1-L6, Q1-Q3, AM1, AM2, S1).

Framework-free: takes a repository, the verified ``user_id`` and plain values, and
returns a result model. The LangChain tools in :mod:`orderpilot.tools.orders` are thin
wrappers around these functions.
"""

from datetime import date, datetime

from orderpilot import policy
from orderpilot.domain import Order, OrderItem, OrderStatus
from orderpilot.repository import OrderRepository
from orderpilot.services.results import (
    ItemView,
    OrderDetails,
    OrderDetailsResult,
    OrderListResult,
    OrderSummary,
    ReasonCode,
)


def _day(value: datetime | None) -> date | None:
    """Return the calendar date of a timestamp.

    Args:
        value: A UTC timestamp, or ``None``.

    Returns:
        Its date, or ``None``.
    """
    return None if value is None else value.date()


def _summarize(order: Order, today: datetime) -> OrderSummary:
    """Summarize an order, with the flags and day counts from the policy rules.

    Args:
        order: The order with its items.
        today: The reference time (``meta.reference_now``).

    Returns:
        The order summary.
    """
    returned = [i.sale_price for i in order.items if i.status is OrderStatus.RETURNED]
    return OrderSummary(
        order_id=order.order_id,
        status=order.status,
        created_on=order.created_at.date(),
        shipped_on=_day(order.shipped_at),
        delivered_on=_day(order.delivered_at),
        returned_on=_day(order.returned_at),
        item_count=len(order.items),
        total=round(sum(i.sale_price for i in order.items), 2),
        days_in_status=policy.days_in_status(order, today),
        is_stuck=policy.is_stuck(order, today),
        is_possibly_lost=policy.is_possibly_lost(order, today),
        refund_total=round(sum(returned), 2) if returned else None,
    )


def _item_view(item: OrderItem) -> ItemView:
    """Convert an order item into its model-facing view.

    Args:
        item: The order item.

    Returns:
        The item without internal ids.
    """
    return ItemView(
        order_item_id=item.order_item_id,
        product_name=item.product_name,
        brand=item.brand,
        category=item.category,
        sale_price=item.sale_price,
        status=item.status,
        shipped_on=_day(item.shipped_at),
        delivered_on=_day(item.delivered_at),
        returned_on=_day(item.returned_at),
    )


def list_my_orders(
    repo: OrderRepository,
    user_id: int,
    *,
    status: OrderStatus | None = None,
    date_from: date | None = None,
    date_to: date | None = None,
    limit: int = 10,
) -> OrderListResult:
    """List the customer's orders, newest first, with stuck/lost flags.

    Args:
        repo: Data access.
        user_id: The verified customer (from graph state, never from the model).
        status: Only orders with this status, if given.
        date_from: Only orders placed on or after this date, if given.
        date_to: Only orders placed on or before this date, if given.
        limit: Maximum number of orders to return.

    Returns:
        ``OK`` with the orders (possibly none) and ``has_more``, or
        ``INVALID_ARGUMENTS`` if the date range is reversed.
    """
    if date_from is not None and date_to is not None and date_from > date_to:
        return OrderListResult(
            reason_code=ReasonCode.INVALID_ARGUMENTS,
            detail="date_from must be on or before date_to",
        )
    today = repo.reference_now()
    # One extra row tells us whether there are more orders than `limit`.
    orders = repo.list_orders(
        user_id, status=status, created_from=date_from, created_to=date_to, limit=limit + 1
    )
    return OrderListResult(
        reason_code=ReasonCode.OK,
        orders=[_summarize(o, today) for o in orders[:limit]],
        has_more=len(orders) > limit,
    )


def get_order_details(repo: OrderRepository, user_id: int, order_id: int) -> OrderDetailsResult:
    """Return one of the customer's orders with its items.

    Args:
        repo: Data access.
        user_id: The verified customer (from graph state, never from the model).
        order_id: The order the customer asked about.

    Returns:
        ``OK`` with the order and items, or ``NOT_FOUND`` if the order doesn't exist or
        belongs to someone else; the two cases are indistinguishable (rule 4).
    """
    order = repo.get_order(user_id, order_id)
    if order is None:
        return OrderDetailsResult(reason_code=ReasonCode.NOT_FOUND)
    summary = _summarize(order, repo.reference_now())
    details = OrderDetails(
        **dict(summary),  # field values as-is (no re-serialisation)
        items=[_item_view(i) for i in order.items],
    )
    return OrderDetailsResult(reason_code=ReasonCode.OK, order=details)


__all__ = ["get_order_details", "list_my_orders"]
