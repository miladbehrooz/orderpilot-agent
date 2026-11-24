"""Cancellation and return eligibility checks (scenarios P1-P8).

Framework-free: takes a repository, the verified ``user_id`` and plain values, and
returns a result model. The decisions come from :mod:`orderpilot.policy`, which the
Milestone 2 write tools will call again before writing (rule 7). These functions only
check; they never change anything.
"""

from collections.abc import Sequence
from datetime import datetime
from typing import Any

from orderpilot import policy
from orderpilot.domain import Order, OrderItem
from orderpilot.repository import OrderRepository
from orderpilot.services.results import (
    CancellationResult,
    ItemReturnView,
    ReasonCode,
    ReturnEligibilityResult,
)


def _order_facts(order: Order, today: datetime) -> dict[str, Any]:
    """Return the order facts both eligibility results carry (``OrderCheckResult``).

    Args:
        order: The customer's order.
        today: TODAY, from ``meta.reference_now``.

    Returns:
        The ``OrderCheckResult`` fields as keyword arguments.
    """
    return {
        "order_id": order.order_id,
        "status": order.status,
        "created_on": order.created_at.date(),
        "shipped_on": None if order.shipped_at is None else order.shipped_at.date(),
        "delivered_on": None if order.delivered_at is None else order.delivered_at.date(),
        "days_in_status": policy.days_in_status(order, today),
        "is_stuck": policy.is_stuck(order, today),
        "is_possibly_lost": policy.is_possibly_lost(order, today),
    }


def check_cancellation_eligibility(
    repo: OrderRepository, user_id: int, order_id: int
) -> CancellationResult:
    """Check whether one of the customer's orders can be cancelled.

    Args:
        repo: Data access.
        user_id: The verified customer (from graph state, never from the model).
        order_id: The order to check.

    Returns:
        The decision with the order's status, dates, days in status and stuck/lost flags
        (the same as in the order summary, so the agent needs no second lookup to flag a
        stuck or lost order), or ``NOT_FOUND`` if the order doesn't exist or belongs to
        someone else (rule 4).
    """
    order = repo.get_order(user_id, order_id)
    if order is None:
        return CancellationResult(reason_code=ReasonCode.NOT_FOUND)
    decision = policy.cancel_decision(order)
    return CancellationResult(
        reason_code=decision.reason_code,
        eligible=decision.eligible,
        **_order_facts(order, repo.reference_now()),
    )


def _item_view(item: OrderItem, decision: policy.ReturnDecision) -> ItemReturnView:
    """Combine an item and its return decision into the model-facing view.

    Args:
        item: The item.
        decision: Its return decision.

    Returns:
        The item's return view.
    """
    return ItemReturnView(
        order_item_id=item.order_item_id,
        product_name=item.product_name,
        category=item.category,
        sale_price=item.sale_price,
        eligible=decision.eligible,
        reason_code=decision.reason_code,
        refund_amount=decision.refund_amount,
        delivered_on=decision.delivered_on,
        days_since_delivery=decision.days_since_delivery,
        return_deadline=decision.return_deadline,
    )


def check_return_eligibility(
    repo: OrderRepository,
    user_id: int,
    order_id: int,
    order_item_ids: Sequence[int] | None = None,
) -> ReturnEligibilityResult:
    """Check whether items of one of the customer's orders can be returned.

    Args:
        repo: Data access.
        user_id: The verified customer (from graph state, never from the model).
        order_id: The order to check.
        order_item_ids: Items to check (duplicates are ignored), or ``None`` for all.

    Returns:
        One decision per item with the combined ``reason_code`` and refund total, and the
        order's status, dates, days in status and stuck/lost flags (as in the order
        summary, so the agent needs no second lookup to explain the decision).
        ``NOT_FOUND`` if the order doesn't exist or belongs to someone else (rule 4), or
        if any requested item is not in this order; then nothing is checked and nothing
        about the order is returned.
    """
    order = repo.get_order(user_id, order_id)
    if order is None:
        return ReturnEligibilityResult(reason_code=ReasonCode.NOT_FOUND)

    items = list(order.items)
    if order_item_ids is not None:
        by_id = {i.order_item_id: i for i in order.items}
        wanted = list(dict.fromkeys(order_item_ids))  # dedupe, keep the caller's order
        unknown = [i for i in wanted if i not in by_id]
        if unknown:
            return ReturnEligibilityResult(
                reason_code=ReasonCode.NOT_FOUND,
                detail=f"order_item_ids not in this order: {unknown}",
            )
        items = [by_id[i] for i in wanted]

    today = repo.reference_now()
    decisions = [policy.return_decision(i, today) for i in items]
    refunds = [d.refund_amount for d in decisions if d.refund_amount is not None]
    return ReturnEligibilityResult(
        reason_code=policy.overall_return_code(decisions),
        eligible=any(d.eligible for d in decisions),
        refund_total=round(sum(refunds), 2) if refunds else None,
        items=[_item_view(i, d) for i, d in zip(items, decisions, strict=True)],
        **_order_facts(order, today),
    )


__all__ = ["check_cancellation_eligibility", "check_return_eligibility"]
