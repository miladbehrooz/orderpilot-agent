"""Structured results returned by services and shown to the model as JSON.

Every result carries a :class:`ReasonCode`. Evals grade on these codes, and the model
only explains them, so business decisions never depend on the model's wording.
"""

from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, SerializerFunctionWrapHandler, model_serializer

from orderpilot.domain import OrderStatus


class ReasonCode(StrEnum):
    """Machine-readable outcome codes shared by all tools."""

    OK = "OK"
    NOT_FOUND = "NOT_FOUND"
    ELIGIBLE = "ELIGIBLE"
    ALREADY_SHIPPED = "ALREADY_SHIPPED"
    NOT_DELIVERED = "NOT_DELIVERED"
    WINDOW_EXPIRED = "WINDOW_EXPIRED"
    ALREADY_RETURNED = "ALREADY_RETURNED"
    ORDER_CANCELLED = "ORDER_CANCELLED"
    VERIFIED = "VERIFIED"
    NOT_MATCHED = "NOT_MATCHED"
    MISSING_DETAILS = "MISSING_DETAILS"
    LOCKED = "LOCKED"
    ALREADY_VERIFIED = "ALREADY_VERIFIED"
    NOT_VERIFIED = "NOT_VERIFIED"
    INVALID_ARGUMENTS = "INVALID_ARGUMENTS"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class ToolResult(BaseModel):
    """Base result returned to the model as JSON.

    Attributes:
        reason_code: Outcome code.
        detail: Optional extra explanation, e.g. why tool arguments were invalid.
    """

    model_config = ConfigDict(frozen=True)

    reason_code: ReasonCode
    detail: str | None = None

    def to_content(self) -> str:
        """Serialize the result as ``ToolMessage`` content.

        Returns:
            Compact JSON without unset (``None``) fields.
        """
        return self.model_dump_json(exclude_none=True)


class VerifyResult(ToolResult):
    """Outcome of an identity check. It never says which detail failed (AU2, AU4).

    Attributes:
        attempts_remaining: Verification attempts left before lockout, when relevant.
    """

    attempts_remaining: int | None = None


@dataclass(frozen=True, slots=True)
class VerificationOutcome:
    """Full output of the verification service.

    Attributes:
        result: The model-facing result.
        user_id: The verified account, set only on success. Kept out of ``result`` so
            the model never sees it.
        counts_as_failure: Whether this call used up one verification attempt.
    """

    result: VerifyResult
    user_id: int | None = None
    counts_as_failure: bool = False


class ShowsNulls(BaseModel):
    """Base for item models whose ``always_shown`` fields stay visible as ``null``.

    ``ToolResult.to_content`` drops ``None`` fields to keep results short. Some fields
    must still appear when empty, so the model can see that e.g. a product name is
    missing instead of guessing one (Q3).
    """

    model_config = ConfigDict(frozen=True)

    always_shown: ClassVar[frozenset[str]] = frozenset()

    @model_serializer(mode="wrap")
    def _keep_always_shown(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        """Serialize normally, then put back ``always_shown`` fields if they were dropped.

        Args:
            handler: Pydantic's default serializer for this model.

        Returns:
            The serialized fields in declaration order.
        """
        data: dict[str, Any] = handler(self)
        return {
            name: data.get(name, getattr(self, name))
            for name in type(self).model_fields
            if name in data or name in self.always_shown
        }


class ItemView(ShowsNulls):
    """One order item as the model sees it. Internal ids (product, inventory) stay hidden.

    Attributes:
        order_item_id: Item id, used to refer to a single item (e.g. a partial return).
        product_name: Product name; serialised as ``null`` when missing (Q3).
        brand: Brand; serialised as ``null`` when missing.
        category: Product category.
        sale_price: Price paid, which is also the refund amount.
        status: Item status.
        shipped_on: Ship date, if shipped.
        delivered_on: Delivery date, if delivered.
        returned_on: Return date, if returned.
    """

    always_shown: ClassVar[frozenset[str]] = frozenset({"product_name", "brand"})

    order_item_id: int
    product_name: str | None
    brand: str | None
    category: str
    sale_price: float
    status: OrderStatus
    shipped_on: date | None = None
    delivered_on: date | None = None
    returned_on: date | None = None


class OrderSummary(BaseModel):
    """An order at a glance, with the flags and numbers the rules computed.

    Attributes:
        order_id: Order id, as the customer refers to it.
        status: Order status.
        created_on: Order date.
        shipped_on: Ship date, if shipped.
        delivered_on: Delivery date, if delivered.
        returned_on: Return date, if returned.
        item_count: Number of items.
        total: Sum of the items' sale prices.
        days_in_status: Calendar days since the order entered its current status.
        is_stuck: Processing for longer than ``STUCK_DAYS`` (Q1).
        is_possibly_lost: Shipped longer than ``LOST_DAYS`` ago without delivery (Q2).
        refund_total: Sum of the returned items' sale prices; only set if any item was
            returned (L5).
    """

    model_config = ConfigDict(frozen=True)

    order_id: int
    status: OrderStatus
    created_on: date
    shipped_on: date | None = None
    delivered_on: date | None = None
    returned_on: date | None = None
    item_count: int
    total: float
    days_in_status: int
    is_stuck: bool
    is_possibly_lost: bool
    refund_total: float | None = None


class OrderDetails(OrderSummary):
    """An order summary plus its items.

    Attributes:
        items: The order's items.
    """

    items: list[ItemView]


class OrderListResult(ToolResult):
    """Result of ``list_my_orders``.

    Attributes:
        orders: Matching orders, newest first; an empty list means none (L6). Unset when
            the call failed.
        has_more: Whether more orders match than were returned.
    """

    orders: list[OrderSummary] | None = None
    has_more: bool | None = None


class OrderDetailsResult(ToolResult):
    """Result of ``get_order_details``.

    Attributes:
        order: The order with its items; unset when it was not found.
    """

    order: OrderDetails | None = None


class CancellationResult(ToolResult):
    """Result of ``check_cancellation_eligibility`` (P1, P2). It only checks; it never cancels.

    ``reason_code`` is ``ELIGIBLE`` or the reason it can't be cancelled. All other fields
    are unset when the order was not found.

    Attributes:
        order_id: The order checked.
        status: Its status, which decides eligibility (only Processing can be cancelled).
        eligible: Whether the order can be cancelled.
        created_on: Order date.
        shipped_on: Ship date, if shipped.
        delivered_on: Delivery date, if delivered.
        days_in_status: Calendar days since the order entered its current status.
        is_stuck: Processing for longer than ``STUCK_DAYS`` (Q1). A stuck order can still
            be cancelled.
        is_possibly_lost: Shipped longer than ``LOST_DAYS`` ago without delivery (Q2).
    """

    order_id: int | None = None
    status: OrderStatus | None = None
    eligible: bool | None = None
    created_on: date | None = None
    shipped_on: date | None = None
    delivered_on: date | None = None
    days_in_status: int | None = None
    is_stuck: bool | None = None
    is_possibly_lost: bool | None = None


class ItemReturnView(ShowsNulls):
    """The return decision for one item (P3-P8).

    Attributes:
        order_item_id: The item checked.
        product_name: Product name; serialised as ``null`` when missing (Q3).
        category: Product category.
        sale_price: Price paid.
        eligible: Whether this item can be returned.
        reason_code: ``ELIGIBLE`` or why it can't be returned.
        refund_amount: Refund if returned (its sale price); only set when eligible.
        delivered_on: Delivery date, if delivered.
        days_since_delivery: Calendar days since delivery, if delivered.
        return_deadline: Last day a return is accepted, if delivered.
    """

    always_shown: ClassVar[frozenset[str]] = frozenset({"product_name"})

    order_item_id: int
    product_name: str | None
    category: str
    sale_price: float
    eligible: bool
    reason_code: ReasonCode
    refund_amount: float | None = None
    delivered_on: date | None = None
    days_since_delivery: int | None = None
    return_deadline: date | None = None


class ReturnEligibilityResult(ToolResult):
    """Result of ``check_return_eligibility``. It only checks; it never records a return.

    ``reason_code`` is ``ELIGIBLE`` if at least one checked item can be returned,
    otherwise the code the items share (or the first item's, if they differ). All other
    fields are unset when the order or an item was not found.

    Attributes:
        order_id: The order checked.
        eligible: Whether at least one checked item can be returned.
        refund_total: Sum of the refunds of the eligible items.
        items: One decision per checked item.
    """

    order_id: int | None = None
    eligible: bool | None = None
    refund_total: float | None = None
    items: list[ItemReturnView] | None = None


__all__ = [
    "CancellationResult",
    "ItemReturnView",
    "ItemView",
    "OrderDetails",
    "OrderDetailsResult",
    "OrderListResult",
    "OrderSummary",
    "ReasonCode",
    "ReturnEligibilityResult",
    "ShowsNulls",
    "ToolResult",
    "VerificationOutcome",
    "VerifyResult",
]
