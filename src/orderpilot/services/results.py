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


class ItemView(BaseModel):
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

    model_config = ConfigDict(frozen=True)

    # Kept as explicit nulls despite `exclude_none`, so the model can see that a name or
    # brand is missing instead of guessing one (Q3).
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

    @model_serializer(mode="wrap")
    def _keep_missing_names(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        """Serialize normally, then put back ``product_name``/``brand`` if they were dropped.

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


__all__ = [
    "ItemView",
    "OrderDetails",
    "OrderDetailsResult",
    "OrderListResult",
    "OrderSummary",
    "ReasonCode",
    "ToolResult",
    "VerificationOutcome",
    "VerifyResult",
]
