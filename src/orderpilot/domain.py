"""Domain objects shared by repositories, services and tools.

These are plain dataclasses with no storage details, so any repository implementation
(SQLite, Postgres, a remote service) can return them.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


@dataclass(frozen=True, slots=True)
class User:
    """A customer account, with the fields needed for identity verification.

    Attributes:
        user_id: Primary key of the account.
        first_name: First name on the account.
        last_name: Last name on the account.
        email: Account email (unique, case-insensitive).
        postal_code: Postal code of the account address.
    """

    user_id: int
    first_name: str
    last_name: str
    email: str
    postal_code: str


class OrderStatus(StrEnum):
    """Lifecycle status of an order or order item."""

    # The values match the CHECK constraint in the database schema. This docstring is
    # part of the tool schema the model sees, so keep it short and customer-facing.

    PROCESSING = "Processing"
    SHIPPED = "Shipped"
    COMPLETE = "Complete"
    RETURNED = "Returned"
    CANCELLED = "Cancelled"


@dataclass(frozen=True, slots=True)
class OrderItem:
    """One purchased item of an order, with its product details.

    Attributes:
        order_item_id: Primary key of the item.
        product_name: Product name, or ``None`` if the catalogue has none (scenario Q3).
        brand: Product brand, or ``None`` if the catalogue has none.
        category: Product category, e.g. ``"Jeans"``.
        sale_price: Price paid for the item, which is also its refund amount.
        status: Item status (the same as its order's until partial returns exist).
        created_at: When the item was ordered (UTC).
        shipped_at: When it shipped, if it has.
        delivered_at: When it was delivered, if it has been.
        returned_at: When it was returned, if it has been.
        return_requested: Whether a return request exists for it (``return_requests``).
    """

    order_item_id: int
    product_name: str | None
    brand: str | None
    category: str
    sale_price: float
    status: OrderStatus
    created_at: datetime
    shipped_at: datetime | None
    delivered_at: datetime | None
    returned_at: datetime | None
    return_requested: bool = False


@dataclass(frozen=True, slots=True)
class Order:
    """A customer's order with all of its items.

    Attributes:
        order_id: Primary key of the order.
        status: Order status.
        created_at: When the order was placed (UTC).
        shipped_at: When it shipped, if it has.
        delivered_at: When it was delivered, if it has been.
        returned_at: When it was returned, if it has been.
        items: The order's items, ordered by ``order_item_id``.
    """

    order_id: int
    status: OrderStatus
    created_at: datetime
    shipped_at: datetime | None
    delivered_at: datetime | None
    returned_at: datetime | None
    items: tuple[OrderItem, ...]


__all__ = ["Order", "OrderItem", "OrderStatus", "User"]
