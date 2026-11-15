"""The data-access interface that services depend on.

Services and tools only use :class:`OrderRepository`, never a concrete database, so
the storage can change (SQLite, Postgres, a remote service) without touching them.
"""

from datetime import date, datetime
from typing import Protocol

from orderpilot.domain import Order, OrderStatus, User


class OrderRepository(Protocol):
    """Read and write access to shop data.

    Every order-level method takes a ``user_id`` and only returns that user's data, so
    another user's order looks exactly like a missing one (rule 4).
    """

    def reference_now(self) -> datetime:
        """Return TODAY, the fixed reference time of the dataset.

        Returns:
            ``meta.reference_now`` as a timezone-aware UTC datetime.
        """
        ...

    def find_user_by_email(self, email: str) -> User | None:
        """Look up an account by email, ignoring case.

        Args:
            email: Email address to look up.

        Returns:
            The matching user, or ``None`` if there is none.
        """
        ...

    def list_orders(
        self,
        user_id: int,
        *,
        status: OrderStatus | None,
        created_from: date | None,
        created_to: date | None,
        limit: int,
    ) -> list[Order]:
        """Return a user's orders, newest first, with their items.

        Args:
            user_id: The verified customer.
            status: Only orders with this status, or all statuses if ``None``.
            created_from: Only orders placed on or after this date (UTC), if given.
            created_to: Only orders placed on or before this date (UTC), if given.
            limit: Maximum number of orders to return.

        Returns:
            At most ``limit`` orders, newest first.
        """
        ...

    def get_order(self, user_id: int, order_id: int) -> Order | None:
        """Return one of the user's orders with its items.

        Args:
            user_id: The verified customer.
            order_id: The order to look up.

        Returns:
            The order, or ``None`` if it doesn't exist or belongs to another user.
        """
        ...


__all__ = ["OrderRepository"]
