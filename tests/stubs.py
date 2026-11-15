"""Test doubles that stand in for the data layer, so most tests need no database."""

from datetime import date, datetime

from orderpilot.domain import Order, OrderStatus, User


def verification_details(user: User) -> dict[str, str]:
    """Return the four details that verify a user, as ``verify_customer`` arguments.

    Args:
        user: User whose details to use.

    Returns:
        ``first_name``, ``last_name``, ``email`` and ``postal_code``.
    """
    return {
        "first_name": user.first_name,
        "last_name": user.last_name,
        "email": user.email,
        "postal_code": user.postal_code,
    }


class StubRepository:
    """In-memory ``OrderRepository`` for tests that don't need the real database.

    Args:
        users: Accounts the stub knows about.
        today: What ``reference_now`` returns.
    """

    def __init__(self, users: list[User], today: datetime) -> None:
        self._users = {u.email.casefold(): u for u in users}
        self._today = today

    def reference_now(self) -> datetime:
        """Return the stub's TODAY.

        Returns:
            The ``today`` given at construction.
        """
        return self._today

    def find_user_by_email(self, email: str) -> User | None:
        """Look up a stub account by email, ignoring case.

        Args:
            email: Email address to look up.

        Returns:
            The matching user, or ``None``.
        """
        return self._users.get(email.casefold())

    def list_orders(
        self,
        user_id: int,
        *,
        status: OrderStatus | None,
        created_from: date | None,
        created_to: date | None,
        limit: int,
    ) -> list[Order]:
        """Return no orders: the stub only knows accounts.

        Args:
            user_id: Ignored.
            status: Ignored.
            created_from: Ignored.
            created_to: Ignored.
            limit: Ignored.

        Returns:
            An empty list.
        """
        return []

    def get_order(self, user_id: int, order_id: int) -> Order | None:
        """Return no order: the stub only knows accounts.

        Args:
            user_id: Ignored.
            order_id: Ignored.

        Returns:
            ``None``.
        """
        return None
