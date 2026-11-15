"""Test doubles that stand in for the data layer, so most tests need no database."""

from datetime import UTC, date, datetime

from orderpilot.domain import Order, OrderStatus, User

STUB_TODAY = datetime(2026, 11, 12, tzinfo=UTC)
STUB_USER = User(
    user_id=42,
    first_name="Ann",
    last_name="Lee",
    email="ann.lee@example.com",
    postal_code="10115",
)


class StubRepository:
    """In-memory ``OrderRepository`` for tests that don't need the real database.

    Args:
        users: Accounts the stub knows about.
    """

    def __init__(self, users: list[User]) -> None:
        self._users = {u.email.casefold(): u for u in users}

    def reference_now(self) -> datetime:
        """Return the fixed stub TODAY.

        Returns:
            ``STUB_TODAY``.
        """
        return STUB_TODAY

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
