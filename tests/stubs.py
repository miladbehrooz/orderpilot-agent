"""Test doubles that stand in for the data layer, so most tests need no database."""

from datetime import UTC, datetime

from orderpilot.domain import User

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
