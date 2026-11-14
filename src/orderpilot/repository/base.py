"""The data-access interface that services depend on.

Services and tools only use :class:`OrderRepository`, never a concrete database, so
the storage can change (SQLite, Postgres, a remote service) without touching them.
"""

from datetime import datetime
from typing import Protocol

from orderpilot.domain import User


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


__all__ = ["OrderRepository"]
