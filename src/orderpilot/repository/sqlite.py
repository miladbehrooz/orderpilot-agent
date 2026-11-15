"""SQLite implementation of :class:`~orderpilot.repository.base.OrderRepository`."""

from contextlib import closing
from datetime import datetime
from pathlib import Path

from orderpilot.db import connect, parse_timestamp
from orderpilot.domain import User
from orderpilot.errors import RepositoryError


class SqliteOrderRepository:
    """``OrderRepository`` backed by the SQLite file built by ``scripts/build_db.py``.

    Each method opens its own short-lived connection (see :func:`orderpilot.db.connect`).

    Args:
        db_path: Path to the SQLite database.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path

    def reference_now(self) -> datetime:
        """Return TODAY, the fixed reference time of the dataset.

        Returns:
            ``meta.reference_now`` as a timezone-aware UTC datetime.

        Raises:
            RepositoryError: If the database has no ``reference_now`` entry.
        """
        with closing(connect(self._db_path)) as con:
            row = con.execute("SELECT value FROM meta WHERE key = 'reference_now'").fetchone()
        if row is None:
            raise RepositoryError("meta.reference_now is missing; rebuild the database")
        return parse_timestamp(row["value"])

    def find_user_by_email(self, email: str) -> User | None:
        """Look up an account by email, ignoring case.

        Args:
            email: Email address to look up.

        Returns:
            The matching user, or ``None`` if there is none.
        """
        # users.email is declared COLLATE NOCASE, so `=` is already case-insensitive.
        with closing(connect(self._db_path)) as con:
            row = con.execute(
                "SELECT id, first_name, last_name, email, postal_code FROM users WHERE email = ?",
                (email,),
            ).fetchone()
        if row is None:
            return None
        return User(
            user_id=row["id"],
            first_name=row["first_name"],
            last_name=row["last_name"],
            email=row["email"],
            postal_code=row["postal_code"],
        )


__all__ = ["SqliteOrderRepository"]
