"""SQLite helpers: opening connections and parsing the database's timestamps."""

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from orderpilot.errors import RepositoryError


def connect(db_path: Path, *, read_only: bool = True) -> sqlite3.Connection:
    """Open a SQLite connection whose rows support access by column name.

    Open one connection per unit of work. ``ToolNode`` runs parallel tool calls in
    worker threads, and a ``sqlite3`` connection must not be shared across threads.

    Args:
        db_path: Path to the database built by ``scripts/build_db.py``.
        read_only: Open with ``mode=ro`` so the connection cannot write.

    Returns:
        An open connection with ``sqlite3.Row`` as its row factory.

    Raises:
        RepositoryError: If the database file does not exist.
    """
    if not db_path.is_file():
        raise RepositoryError(f"Database not found at {db_path}; run scripts/build_db.py first")
    mode = "ro" if read_only else "rw"
    con = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode={mode}", uri=True)
    con.row_factory = sqlite3.Row
    return con


def parse_timestamp(value: str) -> datetime:
    """Parse a database timestamp as a timezone-aware UTC datetime.

    Args:
        value: Timestamp text in the database format ``YYYY-MM-DD HH:MM:SS`` (UTC).

    Returns:
        The timestamp with ``tzinfo=UTC``.
    """
    return datetime.fromisoformat(value).replace(tzinfo=UTC)


__all__ = ["connect", "parse_timestamp"]
