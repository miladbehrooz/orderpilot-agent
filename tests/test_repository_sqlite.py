"""SqliteOrderRepository against the real database (skipped if it isn't built)."""

import sqlite3
from contextlib import closing
from datetime import UTC
from pathlib import Path

import pytest

from orderpilot.db import parse_timestamp
from orderpilot.domain import User
from orderpilot.errors import RepositoryError
from orderpilot.repository import SqliteOrderRepository
from orderpilot.settings import settings


def test_reference_now_matches_meta(sqlite_repo: SqliteOrderRepository) -> None:
    """TODAY is meta.reference_now as an aware UTC datetime (rule 5).

    Args:
        sqlite_repo: Repository on the real database.
    """
    with closing(sqlite3.connect(settings.DB_PATH)) as con:
        (raw,) = con.execute("SELECT value FROM meta WHERE key = 'reference_now'").fetchone()
    now = sqlite_repo.reference_now()
    assert now == parse_timestamp(raw)
    assert now.tzinfo is UTC


def test_find_user_by_email_ignores_case(
    sqlite_repo: SqliteOrderRepository, known_user: User
) -> None:
    """Email lookup is case-insensitive.

    Args:
        sqlite_repo: Repository on the real database.
        known_user: A real user.
    """
    assert sqlite_repo.find_user_by_email(known_user.email.upper()) == known_user


def test_find_user_by_unknown_email_returns_none(sqlite_repo: SqliteOrderRepository) -> None:
    """An unknown email returns None.

    Args:
        sqlite_repo: Repository on the real database.
    """
    assert sqlite_repo.find_user_by_email("nobody@nowhere.invalid") is None


def test_missing_database_raises(tmp_path: Path) -> None:
    """A missing database file raises RepositoryError, not a sqlite3 error.

    Args:
        tmp_path: Pytest temporary directory.
    """
    repo = SqliteOrderRepository(tmp_path / "missing.sqlite")
    with pytest.raises(RepositoryError, match="build_db"):
        repo.reference_now()
