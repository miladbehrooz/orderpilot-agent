"""Shared fixtures: an in-memory stub repository and the real SQLite database."""

import sqlite3
from contextlib import closing

import pytest

from orderpilot.domain import User
from orderpilot.repository import SqliteOrderRepository
from orderpilot.settings import settings
from tests.stubs import STUB_USER, StubRepository


@pytest.fixture(autouse=True)
def _no_tracing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep LangSmith tracing off in every test, whatever .env says.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
    """
    monkeypatch.setattr(settings, "LANGSMITH_TRACING", False)
    monkeypatch.setenv("LANGSMITH_TRACING", "false")


@pytest.fixture
def stub_repo() -> StubRepository:
    """Return a stub repository holding ``STUB_USER``.

    Returns:
        The stub repository.
    """
    return StubRepository([STUB_USER])


@pytest.fixture
def sqlite_repo() -> SqliteOrderRepository:
    """Return a repository on the real database, skipping if it hasn't been built.

    Returns:
        The SQLite repository at ``settings.DB_PATH``.
    """
    if not settings.DB_PATH.is_file():
        pytest.skip(f"database not built at {settings.DB_PATH}")
    return SqliteOrderRepository(settings.DB_PATH)


@pytest.fixture
def known_user(sqlite_repo: SqliteOrderRepository) -> User:
    """Return a real user from the database.

    Args:
        sqlite_repo: Repository on the real database (skips if missing).

    Returns:
        The first user by id.
    """
    with closing(sqlite3.connect(settings.DB_PATH)) as con:
        (email,) = con.execute("SELECT email FROM users ORDER BY id LIMIT 1").fetchone()
    user = sqlite_repo.find_user_by_email(email)
    assert user is not None
    return user
