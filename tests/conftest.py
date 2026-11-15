"""Shared fixtures: a stub repository, a small test shop database and the real database."""

import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

import pytest

from orderpilot.domain import User
from orderpilot.repository import SqliteOrderRepository
from orderpilot.settings import settings
from tests.shop_data import ShopData, build_shop_db
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
def today() -> datetime:
    """Return TODAY of the test shop (its ``meta.reference_now``).

    Returns:
        2026-11-12 at midnight UTC.
    """
    return datetime(2026, 11, 12, tzinfo=UTC)


@pytest.fixture
def shop(tmp_path: Path, today: datetime) -> ShopData:
    """Build a fresh test shop database for one test.

    Args:
        tmp_path: Pytest temporary directory.
        today: TODAY of the dataset.

    Returns:
        Handles to the shop's file and row ids.
    """
    return build_shop_db(tmp_path / "shop.sqlite", today)


@pytest.fixture
def shop_repo(shop: ShopData) -> SqliteOrderRepository:
    """Return a repository on the test shop database.

    Args:
        shop: The test shop.

    Returns:
        The SQLite repository.
    """
    return SqliteOrderRepository(shop.path)


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
