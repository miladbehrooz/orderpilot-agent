"""Shared fixtures: stub repository, test shop, real database, LangSmith run builders."""

import os
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pytest_mock import MockerFixture

from orderpilot.domain import User
from orderpilot.repository import SqliteOrderRepository
from orderpilot.settings import settings
from tests.eval_runs import ExampleFactory, RunFactory, build_example, build_run
from tests.shop_data import ShopData, build_shop_db
from tests.stubs import StubRepository, verification_details


@pytest.fixture(autouse=True)
def _no_tracing(mocker: MockerFixture) -> None:
    """Keep LangSmith tracing off in every test, whatever .env says.

    Args:
        mocker: Pytest-mock fixture.
    """
    mocker.patch.object(settings, "LANGSMITH_TRACING", False)
    mocker.patch.dict(os.environ, {"LANGSMITH_TRACING": "false"})


@pytest.fixture
def stub_user() -> User:
    """Return the one account the stub repository knows.

    Returns:
        Ann Lee, user 42.
    """
    return User(
        user_id=42,
        first_name="Ann",
        last_name="Lee",
        email="ann.lee@example.com",
        postal_code="10115",
    )


@pytest.fixture
def stub_details(stub_user: User) -> dict[str, str]:
    """Return the correct verification details of ``stub_user``.

    Args:
        stub_user: The stub account.

    Returns:
        The four ``verify_customer`` arguments.
    """
    return verification_details(stub_user)


@pytest.fixture
def stub_repo(stub_user: User, today: datetime) -> StubRepository:
    """Return a stub repository holding ``stub_user``.

    Args:
        stub_user: The stub account.
        today: TODAY the stub reports.

    Returns:
        The stub repository.
    """
    return StubRepository([stub_user], today)


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


@pytest.fixture
def make_run() -> RunFactory:
    """Return a builder of target runs.

    Returns:
        ``make_run(answer=..., error=..., messages=...)`` building a real LangSmith ``Run``.
    """
    return build_run


@pytest.fixture
def make_example() -> ExampleFactory:
    """Return a builder of dataset examples.

    Returns:
        ``make_example(metrics=..., expected_tool_calls=..., expected_reason_code=...)``
        building a real LangSmith ``Example``.
    """
    return build_example
