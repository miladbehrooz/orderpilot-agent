"""Unexpected failures inside a real tool reach the model as INTERNAL_ERROR (rule 9)."""

import sqlite3
from collections.abc import Iterator
from typing import Any

import pytest
from pytest_mock import MockerFixture

from orderpilot.observability import logger
from orderpilot.repository import SqliteOrderRepository
from tests.shop_data import ShopData
from tests.tool_runner import run_tool_call


@pytest.fixture
def error_logs() -> Iterator[list[str]]:
    """Collect every ERROR log record, with its traceback, while the test runs.

    Yields:
        The formatted log messages, in order.
    """
    messages: list[str] = []
    sink_id = logger.add(messages.append, level="ERROR", format="{message}")
    yield messages
    logger.remove(sink_id)


# Any order id will do: the connection fails before the order is looked up.
@pytest.mark.parametrize(
    ("name", "args"),
    [
        pytest.param("list_my_orders", {}, id="list-orders"),
        pytest.param("get_order_details", {"order_id": 1}, id="order-details"),
        pytest.param("check_cancellation_eligibility", {"order_id": 1}, id="cancel-check"),
        pytest.param("check_return_eligibility", {"order_id": 1}, id="return-check"),
    ],
)
def test_database_failure_is_internal_error_and_logged(
    mocker: MockerFixture,
    shop_repo: SqliteOrderRepository,
    shop: ShopData,
    error_logs: list[str],
    name: str,
    args: dict[str, Any],
) -> None:
    """A SQLite failure gives the model only INTERNAL_ERROR; the details go to the log.

    Args:
        mocker: Pytest-mock fixture.
        shop_repo: Repository on the test shop.
        shop: Test shop ids.
        error_logs: Captured ERROR log messages.
        name: Tool to call.
        args: Valid arguments for it.
    """
    mocker.patch(
        "orderpilot.repository.db.sqlite3.connect",
        side_effect=sqlite3.OperationalError("disk I/O error"),
    )
    content = run_tool_call(shop_repo, name, args, user_id=shop.customer_id)
    assert content == {"reason_code": "INTERNAL_ERROR"}
    (log,) = error_logs
    assert "Tool raised an unexpected error" in log
    assert "disk I/O error" in log
