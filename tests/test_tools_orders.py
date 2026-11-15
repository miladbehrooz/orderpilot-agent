"""Lookup tools through a real ToolNode: auth guard, user_id from state, argument errors."""

from typing import Any

import pytest
from langchain.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool

from orderpilot.repository import SqliteOrderRepository
from tests.shop_data import ShopData
from tests.tool_runner import run_tool_call, tools_named


@pytest.fixture
def lookup_tools(shop_repo: SqliteOrderRepository) -> list[BaseTool]:
    """Return the two lookup tools.

    Args:
        shop_repo: Repository on the test shop.

    Returns:
        ``list_my_orders`` and ``get_order_details``.
    """
    return tools_named(shop_repo, "list_my_orders", "get_order_details")


def test_lookup_tools_are_registered(lookup_tools: list[BaseTool]) -> None:
    """``build_tools`` provides both lookup tools.

    Args:
        lookup_tools: The customer tools.
    """
    assert [t.name for t in lookup_tools] == ["list_my_orders", "get_order_details"]


@pytest.mark.parametrize(
    ("name", "args"),
    [("list_my_orders", {}), ("get_order_details", {"order_id": 101})],
)
def test_au5_lookup_tools_refuse_without_verification(
    shop_repo: SqliteOrderRepository, name: str, args: dict[str, Any]
) -> None:
    """AU5 / rule 3: called before verification, a tool refuses and shows no data.

    Args:
        shop_repo: Repository on the test shop.
        name: Tool to call.
        args: Valid arguments for it.
    """
    assert run_tool_call(shop_repo, name, args) == {"reason_code": "NOT_VERIFIED"}


def test_s1_tool_schemas_have_no_user_id(lookup_tools: list[BaseTool]) -> None:
    """S1 / rule 2: the model cannot pass a user id; ``runtime`` stays hidden too.

    Args:
        lookup_tools: The customer tools.
    """
    for t in lookup_tools:
        # The function schema exactly as it is sent to the model.
        properties = convert_to_openai_tool(t)["function"]["parameters"]["properties"]
        assert "user_id" not in properties
        assert "runtime" not in properties


def test_s1_user_id_comes_from_state(shop_repo: SqliteOrderRepository, shop: ShopData) -> None:
    """S1: the same call sees an order only when the state's user owns it.

    Args:
        shop_repo: Repository on the test shop.
        shop: Test shop ids.
    """
    args = {"order_id": shop.orders.other_users}
    as_customer = run_tool_call(shop_repo, "get_order_details", args, user_id=shop.customer_id)
    as_owner = run_tool_call(shop_repo, "get_order_details", args, user_id=shop.other_id)
    assert as_customer == {"reason_code": "NOT_FOUND"}
    assert as_owner["reason_code"] == "OK"


def test_list_my_orders_returns_customer_orders(
    shop_repo: SqliteOrderRepository, shop: ShopData
) -> None:
    """A verified call returns the customer's orders, with dates as YYYY-MM-DD.

    Args:
        shop_repo: Repository on the test shop.
        shop: Test shop ids.
    """
    content = run_tool_call(
        shop_repo,
        "list_my_orders",
        {"status": "Shipped", "date_from": "2026-11-01"},
        user_id=shop.customer_id,
    )
    assert content["reason_code"] == "OK"
    (order,) = content["orders"]
    assert order["order_id"] == shop.orders.shipped
    assert order["shipped_on"] == "2026-11-10"


@pytest.mark.parametrize(
    ("name", "args", "field"),
    [
        pytest.param("list_my_orders", {"status": "Lost"}, "status", id="unknown-status"),
        pytest.param("list_my_orders", {"limit": 0}, "limit", id="limit-too-small"),
        pytest.param("list_my_orders", {"limit": 51}, "limit", id="limit-too-large"),
        pytest.param("list_my_orders", {"date_from": "last month"}, "date_from", id="bad-date"),
        pytest.param("get_order_details", {}, "order_id", id="missing-order-id"),
        pytest.param("get_order_details", {"order_id": "abc"}, "order_id", id="non-int-id"),
    ],
)
def test_invalid_arguments_return_structured_error(
    shop_repo: SqliteOrderRepository, shop: ShopData, name: str, args: dict[str, Any], field: str
) -> None:
    """Rule 9: arguments that fail the schema return INVALID_ARGUMENTS naming the field.

    Args:
        shop_repo: Repository on the test shop.
        shop: Test shop ids.
        name: Tool to call.
        args: Invalid arguments.
        field: The argument the error must mention.
    """
    content = run_tool_call(shop_repo, name, args, user_id=shop.customer_id)
    assert content["reason_code"] == "INVALID_ARGUMENTS"
    assert field in content["detail"]
