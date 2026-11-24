"""Eligibility tools through a real ToolNode: auth guard, user_id from state, argument errors."""

from datetime import timedelta
from typing import Any

import pytest
from langchain.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool

from orderpilot.repository import SqliteOrderRepository
from tests.shop_data import ShopData
from tests.tool_runner import run_tool_call, tools_named


@pytest.fixture
def eligibility_tools(shop_repo: SqliteOrderRepository) -> list[BaseTool]:
    """Return the two eligibility tools.

    Args:
        shop_repo: Repository on the test shop.

    Returns:
        ``check_cancellation_eligibility`` and ``check_return_eligibility``.
    """
    return tools_named(shop_repo, "check_cancellation_eligibility", "check_return_eligibility")


def test_eligibility_tools_are_registered(eligibility_tools: list[BaseTool]) -> None:
    """``build_tools`` provides both eligibility tools.

    Args:
        eligibility_tools: The eligibility tools.
    """
    assert [t.name for t in eligibility_tools] == [
        "check_cancellation_eligibility",
        "check_return_eligibility",
    ]


@pytest.mark.parametrize("name", ["check_cancellation_eligibility", "check_return_eligibility"])
def test_eligibility_tools_refuse_without_verification(
    shop_repo: SqliteOrderRepository, shop: ShopData, name: str
) -> None:
    """Rule 3: called before verification, a tool refuses and shows no data.

    Args:
        shop_repo: Repository on the test shop.
        shop: Test shop ids.
        name: Tool to call.
    """
    args = {"order_id": shop.orders.processing}
    assert run_tool_call(shop_repo, name, args) == {"reason_code": "NOT_VERIFIED"}


def test_tool_schemas_have_no_user_id(eligibility_tools: list[BaseTool]) -> None:
    """Rule 2: the model cannot pass a user id; ``runtime`` stays hidden too.

    Args:
        eligibility_tools: The eligibility tools.
    """
    for t in eligibility_tools:
        # The function schema exactly as it is sent to the model.
        properties = convert_to_openai_tool(t)["function"]["parameters"]["properties"]
        assert "user_id" not in properties
        assert "runtime" not in properties


def test_user_id_comes_from_state(shop_repo: SqliteOrderRepository, shop: ShopData) -> None:
    """The same call sees an order only when the state's user owns it.

    Args:
        shop_repo: Repository on the test shop.
        shop: Test shop ids.
    """
    args = {"order_id": shop.orders.other_users}
    name = "check_cancellation_eligibility"
    as_customer = run_tool_call(shop_repo, name, args, user_id=shop.customer_id)
    as_owner = run_tool_call(shop_repo, name, args, user_id=shop.other_id)
    assert as_customer == {"reason_code": "NOT_FOUND"}
    assert as_owner["reason_code"] == "ELIGIBLE"


def test_cancellation_check_shows_the_stuck_flag(
    shop_repo: SqliteOrderRepository, shop: ShopData
) -> None:
    """The model sees ``is_stuck`` and the days in status, so it needs no second lookup.

    Args:
        shop_repo: Repository on the test shop.
        shop: Test shop ids.
    """
    args = {"order_id": shop.orders.stuck}
    content = run_tool_call(
        shop_repo, "check_cancellation_eligibility", args, user_id=shop.customer_id
    )
    assert content["reason_code"] == "ELIGIBLE"
    assert (content["is_stuck"], content["is_possibly_lost"], content["days_in_status"]) == (
        True,
        False,
        400,
    )


def test_return_check_shows_the_order_status(
    shop_repo: SqliteOrderRepository, shop: ShopData
) -> None:
    """The model sees the order's status and ship date, so it needs no second lookup.

    Args:
        shop_repo: Repository on the test shop.
        shop: Test shop ids.
    """
    content = run_tool_call(
        shop_repo,
        "check_return_eligibility",
        {"order_id": shop.orders.shipped},
        user_id=shop.customer_id,
    )
    assert (content["reason_code"], content["status"]) == ("NOT_DELIVERED", "Shipped")
    assert content["shipped_on"] == (shop.today - timedelta(days=2)).date().isoformat()
    assert "delivered_on" not in content


def test_return_check_for_one_item(shop_repo: SqliteOrderRepository, shop: ShopData) -> None:
    """A verified call for one item returns that item's decision as JSON.

    Args:
        shop_repo: Repository on the test shop.
        shop: Test shop ids.
    """
    content = run_tool_call(
        shop_repo,
        "check_return_eligibility",
        {"order_id": shop.orders.complete, "order_item_ids": [1051]},
        user_id=shop.customer_id,
    )
    assert content["reason_code"] == "ELIGIBLE"
    (item,) = content["items"]
    assert (item["order_item_id"], item["refund_amount"]) == (1051, 40.0)
    assert item["return_deadline"] == "2026-11-19"


@pytest.mark.parametrize(
    ("name", "args", "field"),
    [
        pytest.param(
            "check_return_eligibility",
            {"order_id": 105, "order_item_ids": []},
            "order_item_ids",
            id="empty-item-list",
        ),
        pytest.param(
            "check_return_eligibility",
            {"order_id": 105, "order_item_ids": ["abc"]},
            "order_item_ids",
            id="non-int-item-id",
        ),
        pytest.param("check_return_eligibility", {}, "order_id", id="return-missing-id"),
        pytest.param("check_cancellation_eligibility", {}, "order_id", id="cancel-missing-id"),
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
