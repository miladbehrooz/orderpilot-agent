"""Cancellation and return eligibility services on the test shop database."""

import json
import sqlite3
from collections.abc import Callable
from contextlib import closing
from datetime import timedelta

import pytest
from pytest_mock import MockerFixture

from orderpilot.domain import OrderStatus
from orderpilot.repository import SqliteOrderRepository
from orderpilot.services import eligibility as service
from orderpilot.services.results import ReasonCode, ToolResult
from orderpilot.settings import settings
from tests.shop_data import ShopData

Check = Callable[[SqliteOrderRepository, int, int], ToolResult]


def test_processing_order_can_be_cancelled(
    shop_repo: SqliteOrderRepository, shop: ShopData
) -> None:
    """A Processing order can be cancelled.

    Args:
        shop_repo: Repository on the test shop.
        shop: Test shop ids.
    """
    result = service.check_cancellation_eligibility(
        shop_repo, shop.customer_id, shop.orders.processing
    )
    assert (result.reason_code, result.eligible) == (ReasonCode.ELIGIBLE, True)
    assert result.status is OrderStatus.PROCESSING


def test_shipped_order_cannot_be_cancelled(
    shop_repo: SqliteOrderRepository, shop: ShopData
) -> None:
    """A Shipped order can't be cancelled, and the result carries its ship date.

    Args:
        shop_repo: Repository on the test shop.
        shop: Test shop ids.
    """
    result = service.check_cancellation_eligibility(
        shop_repo, shop.customer_id, shop.orders.shipped
    )
    assert (result.reason_code, result.eligible) == (ReasonCode.ALREADY_SHIPPED, False)
    assert result.shipped_on == (shop.today - timedelta(days=2)).date()


def test_cancellation_check_flags_a_stuck_order(
    shop_repo: SqliteOrderRepository, shop: ShopData
) -> None:
    """Processing for longer than ``STUCK_DAYS`` is flagged, and the order can still be cancelled.

    Args:
        shop_repo: Repository on the test shop.
        shop: Test shop ids.
    """
    result = service.check_cancellation_eligibility(shop_repo, shop.customer_id, shop.orders.stuck)
    assert (result.reason_code, result.eligible) == (ReasonCode.ELIGIBLE, True)
    assert (result.is_stuck, result.is_possibly_lost, result.days_in_status) == (True, False, 400)


def test_recent_processing_order_is_not_stuck(
    shop_repo: SqliteOrderRepository, shop: ShopData
) -> None:
    """Two days in Processing is not stuck.

    Args:
        shop_repo: Repository on the test shop.
        shop: Test shop ids.
    """
    result = service.check_cancellation_eligibility(
        shop_repo, shop.customer_id, shop.orders.processing
    )
    assert (result.is_stuck, result.days_in_status) == (False, 2)


def test_cancellation_check_flags_a_possibly_lost_order(
    shop_repo: SqliteOrderRepository, shop: ShopData
) -> None:
    """Shipped 60 days ago and never delivered: it can't be cancelled, and may be lost.

    Args:
        shop_repo: Repository on the test shop.
        shop: Test shop ids.
    """
    result = service.check_cancellation_eligibility(shop_repo, shop.customer_id, shop.orders.lost)
    assert result.reason_code is ReasonCode.ALREADY_SHIPPED
    assert (result.is_possibly_lost, result.is_stuck, result.days_in_status) == (True, False, 60)


def test_stuck_threshold_comes_from_settings(
    shop_repo: SqliteOrderRepository, shop: ShopData, mocker: MockerFixture
) -> None:
    """With ``STUCK_DAYS`` raised above 400 days, the same order is no longer stuck.

    Args:
        shop_repo: Repository on the test shop.
        shop: Test shop ids.
        mocker: Pytest-mock fixture.
    """
    mocker.patch.object(settings, "STUCK_DAYS", 500)
    result = service.check_cancellation_eligibility(shop_repo, shop.customer_id, shop.orders.stuck)
    assert result.is_stuck is False


def test_recent_delivery_is_returnable_with_refund(
    shop_repo: SqliteOrderRepository, shop: ShopData
) -> None:
    """Delivered 5 days ago, so the item can be returned for its sale price.

    Args:
        shop_repo: Repository on the test shop.
        shop: Test shop ids.
    """
    result = service.check_return_eligibility(
        shop_repo, shop.customer_id, shop.orders.delivered_recently
    )
    assert (result.reason_code, result.eligible, result.refund_total) == (
        ReasonCode.ELIGIBLE,
        True,
        45.0,
    )
    assert result.items is not None
    (item,) = result.items
    delivered = (shop.today - timedelta(days=5)).date()
    assert (item.refund_amount, item.days_since_delivery) == (45.0, 5)
    assert item.return_deadline == delivered + timedelta(days=30)


def test_old_delivery_is_outside_window(shop_repo: SqliteOrderRepository, shop: ShopData) -> None:
    """Delivered 45 days ago, so the 30-day window has passed.

    Args:
        shop_repo: Repository on the test shop.
        shop: Test shop ids.
    """
    result = service.check_return_eligibility(
        shop_repo, shop.customer_id, shop.orders.delivered_long_ago
    )
    assert (result.reason_code, result.eligible) == (ReasonCode.WINDOW_EXPIRED, False)
    assert result.refund_total is None
    assert result.items is not None
    assert result.items[0].days_since_delivery == 45


@pytest.mark.parametrize(
    ("order", "code"),
    [
        pytest.param("shipped", ReasonCode.NOT_DELIVERED, id="still-shipped"),
        pytest.param("cancelled", ReasonCode.ORDER_CANCELLED, id="cancelled"),
        pytest.param("returned", ReasonCode.ALREADY_RETURNED, id="already-returned"),
    ],
)
def test_order_cannot_be_returned(
    shop_repo: SqliteOrderRepository, shop: ShopData, order: str, code: ReasonCode
) -> None:
    """Not delivered yet, cancelled, or already returned: nothing to return.

    Args:
        shop_repo: Repository on the test shop.
        shop: Test shop ids.
        order: Name of the order in ``shop.orders``.
        code: Expected reason code for the order and every item.
    """
    order_id = getattr(shop.orders, order)
    result = service.check_return_eligibility(shop_repo, shop.customer_id, order_id)
    assert (result.reason_code, result.eligible) == (code, False)
    assert result.items is not None
    assert {i.reason_code for i in result.items} == {code}


def test_return_check_shows_the_order_status_and_dates(
    shop_repo: SqliteOrderRepository, shop: ShopData
) -> None:
    """A shipped order can't be returned yet, and the result says it is on its way.

    Args:
        shop_repo: Repository on the test shop.
        shop: Test shop ids.
    """
    result = service.check_return_eligibility(shop_repo, shop.customer_id, shop.orders.shipped)
    assert result.reason_code is ReasonCode.NOT_DELIVERED
    assert result.status is OrderStatus.SHIPPED
    assert result.shipped_on == (shop.today - timedelta(days=2)).date()
    assert (result.delivered_on, result.days_in_status) == (None, 2)
    assert (result.is_stuck, result.is_possibly_lost) == (False, False)


@pytest.mark.parametrize(
    ("order", "stuck", "lost", "days"),
    [
        pytest.param("stuck", True, False, 400, id="stuck-in-processing"),
        pytest.param("lost", False, True, 60, id="possibly-lost"),
    ],
)
def test_return_check_flags_stuck_and_lost_orders(
    shop_repo: SqliteOrderRepository,
    shop: ShopData,
    order: str,
    stuck: bool,
    lost: bool,
    days: int,
) -> None:
    """Neither order is delivered, and the result carries the flag the answer must mention.

    Args:
        shop_repo: Repository on the test shop.
        shop: Test shop ids.
        order: Name of the order in ``shop.orders``.
        stuck: Expected ``is_stuck``.
        lost: Expected ``is_possibly_lost``.
        days: Expected ``days_in_status``.
    """
    result = service.check_return_eligibility(
        shop_repo, shop.customer_id, getattr(shop.orders, order)
    )
    assert result.reason_code is ReasonCode.NOT_DELIVERED
    assert (result.is_stuck, result.is_possibly_lost, result.days_in_status) == (stuck, lost, days)


def test_single_item_of_multi_item_order(shop_repo: SqliteOrderRepository, shop: ShopData) -> None:
    """Checking just one item of a two-item order returns that item only.

    Args:
        shop_repo: Repository on the test shop.
        shop: Test shop ids.
    """
    result = service.check_return_eligibility(
        shop_repo, shop.customer_id, shop.orders.complete, [1051]
    )
    assert result.reason_code is ReasonCode.ELIGIBLE
    assert result.items is not None
    (item,) = result.items
    assert (item.order_item_id, item.product_name, item.refund_amount) == (
        1051,
        "Blue Denim Shirt",
        40.0,
    )
    assert result.refund_total == 40.0


def test_all_items_checked_when_none_given(
    shop_repo: SqliteOrderRepository, shop: ShopData
) -> None:
    """Without ``order_item_ids`` every item is checked, and the refunds are summed.

    Args:
        shop_repo: Repository on the test shop.
        shop: Test shop ids.
    """
    result = service.check_return_eligibility(shop_repo, shop.customer_id, shop.orders.complete)
    assert result.items is not None
    assert [i.order_item_id for i in result.items] == [1051, shop.unnamed_item_id]
    assert result.refund_total == 59.99


@pytest.mark.parametrize(
    "check",
    [service.check_cancellation_eligibility, service.check_return_eligibility],
    ids=["cancel", "return"],
)
def test_other_users_order_is_not_found(
    shop_repo: SqliteOrderRepository, shop: ShopData, check: Check
) -> None:
    """Rule 4: another customer's order gives the same answer as a missing one.

    Args:
        shop_repo: Repository on the test shop.
        shop: Test shop ids.
        check: The eligibility service under test.
    """
    theirs = check(shop_repo, shop.customer_id, shop.orders.other_users)
    missing = check(shop_repo, shop.customer_id, 999_999)
    assert theirs == missing
    assert json.loads(missing.to_content()) == {"reason_code": "NOT_FOUND"}


def test_unknown_item_id_fails_whole_call(shop_repo: SqliteOrderRepository, shop: ShopData) -> None:
    """An item that isn't in the order makes the whole call NOT_FOUND; nothing is checked.

    Args:
        shop_repo: Repository on the test shop.
        shop: Test shop ids.
    """
    other_orders_item = 1011  # belongs to the customer's `shipped` order
    result = service.check_return_eligibility(
        shop_repo, shop.customer_id, shop.orders.complete, [1051, other_orders_item]
    )
    assert result.reason_code is ReasonCode.NOT_FOUND
    assert result.items is None
    assert (result.order_id, result.status, result.is_stuck) == (None, None, None)
    assert result.detail is not None and str(other_orders_item) in result.detail


def test_duplicate_item_ids_are_checked_once(
    shop_repo: SqliteOrderRepository, shop: ShopData
) -> None:
    """Repeating an item id doesn't double its refund.

    Args:
        shop_repo: Repository on the test shop.
        shop: Test shop ids.
    """
    result = service.check_return_eligibility(
        shop_repo, shop.customer_id, shop.orders.complete, [1051, 1051]
    )
    assert result.items is not None and len(result.items) == 1
    assert result.refund_total == 40.0


def test_return_request_row_counts_as_returned(
    shop_repo: SqliteOrderRepository, shop: ShopData
) -> None:
    """An item with a row in ``return_requests`` is already returned.

    Args:
        shop_repo: Repository on the test shop.
        shop: Test shop ids.
    """
    with closing(sqlite3.connect(shop.path)) as con:
        con.execute(
            "INSERT INTO return_requests (order_id, order_item_id, refund_amount, created_at)"
            " VALUES (?, ?, ?, ?)",
            (shop.orders.delivered_recently, 1081, 45.0, "2026-11-11 10:00:00"),
        )
        con.commit()
    result = service.check_return_eligibility(
        shop_repo, shop.customer_id, shop.orders.delivered_recently
    )
    assert (result.reason_code, result.eligible) == (ReasonCode.ALREADY_RETURNED, False)


def test_unnamed_item_keeps_null_name(shop_repo: SqliteOrderRepository, shop: ShopData) -> None:
    """An item without a product name is shown with an explicit null name.

    Args:
        shop_repo: Repository on the test shop.
        shop: Test shop ids.
    """
    result = service.check_return_eligibility(
        shop_repo, shop.customer_id, shop.orders.complete, [shop.unnamed_item_id]
    )
    (item,) = json.loads(result.to_content())["items"]
    assert item["product_name"] is None
    assert item["category"] == "Jeans"
