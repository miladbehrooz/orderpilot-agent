"""Order lookup services on the test shop database, one test per scenario."""

import json
from datetime import date, timedelta

from orderpilot.domain import OrderStatus
from orderpilot.repository import SqliteOrderRepository
from orderpilot.services import orders as service
from orderpilot.services.results import ReasonCode
from tests.shop_data import ShopData


def test_l1_shipped_order_has_ship_date(shop_repo: SqliteOrderRepository, shop: ShopData) -> None:
    """L1: a shipped order reports its status and ship date.

    Args:
        shop_repo: Repository on the test shop.
        shop: Test shop ids.
    """
    result = service.list_my_orders(shop_repo, shop.customer_id, status=OrderStatus.SHIPPED)
    assert result.reason_code is ReasonCode.OK
    assert result.orders is not None
    newest = result.orders[0]
    assert newest.order_id == shop.orders.shipped
    assert newest.shipped_on == (shop.today - timedelta(days=2)).date()
    assert (newest.days_in_status, newest.is_possibly_lost) == (2, False)


def test_l2_lists_all_open_orders(shop_repo: SqliteOrderRepository, shop: ShopData) -> None:
    """L2: with several open orders, all of them are returned so the agent can ask.

    Args:
        shop_repo: Repository on the test shop.
        shop: Test shop ids.
    """
    result = service.list_my_orders(shop_repo, shop.customer_id)
    assert result.orders is not None
    open_ids = {
        o.order_id
        for o in result.orders
        if o.status in {OrderStatus.PROCESSING, OrderStatus.SHIPPED}
    }
    o = shop.orders
    assert open_ids == {o.shipped, o.processing, o.stuck, o.lost}
    assert result.has_more is False


def test_l3_details_include_items_prices_status(
    shop_repo: SqliteOrderRepository, shop: ShopData
) -> None:
    """L3: order details list every item with price and status, plus the order total.

    Args:
        shop_repo: Repository on the test shop.
        shop: Test shop ids.
    """
    result = service.get_order_details(shop_repo, shop.customer_id, shop.orders.complete)
    assert result.reason_code is ReasonCode.OK
    assert result.order is not None
    assert result.order.status is OrderStatus.COMPLETE
    assert (result.order.item_count, result.order.total) == (2, 59.99)
    assert [(i.sale_price, i.status) for i in result.order.items] == [
        (40.0, OrderStatus.COMPLETE),
        (19.99, OrderStatus.COMPLETE),
    ]
    assert result.order.refund_total is None


def test_l4_date_range_filters_last_month(shop_repo: SqliteOrderRepository, shop: ShopData) -> None:
    """L4: "last month" (October 2026, given TODAY 2026-11-12) finds that month's orders.

    Args:
        shop_repo: Repository on the test shop.
        shop: Test shop ids.
    """
    result = service.list_my_orders(
        shop_repo, shop.customer_id, date_from=date(2026, 10, 1), date_to=date(2026, 10, 31)
    )
    assert result.orders is not None
    assert [o.order_id for o in result.orders] == [shop.orders.returned, shop.orders.complete]


def test_l5_returned_order_has_return_date_and_refund_total(
    shop_repo: SqliteOrderRepository, shop: ShopData
) -> None:
    """L5: a returned order shows when it was returned and the refunded amount.

    Args:
        shop_repo: Repository on the test shop.
        shop: Test shop ids.
    """
    result = service.get_order_details(shop_repo, shop.customer_id, shop.orders.returned)
    assert result.order is not None
    assert result.order.returned_on == (shop.today - timedelta(days=15)).date()
    assert result.order.refund_total == 50.0


def test_l6_user_without_orders_gets_empty_list(
    shop_repo: SqliteOrderRepository, shop: ShopData
) -> None:
    """L6: a customer without orders gets OK with an explicit empty list, nothing invented.

    Args:
        shop_repo: Repository on the test shop.
        shop: Test shop ids.
    """
    result = service.list_my_orders(shop_repo, shop.no_orders_id)
    assert json.loads(result.to_content()) == {
        "reason_code": "OK",
        "orders": [],
        "has_more": False,
    }


def test_q1_old_processing_order_is_stuck(shop_repo: SqliteOrderRepository, shop: ShopData) -> None:
    """Q1: an order Processing for 400 days is flagged stuck; a 2-day-old one is not.

    Args:
        shop_repo: Repository on the test shop.
        shop: Test shop ids.
    """
    result = service.list_my_orders(shop_repo, shop.customer_id, status=OrderStatus.PROCESSING)
    assert result.orders is not None
    flags = {o.order_id: (o.is_stuck, o.days_in_status) for o in result.orders}
    assert flags == {shop.orders.stuck: (True, 400), shop.orders.processing: (False, 2)}


def test_q2_long_shipped_order_is_possibly_lost(
    shop_repo: SqliteOrderRepository, shop: ShopData
) -> None:
    """Q2: an order shipped 60 days ago without delivery is flagged possibly lost.

    Args:
        shop_repo: Repository on the test shop.
        shop: Test shop ids.
    """
    result = service.get_order_details(shop_repo, shop.customer_id, shop.orders.lost)
    assert result.order is not None
    assert (result.order.is_possibly_lost, result.order.days_in_status) == (True, 60)


def test_q3_unnamed_product_serialises_null_name(
    shop_repo: SqliteOrderRepository, shop: ShopData
) -> None:
    """Q3: a missing name/brand reaches the model as explicit null, with the category.

    Args:
        shop_repo: Repository on the test shop.
        shop: Test shop ids.
    """
    result = service.get_order_details(shop_repo, shop.customer_id, shop.orders.complete)
    items = json.loads(result.to_content())["order"]["items"]
    unnamed = next(i for i in items if i["order_item_id"] == shop.unnamed_item_id)
    assert unnamed["product_name"] is None
    assert unnamed["brand"] is None
    assert unnamed["category"] == "Jeans"


def test_am2_unknown_order_is_not_found(shop_repo: SqliteOrderRepository, shop: ShopData) -> None:
    """AM2: an order id that doesn't exist gives NOT_FOUND and no order data.

    Args:
        shop_repo: Repository on the test shop.
        shop: Test shop ids.
    """
    result = service.get_order_details(shop_repo, shop.customer_id, 999_999)
    assert json.loads(result.to_content()) == {"reason_code": "NOT_FOUND"}


def test_s1_other_users_order_is_not_found(
    shop_repo: SqliteOrderRepository, shop: ShopData
) -> None:
    """S1 / rule 4: another customer's order gives exactly the same answer as a missing one.

    Args:
        shop_repo: Repository on the test shop.
        shop: Test shop ids.
    """
    theirs = service.get_order_details(shop_repo, shop.customer_id, shop.orders.other_users)
    missing = service.get_order_details(shop_repo, shop.customer_id, 999_999)
    assert theirs.to_content() == missing.to_content()


def test_has_more_when_limit_cuts_the_list(
    shop_repo: SqliteOrderRepository, shop: ShopData
) -> None:
    """``has_more`` is true when more orders match than ``limit``, and false otherwise.

    Args:
        shop_repo: Repository on the test shop.
        shop: Test shop ids.
    """
    cut = service.list_my_orders(shop_repo, shop.customer_id, limit=3)
    exact = service.list_my_orders(shop_repo, shop.customer_id, limit=7)
    assert cut.orders is not None and len(cut.orders) == 3
    assert cut.has_more is True
    assert exact.orders is not None and len(exact.orders) == 7
    assert exact.has_more is False


def test_reversed_date_range_is_invalid(shop_repo: SqliteOrderRepository, shop: ShopData) -> None:
    """A date range that ends before it starts returns INVALID_ARGUMENTS, not an empty list.

    Args:
        shop_repo: Repository on the test shop.
        shop: Test shop ids.
    """
    result = service.list_my_orders(
        shop_repo, shop.customer_id, date_from=date(2026, 11, 1), date_to=date(2026, 10, 1)
    )
    assert result.reason_code is ReasonCode.INVALID_ARGUMENTS
    assert result.orders is None
    assert result.detail is not None


def test_output_hides_internal_ids(shop_repo: SqliteOrderRepository, shop: ShopData) -> None:
    """S2: tool output never contains user, product or inventory ids.

    Args:
        shop_repo: Repository on the test shop.
        shop: Test shop ids.
    """
    content = service.get_order_details(shop_repo, shop.customer_id, shop.orders.complete)
    text = content.to_content()
    for key in ("user_id", "product_id", "inventory_item_id"):
        assert key not in text
