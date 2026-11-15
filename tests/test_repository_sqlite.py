"""SqliteOrderRepository: account lookups on the real database, order queries on the test shop.

Tests on the real database are skipped if it isn't built.
"""

import sqlite3
from contextlib import closing
from datetime import UTC, date, timedelta
from pathlib import Path

import pytest

from orderpilot.domain import OrderStatus, User
from orderpilot.errors import RepositoryError
from orderpilot.repository import SqliteOrderRepository
from orderpilot.repository.db import parse_timestamp
from orderpilot.settings import settings
from tests.shop_data import ShopData


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


def _list(
    repo: SqliteOrderRepository,
    user_id: int,
    *,
    status: OrderStatus | None = None,
    created_from: date | None = None,
    created_to: date | None = None,
    limit: int = 50,
) -> list[int]:
    """Return the ids of ``list_orders`` results, with no filters unless given.

    Args:
        repo: Repository under test.
        user_id: Customer whose orders to list.
        status: Status filter.
        created_from: Earliest order date.
        created_to: Latest order date.
        limit: Maximum number of orders.

    Returns:
        Order ids in the returned order.
    """
    orders = repo.list_orders(
        user_id, status=status, created_from=created_from, created_to=created_to, limit=limit
    )
    return [o.order_id for o in orders]


def test_list_orders_newest_first(shop_repo: SqliteOrderRepository, shop: ShopData) -> None:
    """Orders come back newest first, and only the customer's own.

    Args:
        shop_repo: Repository on the test shop.
        shop: Test shop ids.
    """
    o = shop.orders
    assert _list(shop_repo, shop.customer_id) == [
        o.processing,
        o.shipped,
        o.cancelled,
        o.returned,
        o.complete,
        o.lost,
        o.stuck,
    ]


def test_list_orders_filters_by_status(shop_repo: SqliteOrderRepository, shop: ShopData) -> None:
    """The status filter keeps only orders with that status.

    Args:
        shop_repo: Repository on the test shop.
        shop: Test shop ids.
    """
    shipped = _list(shop_repo, shop.customer_id, status=OrderStatus.SHIPPED)
    assert shipped == [shop.orders.shipped, shop.orders.lost]


def test_list_orders_date_bounds_are_inclusive(
    shop_repo: SqliteOrderRepository, shop: ShopData
) -> None:
    """An order placed on the boundary day is included on both ends.

    Args:
        shop_repo: Repository on the test shop.
        shop: Test shop ids.
    """
    day = (shop.today - timedelta(days=28)).date()  # the day order `complete` was placed
    found = _list(shop_repo, shop.customer_id, created_from=day, created_to=day)
    assert found == [shop.orders.complete]


def test_list_orders_respects_limit(shop_repo: SqliteOrderRepository, shop: ShopData) -> None:
    """At most ``limit`` orders are returned, the newest ones.

    Args:
        shop_repo: Repository on the test shop.
        shop: Test shop ids.
    """
    assert _list(shop_repo, shop.customer_id, limit=2) == [
        shop.orders.processing,
        shop.orders.shipped,
    ]


def test_l6_list_orders_of_user_without_orders_is_empty(
    shop_repo: SqliteOrderRepository, shop: ShopData
) -> None:
    """L6: a customer without orders gets an empty list.

    Args:
        shop_repo: Repository on the test shop.
        shop: Test shop ids.
    """
    assert _list(shop_repo, shop.no_orders_id) == []


def test_get_order_loads_its_items(shop_repo: SqliteOrderRepository, shop: ShopData) -> None:
    """An order comes with exactly its own items and their product details.

    Args:
        shop_repo: Repository on the test shop.
        shop: Test shop ids.
    """
    order = shop_repo.get_order(shop.customer_id, shop.orders.complete)
    assert order is not None
    assert order.status is OrderStatus.COMPLETE
    assert [i.order_item_id for i in order.items] == [1051, shop.unnamed_item_id]
    assert (order.items[0].product_name, order.items[0].brand) == ("Blue Denim Shirt", "Acme")
    assert order.delivered_at is not None and order.delivered_at.tzinfo is UTC


def test_q3_missing_product_name_and_brand_are_none(
    shop_repo: SqliteOrderRepository, shop: ShopData
) -> None:
    """Q3: a product without name or brand comes back as None, with its category.

    Args:
        shop_repo: Repository on the test shop.
        shop: Test shop ids.
    """
    order = shop_repo.get_order(shop.customer_id, shop.orders.complete)
    assert order is not None
    unnamed = next(i for i in order.items if i.order_item_id == shop.unnamed_item_id)
    assert (unnamed.product_name, unnamed.brand, unnamed.category) == (None, None, "Jeans")


def test_s1_get_order_of_another_user_returns_none(
    shop_repo: SqliteOrderRepository, shop: ShopData
) -> None:
    """S1 / rule 4: another customer's order is invisible, exactly like a missing one.

    Args:
        shop_repo: Repository on the test shop.
        shop: Test shop ids.
    """
    assert shop_repo.get_order(shop.customer_id, shop.orders.other_users) is None
    assert shop_repo.get_order(shop.other_id, shop.orders.other_users) is not None


def test_am2_get_unknown_order_returns_none(
    shop_repo: SqliteOrderRepository, shop: ShopData
) -> None:
    """AM2: an order id that doesn't exist returns None.

    Args:
        shop_repo: Repository on the test shop.
        shop: Test shop ids.
    """
    assert shop_repo.get_order(shop.customer_id, 999_999) is None
