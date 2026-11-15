"""A tiny, hand-made shop database for tool, service and repository tests.

Built from the real schema in ``scripts/build_db.py``, with every timestamp placed at a
known distance from TODAY, so each scenario's data condition holds exactly. Tests never
depend on ``data/`` or ``ecommerce.sqlite``.
"""

import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import build_db


@dataclass(frozen=True, slots=True)
class ShopOrders:
    """Order ids in the test shop, named by the data condition they cover.

    Attributes:
        shipped: Customer's order shipped 2 days ago (L1).
        processing: Customer's order placed 2 days ago, not stuck.
        stuck: Customer's order Processing for 400 days (Q1).
        lost: Customer's order shipped 60 days ago, never delivered (Q2).
        complete: Customer's order placed last month and delivered, with two items, one
            of them a product without name or brand (L3, L4, Q3).
        returned: Customer's order with both items returned (L5).
        cancelled: Customer's cancelled order.
        other_users: An order of a different customer (S1).
    """

    shipped: int
    processing: int
    stuck: int
    lost: int
    complete: int
    returned: int
    cancelled: int
    other_users: int


@dataclass(frozen=True, slots=True)
class ShopData:
    """Handles to the test shop.

    Attributes:
        path: The SQLite file.
        today: Its ``meta.reference_now``.
        customer_id: The customer with one order of every kind.
        no_orders_id: A customer without orders (L6).
        other_id: Another customer, owner of ``orders.other_users``.
        orders: The order ids.
        unnamed_item_id: The item whose product has no name or brand (Q3).
    """

    path: Path
    today: datetime
    customer_id: int
    no_orders_id: int
    other_id: int
    orders: ShopOrders
    unnamed_item_id: int


def _insert(con: sqlite3.Connection, table: str, **cols: object) -> None:
    """Insert one row.

    Args:
        con: Open connection.
        table: Table name (test code only, never user input).
        **cols: Column values.
    """
    names = ", ".join(cols)
    marks = ", ".join("?" * len(cols))
    con.execute(f"INSERT INTO {table} ({names}) VALUES ({marks})", tuple(cols.values()))


def build_shop_db(path: Path, today: datetime) -> ShopData:
    """Create the test shop database.

    Args:
        path: Where to write the SQLite file.
        today: TODAY of the dataset (``meta.reference_now``), at midnight UTC.

    Returns:
        The ids of the rows the tests refer to.
    """

    def ago(days: int) -> str:
        """Return the timestamp ``days`` calendar days before today, at 10:00 UTC.

        Args:
            days: Days before today.

        Returns:
            Timestamp text in the database format.
        """
        return (today - timedelta(days=days) + timedelta(hours=10)).strftime("%Y-%m-%d %H:%M:%S")

    def add_order(
        order_id: int,
        user_id: int,
        status: str,
        items: list[tuple[int, int, float]],
        **dates: str,
    ) -> None:
        """Insert an order and its items, which share its status and dates.

        Args:
            order_id: Order id.
            user_id: Owner.
            status: Order and item status.
            items: ``(order_item_id, product_id, sale_price)`` per item.
            **dates: ``created_at`` and any of ``shipped_at``/``delivered_at``/``returned_at``.
        """
        _insert(
            con,
            "orders",
            order_id=order_id,
            user_id=user_id,
            status=status,
            num_of_item=len(items),
            **dates,
        )
        for item_id, product_id, price in items:
            _insert(
                con,
                "order_items",
                id=item_id,
                order_id=order_id,
                user_id=user_id,
                product_id=product_id,
                status=status,
                sale_price=price,
                **dates,
            )

    customer, no_orders, other = 1, 2, 3
    shirt, unnamed, socks = 1, 2, 3
    with closing(sqlite3.connect(path)) as con:
        con.executescript(build_db.default_config().schema)
        _insert(con, "meta", key="reference_now", value=today.strftime("%Y-%m-%d %H:%M:%S"))
        _insert(con, "distribution_centers", id=1, name="Test DC")
        for product_id, name, brand, category in [
            (shirt, "Blue Denim Shirt", "Acme", "Tops & Tees"),
            (unnamed, None, None, "Jeans"),
            (socks, "Wool Socks", "Knitco", "Socks"),
        ]:
            _insert(
                con,
                "products",
                id=product_id,
                name=name,
                brand=brand,
                category=category,
                department="Men",
                cost=1.0,
                retail_price=50.0,
                sku=f"SKU-{product_id}",
                distribution_center_id=1,
            )
        for user_id, first, last in [
            (customer, "Ann", "Lee"),
            (no_orders, "Bo", "Kim"),
            (other, "Cy", "Ray"),
        ]:
            _insert(
                con,
                "users",
                id=user_id,
                first_name=first,
                last_name=last,
                email=f"{first.lower()}.{last.lower()}@example.com",
                postal_code="10115",
                created_at=ago(900),
            )

        add_order(
            101, customer, "Shipped", [(1011, shirt, 25.0)], created_at=ago(4), shipped_at=ago(2)
        )
        add_order(102, customer, "Processing", [(1021, socks, 9.5)], created_at=ago(2))
        add_order(103, customer, "Processing", [(1031, socks, 12.0)], created_at=ago(400))
        add_order(
            104, customer, "Shipped", [(1041, shirt, 30.0)], created_at=ago(62), shipped_at=ago(60)
        )
        add_order(
            105,
            customer,
            "Complete",
            [(1051, shirt, 40.0), (1052, unnamed, 19.99)],
            created_at=ago(28),
            shipped_at=ago(27),
            delivered_at=ago(23),
        )
        add_order(
            106,
            customer,
            "Returned",
            [(1061, socks, 15.0), (1062, shirt, 35.0)],
            created_at=ago(20),
            shipped_at=ago(19),
            delivered_at=ago(17),
            returned_at=ago(15),
        )
        add_order(107, customer, "Cancelled", [(1071, socks, 5.0)], created_at=ago(10))
        add_order(301, other, "Processing", [(3011, socks, 8.0)], created_at=ago(3))
        con.commit()

    return ShopData(
        path=path,
        today=today,
        customer_id=customer,
        no_orders_id=no_orders,
        other_id=other,
        orders=ShopOrders(
            shipped=101,
            processing=102,
            stuck=103,
            lost=104,
            complete=105,
            returned=106,
            cancelled=107,
            other_users=301,
        ),
        unnamed_item_id=1052,
    )


__all__ = ["ShopData", "ShopOrders", "build_shop_db"]
