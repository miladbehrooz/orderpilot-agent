"""SQLite implementation of :class:`~orderpilot.repository.base.OrderRepository`."""

import sqlite3
from collections.abc import Sequence
from contextlib import closing
from datetime import date, datetime
from pathlib import Path

from orderpilot.db import connect, parse_timestamp
from orderpilot.domain import Order, OrderItem, OrderStatus, User
from orderpilot.errors import RepositoryError


def _optional_timestamp(value: str | None) -> datetime | None:
    """Parse a nullable database timestamp.

    Args:
        value: Timestamp text, or ``None``.

    Returns:
        The UTC datetime, or ``None`` if the column is NULL.
    """
    return None if value is None else parse_timestamp(value)


def _item_from_row(row: sqlite3.Row) -> OrderItem:
    """Build an order item from a joined ``order_items``/``products`` row.

    Args:
        row: Row with the item and product columns selected in ``_load_items``.

    Returns:
        The order item.
    """
    return OrderItem(
        order_item_id=row["id"],
        product_name=row["name"],
        brand=row["brand"],
        category=row["category"],
        sale_price=row["sale_price"],
        status=OrderStatus(row["status"]),
        created_at=parse_timestamp(row["created_at"]),
        shipped_at=_optional_timestamp(row["shipped_at"]),
        delivered_at=_optional_timestamp(row["delivered_at"]),
        returned_at=_optional_timestamp(row["returned_at"]),
    )


def _order_from_row(row: sqlite3.Row, items: Sequence[OrderItem]) -> Order:
    """Build an order from an ``orders`` row and its items.

    Args:
        row: Row from the ``orders`` table.
        items: The order's items.

    Returns:
        The order.
    """
    return Order(
        order_id=row["order_id"],
        status=OrderStatus(row["status"]),
        created_at=parse_timestamp(row["created_at"]),
        shipped_at=_optional_timestamp(row["shipped_at"]),
        delivered_at=_optional_timestamp(row["delivered_at"]),
        returned_at=_optional_timestamp(row["returned_at"]),
        items=tuple(items),
    )


def _load_items(con: sqlite3.Connection, order_ids: Sequence[int]) -> dict[int, list[OrderItem]]:
    """Load the items of several orders in one query, with their product details.

    Args:
        con: Open connection.
        order_ids: Orders whose items to load.

    Returns:
        Each order's items, ordered by item id. Orders without items are absent.
    """
    if not order_ids:
        return {}
    # Only "?" placeholders are interpolated into the SQL; the ids are bound parameters.
    placeholders = ", ".join("?" * len(order_ids))
    # LEFT JOIN so an item whose product has no name/brand still comes back (Q3).
    rows = con.execute(
        f"""
        SELECT oi.order_id, oi.id, oi.sale_price, oi.status, oi.created_at, oi.shipped_at,
               oi.delivered_at, oi.returned_at, p.name, p.brand, p.category
        FROM order_items AS oi
        LEFT JOIN products AS p ON p.id = oi.product_id
        WHERE oi.order_id IN ({placeholders})
        ORDER BY oi.id
        """,
        tuple(order_ids),
    ).fetchall()
    items: dict[int, list[OrderItem]] = {}
    for row in rows:
        items.setdefault(row["order_id"], []).append(_item_from_row(row))
    return items


class SqliteOrderRepository:
    """``OrderRepository`` backed by the SQLite file built by ``scripts/build_db.py``.

    Each method opens its own short-lived connection (see :func:`orderpilot.db.connect`).

    Args:
        db_path: Path to the SQLite database.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path

    def reference_now(self) -> datetime:
        """Return TODAY, the fixed reference time of the dataset.

        Returns:
            ``meta.reference_now`` as a timezone-aware UTC datetime.

        Raises:
            RepositoryError: If the database has no ``reference_now`` entry.
        """
        with closing(connect(self._db_path)) as con:
            row = con.execute("SELECT value FROM meta WHERE key = 'reference_now'").fetchone()
        if row is None:
            raise RepositoryError("meta.reference_now is missing; rebuild the database")
        return parse_timestamp(row["value"])

    def find_user_by_email(self, email: str) -> User | None:
        """Look up an account by email, ignoring case.

        Args:
            email: Email address to look up.

        Returns:
            The matching user, or ``None`` if there is none.
        """
        # users.email is declared COLLATE NOCASE, so `=` is already case-insensitive.
        with closing(connect(self._db_path)) as con:
            row = con.execute(
                "SELECT id, first_name, last_name, email, postal_code FROM users WHERE email = ?",
                (email,),
            ).fetchone()
        if row is None:
            return None
        return User(
            user_id=row["id"],
            first_name=row["first_name"],
            last_name=row["last_name"],
            email=row["email"],
            postal_code=row["postal_code"],
        )

    def list_orders(
        self,
        user_id: int,
        *,
        status: OrderStatus | None,
        created_from: date | None,
        created_to: date | None,
        limit: int,
    ) -> list[Order]:
        """Return a user's orders, newest first, with their items.

        Args:
            user_id: The verified customer.
            status: Only orders with this status, or all statuses if ``None``.
            created_from: Only orders placed on or after this date (UTC), if given.
            created_to: Only orders placed on or before this date (UTC), if given.
            limit: Maximum number of orders to return.

        Returns:
            At most ``limit`` orders, newest first.
        """
        start = None if created_from is None else created_from.isoformat()
        end = None if created_to is None else created_to.isoformat()
        with closing(connect(self._db_path)) as con:
            rows = con.execute(
                """
                SELECT order_id, status, created_at, shipped_at, delivered_at, returned_at
                FROM orders
                WHERE user_id = ?
                  AND (? IS NULL OR status = ?)
                  AND (? IS NULL OR date(created_at) >= ?)
                  AND (? IS NULL OR date(created_at) <= ?)
                ORDER BY created_at DESC, order_id DESC
                LIMIT ?
                """,
                (user_id, status, status, start, start, end, end, limit),
            ).fetchall()
            items = _load_items(con, [row["order_id"] for row in rows])
        return [_order_from_row(row, items.get(row["order_id"], [])) for row in rows]

    def get_order(self, user_id: int, order_id: int) -> Order | None:
        """Return one of the user's orders with its items.

        Args:
            user_id: The verified customer.
            order_id: The order to look up.

        Returns:
            The order, or ``None`` if it doesn't exist or belongs to another user. The
            ownership check is part of the query, so both cases look the same (rule 4).
        """
        with closing(connect(self._db_path)) as con:
            row = con.execute(
                """
                SELECT order_id, status, created_at, shipped_at, delivered_at, returned_at
                FROM orders
                WHERE order_id = ? AND user_id = ?
                """,
                (order_id, user_id),
            ).fetchone()
            if row is None:
                return None
            items = _load_items(con, [order_id])
        return _order_from_row(row, items.get(order_id, []))


__all__ = ["SqliteOrderRepository"]
