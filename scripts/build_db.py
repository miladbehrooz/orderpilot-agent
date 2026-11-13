"""Build the e-commerce agent's SQLite database from the theLook CSV files.

Steps
  1. Load the CSVs (timestamps read as text, parsed as UTC).
  2. Clean:
     - order_items.created_at := orders.created_at (the source value is 0-4 days off)
     - prices rounded to 2 decimals
     - country variants normalised (Deutschland -> Germany, España -> Spain)
     - empty product name/brand kept as NULL
     - duplicate emails made unique with realistic name-based patterns
  3. Shift every timestamp by one constant offset (whole weeks) so the latest
     event in the data lands just before the reference "now".
  4. Write SQLite with primary/foreign keys, indexes, empty Milestone-2 tables,
     and a `meta` table holding the fixed reference time (TODAY) for tools and evals.
  5. Validate the result.

Usage
  python build_db.py --data-dir data --out ecommerce.sqlite [--today 2026-11-12] [--seed 42]
"""

from __future__ import annotations

import argparse
import random
import re
import sqlite3
import unicodedata
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

TS_FMT = "%Y-%m-%d %H:%M:%S"
WEEK = pd.Timedelta(weeks=1)

TIMESTAMP_COLS = {
    "users": ["created_at"],
    "orders": ["created_at", "shipped_at", "delivered_at", "returned_at"],
    "order_items": ["created_at", "shipped_at", "delivered_at", "returned_at"],
}
COUNTRY_FIXES = {"Deutschland": "Germany", "España": "Spain"}


# --------------------------------------------------------------------------- load
def load(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Load the theLook CSV files into DataFrames.

    Timestamp columns are read as text and parsed as UTC, truncated to whole seconds.

    Args:
        data_dir: Directory containing the source CSV files.

    Returns:
        A mapping from table name to its DataFrame.
    """
    dfs = {}
    for name in ["users", "orders", "order_items", "products", "distribution_centers"]:
        ts_cols = TIMESTAMP_COLS.get(name, [])
        df = pd.read_csv(
            data_dir / f"{name}.csv",
            dtype=dict.fromkeys(ts_cols, "string")
            | ({"postal_code": "string"} if name == "users" else {}),
            keep_default_na=True,
        )
        for c in ts_cols:
            df[c] = pd.to_datetime(df[c], utc=True, format="ISO8601").dt.floor("s")
        dfs[name] = df
    return dfs


# --------------------------------------------------------------------------- clean
def _slug(s: str) -> str:
    """Reduce a name to lowercase ASCII letters for use in an email local part.

    Args:
        s: The name to convert. Accents are stripped; other non-letters are dropped.

    Returns:
        The slug, e.g. ``"José-Luis"`` becomes ``"joseluis"``.
    """
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z]", "", s)


def _email_candidates(first: str, last: str, birth_year: int, rng: random.Random) -> Iterator[str]:
    """Generate realistic email local parts for a person, in random order.

    Args:
        first: First name.
        last: Last name.
        birth_year: Year of birth, used for the ``yy``/``yyyy`` suffix patterns.
        rng: Seeded random generator, so the output is reproducible.

    Yields:
        Ten name-based patterns in shuffled order, then an endless fallback of
        ``first.last<number>``.
    """
    f, l = _slug(first), _slug(last)  # noqa: E741
    yy, yyyy = f"{birth_year % 100:02d}", str(birth_year)
    patterns = [
        f"{f}.{l}",
        f"{f}_{l}",
        f"{f[0]}{l}",
        f"{f[0]}.{l}",
        f"{l}.{f}",
        f"{f}{l}{yy}",
        f"{f}.{l}{yy}",
        f"{f}{l}{yyyy}",
        f"{f[0]}{l}{yy}",
        f"{f}.{l[0]}{yy}",
    ]
    rng.shuffle(patterns)
    yield from patterns
    while True:  # fallback, not needed on the current data
        yield f"{f}.{l}{rng.randint(10, 999)}"


def make_emails_unique(users: pd.DataFrame, ref_year: int, seed: int) -> pd.Series:
    """Make user emails unique by replacing later duplicates with name-based variants.

    The earliest sign-up keeps the original email. Later duplicates get a variant on
    the same reserved domain (example.com/.net/.org).

    Args:
        users: Users table with ``id``, ``email``, ``first_name``, ``last_name``,
            ``age`` and ``created_at`` columns.
        ref_year: Reference year used to derive each user's birth year from ``age``.
        seed: Seed for the random generator, so the result is reproducible.

    Returns:
        The new emails, aligned with the rows of ``users``.
    """
    rng = random.Random(seed)
    used: set[str] = set()
    new: dict[int, str] = {}
    ordered = users.sort_values(["created_at", "id"])
    rows: Iterator[tuple[int, str, str, str, int]] = zip(
        ordered["id"].tolist(),
        ordered["email"].tolist(),
        ordered["first_name"].tolist(),
        ordered["last_name"].tolist(),
        ordered["age"].tolist(),
        strict=True,
    )
    for user_id, raw_email, first_name, last_name, age in rows:
        email = raw_email.strip().lower()
        if email not in used:
            used.add(email)
            new[user_id] = email
            continue
        domain = email.split("@", 1)[1]
        for local in _email_candidates(first_name, last_name, ref_year - int(age), rng):
            cand = f"{local}@{domain}"
            if cand not in used:
                used.add(cand)
                new[user_id] = cand
                break
    return users["id"].map(new)


def clean(dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Clean the loaded tables in place.

    Sets each item's ``created_at`` to its order's, rounds prices to 2 decimals,
    normalises country names, strips text fields and turns empty product names and
    brands into NULL.

    Args:
        dfs: Tables returned by :func:`load`.

    Returns:
        The same mapping, with the DataFrames modified in place.
    """
    users, orders, items, products = (
        dfs[k] for k in ["users", "orders", "order_items", "products"]
    )

    # item created_at follows its order
    items["created_at"] = items["order_id"].map(orders.set_index("order_id")["created_at"])

    # prices
    items["sale_price"] = items["sale_price"].round(2)
    products["retail_price"] = products["retail_price"].round(2)
    products["cost"] = products["cost"].round(2)

    # text fields
    users["country"] = users["country"].replace(COUNTRY_FIXES)
    for c in ["first_name", "last_name", "email", "postal_code", "city", "state", "street_address"]:
        users[c] = users[c].astype("string").str.strip()
    for c in ["name", "brand"]:
        products[c] = products[c].astype("string").str.strip().replace({"": pd.NA})
    return dfs


# --------------------------------------------------------------------------- date shift
def shift_dates(
    dfs: dict[str, pd.DataFrame], reference_now: pd.Timestamp
) -> tuple[pd.Timedelta, pd.Timestamp]:
    """Shift all timestamps by whole weeks so the latest event lands just before TODAY.

    Shifting by whole weeks keeps every date on the same weekday.

    Args:
        dfs: Cleaned tables; timestamp columns are modified in place.
        reference_now: The fixed reference time (TODAY).

    Returns:
        A tuple of the applied offset and the latest event time before the shift.
    """
    anchor = max(dfs[t][c].max() for t, cols in TIMESTAMP_COLS.items() for c in cols)
    offset = ((reference_now - anchor) // WEEK) * WEEK  # whole weeks, keeps weekdays
    for t, cols in TIMESTAMP_COLS.items():
        for c in cols:
            dfs[t][c] = dfs[t][c] + offset
    return offset, anchor


# --------------------------------------------------------------------------- write
SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE distribution_centers (
    id        INTEGER PRIMARY KEY,
    name      TEXT NOT NULL,
    latitude  REAL,
    longitude REAL
);

CREATE TABLE products (
    id                     INTEGER PRIMARY KEY,
    name                   TEXT,
    brand                  TEXT,
    category               TEXT NOT NULL,
    department             TEXT NOT NULL,
    cost                   REAL NOT NULL,
    retail_price           REAL NOT NULL,
    sku                    TEXT NOT NULL UNIQUE,
    distribution_center_id INTEGER NOT NULL REFERENCES distribution_centers(id)
);

CREATE TABLE users (
    id             INTEGER PRIMARY KEY,
    first_name     TEXT NOT NULL,
    last_name      TEXT NOT NULL,
    email          TEXT NOT NULL UNIQUE COLLATE NOCASE,
    age            INTEGER,
    gender         TEXT,
    street_address TEXT,
    postal_code    TEXT NOT NULL,
    city           TEXT,
    state          TEXT,
    country        TEXT,
    latitude       REAL,
    longitude      REAL,
    traffic_source TEXT,
    created_at     TEXT NOT NULL
);

CREATE TABLE orders (
    order_id     INTEGER PRIMARY KEY,
    user_id      INTEGER NOT NULL REFERENCES users(id),
    status       TEXT NOT NULL
                 CHECK (status IN ('Processing','Shipped','Complete','Returned','Cancelled')),
    gender       TEXT,
    num_of_item  INTEGER NOT NULL,
    created_at   TEXT NOT NULL,
    shipped_at   TEXT,
    delivered_at TEXT,
    returned_at  TEXT
);

CREATE TABLE order_items (
    id                INTEGER PRIMARY KEY,
    order_id          INTEGER NOT NULL REFERENCES orders(order_id),
    user_id           INTEGER NOT NULL REFERENCES users(id),
    product_id        INTEGER NOT NULL REFERENCES products(id),
    inventory_item_id INTEGER,  -- inventory_items table not loaded
    status            TEXT NOT NULL
                      CHECK (status IN ('Processing','Shipped','Complete','Returned','Cancelled')),
    sale_price        REAL NOT NULL,
    created_at        TEXT NOT NULL,
    shipped_at        TEXT,
    delivered_at      TEXT,
    returned_at       TEXT
);

-- Milestone 2 (writes): empty for now
CREATE TABLE return_requests (
    return_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id      INTEGER NOT NULL REFERENCES orders(order_id),
    order_item_id INTEGER NOT NULL UNIQUE REFERENCES order_items(id),
    refund_amount REAL NOT NULL,
    reason        TEXT,
    status        TEXT NOT NULL DEFAULT 'requested',
    created_at    TEXT NOT NULL
);

CREATE TABLE support_tickets (
    ticket_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL REFERENCES users(id),
    order_id    INTEGER REFERENCES orders(order_id),
    category    TEXT NOT NULL CHECK (category IN ('stuck_order','lost_package','other')),
    description TEXT,
    status      TEXT NOT NULL DEFAULT 'open',
    created_at  TEXT NOT NULL
);
CREATE UNIQUE INDEX one_open_ticket_per_order ON support_tickets(order_id)
    WHERE status = 'open' AND order_id IS NOT NULL;

CREATE INDEX idx_orders_user        ON orders(user_id, created_at);
CREATE INDEX idx_items_order        ON order_items(order_id);
CREATE INDEX idx_items_user         ON order_items(user_id);
CREATE INDEX idx_items_product      ON order_items(product_id);
CREATE INDEX idx_users_email_postal ON users(email, postal_code);
"""

COLUMNS = {
    "distribution_centers": ["id", "name", "latitude", "longitude"],
    "products": [
        "id",
        "name",
        "brand",
        "category",
        "department",
        "cost",
        "retail_price",
        "sku",
        "distribution_center_id",
    ],
    "users": [
        "id",
        "first_name",
        "last_name",
        "email",
        "age",
        "gender",
        "street_address",
        "postal_code",
        "city",
        "state",
        "country",
        "latitude",
        "longitude",
        "traffic_source",
        "created_at",
    ],
    "orders": [
        "order_id",
        "user_id",
        "status",
        "gender",
        "num_of_item",
        "created_at",
        "shipped_at",
        "delivered_at",
        "returned_at",
    ],
    "order_items": [
        "id",
        "order_id",
        "user_id",
        "product_id",
        "inventory_item_id",
        "status",
        "sale_price",
        "created_at",
        "shipped_at",
        "delivered_at",
        "returned_at",
    ],
}


def _to_sql_values(df: pd.DataFrame, table: str) -> list[tuple[object, ...]]:
    """Convert a table to rows ready for ``executemany``.

    Args:
        df: The table to convert.
        table: Table name, used to pick the column order and the timestamp columns.

    Returns:
        One tuple per row, with timestamps as text and missing values as ``None``.
    """
    df = df[COLUMNS[table]].copy()
    for c in TIMESTAMP_COLS.get(table, []):
        df[c] = df[c].dt.strftime(TS_FMT)
    df = df.astype(object).where(df.notna(), None)
    return list(df.itertuples(index=False, name=None))


def write(dfs: dict[str, pd.DataFrame], out: Path, meta: dict[str, str]) -> None:
    """Write the tables and metadata to a new SQLite database.

    Any existing file at ``out`` is replaced.

    Args:
        dfs: Cleaned and shifted tables.
        out: Path of the SQLite file to create.
        meta: Key/value pairs for the ``meta`` table, including ``reference_now``.
    """
    out.unlink(missing_ok=True)
    con = sqlite3.connect(out)
    con.executescript(SCHEMA)
    for table in ["distribution_centers", "products", "users", "orders", "order_items"]:
        cols = COLUMNS[table]
        con.executemany(
            f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
            _to_sql_values(dfs[table], table),
        )
    con.executemany("INSERT INTO meta VALUES (?, ?)", meta.items())
    con.commit()
    con.close()


# --------------------------------------------------------------------------- validate
CHECKS = {
    "foreign keys": "SELECT COUNT(*) FROM pragma_foreign_key_check",
    "duplicate emails": "SELECT COUNT(*) - COUNT(DISTINCT lower(email)) FROM users",
    "order chronology": """SELECT COUNT(*) FROM orders WHERE shipped_at < created_at
                           OR delivered_at < shipped_at OR returned_at < delivered_at""",
    "item chronology": """SELECT COUNT(*) FROM order_items WHERE shipped_at < created_at
                          OR delivered_at < shipped_at OR returned_at < delivered_at""",
    "events after TODAY": """SELECT COUNT(*) FROM orders, meta m WHERE m.key='reference_now'
                             AND max(created_at, ifnull(shipped_at,''), ifnull(delivered_at,''),
                                     ifnull(returned_at,'')) > m.value""",
    "status vs timestamps": """SELECT COUNT(*) FROM orders WHERE
            (status IN ('Processing','Cancelled')
                AND (shipped_at IS NOT NULL OR delivered_at IS NOT NULL))
         OR (status = 'Shipped'  AND (shipped_at IS NULL OR delivered_at IS NOT NULL))
         OR (status = 'Complete' AND (delivered_at IS NULL OR returned_at IS NOT NULL))
         OR (status = 'Returned' AND returned_at IS NULL)""",
    "item count mismatch": """SELECT COUNT(*) FROM orders o
                              JOIN (SELECT order_id, COUNT(*) n FROM order_items
                                    GROUP BY order_id) i
                              USING (order_id) WHERE o.num_of_item <> i.n""",
    "item/order status mismatch": """SELECT COUNT(*) FROM order_items i
                                     JOIN orders o USING (order_id) WHERE i.status <> o.status""",
}


def validate(out: Path) -> None:
    """Run integrity checks on the built database and print a report.

    Args:
        out: Path of the SQLite file to check.

    Raises:
        SystemExit: If any check finds violations.
    """
    con = sqlite3.connect(out)
    failed = False
    print("\nValidation")
    for name, sql in CHECKS.items():
        n = con.execute(sql).fetchone()[0]
        failed |= n != 0
        print(f"  {'OK  ' if n == 0 else 'FAIL'} {name}: {n}")
    for t in ["users", "products", "orders", "order_items", "distribution_centers"]:
        print(f"  rows {t}: {con.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]:,}")
    con.close()
    if failed:
        raise SystemExit("validation failed")


# --------------------------------------------------------------------------- main
def main() -> None:
    """Parse command-line arguments and build the database."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--out", type=Path, default=Path("ecommerce.sqlite"))
    p.add_argument("--today", help="reference date YYYY-MM-DD (default: today, UTC)")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    today = a.today or datetime.now(UTC).strftime("%Y-%m-%d")
    reference_now = pd.Timestamp(f"{today} 00:00:00", tz="UTC")

    dfs = clean(load(a.data_dir))
    offset, anchor = shift_dates(dfs, reference_now)
    users = dfs["users"]
    original = users["email"].str.lower()
    users["email"] = make_emails_unique(users, ref_year=reference_now.year, seed=a.seed)

    meta = {
        "reference_now": reference_now.strftime(TS_FMT),
        "date_shift_days": str(offset.days),
        "source_latest_event": anchor.strftime(TS_FMT),
        "email_seed": str(a.seed),
        "built_at": datetime.now(UTC).strftime(TS_FMT),
    }
    write(dfs, a.out, meta)

    print(f"Wrote {a.out}")
    print(f"  reference_now (TODAY): {meta['reference_now']}")
    print(
        f"  shifted by {offset.days} days ({offset.days // 7} weeks); "
        f"source latest event {meta['source_latest_event']}"
    )
    print(f"  emails changed: {(users['email'] != original).sum():,}")
    validate(a.out)


if __name__ == "__main__":
    main()
