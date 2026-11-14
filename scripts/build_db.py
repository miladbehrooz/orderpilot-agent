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
import sys
import unicodedata
from collections.abc import Iterator, Sequence
from contextlib import closing
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd

from orderpilot.errors import DatabaseBuildError
from orderpilot.logging import logger

TS_FMT = "%Y-%m-%d %H:%M:%S"
WEEK = pd.Timedelta(weeks=1)

TIMESTAMP_COLS = {
    "users": ["created_at"],
    "orders": ["created_at", "shipped_at", "delivered_at", "returned_at"],
    "order_items": ["created_at", "shipped_at", "delivered_at", "returned_at"],
}
COUNTRY_FIXES = {"Deutschland": "Germany", "España": "Spain"}
TABLES = ["users", "orders", "order_items", "products", "distribution_centers"]


# --------------------------------------------------------------------------- inputs
def check_inputs(data_dir: Path) -> None:
    """Check that the data directory and every source CSV exist.

    Args:
        data_dir: Directory expected to contain the source CSV files.

    Raises:
        DatabaseBuildError: If the directory or any CSV is missing; all missing files are listed.
    """
    if not data_dir.is_dir():
        raise DatabaseBuildError(f"Data directory not found: {data_dir}")
    missing = [f"{t}.csv" for t in TABLES if not (data_dir / f"{t}.csv").is_file()]
    if missing:
        raise DatabaseBuildError(f"Missing CSV files in {data_dir}: {', '.join(missing)}")


# --------------------------------------------------------------------------- load
def load(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Load the theLook CSV files into DataFrames.

    Timestamp columns are read as text and parsed as UTC, truncated to whole seconds.

    Args:
        data_dir: Directory containing the source CSV files.

    Returns:
        A mapping from table name to its DataFrame.

    Raises:
        DatabaseBuildError: If a file cannot be parsed, lacks a required column, or holds a
            timestamp that cannot be parsed.
    """
    dfs = {}
    for name in TABLES:
        path = data_dir / f"{name}.csv"
        ts_cols = TIMESTAMP_COLS.get(name, [])
        try:
            df = pd.read_csv(
                path,
                dtype=dict.fromkeys(ts_cols, "string")
                | ({"postal_code": "string"} if name == "users" else {}),
                keep_default_na=True,
            )
        except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as e:
            raise DatabaseBuildError(f"Cannot read {path}: {e}") from e

        missing = [c for c in COLUMNS[name] if c not in df.columns]
        if missing:
            raise DatabaseBuildError(f"{path.name} is missing columns: {', '.join(missing)}")

        for c in ts_cols:
            try:
                df[c] = pd.to_datetime(df[c], utc=True, format="ISO8601").dt.floor("s")
            except (ValueError, TypeError) as e:
                raise DatabaseBuildError(
                    f"{path.name}: cannot parse timestamps in '{c}': {e}"
                ) from e
        dfs[name] = df
        logger.debug("Loaded {}.csv: {:,} rows", name, len(df))
    return dfs


def _examples(values: pd.Series, n: int = 5) -> str:
    """Format a few example values for an error message.

    Args:
        values: The offending values (e.g. IDs).
        n: Maximum number of examples to show.

    Returns:
        A string like ``"3 rows, e.g. 7, 12, 40"``.
    """
    unique = values.drop_duplicates()
    shown = ", ".join(str(v) for v in unique.head(n).tolist())
    return f"{len(values):,} rows, e.g. {shown}"


def check_source_data(dfs: dict[str, pd.DataFrame]) -> None:
    """Check the loaded tables for problems that would break cleaning or writing.

    Checks unique primary keys, that order items, orders and products point at existing
    rows, and that user fields needed for email deduplication are usable.

    Args:
        dfs: Tables returned by :func:`load`.

    Raises:
        DatabaseBuildError: Listing every problem found, with counts and example IDs.
    """
    problems: list[str] = []
    users, orders, items = dfs["users"], dfs["orders"], dfs["order_items"]
    products, dcs = dfs["products"], dfs["distribution_centers"]

    primary_keys = {
        "users": "id",
        "orders": "order_id",
        "order_items": "id",
        "products": "id",
        "distribution_centers": "id",
    }
    for table, key in primary_keys.items():
        ids = dfs[table][key]
        dup = ids[ids.duplicated()]
        if not dup.empty:
            problems.append(f"{table}: duplicate {key} ({_examples(dup)})")
        if ids.isna().any():
            problems.append(f"{table}: {ids.isna().sum():,} rows with empty {key}")

    references = [
        ("order_items", items, "order_id", orders["order_id"]),
        ("order_items", items, "product_id", products["id"]),
        ("orders", orders, "user_id", users["id"]),
        ("products", products, "distribution_center_id", dcs["id"]),
    ]
    for table, df, col, target in references:
        orphans = df.loc[~df[col].isin(target), col]
        if not orphans.empty:
            problems.append(f"{table}: {col} not found ({_examples(orphans)})")

    email = users["email"].astype("string").str.strip()
    bad_email = users.loc[email.isna() | ~email.str.contains("@", na=False), "id"]
    if not bad_email.empty:
        problems.append(f"users: missing or malformed email ({_examples(bad_email)})")
    if users["age"].isna().any():
        problems.append(f"users: empty age ({_examples(users.loc[users['age'].isna(), 'id'])})")

    # Users with a duplicate email get a name-based replacement, which needs usable names.
    dup_email = email.str.lower().duplicated(keep=False)
    unusable = users["first_name"].map(_slug).eq("") | users["last_name"].map(_slug).eq("")
    bad_names = users.loc[dup_email & unusable, "id"]
    if not bad_names.empty:
        problems.append(
            f"users: duplicate email but no usable first/last name ({_examples(bad_names)})"
        )

    if problems:
        raise DatabaseBuildError("Source data problems:\n  - " + "\n  - ".join(problems))


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

    Raises:
        DatabaseBuildError: If the tables contain no timestamps to anchor the shift on.
    """
    latest = [dfs[t][c].max() for t, cols in TIMESTAMP_COLS.items() for c in cols]
    latest = [ts for ts in latest if pd.notna(ts)]
    if not latest:
        raise DatabaseBuildError("No timestamps found in the source data; cannot shift dates")
    anchor = max(latest)
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

    Any existing file at ``out`` is replaced. Callers should pass a temporary path and
    move it into place only after validation (see :func:`build`).

    Args:
        dfs: Cleaned and shifted tables.
        out: Path of the SQLite file to create.
        meta: Key/value pairs for the ``meta`` table, including ``reference_now``.

    Raises:
        DatabaseBuildError: If SQLite rejects the schema or a table's rows.
    """
    out.unlink(missing_ok=True)
    step = "schema"
    try:
        with closing(sqlite3.connect(out)) as con:
            con.executescript(SCHEMA)
            for table in ["distribution_centers", "products", "users", "orders", "order_items"]:
                step = f"table '{table}'"
                cols = COLUMNS[table]
                placeholders = ", ".join("?" * len(cols))
                con.executemany(
                    f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({placeholders})",
                    _to_sql_values(dfs[table], table),
                )
            step = "table 'meta'"
            con.executemany("INSERT INTO meta VALUES (?, ?)", meta.items())
            con.commit()
    except sqlite3.Error as e:
        raise DatabaseBuildError(f"Failed writing {step} to {out}: {e}") from e


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


def validate(db: Path) -> list[str]:
    """Run integrity checks on a built database and log a report.

    Args:
        db: Path of the SQLite file to check.

    Returns:
        Names of the checks that found violations; empty if all passed.
    """
    failed: list[str] = []
    logger.info("Validating {}", db)
    with closing(sqlite3.connect(db)) as con:
        for name, sql in CHECKS.items():
            n = con.execute(sql).fetchone()[0]
            if n == 0:
                logger.info("  OK   {}: {}", name, n)
            else:
                failed.append(name)
                logger.error("  FAIL {}: {}", name, n)
        for t in TABLES:
            count = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            logger.info("  rows {}: {:,}", t, count)
    return failed


# --------------------------------------------------------------------------- main
def _iso_date(value: str) -> date:
    """Parse a ``YYYY-MM-DD`` command-line value.

    Args:
        value: The raw argument.

    Returns:
        The parsed date.

    Raises:
        argparse.ArgumentTypeError: If the value is not a valid ISO date.
    """
    try:
        return date.fromisoformat(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"invalid date {value!r}, expected YYYY-MM-DD") from e


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Arguments to parse; defaults to ``sys.argv[1:]``.

    Returns:
        The parsed arguments.
    """
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--out", type=Path, default=Path("ecommerce.sqlite"))
    p.add_argument(
        "--today", type=_iso_date, help="reference date YYYY-MM-DD (default: today, UTC)"
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def build(data_dir: Path, out: Path, today: date, seed: int) -> None:
    """Build and validate the database, replacing ``out`` only if validation passes.

    The database is written to ``<out>.tmp`` first. On success it atomically replaces
    ``out``. If validation fails it is kept as ``<out>.failed`` for inspection and
    ``out`` is left untouched. On any other error the temporary file is removed.

    Args:
        data_dir: Directory containing the source CSV files.
        out: Path of the SQLite file to produce.
        today: Reference date (TODAY) stored in the ``meta`` table.
        seed: Seed for the email deduplication.

    Raises:
        DatabaseBuildError: If the inputs, source data, write or validation fail.
    """
    reference_now = pd.Timestamp(f"{today.isoformat()} 00:00:00", tz="UTC")
    logger.info("Building {} from {} (TODAY = {})", out, data_dir, today)

    check_inputs(data_dir)
    logger.info("Loading CSVs")
    dfs = load(data_dir)
    logger.info("Checking source data")
    check_source_data(dfs)
    logger.info("Cleaning tables")
    dfs = clean(dfs)

    offset, anchor = shift_dates(dfs, reference_now)
    logger.info(
        "Shifted timestamps by {} days ({} weeks); source latest event {}",
        offset.days,
        offset.days // 7,
        anchor.strftime(TS_FMT),
    )

    users = dfs["users"]
    original = users["email"].str.lower()
    users["email"] = make_emails_unique(users, ref_year=reference_now.year, seed=seed)
    logger.info("Made emails unique: {:,} changed", (users["email"] != original).sum())

    meta = {
        "reference_now": reference_now.strftime(TS_FMT),
        "date_shift_days": str(offset.days),
        "source_latest_event": anchor.strftime(TS_FMT),
        "email_seed": str(seed),
        "built_at": datetime.now(UTC).strftime(TS_FMT),
    }

    tmp = out.with_name(out.name + ".tmp")
    failed_copy = out.with_name(out.name + ".failed")
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        logger.info("Writing {}", tmp)
        write(dfs, tmp, meta)
        failures = validate(tmp)
        if failures:
            tmp.replace(failed_copy)
            raise DatabaseBuildError(
                f"Validation failed ({', '.join(failures)}). "
                f"Kept {failed_copy} for inspection; {out} is unchanged."
            )
        tmp.replace(out)
        failed_copy.unlink(missing_ok=True)
    finally:
        tmp.unlink(missing_ok=True)

    logger.info("reference_now (TODAY): {}", meta["reference_now"])
    logger.success("Built {}", out)


@logger.catch(onerror=lambda _: sys.exit(1))
def main(argv: Sequence[str] | None = None) -> int:
    """Run the build from the command line.

    Expected failures (:class:`DatabaseBuildError`) are logged without a traceback. Unexpected
    errors are logged with a full traceback by ``logger.catch`` and exit with code 1.

    Args:
        argv: Command-line arguments; defaults to ``sys.argv[1:]``.

    Returns:
        The process exit code: 0 on success, 1 on a build error, 130 if interrupted.
    """
    args = parse_args(argv)
    today = args.today or datetime.now(UTC).date()
    try:
        build(args.data_dir, args.out, today, args.seed)
    except DatabaseBuildError as e:
        logger.error("Build failed: {}", e)
        return 1
    except KeyboardInterrupt:
        logger.warning("Interrupted; {} is unchanged", args.out)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
