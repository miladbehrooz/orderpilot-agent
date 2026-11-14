"""Error handling in scripts/build_db.py, using a tiny generated dataset (no real data)."""

import hashlib
import sqlite3
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytest
from pytest_mock import MockerFixture

import build_db
from build_db import BuildError, build, main, parse_args

TODAY = date(2026, 11, 12)

WriteCsvs = Callable[[dict[str, pd.DataFrame]], Path]


@pytest.fixture
def tables() -> dict[str, pd.DataFrame]:
    """Return a minimal, valid source dataset that passes every validation check.

    Users 1 and 2 share an email, so the email deduplication path is exercised.
    """
    ts = "2024-01-{:02d} 10:00:00+00:00".format
    return {
        "distribution_centers": pd.DataFrame(
            {"id": [1], "name": ["DC1"], "latitude": [0.0], "longitude": [0.0]}
        ),
        "products": pd.DataFrame(
            {
                "id": [1],
                "name": ["Shirt"],
                "brand": ["Acme"],
                "category": ["Tops"],
                "department": ["Men"],
                "cost": [5.0],
                "retail_price": [10.0],
                "sku": ["SKU1"],
                "distribution_center_id": [1],
            }
        ),
        "users": pd.DataFrame(
            {
                "id": [1, 2],
                "first_name": ["Ann", "Bob"],
                "last_name": ["Lee", "Ray"],
                "email": ["ann@example.com", "ann@example.com"],
                "age": [30, 40],
                "gender": ["F", "M"],
                "street_address": ["1 Main St", "2 Main St"],
                "postal_code": ["10115", "10117"],
                "city": ["Berlin", "Berlin"],
                "state": ["Berlin", "Berlin"],
                "country": ["Deutschland", "Germany"],
                "latitude": [52.5, 52.5],
                "longitude": [13.4, 13.4],
                "traffic_source": ["Search", "Email"],
                "created_at": [ts(1), ts(2)],
            }
        ),
        "orders": pd.DataFrame(
            {
                "order_id": [1, 2],
                "user_id": [1, 2],
                "status": ["Complete", "Processing"],
                "gender": ["F", "M"],
                "num_of_item": [1, 1],
                "created_at": [ts(3), ts(6)],
                "shipped_at": [ts(4), None],
                "delivered_at": [ts(5), None],
                "returned_at": [None, None],
            }
        ),
        "order_items": pd.DataFrame(
            {
                "id": [1, 2],
                "order_id": [1, 2],
                "user_id": [1, 2],
                "product_id": [1, 1],
                "inventory_item_id": [1, 2],
                "status": ["Complete", "Processing"],
                "sale_price": [9.999, 10.0],
                "created_at": [ts(3), ts(6)],
                "shipped_at": [ts(4), None],
                "delivered_at": [ts(5), None],
                "returned_at": [None, None],
            }
        ),
    }


@pytest.fixture
def write_csvs(tmp_path: Path) -> WriteCsvs:
    """Return a function that writes tables as CSVs into a fresh data directory.

    Args:
        tmp_path: pytest's per-test temporary directory.
    """

    def _write(tables: dict[str, pd.DataFrame]) -> Path:
        """Write each table to ``<tmp>/data/<name>.csv`` and return the directory."""
        data_dir = tmp_path / "data"
        data_dir.mkdir(exist_ok=True)
        for name, df in tables.items():
            df.to_csv(data_dir / f"{name}.csv", index=False)
        return data_dir

    return _write


def _sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of a file.

    Args:
        path: File to hash.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_build_creates_valid_database(
    tables: dict[str, pd.DataFrame], write_csvs: WriteCsvs, tmp_path: Path
) -> None:
    """A valid dataset builds a database with unique emails and no leftover temp file."""
    out = tmp_path / "out" / "ecommerce.sqlite"  # parent folder does not exist yet
    build(write_csvs(tables), out, TODAY, seed=42)

    assert out.is_file()
    assert not out.with_name("ecommerce.sqlite.tmp").exists()
    with sqlite3.connect(out) as con:
        emails = [r[0] for r in con.execute("SELECT email FROM users")]
    assert len(emails) == len(set(emails)) == 2


def test_missing_data_dir_raises(tmp_path: Path) -> None:
    """A non-existent data directory fails with a clear message."""
    with pytest.raises(BuildError, match="Data directory not found"):
        build(tmp_path / "nope", tmp_path / "db.sqlite", TODAY, seed=42)


def test_missing_csvs_are_all_listed(
    tables: dict[str, pd.DataFrame], write_csvs: WriteCsvs, tmp_path: Path
) -> None:
    """Every missing CSV is named in one error, not just the first."""
    del tables["orders"], tables["products"]
    with pytest.raises(BuildError, match=r"orders\.csv, products\.csv"):
        build(write_csvs(tables), tmp_path / "db.sqlite", TODAY, seed=42)


def test_missing_column_raises(
    tables: dict[str, pd.DataFrame], write_csvs: WriteCsvs, tmp_path: Path
) -> None:
    """A CSV without a required column names the file and the column."""
    tables["orders"] = tables["orders"].drop(columns=["status"])
    with pytest.raises(BuildError, match=r"orders\.csv is missing columns: status"):
        build(write_csvs(tables), tmp_path / "db.sqlite", TODAY, seed=42)


def test_unparseable_timestamp_raises(
    tables: dict[str, pd.DataFrame], write_csvs: WriteCsvs, tmp_path: Path
) -> None:
    """A malformed timestamp names the file and the column."""
    tables["orders"].loc[0, "shipped_at"] = "not a date"
    with pytest.raises(BuildError, match=r"orders\.csv: cannot parse timestamps in 'shipped_at'"):
        build(write_csvs(tables), tmp_path / "db.sqlite", TODAY, seed=42)


def test_orphan_order_item_raises(
    tables: dict[str, pd.DataFrame], write_csvs: WriteCsvs, tmp_path: Path
) -> None:
    """An order item pointing at a non-existent order fails before cleaning."""
    tables["order_items"].loc[1, "order_id"] = 999
    with pytest.raises(BuildError, match=r"order_items: order_id not found \(1 rows, e\.g\. 999\)"):
        build(write_csvs(tables), tmp_path / "db.sqlite", TODAY, seed=42)


def test_duplicate_order_id_raises(
    tables: dict[str, pd.DataFrame], write_csvs: WriteCsvs, tmp_path: Path
) -> None:
    """Duplicate primary keys are reported with the offending ID."""
    tables["orders"].loc[1, "order_id"] = 1
    with pytest.raises(BuildError, match=r"orders: duplicate order_id \(1 rows, e\.g\. 1\)"):
        build(write_csvs(tables), tmp_path / "db.sqlite", TODAY, seed=42)


def test_malformed_email_raises(
    tables: dict[str, pd.DataFrame], write_csvs: WriteCsvs, tmp_path: Path
) -> None:
    """An email without '@' is reported before the deduplication step can crash."""
    tables["users"].loc[0, "email"] = "no-at-sign"
    with pytest.raises(BuildError, match="missing or malformed email"):
        build(write_csvs(tables), tmp_path / "db.sqlite", TODAY, seed=42)


def test_validation_failure_keeps_previous_database(
    tables: dict[str, pd.DataFrame],
    write_csvs: WriteCsvs,
    tmp_path: Path,
    mocker: MockerFixture,
) -> None:
    """A failed validation leaves the existing database untouched and keeps a .failed copy."""
    out = tmp_path / "ecommerce.sqlite"
    build(write_csvs(tables), out, TODAY, seed=42)
    before = _sha256(out)

    validate_spy = mocker.spy(build_db, "validate")
    tables["orders"].loc[0, "num_of_item"] = 2  # passes source checks, fails validation
    with pytest.raises(BuildError, match="item count mismatch"):
        build(write_csvs(tables), out, TODAY, seed=42)

    validate_spy.assert_called_once_with(tmp_path / "ecommerce.sqlite.tmp")  # never `out`
    assert validate_spy.spy_return == ["item count mismatch"]
    assert _sha256(out) == before
    assert (tmp_path / "ecommerce.sqlite.failed").is_file()
    assert not (tmp_path / "ecommerce.sqlite.tmp").exists()


def test_sqlite_error_is_reported_and_temp_file_removed(
    tables: dict[str, pd.DataFrame],
    write_csvs: WriteCsvs,
    tmp_path: Path,
    mocker: MockerFixture,
) -> None:
    """A SQLite error during the write becomes a BuildError naming the step."""
    # Patch where build_db looks it up.
    mocker.patch("build_db.sqlite3.connect", side_effect=sqlite3.OperationalError("disk I/O error"))
    out = tmp_path / "ecommerce.sqlite"
    with pytest.raises(BuildError, match=r"Failed writing schema .*: disk I/O error"):
        build(write_csvs(tables), out, TODAY, seed=42)
    assert not out.exists()
    assert not (tmp_path / "ecommerce.sqlite.tmp").exists()


def test_interrupt_removes_temp_file_and_keeps_database(
    tables: dict[str, pd.DataFrame],
    write_csvs: WriteCsvs,
    tmp_path: Path,
    mocker: MockerFixture,
) -> None:
    """Ctrl-C during the write exits with 130, removes the temp file and keeps the old DB."""
    data_dir = write_csvs(tables)
    out = tmp_path / "ecommerce.sqlite"
    build(data_dir, out, TODAY, seed=42)
    before = _sha256(out)

    def interrupted_write(dfs: object, path: Path, meta: object) -> None:
        """Leave a partial file behind, then simulate Ctrl-C."""
        path.write_bytes(b"partial")
        raise KeyboardInterrupt

    mocker.patch.object(build_db, "write", autospec=True, side_effect=interrupted_write)
    args = ["--data-dir", str(data_dir), "--out", str(out), "--today", TODAY.isoformat()]

    assert main(args) == 130
    assert _sha256(out) == before
    assert not (tmp_path / "ecommerce.sqlite.tmp").exists()


def test_invalid_today_is_rejected(capsys: pytest.CaptureFixture[str]) -> None:
    """An invalid --today exits with argparse's usage error (code 2) and a clear message."""
    with pytest.raises(SystemExit) as exc:
        parse_args(["--today", "2026-13-01"])
    assert exc.value.code == 2
    assert "expected YYYY-MM-DD" in capsys.readouterr().err


# --------------------------------------------------------------------------- main()
# These tests check main() on its own, so build() is replaced by a mock.


def test_main_passes_arguments_to_build(mocker: MockerFixture) -> None:
    """main() parses the CLI arguments, calls build() with them and returns 0."""
    mock_build = mocker.patch.object(build_db, "build", autospec=True)
    code = main(["--data-dir", "d", "--out", "o.sqlite", "--today", "2026-11-12", "--seed", "7"])
    assert code == 0
    mock_build.assert_called_once_with(Path("d"), Path("o.sqlite"), date(2026, 11, 12), 7)


def test_main_defaults_today_to_current_utc_date(mocker: MockerFixture) -> None:
    """Without --today, main() uses the current UTC date (the clock is mocked)."""
    mock_build = mocker.patch.object(build_db, "build", autospec=True)
    mock_datetime = mocker.patch.object(build_db, "datetime", autospec=True)
    mock_datetime.now.return_value = datetime(2030, 1, 2, 23, 59, tzinfo=UTC)

    assert main([]) == 0
    mock_datetime.now.assert_called_once_with(UTC)
    assert mock_build.call_args.args[2] == date(2030, 1, 2)


def test_main_returns_1_on_build_error(mocker: MockerFixture) -> None:
    """main() turns a BuildError into exit code 1 instead of a traceback."""
    mocker.patch.object(build_db, "build", autospec=True, side_effect=BuildError("boom"))
    assert main([]) == 1


def test_main_exits_1_on_unexpected_error(mocker: MockerFixture) -> None:
    """An unexpected exception is caught by logger.catch and exits with code 1."""
    mocker.patch.object(build_db, "build", autospec=True, side_effect=RuntimeError("bug"))
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 1
