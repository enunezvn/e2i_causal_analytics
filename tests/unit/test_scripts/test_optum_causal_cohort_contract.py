"""Migration 148 carries one column per field the causal export emits — no more,
no fewer (spec 2026-09-22 §3A.2). The export is the SSOT: a feature added to
MART_SAFE_FEATURES or a new extra column shows up here as a missing SQL column.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.convert_optum_mart import (  # noqa: E402
    CAUSAL_EXTRA_COLS,
    SWITCH_FLAG,
    TARGET_PERSISTENT_G28,
    TREATMENT_COL,
    build_journey_records,
)
from src.data.manifests import MART_SAFE_FEATURES  # noqa: E402

MIGRATION = _REPO_ROOT / "database/migrations/148_optum_biologic_persistence_causal.sql"
ROLLBACK = _REPO_ROOT / "database/migrations/rollback_148_optum_biologic_persistence_causal.sql"
TABLE = "optum_biologic_persistence_causal"
# Server-defaulted bookkeeping the loader never writes.
_SERVER_COLUMNS = {"created_at", "updated_at"}
# The types this table is allowed to use. The regex below deliberately matches
# ANY bare identifier-shaped word — including one with digits or underscores,
# e.g. INT8 / MY_TYPE — (so it never silently skips a column whose type it
# doesn't recognise, e.g. DOUBLE PRECISION / BIGINT / JSONB / TIMESTAMP) —
# that parsed type is then checked against this allow-list in ``_sql_columns``.
_KNOWN_TYPES = {
    "TEXT",
    "VARCHAR",
    "INTEGER",
    "SMALLINT",
    "NUMERIC",
    "DATE",
    "BOOLEAN",
    "TIMESTAMPTZ",
}
_COLUMN_RE = re.compile(
    r"^\s*([a-z_][a-z0-9_]*)\s+([A-Za-z_][A-Za-z0-9_]*(?:\s+PRECISION)?)\b",
    re.IGNORECASE,
)
# Table-level declarations (not columns) that legitimately don't match
# ``_COLUMN_RE`` as a (name, type) pair — checked BEFORE the regex so a
# standalone ``PRIMARY KEY (...)`` line is never mistaken for a column named
# "primary" of type "key".
_TABLE_CONSTRAINT_PREFIXES = ("CONSTRAINT", "PRIMARY KEY", "UNIQUE", "CHECK", "FOREIGN KEY")


def _sql_columns(sql: str) -> dict[str, str]:
    body = sql.split("CREATE TABLE IF NOT EXISTS", 1)[1]
    body = body[body.index("(") + 1 :]
    depth, end = 1, 0
    for i, ch in enumerate(body):
        depth += ch == "("
        depth -= ch == ")"
        if depth == 0:
            end = i
            break
    columns: dict[str, str] = {}
    for line in body[:end].splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("--"):
            continue
        if stripped.upper().startswith(_TABLE_CONSTRAINT_PREFIXES):
            continue
        m = _COLUMN_RE.match(line)
        if not m:
            raise AssertionError(f"unparsed declaration line {line!r}")
        col_type = m.group(2).upper()
        if col_type not in _KNOWN_TYPES:
            raise AssertionError(f"unrecognised column type {col_type!r} on line {line!r}")
        columns[m.group(1)] = col_type
    return columns


def _export_record_keys() -> set[str]:
    row = {
        "patid": 1,
        "index_date": pd.Timestamp("2020-01-01"),
        "treatment_start_date": pd.Timestamp("2020-02-01"),
        "elig_start_date": pd.Timestamp("2019-01-01"),
        "zipcode_5": "10001",
        "index_biologic_brand": "XOLAIR",
        TREATMENT_COL: 0,
        TARGET_PERSISTENT_G28: 1,
        "discontinued_180d": 0,
        SWITCH_FLAG: 0,
        "persistent_at_180d": 1,
    }
    # every raw allow-list feature present so raw_features is the full list
    for name in MART_SAFE_FEATURES:
        row.setdefault(name, 0)
    rec = build_journey_records(
        pd.DataFrame([row]),
        target=TARGET_PERSISTENT_G28,
        anchor_col="treatment_start_date",
        extra_cols=CAUSAL_EXTRA_COLS,
    )[0]
    return set(rec) | {"is_synthetic", "data_split"}


def test_migration_148_columns_equal_the_export_fields():
    columns = _sql_columns(MIGRATION.read_text())
    exported = _export_record_keys()
    assert set(columns) - _SERVER_COLUMNS == exported, {
        "sql_only": sorted(set(columns) - _SERVER_COLUMNS - exported),
        "export_only": sorted(exported - set(columns)),
    }


def test_migration_148_types_and_constraints():
    sql = MIGRATION.read_text()
    columns = _sql_columns(sql)
    assert f"CREATE TABLE IF NOT EXISTS public.{TABLE}" in sql
    # the header comment also says "patient_id" — match the column line, not the prose
    assert re.search(r"^\s*patient_id\s+VARCHAR\(40\) PRIMARY KEY,", sql, re.MULTILINE)
    assert columns["is_synthetic"] == "BOOLEAN"
    assert re.search(r"^\s*is_synthetic\s+BOOLEAN NOT NULL DEFAULT false,", sql, re.MULTILINE)
    for col in (
        TREATMENT_COL,
        TARGET_PERSISTENT_G28,
        "discontinued_180d",
        SWITCH_FLAG,
        "persistent_at_180d",
    ):
        assert columns[col] == "SMALLINT", col
        assert f"CHECK ({col} IN (0, 1))" in sql, col
    for col in (
        "gdr_cd",
        "payer_category",
        "payer_product",
        "payer_bus",
        "charlson_risk_band",
        "elixhauser_risk_band",
        "geographic_region",
        "index_biologic_brand",
    ):
        assert columns[col] == "TEXT", col
    for col in ("index_date", "journey_start_date", "treatment_start_date"):
        assert columns[col] == "DATE", col
    assert f"idx_{TABLE}_treatment" in sql and f"idx_{TABLE}_outcomes" in sql
    statements = "\n".join(l for l in sql.splitlines() if not l.strip().startswith("--"))
    assert (
        "BEGIN;" not in statements.upper() and "COMMIT;" not in statements.upper()
    )  # the runner wraps each file


def test_rollback_148_drops_only_the_new_table():
    sql = ROLLBACK.read_text()
    assert f"DROP TABLE IF EXISTS public.{TABLE};" in sql
    assert sql.count("DROP") == 1


def test_sql_columns_rejects_an_unrecognised_type():
    """A type outside ``_KNOWN_TYPES`` (e.g. DOUBLE PRECISION, BIGINT, JSONB) must
    fail loud, never be silently skipped — a skipped column would pass the "no
    more" half of the contract test unnoticed."""
    sql = "CREATE TABLE IF NOT EXISTS public.t (\n    foo DOUBLE PRECISION,\n);\n"
    with pytest.raises(AssertionError, match="foo"):
        _sql_columns(sql)


def test_sql_columns_rejects_a_bare_type_with_digits():
    """INT8 is a valid Postgres type spelling; it must still parse to a (name,
    type) pair and then fail the ``_KNOWN_TYPES`` check, not be silently
    skipped by the regex for containing a digit."""
    sql = "CREATE TABLE IF NOT EXISTS public.t (\n    extra_col INT8,\n);\n"
    with pytest.raises(AssertionError, match="extra_col"):
        _sql_columns(sql)


def test_sql_columns_rejects_a_bare_type_with_underscore():
    """Same as above for an underscored type name (e.g. a custom domain)."""
    sql = "CREATE TABLE IF NOT EXISTS public.t (\n    extra_col MY_TYPE,\n);\n"
    with pytest.raises(AssertionError, match="extra_col"):
        _sql_columns(sql)


def test_sql_columns_rejects_an_unparsed_declaration_line():
    """A line that is neither a (name, type) column nor a recognised
    table-level constraint keyword must fail loud rather than vanish."""
    sql = "CREATE TABLE IF NOT EXISTS public.t (\n    ???,\n);\n"
    with pytest.raises(AssertionError, match="unparsed declaration line"):
        _sql_columns(sql)
