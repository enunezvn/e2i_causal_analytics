"""Migration 139 content lock: causal_validations.delta_percent DECIMAL(8,4) -> NUMERIC(12,4).

Every refuter's ``delta_percent`` is ``|delta| / |original| * 100``; the four
perturbation tests compute it UNCLAMPED with a 1e-10 floor, so a near-zero
claim overflows ``DECIMAL(8,4)`` (max 9999.9999,
``database/ml/010_causal_validation_tables.sql``). ``save_suite`` inserts the
whole suite in ONE call, so one overflowing row loses every row's persistence
(#2029, lane 1 of the #1991 debts 3/4 wave).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
MIGRATION = (
    REPO_ROOT / "database" / "migrations" / "139_causal_validations_delta_percent_numeric_12_4.sql"
)


def _statements() -> str:
    return "\n".join(
        ln
        for ln in MIGRATION.read_text(encoding="utf-8").splitlines()
        if not ln.strip().startswith("--")
    )


@pytest.mark.unit
def test_migration_exists_and_is_numbered_139():
    assert MIGRATION.exists()
    assert len(list((REPO_ROOT / "database").rglob("139_*"))) == 1


@pytest.mark.unit
def test_widens_the_column_idempotently():
    sql = _statements()
    assert re.search(r"ALTER\s+TABLE\s+public\.causal_validations", sql, re.I)
    assert re.search(r"ALTER\s+COLUMN\s+delta_percent\s+TYPE\s+NUMERIC\(12,\s*4\)", sql, re.I)
    # idempotent AND never narrowing: the guard fires only when the column is
    # narrower than 12 (or untyped), so a re-run -- or a later wider column -- is a no-op
    assert re.search(r"numeric_precision\s+IS\s+NULL", sql, re.I)
    assert re.search(r"numeric_precision\s+<\s+12", sql)
    assert not re.search(r"numeric_precision\s+IS\s+DISTINCT\s+FROM\s+12", sql, re.I)


@pytest.mark.unit
def test_no_transaction_control_of_its_own():
    sql = _statements().upper()
    assert "BEGIN;" not in sql and "COMMIT;" not in sql
