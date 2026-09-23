"""Migration 155 content lock (option d1, owner decision 2026-09-23, Part of #2207):
``ab_experiment_unit_outcomes`` — one observed outcome per (experiment, unit,
metric), time-indexed by observed_at, the MEASURED side of the twin fidelity loop.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
M = REPO / "database" / "migrations" / "155_ab_experiment_unit_outcomes.sql"
R = REPO / "database" / "migrations" / "rollback_155_ab_experiment_unit_outcomes.sql"
TABLE = "ab_experiment_unit_outcomes"


def _sql(path: Path) -> str:
    return "\n".join(
        line for line in path.read_text().splitlines() if not line.strip().startswith("--")
    )


def _ws(s: str) -> str:
    return re.sub(r"\s+", " ", s)


@pytest.mark.unit
def test_creates_the_table_idempotently_with_the_contract_columns():
    s = _ws(_sql(M))
    assert re.search(rf"CREATE TABLE IF NOT EXISTS (public\.)?{TABLE} \(", s)
    assert "id UUID PRIMARY KEY DEFAULT gen_random_uuid()" in s
    assert re.search(
        r"assignment_id UUID NOT NULL REFERENCES (public\.)?ab_experiment_assignments\(id\) ON DELETE CASCADE",
        s,
    )
    assert re.search(
        r"experiment_id UUID NOT NULL REFERENCES (public\.)?ml_experiments\(id\) ON DELETE RESTRICT",
        s,
    )
    assert "unit_id VARCHAR(255) NOT NULL" in s
    assert "metric_name VARCHAR(255) NOT NULL" in s
    assert "outcome_value DOUBLE PRECISION NOT NULL" in s
    assert re.search(r"observed_at TIMESTAMP(TZ| WITH TIME ZONE) NOT NULL", s)
    assert "is_synthetic BOOLEAN NOT NULL DEFAULT false" in s
    assert re.search(r"UNIQUE \(experiment_id, unit_id, metric_name\)", s)
    assert "BEGIN;" not in s and "COMMIT;" not in s  # the runner wraps the file


@pytest.mark.unit
def test_indexes_and_comments():
    s = _ws(_sql(M))
    assert re.search(
        rf"CREATE INDEX IF NOT EXISTS \w+ ON (public\.)?{TABLE} ?\(experiment_id, metric_name\)", s
    )
    assert re.search(rf"CREATE INDEX IF NOT EXISTS \w+ ON (public\.)?{TABLE} ?\(assignment_id\)", s)
    assert re.search(rf"COMMENT ON TABLE (public\.)?{TABLE} IS", s)
    assert re.search(rf"COMMENT ON COLUMN (public\.)?{TABLE}\.observed_at IS", s)
    assert re.search(rf"COMMENT ON COLUMN (public\.)?{TABLE}\.is_synthetic IS", s)


@pytest.mark.unit
def test_no_data_is_written_by_the_migration():
    """DDL only: no statement starts with INSERT / UPDATE / DELETE (the words
    appear inside ON DELETE clauses and COMMENT strings, which is fine)."""
    s = _sql(M)
    assert not re.search(r"^\s*(INSERT|UPDATE|DELETE)\b", s, flags=re.I | re.M)


@pytest.mark.unit
def test_rollback_drops_exactly_this_table():
    s = _ws(_sql(R))
    drops = re.findall(r"DROP TABLE IF EXISTS (?:public\.)?(\w+)", s)
    assert drops == [TABLE]
    assert "DROP TABLE" not in _sql(M).upper()


@pytest.mark.unit
def test_rollback_retires_exactly_its_own_ledger_row():
    """A manual rollback must also retire the schema_migrations row (the 143/144
    pattern; codex r2 MED) or the deploy runner skips recreating the table."""
    s = _ws(_sql(R))
    deletes = re.findall(
        r"DELETE FROM (?:public\.)?schema_migrations WHERE filename = '([^']+)'", s
    )
    assert deletes == [M.name]
    assert s.count("DELETE FROM") == 1
