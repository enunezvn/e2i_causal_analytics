"""Migration 150 content lock (#2207 follow-up, owner decision 2026-09-22): the per-model
cohort contract lives on ``ml_model_registry`` as three nullable TEXT columns, the
target is backfilled NULL-only from the experiment's declared prediction target through
the FK, and ``cohort_data_source`` is NOT backfilled (the goldstd training is not
expressible as a (table, target column) contract — see the migration's own comment).

Rehearsed live in BEGIN…ROLLBACK on 2026-09-22 (prod-equivalent local Supabase): 14
is_synthetic=false rows gained a cohort_target_outcome, 0 a cohort_data_source, and the
rollback file restores the pre-150 column set.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
M = REPO / "database" / "migrations" / "150_ml_registry_cohort_contract.sql"
R = REPO / "database" / "migrations" / "rollback_150_ml_registry_cohort_contract.sql"

COLUMNS = ("cohort_data_source", "cohort_target_outcome", "cohort_feature_manifest_source")


def _sql(path: Path) -> str:
    return "\n".join(
        line for line in path.read_text().splitlines() if not line.strip().startswith("--")
    )


@pytest.mark.unit
def test_three_nullable_text_columns_added_idempotently():
    s = _sql(M)
    for col in COLUMNS:
        assert re.search(
            rf"ALTER TABLE ml_model_registry ADD COLUMN IF NOT EXISTS {col} TEXT;", s
        ), col
        assert f"COMMENT ON COLUMN ml_model_registry.{col} IS" in s, col
    assert not re.search(r"ADD COLUMN[^;]*NOT NULL", s)  # nullable; only the backfill's WHERE
    assert "ADD CONSTRAINT" not in s
    assert "BEGIN;" not in s and "COMMIT;" not in s  # the runner wraps the file


@pytest.mark.unit
def test_comments_state_the_sweep_rule():
    s = _sql(M)
    assert s.count("only enqueue a retrain when BOTH cohort_data_source and") == 2


@pytest.mark.unit
def test_target_backfill_is_null_only_from_the_experiment_fk_and_real_models_only():
    s = _sql(M)
    m = re.search(
        r"UPDATE ml_model_registry r\s+SET cohort_target_outcome = e\.prediction_target\s+"
        r"FROM ml_experiments e\s+WHERE(.*?);",
        s,
        re.S,
    )
    assert m, "target backfill missing"
    where = m.group(1)
    assert "e.id = r.experiment_id" in where
    assert "r.cohort_target_outcome IS NULL" in where
    assert "r.is_synthetic = false" in where
    assert "e.prediction_target IS NOT NULL" in where


@pytest.mark.unit
def test_data_source_is_never_backfilled():
    s = _sql(M)
    assert not re.search(r"SET\s+cohort_data_source", s)
    assert not re.search(r"SET\s+cohort_feature_manifest_source", s)


@pytest.mark.unit
def test_rollback_drops_only_the_three_columns():
    s = _sql(R)
    for col in COLUMNS:
        assert f"ALTER TABLE ml_model_registry DROP COLUMN IF EXISTS {col};" in s, col
    assert s.count("DROP COLUMN") == 3
    assert "DROP TABLE" not in s


@pytest.mark.unit
def test_migration_number_is_unique_in_the_directory():
    files = sorted((REPO / "database" / "migrations").glob("150_*.sql"))
    assert [f.name for f in files] == ["150_ml_registry_cohort_contract.sql"]
