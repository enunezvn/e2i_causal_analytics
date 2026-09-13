"""Migration 140 content lock (#1991 debt 3): estimand_key as a stored generated
column, pending-uniqueness on the estimand, and the guarded supersede of
BLOCK-band pending rows."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
M = REPO / "database" / "migrations" / "140_expert_reviews_estimand_key.sql"


def _sql() -> str:
    return "\n".join(
        line for line in M.read_text().splitlines() if not line.strip().startswith("--")
    )


@pytest.mark.unit
def test_estimand_key_is_a_stored_generated_column():
    s = _sql()
    assert re.search(r"ADD COLUMN IF NOT EXISTS estimand_key TEXT\s+GENERATED ALWAYS AS", s)
    assert "STORED" in s
    assert "lower(COALESCE(brand, ''))" in s
    assert "lower(COALESCE(treatment_variable, ''))" in s
    assert "lower(COALESCE(outcome_variable, ''))" in s
    # A plain backfill UPDATE + SET NOT NULL would fight a generated column
    # (writers cannot supply generated values) -- the column derives itself.
    assert not re.search(r"SET estimand_key\s*=", s)
    assert "SET NOT NULL" not in s


@pytest.mark.unit
def test_swaps_the_pending_unique_index():
    s = _sql()
    assert "DROP INDEX IF EXISTS uq_er_pending_dag_brand" in s
    assert re.search(
        r"CREATE UNIQUE INDEX IF NOT EXISTS uq_er_pending_estimand\s+ON public\.expert_reviews \(estimand_key\)\s+WHERE approval_status = 'pending'",
        s,
    )


@pytest.mark.unit
def test_supersede_is_guarded_by_an_asserted_count():
    s = _sql()
    assert "approval_status = 'superseded'" in s
    assert "gate=block" in s
    assert "RAISE EXCEPTION" in s and "GET DIAGNOSTICS" in s
    assert "resolved_at = now()" in s
    assert re.search(r"IF v_done <> v_expected THEN\s+RAISE EXCEPTION", s)
    assert "jsonb_typeof(comments_json)" in s


@pytest.mark.unit
def test_supersede_runs_before_the_new_unique_index():
    # 7 estimands hold more than one pending row today -- uq_er_pending_estimand
    # would fail on duplicates if it were created before those rows are
    # superseded, so the ordering in the file is load-bearing, not incidental.
    s = _sql()
    assert s.index("approval_status = 'superseded'") < s.index("uq_er_pending_estimand")


@pytest.mark.unit
def test_no_delete_and_no_own_transaction():
    s = _sql().upper()
    assert "DELETE FROM" not in s
    assert "BEGIN;" not in s and "COMMIT;" not in s
