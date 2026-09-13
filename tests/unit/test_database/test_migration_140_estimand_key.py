"""Migration 140 content lock (#1991 debt 3): estimand_key, pending-uniqueness on the estimand,
and the guarded supersede of BLOCK-band pending rows."""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
M = REPO / "database" / "migrations" / "140_expert_reviews_estimand_key.sql"


def _sql() -> str:
    return "\n".join(
        line for line in M.read_text().splitlines() if not line.strip().startswith("--")
    )


def test_adds_and_backfills_estimand_key():
    s = _sql()
    assert re.search(r"ADD COLUMN IF NOT EXISTS estimand_key TEXT", s)
    assert re.search(r"UPDATE public\.expert_reviews\s+SET estimand_key\s*=", s)
    assert "lower(" in s and "COALESCE(brand, '')" in s
    assert re.search(r"ALTER COLUMN estimand_key SET NOT NULL", s)


def test_swaps_the_pending_unique_index():
    s = _sql()
    assert "DROP INDEX IF EXISTS uq_er_pending_dag_brand" in s
    assert re.search(
        r"CREATE UNIQUE INDEX IF NOT EXISTS uq_er_pending_estimand\s+ON public\.expert_reviews \(estimand_key\)\s+WHERE approval_status = 'pending'",
        s,
    )


def test_supersede_is_guarded_by_an_asserted_count():
    s = _sql()
    assert "approval_status = 'superseded'" in s
    assert "gate=block" in s
    assert "RAISE EXCEPTION" in s and "GET DIAGNOSTICS" in s
    assert "resolved_at = now()" in s


def test_no_delete_and_no_own_transaction():
    s = _sql().upper()
    assert "DELETE FROM" not in s
    assert "BEGIN;" not in s and "COMMIT;" not in s
