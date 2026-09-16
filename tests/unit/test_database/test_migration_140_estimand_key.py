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


@pytest.mark.unit
def test_reverse_block_deletes_the_ledger_row_and_warns_about_pending_collisions():
    """The REVERSE instructions are read by a human, so they are asserted on the
    RAW text -- ``_sql()`` strips exactly the comment lines they live in.

    Two things were missing. Without the ledger DELETE, run_migrations.sh's
    already-applied check (~:117) skips this file, so a later re-apply silently
    no-ops instead of recreating the column and index -- the same reason
    migration 141's REVERSE block spells it out. And restoring the superseded
    rows to pending re-creates the PRE-140 index on (dag_version_hash,
    COALESCE(brand,'')), which a pending row minted since could already occupy:
    the restore then fails mid-transaction, or takes a slot the live queue is
    using. Naming the check is the difference between a reversal plan and a hope.
    """
    raw = M.read_text()
    reverse = raw[raw.index("-- REVERSE (manual") :]
    assert (
        "DELETE FROM public.schema_migrations WHERE filename =" in reverse
        and "140_expert_reviews_estimand_key.sql" in reverse
    )
    assert "uq_er_pending_dag_brand" in reverse
    # The collision precondition covers EVERY row that will be pending after the
    # reverse, and is stated BEFORE the CREATE UNIQUE INDEX line -- not just
    # before the restoring UPDATE. Two pending reviews of one estimand-keyed
    # queue can share (dag_version_hash, brand) and differ only in
    # treatment/outcome: legal under uq_er_pending_estimand, forbidden under the
    # restored uq_er_pending_dag_brand. So the index creation is itself the first
    # statement that can fail, on rows nobody is restoring.
    assert "collide" in reverse or "collision" in reverse
    precondition = reverse.index("PRECONDITION")
    assert precondition < reverse.index("CREATE UNIQUE INDEX uq_er_pending_dag_brand")
    assert precondition < reverse.index(
        "UPDATE public.expert_reviews SET approval_status = 'pending'"
    )
    # ... and it names both populations, not only the rows being restored
    lowered = reverse.lower()
    assert "already pending" in lowered and "restor" in lowered
    # every added line is still a comment: the executable body is untouched
    assert "DELETE FROM" not in _sql().upper()
