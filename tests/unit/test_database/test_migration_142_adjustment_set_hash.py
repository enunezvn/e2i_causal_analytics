"""Migration 142 content lock (#1991 debt 3, codex round 2): expert_reviews carries the
ADJUSTMENT-SET half of its current version identity.

The review row previously held only ``dag_version_hash``. Every guard keyed on the row --
the resolution filter (``submit_review``), the assessment persist guard
(``update_agent_assessment``) and the advance's compare-and-set (``advance_review``) --
therefore matched on half an identity, and an ADJUSTMENT-ONLY advance (same DAG, different
adjustment set) slipped past all three. This migration gives the row the other half.

Additive only, by design: NULL means "unknown adjustment set", which is what a pre-142 row,
a row minted by the OLD api image during the deploy window, and a mint without a structure
all honestly are. No default, no constraint, no backfill -- there is nothing on
``expert_reviews`` to derive the hash FROM (migration 141's backfilled version rows are NULL
for exactly the same reason).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
M = REPO / "database" / "migrations" / "142_expert_reviews_adjustment_set_hash.sql"


def _sql() -> str:
    """The executable body: every comment line stripped, so a claim asserted here
    cannot be satisfied by prose in the header."""
    return "\n".join(
        line for line in M.read_text().splitlines() if not line.strip().startswith("--")
    )


@pytest.mark.unit
def test_adds_the_adjustment_set_hash_column_matching_dag_version_hash():
    s = _sql()
    # Same type as the column it partners with: expert_reviews.dag_version_hash is
    # character varying(64) (measured live 2026-09-14), and both halves of one
    # identity holding different types would be an invitation to a silent cast.
    assert re.search(
        r"ALTER TABLE public\.expert_reviews\s+ADD COLUMN IF NOT EXISTS adjustment_set_hash "
        r"VARCHAR\(64\)",
        s,
    )
    assert "COMMENT ON COLUMN public.expert_reviews.adjustment_set_hash" in s


@pytest.mark.unit
def test_nullable_with_no_default_no_constraint_and_no_backfill():
    """NULL is the honest value for "unknown", so nothing may manufacture one."""
    s = _sql()
    assert "NOT NULL" not in s
    assert "DEFAULT" not in s.upper()
    assert "CHECK" not in s.upper()
    # No backfill: there is no source column on expert_reviews to derive it from.
    assert "UPDATE public.expert_reviews" not in s
    assert "INSERT INTO" not in s.upper()


@pytest.mark.unit
def test_additive_only_so_the_old_image_keeps_writing_through_the_deploy_window():
    """The 136/140 rule: migrations run BEFORE the image flip, so the column must be
    invisible to the old writer -- nothing dropped, renamed or made required."""
    s = _sql().upper()
    assert "DROP COLUMN" not in s
    assert "DROP TABLE" not in s
    assert "RENAME" not in s
    assert "DELETE FROM" not in s
    # run_migrations.sh wraps the file in --single-transaction and appends the ledger row.
    assert "BEGIN;" not in s and "COMMIT;" not in s


@pytest.mark.unit
def test_reverse_instructions_drop_the_column_and_the_ledger_row():
    """141's lesson: run_migrations.sh skips any file already in schema_migrations, so a
    reverse that leaves the ledger row makes a later re-apply silently no-op."""
    header = M.read_text()
    assert "ALTER TABLE public.expert_reviews DROP COLUMN adjustment_set_hash;" in header, (
        "the REVERSE must name the exact DROP COLUMN statement"
    )
    assert "DELETE FROM public.schema_migrations" in header
    assert "'142_expert_reviews_adjustment_set_hash.sql'" in header
    assert "re-apply" in header, "the REVERSE must say WHY the ledger row has to go"


@pytest.mark.unit
def test_header_records_why_the_row_needs_the_whole_identity():
    header = M.read_text()
    for claim in ("WHAT:", "WHY", "SAFETY:", "REVERSE"):
        assert claim in header, claim
    # The three guards this column exists to complete -- named, so a later reader cannot
    # mistake it for a denormalised copy of the version timeline's column.
    for guard in ("submit_review", "update_agent_assessment", "advance_review"):
        assert guard in header, guard
