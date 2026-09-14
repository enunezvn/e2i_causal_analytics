"""Migration 141 content lock (#1991 debt 3): expert_review_versions is a TIMELINE (no unique
hash per review, a revert can repeat one), backfilled idempotently via NOT EXISTS, with grants
deliberately narrower than expert_reviews (SELECT, INSERT only for service_role)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
M = REPO / "database" / "migrations" / "141_expert_review_versions.sql"


def _sql() -> str:
    return "\n".join(
        line for line in M.read_text().splitlines() if not line.strip().startswith("--")
    )


@pytest.mark.unit
def test_creates_versions_table_with_fk_and_columns():
    s = _sql()
    assert re.search(r"CREATE TABLE IF NOT EXISTS public\.expert_review_versions", s)
    for col in (
        "version_id UUID",
        "review_id UUID NOT NULL REFERENCES public.expert_reviews(review_id)",
        "dag_version_hash VARCHAR(64) NOT NULL",
        "adjustment_set_hash VARCHAR(64)",
        "dag_structure_json JSONB",
        "query_id TEXT",
        "created_at TIMESTAMPTZ NOT NULL DEFAULT now()",
    ):
        assert col in s, col
    # Timeline semantics (#1991 debt 3 fix round): a DAG revert A -> B -> A must be able to
    # append a third row equal to the first hash, so there is no UNIQUE on (review_id,
    # dag_version_hash) -- idempotence for a repeated hash is enforced in Python
    # (expert_review_gate), not by a constraint.
    assert "UNIQUE (review_id, dag_version_hash)" not in s
    assert "ON DELETE CASCADE" in s
    assert "idx_erv_review_created" in s
    # A string snapshot got into expert_reviews once before (migration 137's fix); the diff
    # path must never see a scalar here either.
    assert "ck_erv_snapshot_object" in s


@pytest.mark.unit
def test_backfills_one_version_per_existing_review():
    s = _sql()
    assert re.search(r"INSERT INTO public\.expert_review_versions", s)
    assert "SELECT" in s and "FROM public.expert_reviews" in s
    assert re.search(r"WHERE NOT EXISTS\s*\(\s*SELECT 1 FROM public\.expert_review_versions", s)
    assert (
        "SELECT r.review_id, r.dag_version_hash, r.dag_structure_json, r.reviewer_id, r.created_at"
        in s
    )
    assert "created_at" in s and "updated_at" not in s


@pytest.mark.unit
def test_grants_are_narrower_than_expert_reviews_select_insert_only():
    s = _sql()
    assert "REVOKE ALL ON public.expert_review_versions FROM PUBLIC, anon, authenticated" in s
    assert "REVOKE ALL ON public.expert_review_versions FROM service_role" in s
    assert "GRANT SELECT, INSERT ON public.expert_review_versions TO service_role" in s


@pytest.mark.unit
def test_no_delete_and_no_own_transaction():
    s = _sql().upper()
    assert "DELETE FROM" not in s and "DROP TABLE" not in s
    assert "BEGIN;" not in s and "COMMIT;" not in s
