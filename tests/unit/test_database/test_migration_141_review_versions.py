"""Migration 141 content lock (#1991 debt 3): expert_review_versions table, one backfilled
version per existing review, grants mirroring expert_reviews."""

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
    assert "UNIQUE (review_id, dag_version_hash)" in s


@pytest.mark.unit
def test_backfills_one_version_per_existing_review():
    s = _sql()
    assert re.search(r"INSERT INTO public\.expert_review_versions", s)
    assert "SELECT" in s and "FROM public.expert_reviews" in s
    assert "ON CONFLICT (review_id, dag_version_hash) DO NOTHING" in s
    assert "created_at" in s and "updated_at" not in s


@pytest.mark.unit
def test_grants_match_expert_reviews_pattern():
    s = _sql()
    assert "REVOKE ALL ON public.expert_review_versions FROM PUBLIC, anon, authenticated" in s
    assert "GRANT SELECT, INSERT ON public.expert_review_versions TO service_role" in s


@pytest.mark.unit
def test_no_delete_and_no_own_transaction():
    s = _sql().upper()
    assert "DELETE FROM" not in s and "DROP TABLE" not in s
    assert "BEGIN;" not in s and "COMMIT;" not in s
