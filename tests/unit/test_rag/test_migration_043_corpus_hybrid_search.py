"""CI-collectable regression guard for migration 043 (audit F2/F3a/F4).

tests/rag/ (the faithful live gates in test_hybrid_corpus_retrieval.py) is NOT
collected by any CI lane, so this offline guard ensures the migration ARTIFACT
that deploy applies actually carries the fix: case-insensitive episodic brand/
region matching in the dense RPC (F3a), an episodic_memories branch in the
sparse RPC (F2), and content=em.description in that branch so RRF dedup collapses
the dense+sparse corpus rows (F4). It guards the deploy input, not behavior;
behavior is proven by the live gates.
"""

from pathlib import Path

import pytest

MIGRATION = (
    Path(__file__).resolve().parents[3]
    / "database"
    / "memory"
    / "043_hybrid_search_operational_corpus.sql"
)


@pytest.fixture(scope="module")
def sql() -> str:
    assert MIGRATION.exists(), f"migration 043 missing at {MIGRATION}"
    return MIGRATION.read_text()


def test_replaces_both_hybrid_functions(sql: str):
    assert "CREATE OR REPLACE FUNCTION hybrid_vector_search" in sql
    assert "CREATE OR REPLACE FUNCTION hybrid_fulltext_search" in sql


def test_f3a_case_insensitive_brand_and_region(sql: str):
    # both legs must lower()-normalize brand and region
    assert sql.count("lower(em.brand) = lower(filters->>'brand')") >= 2, (
        "case-insensitive brand match must appear in BOTH dense and sparse legs"
    )
    assert sql.count("lower(em.region) = lower(filters->>'region')") >= 2


def test_f2_sparse_episodic_branch_present(sql: str):
    # the sparse RPC must query episodic_memories via its search_text tsvector
    assert "em.search_text @@" in sql, "sparse leg must search episodic search_text"
    assert "'episodic_memories'::text as source_table" in sql
    # OR-combined query (F2 terse-corpus vs verbose-query AND-mismatch fix)
    assert "v_or_query" in sql


def test_f4_sparse_content_matches_dense_for_dedup(sql: str):
    # content must be em.description (NOT the tsvector) so dedup_key collapses
    # the dense+sparse corpus rows into one fused key.
    assert sql.count("em.description as content") >= 2, (
        "both dense and sparse episodic branches must return em.description as "
        "content for cross-leg RRF dedup"
    )
