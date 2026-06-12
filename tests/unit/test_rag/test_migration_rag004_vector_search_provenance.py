"""CI-collectable regression guard for migration rag/004 (issue #896).

Mirrors the offline-artifact-guard pattern of
test_migration_043_corpus_hybrid_search.py: the faithful live SQL proof for
#896 runs against the docker DB (transaction-rollback, quoted in the PR) and is
NOT collected by CI, so this guard ensures the migration ARTIFACT that deploy
applies actually carries the three fixes:

1. provenance: COALESCE'd is_synthetic default-exclusion on the
   episodic_memories branch of rag_vector_search, with the platform-wide
   ``filters->>'include_synthetic'`` opt-in (mirrors memory/044+045 semantics);
2. case-insensitive brand/region matching (mirrors memory/043);
3. plural/singular filter-key reconciliation: the RPC honors BOTH the singular
   text keys (``brand``/``region``) and the plural list keys
   (``brands``/``regions``) emitted by _build_filters_from_entities, via the
   rag_filter_values normalizer.

It guards the deploy input, not behavior; behavior is proven by the live
transaction-rollback proof.
"""

from pathlib import Path

import pytest

MIGRATION = (
    Path(__file__).resolve().parents[3]
    / "database"
    / "rag"
    / "004_rag_vector_search_provenance.sql"
)


@pytest.fixture(scope="module")
def sql() -> str:
    assert MIGRATION.exists(), f"migration rag/004 missing at {MIGRATION}"
    return MIGRATION.read_text()


def test_replaces_rag_vector_search_and_creates_normalizer(sql: str):
    assert "CREATE OR REPLACE FUNCTION rag_vector_search" in sql
    assert "CREATE OR REPLACE FUNCTION rag_filter_values" in sql


def test_provenance_default_exclusion_on_episodic_branch(sql: str):
    # Exact 044/045 predicate shape: filters-jsonb opt-in (NOT a new SQL
    # parameter -- CREATE OR REPLACE cannot extend a signature, and a second
    # overload would break PostgREST RPC resolution), NULL-safe COALESCE on
    # the column side so a legacy NULL row reads as real and is retained.
    assert (
        "COALESCE(filters->>'include_synthetic','false') = 'true'"
        " OR COALESCE(em.is_synthetic, false) = false" in sql
    ), "episodic branch must carry the 045-style NULL-safe default-exclusion"


def test_provenance_predicate_scoped_to_episodic_only(sql: str):
    # rag_document_chunks and procedural_memories carry NO is_synthetic column
    # (structurally exempt; verified against migrations 063/069 + live \\d).
    # The predicate must reference em.is_synthetic and nothing else.
    assert "dc.is_synthetic" not in sql
    assert "pm.is_synthetic" not in sql


def test_case_insensitive_plural_aware_brand_region(sql: str):
    # Both filterable branches (rag_document_chunks + episodic_memories) must
    # use the lowercased, plural-aware normalized arrays.
    for alias in ("dc", "em"):
        assert f"lower({alias}.brand) = ANY(v_brands)" in sql, (
            f"{alias} branch must match brand case-insensitively against the "
            "normalized filter array"
        )
        assert f"lower({alias}.region) = ANY(v_regions)" in sql
    # the arrays must come from the singular+plural normalizer
    assert "rag_filter_values(filters, 'brand', 'brands')" in sql
    assert "rag_filter_values(filters, 'region', 'regions')" in sql


def test_normalizer_lowercases_and_handles_both_shapes(sql: str):
    assert "jsonb_typeof" in sql, "normalizer must type-dispatch string vs array"
    assert "jsonb_array_elements_text" in sql


def test_grants_preserved(sql: str):
    assert "GRANT EXECUTE ON FUNCTION rag_vector_search TO authenticated" in sql
    assert "GRANT EXECUTE ON FUNCTION rag_filter_values" in sql
