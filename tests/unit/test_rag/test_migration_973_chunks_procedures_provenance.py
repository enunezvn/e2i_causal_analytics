"""CI-collectable regression guard for issue #973 — close the provenance gap that
rag/004 left: rag_document_chunks and procedural_memories had no is_synthetic
column, so they were structurally EXEMPT from the default-exclude predicate that
rag/004 applied to the episodic_memories branch.

This fix adds the column to BOTH tables and extends the SAME platform-wide
default-exclude predicate (NULL-safe COALESCE, ``filters->>'include_synthetic'``
opt-in) to EVERY retrieval RPC that reads their content directly:

  * database/rag/005   -> rag_vector_search (dc + pm branches), rag_fulltext_search (dc)
  * database/memory/047-> hybrid_vector_search (pm branch), find_relevant_procedures (pm)

(find_similar_documents and rag_hybrid_search DELEGATE to rag_vector_search /
rag_fulltext_search, so they inherit the fix; get_search_stats is a counts-only
surface and is intentionally out of scope.)

This guard checks the migration ARTIFACTS that deploy applies; faithful behavior
is proven by the live docker-DB integration test (test_rag973_*_realdb.py).

Supersedes the exemption asserted by
test_migration_rag004_vector_search_provenance.py::test_provenance_predicate_scoped_to_episodic_only
(that test still passes — it guards the UNCHANGED rag/004 file; #973 lives in
rag/005 + memory/047).
"""

from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[3]
RAG_005 = _ROOT / "database" / "rag" / "005_chunks_synthetic_provenance.sql"
MEM_047 = _ROOT / "database" / "memory" / "047_procedural_synthetic_provenance.sql"


# The platform default-exclude predicate, parameterised by table alias. Mirrors
# the exact 045 / rag-004 shape: filters-jsonb opt-in (NOT a new SQL parameter,
# so CREATE OR REPLACE keeps the signature and PostgREST RPC resolution stable),
# NULL-safe COALESCE on the column side so a legacy NULL row reads as real.
def _optin_predicate(alias: str) -> str:
    return (
        "COALESCE(filters->>'include_synthetic','false') = 'true'"
        f" OR COALESCE({alias}.is_synthetic, false) = false"
    )


@pytest.fixture(scope="module")
def rag005() -> str:
    assert RAG_005.exists(), f"migration rag/005 missing at {RAG_005}"
    return RAG_005.read_text()


@pytest.fixture(scope="module")
def mem047() -> str:
    assert MEM_047.exists(), f"migration memory/047 missing at {MEM_047}"
    return MEM_047.read_text()


# --- rag/005: rag_document_chunks column + rag_* RPC predicates -----------------


def test_rag005_adds_is_synthetic_column_to_chunks(rag005: str):
    assert "ALTER TABLE rag_document_chunks" in rag005
    assert "is_synthetic BOOLEAN NOT NULL DEFAULT false" in rag005


def test_rag005_replaces_both_rag_search_functions(rag005: str):
    assert "CREATE OR REPLACE FUNCTION rag_vector_search" in rag005
    assert "CREATE OR REPLACE FUNCTION rag_fulltext_search" in rag005


def test_rag005_default_exclude_on_chunks_branch_in_both_rpcs(rag005: str):
    # rag_document_chunks is searched by BOTH rag_vector_search and
    # rag_fulltext_search -> the dc predicate must appear >= 2 times.
    assert rag005.count(_optin_predicate("dc")) >= 2, (
        "rag_document_chunks default-exclude predicate must be present in BOTH "
        "the vector and fulltext RPCs"
    )


def test_rag005_default_exclude_on_procedural_branch_in_vector_rpc(rag005: str):
    # rag_vector_search also unions procedural_memories -> pm predicate present.
    assert _optin_predicate("pm") in rag005


def test_rag005_preserves_grants(rag005: str):
    assert "GRANT EXECUTE ON FUNCTION rag_vector_search TO authenticated" in rag005
    assert "GRANT EXECUTE ON FUNCTION rag_fulltext_search TO authenticated" in rag005


# --- memory/047: procedural_memories column + hybrid + matcher predicates -------


def test_mem047_adds_is_synthetic_column_to_procedures(mem047: str):
    assert "ALTER TABLE procedural_memories" in mem047
    assert "is_synthetic BOOLEAN NOT NULL DEFAULT false" in mem047


def test_mem047_replaces_hybrid_and_matcher_functions(mem047: str):
    assert "CREATE OR REPLACE FUNCTION hybrid_vector_search" in mem047
    assert "CREATE OR REPLACE FUNCTION find_relevant_procedures" in mem047


def test_mem047_default_exclude_on_hybrid_procedural_branch(mem047: str):
    # hybrid_vector_search has a filters jsonb arg -> full opt-in predicate.
    assert _optin_predicate("pm") in mem047


def test_mem047_default_exclude_on_matcher(mem047: str):
    # find_relevant_procedures has a TYPED signature (no filters jsonb) so it
    # carries a bare default-exclude (no opt-in path; internal matcher, no caller
    # passes include_synthetic). Synthetic procedures must never surface.
    assert "COALESCE(pm.is_synthetic, false) = false" in mem047


def test_mem047_preserves_grants(mem047: str):
    assert "GRANT EXECUTE ON FUNCTION hybrid_vector_search TO authenticated" in mem047
    assert "GRANT EXECUTE ON FUNCTION find_relevant_procedures" in mem047
