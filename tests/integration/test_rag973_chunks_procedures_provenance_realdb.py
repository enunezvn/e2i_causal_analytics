"""Faithful gate (issue #973): the two RAG sources that rag/004 left exempt —
rag_document_chunks and procedural_memories — now default-exclude synthetic rows
across every retrieval RPC that reads their content.

Inserts ONE real + ONE synthetic row into EACH table (identical embeddings so
cosine similarity is 1.0 and retrieval is driven solely by the is_synthetic
predicate under test), then asserts against the live docker supabase-db:

  rag_document_chunks:
    * rag_vector_search   '{}'  -> real present, synthetic absent; opt-in -> both
    * rag_fulltext_search '{}'  -> real present, synthetic absent; opt-in -> both
  procedural_memories:
    * rag_vector_search   '{}'  -> real present, synthetic absent; opt-in -> both
    * hybrid_vector_search'{}'  -> real present, synthetic absent; opt-in -> both
    * find_relevant_procedures  -> real present, synthetic absent (bare
                                    default-exclude; no opt-in path by design)

All rows are always cleaned up (finally). Faithful DB = local docker supabase-db
(same pattern as test_hybrid_search_excludes_synthetic.py).
"""

import os
import subprocess
import uuid

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("E2I_DB_INTEGRATION") != "1",
    reason="faithful docker-Supabase test; set E2I_DB_INTEGRATION=1",
)

# Same vector for row embedding AND query embedding -> cosine similarity 1.0,
# above every branch's similarity floor (0.3 rag_*, 0.5 hybrid_*, 0.6 matcher).
_VEC = "[" + ",".join(["0.1"] * 1536) + "]"


def _psql(sql: str) -> str:
    out = subprocess.run(
        ["docker", "exec", "supabase-db", "psql", "-U", "postgres", "-d", "postgres", "-tAc", sql],
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip()


def _ids(sql: str) -> set[str]:
    return {r.strip() for r in _psql(sql).splitlines() if r.strip()}


# ---------------------------------------------------------------------------
# rag_document_chunks
# ---------------------------------------------------------------------------


def _insert_chunk(doc_id: str, is_synthetic: bool) -> None:
    # document_id is the metadata id surfaced by the RPCs; chunk_id is the PK.
    _psql(
        "INSERT INTO rag_document_chunks "
        "(document_id, document_type, chunk_index, content, embedding, is_synthetic) "
        f"VALUES ('{doc_id}', 'kpi_report', 0, "
        f"'rag973 provenance chunk {doc_id}', '{_VEC}'::vector, {str(is_synthetic).lower()});"
    )


def _chunk_doc_ids(rpc_call: str) -> set[str]:
    # both rag RPCs surface document_id in metadata; filter to our source rows.
    return _ids(
        f"SELECT metadata->>'document_id' FROM {rpc_call} "
        "WHERE source_table = 'rag_document_chunks';"
    )


def test_rag_document_chunks_excluded_by_default_across_rag_rpcs():
    real_id = f"rag973-real-{uuid.uuid4()}"
    synth_id = f"rag973-synth-{uuid.uuid4()}"
    try:
        _insert_chunk(real_id, is_synthetic=False)
        _insert_chunk(synth_id, is_synthetic=True)

        for rpc in (
            f"rag_vector_search('{_VEC}'::vector, 100, '{{}}'::jsonb)",
            "rag_fulltext_search('provenance', 100, '{}'::jsonb)",
        ):
            prod = _chunk_doc_ids(rpc)
            assert real_id in prod, f"real chunk must be returned in prod mode by {rpc}"
            assert synth_id not in prod, f"synthetic chunk must NOT be returned by {rpc}"

        for rpc in (
            f"rag_vector_search('{_VEC}'::vector, 100, '{{\"include_synthetic\":\"true\"}}'::jsonb)",
            "rag_fulltext_search('provenance', 100, '{\"include_synthetic\":\"true\"}'::jsonb)",
        ):
            optin = _chunk_doc_ids(rpc)
            assert real_id in optin and synth_id in optin, (
                f"opt-in must return BOTH chunks via {rpc}"
            )
    finally:
        _psql(
            "DELETE FROM rag_document_chunks WHERE document_id IN "
            f"('{real_id}', '{synth_id}');"
        )


# ---------------------------------------------------------------------------
# procedural_memories
# ---------------------------------------------------------------------------


def _insert_procedure(name: str, is_synthetic: bool) -> str:
    pid = str(uuid.uuid4())
    _psql(
        "INSERT INTO procedural_memories "
        "(procedure_id, procedure_name, procedure_type, tool_sequence, trigger_pattern, "
        " trigger_embedding, is_active, success_count, usage_count, is_synthetic) "
        f"VALUES ('{pid}', '{name}', 'tool_sequence', '[]'::jsonb, 'rag973 trigger', "
        f"'{_VEC}'::vector, true, 1, 1, {str(is_synthetic).lower()});"
    )
    return pid


def test_procedural_memories_excluded_by_default_across_rpcs():
    real_name = f"rag973-real-{uuid.uuid4().hex[:8]}"
    synth_name = f"rag973-synth-{uuid.uuid4().hex[:8]}"
    real_pid = synth_pid = None
    try:
        real_pid = _insert_procedure(real_name, is_synthetic=False)
        synth_pid = _insert_procedure(synth_name, is_synthetic=True)

        # rag_vector_search + hybrid_vector_search surface procedure_id as `id`.
        for rpc in (
            f"rag_vector_search('{_VEC}'::vector, 200, '{{}}'::jsonb)",
            f"hybrid_vector_search('{_VEC}'::vector, 200, '{{}}'::jsonb)",
        ):
            prod = _ids(f"SELECT id FROM {rpc} WHERE source_table = 'procedural_memories';")
            assert real_pid in prod, f"real procedure must be returned in prod mode by {rpc}"
            assert synth_pid not in prod, f"synthetic procedure must NOT be returned by {rpc}"

        for rpc in (
            f"rag_vector_search('{_VEC}'::vector, 200, '{{\"include_synthetic\":\"true\"}}'::jsonb)",
            f"hybrid_vector_search('{_VEC}'::vector, 200, '{{\"include_synthetic\":\"true\"}}'::jsonb)",
        ):
            optin = _ids(f"SELECT id FROM {rpc} WHERE source_table = 'procedural_memories';")
            assert real_pid in optin and synth_pid in optin, (
                f"opt-in must return BOTH procedures via {rpc}"
            )

        # find_relevant_procedures: bare default-exclude (no opt-in path).
        matcher = _ids(
            f"SELECT procedure_id FROM find_relevant_procedures('{_VEC}'::vector, 0.3, 200);"
        )
        assert real_pid in matcher, "real procedure must be matched by find_relevant_procedures"
        assert synth_pid not in matcher, (
            "synthetic procedure must NOT be matched by find_relevant_procedures"
        )
    finally:
        ids = [p for p in (real_pid, synth_pid) if p]
        if ids:
            joined = ", ".join(f"'{p}'" for p in ids)
            _psql(f"DELETE FROM procedural_memories WHERE procedure_id IN ({joined});")
