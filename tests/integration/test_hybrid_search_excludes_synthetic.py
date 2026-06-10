"""Faithful gate (Shard 07 R13): hybrid_vector_search default-excludes synthetic
episodic_memories rows; opt in via filters->>'include_synthetic'='true'.

Inserts ONE synthetic + ONE real episodic_memories row (agent_name='corpus_ingestion')
with an IDENTICAL embedding, then calls migration 044's hybrid_vector_search against
the live docker supabase-db:
  - prod mode  (filters '{}')                     -> real id returned, synthetic NOT.
  - opt-in     (filters '{"include_synthetic":"true"}') -> BOTH returned.
Both rows are always cleaned up (finally).

Faithful DB = local docker `supabase-db` (same pattern as the M1 provenance test).
"""

import os
import subprocess
import uuid

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("E2I_DB_INTEGRATION") != "1",
    reason="faithful docker-Supabase test; set E2I_DB_INTEGRATION=1",
)

# A deterministic non-zero 1536-dim unit-ish vector. Using the SAME vector for the
# row embedding AND the query embedding gives cosine similarity 1.0 (> the 0.5
# floor), so retrieval is driven solely by the is_synthetic predicate under test.
_VEC = "[" + ",".join(["0.1"] * 1536) + "]"


def _psql(sql: str) -> str:
    out = subprocess.run(
        ["docker", "exec", "supabase-db", "psql", "-U", "postgres", "-d", "postgres", "-tAc", sql],
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip()


def _insert_row(memory_id: str, is_synthetic: bool, label: str) -> None:
    _psql(
        "INSERT INTO episodic_memories "
        "(memory_id, event_type, description, agent_name, embedding, is_synthetic) "
        f"VALUES ('{memory_id}', 'system_event', "
        f"'provenance gate {label} {memory_id}', 'corpus_ingestion', "
        f"'{_VEC}'::vector, {str(is_synthetic).lower()});"
    )


def _search_ids(include_synthetic: bool) -> set[str]:
    filt = '{"include_synthetic":"true"}' if include_synthetic else "{}"
    # cast jsonb literal inside SQL; single-quote-escape the json braces.
    rows = _psql(f"SELECT id FROM hybrid_vector_search('{_VEC}'::vector, 50, '{filt}'::jsonb);")
    return {r.strip() for r in rows.splitlines() if r.strip()}


def test_hybrid_vector_search_excludes_synthetic_episodic_by_default():
    real_id = str(uuid.uuid4())
    synth_id = str(uuid.uuid4())
    try:
        _insert_row(real_id, is_synthetic=False, label="real")
        _insert_row(synth_id, is_synthetic=True, label="synthetic")

        # prod mode: real present, synthetic absent
        prod = _search_ids(include_synthetic=False)
        assert real_id in prod, "real episodic row must be returned in prod mode"
        assert synth_id not in prod, "synthetic episodic row must NOT be returned in prod mode"

        # opt-in: both present (flip)
        optin = _search_ids(include_synthetic=True)
        assert real_id in optin
        assert synth_id in optin, "synthetic row must be returned when include_synthetic=true"
    finally:
        _psql(f"DELETE FROM episodic_memories WHERE memory_id IN ('{real_id}', '{synth_id}');")
