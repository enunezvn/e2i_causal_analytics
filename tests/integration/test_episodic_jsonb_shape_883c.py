"""#883 read-side follow-up: episodic JSONB double-encode (writers + migration 073).

Cross-verified with the #883 learning-signals sibling (PR #887 / migration
072), which proved the same writer bug class live for
``procedural_memories.tool_sequence`` (1566/1566 string scalars) and
``ml_hpo_patterns`` (887/887): the writers run ``json.dumps(...)`` on
structured payloads before the supabase insert, and postgrest JSON-encodes
the payload itself — so the JSONB columns store JSON **string scalars**, not
objects. Verified live on THIS surface 2026-06-12 pre-fix::

    SELECT jsonb_typeof(raw_content), count(*) FROM episodic_memories GROUP BY 1;
    -- string | 628          (entities and outcome_details identical)

That double-encode is the ROOT CAUSE of the raw_content reader gap this
branch fixes: the hydration helper had to parse the string scalars back.
This change closes the loop:

* writers (``insert_episodic_memory`` + ``bulk_insert_episodic_memories``;
  ``insert_episodic_memory_with_text`` delegates) pass the dicts through so
  new rows land as real JSONB objects — RED quoted below;
* migration 073 repairs the historical rows (072 §2 loop pattern:
  per-row exception guard, NOTICE counts, idempotent re-run);
* ``hydrate_raw_content`` stays tolerant of BOTH shapes (legacy string
  scalar and repaired/new object) — proven here against the live DB, so a
  partially-repaired or skipped row can never resurrect the dropped-rows bug.

Blast-radius census (live + repo-wide, 2026-06-12): the search RPC's TABLE
shape carries none of the three columns; ``search_episodic_by_e2i_entity`` /
``get_enriched_episodic_memory`` (which select them) have zero non-module
consumers; no consumer reads episodic ``entities``/``outcome_details``
shapes; tool_composer's planner reads ``raw_content`` only off search-RPC
rows where the key is absent (-> ``{}``, unchanged).

Run with the shared-DB lock::

    flock /tmp/e2i_db_verify.lock -c \\
        'E2I_DB_INTEGRATION=1 PYTHONPATH=$PWD .venv/bin/pytest -n0 \\
         tests/integration/test_episodic_jsonb_shape_883c.py'
"""

import json
import os
import uuid

import pytest

_GATE = os.environ.get("E2I_DB_INTEGRATION") == "1"
_HAS_CREDS = bool(os.environ.get("OPENAI_API_KEY")) and bool(os.environ.get("SUPABASE_URL"))

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (_GATE and _HAS_CREDS),
        reason="faithful real-DB JSONB-shape test; set E2I_DB_INTEGRATION=1 + creds in .env",
    ),
]


def _cleanup(memory_id: str) -> None:
    from src.memory.episodic_memory import get_supabase_client

    get_supabase_client().table("episodic_memories").delete().eq("memory_id", memory_id).execute()


def _typeof(memory_id: str) -> dict:
    """jsonb_typeof for the three columns via a raw select (PostgREST returns
    the decoded value, so shape-check on the PYTHON side: dict == object,
    str == string scalar)."""
    from src.memory.episodic_memory import get_supabase_client

    row = (
        get_supabase_client()
        .table("episodic_memories")
        .select("raw_content, entities, outcome_details")
        .eq("memory_id", memory_id)
        .execute()
    ).data[0]
    return {k: type(v).__name__ for k, v in row.items()}


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_writer_stores_jsonb_objects_not_string_scalars():
    """RED pre-fix: a row written through the real path came back with
    raw_content/entities/outcome_details as ``str`` (JSON-string scalars in
    jsonb) — the writers json.dumps'd dicts that postgrest re-encoded.
    GREEN: all three land as real objects (``dict`` through PostgREST)."""
    from src.memory.episodic_memory import (
        EpisodicMemoryInput,
        insert_episodic_memory_with_text,
    )

    marker = f"883c-shape-{uuid.uuid4().hex[:12]}"
    memory_id = await insert_episodic_memory_with_text(
        memory=EpisodicMemoryInput(
            event_type="gap_analysis_completed",
            description=f"jsonb shape probe ({marker})",
            raw_content={"marker": marker, "metrics": ["trx_rate"]},
            entities={"brands": ["remibrutinib"]},
            outcome_type="success",
            outcome_details={"probe": True},
            agent_name="gap_analyzer",
        ),
        text_to_embed=f"jsonb shape probe {marker}",
        session_id=str(uuid.uuid4()),
    )
    try:
        shapes = _typeof(memory_id)
        assert shapes == {
            "raw_content": "dict",
            "entities": "dict",
            "outcome_details": "dict",
        }, (
            f"writer double-encode: columns landed as {shapes} — json.dumps'd "
            "payloads become JSON string SCALARS in jsonb (the root cause of "
            "the #883 raw_content reader gap; same class as migration 072's "
            "procedural/hpo repair)"
        )
    finally:
        _cleanup(memory_id)


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_no_reader_opaque_string_scalars_remain():
    """Census invariant the read side depends on, pinned shape-tolerantly:

    * any episodic JSONB value still stored as a string scalar must be
      PYTHON-parseable (``json.loads`` accepts the bare NaN/Infinity tokens
      Python's own ``json.dumps`` emitted; Postgres ``::jsonb`` does not) —
      i.e. ``hydrate_raw_content`` can recover its content. A string scalar
      the reader CANNOT parse would silently hydrate to ``{}`` and resurrect
      the dropped-rows bug.

    Live state behind this: migration 073 repaired 491 raw_content + 628
    entities + 628 outcome_details rows in-session; the 137 remaining
    raw_content string scalars are all model_trainer payloads whose bare-NaN
    tokens fail the Postgres cast. They stay string scalars BY DESIGN
    (codex R2: an in-SQL text rewrite cannot be made quote-aware, so the
    migration only plain-casts and skips them; the safe convergence path,
    if ever wanted, is a Python json.loads/json.dumps repair). The writers
    are fixed, so this set can only shrink — a GROWING set means a writer
    regressed to double-encoding."""
    import math

    from src.memory.episodic_memory import get_supabase_client

    client = get_supabase_client()
    rows = (
        client.table("episodic_memories")
        .select("memory_id, raw_content, entities, outcome_details")
        .limit(2000)
        .execute()
    ).data or []
    assert rows, "episodic_memories unexpectedly empty"

    opaque = []
    for r in rows:
        for c in ("raw_content", "entities", "outcome_details"):
            v = r.get(c)
            if isinstance(v, str):
                try:
                    parsed = json.loads(v)  # tolerant of NaN/Infinity tokens
                    assert isinstance(parsed, (dict, list))
                except (ValueError, AssertionError):
                    opaque.append((r["memory_id"], c))
    assert not opaque, (
        f"{len(opaque)} episodic JSONB values are string scalars the Python "
        f"reader cannot parse (e.g. {opaque[:3]}) — these hydrate to {{}} and "
        "silently drop content (writer regression or corrupt restore)"
    )

    # NaN-class detector: math.isnan must be reachable for the known 137-row
    # class (proves json.loads really is the tolerant parser we rely on).
    assert math.isnan(json.loads('{"x": NaN}')["x"])


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_hydrate_raw_content_tolerates_both_shapes():
    """The reader fix must survive BOTH shapes (the 073 repair skips rows
    whose inner text is not valid JSON, and pre-restore backups may
    reintroduce legacy rows): one LEGACY row inserted directly with a
    json.dumps'd string scalar, one row through the fixed writer (object).
    Both hydrate to dicts."""
    from src.memory.episodic_memory import (
        EpisodicMemoryInput,
        get_supabase_client,
        hydrate_raw_content,
        insert_episodic_memory_with_text,
    )

    marker = f"883c-tolerant-{uuid.uuid4().hex[:12]}"
    session_id = str(uuid.uuid4())

    # Legacy shape: direct insert, payload pre-json.dumps'd (what the old
    # writer produced). Embedding column is nullable; hydration is by PK.
    legacy_id = str(uuid.uuid4())
    get_supabase_client().table("episodic_memories").insert(
        {
            "memory_id": legacy_id,
            "session_id": session_id,
            "event_type": "gap_analysis_completed",
            "description": f"legacy string-scalar row ({marker})",
            "raw_content": json.dumps({"marker": marker, "shape": "legacy"}),
            "agent_name": "gap_analyzer",
        }
    ).execute()

    new_id = None
    try:
        new_id = await insert_episodic_memory_with_text(
            memory=EpisodicMemoryInput(
                event_type="gap_analysis_completed",
                description=f"object row ({marker})",
                raw_content={"marker": marker, "shape": "object"},
                agent_name="gap_analyzer",
            ),
            text_to_embed=f"object row {marker}",
            session_id=session_id,
        )

        hydrated = await hydrate_raw_content([{"memory_id": legacy_id}, {"memory_id": new_id}])
        by_id = {str(r["memory_id"]): r["raw_content"] for r in hydrated}
        assert by_id[legacy_id] == {"marker": marker, "shape": "legacy"}, (
            "legacy JSON-string-scalar raw_content must parse back to its dict"
        )
        assert by_id[str(new_id)] == {"marker": marker, "shape": "object"}, (
            "object-shaped raw_content must pass through unchanged"
        )
    finally:
        _cleanup(legacy_id)
        if new_id:
            _cleanup(new_id)


@pytest.mark.asyncio
@pytest.mark.timeout(120)
async def test_hydrate_parses_nan_bearing_legacy_rows():
    """Pins the live 137-row class AND the codex-R2 corruption scenario:
    Python's json.dumps emitted bare NaN tokens that Postgres ::jsonb rejects
    (which is why 073's plain cast skips these rows BY DESIGN — an in-SQL
    regex rewrite would also hit ': NaN' inside legitimate string values and
    the per-row guard could not catch it). json.loads — the hydration
    parser — accepts the bare tokens, so the reader recovers the full
    payload, INCLUDING a string value that contains the dangerous text
    verbatim."""
    import math

    from src.memory.episodic_memory import get_supabase_client, hydrate_raw_content

    marker = f"883c-nan-{uuid.uuid4().hex[:12]}"
    legacy_id = str(uuid.uuid4())
    # Exactly what the old writer produced for a NaN metric payload — plus
    # the codex-R2 regression: a STRING value containing ': NaN'/', Infinity'
    # text that any quote-unaware rewriter would corrupt.
    note = "threshold: NaN means missing, Infinity capped"
    payload_text = json.dumps({"marker": marker, "auc": float("nan"), "note": note})
    assert "NaN" in payload_text  # the non-standard token under test
    get_supabase_client().table("episodic_memories").insert(
        {
            "memory_id": legacy_id,
            "session_id": str(uuid.uuid4()),
            "event_type": "model_training_completed",
            "description": f"NaN-bearing legacy row ({marker})",
            "raw_content": payload_text,
            "agent_name": "model_trainer",
        }
    ).execute()
    try:
        hydrated = await hydrate_raw_content([{"memory_id": legacy_id}])
        rc = hydrated[0]["raw_content"]
        assert rc.get("marker") == marker, "NaN-bearing legacy row hydrated to {} — content lost"
        assert math.isnan(rc["auc"])
        assert rc["note"] == note, (
            "string content containing ': NaN' text must survive VERBATIM — "
            "the reader path is the only endorsed repair path precisely "
            "because it cannot corrupt string values"
        )
    finally:
        _cleanup(legacy_id)
