"""#891 convergence script proof — single BEGIN..ROLLBACK transaction against
the live docker DB (self-cleaning by rollback; nothing persists).

Proves, per the issue's red-first contract:

(a) a NaN-bearing string-scalar row (the 137-row class migration 073 skipped
    BY DESIGN: bare ``NaN``/``Infinity`` tokens are valid for Python
    ``json.loads`` but rejected by ``::jsonb``) becomes a proper JSONB object
    with the non-finite floats as JSON nulls;
(b) the codex-R2 corruption payload — a string VALUE containing the literal
    text ``"threshold: NaN means missing, Infinity capped"`` — survives
    BYTE-IDENTICAL (json.loads/json.dumps is quote-aware by construction;
    the rejected in-SQL regex rewrite was not);
(c) rows already stored as proper objects are not touched at all (byte-stable
    ``raw_content::text``), even when they contain the dangerous text;
(d) idempotency: a second pass finds nothing left to converge.

Gates: ``E2I_DB_INTEGRATION=1`` plus a direct-postgres DSN in ``E2I_PG_DSN``
(the .env ``SUPABASE_DB_URL`` points at the supavisor pooler and fails with
"Tenant or user not found"; use the worker container's DSN rewritten to host
port 5433 — see the script header for the recipe). Run under the shared DB
lock::

    flock /tmp/e2i_dbtest.lock -c \\
        'E2I_DB_INTEGRATION=1 E2I_PG_DSN=postgresql://... PYTHONPATH=$PWD \\
         .venv/bin/pytest -n0 tests/integration/test_episodic_nan_convergence_891.py'
"""

import importlib.util
import json
import os
import uuid
from pathlib import Path

import pytest

_GATE = os.environ.get("E2I_DB_INTEGRATION") == "1"
_DSN = os.environ.get("E2I_PG_DSN", "")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (_GATE and _DSN),
        reason=(
            "faithful real-DB convergence proof; set E2I_DB_INTEGRATION=1 and "
            "E2I_PG_DSN (direct postgres DSN, e.g. worker SUPABASE_DB_URL "
            "rewritten to 127.0.0.1:5433 — the .env pooler URL does not work)"
        ),
    ),
]

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _PROJECT_ROOT / "scripts" / "maintenance" / "converge_episodic_nan_rows_891.py"

# The exact string value codex R2 flagged: a quote-unaware rewriter corrupts
# it; the endorsed Python repair must preserve it verbatim. Also pinned by
# tests/integration/test_episodic_jsonb_shape_883c.py.
CODEX_R2_NOTE = "threshold: NaN means missing, Infinity capped"


def _load_script():
    spec = importlib.util.spec_from_file_location("converge_episodic_nan_rows_891", _SCRIPT)
    assert spec and spec.loader, f"convergence script missing: {_SCRIPT}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.timeout(180)
def test_convergence_red_green_single_transaction():
    import psycopg2

    mod = _load_script()
    conn = psycopg2.connect(_DSN)
    try:
        conn.autocommit = False
        cur = conn.cursor()
        marker = f"891-conv-{uuid.uuid4().hex[:12]}"
        nan_id, note_id, obj_id = (str(uuid.uuid4()) for _ in range(3))

        # Exactly what the pre-#888 writer produced for NaN metric payloads:
        # json.dumps (allow_nan default True) -> bare NaN/Infinity tokens ->
        # stored as a jsonb STRING SCALAR (to_jsonb of text).
        nan_txt = json.dumps(
            {
                "marker": marker,
                "auc": float("nan"),
                "rmse": float("inf"),
                "neg": float("-inf"),
                "nested": {"vals": [1.0, float("nan")], "deep": {"x": float("inf")}},
            }
        )
        assert "NaN" in nan_txt and "Infinity" in nan_txt  # the tokens under test
        note_txt = json.dumps({"marker": marker, "auc": float("nan"), "note": CODEX_R2_NOTE})

        for mid, txt in ((nan_id, nan_txt), (note_id, note_txt)):
            cur.execute(
                "INSERT INTO episodic_memories "
                "(memory_id, event_type, description, agent_name, raw_content) "
                "VALUES (%s, 'model_training_completed', %s, 'model_trainer', "
                "to_jsonb(%s::text))",
                (mid, f"891 convergence probe ({marker})", txt),
            )

        # Control: a row already stored as a proper object — containing the
        # dangerous text in a string value — must not be touched at all.
        cur.execute(
            "INSERT INTO episodic_memories "
            "(memory_id, event_type, description, agent_name, raw_content) "
            "VALUES (%s, 'model_training_completed', %s, 'model_trainer', %s::jsonb)",
            (
                obj_id,
                f"891 convergence control ({marker})",
                json.dumps({"marker": marker, "note": CODEX_R2_NOTE, "auc": 0.81}),
            ),
        )
        cur.execute(
            "SELECT raw_content::text FROM episodic_memories WHERE memory_id = %s", (obj_id,)
        )
        obj_text_before = cur.fetchone()[0]

        # ---- first pass: converges our two probes (+ any live backlog,
        # transiently — everything rolls back) -------------------------------
        stats = mod.converge(conn)
        converged_ids = set(stats.converged_ids)
        assert nan_id in converged_ids, "NaN-bearing string-scalar row was not converged"
        assert note_id in converged_ids, "corruption-canary row was not converged"
        assert obj_id not in converged_ids, "object row must never be a candidate"

        # (a) NaN row -> proper object, non-finite floats -> JSON null,
        # finite values and structure intact.
        cur.execute(
            "SELECT jsonb_typeof(raw_content), raw_content "
            "FROM episodic_memories WHERE memory_id = %s",
            (nan_id,),
        )
        typ, rc = cur.fetchone()
        assert typ == "object"
        assert rc["marker"] == marker
        assert "auc" in rc and rc["auc"] is None
        assert rc["rmse"] is None and rc["neg"] is None
        assert rc["nested"]["vals"] == [1.0, None]
        assert rc["nested"]["deep"] == {"x": None}

        # (b) the pinned corruption payload survives byte-identical.
        cur.execute(
            "SELECT jsonb_typeof(raw_content), raw_content "
            "FROM episodic_memories WHERE memory_id = %s",
            (note_id,),
        )
        typ, rc = cur.fetchone()
        assert typ == "object"
        assert rc["note"] == CODEX_R2_NOTE
        assert rc["note"].encode() == CODEX_R2_NOTE.encode()
        assert "auc" in rc and rc["auc"] is None

        # (c) object rows byte-stable at rest.
        cur.execute(
            "SELECT raw_content::text FROM episodic_memories WHERE memory_id = %s", (obj_id,)
        )
        assert cur.fetchone()[0] == obj_text_before, "object row was rewritten — must be untouched"

        # (d) idempotency: nothing left to converge; only deliberate skips
        # (rows whose payload is not an object/array) may remain as candidates.
        stats2 = mod.converge(conn)
        assert stats2.converged == 0, (
            f"second pass converged {stats2.converged} rows — not idempotent"
        )
        assert stats2.candidates == len(stats2.skipped)
        assert not {nan_id, note_id, obj_id} & set(stats2.converged_ids)
    finally:
        conn.rollback()  # self-cleaning: probes AND transient backlog convergence vanish
        conn.close()


@pytest.mark.timeout(60)
def test_non_object_string_scalars_are_skipped_not_mutated():
    """A string scalar whose inner text parses to a NON-object (triple-encoded
    or bare scalar) is reported and left untouched — the script only writes
    payloads it can verify as JSON objects/arrays."""
    import psycopg2

    mod = _load_script()
    conn = psycopg2.connect(_DSN)
    try:
        conn.autocommit = False
        cur = conn.cursor()
        marker = f"891-skip-{uuid.uuid4().hex[:12]}"
        scalar_id = str(uuid.uuid4())
        # Inner text is valid JSON but a bare scalar, not an object.
        cur.execute(
            "INSERT INTO episodic_memories "
            "(memory_id, event_type, description, agent_name, raw_content) "
            "VALUES (%s, 'model_training_completed', %s, 'model_trainer', "
            "to_jsonb(%s::text))",
            (scalar_id, f"891 skip probe ({marker})", json.dumps(f"just a string {marker}")),
        )
        cur.execute(
            "SELECT raw_content::text FROM episodic_memories WHERE memory_id = %s", (scalar_id,)
        )
        before = cur.fetchone()[0]

        stats = mod.converge(conn)
        assert scalar_id not in set(stats.converged_ids)
        assert any(mid == scalar_id for mid, _col, _reason in stats.skipped)

        cur.execute(
            "SELECT raw_content::text FROM episodic_memories WHERE memory_id = %s", (scalar_id,)
        )
        assert cur.fetchone()[0] == before
    finally:
        conn.rollback()
        conn.close()
