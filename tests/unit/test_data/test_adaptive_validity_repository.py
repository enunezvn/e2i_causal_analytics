"""Phase 7.1 forcing tests for adaptive_validity_repository.

Plan reference: ``.claude/plans/causal_role_propagation_FINAL.md`` §7.1 +
v3 §7.3.

The repository module is the **single SQL source-of-truth** for active
``RoleAttribution`` rows. It reads ``adaptive_validity_verdicts``
columns ``causal_role_final`` / ``causal_role_source`` (Migration 041)
plus ``evaluator_audit->>'satisfied'`` and produces typed
``RoleAttribution`` rows that downstream tools (tool_composer, ML
foundation) can consume.

**Falsifiability**: each test pins exact SQL text fragments and rows the
repository must yield. The repository may not silently widen the result
set (else the tool composer would auto-populate wrong confounders) and
may not silently drop the ``evaluator_satisfied=true`` filter (else
unverified LLM roles leak past the C1 trust-gate).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from src.data.adaptive_validity_repository import (
    query_active_role_attributions,
)
from src.data.role_attribution import RoleAttribution


def _make_conn_returning(rows: list[tuple[Any, ...]]) -> MagicMock:
    """Build a MagicMock psycopg-style connection that yields ``rows``.

    Mirrors the cursor protocol used in ``scripts/query_audit_trail.py``:
    ``with conn.cursor() as cur: cur.execute(...); cur.fetchall()``.
    """
    cur = MagicMock()
    cur.fetchall.return_value = rows
    cur.execute = MagicMock()
    # Context-manager: __enter__ returns cur, __exit__ returns None
    cur.__enter__ = MagicMock(return_value=cur)
    cur.__exit__ = MagicMock(return_value=None)

    conn = MagicMock()
    conn.cursor.return_value = cur
    return conn


# ----------------------------------------------------------------------------
# Case 1: happy path — manifest + llm-satisfied rows returned;
#         evaluator_satisfied flag forwarded into the typed dict.
# ----------------------------------------------------------------------------
def test_query_returns_typed_role_attributions_for_satisfied_rows() -> None:
    """Falsifiability: revert the SQL ``::boolean`` cast to plain
    ``->>'satisfied'`` — the LLM row (text 'true') still leaks through
    (postgres-compares-text). The assertion ``len(result) == 2`` still
    passes, but rotating evaluator_audit to ``{"satisfied": "false"}``
    in the fixture row 1 would START passing the satisfied filter
    under the buggy SQL, defeating C1. Pinning the SQL text below is
    therefore the load-bearing falsifiability anchor.
    """
    rows = [
        # (feature, causal_role_final, causal_role_source, evaluator_audit,
        #  verdict)
        (
            "age",
            "confounder",
            "manifest",
            None,  # manifest sources have no evaluator audit
            {"evaluator_model": "n/a"},
        ),
        (
            "weight",
            "instrument",
            "llm",
            {"satisfied": True},
            {"evaluator_model": "anthropic/claude-haiku-4-5"},
        ),
    ]
    conn = _make_conn_returning(rows)

    result = query_active_role_attributions("exp-001", conn=conn)

    assert len(result) == 2
    assert result[0]["feature"] == "age"
    assert result[0]["causal_role"] == "confounder"
    assert result[0]["source"] == "manifest"
    assert result[0]["evaluator_satisfied"] is True
    assert result[0]["evaluator_model"] == "n/a"

    assert result[1]["feature"] == "weight"
    assert result[1]["causal_role"] == "instrument"
    assert result[1]["source"] == "llm"
    assert result[1]["evaluator_satisfied"] is True
    assert result[1]["evaluator_model"] == "anthropic/claude-haiku-4-5"


# ----------------------------------------------------------------------------
# Case 2: SQL contains the ``::boolean`` cast (codex-2 fix).
# ----------------------------------------------------------------------------
def test_query_sql_uses_boolean_cast_for_satisfied_flag() -> None:
    """Falsifiability: drop ``::boolean`` from the SQL — this assertion
    trips. The plan §7.1 explicitly mandates the cast because
    ``->>`` returns TEXT in postgres; comparing text to a boolean
    literal silently mis-filters (``'true' = true`` is a type error
    in strict mode, and ``'true' = 'true'`` matches but a non-canonical
    string like ``'TRUE'`` slips through).
    """
    rows: list[tuple[Any, ...]] = []
    conn = _make_conn_returning(rows)

    query_active_role_attributions("exp-001", conn=conn)

    cur = conn.cursor.return_value
    assert cur.execute.called, "cursor.execute must be called"
    sql_arg = cur.execute.call_args[0][0]
    # Plan §7.1 anchor — the explicit cast.
    assert "(evaluator_audit->>'satisfied')::boolean" in sql_arg, (
        f"SQL must contain the explicit boolean cast per plan §7.1; got: {sql_arg!r}"
    )


# ----------------------------------------------------------------------------
# Case 3: only_evaluator_satisfied=False relaxes the gate so all rows
#         come back (audit / debug use cases).
# ----------------------------------------------------------------------------
def test_query_without_satisfied_gate_omits_satisfied_filter() -> None:
    """Falsifiability: if the implementation always appends the
    satisfied filter regardless of ``only_evaluator_satisfied``, this
    test trips on the SQL text inspection. The plan §7.1 makes the
    gate a keyword arg with True default — overrideable for audit.
    """
    rows: list[tuple[Any, ...]] = []
    conn = _make_conn_returning(rows)

    query_active_role_attributions("exp-001", only_evaluator_satisfied=False, conn=conn)

    cur = conn.cursor.return_value
    sql_arg = cur.execute.call_args[0][0]
    assert "satisfied" not in sql_arg, (
        f"When only_evaluator_satisfied=False, the SQL must not "
        f"filter on evaluator_audit.satisfied; got: {sql_arg!r}"
    )


# ----------------------------------------------------------------------------
# Case 4: experiment_id is bound as a parameter (no SQL injection).
# ----------------------------------------------------------------------------
def test_query_binds_experiment_id_as_parameter() -> None:
    """Falsifiability: f-string-injecting experiment_id (e.g.
    ``f"... WHERE experiment_id = '{experiment_id}'"``) would pass
    Case 1+2+3 because the SQL still reads correctly, but trips here
    on the params tuple inspection.
    """
    rows: list[tuple[Any, ...]] = []
    conn = _make_conn_returning(rows)

    query_active_role_attributions("exp-foo'; DROP TABLE x; --", conn=conn)

    cur = conn.cursor.return_value
    call_args = cur.execute.call_args
    # cur.execute(sql, params) — params must include the experiment_id.
    assert len(call_args[0]) == 2, "execute must pass params separately"
    params = call_args[0][1]
    assert "exp-foo'; DROP TABLE x; --" in tuple(params), (
        f"experiment_id must be in the params tuple; got: {params!r}"
    )


# ----------------------------------------------------------------------------
# Case 5: rows that fail the RoleAttribution invariants are skipped
#         defensively (e.g. NULL causal_role_final — should not occur
#         given the copresence CHECK in migration 041, but defense in
#         depth).
# ----------------------------------------------------------------------------
def test_query_skips_rows_with_null_causal_role_final() -> None:
    """Falsifiability: removing the NULL guard means a row with
    ``causal_role_final=None`` produces a malformed RoleAttribution
    (``causal_role=None`` violating the typed-dict shape). The plan
    §7.1 doesn't say "skip nulls" verbatim but the copresence CHECK
    (migration 041 §4) already forbids it; defensive skip preserves
    the invariant under hypothetical schema drift.
    """
    rows = [
        (None, None, None, None, {}),  # all-null — should be skipped
        ("age", "confounder", "manifest", None, {"evaluator_model": "n/a"}),
    ]
    conn = _make_conn_returning(rows)

    result = query_active_role_attributions("exp-001", conn=conn)

    assert len(result) == 1
    assert result[0]["feature"] == "age"


# ----------------------------------------------------------------------------
# Case 6: return type is list[RoleAttribution] (TypedDict invariants).
# ----------------------------------------------------------------------------
def test_returned_dicts_satisfy_role_attribution_shape() -> None:
    """Falsifiability: returning bare dicts without the five required
    keys means downstream consumers (Phase 7.2 tool composer) raise
    KeyError. Tested explicitly against the TypedDict's required keys.
    """
    rows = [
        (
            "treatment_age",
            "confounder",
            "llm",
            {"satisfied": True},
            {"evaluator_model": "haiku"},
        ),
    ]
    conn = _make_conn_returning(rows)

    result = query_active_role_attributions("exp-001", conn=conn)

    required_keys = set(RoleAttribution.__required_keys__)
    assert required_keys <= set(result[0].keys()), (
        f"Returned dict missing required RoleAttribution keys. "
        f"Required: {required_keys}, got: {set(result[0].keys())}"
    )
