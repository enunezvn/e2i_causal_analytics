"""Red-first test for the RPC-vs-DDL column guard (audit 2026-06-05, Rec 5 / F1).

This is the static check that would have caught the broken `016` RPC at commit
time. It runs against the REAL schema files — no mocks, no DB — so a passing
test is a real result: the guard genuinely flags the 7 phantom columns the
`016` `search_similar_conversations` / `get_conversations_with_feedback`
functions reference on `cognitive_cycles`, columns that are absent from the
`cognitive_cycles` DDL.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_GUARD = _REPO_ROOT / "scripts" / "ci" / "rpc_ddl_column_guard.py"
_DB_DIR = _REPO_ROOT / "database"


def _load_guard():
    spec = importlib.util.spec_from_file_location("rpc_ddl_column_guard", _GUARD)
    assert spec and spec.loader, f"cannot load guard at {_GUARD}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_guard_flags_the_016_phantom_columns() -> None:
    """The guard must report every phantom `cc.<col>` the 016 RPCs reference."""
    guard = _load_guard()
    findings = guard.find_phantom_column_references(_DB_DIR)
    # Map: (function, column) pairs the guard flagged on cognitive_cycles.
    flagged = {
        (f["function"], f["column"])
        for f in findings
        if f["table"] == "cognitive_cycles"
    }
    # The 7 columns the 016 RPCs reference but cognitive_cycles does not define.
    expected_phantom = {
        "agent_response",
        "response_type",
        "feedback_type",
        "feedback_text",
        "feedback_score",
        "feedback_at",
        "created_at",
    }
    flagged_cols = {col for (_fn, col) in flagged}
    missing = expected_phantom - flagged_cols
    assert not missing, (
        f"guard failed to flag phantom cognitive_cycles columns: {sorted(missing)}; "
        f"it flagged: {sorted(flagged_cols)}"
    )


def test_guard_does_not_flag_real_columns() -> None:
    """The guard must NOT report columns that genuinely exist on their table.

    `query_embedding`, `cycle_id`, `session_id`, `user_id`, `user_query`,
    `detected_intent`, `detected_entities` ARE real cognitive_cycles columns the
    016 RPCs also reference; the guard must leave them alone (no false positives).
    """
    guard = _load_guard()
    findings = guard.find_phantom_column_references(_DB_DIR)
    flagged_cols = {
        f["column"] for f in findings if f["table"] == "cognitive_cycles"
    }
    real_cols = {
        "cycle_id",
        "session_id",
        "user_id",
        "user_query",
        "query_embedding",
        "detected_intent",
        "detected_entities",
    }
    false_positives = real_cols & flagged_cols
    assert not false_positives, (
        f"guard false-positived on real cognitive_cycles columns: {sorted(false_positives)}"
    )
