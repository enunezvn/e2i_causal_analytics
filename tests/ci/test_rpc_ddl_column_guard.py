"""Tests for the RPC-vs-DDL column guard (audit 2026-06-05, Rec 5 / F1).

The capability checks use a SELF-CONTAINED synthetic schema fixture written to a
tmp dir — real parsing, no mocks, no DB, and independent of repo state. A final
check runs the guard against the REAL `database/` tree and asserts it is clean
(no phantom references), which is true once the `016` RPCs are retired by
migration 031 (their definitions are excluded as dropped functions).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_GUARD = _REPO_ROOT / "scripts" / "ci" / "rpc_ddl_column_guard.py"


def _load_guard():
    spec = importlib.util.spec_from_file_location("rpc_ddl_column_guard", _GUARD)
    assert spec and spec.loader, f"cannot load guard at {_GUARD}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_GOOD_TABLE = """
CREATE TABLE IF NOT EXISTS foo (
    id uuid PRIMARY KEY,
    name text,
    score float
);
ALTER TABLE foo ADD COLUMN IF NOT EXISTS extra_a int, ADD COLUMN IF NOT EXISTS extra_b int;
"""

_BAD_FUNCTION = """
CREATE OR REPLACE FUNCTION search_foo(q text)
RETURNS TABLE (id uuid, name text) LANGUAGE plpgsql AS $$
BEGIN
    RETURN QUERY
    SELECT f.id, f.name, f.ghost_col, f.extra_a
    FROM foo f
    WHERE f.name = q;
END;
$$;
"""


def test_guard_flags_phantom_column_but_not_real_ones(tmp_path: Path) -> None:
    """Synthetic: f.ghost_col (absent) is flagged; real cols are not.

    `extra_a` is added via a MULTI-COLUMN ALTER — it must be recognised as real
    (regression guard for the multi-column-ALTER parser fix).
    """
    db = tmp_path / "database"
    db.mkdir()
    (db / "001_schema.sql").write_text(_GOOD_TABLE)
    (db / "002_func.sql").write_text(_BAD_FUNCTION)
    guard = _load_guard()
    findings = guard.find_phantom_column_references(db)
    flagged = {(f["function"], f["column"]) for f in findings}
    assert ("search_foo", "ghost_col") in flagged, (
        f"guard failed to flag phantom column f.ghost_col; flagged={flagged}"
    )
    real_cols = {"id", "name", "extra_a", "extra_b"}
    false_pos = {col for (_fn, col) in flagged if col in real_cols}
    assert not false_pos, f"guard false-positived on real columns: {false_pos}"


def test_guard_excludes_functions_dropped_by_a_later_migration(tmp_path: Path) -> None:
    """A function dropped by a later migration must not be flagged (016 pattern)."""
    db = tmp_path / "database"
    db.mkdir()
    (db / "001_schema.sql").write_text(_GOOD_TABLE)
    (db / "002_func.sql").write_text(_BAD_FUNCTION)
    # Retire it the same way migration 031 does (pg_proc proname IN (...) DO-block).
    (db / "003_retire.sql").write_text(
        "DO $$ DECLARE r record; BEGIN "
        "FOR r IN SELECT 'DROP FUNCTION IF EXISTS ' || p.oid::regprocedure AS stmt "
        "FROM pg_proc p WHERE p.proname IN ('search_foo') LOOP EXECUTE r.stmt; END LOOP; END $$;"
    )
    guard = _load_guard()
    findings = guard.find_phantom_column_references(db)
    assert not findings, (
        f"guard flagged a function retired by a later migration: {findings}"
    )


def test_real_schema_is_blocking_clean() -> None:
    """Against the REAL database/ tree the guard must report ZERO phantom refs.

    True post-retirement: migration 031 drops the 016 RPCs, so their historical
    definitions are excluded. A non-zero result here means a live RPC references
    a column its table does not define — exactly the 016-class defect.
    """
    guard = _load_guard()
    findings = guard.find_phantom_column_references(_REPO_ROOT / "database")
    assert not findings, (
        "real schema has phantom RPC column references (016-class defect):\n"
        + "\n".join(
            f"  {f['file']}: {f['function']}() -> {f['alias']}.{f['column']} "
            f"absent from '{f['table']}'"
            for f in findings
        )
    )
