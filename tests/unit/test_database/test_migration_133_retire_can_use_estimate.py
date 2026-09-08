"""Migration 133 content lock -- retire the SQL ``can_use_estimate`` (#1971).

Two functions shared the name ``can_use_estimate`` and did not share a contract
(#1982): the SQL one in ``database/ml/010_causal_validation_tables.sql`` took
``(estimate_id, dag_hash)`` and combined the refutation gate with expert
approval; the Python one in ``src/repositories/causal_validation.py`` took one
argument, never consulted approval and was fail-open. Neither had a caller.
#1971 retires BOTH: the Python method is deleted, and migration 133 drops the
SQL function. ``causal_paths.validation_status`` -- maintained by the sole
promoter, the causal_impact RefutationNode -- is the ONE definition of
"usable" (the #1902 lesson: one definition across readers), and the gate
contract the SQL function promised now runs on the live agent path
(``CAUSAL_IMPACT_REQUIRE_DAG_APPROVAL``).

Validated by READING the migration (DB application is deferred to
batch-deploy -- the local DB is prod; the rehearsal is a BEGIN/ROLLBACK).
Mirrors test_migration_122_cohort_profiler_maxage.py.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
MIGRATION = REPO_ROOT / "database" / "migrations" / "133_retire_can_use_estimate.sql"
BASELINE_SQL = REPO_ROOT / "database" / "ml" / "010_causal_validation_tables.sql"
SRC = REPO_ROOT / "src"


def _migration() -> str:
    return MIGRATION.read_text(encoding="utf-8")


def _statements(text: str) -> str:
    """The executable part of a migration: every line that is not a comment."""
    return "\n".join(ln for ln in text.splitlines() if not ln.strip().startswith("--"))


def _baseline_param_types() -> list[str]:
    """Parameter types of the function ml/010 defines, in order."""
    sql = BASELINE_SQL.read_text(encoding="utf-8")
    match = re.search(
        r"CREATE OR REPLACE FUNCTION can_use_estimate\s*\((.*?)\)\s*RETURNS",
        sql,
        re.S | re.I,
    )
    assert match, "ml/010 no longer defines can_use_estimate; the DROP has nothing to retire"
    types: list[str] = []
    for param in match.group(1).split(","):
        param = param.strip()
        if not param:
            continue
        # "p_estimate_id UUID" / "p_dag_hash VARCHAR(64) DEFAULT NULL" -> base type
        type_token = param.split()[1]
        types.append(re.sub(r"\(.*\)$", "", type_token).upper())
    return types


def test_migration_file_exists():
    assert MIGRATION.exists(), f"missing migration: {MIGRATION.name}"


def test_drop_targets_exactly_the_baseline_signature():
    """``DROP FUNCTION IF EXISTS`` must name the identity arguments ml/010
    declares, or Postgres would report "function does not exist" (IF EXISTS
    then silently no-ops and the function stays)."""
    match = re.search(
        r"DROP FUNCTION IF EXISTS\s+public\.can_use_estimate\s*\((.*?)\)\s*;",
        _statements(_migration()),
        re.S | re.I,
    )
    assert match, "migration 133 must DROP FUNCTION IF EXISTS public.can_use_estimate(...)"
    dropped = [p.strip().upper() for p in match.group(1).split(",") if p.strip()]
    assert dropped == _baseline_param_types(), (
        f"DROP names {dropped} but ml/010 declares {_baseline_param_types()}"
    )


def test_runner_owns_the_transaction():
    """run_migrations.sh wraps each file in --single-transaction; a file that
    manages its own transaction breaks that (migration 119 header)."""
    text = _migration()
    statements = [
        ln.strip().upper()
        for ln in text.splitlines()
        if ln.strip() and not ln.strip().startswith("--")
    ]
    assert not any(s.startswith(("BEGIN", "COMMIT", "ROLLBACK")) for s in statements)
    assert "NOTIFY pgrst, 'reload schema';" in text


def test_reasoning_is_recorded_at_the_definition():
    text = _migration()
    assert "#1971" in text
    assert "validation_status" in text, "must name the ONE remaining definition of usable"
    assert "CAUSAL_IMPACT_REQUIRE_DAG_APPROVAL" in text, (
        "must point at the live switch that replaced the SQL contract"
    )
    # The fresh-bootstrap caveat: database/migrations applies BEFORE database/ml,
    # so on a brand-new database ml/010 re-creates the function after this DROP
    # no-ops. That is documented, not hidden.
    assert "ml/010" in text


def test_no_python_caller_exists():
    """The retirement premise: nothing in src/ calls either function.

    Code lines only -- a comment may still NAME the retired function (the
    refutation node explains which contract it replaced); a call or attribute
    reference may not.
    """
    offenders: list[str] = []
    for path in SRC.rglob("*.py"):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip().startswith("#"):
                continue
            if "can_use_estimate" in line:
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{lineno}: {line.strip()}")
    assert offenders == [], "src/ still references can_use_estimate:\n" + "\n".join(offenders)
    # Positive control: the same grep finds the name in the migration file, so
    # an empty result above means "absent", not "grep found nothing at all".
    control = subprocess.run(
        ["grep", "-l", "can_use_estimate", str(MIGRATION)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert control.stdout.strip() == str(MIGRATION)
