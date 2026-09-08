"""Pins for two semantics traps in the expert-review path (#1971, #1972).

#1972 is not a behaviour change -- the code is defensible but what it
*communicates* was wrong. #1971 WAS a behaviour change: the two same-named
``can_use_estimate`` functions are now retired (see TestCanUseEstimateRetired).

Parsed from source rather than imported: these modules pull the Supabase client
and scipy/dowhy transitively, which makes the tests unrunnable where those are
absent and expensive on a memory-capped box. Every fact asserted here is a
declared signature or a docstring, so reading the declaration is the faithful
check.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CAUSAL_VALIDATION = REPO_ROOT / "src" / "repositories" / "causal_validation.py"
VALIDATION_SQL = REPO_ROOT / "database" / "ml" / "010_causal_validation_tables.sql"
SUMMARY_SCHEMA = REPO_ROOT / "src" / "api" / "schemas" / "expert_review.py"
EXPERT_REVIEW_REPO = REPO_ROOT / "src" / "repositories" / "expert_review.py"


def _func(path: Path, name: str) -> ast.AsyncFunctionDef | ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.AsyncFunctionDef, ast.FunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in {path.name}")


class TestCanUseEstimateRetired:
    """#1971: both ``can_use_estimate`` are retired, not reconciled.

    They shared a name and not a contract (SQL: ``(estimate_id, dag_hash)``,
    gate + expert approval; Python: one arg, no approval, fail-open). Neither
    had a caller. Rather than wire one up, #1971 makes the gate real on the live
    agent path (``CAUSAL_IMPACT_REQUIRE_DAG_APPROVAL``) and leaves ONE
    definition of "usable": ``causal_paths.validation_status``, moved only by
    the sole promoter (migration 119). The Python method is deleted here;
    migration 133 drops the SQL function.
    """

    def test_the_python_method_is_gone(self):
        tree = ast.parse(CAUSAL_VALIDATION.read_text(encoding="utf-8"))
        names = {
            n.name for n in ast.walk(tree) if isinstance(n, (ast.AsyncFunctionDef, ast.FunctionDef))
        }
        assert "can_use_estimate" not in names, (
            "Python can_use_estimate is back. It was retired because it never "
            "consulted expert approval and was fail-open (#1971/#1982); the live "
            "gate is RefutationNode + CAUSAL_IMPACT_REQUIRE_DAG_APPROVAL."
        )
        # Positive control: the walk sees the surviving neighbours.
        assert {"get_gate_decision", "get_validation_summary"} <= names

    def test_the_sql_function_is_dropped_by_migration_133(self):
        migration = REPO_ROOT / "database" / "migrations" / "133_retire_can_use_estimate.sql"
        assert migration.exists()
        text = migration.read_text(encoding="utf-8")
        assert re.search(r"DROP FUNCTION IF EXISTS\s+public\.can_use_estimate\s*\(", text, re.I), (
            "migration 133 must drop the SQL can_use_estimate"
        )
        # The baseline file is deliberately untouched (applied 2026-06-04; the
        # DROP is the forward migration), so the function it defines is real.
        assert "CREATE OR REPLACE FUNCTION can_use_estimate" in VALIDATION_SQL.read_text(
            encoding="utf-8"
        )


class TestSummaryCountsAreNotDisjoint:
    """#1972: `expiring_soon` is a SUBSET of `approved`, not a fifth bucket."""

    def test_the_repository_really_does_count_an_expiring_row_twice(self):
        """The behaviour is correct; it is the presentation that misled.

        Asserting it here means the docs below cannot quietly stop matching it.
        """
        src = EXPERT_REVIEW_REPO.read_text(encoding="utf-8")
        block = re.search(
            r"elif exp_date <= soon:\s*\n\s*expiring_soon \+= 1\s*\n\s*approved \+= 1",
            src,
        )
        assert block, (
            "expected the expiring branch to increment BOTH expiring_soon and "
            "approved. If that changed, expiring_soon is now disjoint and the "
            "subset wording in ReviewSummaryResponse must change with it."
        )

    def test_the_response_schema_states_the_subset_relationship(self):
        text = SUMMARY_SCHEMA.read_text(encoding="utf-8")
        cls = text[text.index("class ReviewSummaryResponse") :]
        cls = cls[: cls.index("class ", 10)] if "class " in cls[10:] else cls
        low = cls.lower()
        assert "subset" in low, "the schema must say expiring_soon is a subset"
        assert "#1972" in cls

    def test_the_sql_vocabulary_no_longer_claims_expired_is_written(self):
        """#1972 option (a): documented values must match written values."""
        sql = VALIDATION_SQL.read_text(encoding="utf-8")
        line = next(ln for ln in sql.splitlines() if "approval_status VARCHAR(30)" in ln)
        assert "expired" not in line, (
            "the approval_status declaration still lists 'expired' as a value; "
            "nothing writes it -- expiry is derived from valid_until"
        )
        assert "#1972" in sql, "the correction should reference the issue"
