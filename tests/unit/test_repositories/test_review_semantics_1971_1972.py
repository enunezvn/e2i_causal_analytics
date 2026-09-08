"""Pins for two semantics traps in the expert-review path (#1971, #1972).

Neither is a behaviour change -- both are cases where the code is defensible but
what it *communicates* is wrong, which is how a future change goes wrong.

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


class TestCanUseEstimateDivergence:
    """#1971: two functions share a name and do NOT share a contract.

    SQL `can_use_estimate(estimate_id, dag_hash)` combines the validation gate
    AND expert approval. The Python method of the same name takes no dag_hash,
    never consults approval, and is fail-open. Both are uncalled today, so the
    risk is not current behaviour -- it is that someone wires the Python one up
    believing the SQL contract and silently enforces nothing.
    """

    def test_the_two_signatures_really_do_differ(self):
        py = _func(CAUSAL_VALIDATION, "can_use_estimate")
        py_args = [a.arg for a in py.args.args if a.arg != "self"]
        assert py_args == ["estimate_id"], (
            f"Python can_use_estimate now takes {py_args}. If it grew a dag_hash "
            "it may finally match the SQL contract -- update this pin and the "
            "warning docstring together, deliberately."
        )

        sql = VALIDATION_SQL.read_text(encoding="utf-8")
        match = re.search(
            r"CREATE OR REPLACE FUNCTION can_use_estimate\s*\((.*?)\)\s*RETURNS",
            sql,
            re.S | re.I,
        )
        assert match, "SQL can_use_estimate signature not found"
        sql_params = [p for p in match.group(1).split(",") if p.strip()]
        assert len(sql_params) == 2, (
            f"SQL can_use_estimate takes {len(sql_params)} params; the divergence "
            "this test pins is that it takes a dag_hash the Python one does not"
        )

    def test_the_python_method_warns_that_it_is_not_the_sql_one(self):
        doc = ast.get_docstring(_func(CAUSAL_VALIDATION, "can_use_estimate")) or ""
        assert "#1971" in doc
        assert "expert approval" in doc.lower(), (
            "the docstring must say it does NOT consult expert approval"
        )
        assert "fail-open" in doc.lower(), (
            "the docstring must say it returns True when there are no validations"
        )

    def test_the_fail_open_branch_is_still_there(self):
        """Pins the behaviour the warning describes, so they cannot drift apart."""
        src = ast.unparse(_func(CAUSAL_VALIDATION, "can_use_estimate"))
        assert "if gate else True" in src.replace("\n", " "), (
            "can_use_estimate no longer returns True on 'no validations'. That may "
            "be a real improvement -- but the docstring calls it fail-open, so "
            "change both together."
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
