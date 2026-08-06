"""Migration 034: the combined-score COALESCE and stale-component traps (#1489 deferral 4).

Migration 022 shipped ``calculate_combined_score`` / ``update_learning_signal_evaluation``
under two assumptions that were true when it landed and are false now:

1. every RAGAS bundle carries all five metrics, so ``COALESCE(metric, 0)`` is
   harmless; and
2. nothing else on the row describes the bundle, so replacing the score columns
   leaves the row self-consistent.

#1485 made partial bundles the NORMAL shape (the real-pipeline replay reports
only faithfulness and answer_relevancy — the other three need a ground truth it
deliberately refuses to fabricate), #1488 made an unmeasured metric a NULL
rather than a 0.0, and #1487 added ``signal_details.ragas_coverage`` describing
which metrics a row's bundle holds.

Migration 033 documented both traps in COMMENTs. That kept the wrong arithmetic
reachable: the deferral asks for the functional fix. Measured against real
plpgsql (PostgreSQL 15.8, throwaway database) BEFORE this migration:

    calculate_combined_score('{"faithfulness":1.0,"answer_relevancy":1.0}', 5.0)
        -> 0.78     (a PERFECT partial row; ragas_scoring.py gives 1.0)
    calculate_combined_score(<all five 1.0>, NULL)
        -> 0.40     (40%-of-nothing published as a two-half blend)
    calculate_combined_score('{}', 5.0)
        -> 0.60     (zero RAGAS measurement, still routes work)
    determine_improvement_priority(NULL)      -> 'critical'
    determine_improvement_type(NULL, 0.75)    -> 'workflow'

and at the MEASURED #1485 baseline (faithfulness 0.524, answer_relevancy 0.179,
rubric 4.0) the trap moved the priority from 'medium' to 'high', while a
perfect partial bundle was routed to 'retrieval' — blaming retrieval for a row
retrieval had served perfectly.

The stale half, also measured: calling ``update_learning_signal_evaluation``
on a row the #1487 Python writer had inserted left ``signal_value`` at the OLD
rubric total (4.0 while ``rubric_total`` became 2.0) and left
``signal_details.ragas_coverage.measured`` naming two metrics after the stored
bundle had been replaced by a one-metric one.

These tests pin the migration's text and its agreement with the Python source
of truth. The behaviour itself is pinned by execution against real PostgreSQL
in ``tests/integration/test_combined_score_sql_semantics_1489.py``.
"""

import re
from pathlib import Path

import pytest

from src.agents.feedback_learner.ragas_scoring import (
    RAGAS_BLEND_WEIGHT,
    RAGAS_METRIC_WEIGHTS,
    RUBRIC_BLEND_WEIGHT,
)

DATABASE_ML = Path(__file__).parent.parent.parent.parent / "database" / "ml"
MIGRATION_PATH = DATABASE_ML / "034_combined_score_partial_bundles.sql"


@pytest.fixture
def migration_sql() -> str:
    return MIGRATION_PATH.read_text()


def _function_body(sql: str, name: str) -> str:
    """The text of one ``CREATE OR REPLACE FUNCTION`` block."""
    start = sql.index(f"CREATE OR REPLACE FUNCTION {name}")
    end = sql.index("$$ LANGUAGE plpgsql", start)
    return sql[start:end]


class TestMigrationShape:
    def test_migration_file_exists(self):
        assert MIGRATION_PATH.exists(), f"Migration file not found: {MIGRATION_PATH}"

    def test_number_is_unique_in_the_directory(self, migration_sql: str):
        """A duplicate number applies in an ambiguous order.

        Deliberately NOT ``max(numbers) == 34``: that holds only until the next
        migration lands, and being the highest number was never the property
        worth pinning. The #1487 test asserted it and this migration broke it.
        """
        numbers = [
            int(match.group(1))
            for path in DATABASE_ML.glob("*.sql")
            if (match := re.match(r"(\d+)_", path.name))
        ]
        assert numbers.count(34) == 1

    def test_supersedes_022_as_the_live_definition(self, migration_sql: str):
        """The deploy applies database/ml in numeric order, so the LAST
        definition wins. 034 must outrank 022 or the trap stays live."""
        definers = sorted(
            int(re.match(r"(\d+)_", path.name).group(1))
            for path in DATABASE_ML.glob("*.sql")
            if re.match(r"\d+_", path.name)
            and "CREATE OR REPLACE FUNCTION calculate_combined_score" in path.read_text()
        )
        assert definers[-1] == 34, f"022 still wins the apply order: {definers}"

    def test_touches_no_table_structure_or_data(self, migration_sql: str):
        """Redefining functions is safe to batch at deploy time; a DDL/DML
        statement in the same file would not be."""
        forbidden = re.compile(
            r"^\s*(CREATE\s+TABLE|ALTER\s+TABLE|DROP\s+(TABLE|COLUMN)|INSERT|UPDATE\s+\w|"
            r"DELETE\s+FROM|TRUNCATE)\b",
            re.IGNORECASE,
        )
        offenders = [
            line.strip()
            for line in migration_sql.splitlines()
            # The UPDATE inside update_learning_signal_evaluation's body is the
            # function's own statement, not a migration-time data change; it is
            # indented inside the $$ block.
            if not line.startswith(" ")
            and not line.strip().startswith("--")
            and forbidden.match(line)
        ]
        assert offenders == [], f"structure/data statements found: {offenders}"

    def test_is_idempotent(self, migration_sql: str):
        """Every function definition must be CREATE OR REPLACE — the deploy
        applies the whole ml/ directory, so a bare CREATE FUNCTION would fail
        the second time."""
        bare = re.findall(r"^CREATE FUNCTION\b", migration_sql, re.MULTILINE)
        assert bare == []


class TestCoalesceTrapIsGone:
    def test_redefines_both_flagged_functions(self, migration_sql: str):
        assert "CREATE OR REPLACE FUNCTION calculate_combined_score" in migration_sql
        assert "CREATE OR REPLACE FUNCTION update_learning_signal_evaluation" in migration_sql

    def test_no_metric_is_coalesced_to_zero(self, migration_sql: str):
        """The trap itself: an absent metric scored as a judged 0.0."""
        offenders = re.findall(
            r"COALESCE\(\(p_ragas_scores->>'(\w+)'\)::float,\s*0\)", migration_sql
        )
        assert offenders == [], f"COALESCE-to-zero survives for {offenders}"

    def test_rubric_total_is_not_coalesced_to_the_bottom_of_the_scale(self, migration_sql: str):
        assert "COALESCE((p_rubric_total - 1) / 4.0, 0)" not in migration_sql

    def test_renormalises_over_the_measured_weight(self, migration_sql: str):
        """A partial bundle must be scored on what was judged, which requires
        dividing by the weight that was actually measured."""
        body = _function_body(migration_sql, "ragas_weighted_measured")
        assert "v_measured_weight" in body
        assert re.search(r"v_weighted_sum\s*/\s*v_measured_weight", body), body

    def test_returns_null_when_a_half_is_missing(self, migration_sql: str):
        """A blend of a half that does not exist is not a measurement."""
        body = _function_body(migration_sql, "calculate_combined_score")
        assert "RETURN NULL" in body
        assert "p_rubric_total IS NULL" in body

    def test_routing_helpers_are_null_safe(self, migration_sql: str):
        """determine_improvement_priority(NULL) returned 'critical' and
        determine_improvement_type(NULL, x) returned 'workflow' — a fabricated
        verdict off an absent measurement."""
        priority = _function_body(migration_sql, "determine_improvement_priority")
        assert "p_combined_score IS NULL" in priority
        assert "RETURN NULL" in priority

        itype = _function_body(migration_sql, "determine_improvement_type")
        assert "p_ragas_weighted IS NULL" in itype
        assert "p_rubric_normalized IS NULL" in itype
        assert "RETURN NULL" in itype


class TestStaleComponentsAreRefreshed:
    def test_signal_value_follows_rubric_total(self, migration_sql: str):
        """signal_value carried the rubric score at INSERT (rubric_node.py sets
        signal_value = evaluation.weighted_score). Updating rubric_total without
        it leaves the row contradicting itself."""
        body = _function_body(migration_sql, "update_learning_signal_evaluation")
        assert re.search(r"signal_value\s*=\s*p_rubric_total", body), body

    def test_ragas_coverage_is_recomputed_from_the_stored_bundle(self, migration_sql: str):
        """signal_details.ragas_coverage described the PREVIOUS bundle and was
        left untouched, so it could name metrics the row no longer holds."""
        body = _function_body(migration_sql, "update_learning_signal_evaluation")
        assert "ragas_coverage" in body
        assert "signal_details" in body

    def test_does_not_reintroduce_the_duplicated_weight_block(self, migration_sql: str):
        """022 copy-pasted the weighted-sum expression into BOTH functions, so
        a future weight change had two places to miss. The weights now have a
        single definition that every other function selects from — the same
        desync class #1499 hit when a loader and its cache-probe each sorted."""
        assert migration_sql.count("CREATE OR REPLACE FUNCTION ragas_metric_weights") == 1
        # Nobody restates a metric weight outside that one table.
        assert len(re.findall(r"'faithfulness'", migration_sql)) == 1, (
            "a metric name appears outside the single weight table"
        )
        body = _function_body(migration_sql, "update_learning_signal_evaluation")
        assert "ragas_weighted_measured(" in body
        assert "ragas_metric_weights()" in body, "coverage must use the same metric list"


class TestWeightsStillPinnedToPython:
    """``ragas_scoring.py`` is the Python source of truth and states its weights
    are transcribed from this SQL. Parsing them keeps the two from drifting —
    the same guard the #1487 tests apply, repointed at the CURRENT definition."""

    def test_metric_weights_match_the_python_source_of_truth(self, migration_sql: str):
        body = _function_body(migration_sql, "ragas_metric_weights")
        sql_weights = {
            name: float(weight) for name, weight in re.findall(r"\('(\w+)',\s*([\d.]+)\)", body)
        }
        assert sql_weights, "failed to parse metric weights out of migration 034"
        assert sql_weights == RAGAS_METRIC_WEIGHTS

    def test_blend_weights_match_the_python_source_of_truth(self, migration_sql: str):
        body = _function_body(migration_sql, "calculate_combined_score")
        assert f"p_ragas_weight FLOAT DEFAULT {RAGAS_BLEND_WEIGHT}" in body
        assert f"p_rubric_weight FLOAT DEFAULT {RUBRIC_BLEND_WEIGHT}" in body


class TestMigrationDocumentsWhy:
    def test_names_the_issue_and_the_superseded_migration(self, migration_sql: str):
        assert "#1489" in migration_sql
        assert "033" in migration_sql, "must say it supersedes 033's comment-only warning"

    def test_names_the_python_source_of_truth(self, migration_sql: str):
        assert "ragas_scoring.py" in migration_sql
