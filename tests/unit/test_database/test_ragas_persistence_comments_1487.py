"""Migration 033: RAGAS persistence semantics, documented in the schema (#1487).

Wiring the RAGAS writers made something reachable that was previously moot: a
learning_signals row whose ragas_scores holds only SOME of the five metrics.
Migration 022's SQL helpers ``COALESCE`` a missing metric to 0, which on a
partial row understates the score and — via
``determine_improvement_priority`` — mis-routes the improvement work off it.
While ragas_scores was always '{}' that trap could not fire. It can now.

This migration adds no structure and changes no data; it records those
semantics where a SQL-side reader will actually find them. These tests pin that
it stays comment-only, and that it names the specific traps rather than
restating the obvious.
"""

import re
from pathlib import Path

import pytest

DATABASE_ML = Path(__file__).parent.parent.parent.parent / "database" / "ml"
MIGRATION_PATH = DATABASE_ML / "033_ragas_persistence_semantics.sql"


@pytest.fixture
def migration_sql() -> str:
    return MIGRATION_PATH.read_text()


class TestMigrationIsSafe:
    def test_migration_file_exists(self):
        assert MIGRATION_PATH.exists(), f"Migration file not found: {MIGRATION_PATH}"

    def test_number_is_the_next_free_one_in_the_directory(self):
        """A duplicate number applies in an ambiguous order."""
        numbers = [
            int(match.group(1))
            for path in DATABASE_ML.glob("*.sql")
            if (match := re.match(r"(\d+)_", path.name))
        ]
        assert numbers.count(33) == 1
        assert max(numbers) == 33

    def test_is_comment_only(self):
        """A clarification migration must not be able to change structure or
        data — that is the whole reason it is safe to batch at deploy time."""
        statements = [
            line.strip()
            for line in migration_sql_lines()
            if line.strip() and not line.strip().startswith("--")
        ]
        forbidden = re.compile(
            r"^\s*(CREATE|ALTER|DROP|INSERT|UPDATE|DELETE|TRUNCATE|GRANT|REVOKE)\b",
            re.IGNORECASE,
        )
        offenders = [s for s in statements if forbidden.match(s)]
        assert offenders == [], f"non-comment statements found: {offenders}"


def migration_sql_lines():
    return MIGRATION_PATH.read_text().splitlines()


class TestDocumentsTheNullSemantics:
    def test_learning_signals_ragas_columns_are_commented(self, migration_sql: str):
        assert "COMMENT ON COLUMN learning_signals.ragas_scores" in migration_sql
        assert "COMMENT ON COLUMN learning_signals.ragas_weighted" in migration_sql
        assert "COMMENT ON COLUMN learning_signals.combined_score" in migration_sql

    def test_evaluation_results_metric_columns_are_commented(self, migration_sql: str):
        for column in (
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
            "answer_correctness",
            "ragas_aggregate",
        ):
            assert f"COMMENT ON COLUMN evaluation_results.{column}" in migration_sql

    def test_says_null_means_unmeasured_not_zero(self, migration_sql: str):
        """The single most important thing a reader of these columns needs."""
        assert "never a judged 0.0" in migration_sql

    def test_says_combined_score_is_null_without_both_halves(self, migration_sql: str):
        lowered = migration_sql.lower()
        assert "both halves" in lowered


class TestDocumentsTheCoalesceTrap:
    def test_warns_on_calculate_combined_score(self, migration_sql: str):
        """The function silently scores an unmeasured metric as 0.0."""
        assert "COMMENT ON FUNCTION calculate_combined_score" in migration_sql
        assert "COALESCE" in migration_sql

    def test_warns_on_update_learning_signal_evaluation(self, migration_sql: str):
        """Worse than a wrong number: it routes improvement_priority off it."""
        assert "COMMENT ON FUNCTION update_learning_signal_evaluation" in migration_sql
        assert "improvement_priority" in migration_sql

    def test_names_the_python_source_of_truth(self, migration_sql: str):
        assert "ragas_scoring" in migration_sql


class TestDocumentsTheView:
    def test_view_comment_explains_differing_denominators(self, migration_sql: str):
        """AVG() skips NULLs, so each metric averages over its own sample count
        — evaluation_count is NOT the denominator of every column."""
        assert "COMMENT ON VIEW v_ragas_performance_trends" in migration_sql
        assert "evaluation_count" in migration_sql
