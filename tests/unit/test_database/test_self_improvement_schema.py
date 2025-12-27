"""Tests for self-improvement database schema (migration 022).

Tests verify:
1. Migration SQL syntax is valid
2. New tables have correct structure
3. ENUMs and functions are properly defined
"""

import pytest
import re
from pathlib import Path


MIGRATION_PATH = Path(__file__).parent.parent.parent.parent / "database" / "ml" / "022_self_improvement_tables.sql"


@pytest.fixture
def migration_sql() -> str:
    """Load the migration SQL file."""
    return MIGRATION_PATH.read_text()


class TestMigrationSyntax:
    """Test migration SQL syntax and structure."""

    def test_migration_file_exists(self):
        """Migration file should exist."""
        assert MIGRATION_PATH.exists(), f"Migration file not found: {MIGRATION_PATH}"

    def test_migration_not_empty(self, migration_sql: str):
        """Migration file should have content."""
        assert len(migration_sql) > 1000, "Migration file appears to be too short"

    def test_has_required_enums(self, migration_sql: str):
        """Migration should define improvement_type and improvement_priority ENUMs."""
        assert "CREATE TYPE improvement_type AS ENUM" in migration_sql
        assert "CREATE TYPE improvement_priority AS ENUM" in migration_sql

    def test_enum_values_correct(self, migration_sql: str):
        """ENUMs should have correct values."""
        # improvement_type values
        assert "'retrieval'" in migration_sql
        assert "'prompt'" in migration_sql
        assert "'workflow'" in migration_sql
        assert "'none'" in migration_sql

        # improvement_priority values
        assert "'critical'" in migration_sql
        assert "'high'" in migration_sql
        assert "'medium'" in migration_sql
        assert "'low'" in migration_sql


class TestNewTables:
    """Test new table definitions."""

    def test_creates_evaluation_results(self, migration_sql: str):
        """Should create evaluation_results table."""
        assert "CREATE TABLE IF NOT EXISTS evaluation_results" in migration_sql

    def test_creates_retrieval_configurations(self, migration_sql: str):
        """Should create retrieval_configurations table."""
        assert "CREATE TABLE IF NOT EXISTS retrieval_configurations" in migration_sql

    def test_creates_prompt_configurations(self, migration_sql: str):
        """Should create prompt_configurations table."""
        assert "CREATE TABLE IF NOT EXISTS prompt_configurations" in migration_sql

    def test_creates_improvement_actions(self, migration_sql: str):
        """Should create improvement_actions table."""
        assert "CREATE TABLE IF NOT EXISTS improvement_actions" in migration_sql

    def test_creates_experiment_knowledge_store(self, migration_sql: str):
        """Should create experiment_knowledge_store table."""
        assert "CREATE TABLE IF NOT EXISTS experiment_knowledge_store" in migration_sql

    def test_evaluation_results_has_ragas_columns(self, migration_sql: str):
        """evaluation_results should have RAGAS metric columns."""
        ragas_columns = [
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
            "answer_correctness",
            "ragas_aggregate",
        ]
        for col in ragas_columns:
            assert col in migration_sql, f"Missing column: {col}"

    def test_evaluation_results_has_rubric_columns(self, migration_sql: str):
        """evaluation_results should have rubric metric columns."""
        rubric_columns = [
            "causal_validity",
            "actionability",
            "evidence_chain",
            "regulatory_awareness",
            "uncertainty_communication",
            "rubric_aggregate",
        ]
        for col in rubric_columns:
            assert col in migration_sql, f"Missing column: {col}"

    def test_retrieval_config_has_hybrid_weights(self, migration_sql: str):
        """retrieval_configurations should have hybrid RAG weight columns."""
        weight_columns = ["vector_weight", "fulltext_weight", "graph_weight"]
        for col in weight_columns:
            assert col in migration_sql, f"Missing column: {col}"

    def test_weights_constraint_exists(self, migration_sql: str):
        """Should have constraint ensuring weights sum to 1.0."""
        assert "chk_weights_sum" in migration_sql
        assert "vector_weight + fulltext_weight + graph_weight" in migration_sql


class TestAlterExistingTables:
    """Test alterations to existing tables."""

    def test_alters_learning_signals(self, migration_sql: str):
        """Should alter learning_signals to add new columns."""
        assert "ALTER TABLE learning_signals ADD COLUMN" in migration_sql

    def test_adds_combined_score(self, migration_sql: str):
        """Should add combined_score column to learning_signals."""
        assert "combined_score" in migration_sql

    def test_adds_improvement_columns(self, migration_sql: str):
        """Should add improvement_type and improvement_priority columns."""
        pattern = r"learning_signals.*ADD COLUMN.*improvement_type"
        assert re.search(pattern, migration_sql, re.DOTALL)


class TestFunctions:
    """Test helper function definitions."""

    def test_creates_calculate_combined_score(self, migration_sql: str):
        """Should create calculate_combined_score function."""
        assert "CREATE OR REPLACE FUNCTION calculate_combined_score" in migration_sql

    def test_creates_determine_improvement_type(self, migration_sql: str):
        """Should create determine_improvement_type function."""
        assert "CREATE OR REPLACE FUNCTION determine_improvement_type" in migration_sql

    def test_creates_determine_improvement_priority(self, migration_sql: str):
        """Should create determine_improvement_priority function."""
        assert "CREATE OR REPLACE FUNCTION determine_improvement_priority" in migration_sql

    def test_creates_update_learning_signal_evaluation(self, migration_sql: str):
        """Should create update_learning_signal_evaluation function."""
        assert "CREATE OR REPLACE FUNCTION update_learning_signal_evaluation" in migration_sql

    def test_creates_get_active_retrieval_config(self, migration_sql: str):
        """Should create get_active_retrieval_config function."""
        assert "CREATE OR REPLACE FUNCTION get_active_retrieval_config" in migration_sql


class TestViews:
    """Test view definitions."""

    def test_creates_ragas_performance_view(self, migration_sql: str):
        """Should create v_ragas_performance_trends view."""
        assert "CREATE OR REPLACE VIEW v_ragas_performance_trends" in migration_sql

    def test_creates_improvement_summary_view(self, migration_sql: str):
        """Should create v_improvement_summary view."""
        assert "CREATE OR REPLACE VIEW v_improvement_summary" in migration_sql

    def test_creates_learning_signal_distribution_view(self, migration_sql: str):
        """Should create v_learning_signal_distribution view."""
        assert "CREATE OR REPLACE VIEW v_learning_signal_distribution" in migration_sql


class TestIndexes:
    """Test index definitions."""

    def test_creates_key_indexes(self, migration_sql: str):
        """Should create important indexes."""
        key_indexes = [
            "idx_eval_learning_signal",
            "idx_eval_ragas",
            "idx_retrieval_active",
            "idx_prompt_active",
            "idx_improvement_signal",
            "idx_knowledge_type",
        ]
        for idx in key_indexes:
            assert idx in migration_sql, f"Missing index: {idx}"


class TestBaselineData:
    """Test baseline configuration data."""

    def test_inserts_baseline_retrieval_config(self, migration_sql: str):
        """Should insert baseline retrieval configuration."""
        assert "INSERT INTO retrieval_configurations" in migration_sql
        assert "baseline_hybrid_rag" in migration_sql

    def test_baseline_has_correct_weights(self, migration_sql: str):
        """Baseline should have correct default weights (0.4, 0.3, 0.3)."""
        # Check the INSERT statement contains these values
        assert "0.4" in migration_sql
        assert "0.3" in migration_sql
