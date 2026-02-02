"""
Unit tests for validation_outcome_store.py

Tests cover:
- InMemoryValidationOutcomeStore
- SupabaseValidationOutcomeStore
- ExperimentKnowledgeStore
- Global store functions
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from src.causal_engine.validation_outcome_store import (
    InMemoryValidationOutcomeStore,
    SupabaseValidationOutcomeStore,
    ExperimentKnowledgeStore,
    get_validation_outcome_store,
    reset_validation_outcome_store,
    get_experiment_knowledge_store,
    log_validation_outcome,
    ValidationLearning,
)
from src.causal_engine.validation_outcome import (
    ValidationOutcome,
    ValidationOutcomeType,
    FailureCategory,
    ValidationFailurePattern,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_outcome():
    """Sample validation outcome for testing."""
    return ValidationOutcome(
        outcome_id="test-outcome-001",
        estimate_id="est-123",
        outcome_type=ValidationOutcomeType.FAILED_MULTIPLE,
        treatment_variable="hcp_engagement",
        outcome_variable="conversion_rate",
        brand="Kisqali",
        sample_size=1000,
        effect_size=0.15,
        gate_decision="block",
        confidence_score=0.45,
        tests_passed=2,
        tests_failed=3,
        tests_total=5,
        failure_patterns=[
            ValidationFailurePattern(
                category=FailureCategory.SPURIOUS_CORRELATION,
                test_name="placebo_treatment",
                description="Placebo effect is significant",
                severity="high",
                original_effect=0.15,
                refuted_effect=0.12,
                delta_percent=20.0,
                recommendation="Review confounders",
            ),
            ValidationFailurePattern(
                category=FailureCategory.MODEL_MISSPECIFICATION,
                test_name="random_common_cause",
                description="Effect sensitive to random confounders",
                severity="medium",
                original_effect=0.15,
                refuted_effect=0.08,
                delta_percent=46.7,
                recommendation="Add control variables",
            ),
        ],
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@pytest.fixture
def sample_passed_outcome():
    """Sample passed validation outcome."""
    return ValidationOutcome(
        outcome_id="test-outcome-002",
        estimate_id="est-456",
        outcome_type=ValidationOutcomeType.PASSED,
        treatment_variable="marketing_spend",
        outcome_variable="revenue",
        brand="Fabhalta",
        sample_size=1500,
        effect_size=0.25,
        gate_decision="proceed",
        confidence_score=0.85,
        tests_passed=5,
        tests_failed=0,
        tests_total=5,
        failure_patterns=[],
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


# ============================================================================
# InMemoryValidationOutcomeStore TESTS
# ============================================================================

class TestInMemoryValidationOutcomeStore:
    """Tests for InMemoryValidationOutcomeStore."""

    @pytest.mark.asyncio
    async def test_store_outcome(self, sample_outcome):
        """Test storing a validation outcome."""
        store = InMemoryValidationOutcomeStore()
        outcome_id = await store.store(sample_outcome)

        assert outcome_id == sample_outcome.outcome_id
        assert store.count == 1

    @pytest.mark.asyncio
    async def test_get_outcome(self, sample_outcome):
        """Test retrieving a validation outcome by ID."""
        store = InMemoryValidationOutcomeStore()
        await store.store(sample_outcome)

        retrieved = await store.get(sample_outcome.outcome_id)

        assert retrieved is not None
        assert retrieved.outcome_id == sample_outcome.outcome_id
        assert retrieved.treatment_variable == sample_outcome.treatment_variable

    @pytest.mark.asyncio
    async def test_get_nonexistent_outcome(self):
        """Test retrieving a non-existent outcome returns None."""
        store = InMemoryValidationOutcomeStore()
        retrieved = await store.get("nonexistent-id")

        assert retrieved is None

    @pytest.mark.asyncio
    async def test_query_failures_basic(self, sample_outcome, sample_passed_outcome):
        """Test querying validation failures."""
        store = InMemoryValidationOutcomeStore()
        await store.store(sample_outcome)
        await store.store(sample_passed_outcome)

        failures = await store.query_failures(limit=10)

        # Should only return failed outcome, not passed
        assert len(failures) == 1
        assert failures[0].outcome_type == ValidationOutcomeType.FAILED_MULTIPLE

    @pytest.mark.asyncio
    async def test_query_failures_by_treatment(self, sample_outcome):
        """Test querying failures filtered by treatment variable."""
        store = InMemoryValidationOutcomeStore()
        await store.store(sample_outcome)

        failures = await store.query_failures(
            treatment_variable="hcp_engagement",
            limit=10
        )

        assert len(failures) == 1
        assert failures[0].treatment_variable == "hcp_engagement"

        # Should return empty for different variable
        failures = await store.query_failures(
            treatment_variable="other_variable",
            limit=10
        )
        assert len(failures) == 0

    @pytest.mark.asyncio
    async def test_query_failures_by_outcome(self, sample_outcome):
        """Test querying failures filtered by outcome variable."""
        store = InMemoryValidationOutcomeStore()
        await store.store(sample_outcome)

        failures = await store.query_failures(
            outcome_variable="conversion_rate",
            limit=10
        )

        assert len(failures) == 1
        assert failures[0].outcome_variable == "conversion_rate"

    @pytest.mark.asyncio
    async def test_query_failures_by_brand(self, sample_outcome):
        """Test querying failures filtered by brand."""
        store = InMemoryValidationOutcomeStore()
        await store.store(sample_outcome)

        failures = await store.query_failures(brand="Kisqali", limit=10)

        assert len(failures) == 1
        assert failures[0].brand == "Kisqali"

    @pytest.mark.asyncio
    async def test_query_failures_by_category(self, sample_outcome):
        """Test querying failures filtered by failure category."""
        store = InMemoryValidationOutcomeStore()
        await store.store(sample_outcome)

        failures = await store.query_failures(
            failure_category=FailureCategory.SPURIOUS_CORRELATION,
            limit=10
        )

        assert len(failures) == 1
        assert any(p.category == FailureCategory.SPURIOUS_CORRELATION for p in failures[0].failure_patterns)

    @pytest.mark.asyncio
    async def test_query_failures_limit(self):
        """Test query failures respects limit."""
        store = InMemoryValidationOutcomeStore()

        # Create multiple failures
        for i in range(10):
            outcome = ValidationOutcome(
                outcome_id=f"test-{i}",
                outcome_type=ValidationOutcomeType.FAILED_CRITICAL,
                treatment_variable="test",
                outcome_variable="test",
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
            await store.store(outcome)

        failures = await store.query_failures(limit=5)
        assert len(failures) == 5

    @pytest.mark.asyncio
    async def test_get_failure_patterns(self, sample_outcome):
        """Test getting aggregated failure patterns."""
        store = InMemoryValidationOutcomeStore()

        # Store multiple outcomes with same pattern
        for i in range(3):
            outcome = ValidationOutcome(
                outcome_id=f"test-{i}",
                outcome_type=ValidationOutcomeType.FAILED_CRITICAL,
                failure_patterns=[
                    ValidationFailurePattern(
                        category=FailureCategory.SPURIOUS_CORRELATION,
                        test_name="placebo_treatment",
                        description="Placebo effect",
                        severity="high",
                        original_effect=0.15,
                        refuted_effect=0.12,
                        delta_percent=20.0,
                        recommendation="Review confounders",
                    ),
                ],
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
            await store.store(outcome)

        patterns = await store.get_failure_patterns(limit=10)

        assert len(patterns) > 0
        assert patterns[0]["count"] == 3
        assert patterns[0]["test_name"] == "placebo_treatment"

    @pytest.mark.asyncio
    async def test_get_failure_patterns_by_category(self, sample_outcome):
        """Test getting failure patterns filtered by category."""
        store = InMemoryValidationOutcomeStore()
        await store.store(sample_outcome)

        patterns = await store.get_failure_patterns(
            category=FailureCategory.SPURIOUS_CORRELATION,
            limit=10
        )

        assert len(patterns) > 0
        assert all(p["category"] == FailureCategory.SPURIOUS_CORRELATION.value for p in patterns)

    @pytest.mark.asyncio
    async def test_get_similar_failures(self, sample_outcome):
        """Test getting similar past failures."""
        store = InMemoryValidationOutcomeStore()
        await store.store(sample_outcome)

        # Create another outcome with similar variables
        similar_outcome = ValidationOutcome(
            outcome_id="test-similar",
            outcome_type=ValidationOutcomeType.FAILED_CRITICAL,
            treatment_variable="hcp_engagement",
            outcome_variable="different_outcome",
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        await store.store(similar_outcome)

        similar = await store.get_similar_failures(
            treatment_variable="hcp_engagement",
            outcome_variable="conversion_rate",
            limit=5
        )

        assert len(similar) > 0
        assert similar[0].treatment_variable == "hcp_engagement"

    @pytest.mark.asyncio
    async def test_clear(self, sample_outcome):
        """Test clearing the store."""
        store = InMemoryValidationOutcomeStore()
        await store.store(sample_outcome)

        assert store.count == 1

        store.clear()

        assert store.count == 0

    @pytest.mark.asyncio
    async def test_count_property(self, sample_outcome, sample_passed_outcome):
        """Test count property."""
        store = InMemoryValidationOutcomeStore()

        assert store.count == 0

        await store.store(sample_outcome)
        assert store.count == 1

        await store.store(sample_passed_outcome)
        assert store.count == 2


# ============================================================================
# SupabaseValidationOutcomeStore TESTS
# ============================================================================

class TestSupabaseValidationOutcomeStore:
    """Tests for SupabaseValidationOutcomeStore."""

    @pytest.mark.asyncio
    async def test_store_outcome_with_supabase(self, sample_outcome):
        """Test storing outcome with Supabase client."""
        store = SupabaseValidationOutcomeStore()

        # Mock Supabase client
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.data = [{"outcome_id": sample_outcome.outcome_id}]
        mock_client.table().insert().execute.return_value = mock_result

        with patch.object(store, '_get_client', return_value=mock_client):
            outcome_id = await store.store(sample_outcome)

            assert outcome_id == sample_outcome.outcome_id
            # Check that table was called with correct name (can be called multiple times in chain)
            mock_client.table.assert_any_call("validation_outcomes")

    @pytest.mark.asyncio
    async def test_store_outcome_fallback_to_memory(self, sample_outcome):
        """Test fallback to in-memory store when Supabase fails."""
        store = SupabaseValidationOutcomeStore()

        with patch.object(store, '_get_client', return_value=None):
            outcome_id = await store.store(sample_outcome)

            assert outcome_id == sample_outcome.outcome_id
            # Should have created fallback store
            assert store._fallback_store is not None

    @pytest.mark.asyncio
    async def test_get_outcome_with_supabase(self, sample_outcome):
        """Test retrieving outcome with Supabase."""
        store = SupabaseValidationOutcomeStore()

        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.data = [store._outcome_to_row(sample_outcome)]
        mock_client.table().select().eq().execute.return_value = mock_result

        with patch.object(store, '_get_client', return_value=mock_client):
            retrieved = await store.get(sample_outcome.outcome_id)

            assert retrieved is not None
            assert retrieved.outcome_id == sample_outcome.outcome_id

    @pytest.mark.asyncio
    async def test_get_outcome_not_found(self):
        """Test retrieving non-existent outcome."""
        store = SupabaseValidationOutcomeStore()

        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.data = []
        mock_client.table().select().eq().execute.return_value = mock_result

        with patch.object(store, '_get_client', return_value=mock_client):
            retrieved = await store.get("nonexistent")

            assert retrieved is None

    @pytest.mark.asyncio
    async def test_query_failures_with_supabase(self, sample_outcome):
        """Test querying failures with Supabase."""
        store = SupabaseValidationOutcomeStore()

        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.data = [store._outcome_to_row(sample_outcome)]

        # Mock the chained query builder
        mock_query = MagicMock()
        mock_query.neq().eq().order().limit().execute.return_value = mock_result
        mock_client.table().select.return_value = mock_query

        with patch.object(store, '_get_client', return_value=mock_client):
            failures = await store.query_failures(
                treatment_variable="hcp_engagement",
                limit=10
            )

            assert len(failures) == 1

    @pytest.mark.asyncio
    async def test_outcome_to_row_conversion(self, sample_outcome):
        """Test converting ValidationOutcome to database row."""
        store = SupabaseValidationOutcomeStore()
        row = store._outcome_to_row(sample_outcome)

        assert row["outcome_id"] == sample_outcome.outcome_id
        assert row["outcome_type"] == sample_outcome.outcome_type.value
        assert row["treatment_variable"] == sample_outcome.treatment_variable
        assert len(row["failure_patterns"]) == len(sample_outcome.failure_patterns)

    @pytest.mark.asyncio
    async def test_row_to_outcome_conversion(self, sample_outcome):
        """Test converting database row to ValidationOutcome."""
        store = SupabaseValidationOutcomeStore()
        row = store._outcome_to_row(sample_outcome)

        outcome = store._row_to_outcome(row)

        assert outcome.outcome_id == sample_outcome.outcome_id
        assert outcome.outcome_type == sample_outcome.outcome_type
        assert len(outcome.failure_patterns) == len(sample_outcome.failure_patterns)


# ============================================================================
# ExperimentKnowledgeStore TESTS
# ============================================================================

class TestExperimentKnowledgeStore:
    """Tests for ExperimentKnowledgeStore."""

    @pytest.mark.asyncio
    async def test_get_similar_experiments(self, sample_outcome):
        """Test getting similar experiments."""
        store = InMemoryValidationOutcomeStore()
        await store.store(sample_outcome)

        knowledge_store = ExperimentKnowledgeStore(store)

        experiments = await knowledge_store.get_similar_experiments(
            business_question="What is the impact of hcp engagement on conversion?",
            limit=5
        )

        assert len(experiments) > 0
        assert experiments[0]["outcome"] == sample_outcome.outcome_variable

    @pytest.mark.asyncio
    async def test_get_recent_assumption_violations(self, sample_outcome):
        """Test getting recent assumption violations."""
        store = InMemoryValidationOutcomeStore()
        await store.store(sample_outcome)

        knowledge_store = ExperimentKnowledgeStore(store)

        violations = await knowledge_store.get_recent_assumption_violations(limit=5)

        assert len(violations) > 0
        assert "violation_type" in violations[0]

    @pytest.mark.asyncio
    async def test_get_validation_learnings(self, sample_outcome):
        """Test getting validation learnings."""
        store = InMemoryValidationOutcomeStore()
        await store.store(sample_outcome)

        knowledge_store = ExperimentKnowledgeStore(store)

        learnings = await knowledge_store.get_validation_learnings(limit=10)

        assert len(learnings) > 0
        assert isinstance(learnings[0], ValidationLearning)
        assert learnings[0].frequency > 0

    @pytest.mark.asyncio
    async def test_get_validation_learnings_by_category(self, sample_outcome):
        """Test getting validation learnings filtered by category."""
        store = InMemoryValidationOutcomeStore()
        await store.store(sample_outcome)

        knowledge_store = ExperimentKnowledgeStore(store)

        learnings = await knowledge_store.get_validation_learnings(
            category=FailureCategory.SPURIOUS_CORRELATION,
            limit=10
        )

        assert len(learnings) > 0
        assert learnings[0].failure_category == FailureCategory.SPURIOUS_CORRELATION.value

    @pytest.mark.asyncio
    async def test_should_warn_for_design(self, sample_outcome):
        """Test getting warnings for proposed design."""
        store = InMemoryValidationOutcomeStore()
        await store.store(sample_outcome)

        knowledge_store = ExperimentKnowledgeStore(store)

        warnings = await knowledge_store.should_warn_for_design(
            treatment_variable="hcp_engagement",
            outcome_variable="conversion_rate"
        )

        assert len(warnings) > 0
        assert any("Review confounders" in w for w in warnings)

    @pytest.mark.asyncio
    async def test_should_warn_no_similar_failures(self):
        """Test warnings when no similar failures exist."""
        store = InMemoryValidationOutcomeStore()
        knowledge_store = ExperimentKnowledgeStore(store)

        warnings = await knowledge_store.should_warn_for_design(
            treatment_variable="new_treatment",
            outcome_variable="new_outcome"
        )

        assert len(warnings) == 0


# ============================================================================
# GLOBAL STORE FUNCTIONS TESTS
# ============================================================================

class TestGlobalStoreFunctions:
    """Tests for global store functions."""

    def setup_method(self):
        """Reset global stores before each test."""
        reset_validation_outcome_store()

    def test_get_validation_outcome_store_default(self):
        """Test getting validation outcome store with defaults."""
        with patch.dict('os.environ', {'SUPABASE_URL': 'https://test.supabase.co'}):
            store = get_validation_outcome_store(use_supabase=True)
            assert isinstance(store, SupabaseValidationOutcomeStore)

    def test_get_validation_outcome_store_in_memory(self):
        """Test getting in-memory validation outcome store."""
        store = get_validation_outcome_store(use_supabase=False)
        assert isinstance(store, InMemoryValidationOutcomeStore)

    def test_get_validation_outcome_store_no_supabase_url(self):
        """Test fallback to in-memory when SUPABASE_URL not set."""
        with patch.dict('os.environ', {}, clear=True):
            store = get_validation_outcome_store(use_supabase=True)
            assert isinstance(store, InMemoryValidationOutcomeStore)

    def test_get_validation_outcome_store_singleton(self):
        """Test that get_validation_outcome_store returns singleton."""
        store1 = get_validation_outcome_store(use_supabase=False)
        store2 = get_validation_outcome_store(use_supabase=False)

        assert store1 is store2

    def test_reset_validation_outcome_store(self):
        """Test resetting validation outcome store."""
        store1 = get_validation_outcome_store(use_supabase=False)
        reset_validation_outcome_store()
        store2 = get_validation_outcome_store(use_supabase=False)

        assert store1 is not store2

    def test_get_experiment_knowledge_store(self):
        """Test getting experiment knowledge store."""
        knowledge_store = get_experiment_knowledge_store()
        assert isinstance(knowledge_store, ExperimentKnowledgeStore)

    def test_get_experiment_knowledge_store_singleton(self):
        """Test that get_experiment_knowledge_store returns singleton."""
        store1 = get_experiment_knowledge_store()
        store2 = get_experiment_knowledge_store()

        assert store1 is store2

    @pytest.mark.asyncio
    async def test_log_validation_outcome(self, sample_outcome):
        """Test convenience function for logging validation outcome."""
        with patch('src.causal_engine.validation_outcome_store.get_validation_outcome_store') as mock_get_store:
            mock_store = AsyncMock()
            mock_store.store = AsyncMock(return_value=sample_outcome.outcome_id)
            mock_get_store.return_value = mock_store

            outcome_id = await log_validation_outcome(sample_outcome)

            assert outcome_id == sample_outcome.outcome_id
            mock_store.store.assert_called_once_with(sample_outcome)
