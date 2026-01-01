"""Tests for ValidationOutcome and related Phase 4 components.

Version: 4.3
Tests the Feedback Learner integration from Causal Validation Protocol.

Phase 4: Connect Feedback Learner to validation outcomes
"""

import os
from unittest.mock import patch

import pytest

from src.causal_engine import (
    ExperimentKnowledgeStore,
    FailureCategory,
    GateDecision,
    # Store classes
    InMemoryValidationOutcomeStore,
    # Refutation types for test creation
    RefutationResult,
    RefutationStatus,
    RefutationSuite,
    RefutationTestType,
    ValidationFailurePattern,
    ValidationLearning,
    # Dataclasses
    ValidationOutcome,
    # ENUMs
    ValidationOutcomeType,
    # Functions
    create_validation_outcome,
    extract_failure_patterns,
    get_experiment_knowledge_store,
    # Global accessors
    get_validation_outcome_store,
    log_validation_outcome,
)
from src.causal_engine.validation_outcome_store import reset_validation_outcome_store


class TestValidationOutcomeType:
    """Test ValidationOutcomeType enum."""

    def test_validation_outcome_type_values(self):
        """Test enum has correct values."""
        assert ValidationOutcomeType.PASSED.value == "passed"
        assert ValidationOutcomeType.FAILED_CRITICAL.value == "failed_critical"
        assert ValidationOutcomeType.FAILED_MULTIPLE.value == "failed_multiple"
        assert ValidationOutcomeType.NEEDS_REVIEW.value == "needs_review"
        assert ValidationOutcomeType.BLOCKED.value == "blocked"


class TestFailureCategory:
    """Test FailureCategory enum."""

    def test_failure_category_values(self):
        """Test enum has correct values."""
        assert FailureCategory.INSUFFICIENT_SAMPLE.value == "insufficient_sample"
        assert FailureCategory.UNOBSERVED_CONFOUNDING.value == "unobserved_confounding"
        assert FailureCategory.SPURIOUS_CORRELATION.value == "spurious_correlation"
        assert FailureCategory.MODEL_MISSPECIFICATION.value == "model_misspecification"
        assert FailureCategory.EFFECT_INSTABILITY.value == "effect_instability"
        assert FailureCategory.UNKNOWN.value == "unknown"


class TestValidationFailurePattern:
    """Test ValidationFailurePattern dataclass."""

    def test_create_failure_pattern(self):
        """Test creating a failure pattern."""
        pattern = ValidationFailurePattern(
            category=FailureCategory.SPURIOUS_CORRELATION,
            test_name="placebo_treatment",
            description="Placebo treatment showed 60% of original effect",
            severity="critical",
            original_effect=0.50,
            refuted_effect=0.30,
            delta_percent=60.0,
            recommendation="Check for spurious correlations",
        )

        assert pattern.category == FailureCategory.SPURIOUS_CORRELATION
        assert pattern.test_name == "placebo_treatment"
        assert pattern.severity == "critical"
        assert pattern.delta_percent == 60.0

    def test_to_dict(self):
        """Test serialization to dict."""
        pattern = ValidationFailurePattern(
            category=FailureCategory.INSUFFICIENT_SAMPLE,
            test_name="data_subset",
            description="Effect varied across subsets",
            severity="medium",
            original_effect=0.30,
            refuted_effect=0.25,
            delta_percent=16.7,
            recommendation="Increase sample size",
        )

        d = pattern.to_dict()

        assert d["category"] == "insufficient_sample"
        assert d["test_name"] == "data_subset"
        assert d["severity"] == "medium"
        assert d["recommendation"] == "Increase sample size"


class TestValidationOutcome:
    """Test ValidationOutcome dataclass."""

    def test_create_validation_outcome(self):
        """Test creating a validation outcome."""
        outcome = ValidationOutcome(
            outcome_id="vo_test123",
            outcome_type=ValidationOutcomeType.PASSED,
            timestamp="2024-01-15T10:00:00Z",
            estimate_id="est_001",
            treatment_variable="rep_visits",
            outcome_variable="trx_total",
            brand="TestBrand",
            gate_decision="proceed",
            confidence_score=0.85,
            tests_passed=5,
            tests_failed=0,
            tests_total=5,
        )

        assert outcome.outcome_id == "vo_test123"
        assert outcome.outcome_type == ValidationOutcomeType.PASSED
        assert outcome.treatment_variable == "rep_visits"
        assert outcome.confidence_score == 0.85
        assert outcome.tests_passed == 5

    def test_to_dict(self):
        """Test serialization to dict."""
        outcome = ValidationOutcome(
            outcome_id="vo_test456",
            outcome_type=ValidationOutcomeType.FAILED_CRITICAL,
            timestamp="2024-01-15T11:00:00Z",
            treatment_variable="digital_engagement",
            outcome_variable="nrx",
            gate_decision="block",
            confidence_score=0.35,
            tests_passed=2,
            tests_failed=3,
            tests_total=5,
            failure_patterns=[
                ValidationFailurePattern(
                    category=FailureCategory.SPURIOUS_CORRELATION,
                    test_name="placebo_treatment",
                    description="Test failed",
                    severity="critical",
                    original_effect=0.40,
                    refuted_effect=0.25,
                    delta_percent=37.5,
                    recommendation="Review confounders",
                )
            ],
        )

        d = outcome.to_dict()

        assert d["outcome_id"] == "vo_test456"
        assert d["outcome_type"] == "failed_critical"
        assert d["gate_decision"] == "block"
        assert len(d["failure_patterns"]) == 1
        assert d["failure_patterns"][0]["category"] == "spurious_correlation"

    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "outcome_id": "vo_test789",
            "outcome_type": "needs_review",
            "timestamp": "2024-01-15T12:00:00Z",
            "treatment_variable": "conference_attendance",
            "outcome_variable": "market_share",
            "gate_decision": "review",
            "confidence_score": 0.55,
            "tests_passed": 3,
            "tests_failed": 2,
            "tests_total": 5,
            "failure_patterns": [
                {
                    "category": "model_misspecification",
                    "test_name": "random_common_cause",
                    "description": "Effect changed with random cause",
                    "severity": "high",
                    "original_effect": 0.30,
                    "refuted_effect": 0.22,
                    "delta_percent": 26.7,
                    "recommendation": "Review DAG",
                }
            ],
        }

        outcome = ValidationOutcome.from_dict(data)

        assert outcome.outcome_id == "vo_test789"
        assert outcome.outcome_type == ValidationOutcomeType.NEEDS_REVIEW
        assert outcome.confidence_score == 0.55
        assert len(outcome.failure_patterns) == 1
        assert outcome.failure_patterns[0].category == FailureCategory.MODEL_MISSPECIFICATION

    def test_get_learning_summary_passed(self):
        """Test learning summary for passed outcome."""
        outcome = ValidationOutcome(
            outcome_id="vo_passed",
            outcome_type=ValidationOutcomeType.PASSED,
            timestamp="2024-01-15T10:00:00Z",
            treatment_variable="rep_visits",
            outcome_variable="trx_total",
            confidence_score=0.85,
        )

        summary = outcome.get_learning_summary()

        assert "passed" in summary.lower()
        assert "rep_visits" in summary
        assert "trx_total" in summary

    def test_get_learning_summary_failed(self):
        """Test learning summary for failed outcome."""
        outcome = ValidationOutcome(
            outcome_id="vo_failed",
            outcome_type=ValidationOutcomeType.FAILED_CRITICAL,
            timestamp="2024-01-15T10:00:00Z",
            treatment_variable="digital_engagement",
            outcome_variable="nrx",
            confidence_score=0.35,
            failure_patterns=[
                ValidationFailurePattern(
                    category=FailureCategory.SPURIOUS_CORRELATION,
                    test_name="placebo_treatment",
                    description="Detected spurious effect",
                    severity="critical",
                    original_effect=0.40,
                    refuted_effect=0.25,
                    delta_percent=37.5,
                    recommendation="Check confounders",
                )
            ],
        )

        summary = outcome.get_learning_summary()

        assert "failed_critical" in summary.lower()
        assert "spurious_correlation" in summary


class TestExtractFailurePatterns:
    """Test extract_failure_patterns function."""

    def test_extract_patterns_from_failed_tests(self):
        """Test extracting patterns from failed tests."""
        # Create a mock suite with failed tests
        tests = [
            RefutationResult(
                test_name=RefutationTestType.PLACEBO_TREATMENT,
                status=RefutationStatus.FAILED,
                original_effect=0.50,
                refuted_effect=0.30,
                delta_percent=60.0,
            ),
            RefutationResult(
                test_name=RefutationTestType.DATA_SUBSET,
                status=RefutationStatus.PASSED,
                original_effect=0.50,
                refuted_effect=0.48,
                delta_percent=4.0,
            ),
            RefutationResult(
                test_name=RefutationTestType.RANDOM_COMMON_CAUSE,
                status=RefutationStatus.WARNING,
                original_effect=0.50,
                refuted_effect=0.35,
                delta_percent=30.0,
            ),
        ]

        suite = RefutationSuite(
            passed=False,
            confidence_score=0.40,
            gate_decision=GateDecision.BLOCK,
            tests=tests,
        )

        patterns = extract_failure_patterns(suite)

        # Should have 2 patterns (failed and warning)
        assert len(patterns) == 2

        # First pattern should be placebo (failed)
        placebo_pattern = next(p for p in patterns if p.test_name == "placebo_treatment")
        assert placebo_pattern.category == FailureCategory.SPURIOUS_CORRELATION
        assert placebo_pattern.severity in ("critical", "high")

        # Second pattern should be random common cause (warning)
        rcc_pattern = next(p for p in patterns if p.test_name == "random_common_cause")
        assert rcc_pattern.category == FailureCategory.MODEL_MISSPECIFICATION

    def test_extract_patterns_all_passed(self):
        """Test that no patterns are extracted when all tests pass."""
        tests = [
            RefutationResult(
                test_name=RefutationTestType.PLACEBO_TREATMENT,
                status=RefutationStatus.PASSED,
                original_effect=0.50,
                refuted_effect=0.02,
                delta_percent=4.0,
            ),
        ]

        suite = RefutationSuite(
            passed=True,
            confidence_score=0.95,
            gate_decision=GateDecision.PROCEED,
            tests=tests,
        )

        patterns = extract_failure_patterns(suite)

        assert len(patterns) == 0


class TestCreateValidationOutcome:
    """Test create_validation_outcome function."""

    def test_create_from_passing_suite(self):
        """Test creating outcome from passing suite."""
        tests = [
            RefutationResult(
                test_name=RefutationTestType.PLACEBO_TREATMENT,
                status=RefutationStatus.PASSED,
                original_effect=0.50,
                refuted_effect=0.02,
                delta_percent=4.0,
            ),
            RefutationResult(
                test_name=RefutationTestType.BOOTSTRAP,
                status=RefutationStatus.PASSED,
                original_effect=0.50,
                refuted_effect=0.48,
                delta_percent=4.0,
            ),
        ]

        suite = RefutationSuite(
            passed=True,
            confidence_score=0.90,
            gate_decision=GateDecision.PROCEED,
            tests=tests,
            treatment_variable="rep_visits",
            outcome_variable="trx_total",
            brand="TestBrand",
            estimate_id="est_001",
        )

        outcome = create_validation_outcome(suite)

        assert outcome.outcome_type == ValidationOutcomeType.PASSED
        assert outcome.gate_decision == "proceed"
        assert outcome.confidence_score == 0.90
        assert outcome.tests_passed == 2
        assert outcome.tests_failed == 0
        assert len(outcome.failure_patterns) == 0

    def test_create_from_blocked_suite(self):
        """Test creating outcome from blocked suite."""
        tests = [
            RefutationResult(
                test_name=RefutationTestType.PLACEBO_TREATMENT,
                status=RefutationStatus.FAILED,
                original_effect=0.50,
                refuted_effect=0.35,
                delta_percent=70.0,
            ),
        ]

        suite = RefutationSuite(
            passed=False,
            confidence_score=0.30,
            gate_decision=GateDecision.BLOCK,
            tests=tests,
            treatment_variable="digital_engagement",
            outcome_variable="nrx",
        )

        outcome = create_validation_outcome(
            suite,
            agent_context={"agent": "test", "query_id": "q123"},
            dag_hash="abc123",
            sample_size=1000,
        )

        assert outcome.outcome_type == ValidationOutcomeType.FAILED_CRITICAL
        assert outcome.gate_decision == "block"
        assert len(outcome.failure_patterns) == 1
        assert outcome.agent_context["agent"] == "test"
        assert outcome.dag_hash == "abc123"
        assert outcome.sample_size == 1000


class TestInMemoryValidationOutcomeStore:
    """Test InMemoryValidationOutcomeStore."""

    @pytest.fixture
    def store(self):
        """Create fresh store for each test."""
        return InMemoryValidationOutcomeStore()

    @pytest.fixture
    def sample_outcome(self):
        """Create sample validation outcome."""
        return ValidationOutcome(
            outcome_id="vo_sample",
            outcome_type=ValidationOutcomeType.FAILED_CRITICAL,
            timestamp="2024-01-15T10:00:00Z",
            treatment_variable="rep_visits",
            outcome_variable="trx_total",
            brand="TestBrand",
            gate_decision="block",
            confidence_score=0.35,
            tests_passed=2,
            tests_failed=3,
            tests_total=5,
            failure_patterns=[
                ValidationFailurePattern(
                    category=FailureCategory.SPURIOUS_CORRELATION,
                    test_name="placebo_treatment",
                    description="Test failed",
                    severity="critical",
                    original_effect=0.40,
                    refuted_effect=0.25,
                    delta_percent=37.5,
                    recommendation="Review confounders",
                )
            ],
        )

    @pytest.mark.asyncio
    async def test_store_and_get(self, store, sample_outcome):
        """Test storing and retrieving an outcome."""
        outcome_id = await store.store(sample_outcome)

        assert outcome_id == "vo_sample"
        assert store.count == 1

        retrieved = await store.get(outcome_id)

        assert retrieved is not None
        assert retrieved.outcome_id == "vo_sample"
        assert retrieved.treatment_variable == "rep_visits"

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, store):
        """Test getting nonexistent outcome returns None."""
        result = await store.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_query_failures(self, store, sample_outcome):
        """Test querying failures."""
        await store.store(sample_outcome)

        # Add a passed outcome
        passed_outcome = ValidationOutcome(
            outcome_id="vo_passed",
            outcome_type=ValidationOutcomeType.PASSED,
            timestamp="2024-01-15T11:00:00Z",
            treatment_variable="digital_engagement",
            outcome_variable="nrx",
            gate_decision="proceed",
            confidence_score=0.90,
        )
        await store.store(passed_outcome)

        # Query failures only
        failures = await store.query_failures(limit=10)

        assert len(failures) == 1
        assert failures[0].outcome_id == "vo_sample"

    @pytest.mark.asyncio
    async def test_query_failures_by_treatment(self, store, sample_outcome):
        """Test filtering by treatment variable."""
        await store.store(sample_outcome)

        # Add another failure with different treatment
        other_outcome = ValidationOutcome(
            outcome_id="vo_other",
            outcome_type=ValidationOutcomeType.FAILED_MULTIPLE,
            timestamp="2024-01-15T12:00:00Z",
            treatment_variable="digital_engagement",
            outcome_variable="nrx",
            gate_decision="block",
            confidence_score=0.40,
            failure_patterns=[],
        )
        await store.store(other_outcome)

        failures = await store.query_failures(treatment_variable="rep_visits")

        assert len(failures) == 1
        assert failures[0].treatment_variable == "rep_visits"

    @pytest.mark.asyncio
    async def test_get_failure_patterns(self, store, sample_outcome):
        """Test aggregating failure patterns."""
        await store.store(sample_outcome)

        patterns = await store.get_failure_patterns(limit=10)

        assert len(patterns) >= 1
        assert patterns[0]["test_name"] == "placebo_treatment"
        assert patterns[0]["count"] == 1

    @pytest.mark.asyncio
    async def test_get_similar_failures(self, store, sample_outcome):
        """Test finding similar failures."""
        await store.store(sample_outcome)

        similar = await store.get_similar_failures(
            treatment_variable="rep_visits",
            outcome_variable="trx_total",
        )

        assert len(similar) >= 1
        assert similar[0].treatment_variable == "rep_visits"

    @pytest.mark.asyncio
    async def test_clear(self, store, sample_outcome):
        """Test clearing the store."""
        await store.store(sample_outcome)
        assert store.count == 1

        store.clear()
        assert store.count == 0


class TestExperimentKnowledgeStore:
    """Test ExperimentKnowledgeStore for Experiment Designer integration."""

    @pytest.fixture
    def knowledge_store(self):
        """Create knowledge store with in-memory backend."""
        outcome_store = InMemoryValidationOutcomeStore()
        return ExperimentKnowledgeStore(outcome_store=outcome_store)

    @pytest.fixture
    async def populated_store(self):
        """Create knowledge store with sample data."""
        outcome_store = InMemoryValidationOutcomeStore()

        # Add sample failures
        for i in range(3):
            outcome = ValidationOutcome(
                outcome_id=f"vo_test_{i}",
                outcome_type=ValidationOutcomeType.FAILED_CRITICAL,
                timestamp=f"2024-01-{15+i}T10:00:00Z",
                treatment_variable="rep_visits" if i % 2 == 0 else "digital_engagement",
                outcome_variable="trx_total" if i % 2 == 0 else "nrx",
                gate_decision="block",
                confidence_score=0.30 + i * 0.05,
                failure_patterns=[
                    ValidationFailurePattern(
                        category=FailureCategory.SPURIOUS_CORRELATION,
                        test_name="placebo_treatment",
                        description=f"Test failed {i}",
                        severity="critical",
                        original_effect=0.40,
                        refuted_effect=0.25,
                        delta_percent=37.5,
                        recommendation="Review confounders",
                    )
                ],
            )
            await outcome_store.store(outcome)

        return ExperimentKnowledgeStore(outcome_store=outcome_store)

    @pytest.mark.asyncio
    async def test_get_similar_experiments_empty(self, knowledge_store):
        """Test getting similar experiments from empty store."""
        experiments = await knowledge_store.get_similar_experiments(
            business_question="What is the effect of rep visits on prescriptions?"
        )

        assert isinstance(experiments, list)

    @pytest.mark.asyncio
    async def test_get_similar_experiments_with_data(self, populated_store):
        """Test getting similar experiments with data."""
        experiments = await populated_store.get_similar_experiments(
            business_question="What is the effect of rep visits on trx?"
        )

        assert len(experiments) >= 1
        # Should match rep_visits experiments more
        assert any("rep_visits" in str(exp) for exp in experiments)

    @pytest.mark.asyncio
    async def test_get_recent_assumption_violations(self, populated_store):
        """Test getting recent violations."""
        violations = await populated_store.get_recent_assumption_violations(limit=5)

        assert len(violations) >= 1
        assert "violation_type" in violations[0]
        assert "recommendation" in violations[0]

    @pytest.mark.asyncio
    async def test_get_validation_learnings(self, populated_store):
        """Test getting structured learnings."""
        learnings = await populated_store.get_validation_learnings(limit=5)

        assert isinstance(learnings, list)
        for learning in learnings:
            assert isinstance(learning, ValidationLearning)
            assert learning.failure_category
            assert learning.recommendation

    @pytest.mark.asyncio
    async def test_should_warn_for_design(self, populated_store):
        """Test getting design warnings."""
        warnings = await populated_store.should_warn_for_design(
            treatment_variable="rep_visits",
            outcome_variable="trx_total",
        )

        assert isinstance(warnings, list)
        # Should have warnings for similar past failures
        assert any("⚠️" in w for w in warnings) or len(warnings) == 0


class TestGlobalAccessors:
    """Test global accessor functions."""

    def test_get_validation_outcome_store(self):
        """Test getting global store instance (in-memory mode)."""
        # Reset global singleton to ensure clean state
        reset_validation_outcome_store()

        # Isolate environment by removing SUPABASE_URL
        env_without_supabase = {k: v for k, v in os.environ.items() if k != "SUPABASE_URL"}
        with patch.dict(os.environ, env_without_supabase, clear=True):
            store = get_validation_outcome_store()

            assert store is not None
            assert isinstance(store, InMemoryValidationOutcomeStore)

        # Cleanup: reset again for subsequent tests
        reset_validation_outcome_store()

    def test_get_experiment_knowledge_store(self):
        """Test getting global knowledge store instance."""
        store = get_experiment_knowledge_store()

        assert store is not None
        assert isinstance(store, ExperimentKnowledgeStore)

    @pytest.mark.asyncio
    async def test_log_validation_outcome(self):
        """Test convenience logging function."""
        outcome = ValidationOutcome(
            outcome_id="vo_log_test",
            outcome_type=ValidationOutcomeType.PASSED,
            timestamp="2024-01-15T10:00:00Z",
        )

        outcome_id = await log_validation_outcome(outcome)

        assert outcome_id == "vo_log_test"
