"""Tests for ValidationOutcome and related Phase 4 components.

Version: 4.3
Tests the Feedback Learner integration from Causal Validation Protocol.

Phase 4: Connect Feedback Learner to validation outcomes
"""

import os
import uuid
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

    def test_sensitivity_readings_are_categorized_by_reading_not_by_a_cutoff(self):
        """2026-09-10: the sensitivity pattern follows the READING, not an E-value.

        The retired rule labelled any sensitivity row with ``e_value < 1.5`` a
        CRITICAL unobserved-confounding pattern. Under the benchmarked reading a
        null finding typically carries a small E-value and is a PRECISION finding,
        not a confounding one -- storing it as critical confounding teaches the
        Experiment Designer something false. Every row below carries the same
        e_value=1.2 so only the reading can be doing the work.
        """
        readings = {
            "null_finding": (FailureCategory.INSUFFICIENT_SAMPLE, "low", "null finding"),
            "within_measured_confounding": (
                FailureCategory.UNOBSERVED_CONFOUNDING,
                "high",
                "measured confounding",
            ),
            "unbenchmarked": (
                FailureCategory.UNOBSERVED_CONFOUNDING,
                "medium",
                "no measured confounders",
            ),
        }

        tests = [
            RefutationResult(
                test_name=RefutationTestType.SENSITIVITY_E_VALUE,
                status=RefutationStatus.WARNING,
                original_effect=0.50,
                refuted_effect=0.50,
                delta_percent=0.0,
                details={"reading": reading, "e_value": 1.2, "message": "m"},
            )
            for reading in readings
        ]

        suite = RefutationSuite(
            passed=False,
            confidence_score=0.55,
            gate_decision=GateDecision.REVIEW,
            tests=tests,
        )

        patterns = extract_failure_patterns(suite)

        assert len(patterns) == len(readings)
        for pattern, reading in zip(patterns, readings, strict=True):
            category, severity, phrase = readings[reading]
            assert pattern.category == category, reading
            assert pattern.severity == severity, reading
            assert phrase in pattern.recommendation.lower(), reading

        by_reading = {p.test_name and r: p for r, p in zip(readings, patterns, strict=True)}

        # Review round 1 (2026-09-10): for the `within` reading the E-value bound
        # leaves the DIRECTION unestablished too, so the advice may not present it
        # as the reliable half. And a null finding is not fixed only by more rows.
        within = by_reading["within_measured_confounding"].recommendation
        assert "more reliable than the size" not in within
        assert (
            "Do not act on the size of this effect, and treat its direction as "
            "unconfirmed against confounding of that strength" in within
        )

        null = by_reading["null_finding"].recommendation
        assert "the only way" not in null
        assert (
            "a larger sample, a longer window, or a more precise outcome measure "
            "is needed to detect a smaller effect." in null
        )

    @staticmethod
    def _unbenchmarked_pattern(details):
        suite = RefutationSuite(
            passed=False,
            confidence_score=0.55,
            gate_decision=GateDecision.REVIEW,
            tests=[
                RefutationResult(
                    test_name=RefutationTestType.SENSITIVITY_E_VALUE,
                    status=RefutationStatus.WARNING,
                    original_effect=0.30,
                    refuted_effect=0.30,
                    delta_percent=0.0,
                    details={"reading": "unbenchmarked", "e_value": 1.2, "message": "m", **details},
                )
            ],
        )
        (pattern,) = extract_failure_patterns(suite)
        return pattern

    def test_unbenchmarked_recommendation_follows_the_measured_unscoreable_basis(self):
        """Whole-diff review F2: the persisted recommendation must follow the
        benchmark basis. On ``measured_unscoreable`` the confounders WERE measured
        (the live ``peer_influence_score -> adopted`` runs declared ``centrality_z``,
        collinear with the treatment); "no measured confounders exist for this
        design" is false there and "add covariates" is the wrong instruction."""
        pattern = self._unbenchmarked_pattern(
            {"benchmark_basis": "measured_unscoreable", "covariates_measured": 1}
        )
        assert pattern.category == FailureCategory.UNOBSERVED_CONFOUNDING
        assert pattern.severity == "medium"
        assert "the 1 measured confounder(s) could not be scored on this frame" in (
            pattern.recommendation
        )
        assert "collinear with the treatment" in pattern.recommendation
        assert "varies independently of the treatment" in pattern.recommendation
        assert "no measured confounders exist" not in pattern.recommendation

    def test_unbenchmarked_recommendation_without_a_count_still_names_the_measured_set(self):
        pattern = self._unbenchmarked_pattern({"benchmark_basis": "measured_unscoreable"})
        assert "the measured confounders could not be scored on this frame" in (
            pattern.recommendation
        )
        assert "no measured confounders exist" not in pattern.recommendation

    def test_unbenchmarked_recommendation_keeps_todays_text_on_none_measured(self):
        for details in ({"benchmark_basis": "none_measured"}, {}):
            pattern = self._unbenchmarked_pattern(details)
            assert "no measured confounders exist for this design" in pattern.recommendation
            assert "Add covariates so the E-value has a benchmark" in pattern.recommendation
            assert "could not be scored" not in pattern.recommendation

    def test_legacy_sensitivity_row_without_a_reading_is_never_critical(self):
        """Rows persisted before 2026-09-10 carry an e_value and no ``reading``.

        They must degrade to the generic confounding pattern -- the 1.5 cutoff that
        used to make this row "critical" no longer exists anywhere.
        """
        suite = RefutationSuite(
            passed=False,
            confidence_score=0.55,
            gate_decision=GateDecision.REVIEW,
            tests=[
                RefutationResult(
                    test_name=RefutationTestType.SENSITIVITY_E_VALUE,
                    status=RefutationStatus.WARNING,
                    original_effect=0.50,
                    refuted_effect=0.50,
                    delta_percent=0.0,
                    details={"e_value": 1.2},
                )
            ],
        )

        (pattern,) = extract_failure_patterns(suite)

        assert pattern.category == FailureCategory.UNOBSERVED_CONFOUNDING
        assert pattern.severity == "high"

    def test_sensitivity_description_follows_the_reading(self):
        """2026-09-10: the stored DESCRIPTION must match the stored category.

        The retired sentence asserted "sensitivity to unmeasured confounding" for
        every sensitivity row. On a ``null_finding`` -- categorized
        insufficient_sample / low, with a recommendation saying no unmeasured
        confounder is needed -- that made one record contradict itself.
        """
        headline = "No detectable effect at this sample size"
        suite = RefutationSuite(
            passed=False,
            confidence_score=0.55,
            gate_decision=GateDecision.REVIEW,
            tests=[
                RefutationResult(
                    test_name=RefutationTestType.SENSITIVITY_E_VALUE,
                    status=RefutationStatus.WARNING,
                    original_effect=0.02,
                    refuted_effect=0.02,
                    delta_percent=0.0,
                    details={
                        "reading": "null_finding",
                        "e_value": 1.2,
                        "headline": headline,
                        "message": "The 95 % CI includes zero at n=1500.",
                    },
                )
            ],
        )

        (pattern,) = extract_failure_patterns(suite)

        assert pattern.description.startswith(headline)
        assert "unmeasured confounding" not in pattern.description
        assert "The 95 % CI includes zero at n=1500." in pattern.description

    def test_sensitivity_description_falls_back_to_the_canonical_headline(self):
        """A row carrying a ``reading`` but no stored headline still reads right."""
        from src.causal_engine import evalue

        suite = RefutationSuite(
            passed=False,
            confidence_score=0.55,
            gate_decision=GateDecision.REVIEW,
            tests=[
                RefutationResult(
                    test_name=RefutationTestType.SENSITIVITY_E_VALUE,
                    status=RefutationStatus.WARNING,
                    original_effect=0.30,
                    refuted_effect=0.30,
                    delta_percent=0.0,
                    details={"reading": "unbenchmarked", "e_value": 1.2, "message": "m"},
                )
            ],
        )

        (pattern,) = extract_failure_patterns(suite)

        assert pattern.description.startswith(evalue.HEADLINES["unbenchmarked"])
        assert "unmeasured confounding" not in pattern.description

    def test_legacy_sensitivity_description_is_unchanged(self):
        """Rows persisted before the reading existed keep today's sentence."""
        suite = RefutationSuite(
            passed=False,
            confidence_score=0.55,
            gate_decision=GateDecision.REVIEW,
            tests=[
                RefutationResult(
                    test_name=RefutationTestType.SENSITIVITY_E_VALUE,
                    status=RefutationStatus.WARNING,
                    original_effect=0.30,
                    refuted_effect=0.30,
                    delta_percent=0.0,
                    details={"e_value": 1.2},
                )
            ],
        )

        (pattern,) = extract_failure_patterns(suite)

        assert pattern.description == (
            "E-value of 1.2 indicates sensitivity to unmeasured confounding"
        )

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

    def test_random_common_cause_failed_block_is_failed_critical(self):
        """random_common_cause IS a critical test; a BLOCK on it is FAILED_CRITICAL.

        The retired hardcoded set was ``("placebo_treatment", "sensitivity_e_value")``:
        it omitted random_common_cause, so this suite was stored as FAILED_MULTIPLE,
        and it named sensitivity_e_value, which can no longer FAIL at all.
        """
        tests = [
            RefutationResult(
                test_name=RefutationTestType.PLACEBO_TREATMENT,
                status=RefutationStatus.PASSED,
                original_effect=0.50,
                refuted_effect=0.02,
                delta_percent=4.0,
            ),
            RefutationResult(
                test_name=RefutationTestType.RANDOM_COMMON_CAUSE,
                status=RefutationStatus.FAILED,
                original_effect=0.50,
                refuted_effect=0.15,
                delta_percent=70.0,
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
            passed=False,
            confidence_score=0.35,
            gate_decision=GateDecision.BLOCK,
            tests=tests,
        )

        outcome = create_validation_outcome(suite)

        assert outcome.outcome_type == ValidationOutcomeType.FAILED_CRITICAL

    def test_critical_set_matches_the_runner_config(self):
        """One source of truth: the runner config's ``critical`` flags.

        A hardcoded copy in this module drifts the moment a test's criticality
        changes -- which is exactly what happened to sensitivity_e_value on
        2026-09-10.
        """
        from src.causal_engine.refutation_runner import RefutationRunner
        from src.causal_engine.validation_outcome import _critical_test_names

        expected = {
            name
            for name, cfg in RefutationRunner.DEFAULT_CONFIG.items()
            if isinstance(cfg, dict) and cfg.get("critical")
        }

        assert _critical_test_names() == expected
        assert expected == {"placebo_treatment", "random_common_cause"}

    def test_generated_outcome_id_is_a_bare_uuid(self):
        """The minted outcome_id must be UUID-coercible (#1611).

        validation_outcomes.outcome_id is UUID PRIMARY KEY, so the ``vo_<12 hex>``
        id this used to mint made every Supabase insert fail 22P02 and silently
        degrade to the ephemeral in-memory fallback.
        """
        suite = RefutationSuite(
            passed=True,
            confidence_score=0.90,
            gate_decision=GateDecision.PROCEED,
            tests=[
                RefutationResult(
                    test_name=RefutationTestType.BOOTSTRAP,
                    status=RefutationStatus.PASSED,
                    original_effect=0.50,
                    refuted_effect=0.48,
                    delta_percent=4.0,
                )
            ],
            treatment_variable="rep_visits",
            outcome_variable="trx_total",
        )

        outcome = create_validation_outcome(suite)

        # Raises ValueError on the old ``vo_``-prefixed id.
        assert str(uuid.UUID(outcome.outcome_id)) == outcome.outcome_id

    def test_generated_outcome_ids_are_unique(self):
        """Distinct outcomes must not collide on the primary key."""
        suite = RefutationSuite(
            passed=True,
            confidence_score=0.90,
            gate_decision=GateDecision.PROCEED,
            tests=[
                RefutationResult(
                    test_name=RefutationTestType.BOOTSTRAP,
                    status=RefutationStatus.PASSED,
                    original_effect=0.50,
                    refuted_effect=0.48,
                    delta_percent=4.0,
                )
            ],
        )

        ids = {create_validation_outcome(suite).outcome_id for _ in range(50)}

        assert len(ids) == 50


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
                timestamp=f"2024-01-{15 + i}T10:00:00Z",
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


class TestRandomCommonCauseSeverityFollowsTheSeUnitsRule:
    """#2005: the rcc pattern's severity follows the verdict's own statistic.

    The retired label was ``"high" if delta_percent > 30 else "medium"`` on
    ``|Δ| / |ATE|``, which the runner no longer scores: a FAILED row now means
    the shift exceeded 2 SE of the reported interval, and the percentage can be
    huge on a null (measured 2026-09-11: 174 % at 0.34 SE). Severity reads
    ``details["shift_se_units"]``; a row persisted before the key existed
    follows its stored status; only a row with neither keeps the percentage.
    """

    @staticmethod
    def _rcc(status, delta_percent, details=None, original_effect=0.039):
        suite = RefutationSuite(
            passed=False,
            confidence_score=0.40,
            gate_decision=GateDecision.BLOCK,
            tests=[
                RefutationResult(
                    test_name=RefutationTestType.RANDOM_COMMON_CAUSE,
                    status=status,
                    original_effect=original_effect,
                    refuted_effect=0.045,
                    delta_percent=delta_percent,
                    details=details or {},
                )
            ],
        )
        return extract_failure_patterns(suite)[0]

    def test_the_persisted_shift_outranks_the_status(self):
        """Precedence pinned with CONFLICTING inputs: a status-only reading would
        invert both of these."""
        p = self._rcc(RefutationStatus.WARNING, 5.0, {"shift_se_units": 2.7})
        assert p.severity == "high"
        p = self._rcc(RefutationStatus.FAILED, 300.0, {"shift_se_units": 0.9})
        assert p.severity == "medium"

    def test_the_persisted_cutoff_that_produced_the_verdict_wins(self):
        """A runner with overridden thresholds persists them as ``thresholds_se``;
        severity follows THAT cutoff, not the class default (whole-diff codex F1)."""
        p = self._rcc(
            RefutationStatus.WARNING,
            5.0,
            {"shift_se_units": 2.35, "thresholds_se": {"pass": 3.0, "warning": 4.0}},
        )
        assert p.severity == "medium"  # 2.35 is inside the overridden warning band
        p = self._rcc(
            RefutationStatus.FAILED,
            5.0,
            {"shift_se_units": 1.57, "thresholds_se": {"pass": 0.5, "warning": 1.0}},
        )
        assert p.severity == "high"  # 1.57 is FAILED under the overridden cutoff
        # a row carrying the shift but no thresholds: the class default (2.0)
        assert self._rcc(RefutationStatus.WARNING, 5.0, {"shift_se_units": 2.35}).severity == "high"

    def test_the_cutoff_is_inclusive_at_exactly_two_se(self):
        assert (
            self._rcc(RefutationStatus.WARNING, 5.0, {"shift_se_units": 2.0}).severity == "medium"
        )
        assert (
            self._rcc(RefutationStatus.WARNING, 5.0, {"shift_se_units": 2.0000001}).severity
            == "high"
        )

    def test_description_names_the_refit_frame_scale_when_applied(self):
        p = self._rcc(
            RefutationStatus.WARNING,
            15.4,
            {"shift_se_units": 1.47, "reference_se_scale": 2.7391604553220317},
        )
        assert "1.47 SE" in p.description
        assert "scaled x2.74 to the refit frame" in p.description
        unscaled = self._rcc(
            RefutationStatus.WARNING, 15.4, {"shift_se_units": 1.47, "reference_se_scale": 1.0}
        )
        assert "scaled" not in unscaled.description

    def test_description_does_not_call_a_floored_percentage_a_share_of_a_zero_effect(self):
        p = self._rcc(
            RefutationStatus.FAILED,
            6.0e9,
            {"shift_se_units": 2.4},
            original_effect=0.0,
        )
        assert "2.40 SE" in p.description
        assert "% of the effect" not in p.description
        assert "floored denominator" in p.description

    def test_failed_beyond_two_se_is_high_even_at_a_small_percentage(self):
        p = self._rcc(
            RefutationStatus.FAILED, 15.4, {"shift_se_units": 2.35, "rule": "shift_vs_reported_se"}
        )
        assert p.severity == "high"
        assert p.category == FailureCategory.MODEL_MISSPECIFICATION
        assert "2.35 SE" in p.description
        assert "15.4" in p.description  # the percentage stays, as context

    def test_warning_band_is_medium_even_at_a_huge_percentage(self):
        p = self._rcc(
            RefutationStatus.WARNING, 174.0, {"shift_se_units": 1.4, "rule": "shift_vs_reported_se"}
        )
        assert p.severity == "medium"
        assert "1.40 SE" in p.description

    def test_legacy_row_without_the_key_follows_its_stored_status(self):
        assert self._rcc(RefutationStatus.FAILED, 12.0).severity == "high"
        assert self._rcc(RefutationStatus.WARNING, 95.0).severity == "medium"
        assert "SE" not in self._rcc(RefutationStatus.WARNING, 95.0).description
        assert "95.0%" in self._rcc(RefutationStatus.WARNING, 95.0).description
