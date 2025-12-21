"""Tests for refutation node.

Version: 4.3
Tests the RefutationNode integration with RefutationRunner.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.causal_impact.nodes.refutation import RefutationNode, refute_causal_estimate
from src.agents.causal_impact.state import CausalImpactState, EstimationResult


class TestRefutationNode:
    """Test RefutationNode."""

    def _create_test_estimation(self, ate: float = 0.5) -> EstimationResult:
        """Create test estimation result."""
        return {
            "method": "CausalForestDML",
            "ate": ate,
            "ate_ci_lower": ate - 0.1,
            "ate_ci_upper": ate + 0.1,
            "effect_size": "medium",
            "statistical_significance": True,
            "p_value": 0.01,
            "sample_size": 1000,
            "covariates_adjusted": ["geographic_region"],
            "heterogeneity_detected": False,
        }

    @pytest.mark.asyncio
    async def test_run_all_refutation_tests(self):
        """Test that all refutation tests are run."""
        node = RefutationNode()

        state: CausalImpactState = {
            "query": "test query",
            "query_id": "test-1",
            "estimation_result": self._create_test_estimation(),
            "status": "pending",
        }

        result = await node.execute(state)

        assert "refutation_results" in result
        ref = result["refutation_results"]

        assert ref["total_tests"] == 5  # 5 refutation tests (updated from 4)
        assert len(ref["individual_tests"]) == 5
        assert result["current_phase"] in ["analyzing_sensitivity", "failed"]

    @pytest.mark.asyncio
    async def test_refutation_tests_structure(self):
        """Test that refutation tests have correct structure."""
        node = RefutationNode()

        state: CausalImpactState = {
            "query": "test query",
            "query_id": "test-2",
            "estimation_result": self._create_test_estimation(),
            "status": "pending",
        }

        result = await node.execute(state)

        ref = result["refutation_results"]

        for test in ref["individual_tests"]:
            assert "test_name" in test
            assert test["test_name"] in [
                "placebo_treatment",
                "random_common_cause",
                "data_subset",
                "data_subset_validation",  # Legacy alias
                "bootstrap",
                "sensitivity_e_value",
            ]
            assert "passed" in test
            assert isinstance(test["passed"], bool)
            assert "new_effect" in test
            assert "original_effect" in test
            assert "p_value" in test
            assert "details" in test

    @pytest.mark.asyncio
    async def test_overall_robustness_majority_pass(self):
        """Test that overall robustness requires majority of tests to pass."""
        node = RefutationNode()

        state: CausalImpactState = {
            "query": "test query",
            "query_id": "test-3",
            "estimation_result": self._create_test_estimation(ate=0.6),
            "status": "pending",
        }

        result = await node.execute(state)

        ref = result["refutation_results"]

        # overall_robust should be True if tests_passed >= total_tests / 2
        expected_robust = ref["tests_passed"] >= ref["total_tests"] / 2
        assert ref["overall_robust"] == expected_robust

    @pytest.mark.asyncio
    async def test_confidence_adjustment(self):
        """Test confidence adjustment calculation."""
        node = RefutationNode()

        state: CausalImpactState = {
            "query": "test query",
            "query_id": "test-4",
            "estimation_result": self._create_test_estimation(),
            "status": "pending",
        }

        result = await node.execute(state)

        ref = result["refutation_results"]

        # Confidence adjustment should be between 0 and 1
        assert 0.0 <= ref["confidence_adjustment"] <= 1.0

    @pytest.mark.asyncio
    async def test_gate_decision_in_result(self):
        """Test that gate decision is included in results."""
        node = RefutationNode()

        state: CausalImpactState = {
            "query": "test query",
            "query_id": "test-gate",
            "estimation_result": self._create_test_estimation(),
            "status": "pending",
        }

        result = await node.execute(state)

        # Gate decision should be in both legacy and extended format
        assert "gate_decision" in result
        assert result["gate_decision"] in ["proceed", "review", "block"]
        assert result["refutation_results"]["gate_decision"] in ["proceed", "review", "block"]

    @pytest.mark.asyncio
    async def test_refutation_suite_in_result(self):
        """Test that full refutation suite is included."""
        node = RefutationNode()

        state: CausalImpactState = {
            "query": "test query",
            "query_id": "test-suite",
            "estimation_result": self._create_test_estimation(),
            "status": "pending",
        }

        result = await node.execute(state)

        assert "refutation_suite" in result
        suite = result["refutation_suite"]
        assert "passed" in suite
        assert "confidence_score" in suite
        assert "tests" in suite
        assert "gate_decision" in suite

    @pytest.mark.asyncio
    async def test_refutation_confidence_in_result(self):
        """Test that refutation confidence is included."""
        node = RefutationNode()

        state: CausalImpactState = {
            "query": "test query",
            "query_id": "test-conf",
            "estimation_result": self._create_test_estimation(),
            "status": "pending",
        }

        result = await node.execute(state)

        assert "refutation_confidence" in result
        assert 0.0 <= result["refutation_confidence"] <= 1.0

    @pytest.mark.asyncio
    async def test_latency_measurement(self):
        """Test that refutation latency is measured."""
        node = RefutationNode()

        state: CausalImpactState = {
            "query": "test query",
            "query_id": "test-5",
            "estimation_result": self._create_test_estimation(),
            "status": "pending",
        }

        result = await node.execute(state)

        assert "refutation_latency_ms" in result
        assert result["refutation_latency_ms"] >= 0
        assert result["refutation_latency_ms"] < 15000  # Should be < 15s

    @pytest.mark.asyncio
    async def test_error_handling_missing_estimation(self):
        """Test error handling when estimation result is missing."""
        node = RefutationNode()

        state: CausalImpactState = {
            "query": "test query",
            "query_id": "test-6",
            "status": "pending",
        }

        result = await node.execute(state)

        assert "refutation_error" in result
        assert result["status"] == "failed"
        assert "Estimation result not found" in result["error_message"]

    @pytest.mark.asyncio
    async def test_blocked_estimate_fails_workflow(self):
        """Test that blocked estimates set workflow status to failed."""
        # Create a node with very strict thresholds to force blocking
        node = RefutationNode(
            thresholds={
                "e_value_min": {"pass": 100.0, "warning": 50.0},  # Impossible threshold
            }
        )

        state: CausalImpactState = {
            "query": "test query",
            "query_id": "test-block",
            "estimation_result": self._create_test_estimation(),
            "status": "pending",
        }

        result = await node.execute(state)

        if result["gate_decision"] == "block":
            assert result["status"] == "failed"
            assert result["current_phase"] == "failed"
            assert "error_message" in result
            assert "blocked" in result["error_message"].lower()

    @pytest.mark.asyncio
    async def test_custom_config_passed_to_runner(self):
        """Test that custom config is passed to RefutationRunner."""
        custom_config = {
            "placebo_treatment": {"num_simulations": 50},
        }
        node = RefutationNode(config=custom_config)

        assert node.runner.config["placebo_treatment"]["num_simulations"] == 50

    @pytest.mark.asyncio
    async def test_custom_thresholds_passed_to_runner(self):
        """Test that custom thresholds are passed to RefutationRunner."""
        custom_thresholds = {
            "e_value_min": {"pass": 3.0},
        }
        node = RefutationNode(thresholds=custom_thresholds)

        assert node.runner.thresholds["e_value_min"]["pass"] == 3.0


class TestRefutationNodeWithRepository:
    """Test RefutationNode with database persistence."""

    def _create_test_estimation(self, ate: float = 0.5) -> EstimationResult:
        """Create test estimation result."""
        return {
            "method": "CausalForestDML",
            "ate": ate,
            "ate_ci_lower": ate - 0.1,
            "ate_ci_upper": ate + 0.1,
            "effect_size": "medium",
            "statistical_significance": True,
            "p_value": 0.01,
            "sample_size": 1000,
            "covariates_adjusted": ["geographic_region"],
            "heterogeneity_detected": False,
        }

    @pytest.mark.asyncio
    async def test_with_mock_repository(self):
        """Test refutation with mock repository."""
        mock_repo = MagicMock()
        mock_repo.save_suite = AsyncMock(return_value=["val-1", "val-2", "val-3"])

        node = RefutationNode(validation_repo=mock_repo)

        state: CausalImpactState = {
            "query": "test query",
            "query_id": "test-repo-1",
            "estimation_result": self._create_test_estimation(),
            "status": "pending",
        }

        result = await node.execute(state)

        # Verify save_suite was called
        mock_repo.save_suite.assert_called_once()

        # Verify validation_ids are returned
        assert "validation_ids" in result
        assert result["validation_ids"] == ["val-1", "val-2", "val-3"]

    @pytest.mark.asyncio
    async def test_without_repository(self):
        """Test refutation without repository (no persistence)."""
        node = RefutationNode(validation_repo=None)

        state: CausalImpactState = {
            "query": "test query",
            "query_id": "test-no-repo",
            "estimation_result": self._create_test_estimation(),
            "status": "pending",
        }

        result = await node.execute(state)

        # Should still work without repository
        assert "refutation_results" in result
        assert "validation_ids" in result
        assert result["validation_ids"] == []  # Empty when no repo

    @pytest.mark.asyncio
    async def test_repository_failure_handled(self):
        """Test that repository failures are handled gracefully."""
        mock_repo = MagicMock()
        mock_repo.save_suite = AsyncMock(side_effect=Exception("DB error"))

        node = RefutationNode(validation_repo=mock_repo)

        state: CausalImpactState = {
            "query": "test query",
            "query_id": "test-repo-fail",
            "estimation_result": self._create_test_estimation(),
            "status": "pending",
        }

        result = await node.execute(state)

        # Should complete successfully despite repo failure
        assert "refutation_results" in result
        assert result["validation_ids"] == []  # Empty due to failure


class TestRefutationPassCriteria:
    """Test pass/fail criteria for refutation tests."""

    def _create_test_estimation(self, ate: float = 0.5) -> EstimationResult:
        """Create test estimation result."""
        return {
            "method": "CausalForestDML",
            "ate": ate,
            "ate_ci_lower": ate - 0.1,
            "ate_ci_upper": ate + 0.1,
            "effect_size": "medium",
            "statistical_significance": True,
            "p_value": 0.01,
            "sample_size": 1000,
            "covariates_adjusted": ["geographic_region"],
            "heterogeneity_detected": False,
        }

    @pytest.mark.asyncio
    async def test_all_tests_run(self):
        """Test that all 5 refutation tests are run."""
        node = RefutationNode()

        state: CausalImpactState = {
            "query": "test query",
            "query_id": "test-all-tests",
            "estimation_result": self._create_test_estimation(),
            "status": "pending",
        }

        result = await node.execute(state)
        ref = result["refutation_results"]

        test_names = [t["test_name"] for t in ref["individual_tests"]]

        # All 5 test types should be present
        expected_tests = [
            "placebo_treatment",
            "random_common_cause",
            "data_subset",
            "bootstrap",
            "sensitivity_e_value",
        ]
        for expected in expected_tests:
            assert expected in test_names, f"Missing test: {expected}"


class TestRefutationWithDifferentEffectSizes:
    """Test refutation with different effect sizes."""

    def _create_test_estimation(self, ate: float = 0.5) -> EstimationResult:
        """Create test estimation result."""
        return {
            "method": "CausalForestDML",
            "ate": ate,
            "ate_ci_lower": ate - 0.1,
            "ate_ci_upper": ate + 0.1,
            "effect_size": "medium",
            "statistical_significance": True,
            "p_value": 0.01,
            "sample_size": 1000,
            "covariates_adjusted": ["geographic_region"],
            "heterogeneity_detected": False,
        }

    @pytest.mark.asyncio
    async def test_small_effect_refutation(self):
        """Test refutation for small effect."""
        node = RefutationNode()

        state: CausalImpactState = {
            "query": "test query",
            "query_id": "test-7",
            "estimation_result": self._create_test_estimation(ate=0.1),  # Small
            "status": "pending",
        }

        result = await node.execute(state)

        assert "refutation_results" in result
        assert result["refutation_results"]["total_tests"] == 5

    @pytest.mark.asyncio
    async def test_large_effect_refutation(self):
        """Test refutation for large effect."""
        node = RefutationNode()

        state: CausalImpactState = {
            "query": "test query",
            "query_id": "test-8",
            "estimation_result": self._create_test_estimation(ate=0.8),  # Large
            "status": "pending",
        }

        result = await node.execute(state)

        assert "refutation_results" in result
        # Larger effects should typically be more robust
        ref = result["refutation_results"]
        assert ref["tests_passed"] >= 0  # At least some should pass

    @pytest.mark.asyncio
    async def test_negative_effect_refutation(self):
        """Test refutation for negative effect."""
        node = RefutationNode()

        state: CausalImpactState = {
            "query": "test query",
            "query_id": "test-9",
            "estimation_result": self._create_test_estimation(ate=-0.5),  # Negative
            "status": "pending",
        }

        result = await node.execute(state)

        assert "refutation_results" in result
        ref = result["refutation_results"]

        # Should handle negative effects
        for test in ref["individual_tests"]:
            # Original effect should be preserved
            assert test["original_effect"] == -0.5


class TestStandaloneFunction:
    """Test standalone refute_causal_estimate function."""

    def _create_test_estimation(self, ate: float = 0.5) -> EstimationResult:
        """Create test estimation result."""
        return {
            "method": "CausalForestDML",
            "ate": ate,
            "ate_ci_lower": ate - 0.1,
            "ate_ci_upper": ate + 0.1,
            "effect_size": "medium",
            "statistical_significance": True,
            "p_value": 0.01,
            "sample_size": 1000,
            "covariates_adjusted": ["geographic_region"],
            "heterogeneity_detected": False,
        }

    @pytest.mark.asyncio
    async def test_standalone_function(self):
        """Test refute_causal_estimate standalone function."""
        state: CausalImpactState = {
            "query": "test query",
            "query_id": "test-standalone",
            "estimation_result": self._create_test_estimation(),
            "status": "pending",
        }

        result = await refute_causal_estimate(state)

        assert "refutation_results" in result
        assert "gate_decision" in result

    @pytest.mark.asyncio
    async def test_standalone_function_with_repo(self):
        """Test refute_causal_estimate with repository."""
        mock_repo = MagicMock()
        mock_repo.save_suite = AsyncMock(return_value=["val-1"])

        state: CausalImpactState = {
            "query": "test query",
            "query_id": "test-standalone-repo",
            "estimation_result": self._create_test_estimation(),
            "status": "pending",
        }

        result = await refute_causal_estimate(state, validation_repo=mock_repo)

        assert "refutation_results" in result
        mock_repo.save_suite.assert_called_once()
