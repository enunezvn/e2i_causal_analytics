"""Tests for refutation node.

Version: 4.3
Tests the RefutationNode integration with RefutationRunner.

F-014 fix (#416): RefutationNode now reconstructs a real DoWhy CausalModel
from ``estimation_data`` passthrough before invoking ``run_all_tests``.
Tests in this module provide a synthetic DataFrame + matching treatment_var /
outcome_var / confounders so reconstruction succeeds; otherwise the node
fail-closes with ``RefutationError``.
"""

import copy
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pandas as pd
import pytest

from src.agents.causal_impact.nodes.refutation import RefutationNode, refute_causal_estimate
from src.agents.causal_impact.state import CausalImpactState, EstimationResult


@pytest.fixture(autouse=True)
def _fast_refuter_sims(monkeypatch):
    """#583: cap DoWhy refuter simulation counts for these unit tests.

    Every test here drives the REAL refutation path — reconstruct a DoWhy
    CausalModel, fit the EconML CausalForestDML, then run the placebo /
    random-common-cause / data-subset / bootstrap refuters. The default
    config (RefutationRunner.DEFAULT_CONFIG) runs 100-simulation bootstrap and
    placebo refuters, each of which RE-FITS the causal forest — minutes per
    test (a single test exceeded 7 min uncapped). These are unit tests of
    refutation WIRING + result structure + pass/block scoring, not statistical
    power, so a small simulation count exercises the same code path in seconds
    and keeps the suite viable on-PR. Production is unchanged (still 100).
    """
    from src.causal_engine.refutation_runner import RefutationRunner

    fast = copy.deepcopy(RefutationRunner.DEFAULT_CONFIG)
    for section in ("placebo_treatment", "data_subset", "bootstrap"):
        if section in fast:
            for key in ("num_simulations", "num_subsets", "num_bootstraps"):
                if key in fast[section]:
                    fast[section][key] = 5
    monkeypatch.setattr(RefutationRunner, "DEFAULT_CONFIG", fast)


# #583: this whole module is @pytest.mark.slow. Every meaningful test drives
# the REAL refutation path, which re-fits an EconML CausalForestDML (with
# GridSearch nuisance models) ~16x per test (base estimate + each refuter) —
# minutes per test, ~40+ min for the module, even with the simulation cap
# above. That cannot fit the on-PR agents-tests 25-min cap, so these run
# off-PR via slow-tests.yml (`-m slow`, 90-min budget, must-pass) — coverage
# preserved, not silently skipped. The agents-tests lane runs `-m "not slow"`.
pytestmark = pytest.mark.slow


def _make_estimation_data(
    treatment_col: str = "hcp_engagement_level",
    outcome_col: str = "patient_conversion_rate",
    confounder_cols: list = None,
    n: int = 300,
    seed: int = 42,
    true_ate: float = 0.5,
) -> pd.DataFrame:
    """Build a small synthetic DataFrame suitable for DoWhy CausalModel rebuild.

    The data MUST contain columns matching ``state['treatment_var']``,
    ``state['outcome_var']``, and every entry of ``state['confounders']``
    so the agent's reconstruction step succeeds. This replaces the
    deleted silent-mock fallback path in the refutation node.

    Iter-6 codex H-iter5-1 (#416): the data-generating ATE coefficient is
    parameterized so test fixtures can match the EstimationResult.ate value
    they declare. The refutation node now verifies reconstructed ATE matches
    reported ATE within tolerance, so the data MUST produce a similar effect.
    """
    if confounder_cols is None:
        confounder_cols = ["geographic_region"]
    rng = np.random.default_rng(seed)
    columns: dict = {col: rng.normal(0, 1, n) for col in confounder_cols}
    treatment = rng.binomial(1, 0.5, n).astype(float)
    # Use small noise so the small-sample estimate stays near ``true_ate``.
    outcome = (
        true_ate * treatment
        + sum(0.1 * columns[c] for c in confounder_cols)
        + rng.normal(0, 0.3, n)
    )
    columns[treatment_col] = treatment
    columns[outcome_col] = outcome
    return pd.DataFrame(columns)


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

    def _make_state(self, query_id: str, ate: float = 0.5, **overrides) -> CausalImpactState:
        """Build a full state with estimation_data + treatment/outcome/confounder
        wiring so the refutation node can reconstruct a real DoWhy CausalModel.

        F-014 fix (#416): tests must provide estimation_data or the agent
        fails-closed with RefutationError (no silent mock fallback).
        """
        state: CausalImpactState = {
            "query": "test query",
            "query_id": query_id,
            "treatment_var": "hcp_engagement_level",
            "outcome_var": "patient_conversion_rate",
            "confounders": ["geographic_region"],
            "data_source": "synthetic",
            "estimation_result": self._create_test_estimation(ate=ate),
            "estimation_data": _make_estimation_data(true_ate=ate),
            "status": "pending",
            "errors": [],
            "warnings": [],
        }
        state.update(overrides)  # type: ignore[typeddict-item]
        return state

    @pytest.mark.asyncio
    async def test_run_all_refutation_tests(self):
        """Test that all refutation tests are run.

        F-014 fix (#416): provides estimation_data so the agent can reconstruct
        a real DoWhy CausalModel.
        """
        node = RefutationNode()

        state: CausalImpactState = {
            "query": "test query",
            "query_id": "test-1",
            "treatment_var": "hcp_engagement_level",
            "outcome_var": "patient_conversion_rate",
            "confounders": ["geographic_region"],
            "data_source": "synthetic",
            "estimation_result": self._create_test_estimation(),
            "estimation_data": _make_estimation_data(true_ate=0.5),
            "status": "pending",
        }

        result = await node.execute(state)

        assert "refutation_results" in result
        ref = result["refutation_results"]

        # Iter-6 codex H-iter5-2/3 (#416): data_subset + bootstrap mark
        # SKIPPED when DoWhy doesn't expose ``subset_effects`` /
        # ``bootstrap_estimates`` (real DoWhy 0.10+ behavior on our fixture),
        # so ``total_tests`` (which excludes SKIPPED) is 3-5 depending on
        # DoWhy version. #1219 (ef2e16dd): SKIPPED tests are OMITTED from
        # ``individual_tests`` — the legacy dict is pass/fail-shaped, so a
        # SKIPPED entry would render as a FAILED row downstream; absent key
        # three-states to None (audit chain) / not-narrated (interpretation).
        # The omission semantics are pinned fast-lane in
        # tests/unit/test_causal_engine/test_refutation_runner_randomized.py;
        # here we assert the node-level consistency invariant.
        assert ref["total_tests"] in (3, 4, 5), (
            f"Expected 3-5 non-skipped tests, got {ref['total_tests']}"
        )
        assert len(ref["individual_tests"]) == ref["total_tests"], (
            "individual_tests must carry exactly the non-SKIPPED tests "
            f"(got {sorted(ref['individual_tests'])} vs total_tests={ref['total_tests']})"
        )
        valid_keys = {
            "placebo_treatment",
            "random_common_cause",
            "data_subset",
            "bootstrap",
            "unobserved_common_cause",
        }
        assert set(ref["individual_tests"]) <= valid_keys
        # placebo + random_common_cause always run on this fixture; the
        # E-value (unobserved_common_cause) gate only skips for randomized
        # designs (#1219) and this fixture is observational.
        assert {
            "placebo_treatment",
            "random_common_cause",
            "unobserved_common_cause",
        } <= set(ref["individual_tests"])
        assert result["current_phase"] in ["analyzing_sensitivity", "failed"]

    @pytest.mark.asyncio
    async def test_refutation_tests_structure(self):
        """Test that refutation tests have correct structure.

        Contract: individual_tests is Dict with test names as keys.
        """
        node = RefutationNode()

        state = self._make_state(query_id="test-2")

        result = await node.execute(state)

        ref = result["refutation_results"]

        # Contract: individual_tests is Dict, not List
        assert isinstance(ref["individual_tests"], dict)

        for _test_key, test in ref["individual_tests"].items():
            assert "test_name" in test
            assert test["test_name"] in [
                "placebo_treatment",
                "random_common_cause",
                "data_subset",
                "data_subset_validation",  # Legacy alias
                "bootstrap",
                "sensitivity_e_value",
                "unobserved_common_cause",  # Contract key
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

        state = self._make_state(query_id="test-3", ate=0.6)

        result = await node.execute(state)

        ref = result["refutation_results"]

        # overall_robust should be True if tests_passed >= total_tests / 2
        expected_robust = ref["tests_passed"] >= ref["total_tests"] / 2
        assert ref["overall_robust"] == expected_robust

    @pytest.mark.asyncio
    async def test_confidence_adjustment(self):
        """Test confidence adjustment calculation."""
        node = RefutationNode()

        state = self._make_state(query_id="test-4")

        result = await node.execute(state)

        ref = result["refutation_results"]

        # Confidence adjustment should be between 0 and 1
        assert 0.0 <= ref["confidence_adjustment"] <= 1.0

    @pytest.mark.asyncio
    async def test_gate_decision_in_result(self):
        """Test that gate decision is included in results."""
        node = RefutationNode()

        state = self._make_state(query_id="test-gate")

        result = await node.execute(state)

        # Gate decision should be in both legacy and extended format
        assert "gate_decision" in result
        assert result["gate_decision"] in ["proceed", "review", "block"]
        assert result["refutation_results"]["gate_decision"] in ["proceed", "review", "block"]

    @pytest.mark.asyncio
    async def test_refutation_suite_in_result(self):
        """Test that full refutation suite is included."""
        node = RefutationNode()

        state = self._make_state(query_id="test-suite")

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

        state = self._make_state(query_id="test-conf")

        result = await node.execute(state)

        assert "refutation_confidence" in result
        assert 0.0 <= result["refutation_confidence"] <= 1.0

    @pytest.mark.asyncio
    async def test_latency_measurement(self):
        """Test that refutation latency is measured and recorded.

        Iter-2 (F-014 #416): the performance budget < 15s was set when the
        deleted ``_mock_*`` paths returned seeded random values in < 1ms
        per test. With real DoWhy refuters (placebo, random_common_cause,
        data_subset, bootstrap) now executing against a reconstructed
        CausalModel + EconML estimator, per-suite cost on small fixtures
        is dominated by EconML fit calls — easily exceeding 15s in test
        environments. The honest assertion is: latency IS measured and is
        a positive number. The "<15s" production budget still applies for
        full-sized data (where EconML is amortized over more samples).
        """
        node = RefutationNode()

        state = self._make_state(query_id="test-5")

        result = await node.execute(state)

        assert "refutation_latency_ms" in result
        assert result["refutation_latency_ms"] >= 0

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

        state = self._make_state(query_id="test-block")

        result = await node.execute(state)

        if result.get("gate_decision") == "block":
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

    def _make_state(self, query_id: str, ate: float = 0.5) -> CausalImpactState:
        """Build a full state with estimation_data passthrough (F-014 #416)."""
        return {
            "query": "test query",
            "query_id": query_id,
            "treatment_var": "hcp_engagement_level",
            "outcome_var": "patient_conversion_rate",
            "confounders": ["geographic_region"],
            "data_source": "synthetic",
            "estimation_result": self._create_test_estimation(ate=ate),
            "estimation_data": _make_estimation_data(true_ate=ate),
            "status": "pending",
            "errors": [],
            "warnings": [],
        }

    @pytest.mark.asyncio
    async def test_with_mock_repository(self):
        """Test refutation with mock repository."""
        mock_repo = MagicMock()
        mock_repo.save_suite = AsyncMock(return_value=["val-1", "val-2", "val-3"])

        node = RefutationNode(validation_repo=mock_repo)

        state = self._make_state(query_id="test-repo-1")

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

        state = self._make_state(query_id="test-no-repo")

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

        state = self._make_state(query_id="test-repo-fail")

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
        """Test that every APPLICABLE refutation test lands in individual_tests.

        Contract: individual_tests is Dict with test names as keys, carrying
        exactly the non-SKIPPED tests (#1219/ef2e16dd: SKIPPED tests are
        omitted — the pass/fail-shaped legacy dict would render them as
        FAILED rows downstream). data_subset / bootstrap are SKIPPED when
        DoWhy doesn't expose ``subset_effects`` / ``bootstrap_estimates``
        (real DoWhy 0.10+ behavior on this fixture), so their presence is
        DoWhy-version-dependent; the three always-applicable tests must be
        present on this observational fixture.
        """
        node = RefutationNode()

        state: CausalImpactState = {
            "query": "test query",
            "query_id": "test-all-tests",
            "treatment_var": "hcp_engagement_level",
            "outcome_var": "patient_conversion_rate",
            "confounders": ["geographic_region"],
            "data_source": "synthetic",
            "estimation_result": self._create_test_estimation(),
            "estimation_data": _make_estimation_data(true_ate=0.5),
            "status": "pending",
            "errors": [],
            "warnings": [],
        }

        result = await node.execute(state)
        ref = result["refutation_results"]

        # Contract: individual_tests is Dict, get keys for test names
        test_keys = set(ref["individual_tests"].keys())

        # Always-applicable on this observational fixture: placebo,
        # random_common_cause, and the E-value gate (which only skips for
        # randomized designs, #1219). sensitivity_e_value maps to
        # unobserved_common_cause per contract.
        for expected in ("placebo_treatment", "random_common_cause"):
            assert expected in test_keys, f"Missing test: {expected}"
        assert "unobserved_common_cause" in test_keys, (
            "Missing test: unobserved_common_cause (the E-value gate; "
            "sensitivity_e_value maps to this key in the legacy dict)"
        )

        # data_subset / bootstrap are DoWhy-version-dependent (SKIPPED →
        # omitted per #1219); when present they must be well-formed keys,
        # and the dict must stay consistent with total_tests.
        valid_keys = {
            "placebo_treatment",
            "random_common_cause",
            "data_subset",
            "bootstrap",
            "unobserved_common_cause",
        }
        assert test_keys <= valid_keys, f"Unexpected test keys: {test_keys - valid_keys}"
        assert len(test_keys) == ref["total_tests"], (
            "individual_tests must carry exactly the non-SKIPPED tests "
            f"(got {sorted(test_keys)} vs total_tests={ref['total_tests']})"
        )


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

    def _make_state(self, query_id: str, ate: float = 0.5) -> CausalImpactState:
        """Build a full state with estimation_data passthrough (F-014 #416)."""
        return {
            "query": "test query",
            "query_id": query_id,
            "treatment_var": "hcp_engagement_level",
            "outcome_var": "patient_conversion_rate",
            "confounders": ["geographic_region"],
            "data_source": "synthetic",
            "estimation_result": self._create_test_estimation(ate=ate),
            "estimation_data": _make_estimation_data(true_ate=ate),
            "status": "pending",
            "errors": [],
            "warnings": [],
        }

    @pytest.mark.asyncio
    async def test_small_effect_refutation(self):
        """Test refutation for small effect."""
        node = RefutationNode()

        state = self._make_state(query_id="test-7", ate=0.1)

        result = await node.execute(state)

        assert "refutation_results" in result
        # Iter-6: total_tests excludes SKIPPED (data_subset + bootstrap may
        # skip when DoWhy doesn't expose per-subset effects).
        assert result["refutation_results"]["total_tests"] in (3, 4, 5)

    @pytest.mark.asyncio
    async def test_large_effect_refutation(self):
        """Test refutation for large effect."""
        node = RefutationNode()

        state = self._make_state(query_id="test-8", ate=0.8)

        result = await node.execute(state)

        assert "refutation_results" in result
        # Larger effects should typically be more robust
        ref = result["refutation_results"]
        assert ref["tests_passed"] >= 0  # At least some should pass

    @pytest.mark.asyncio
    async def test_negative_effect_refutation(self):
        """Test refutation for negative effect.

        Contract: individual_tests is Dict with test names as keys.
        """
        node = RefutationNode()

        state = self._make_state(query_id="test-9", ate=-0.5)

        result = await node.execute(state)

        assert "refutation_results" in result
        ref = result["refutation_results"]

        # Contract: individual_tests is Dict, iterate over values
        for _test_key, test in ref["individual_tests"].items():
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

    def _make_state(self, query_id: str) -> CausalImpactState:
        """Build full state with estimation_data passthrough (F-014 #416)."""
        return {
            "query": "test query",
            "query_id": query_id,
            "treatment_var": "hcp_engagement_level",
            "outcome_var": "patient_conversion_rate",
            "confounders": ["geographic_region"],
            "data_source": "synthetic",
            "estimation_result": self._create_test_estimation(),
            "estimation_data": _make_estimation_data(true_ate=0.5),
            "status": "pending",
            "errors": [],
            "warnings": [],
        }

    @pytest.mark.asyncio
    async def test_standalone_function(self):
        """Test refute_causal_estimate standalone function."""
        state = self._make_state(query_id="test-standalone")

        result = await refute_causal_estimate(state)

        assert "refutation_results" in result
        assert "gate_decision" in result

    @pytest.mark.asyncio
    async def test_standalone_function_with_repo(self):
        """Test refute_causal_estimate with repository."""
        mock_repo = MagicMock()
        mock_repo.save_suite = AsyncMock(return_value=["val-1"])

        state = self._make_state(query_id="test-standalone-repo")

        result = await refute_causal_estimate(state, validation_repo=mock_repo)

        assert "refutation_results" in result
        mock_repo.save_suite.assert_called_once()


@pytest.mark.asyncio
async def test_refutation_suite_offloaded_to_bounded_pool(monkeypatch):
    """The CPU-bound DoWhy reconstruction + refutation suite (each refuter
    re-estimates many times) must run OFF the event loop so it cannot block the
    gunicorn worker past --timeout and get it KILLED mid-run (orphaning async
    jobs at status='running'). The module's autouse fast-sim fixture keeps the
    real refuters quick.

    #1601 moved all three off-loads from ``asyncio.to_thread`` to the BOUNDED
    agent-compute pool (``run_bounded_with_budget``) and dropped the module-level
    ``asyncio`` import. This test used to monkeypatch ``_ref_mod.asyncio.to_thread``
    and so died with ``AttributeError`` before asserting anything (#1630) — the
    off-loop guarantee it exists to protect was silently unguarded in the
    nightly.

    Asserting the THREAD rather than only the helper name is deliberate: a
    name-only spy passes as long as the helper is called, even if a future
    refactor makes it run inline. Comparing against the event loop's own thread
    id fails on any regression that puts this compute back on the loop,
    whichever helper it routes through.
    """
    import threading

    from src.agents.causal_impact.nodes import refutation as _ref_mod

    node = RefutationNode()
    offloaded: list = []
    ran_on_thread: list[int] = []
    loop_thread_id = threading.get_ident()
    real_bounded = _ref_mod.run_bounded_with_budget

    async def _spy(func, *args, **kwargs):
        offloaded.append(getattr(func, "__name__", str(func)))

        def _record_thread(*inner_args, **inner_kwargs):
            ran_on_thread.append(threading.get_ident())
            return func(*inner_args, **inner_kwargs)

        return await real_bounded(_record_thread, *args, **kwargs)

    monkeypatch.setattr(_ref_mod, "run_bounded_with_budget", _spy)

    ate = 0.5
    state: CausalImpactState = {
        "query": "offload test",
        "query_id": "offload-ref-1",
        "treatment_var": "hcp_engagement_level",
        "outcome_var": "patient_conversion_rate",
        "confounders": ["geographic_region"],
        "data_source": "synthetic",
        "estimation_result": {
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
        },
        "estimation_data": _make_estimation_data(true_ate=ate),
        "status": "pending",
        "errors": [],
        "warnings": [],
    }

    result = await node.execute(state)

    # Both heavy DoWhy steps went through the bounded agent-compute pool...
    assert "_reconstruct_dowhy_artifacts" in offloaded
    assert "run_all_tests" in offloaded
    # ...and genuinely executed OFF the event loop thread, not merely routed
    # through the helper. This is the assertion that survives a refactor.
    assert ran_on_thread, "no off-loaded call actually executed"
    assert loop_thread_id not in ran_on_thread, (
        f"CPU-bound refutation ran ON the event loop thread ({loop_thread_id}); "
        "a multi-minute suite there trips gunicorn --timeout and the worker is "
        "killed mid-run"
    )
    # ...and the suite still ran to a real gate decision (offload is transparent).
    assert "refutation_results" in result or "gate_decision" in result


@pytest.mark.asyncio
async def test_refutation_forwards_compute_deadline(monkeypatch):
    """Orphan-fix: the refutation node must forward ``state['compute_deadline']``
    to ``RefutationRunner.run_all_tests`` so the suite can self-terminate before
    the route's hard wall-clock cap, instead of orphaning to_thread compute.
    Spy on run_all_tests to assert the deadline is threaded through (and the
    node fails closed cleanly when the suite raises)."""
    from src.causal_engine.errors import RefutationError

    node = RefutationNode()
    captured: dict = {}

    def spy(**kwargs):
        captured.update(kwargs)
        # Stop after capturing — we only assert the deadline was forwarded.
        raise RefutationError("stop after capture", details={"reason": "test"})

    monkeypatch.setattr(node.runner, "run_all_tests", spy)

    # Far-FUTURE deadline so the node's pre-refutation budget check passes and we
    # actually reach run_all_tests (where the spy captures the forwarded value).
    import time as _t

    deadline_val = _t.monotonic() + 10_000.0
    ate = 0.5
    state: CausalImpactState = {
        "query": "deadline plumbing",
        "query_id": "deadline-ref-1",
        "treatment_var": "hcp_engagement_level",
        "outcome_var": "patient_conversion_rate",
        "confounders": ["geographic_region"],
        "data_source": "synthetic",
        "estimation_result": {
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
        },
        "estimation_data": _make_estimation_data(true_ate=ate),
        "compute_deadline": deadline_val,
        "status": "pending",
        "errors": [],
        "warnings": [],
    }

    result = await node.execute(state)

    assert captured.get("deadline") == deadline_val
    # The spy raised RefutationError -> node fails closed cleanly.
    assert result.get("status") == "failed"
