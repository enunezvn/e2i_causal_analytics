"""Regression tests pinning fail-closed semantics for F-014 + F-006.

These tests assert the contract that:

F-014: ``RefutationNode.execute`` MUST construct a real ``CausalModel`` and
       pass it into ``RefutationRunner.run_all_tests``. It MUST NOT call
       ``run_all_tests`` with ``causal_model=None``, which would silently
       dispatch to the ``_mock_*`` paths.

F-006: ``EstimationNode._estimate_*`` legacy methods MUST raise
       ``EstimationError`` (fail-closed) instead of returning
       ``np.corrcoef``-based mock values.

These pins protect against future regressions to silent-fallback behavior
per ``CLAUDE.md`` §"CRITICAL — Anti-Mocking & Verification Discipline".

References:
- #416 (F-014 refutation_runner mock paths)
- #417 (F-006 estimation legacy mocks)
- Parent: #354 (causal-engine canonical-routing)
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.agents.causal_impact.nodes.estimation import EstimationNode
from src.agents.causal_impact.nodes.refutation import RefutationNode
from src.agents.causal_impact.state import CausalGraph, CausalImpactState, EstimationResult

# ============================================================================
# F-014: refutation node MUST pass real CausalModel, not None
# ============================================================================


def _make_estimation_result(ate: float = 0.25) -> EstimationResult:
    return {
        "method": "CausalForestDML",
        "ate": ate,
        "ate_ci_lower": ate - 0.05,
        "ate_ci_upper": ate + 0.05,
        "standard_error": 0.025,
        "effect_size": "medium",
        "statistical_significance": True,
        "p_value": 0.01,
        "sample_size": 1000,
        "covariates_adjusted": ["geographic_region"],
        "heterogeneity_detected": False,
    }


@pytest.mark.asyncio
async def test_f014_refutation_node_does_not_pass_none_causal_model():
    """F-014 RED pin: refutation must reconstruct CausalModel from state.

    Asserts: after running ``RefutationNode.execute``, the call into
    ``RefutationRunner.run_all_tests`` MUST have ``causal_model`` that is
    NOT None (i.e., reconstructed) OR the node MUST fail-closed (raise/log
    a structured error).
    """
    node = RefutationNode()

    # Provide full data passthrough as estimation node would
    import numpy as np
    import pandas as pd

    np.random.seed(123)
    n = 200
    data = pd.DataFrame(
        {
            "hcp_engagement_level": np.random.normal(0, 1, n),
            "patient_conversion_rate": np.random.normal(0, 1, n),
            "geographic_region": np.random.randint(0, 2, n).astype(float),
        }
    )

    state: CausalImpactState = {
        "query": "test",
        "query_id": "test-f014-1",
        "treatment_var": "hcp_engagement_level",
        "outcome_var": "patient_conversion_rate",
        "confounders": ["geographic_region"],
        "data_source": "synthetic",
        "estimation_result": _make_estimation_result(),
        "estimation_data": data,
        "status": "pending",
    }

    # Spy on run_all_tests to capture the causal_model argument
    captured_kwargs: dict = {}
    original_run_all_tests = node.runner.run_all_tests

    def spy(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return original_run_all_tests(*args, **kwargs)

    with patch.object(node.runner, "run_all_tests", side_effect=spy):
        result = await node.execute(state)

    # Contract: refutation must either (a) pass a real CausalModel OR
    # (b) fail-closed with a structured error.
    refutation_error = result.get("refutation_error")
    if refutation_error is not None:
        # Fail-closed path is acceptable
        assert "refutation" in refutation_error.lower() or "service" in refutation_error.lower()
    else:
        # If we have a successful run, causal_model MUST NOT be None
        # (otherwise we'd be hitting the _mock_* paths silently)
        assert captured_kwargs.get("causal_model") is not None, (
            "F-014 regression: refutation node called run_all_tests with "
            "causal_model=None, which silently dispatches to _mock_* paths. "
            "Either reconstruct CausalModel from estimation_data OR fail-closed "
            "with a RefutationError."
        )


@pytest.mark.asyncio
async def test_f014_refutation_node_fails_closed_when_data_missing():
    """F-014 RED pin: if estimation_data is missing, refutation must fail-closed.

    Reconstruction requires data; if data is missing, the node MUST NOT
    silently fall back to the mock paths.
    """
    node = RefutationNode()

    state: CausalImpactState = {
        "query": "test",
        "query_id": "test-f014-2",
        "treatment_var": "hcp_engagement_level",
        "outcome_var": "patient_conversion_rate",
        "confounders": ["geographic_region"],
        "data_source": "synthetic",
        "estimation_result": _make_estimation_result(),
        # estimation_data deliberately absent
        "status": "pending",
    }

    result = await node.execute(state)

    # Must fail-closed
    assert result.get("refutation_error") is not None or result.get("status") == "failed", (
        "F-014 regression: missing estimation_data did NOT cause fail-closed. "
        "Refutation node silently dispatched to mock paths or proceeded with "
        "None CausalModel."
    )


# ============================================================================
# F-006: estimation node legacy _estimate_* methods MUST fail-closed
# ============================================================================


class TestF006FailClosed:
    """F-006 RED pins: legacy _estimate_* methods must raise EstimationError."""

    def _make_state_with_method(self, method: str) -> CausalImpactState:
        graph: CausalGraph = {
            "nodes": ["hcp_engagement_level", "patient_conversion_rate", "geographic_region"],
            "edges": [
                ("geographic_region", "hcp_engagement_level"),
                ("geographic_region", "patient_conversion_rate"),
                ("hcp_engagement_level", "patient_conversion_rate"),
            ],
            "treatment_nodes": ["hcp_engagement_level"],
            "outcome_nodes": ["patient_conversion_rate"],
            "adjustment_sets": [["geographic_region"]],
            "dag_dot": "...",
            "confidence": 0.85,
        }
        return {
            "query": "test",
            "query_id": f"test-f006-{method}",
            "treatment_var": "hcp_engagement_level",
            "outcome_var": "patient_conversion_rate",
            "confounders": ["geographic_region"],
            "data_source": "synthetic",
            "causal_graph": graph,
            "parameters": {"method": method},
            "status": "pending",
            "errors": [],
            "warnings": [],
        }

    @pytest.mark.asyncio
    async def test_f006_legacy_causal_forest_does_not_return_corrcoef_ate(self):
        """F-006 RED pin: _estimate_causal_forest must not return np.corrcoef-based ATE.

        When energy_score=False AND method=CausalForestDML:
        - OLD behavior: returns ate = corrcoef(t,o) * std(o), ate_se = 0.05 (mock)
        - NEW behavior: delegates to energy-score selector; on failure raises
          EstimationError (fail-closed).
        """
        node = EstimationNode()

        # Force the legacy path: explicit method
        state = self._make_state_with_method("CausalForestDML")
        # Disable energy score so we must use legacy path
        state["parameters"]["use_energy_score"] = False  # type: ignore[index]

        result = await node.execute(state)

        # Contract:
        # If estimation succeeded → must be from REAL econml/dowhy (not mock corrcoef
        #   with ate_se=0.05 fixed).
        # If failed → must be EstimationError-shaped failure (structured fail-closed).
        if result.get("status") == "failed":
            # Fail-closed is acceptable
            assert (
                "EstimationError" in result.get("error_message", "")
                or "estimat" in result.get("error_message", "").lower()
            )
        else:
            est = result["estimation_result"]
            # The mock returned ate_se=0.05 ALWAYS. Real estimator has variable se.
            # If we still see ate_se exactly 0.05 with corrcoef-derived ate, we're
            # in the mock path. Real implementations vary.
            # Note: the mock cate_segments had "High Engagement" / "Low Engagement"
            # hard-coded — assert these specific mock artifacts are gone.
            cate_segments = est.get("cate_segments", [])
            for seg in cate_segments:
                # These exact strings are mock fingerprints
                assert (
                    seg.get("segment") != "High Engagement"
                    or seg.get("description") != "HCPs with high engagement"
                ), (
                    "F-006 regression: legacy _estimate_causal_forest still returns "
                    "hardcoded mock CATE segments with 'High Engagement' label."
                )

    @pytest.mark.asyncio
    async def test_f006_explicit_method_legacy_path_fails_closed_on_energy_score_failure(self):
        """F-006 RED pin: when energy-score path is the only real path and it fails,
        explicit legacy method must raise EstimationError, not return mock values.
        """
        node = EstimationNode()

        # Force a state where method is set explicitly AND energy_score is disabled.
        # The mock _estimate_causal_forest currently returns np.corrcoef * std.
        # After the fix, this MUST raise EstimationError.
        state = self._make_state_with_method("CausalForestDML")
        state["parameters"]["use_energy_score"] = False  # type: ignore[index]

        # Patch the energy-score selector to FORCE failure, simulating the
        # silent-fallback trapdoor that the fix must eliminate.
        with patch.object(
            node,
            "_select_estimator_with_energy_score",
            side_effect=Exception("Forced energy-score failure"),
        ):
            result = await node.execute(state)

        # Must fail-closed: status=failed with EstimationError-shaped message
        assert result.get("status") == "failed", (
            "F-006 regression: explicit legacy method with energy-score failure did "
            "NOT fail-closed. Silent mock fallback (np.corrcoef path) is active."
        )
        error_msg = result.get("error_message", "").lower()
        assert "estimat" in error_msg, (
            f"F-006 regression: error message does not indicate estimation failure: "
            f"{result.get('error_message')!r}"
        )

    @pytest.mark.asyncio
    async def test_f006_get_data_fails_closed_without_data_source_opt_in(self):
        """F-006 iter-2 (#417, codex H2): _get_data() must fail-closed when
        no real data is in data_cache AND data_source is not 'synthetic'.

        The previous unconditional synthetic-data fallback was a silent-wrong
        path — real estimators produced polished causal answers over
        fabricated HCP/conversion data. Production callers that omit
        data_source must now fail-closed.
        """
        node = EstimationNode()

        graph: CausalGraph = {
            "nodes": ["hcp_engagement_level", "patient_conversion_rate", "geographic_region"],
            "edges": [
                ("geographic_region", "hcp_engagement_level"),
                ("geographic_region", "patient_conversion_rate"),
                ("hcp_engagement_level", "patient_conversion_rate"),
            ],
            "treatment_nodes": ["hcp_engagement_level"],
            "outcome_nodes": ["patient_conversion_rate"],
            "adjustment_sets": [["geographic_region"]],
            "dag_dot": "...",
            "confidence": 0.85,
        }
        state: CausalImpactState = {
            "query": "test query",
            "query_id": "test-f006-get-data-no-optin",
            "treatment_var": "hcp_engagement_level",
            "outcome_var": "patient_conversion_rate",
            "confounders": ["geographic_region"],
            # Deliberately omit data_source — production caller forgot to
            # provide real data AND did not opt in to synthetic.
            "causal_graph": graph,
            "status": "pending",
            "errors": [],
            "warnings": [],
        }

        result = await node.execute(state)

        assert result.get("status") == "failed", (
            "F-006 regression: missing data_source did NOT cause fail-closed. "
            "Production caller was silently served synthetic data."
        )
        error_msg = result.get("error_message", "").lower()
        # Either "estimat" (from EstimationError wrapping) or "data" (raw)
        assert "estimat" in error_msg or "data" in error_msg, (
            f"F-006 regression: unexpected error message: {result.get('error_message')!r}"
        )

    @pytest.mark.asyncio
    async def test_f006_get_data_allows_explicit_synthetic_opt_in(self):
        """F-006 iter-2 (#417): when data_source='synthetic' is explicit,
        the synthetic-data path is allowed (preserves test fixtures).
        """
        node = EstimationNode()

        graph: CausalGraph = {
            "nodes": ["hcp_engagement_level", "patient_conversion_rate", "geographic_region"],
            "edges": [
                ("geographic_region", "hcp_engagement_level"),
                ("geographic_region", "patient_conversion_rate"),
                ("hcp_engagement_level", "patient_conversion_rate"),
            ],
            "treatment_nodes": ["hcp_engagement_level"],
            "outcome_nodes": ["patient_conversion_rate"],
            "adjustment_sets": [["geographic_region"]],
            "dag_dot": "...",
            "confidence": 0.85,
        }
        state: CausalImpactState = {
            "query": "test query",
            "query_id": "test-f006-get-data-synthetic-optin",
            "treatment_var": "hcp_engagement_level",
            "outcome_var": "patient_conversion_rate",
            "confounders": ["geographic_region"],
            "data_source": "synthetic",  # explicit opt-in
            "causal_graph": graph,
            "status": "pending",
            "errors": [],
            "warnings": [],
        }

        result = await node.execute(state)

        # Should NOT fail-closed when opt-in is explicit.
        assert (
            result.get("status") != "failed"
            or "data_source" not in result.get("error_message", "").lower()
        ), (
            "F-006 regression: explicit data_source='synthetic' opt-in was "
            "still rejected. The synthetic path must remain available for "
            "test fixtures and developer workflows."
        )

    @pytest.mark.asyncio
    async def test_f006_p_value_is_computed_not_hardcoded(self):
        """F-006 iter-3 (#417, codex iter-2 H2): p_value must come from a
        real two-sided z-test on (ate, ate_std), not the hardcoded
        ``0.001 if abs(ate) > 1.96 * ate_std else 0.15`` sentinel pair.

        Anti-mocking: hardcoded p-values are placeholder evidence; downstream
        code treats p_value < 0.05 as a real statistical signal.
        """
        node = EstimationNode()
        graph: CausalGraph = {
            "nodes": ["hcp_engagement_level", "patient_conversion_rate", "geographic_region"],
            "edges": [
                ("geographic_region", "hcp_engagement_level"),
                ("geographic_region", "patient_conversion_rate"),
                ("hcp_engagement_level", "patient_conversion_rate"),
            ],
            "treatment_nodes": ["hcp_engagement_level"],
            "outcome_nodes": ["patient_conversion_rate"],
            "adjustment_sets": [["geographic_region"]],
            "dag_dot": "...",
            "confidence": 0.85,
        }
        state: CausalImpactState = {
            "query": "test query",
            "query_id": "test-f006-p-value-real",
            "treatment_var": "hcp_engagement_level",
            "outcome_var": "patient_conversion_rate",
            "confounders": ["geographic_region"],
            "data_source": "synthetic",
            "causal_graph": graph,
            "parameters": {"method": "CausalForestDML"},
            "status": "pending",
            "errors": [],
            "warnings": [],
        }
        result = await node.execute(state)
        if result.get("status") == "failed":
            return  # fail-closed path acceptable
        est = result["estimation_result"]
        p_value = est.get("p_value")
        # Sentinel pair: 0.001 or 0.15 was the previous hardcoded value pair.
        # A real z-test produces a continuous p-value, so it should NOT be
        # exactly one of those two sentinels for non-trivial data.
        assert p_value != 0.001 and p_value != 0.15, (
            f"F-006 regression iter-3: p_value={p_value!r} is one of the "
            "hardcoded sentinels (0.001, 0.15) — must be computed from real "
            "estimator uncertainty (z-test on ate / ate_std)."
        )

    @pytest.mark.asyncio
    async def test_f006_estimation_fails_closed_when_selector_returns_no_ci(self):
        """F-006 iter-4 (#417, codex iter-3 H1): estimation must fail-closed
        when EstimatorSelector returns success=True but ate_ci_lower /
        ate_ci_upper / ate_std are None.

        The previous default of ``0.0`` materialized a degenerate CI
        ``(0.0, 0.0)`` and zero standard error, which propagated as
        silent-wrong evidence into refutation scoring (the refutation guards
        added in iter-3 check finiteness + ordering but NOT zero-width).
        Same silent-evidence class as iter-1 H4 / iter-2 H1.
        """
        from unittest.mock import MagicMock

        node = EstimationNode()

        from src.causal_engine.energy_score import (
            EstimatorResult,
            EstimatorType,
            SelectionResult,
            SelectionStrategy,
        )

        # success=True but ate_ci_lower/ate_ci_upper/ate_std are None.
        # This shape used to materialize as ate_ci=(0.0, 0.0), ate_se=0.0.
        partial_result = EstimatorResult(
            estimator_type=EstimatorType.OLS,
            success=True,
            ate=0.25,
            ate_std=None,
            ate_ci_lower=None,
            ate_ci_upper=None,
            energy_score_result=None,
        )
        fake_selection = SelectionResult(
            selected=partial_result,
            selection_strategy=SelectionStrategy.BEST_ENERGY_SCORE,
            all_results=[partial_result],
            selection_reason="Only OLS succeeded",
            total_time_ms=10.0,
            energy_scores={"ols": 0.0},
            energy_score_gap=0.0,
        )

        graph: CausalGraph = {
            "nodes": ["hcp_engagement_level", "patient_conversion_rate", "geographic_region"],
            "edges": [
                ("geographic_region", "hcp_engagement_level"),
                ("geographic_region", "patient_conversion_rate"),
                ("hcp_engagement_level", "patient_conversion_rate"),
            ],
            "treatment_nodes": ["hcp_engagement_level"],
            "outcome_nodes": ["patient_conversion_rate"],
            "adjustment_sets": [["geographic_region"]],
            "dag_dot": "...",
            "confidence": 0.85,
        }
        state: CausalImpactState = {
            "query": "test query",
            "query_id": "test-f006-no-ci-on-success",
            "treatment_var": "hcp_engagement_level",
            "outcome_var": "patient_conversion_rate",
            "confounders": ["geographic_region"],
            "data_source": "synthetic",
            "causal_graph": graph,
            "status": "pending",
            "errors": [],
            "warnings": [],
        }

        fake_selector = MagicMock()
        fake_selector.select.return_value = fake_selection
        with patch.object(node, "_get_estimator_selector", return_value=fake_selector):
            result = await node.execute(state)

        assert result.get("status") == "failed", (
            "F-006 regression iter-4: estimator returned success=True with "
            "ate_ci_lower=None and ate_ci_upper=None — node did NOT fail-closed "
            "and silently materialized ate_ci=(0.0, 0.0) which propagates as "
            "silent-wrong evidence into refutation scoring."
        )
        error_msg = result.get("error_message", "").lower()
        assert "ci" in error_msg or "confidence" in error_msg or "estimat" in error_msg, (
            f"F-006 regression iter-4: error message does not indicate "
            f"CI-bounds failure: {result.get('error_message')!r}"
        )

    @pytest.mark.asyncio
    async def test_f006_estimation_fails_closed_on_zero_ate_std(self):
        """F-006 iter-5 (#417, codex iter-4 H1): estimation must fail-closed
        when selected estimator returns success=True with ate_std == 0.0
        (or NaN / non-finite). The iter-4 fix only rejected None ate_std,
        but a 0.0 / NaN ate_std produces z_score=inf and is functionally
        unusable uncertainty.
        """
        from unittest.mock import MagicMock

        node = EstimationNode()

        from src.causal_engine.energy_score import (
            EstimatorResult,
            EstimatorType,
            SelectionResult,
            SelectionStrategy,
        )

        partial_result = EstimatorResult(
            estimator_type=EstimatorType.OLS,
            success=True,
            ate=0.25,
            ate_std=0.0,  # zero — unusable
            ate_ci_lower=0.20,
            ate_ci_upper=0.30,
            energy_score_result=None,
        )
        fake_selection = SelectionResult(
            selected=partial_result,
            selection_strategy=SelectionStrategy.BEST_ENERGY_SCORE,
            all_results=[partial_result],
            selection_reason="Only OLS succeeded",
            total_time_ms=10.0,
            energy_scores={"ols": 0.0},
            energy_score_gap=0.0,
        )
        graph: CausalGraph = {
            "nodes": ["hcp_engagement_level", "patient_conversion_rate", "geographic_region"],
            "edges": [
                ("geographic_region", "hcp_engagement_level"),
                ("geographic_region", "patient_conversion_rate"),
                ("hcp_engagement_level", "patient_conversion_rate"),
            ],
            "treatment_nodes": ["hcp_engagement_level"],
            "outcome_nodes": ["patient_conversion_rate"],
            "adjustment_sets": [["geographic_region"]],
            "dag_dot": "...",
            "confidence": 0.85,
        }
        state: CausalImpactState = {
            "query": "test query",
            "query_id": "test-f006-zero-ate-std",
            "treatment_var": "hcp_engagement_level",
            "outcome_var": "patient_conversion_rate",
            "confounders": ["geographic_region"],
            "data_source": "synthetic",
            "causal_graph": graph,
            "status": "pending",
            "errors": [],
            "warnings": [],
        }
        fake_selector = MagicMock()
        fake_selector.select.return_value = fake_selection
        with patch.object(node, "_get_estimator_selector", return_value=fake_selector):
            result = await node.execute(state)

        assert result.get("status") == "failed", (
            "F-006 regression iter-5: ate_std=0.0 did NOT cause fail-closed. "
            "Estimation emitted classification with unusable uncertainty."
        )

    @pytest.mark.asyncio
    async def test_f006_estimation_fails_closed_on_degenerate_ci(self):
        """F-006 iter-5 (#417, codex iter-4 H2): estimation must fail-closed
        when CI bounds are degenerate (ate_ci_lower == ate_ci_upper) on a
        successful EstimatorResult. Zero-width CI propagates as silent-wrong
        evidence into refutation scoring.
        """
        from unittest.mock import MagicMock

        node = EstimationNode()

        from src.causal_engine.energy_score import (
            EstimatorResult,
            EstimatorType,
            SelectionResult,
            SelectionStrategy,
        )

        partial_result = EstimatorResult(
            estimator_type=EstimatorType.OLS,
            success=True,
            ate=0.25,
            ate_std=0.05,
            ate_ci_lower=0.25,  # degenerate (zero-width)
            ate_ci_upper=0.25,
            energy_score_result=None,
        )
        fake_selection = SelectionResult(
            selected=partial_result,
            selection_strategy=SelectionStrategy.BEST_ENERGY_SCORE,
            all_results=[partial_result],
            selection_reason="Only OLS succeeded",
            total_time_ms=10.0,
            energy_scores={"ols": 0.0},
            energy_score_gap=0.0,
        )
        graph: CausalGraph = {
            "nodes": ["hcp_engagement_level", "patient_conversion_rate", "geographic_region"],
            "edges": [
                ("geographic_region", "hcp_engagement_level"),
                ("geographic_region", "patient_conversion_rate"),
                ("hcp_engagement_level", "patient_conversion_rate"),
            ],
            "treatment_nodes": ["hcp_engagement_level"],
            "outcome_nodes": ["patient_conversion_rate"],
            "adjustment_sets": [["geographic_region"]],
            "dag_dot": "...",
            "confidence": 0.85,
        }
        state: CausalImpactState = {
            "query": "test query",
            "query_id": "test-f006-degenerate-ci",
            "treatment_var": "hcp_engagement_level",
            "outcome_var": "patient_conversion_rate",
            "confounders": ["geographic_region"],
            "data_source": "synthetic",
            "causal_graph": graph,
            "status": "pending",
            "errors": [],
            "warnings": [],
        }
        fake_selector = MagicMock()
        fake_selector.select.return_value = fake_selection
        with patch.object(node, "_get_estimator_selector", return_value=fake_selector):
            result = await node.execute(state)

        assert result.get("status") == "failed", (
            "F-006 regression iter-5: degenerate CI (ate_ci_lower == ate_ci_upper) "
            "did NOT cause fail-closed."
        )

    @pytest.mark.asyncio
    async def test_f014_ci_bounds_required_for_refutation(self):
        """F-014 iter-3 (#416, codex iter-2 H1): refutation must fail-closed
        when EstimationResult is missing ate_ci_lower / ate_ci_upper.

        The previous default of ``original_ate +/- 0.1`` fabricated
        uncertainty that fed directly into data_subset and bootstrap
        pass/review/block scoring. Same silent-evidence class as the iter-1
        H4 finding on placebo p_value defaults.
        """
        node = RefutationNode()

        import numpy as np
        import pandas as pd

        np.random.seed(123)
        n = 100
        data = pd.DataFrame(
            {
                "hcp_engagement_level": np.random.normal(0, 1, n),
                "patient_conversion_rate": np.random.normal(0, 1, n),
                "geographic_region": np.random.randint(0, 2, n).astype(float),
            }
        )

        # EstimationResult deliberately missing CI bounds
        estimation_result_no_ci = {
            "method": "CausalForestDML",
            "ate": 0.25,
            "effect_size": "medium",
            "statistical_significance": True,
            "p_value": 0.01,
            "sample_size": n,
            "covariates_adjusted": ["geographic_region"],
            "heterogeneity_detected": False,
        }

        state: CausalImpactState = {
            "query": "test",
            "query_id": "test-f014-no-ci",
            "treatment_var": "hcp_engagement_level",
            "outcome_var": "patient_conversion_rate",
            "confounders": ["geographic_region"],
            "data_source": "synthetic",
            "estimation_result": estimation_result_no_ci,  # type: ignore[typeddict-item]
            "estimation_data": data,
            "status": "pending",
        }
        result = await node.execute(state)
        assert result.get("refutation_error") is not None or result.get("status") == "failed", (
            "F-014 regression iter-3: missing ate_ci_lower/ate_ci_upper did "
            "NOT cause fail-closed. Refutation silently fabricated a +/- 0.1 "
            "CI which feeds invented evidence into data_subset and bootstrap "
            "scoring."
        )

    @pytest.mark.asyncio
    async def test_f006_all_estimators_fail_fails_closed(self):
        """F-006 iter-2 (#417, codex H1): when EstimatorSelector returns a
        success=False EstimatorResult (all configured estimators failed),
        the node MUST raise EstimationError instead of emitting ate=0.0
        / ate_se=0.0 / energy_score=0.0 silent-wrong defaults.
        """
        from unittest.mock import MagicMock

        node = EstimationNode()

        # Build a fake SelectionResult where the selected estimator failed.
        from src.causal_engine.energy_score import (
            EstimatorResult,
            EstimatorType,
            SelectionResult,
            SelectionStrategy,
        )

        failed_result = EstimatorResult(
            estimator_type=EstimatorType.OLS,
            success=False,
            error_message="forced all-estimators-fail for codex H1 pin",
            error_type="RuntimeError",
        )
        fake_selection = SelectionResult(
            selected=failed_result,
            selection_strategy=SelectionStrategy.BEST_ENERGY_SCORE,
            all_results=[failed_result],
            selection_reason="All estimators failed",
            total_time_ms=10.0,
            energy_scores={},
            energy_score_gap=0.0,
        )

        graph: CausalGraph = {
            "nodes": ["hcp_engagement_level", "patient_conversion_rate", "geographic_region"],
            "edges": [
                ("geographic_region", "hcp_engagement_level"),
                ("geographic_region", "patient_conversion_rate"),
                ("hcp_engagement_level", "patient_conversion_rate"),
            ],
            "treatment_nodes": ["hcp_engagement_level"],
            "outcome_nodes": ["patient_conversion_rate"],
            "adjustment_sets": [["geographic_region"]],
            "dag_dot": "...",
            "confidence": 0.85,
        }
        state: CausalImpactState = {
            "query": "test query",
            "query_id": "test-f006-all-fail",
            "treatment_var": "hcp_engagement_level",
            "outcome_var": "patient_conversion_rate",
            "confounders": ["geographic_region"],
            "data_source": "synthetic",
            "causal_graph": graph,
            "status": "pending",
            "errors": [],
            "warnings": [],
        }

        # Inject a fake selector that returns the all-failed SelectionResult.
        fake_selector = MagicMock()
        fake_selector.select.return_value = fake_selection
        with patch.object(node, "_get_estimator_selector", return_value=fake_selector):
            result = await node.execute(state)

        assert result.get("status") == "failed", (
            "F-006 regression iter-2: all-estimators-fail did NOT trigger "
            "fail-closed. The node returned a silent-wrong ate=0.0."
        )
        error_msg = result.get("error_message", "").lower()
        assert "estimat" in error_msg and ("fail" in error_msg or "0.0" in error_msg), (
            f"F-006 regression iter-2: error message does not indicate "
            f"all-estimators-fail: {result.get('error_message')!r}"
        )

    @pytest.mark.asyncio
    async def test_f006_fails_closed_when_treatment_outcome_not_in_data(self):
        """#354 trapdoor #2: execute() must fail-closed when the graph's
        treatment/outcome are NOT columns of the estimation data.

        ``_select_estimator_with_energy_score`` resolves the treatment/outcome
        arrays with ``data.get(treatment, data.iloc[:, 0])`` /
        ``data.iloc[:, 1]`` — a silent POSITIONAL fallback. If the graph's
        treatment/outcome names diverge from the loaded frame's columns
        (query-inference, the hardcoded ``outcome='patient_conversion_rate'``
        default in graph_builder, or any upstream rename), the node would
        silently estimate over the FIRST TWO columns and emit a polished ATE
        over the WRONG variables. The node MUST instead fail-closed.
        """
        import numpy as np
        import pandas as pd

        node = EstimationNode()

        np.random.seed(7)
        n = 200
        # Frame columns deliberately NOT named like the graph's treatment/outcome.
        # col_unrelated_x/y would be silently picked as treatment/outcome (iloc 0/1)
        # if the positional-fallback trapdoor is live.
        frame = pd.DataFrame(
            {
                "col_unrelated_x": np.random.normal(0, 1, n),
                "col_unrelated_y": 0.9 * np.random.normal(0, 1, n) + np.random.normal(0, 0.2, n),
                "geographic_region": np.random.randint(0, 2, n).astype(float),
            }
        )
        graph: CausalGraph = {
            "nodes": ["real_treatment", "real_outcome", "geographic_region"],
            "edges": [
                ("geographic_region", "real_treatment"),
                ("geographic_region", "real_outcome"),
                ("real_treatment", "real_outcome"),
            ],
            "treatment_nodes": ["real_treatment"],  # NOT a frame column
            "outcome_nodes": ["real_outcome"],  # NOT a frame column
            "adjustment_sets": [["geographic_region"]],
            "dag_dot": "...",
            "confidence": 0.85,
        }
        state: CausalImpactState = {
            "query": "trapdoor-2 column-presence",
            "query_id": "test-f006-cols-missing",
            "treatment_var": "real_treatment",
            "outcome_var": "real_outcome",
            "confounders": ["geographic_region"],
            "causal_graph": graph,
            # Real-data passthrough so _get_data does NOT fail first — the frame
            # exists, it just lacks the treatment/outcome columns.
            "data_cache": {"estimation_data": frame},
            "status": "pending",
            "errors": [],
            "warnings": [],
        }

        result = await node.execute(state)

        assert result.get("status") == "failed", (
            "#354 trapdoor #2 regression: treatment/outcome absent from the data "
            "did NOT cause fail-closed. The node silently estimated over positional "
            "columns (data.iloc[:, 0]/[:, 1]) and emitted a wrong-variable ATE."
        )
        error_msg = result.get("error_message", "").lower()
        assert "column" in error_msg or "treatment" in error_msg or "outcome" in error_msg, (
            f"#354 trapdoor #2 regression: error message does not indicate the "
            f"missing-column cause: {result.get('error_message')!r}"
        )

    @pytest.mark.asyncio
    async def test_f006_estimation_error_type_is_structured(self):
        """F-006 RED pin: EstimationError must be importable and structured."""
        from src.causal_engine import EstimationError as ImportedEstimationError

        err = ImportedEstimationError(
            "test message",
            details={"method": "CausalForestDML"},
        )
        assert err.message == "test message"
        assert err.details == {"method": "CausalForestDML"}
        as_dict = err.to_dict()
        assert as_dict["error_type"] == "EstimationError"
        assert as_dict["message"] == "test message"

    @pytest.mark.asyncio
    async def test_f006_no_corrcoef_in_estimation_module(self):
        """F-006 RED pin (codebase invariant): no np.corrcoef CALLS in
        estimation.py production code path. corrcoef-based ATE is the mock
        fingerprint. References in comments/docstrings are allowed (they
        document the deletion).
        """
        import ast
        import pathlib

        estimation_path = (
            pathlib.Path(__file__).resolve().parents[4]
            / "src"
            / "agents"
            / "causal_impact"
            / "nodes"
            / "estimation.py"
        )
        tree = ast.parse(estimation_path.read_text())

        # Walk AST to find any attribute access that looks like np.corrcoef(...)
        # or corrcoef(...) calls. Comments and docstrings are NOT in the AST,
        # so they're naturally exempt.
        offending = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                # np.corrcoef(...)
                if isinstance(func, ast.Attribute) and func.attr == "corrcoef":
                    offending.append(ast.dump(node)[:120])
                # bare corrcoef(...)
                elif isinstance(func, ast.Name) and func.id == "corrcoef":
                    offending.append(ast.dump(node)[:120])

        assert not offending, (
            "F-006 regression: np.corrcoef CALL found in estimation.py: "
            f"{offending}. This is the fingerprint of the legacy mock "
            "_estimate_* methods. Replace with delegators to energy-score "
            "selector + fail-closed semantics."
        )
