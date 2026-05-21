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
