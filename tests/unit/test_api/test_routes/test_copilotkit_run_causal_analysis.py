"""
Tests for src.api.routes.copilotkit.run_causal_analysis — F-001 fix verification.

Background:
    Prior to F-001, `run_causal_analysis()` fell back to `random.uniform(...)` ATE/p_value
    when the orchestrator was missing or raised, presenting fabricated statistics to the
    CopilotKit chat UI as natural-language analysis.

These tests assert the fail-closed contract:
    - When orchestrator is unavailable → return a structured error response
      (no fabricated ATE/p_value/CI/sample_size in the default code path).
    - When orchestrator raises → propagate the failure as a structured error
      (no silent fabrication).
    - When orchestrator returns valid result → pass through.
    - When the dev-mode flag `E2I_ENABLE_SIMULATED_FALLBACK=1` is set, the fallback
      block runs but emits PINNED ZEROS with `data_source="dev_mock"` (never
      `random.uniform` values).

Reference: GitHub issue #418.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from src.api.routes.copilotkit import run_causal_analysis

# =============================================================================
# F-001: Orchestrator unavailable → structured error
# =============================================================================


class TestRunCausalAnalysisFailClosed:
    """run_causal_analysis must fail-closed when orchestrator unavailable or failed."""

    @pytest.mark.asyncio
    async def test_orchestrator_missing_returns_structured_error_not_fabrication(self):
        """When _get_orchestrator() returns None, must return error envelope (no RNG)."""
        with patch("src.api.routes.copilotkit._get_orchestrator", return_value=None):
            result = await run_causal_analysis(
                intervention="HCP Engagement",
                target_kpi="TRx Volume",
                brand="Remibrutinib",
            )
        # Must surface as structured error, NOT a 200-shaped success with RNG
        assert result.get("success") is False, (
            f"Expected success=False (fail-closed), got: {result}"
        )
        assert "error" in result, f"Expected 'error' key in response, got: {result}"
        # Must NOT contain fabricated numeric fields in the default path
        results_section = result.get("results", {})
        assert results_section.get("average_treatment_effect") in (None, 0.0, 0), (
            "Default-path fail-closed must NOT return a fabricated ATE"
        )
        # data_source must NOT claim "simulated" — that string was the old user-facing label
        # (also ensures we did not silently introduce a label-style fix instead of a real fix)
        assert result.get("data_source") != "simulated", (
            "data_source='simulated' indicates the old fabrication fallback is still active"
        )

    @pytest.mark.asyncio
    async def test_orchestrator_raises_returns_structured_error_not_fabrication(self):
        """When orchestrator.run() raises, must return error envelope (no RNG)."""
        mock_orch = AsyncMock()
        mock_orch.run = AsyncMock(side_effect=RuntimeError("simulated upstream failure"))
        with patch("src.api.routes.copilotkit._get_orchestrator", return_value=mock_orch):
            result = await run_causal_analysis(
                intervention="HCP Engagement",
                target_kpi="TRx Volume",
                brand="Remibrutinib",
            )
        assert result.get("success") is False
        assert "error" in result
        results_section = result.get("results", {})
        # Most critical assertion: no fabricated ATE in the pharma-uplift range
        ate = results_section.get("average_treatment_effect")
        assert ate in (None, 0.0, 0), f"Expected no fabricated ATE on failure, got: {ate}"
        # No fabricated p_value
        p_value = results_section.get("p_value")
        assert p_value in (None, 0.0, 0), (
            f"Expected no fabricated p_value on failure, got: {p_value}"
        )

    @pytest.mark.asyncio
    async def test_orchestrator_returns_empty_response_returns_structured_error(self):
        """When orchestrator returns dict without response_text, must fail-closed."""
        mock_orch = AsyncMock()
        mock_orch.run = AsyncMock(return_value={})  # no "response_text"
        with patch("src.api.routes.copilotkit._get_orchestrator", return_value=mock_orch):
            result = await run_causal_analysis(
                intervention="Marketing Campaign",
                target_kpi="Market Share",
                brand="Kisqali",
            )
        # Falling out of the if-block must NOT silently route to RNG fabrication
        assert result.get("success") is False, (
            f"Expected success=False on empty orchestrator result, got: {result}"
        )

    @pytest.mark.asyncio
    async def test_orchestrator_success_passes_through_real_results(self):
        """When orchestrator returns valid result, pass through (no RNG)."""
        mock_orch = AsyncMock()
        mock_orch.run = AsyncMock(
            return_value={
                "response_text": "Real causal analysis: ATE=0.184",
                "ate": 0.184,
                "ci": [0.142, 0.226],
                "p_value": 0.012,
                "significant": True,
                "agents_dispatched": ["causal_impact"],
            }
        )
        with patch("src.api.routes.copilotkit._get_orchestrator", return_value=mock_orch):
            result = await run_causal_analysis(
                intervention="HCP Engagement",
                target_kpi="TRx Volume",
                brand="Remibrutinib",
            )
        # Real path returns real results
        assert result.get("data_source") == "orchestrator"
        assert result["results"]["average_treatment_effect"] == 0.184
        assert result["results"]["p_value"] == 0.012


# =============================================================================
# F-001: Dev-mode flag uses pinned-zero placeholder, not random.uniform
# =============================================================================


class TestRunCausalAnalysisDevMode:
    """When E2I_ENABLE_SIMULATED_FALLBACK=1, fallback must use pinned zeros, never RNG."""

    @pytest.mark.asyncio
    async def test_dev_mode_returns_pinned_zeros_not_random_uniform(self, monkeypatch):
        """Dev-mode fallback must emit zeros + data_source='dev_mock'."""
        monkeypatch.setenv("E2I_ENABLE_SIMULATED_FALLBACK", "1")
        with patch("src.api.routes.copilotkit._get_orchestrator", return_value=None):
            result = await run_causal_analysis(
                intervention="HCP Engagement",
                target_kpi="TRx Volume",
                brand="Remibrutinib",
            )
        # Dev-mode label must be explicit
        assert result.get("data_source") == "dev_mock", (
            f"dev-mode fallback must label data_source='dev_mock', got: {result.get('data_source')}"
        )
        # Values must be pinned zeros, not random.uniform() outputs
        results_section = result.get("results", {})
        assert results_section.get("average_treatment_effect") == 0.0
        assert results_section.get("p_value") == 0.0
        # statistical_significance must NOT default to True in fabricated path
        assert results_section.get("statistical_significance") is False

    @pytest.mark.asyncio
    async def test_dev_mode_disabled_by_default_returns_error(self, monkeypatch):
        """With no env override, fallback must return error (default-OFF)."""
        monkeypatch.delenv("E2I_ENABLE_SIMULATED_FALLBACK", raising=False)
        with patch("src.api.routes.copilotkit._get_orchestrator", return_value=None):
            result = await run_causal_analysis(
                intervention="HCP Engagement",
                target_kpi="TRx Volume",
                brand="Remibrutinib",
            )
        assert result.get("success") is False, (
            "Default state (no E2I_ENABLE_SIMULATED_FALLBACK env var) must fail-closed"
        )


# =============================================================================
# F-001: Source-grep regression pin — production code path has no random.uniform
# =============================================================================


class TestNoRandomUniformInRunCausalAnalysisSource:
    """
    Regression pin: ensure the production fallback block contains no
    random.uniform() / random.randint() calls.

    This is a static-source check that prevents future re-introduction of
    fabrication in the run_causal_analysis() body. It targets ONLY the
    function source (not the surrounding module), to avoid coupling to
    unrelated functions.
    """

    def test_run_causal_analysis_function_source_has_no_random_uniform(self):
        """Static check: run_causal_analysis source must not call random.uniform/randint."""
        import inspect

        source = inspect.getsource(run_causal_analysis)
        # The honest-rename assertion: forbid the exact fabrication primitives
        assert "random.uniform" not in source, (
            "F-001 regression: random.uniform reintroduced in run_causal_analysis"
        )
        assert "random.randint" not in source, (
            "F-001 regression: random.randint reintroduced in run_causal_analysis"
        )
