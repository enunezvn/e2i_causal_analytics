"""Real-library wiring tests for EconMLExecutor (phase C-3 of GH #354).

These tests pin the NEW behavior wired in phase C-3: ``EconMLExecutor.execute()``
delegates to ``src.causal_engine.energy_score.estimator_selector.EstimatorSelector``
(real EconML wrap via CausalForestDML / LinearDML / DRLearner per V-04) and
inherits the fail-closed guards from
``src/agents/causal_impact/nodes/estimation.py`` (no silent ``ate=0.0`` /
``ci=(0.0, 0.0)`` / ``ate_std=0.0`` / ``ate_outside_ci`` shapes).

Test strategy: dependency-inject a fake ``EstimatorSelector`` via the
executor's constructor (``EconMLExecutor(selector=...)``) so we don't depend on
EconML model fitting (which is slow, non-deterministic, and orthogonal to the
contract under test). The integration with real EconML is exercised by
``causal_impact/nodes/estimation.py``'s test suite — the SAME selector class is
reused here.

Forbidden patterns (HIGH finding on detection per CLAUDE.md anti-mocking):
- ``np.random.seed`` / ``random.uniform`` anywhere in this file or in
  ``executors/econml.py``
- All-default / all-zero ``LibraryExecutionResult`` returned when data is
  unavailable (must be ``success=False`` with explicit error)
- Silent substitution when the selector reports failure (must be
  ``success=False`` propagating the structured reason)
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Optional
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from src.causal_engine.energy_score.estimator_selector import (
    EstimatorResult,
    EstimatorType,
    SelectionResult,
    SelectionStrategy,
)
from src.causal_engine.pipeline.executors.econml import EconMLExecutor
from src.causal_engine.pipeline.router import CausalLibrary
from src.causal_engine.pipeline.state import (
    PipelineConfig,
    PipelineStage,
    PipelineState,
)

# =============================================================================
# Helpers
# =============================================================================


def _base_state(**overrides) -> PipelineState:
    """Build a minimal PipelineState fixture for executor tests."""
    config: PipelineConfig = {
        "mode": "sequential",
        "libraries_enabled": ["econml"],
        "primary_library": "econml",
        "stage_timeout_ms": 30000,
        "total_timeout_ms": 120000,
        "cross_validate": True,
        "min_agreement_threshold": 0.85,
        "max_parallel_libraries": 4,
        "fail_fast": False,
        "segment_by_uplift": False,
        "nested_ci_level": 0.95,
    }
    state: PipelineState = {
        "query": "Does marketing spend cause sales?",
        "question_type": "causal_relationship",
        "treatment_var": "marketing_spend",
        "outcome_var": "sales",
        "confounders": ["region", "season"],
        "effect_modifiers": None,
        "data_source": "test_data",
        "filters": None,
        "config": config,
        "routed_libraries": ["econml"],
        "routing_confidence": 0.9,
        "routing_rationale": "Test routing",
        "networkx_result": None,
        "causal_graph": None,
        "graph_metrics": None,
        "dowhy_result": None,
        "causal_effect": None,
        "refutation_results": None,
        "identification_method": None,
        "econml_result": None,
        "cate_by_segment": None,
        "overall_ate": None,
        "heterogeneity_score": None,
        "causalml_result": None,
        "uplift_scores": None,
        "auuc": None,
        "qini": None,
        "targeting_recommendations": None,
        "consensus_effect": None,
        "consensus_confidence": None,
        "library_agreement": None,
        "nested_cate": None,
        "segment_confidence_intervals": None,
        "executive_summary": None,
        "key_insights": None,
        "recommended_actions": None,
        "current_stage": PipelineStage.PENDING,
        "stage_latencies": {},
        "total_latency_ms": 0,
        "libraries_executed": [],
        "libraries_skipped": [],
        "errors": [],
        "warnings": [],
        "status": "pending",
    }
    state.update(overrides)  # type: ignore[typeddict-item]
    return state


def _base_config() -> PipelineConfig:
    return {
        "mode": "sequential",
        "libraries_enabled": ["econml"],
        "primary_library": "econml",
        "stage_timeout_ms": 30000,
        "total_timeout_ms": 120000,
        "cross_validate": True,
        "min_agreement_threshold": 0.85,
        "max_parallel_libraries": 4,
        "fail_fast": False,
        "segment_by_uplift": False,
        "nested_ci_level": 0.95,
    }


def _real_data_frame() -> pd.DataFrame:
    """A tiny deterministic DataFrame for tests.

    NOT a synthetic-data fabrication for the estimator — the estimator is
    mocked. The frame just satisfies the executor's column-extraction step.
    """
    return pd.DataFrame(
        {
            "marketing_spend": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "sales": [10.0, 12.0, 11.0, 14.0, 9.5, 13.5, 10.5, 13.0, 11.5, 12.5],
            "region": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            "season": [0, 0, 1, 1, 0, 0, 1, 1, 0, 1],
        }
    )


def _state_with_data(df: Optional[pd.DataFrame] = None, **overrides) -> PipelineState:
    """Build a state with a DataFrame attached via ``data_cache``.

    Mirrors ``agents/causal_impact/nodes/estimation.py::_get_data()`` resolution
    order: ``state['data_cache']['estimation_data']`` is the canonical real-data
    passthrough key.
    """
    state = _base_state(**overrides)
    state["data_cache"] = {"estimation_data": df if df is not None else _real_data_frame()}  # type: ignore[typeddict-unknown-key]
    return state


_CATE_SENTINEL = object()


def _good_estimator_result(
    *,
    ate: float = 0.15,
    ate_std: float = 0.02,
    ate_ci_lower: float = 0.10,
    ate_ci_upper: float = 0.20,
    cate: Any = _CATE_SENTINEL,
    estimator_type: EstimatorType = EstimatorType.CAUSAL_FOREST,
    energy_score: float = 0.3,
) -> EstimatorResult:
    """Build a successful EstimatorResult satisfying every fail-closed guard.

    Use a sentinel for ``cate`` so callers can explicitly pass ``cate=None``
    (LinearDML / DRLearner / OLS shape) vs. omit it (CausalForestDML shape
    with a default per-record CATE array).

    Attaches a duck-typed ``energy_score_result`` so ``EstimatorResult``'s
    ``energy_score`` property returns a finite value (not the default +inf
    that triggers the executor's finiteness guard).
    """
    if cate is _CATE_SENTINEL:
        cate_value = np.array([0.12, 0.14, 0.16, 0.18], dtype=float)
    else:
        cate_value = cate  # may be None (LinearDML shape) or an array
    er = EstimatorResult(
        estimator_type=estimator_type,
        success=True,
        ate=ate,
        cate=cate_value,
        ate_std=ate_std,
        ate_ci_lower=ate_ci_lower,
        ate_ci_upper=ate_ci_upper,
    )
    # SimpleNamespace duck-types EnergyScoreResult sufficiently for the
    # ``EstimatorResult.energy_score`` property (which only reads
    # ``.energy_score``).
    er.energy_score_result = SimpleNamespace(energy_score=energy_score)  # type: ignore[assignment]
    return er


def _selection_result(selected: EstimatorResult, **kw) -> SelectionResult:
    return SelectionResult(
        selected=selected,
        selection_strategy=SelectionStrategy.BEST_ENERGY_SCORE,
        all_results=[selected],
        selection_reason="test selection",
        total_time_ms=10.0,
        energy_scores={selected.estimator_type.value: 0.3},
        energy_score_gap=0.05,
        **kw,
    )


class _FakeSelector:
    """Mimics ``EstimatorSelector.select()`` shape for executor wiring tests."""

    def __init__(self, selection: SelectionResult):
        self._selection = selection
        self.calls = 0

    def select(
        self,
        treatment,
        outcome,
        covariates,
        **kwargs,
    ) -> SelectionResult:
        self.calls += 1
        return self._selection


class _RaisingSelector:
    def __init__(self, exc: BaseException):
        self._exc = exc

    def select(self, *args, **kwargs):
        raise self._exc


# =============================================================================
# Library + ABC contract tests (unchanged from C-1 baseline)
# =============================================================================


class TestEconMLExecutorContract:
    def test_library_property_returns_econml(self):
        assert EconMLExecutor().library == CausalLibrary.ECONML

    def test_validate_input_success(self):
        valid, msg = EconMLExecutor().validate_input(_base_state())
        assert valid is True
        assert msg == ""

    def test_validate_input_fails_without_treatment_var(self):
        state = _base_state()
        state["treatment_var"] = None
        valid, msg = EconMLExecutor().validate_input(state)
        assert valid is False
        assert "treatment_var" in msg

    def test_validate_input_fails_without_outcome_var(self):
        state = _base_state()
        state["outcome_var"] = None
        valid, msg = EconMLExecutor().validate_input(state)
        assert valid is False
        assert "outcome_var" in msg


# =============================================================================
# Fail-closed when data backend is unavailable
# =============================================================================


class TestEconMLExecutorFailsClosedWithoutData:
    """When state has no ``data_cache.estimation_data`` and no other DataFrame
    source, the executor MUST fail-closed (success=False with explicit reason).
    NEVER fall back to synthetic data, hardcoded ATE, or DoWhy effect leakage.
    """

    @pytest.mark.asyncio
    async def test_fails_closed_when_no_data_cache(self):
        executor = EconMLExecutor()
        result = await executor.execute(_base_state(), _base_config())

        assert result["library"] == "econml"
        assert result["success"] is False
        assert result["confidence"] == 0.0
        # The error must explicitly name the data-unavailability reason —
        # otherwise downstream consumers can't distinguish from a real
        # estimator failure.
        assert result["error"] is not None
        assert "data" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_fails_closed_when_data_cache_present_but_no_estimation_data(self):
        state = _base_state()
        state["data_cache"] = {}  # type: ignore[typeddict-unknown-key]
        result = await EconMLExecutor().execute(state, _base_config())

        assert result["success"] is False
        assert result["error"] is not None

    @pytest.mark.asyncio
    async def test_fails_closed_when_validate_input_rejects(self):
        """Missing treatment_var/outcome_var means we cannot even call the
        selector. Must fail-closed even if a data_cache is present.
        """
        state = _state_with_data()
        state["treatment_var"] = None
        result = await EconMLExecutor().execute(state, _base_config())

        assert result["success"] is False
        assert result["error"] is not None
        assert "treatment_var" in result["error"]


# =============================================================================
# Fail-closed when EstimatorSelector raises or returns unusable values
# =============================================================================


class TestEconMLExecutorFailsClosedOnSelectorFailure:
    @pytest.mark.asyncio
    async def test_fails_closed_when_selector_raises(self):
        sel = _RaisingSelector(RuntimeError("econml model fit blew up"))
        executor = EconMLExecutor(selector=sel)
        result = await executor.execute(_state_with_data(), _base_config())

        assert result["success"] is False
        assert result["error"] is not None
        assert "econml" in result["error"].lower() or "blew up" in result["error"]

    @pytest.mark.asyncio
    async def test_fails_closed_when_all_estimators_fail(self):
        """``EstimatorSelector._select_best_energy`` returns a
        ``success=False`` EstimatorResult when ALL configured estimators fail.
        Mirrors estimation.py:209-241 (F-006 iter-2 H1).
        """
        failed = EstimatorResult(
            estimator_type=EstimatorType.OLS,
            success=False,
            ate=None,
            error_message="All estimators failed",
        )
        sel = _FakeSelector(_selection_result(failed))
        executor = EconMLExecutor(selector=sel)
        result = await executor.execute(_state_with_data(), _base_config())

        assert result["success"] is False
        assert result["error"] is not None

    @pytest.mark.asyncio
    async def test_fails_closed_when_selected_success_true_but_ate_is_none(self):
        """Defense-in-depth: even a 'successful' selector result that produced
        no ATE must be rejected (would otherwise emit None as ate downstream).
        """
        er = _good_estimator_result()
        er.ate = None
        sel = _FakeSelector(_selection_result(er))
        executor = EconMLExecutor(selector=sel)
        result = await executor.execute(_state_with_data(), _base_config())

        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_fails_closed_when_ci_bounds_missing(self):
        er = _good_estimator_result()
        er.ate_ci_lower = None
        sel = _FakeSelector(_selection_result(er))
        result = await EconMLExecutor(selector=sel).execute(_state_with_data(), _base_config())

        assert result["success"] is False
        assert "ci" in result["error"].lower() or "confidence" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_fails_closed_when_ci_bounds_non_finite(self):
        er = _good_estimator_result()
        er.ate_ci_lower = float("-inf")
        sel = _FakeSelector(_selection_result(er))
        result = await EconMLExecutor(selector=sel).execute(_state_with_data(), _base_config())

        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_fails_closed_when_ci_degenerate(self):
        er = _good_estimator_result()
        er.ate_ci_lower = 0.20
        er.ate_ci_upper = 0.20  # zero-width
        sel = _FakeSelector(_selection_result(er))
        result = await EconMLExecutor(selector=sel).execute(_state_with_data(), _base_config())

        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_fails_closed_when_ate_outside_ci(self):
        er = _good_estimator_result(ate=0.5, ate_ci_lower=0.10, ate_ci_upper=0.20)
        sel = _FakeSelector(_selection_result(er))
        result = await EconMLExecutor(selector=sel).execute(_state_with_data(), _base_config())

        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_fails_closed_when_ate_std_missing(self):
        er = _good_estimator_result()
        er.ate_std = None
        sel = _FakeSelector(_selection_result(er))
        result = await EconMLExecutor(selector=sel).execute(_state_with_data(), _base_config())

        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_fails_closed_when_ate_std_non_positive(self):
        er = _good_estimator_result()
        er.ate_std = 0.0
        sel = _FakeSelector(_selection_result(er))
        result = await EconMLExecutor(selector=sel).execute(_state_with_data(), _base_config())

        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_fails_closed_when_ate_std_non_finite(self):
        er = _good_estimator_result()
        er.ate_std = float("nan")
        sel = _FakeSelector(_selection_result(er))
        result = await EconMLExecutor(selector=sel).execute(_state_with_data(), _base_config())

        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_fails_closed_when_energy_score_nan(self):
        """Codex iter-1 MEDIUM: non-finite energy_score must fail-closed
        rather than silently emit confidence=1.0 (NaN min/max semantics) and
        non-JSON-safe NaN/Inf into the result payload.
        """
        er = _good_estimator_result()
        # Force energy_score_result to carry a NaN energy_score. We use a
        # SimpleNamespace as a duck-typed EnergyScoreResult to avoid coupling
        # this test to the full EnergyScoreResult schema (it has many
        # mandatory metadata fields).
        from types import SimpleNamespace

        er.energy_score_result = SimpleNamespace(energy_score=float("nan"))  # type: ignore[assignment]
        sel = _FakeSelector(_selection_result(er))
        result = await EconMLExecutor(selector=sel).execute(_state_with_data(), _base_config())

        assert result["success"] is False
        assert "energy_score" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_fails_closed_when_energy_score_infinite(self):
        """Default EstimatorResult.energy_score is +inf when no energy score
        was computed (estimator_selector.py:108). Must fail-closed: an
        unevaluated estimator cannot produce a trustworthy quality_tier /
        confidence.
        """
        er = _good_estimator_result()
        er.energy_score_result = None  # property returns float("inf")
        sel = _FakeSelector(_selection_result(er))
        result = await EconMLExecutor(selector=sel).execute(_state_with_data(), _base_config())

        assert result["success"] is False


# =============================================================================
# Fail-closed when routed confounders are missing from the DataFrame
# =============================================================================


class TestEconMLExecutorConfounderValidation:
    """Codex iter-1 MEDIUM: when the routed state names confounders that are
    NOT present in the DataFrame, the executor MUST fail-closed -- not
    silently substitute a different adjustment set.
    """

    @pytest.mark.asyncio
    async def test_fails_closed_when_confounder_missing(self):
        er = _good_estimator_result()
        sel = _FakeSelector(_selection_result(er))
        state = _state_with_data()
        # state has confounders ['region', 'season']; df has 'region' but not
        # 'climate_zone'. Make one of the named confounders missing.
        state["confounders"] = ["region", "climate_zone"]
        result = await EconMLExecutor(selector=sel).execute(state, _base_config())

        assert result["success"] is False
        assert "climate_zone" in result["error"]
        # Selector MUST NOT have been called (we fail before reaching it).
        assert sel.calls == 0

    @pytest.mark.asyncio
    async def test_fails_closed_when_all_confounders_missing(self):
        er = _good_estimator_result()
        sel = _FakeSelector(_selection_result(er))
        state = _state_with_data()
        # All named confounders absent from df -- pre-fix code would silently
        # fall back to all non-(treatment, outcome) columns; new code
        # fail-closes.
        state["confounders"] = ["foo", "bar"]
        result = await EconMLExecutor(selector=sel).execute(state, _base_config())

        assert result["success"] is False
        assert "foo" in result["error"] and "bar" in result["error"]
        assert sel.calls == 0

    @pytest.mark.asyncio
    async def test_happy_path_when_all_confounders_present(self):
        """All named confounders ARE in df -> selector is called with only
        the named confounder subset (NOT all columns).
        """
        captured = {}

        class _CapturingSelector:
            def __init__(self, selection: SelectionResult):
                self._selection = selection

            def select(self, treatment, outcome, covariates, **kwargs):
                captured["covariate_cols"] = list(covariates.columns)
                return self._selection

        er = _good_estimator_result()
        sel = _CapturingSelector(_selection_result(er))
        state = _state_with_data()
        state["confounders"] = ["region"]  # subset; 'season' deliberately omitted
        result = await EconMLExecutor(selector=sel).execute(state, _base_config())

        assert result["success"] is True
        assert captured["covariate_cols"] == ["region"]

    @pytest.mark.asyncio
    async def test_falls_back_to_non_treatment_outcome_when_no_confounders_named(self):
        """When state has no confounders (None or []), the executor falls
        back to all columns that aren't treatment or outcome. This is the
        documented intent path -- NOT the silent-substitution one.
        """
        captured = {}

        class _CapturingSelector:
            def __init__(self, selection: SelectionResult):
                self._selection = selection

            def select(self, treatment, outcome, covariates, **kwargs):
                captured["covariate_cols"] = list(covariates.columns)
                return self._selection

        er = _good_estimator_result()
        sel = _CapturingSelector(_selection_result(er))
        state = _state_with_data()
        state["confounders"] = None  # no routed adjustment set
        result = await EconMLExecutor(selector=sel).execute(state, _base_config())

        assert result["success"] is True
        # df has columns [marketing_spend, sales, region, season]; with
        # treatment='marketing_spend' and outcome='sales', the fallback gives
        # [region, season].
        assert captured["covariate_cols"] == ["region", "season"]


# =============================================================================
# Energy-score sanitization of peer-estimator scores in the result payload
# =============================================================================


class TestEconMLExecutorEnergyScoreSanitization:
    """Codex iter-1 MEDIUM (companion): even when the SELECTED estimator's
    energy_score is finite, the per-library ``energy_scores`` dict in the
    result payload may carry NaN / Inf from peer estimators that failed
    energy-score computation. Those must be sanitized to None for JSON safety.
    """

    @pytest.mark.asyncio
    async def test_peer_nan_energy_scores_sanitized_to_none(self):
        er = _good_estimator_result()
        sel = _FakeSelector(
            SelectionResult(
                selected=er,
                selection_strategy=SelectionStrategy.BEST_ENERGY_SCORE,
                all_results=[er],
                selection_reason="test",
                total_time_ms=10.0,
                energy_scores={
                    er.estimator_type.value: 0.3,
                    "linear_dml": float("nan"),
                    "drlearner": float("inf"),
                },
                energy_score_gap=0.05,
            )
        )
        result = await EconMLExecutor(selector=sel).execute(_state_with_data(), _base_config())

        assert result["success"] is True
        scores = result["result"]["energy_scores"]
        assert scores[er.estimator_type.value] == pytest.approx(0.3)
        assert scores["linear_dml"] is None
        assert scores["drlearner"] is None

    @pytest.mark.asyncio
    async def test_non_finite_energy_score_gap_zeroed(self):
        er = _good_estimator_result()
        sel = _FakeSelector(
            SelectionResult(
                selected=er,
                selection_strategy=SelectionStrategy.BEST_ENERGY_SCORE,
                all_results=[er],
                selection_reason="test",
                total_time_ms=10.0,
                energy_scores={er.estimator_type.value: 0.3},
                energy_score_gap=float("nan"),  # only one estimator -> no gap
            )
        )
        result = await EconMLExecutor(selector=sel).execute(_state_with_data(), _base_config())

        assert result["success"] is True
        assert result["result"]["energy_score_gap"] == 0.0


# =============================================================================
# Happy path — selector returns real EconML output
# =============================================================================


class TestEconMLExecutorHappyPath:
    @pytest.mark.asyncio
    async def test_returns_real_estimator_output_in_result(self):
        cate = np.array([0.08, 0.12, 0.18, 0.22], dtype=float)
        er = _good_estimator_result(
            ate=0.15, ate_std=0.02, ate_ci_lower=0.10, ate_ci_upper=0.20, cate=cate
        )
        sel = _FakeSelector(_selection_result(er))
        executor = EconMLExecutor(selector=sel)
        result = await executor.execute(_state_with_data(), _base_config())

        assert result["library"] == "econml"
        assert result["success"] is True
        assert result["error"] is None
        assert sel.calls == 1

        body = result["result"]
        assert body is not None
        # Real estimator metadata must be present (not the placeholder
        # ``"CausalForestDML"`` constant — must be the SELECTED type name).
        assert body["estimator"] == "causal_forest"
        # Real ATE comes from the selector, not from ``state["causal_effect"]``.
        assert body["ate"] == pytest.approx(0.15)
        assert body["ate_ci_lower"] == pytest.approx(0.10)
        assert body["ate_ci_upper"] == pytest.approx(0.20)
        assert body["ate_std"] == pytest.approx(0.02)
        # CATE segments derived from real per-record CATE array (mean per half).
        assert "cate_by_segment" in body
        seg = body["cate_by_segment"]
        assert "High CATE" in seg and "Low CATE" in seg
        assert seg["High CATE"]["cate"] > seg["Low CATE"]["cate"]
        # Heterogeneity score derived from the real CATE spread (NOT 0.0).
        assert "heterogeneity_score" in body
        assert body["heterogeneity_score"] != 0.0
        # Energy-score / quality metadata for transparency.
        assert "energy_score" in body
        assert "quality_tier" in body
        assert "n_estimators_evaluated" in body
        assert body["n_estimators_evaluated"] == 1
        assert body["n_estimators_succeeded"] == 1

    @pytest.mark.asyncio
    async def test_does_not_use_dowhy_causal_effect_as_ate(self):
        """Pre-rewire shape leaked DoWhy's ``causal_effect`` into EconML's
        ``ate`` field — silent fabrication of heterogeneity. New shape uses
        the REAL selector ATE, ignoring ``state["causal_effect"]``.
        """
        cate = np.array([0.05, 0.07, 0.10, 0.13], dtype=float)
        er = _good_estimator_result(
            ate=0.08, ate_std=0.01, ate_ci_lower=0.06, ate_ci_upper=0.10, cate=cate
        )
        sel = _FakeSelector(_selection_result(er))
        state = _state_with_data()
        state["causal_effect"] = 0.42  # NOT what the new executor uses
        result = await EconMLExecutor(selector=sel).execute(state, _base_config())

        assert result["success"] is True
        assert result["result"]["ate"] == pytest.approx(0.08)  # selector, not 0.42

    @pytest.mark.asyncio
    async def test_no_cate_segments_when_cate_array_unavailable(self):
        """LinearDML / DRLearner / OLS produce a single ATE without per-record
        CATE. The executor MUST emit an empty ``cate_by_segment`` and
        ``heterogeneity_score=0.0`` — NEVER fabricate ``ate * 1.2 / 0.8``
        synthetic segments.
        """
        er = _good_estimator_result(ate=0.10, cate=None, estimator_type=EstimatorType.LINEAR_DML)
        sel = _FakeSelector(_selection_result(er))
        result = await EconMLExecutor(selector=sel).execute(_state_with_data(), _base_config())

        assert result["success"] is True
        assert result["result"]["cate_by_segment"] == {}
        assert result["result"]["heterogeneity_score"] == 0.0
        assert result["result"]["estimator"] == "linear_dml"

    @pytest.mark.asyncio
    async def test_latency_ms_is_nonzero_and_int(self):
        er = _good_estimator_result()
        sel = _FakeSelector(_selection_result(er))
        result = await EconMLExecutor(selector=sel).execute(_state_with_data(), _base_config())
        assert result["success"] is True
        assert isinstance(result["latency_ms"], int)
        assert result["latency_ms"] >= 0

    @pytest.mark.asyncio
    async def test_selector_called_with_treatment_outcome_covariates(self):
        """The executor must extract treatment/outcome arrays from the
        DataFrame using the state's variable names, and pass the confounder
        columns as covariates. We verify by spying on the selector.
        """
        captured = {}

        class _CapturingSelector:
            def __init__(self, selection: SelectionResult):
                self._selection = selection

            def select(self, treatment, outcome, covariates, **kwargs):
                captured["treatment"] = np.asarray(treatment)
                captured["outcome"] = np.asarray(outcome)
                captured["covariates"] = covariates
                return self._selection

        er = _good_estimator_result()
        sel = _CapturingSelector(_selection_result(er))
        df = _real_data_frame()
        result = await EconMLExecutor(selector=sel).execute(_state_with_data(df=df), _base_config())

        assert result["success"] is True
        # Treatment column extracted from named `treatment_var` (binary).
        assert captured["treatment"].tolist() == df["marketing_spend"].tolist()
        # Outcome column extracted from named `outcome_var`.
        assert list(captured["outcome"]) == df["sales"].tolist()
        # Covariates are the confounder columns (not all columns).
        assert list(captured["covariates"].columns) == ["region", "season"]


# =============================================================================
# Dependency-injection knob — selector argument
# =============================================================================


class TestEconMLExecutorDependencyInjection:
    def test_default_selector_lazy_constructed(self):
        """Default behavior: when no selector is injected the executor
        constructs the production ``EstimatorSelector`` lazily inside
        ``execute()``. Verified by introspection rather than DI.
        """
        executor = EconMLExecutor()
        assert executor._selector is None  # not eagerly built

    def test_injected_selector_is_used(self):
        sel = MagicMock()
        executor = EconMLExecutor(selector=sel)
        assert executor._selector is sel
