"""EconML executor -- heterogeneous treatment-effect (CATE) stage of the pipeline.

Phase C-3 of GH #354 wired this executor to the production
``EstimatorSelector`` (V-04: ``causal_engine/energy_score/estimator_selector.py``)
which already wraps the real EconML libraries (``CausalForestDML``,
``LinearDML``, ``DRLearner``, ``DMLOrthoForest``). Per the dispatch brief, we
REUSE ``EstimatorSelector`` rather than reimplement the selection logic --
that class is also used by
``agents/causal_impact/nodes/estimation.py::_select_estimator_with_energy_score``
in production, and we deliberately inherit its fail-closed guards (F-006
iter-2..iter-5; #417).

Architectural constraint (pre-C-6): ``PipelineState`` (see
``pipeline/state.py``) carries ``data_source: str`` as a string identifier
only -- there is no in-state ``DataFrame``. Until a future phase (C-6
aggregation or a data-loader phase) wires a real backend, this executor
resolves data via the same key the existing production estimator uses
(``state['data_cache']['estimation_data']``, see
``agents/causal_impact/nodes/estimation.py::_get_data``). When that key is
absent, the executor FAIL-CLOSES with an explicit data-unavailability error
in ``LibraryExecutionResult.error``; it does NOT generate synthetic data,
copy ``state['causal_effect']`` as its own ATE, or return all-default values.

CLAUDE.md REASON-BEFORE-RULES + Anti-Mocking compliance: the pre-C-3 body was
a SCAFFOLDED PLACEHOLDER under product Q1/Q3 ("Yes, still planned -- finish
wiring it"; "scaffolded ahead of integration"). Per the 4-way framework this
called for REWIRE, not DELETE. The REWIRE is this file's GREEN phase.

Fail-closed guards (mirrored from
``agents/causal_impact/nodes/estimation.py::_select_estimator_with_energy_score``):

- Selector raises -> success=False with the underlying exception text.
- ``selected.success is False`` (all estimators failed) -> success=False.
- ``selected.ate is None`` -> success=False (would emit None as ATE).
- ``ate_ci_lower`` or ``ate_ci_upper`` is None -> success=False.
- CI bounds non-finite -> success=False.
- CI degenerate (lower >= upper) -> success=False (zero-width / inverted).
- ATE outside its own CI -> success=False (estimator internally inconsistent).
- ``ate_std`` is None -> success=False.
- ``ate_std`` non-finite or <= 0 -> success=False (no usable uncertainty).

Wave-3 forbidden-pattern audit:
- No ``np.random.seed`` / ``random.uniform`` in this module.
- No all-zero ``LibraryExecutionResult`` on data unavailability.
- No silent substitution of ``state['causal_effect']`` for the EconML ATE.
- No synthetic-data feed to the real estimator.
"""

from __future__ import annotations

import logging
import math
import time
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from ..router import CausalLibrary
from ..state import LibraryExecutionResult, PipelineConfig, PipelineState
from .base import LibraryExecutor

if TYPE_CHECKING:  # pragma: no cover -- import-cycle / heavy-import guard
    import pandas as pd

    from ...energy_score.estimator_selector import EstimatorSelector

logger = logging.getLogger(__name__)


# Quality-tier thresholds reused from
# ``agents/causal_impact/nodes/estimation.py::EstimationNode.QUALITY_TIERS``;
# kept inline here to avoid pulling in agent-layer imports from a causal-engine
# module (layering rule).
_QUALITY_TIERS = (
    ("excellent", 0.25),
    ("good", 0.45),
    ("acceptable", 0.65),
    ("poor", 0.80),
    ("unreliable", 1.0),
)


def _quality_tier(energy_score: float) -> str:
    """Map an energy score onto a quality tier (lower is better)."""
    if energy_score is None or not math.isfinite(energy_score):
        return "unreliable"
    for tier, threshold in _QUALITY_TIERS:
        if energy_score <= threshold:
            return tier
    return "unreliable"


def _build_cate_segments(cate: np.ndarray) -> dict[str, dict[str, Any]]:
    """Build per-segment CATE summary from a real per-record CATE array.

    Mirrors the high/low half-split in
    ``agents/causal_impact/nodes/estimation.py`` (iter-2 onward) -- uses REAL
    CATE means per half, NEVER the legacy ``ate * 1.2`` / ``ate * 0.8`` mock
    multipliers. Returns ``{}`` if the array is too small to split (the caller
    treats this as "heterogeneity not detected"; never substitutes a fake
    segment).
    """
    arr = np.asarray(cate, dtype=float).ravel()
    if arr.size < 2 or not np.all(np.isfinite(arr)):
        return {}
    threshold = float(np.median(arr))
    high_mask = arr >= threshold
    low_mask = ~high_mask
    out: dict[str, dict[str, Any]] = {}
    if high_mask.any():
        out["High CATE"] = {
            "cate": float(np.mean(arr[high_mask])),
            "size": int(high_mask.sum()),
            "description": "Records with CATE at or above median",
        }
    if low_mask.any():
        out["Low CATE"] = {
            "cate": float(np.mean(arr[low_mask])),
            "size": int(low_mask.sum()),
            "description": "Records with CATE below median",
        }
    return out


def _heterogeneity_score(cate: Optional[np.ndarray]) -> float:
    """Quantify CATE spread (coefficient-of-variation-like) when CATE is real.

    For single-ATE estimators (LinearDML, DRLearner, OLS) ``cate`` is None and
    we return 0.0 (NOT a fabricated heterogeneity); the executor flags the
    estimator's lack of per-record CATE by emitting an empty
    ``cate_by_segment`` dict.
    """
    if cate is None:
        return 0.0
    arr = np.asarray(cate, dtype=float).ravel()
    if arr.size < 2 or not np.all(np.isfinite(arr)):
        return 0.0
    spread = float(np.std(arr))
    mean = float(np.mean(np.abs(arr)))
    if mean <= 0.0:
        return 0.0
    return spread / mean


def _resolve_dataframe(state: PipelineState) -> Optional["pd.DataFrame"]:
    """Resolve the input DataFrame for estimation.

    Uses the same key as
    ``agents/causal_impact/nodes/estimation.py::_get_data`` so that future
    orchestrator wiring (likely phase C-6) only needs to populate
    ``state['data_cache']['estimation_data']`` once for all executors.

    Returns ``None`` if no DataFrame is available. The caller is responsible
    for fail-closing in that case -- this helper does NOT raise.
    """
    data_cache = state.get("data_cache")  # type: ignore[call-overload]
    if not isinstance(data_cache, dict):
        return None
    df = data_cache.get("estimation_data")
    if df is None:
        return None
    return df


def _failure(
    *,
    error: str,
    start_time: float,
    warnings_list: Optional[list[str]] = None,
) -> LibraryExecutionResult:
    """Build a fail-closed LibraryExecutionResult.

    Centralized so every fail-closed branch produces the same shape:
    ``success=False``, ``result=None``, ``confidence=0.0``, explicit ``error``.
    NO all-default / all-zero result body that could be mistaken for real
    output.
    """
    latency_ms = int((time.time() - start_time) * 1000)
    return LibraryExecutionResult(
        library="econml",
        success=False,
        latency_ms=latency_ms,
        result=None,
        error=error,
        confidence=0.0,
        warnings=list(warnings_list or []),
    )


class EconMLExecutor(LibraryExecutor):
    """Executor for EconML heterogeneous treatment effects.

    Constructor accepts an optional ``selector`` for dependency injection in
    tests (the production code path lazily constructs an
    ``EstimatorSelector`` with default config inside ``execute()``).
    """

    def __init__(self, selector: Optional["EstimatorSelector"] = None):
        # Stored as the underscore-prefixed attribute so tests can inspect
        # it without taking a dependency on a public accessor.
        self._selector: Optional["EstimatorSelector"] = selector

    @property
    def library(self) -> CausalLibrary:
        return CausalLibrary.ECONML

    async def execute(
        self,
        state: PipelineState,
        config: PipelineConfig,
    ) -> LibraryExecutionResult:
        """Execute EconML CATE estimation via the production ``EstimatorSelector``.

        Fails-closed on any of: missing inputs, missing DataFrame in
        ``state['data_cache']['estimation_data']``, selector exception,
        selector reporting all-estimators-failed, or selector returning
        internally-inconsistent ATE/CI/SE outputs.
        """
        start_time = time.time()

        # The outer try/except is the LAST-resort safety net for genuinely
        # unexpected exceptions (e.g. a DataFrame attribute access blowing up).
        # All EXPECTED failure modes are handled with explicit fail-closed
        # branches inside the try-block so the error messages are structured
        # rather than the raw exception text.
        try:
            # ---- Step 1: validate input vars ----
            valid, error = self.validate_input(state)
            if not valid:
                return _failure(error=error, start_time=start_time)

            treatment_var = state.get("treatment_var")
            outcome_var = state.get("outcome_var")
            confounders = state.get("confounders") or []

            # ---- Step 2: resolve DataFrame ----
            df = _resolve_dataframe(state)
            if df is None:
                return _failure(
                    error=(
                        "EconML executor requires real data via "
                        "state['data_cache']['estimation_data']; none found. "
                        "Refusing to fabricate ATE/CATE from synthetic or "
                        "DoWhy-leakage substitutes."
                    ),
                    start_time=start_time,
                )

            # ---- Step 3: extract treatment / outcome / covariates ----
            #
            # Note: we use named-column access (NOT positional fallback). The
            # production estimation.py path historically had a
            # ``data.iloc[:, 0/1]`` fallback when columns weren't found by
            # name; that's one of the documented silent-fallback trapdoors
            # (see issue_354_stub_vs_real_estimators_20260521.md). We do not
            # inherit it here -- if the column is missing, fail-closed
            # instead of silently picking the first/second column.
            if treatment_var not in df.columns:
                return _failure(
                    error=(
                        f"EconML executor: treatment column '{treatment_var}' "
                        f"not found in data_cache.estimation_data; columns="
                        f"{list(df.columns)}"
                    ),
                    start_time=start_time,
                )
            if outcome_var not in df.columns:
                return _failure(
                    error=(
                        f"EconML executor: outcome column '{outcome_var}' "
                        f"not found in data_cache.estimation_data; columns="
                        f"{list(df.columns)}"
                    ),
                    start_time=start_time,
                )

            treatment_col = np.asarray(df[treatment_var].values)
            outcome_col = np.asarray(df[outcome_var].values, dtype=float)

            # Binarize continuous treatment at the median (mirrors
            # estimation.py:170-174). EconML's CausalForestDML supports
            # continuous treatment, but the EstimatorSelector chain expects a
            # binary treatment for energy-score evaluation.
            if not np.array_equal(treatment_col, treatment_col.astype(int)):
                treatment_binary = (
                    treatment_col > np.median(treatment_col)
                ).astype(int)
            else:
                treatment_binary = treatment_col.astype(int)

            # Covariates: the explicit confounder list when present, else all
            # non-(treatment, outcome) columns. NEVER an empty DataFrame --
            # EconML estimators degenerate without covariates; we fail-closed
            # in that case.
            covariate_cols = [c for c in confounders if c in df.columns]
            if not covariate_cols:
                covariate_cols = [
                    c for c in df.columns if c not in (treatment_var, outcome_var)
                ]
            if not covariate_cols:
                return _failure(
                    error=(
                        "EconML executor: no covariate columns available "
                        "(neither confounders nor non-treatment/outcome "
                        "columns). EconML estimators require covariates."
                    ),
                    start_time=start_time,
                )
            covariates = df[covariate_cols]

            # ---- Step 4: get-or-build the selector and run it ----
            selector = self._selector
            if selector is None:
                # Local import to keep package import-time cheap and break
                # the import cycle (energy_score -> econml package vs. ours).
                from ...energy_score.estimator_selector import EstimatorSelector

                selector = EstimatorSelector()

            try:
                selection_result = selector.select(
                    treatment=treatment_binary,
                    outcome=outcome_col,
                    covariates=covariates,
                )
            except Exception as exc:  # noqa: BLE001 -- structured fail-closed
                logger.warning("EconML EstimatorSelector raised: %s", exc)
                return _failure(
                    error=f"EconML EstimatorSelector failed: {exc}",
                    start_time=start_time,
                )

            # ---- Step 5: enforce the selector's success contract ----
            selected = selection_result.selected
            if selected is None or not selected.success or selected.ate is None:
                failed_estimators = [
                    {
                        "estimator": r.estimator_type.value,
                        "error": r.error_message,
                    }
                    for r in (selection_result.all_results or [])
                    if not r.success
                ]
                return _failure(
                    error=(
                        "EconML: all configured estimators failed; refusing "
                        "to report ate=0.0 silent-wrong. "
                        f"failed_estimators={failed_estimators}"
                    ),
                    start_time=start_time,
                )

            # ---- Step 6: enforce CI / SE invariants (estimation.py iter-2..5) ----
            ate_lower = selected.ate_ci_lower
            ate_upper = selected.ate_ci_upper
            if ate_lower is None or ate_upper is None:
                return _failure(
                    error=(
                        "EconML: selected estimator produced an ATE without "
                        "confidence interval bounds; refusing to materialize "
                        "ate_ci=(0.0, 0.0) silent-wrong shape."
                    ),
                    start_time=start_time,
                )
            ate_lower_f = float(ate_lower)
            ate_upper_f = float(ate_upper)
            if not (math.isfinite(ate_lower_f) and math.isfinite(ate_upper_f)):
                return _failure(
                    error=(
                        f"EconML: non-finite CI bounds "
                        f"(lower={ate_lower_f}, upper={ate_upper_f})."
                    ),
                    start_time=start_time,
                )
            if ate_lower_f >= ate_upper_f:
                return _failure(
                    error=(
                        f"EconML: degenerate CI bounds "
                        f"(lower={ate_lower_f} >= upper={ate_upper_f})."
                    ),
                    start_time=start_time,
                )
            ate_f = float(selected.ate)
            if not (ate_lower_f <= ate_f <= ate_upper_f):
                return _failure(
                    error=(
                        f"EconML: ATE outside its own CI "
                        f"(ate={ate_f}, ci=[{ate_lower_f}, {ate_upper_f}])."
                    ),
                    start_time=start_time,
                )

            ate_std = selected.ate_std
            if ate_std is None:
                return _failure(
                    error=(
                        "EconML: selected estimator produced an ATE without "
                        "a standard error; refusing to materialize ate_std=0.0 "
                        "silent-wrong shape."
                    ),
                    start_time=start_time,
                )
            ate_std_f = float(ate_std)
            if not math.isfinite(ate_std_f) or ate_std_f <= 0.0:
                return _failure(
                    error=(
                        f"EconML: unusable standard error (ate_std={ate_std_f})."
                    ),
                    start_time=start_time,
                )

            # ---- Step 7: pack real outputs into LibraryExecutionResult.result ----
            cate_arr: Optional[np.ndarray] = None
            if selected.cate is not None:
                cate_arr = np.asarray(selected.cate, dtype=float)
            cate_segments = _build_cate_segments(cate_arr) if cate_arr is not None else {}
            het_score = _heterogeneity_score(cate_arr)
            energy_score_f = float(selected.energy_score)

            successful_results = [
                r for r in (selection_result.all_results or []) if r.success
            ]

            result: dict[str, Any] = {
                "estimator": selected.estimator_type.value,
                "ate": ate_f,
                "ate_ci_lower": ate_lower_f,
                "ate_ci_upper": ate_upper_f,
                "ate_std": ate_std_f,
                "cate_by_segment": cate_segments,
                "heterogeneity_score": het_score,
                "energy_score": energy_score_f,
                "quality_tier": _quality_tier(energy_score_f),
                "selection_strategy": selection_result.selection_strategy.value,
                "selection_reason": selection_result.selection_reason,
                "energy_scores": dict(selection_result.energy_scores or {}),
                "energy_score_gap": float(selection_result.energy_score_gap),
                "n_estimators_evaluated": len(selection_result.all_results or []),
                "n_estimators_succeeded": len(successful_results),
            }

            latency_ms = int((time.time() - start_time) * 1000)
            # Confidence reflects the selector's quality assessment, NOT a
            # hardcoded 0.82. Map energy_score (lower better) to a confidence
            # in [0, 1] via 1 - clip(energy_score, 0, 1).
            confidence = max(0.0, min(1.0, 1.0 - energy_score_f))

            return LibraryExecutionResult(
                library="econml",
                success=True,
                latency_ms=latency_ms,
                result=result,
                error=None,
                confidence=confidence,
                warnings=[],
            )

        except Exception as e:  # noqa: BLE001 -- last-resort safety net
            logger.error("EconML execution failed: %s", e)
            return _failure(error=str(e), start_time=start_time)

    def validate_input(self, state: PipelineState) -> tuple[bool, str]:
        """Validate input for EconML analysis."""
        if not state.get("treatment_var"):
            return False, "EconML requires treatment_var"
        if not state.get("outcome_var"):
            return False, "EconML requires outcome_var"
        return True, ""
