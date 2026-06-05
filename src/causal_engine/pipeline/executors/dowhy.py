"""DoWhy executor — causal identification + estimation stage of the pipeline.

Wave-1 phase C-2 of GH #354. Wires `DoWhyExecutor.execute()` to a real
`dowhy.CausalModel` run (identify_effect → estimate_effect) against a
DataFrame conveyed in `state["filters"]["estimation_data"]`. Mirrors the
production-mature wrap point at:

- `causal_engine/refutation_runner.py:35` (`from dowhy import CausalModel`
  — §0 V-03 of the dispatch plan)
- `agents/causal_impact/nodes/refutation.py:_reconstruct_dowhy_artifacts`
  (refutation.py:206-247) — the production-mature pattern for building a
  CausalModel from estimation passthrough data with fail-closed semantics.

Fail-closed semantics (per CLAUDE.md anti-mocking discipline + dispatch
plan R2/R9):

- No DataFrame in state → `success=False, error="DoWhy executor requires
  a DataFrame ..."`. NEVER fall back to synthetic data, NEVER return the
  placeholder `causal_effect=0.0` / `identified_estimand="backdoor"` shape.
- DoWhy unavailable (`DOWHY_AVAILABLE=False`) → `success=False`.
- DataFrame missing treatment/outcome/confounder columns → `success=False`.
- State missing `treatment_var` / `outcome_var` → `success=False`.
- DoWhy `identify_effect` or `estimate_effect` raises → `success=False`
  with the underlying error preserved in `error`. NEVER silently substitute
  a different signal (Wave-3 pattern #4).

The ABC contract (`{library, execute, validate_input}`) and
`LibraryExecutionResult` TypedDict shape are LOCKED in C-1 — this PR
does not touch either.

Cross-refs:
- Dispatch plan: .claude/plans/354_dispatch_plan_v1.md §2.2 C-2
- Design plan: .claude/plans/causal_engine_canonical_routing_v4.md §1-§5
- Brief template: .claude/dispatch/354_executor_brief_template.md
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np

from ..data_resolver import resolve_estimation_dataframe
from ..router import CausalLibrary
from ..state import LibraryExecutionResult, PipelineConfig, PipelineState
from .base import LibraryExecutor

logger = logging.getLogger(__name__)

# Conditional DoWhy import for graceful degradation. Mirrors the pattern at
# `causal_engine/refutation_runner.py:33-40` so DoWhy unavailability surfaces
# as a structured fail-closed at execute() time rather than an import error
# at module load time.
try:
    from dowhy import CausalModel  # type: ignore[import-not-found]

    DOWHY_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised in environments without dowhy
    DOWHY_AVAILABLE = False
    CausalModel = None  # type: ignore[assignment,misc]


# DataFrame resolution is delegated to ``data_resolver.resolve_estimation_dataframe``
# (#458). The resolver prefers the first-class ``state["estimation_data"]``
# slot and back-compats the Wave-1 nested-dict shapes with a
# ``DeprecationWarning`` so per-executor key drift cannot re-appear.


def _resolve_dowhy_method(state: PipelineState) -> str:
    """Choose the DoWhy `method_name` for the estimate_effect call.

    The PipelineState carries no explicit estimator-selection signal (that
    would be C-3 EconML's responsibility under Wave-1). We default to
    ``backdoor.linear_regression`` — DoWhy's most universally-available
    backdoor identifier — as the pipeline-stage DoWhy method. Callers that
    need a different estimator can pass a ``dowhy_method`` key in
    ``state['filters']`` (forward-compatible).

    Important: this is NOT a silent fallback in the harmful sense. It is a
    documented default for an under-specified pipeline stage. The full
    cross-estimator alignment work (mapping selected_estimator → DoWhy
    method, as `agents/causal_impact/nodes/refutation.py:_resolve_dowhy_method`
    does) belongs in C-6 aggregation where heterogeneous executor outputs
    are reconciled.
    """
    filters = state.get("filters") or {}
    if isinstance(filters, dict):
        method = filters.get("dowhy_method")
        if isinstance(method, str) and method:
            return method
    return "backdoor.linear_regression"


def _build_failure_result(
    *,
    start_time: float,
    error: str,
    warnings: Optional[List[str]] = None,
) -> LibraryExecutionResult:
    """Build a fail-closed LibraryExecutionResult.

    NEVER returns hardcoded placeholder values for `causal_effect` /
    `identified_estimand` / `confidence_interval`. The TypedDict's
    `result` field is set to None so downstream aggregation
    (`_update_state_with_result` in orchestrator.py:172-178) sees an
    explicit "no result" rather than fake zeros.
    """
    latency_ms = int((time.time() - start_time) * 1000)
    return LibraryExecutionResult(
        library="dowhy",
        success=False,
        latency_ms=latency_ms,
        result=None,
        error=error,
        confidence=0.0,
        warnings=warnings or [],
    )


class DoWhyExecutor(LibraryExecutor):
    """Executor for DoWhy causal identification + estimation.

    Real-library wiring landed in phase C-2 of GH #354. See module docstring
    for full design notes.
    """

    @property
    def library(self) -> CausalLibrary:
        return CausalLibrary.DOWHY

    async def execute(
        self,
        state: PipelineState,
        config: PipelineConfig,
    ) -> LibraryExecutionResult:
        """Execute real DoWhy identify_effect → estimate_effect on state data.

        Workflow:
        1. Fail-closed if DoWhy is unavailable.
        2. Fail-closed if treatment_var / outcome_var are missing from state.
        3. Fail-closed if no DataFrame is present in state['filters'].
        4. Fail-closed if required columns (treatment, outcome, confounders)
           are missing from the DataFrame.
        5. Build `dowhy.CausalModel(data, treatment, outcome, common_causes)`.
        6. Call `model.identify_effect(proceed_when_unidentifiable=True)`.
        7. Call `model.estimate_effect(identified_estimand, method_name=...)`.
        8. Pack outputs into LibraryExecutionResult.result (TypedDict-locked
           shape: `result: Optional[Dict[str, Any]]`).

        Returns:
            LibraryExecutionResult with success=True + populated result on the
            happy path; success=False + descriptive error on any failure
            path. The result payload on success carries:
            - `causal_effect: float` — DoWhy-derived numeric ATE
            - `identified_estimand: str` — `repr(identified_estimand)`-derived label
            - `dowhy_method: str` — the method_name actually used
            - `treatment_var`, `outcome_var`, `common_causes` — what we passed to DoWhy
            - `graph_source` — `"networkx"` if NetworkX upstream populated
              `state['causal_graph']`, else `"inferred"`
        """
        start_time = time.time()
        warnings_acc: List[str] = []

        # === Step 1: DoWhy availability ===
        if not DOWHY_AVAILABLE or CausalModel is None:
            return _build_failure_result(
                start_time=start_time,
                error=(
                    "DoWhy library is not available in this environment. "
                    "Install via `pip install dowhy` to enable the DoWhyExecutor."
                ),
            )

        # === Step 2: validate state (treatment_var + outcome_var present) ===
        is_valid, validation_error = self.validate_input(state)
        if not is_valid:
            return _build_failure_result(
                start_time=start_time,
                error=f"DoWhy input validation failed: {validation_error}",
            )

        treatment_var = state["treatment_var"]
        outcome_var = state["outcome_var"]
        assert treatment_var is not None  # validated above; help mypy
        assert outcome_var is not None

        confounders: List[str] = list(state.get("confounders") or [])

        # === Step 3: DataFrame passthrough ===
        data = resolve_estimation_dataframe(state)
        if data is None:
            return _build_failure_result(
                start_time=start_time,
                error=(
                    "DoWhy executor requires a DataFrame in state "
                    "(first-class estimation_data field, or legacy filters/"
                    "data_cache slots); no DataFrame was provided. Refusing "
                    "to fabricate synthetic data."
                ),
            )

        # === Step 4: column validation on the DataFrame ===
        try:
            df_columns = set(data.columns)  # type: ignore[union-attr]
        except Exception as col_exc:  # noqa: BLE001 - guard against non-DataFrame input
            return _build_failure_result(
                start_time=start_time,
                error=(
                    "DoWhy executor: resolved DataFrame is malformed "
                    f"(cannot read .columns): {col_exc}"
                ),
            )

        missing_columns: List[str] = []
        if treatment_var not in df_columns:
            missing_columns.append(treatment_var)
        if outcome_var not in df_columns:
            missing_columns.append(outcome_var)
        for cc in confounders:
            if cc not in df_columns:
                missing_columns.append(cc)
        if missing_columns:
            return _build_failure_result(
                start_time=start_time,
                error=(
                    f"DoWhy executor: DataFrame is missing required columns "
                    f"{missing_columns}. Available columns: {sorted(df_columns)}. "
                    "Refusing to silently substitute."
                ),
            )

        # === Step 5: build CausalModel ===
        dowhy_method = _resolve_dowhy_method(state)
        try:
            model = CausalModel(
                data=data,
                treatment=treatment_var,
                outcome=outcome_var,
                common_causes=confounders if confounders else None,
            )
        except Exception as exc:  # noqa: BLE001 - DoWhy raises broad-typed errors
            return _build_failure_result(
                start_time=start_time,
                error=(
                    f"DoWhy CausalModel construction failed for treatment="
                    f"{treatment_var!r}, outcome={outcome_var!r}, "
                    f"common_causes={confounders!r}: {exc}"
                ),
            )

        # === Step 6: identify_effect ===
        try:
            identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
        except Exception as exc:  # noqa: BLE001
            return _build_failure_result(
                start_time=start_time,
                error=f"DoWhy identify_effect failed: {exc}",
            )

        # === Step 7: estimate_effect ===
        try:
            estimate = model.estimate_effect(
                identified_estimand,
                method_name=dowhy_method,
                test_significance=False,
            )
        except Exception as exc:  # noqa: BLE001
            return _build_failure_result(
                start_time=start_time,
                error=(f"DoWhy estimate_effect failed for method_name={dowhy_method!r}: {exc}"),
            )

        # === Step 8: extract real numeric ATE; fail-closed on non-finite ===
        try:
            causal_effect_raw = getattr(estimate, "value", None)
            if causal_effect_raw is None:
                raise ValueError("DoWhy estimate has no .value attribute")
            causal_effect = float(causal_effect_raw)
        except (TypeError, ValueError, AttributeError) as exc:
            return _build_failure_result(
                start_time=start_time,
                error=(
                    "DoWhy estimate_effect returned a non-numeric value "
                    f"(method_name={dowhy_method!r}): {exc}"
                ),
            )
        if not np.isfinite(causal_effect):
            return _build_failure_result(
                start_time=start_time,
                error=(
                    f"DoWhy estimate_effect returned non-finite value "
                    f"{causal_effect} (method_name={dowhy_method!r})."
                ),
            )

        # === Step 9: derive identified_estimand label ===
        # DoWhy's identified_estimand has rich structure (estimand_type,
        # backdoor_variables, etc.). The orchestrator's
        # _update_state_with_result (orchestrator.py:178) only reads it via
        # .get("identified_estimand") on the dict, so we surface a stable
        # human-readable label and also stash the full repr for inspection.
        estimand_label = self._extract_estimand_label(identified_estimand)

        # === Step 10: graph source bookkeeping ===
        graph_source = "networkx" if state.get("causal_graph") else "inferred"

        # H9: best-effort standard error of the ATE so the consensus aggregator
        # can weight DoWhy by PRECISION (inverse-variance) instead of its
        # hardcoded confidence=1.0, which otherwise structurally dominates the
        # blended consensus. None when DoWhy's estimator does not expose one.
        dowhy_se: Optional[float] = None
        try:
            se_raw = estimate.get_standard_error()
            se_val = float(np.ravel(se_raw)[0]) if se_raw is not None else None
            if se_val is not None and np.isfinite(se_val) and se_val > 0:
                dowhy_se = se_val
        except Exception:  # noqa: BLE001 - SE is method-dependent; absence is fine
            dowhy_se = None

        # === Step 11: build success result ===
        latency_ms = int((time.time() - start_time) * 1000)
        result_payload: Dict[str, Any] = {
            "causal_effect": causal_effect,
            "standard_error": dowhy_se,
            "identified_estimand": estimand_label,
            "identified_estimand_repr": repr(identified_estimand),
            "dowhy_method": dowhy_method,
            "treatment_var": treatment_var,
            "outcome_var": outcome_var,
            "common_causes": confounders,
            "graph_source": graph_source,
            # Empty until C-2/C-6 wires refutation as a pipeline stage;
            # the placeholder shape (empty dict) is intentional and
            # documented — see refutation_runner for the real refutation
            # path that C-6+ may invoke.
            "refutation_results": {},
        }

        # Confidence policy: this executor reports a confidence of 1.0 for
        # a successful identify+estimate call and 0.0 for any fail-closed
        # path. A more nuanced confidence (e.g., based on refutation pass
        # rate) belongs in C-6 aggregation where refutation results are
        # available. Using `1.0` rather than `0.85` makes the post-wiring
        # value distinguishable from the C-1 stub-shape constant `0.85`.
        confidence_value = 1.0

        return LibraryExecutionResult(
            library="dowhy",
            success=True,
            latency_ms=latency_ms,
            result=result_payload,
            error=None,
            confidence=confidence_value,
            warnings=warnings_acc,
        )

    @staticmethod
    def _extract_estimand_label(identified_estimand: Any) -> str:
        """Derive a stable human-readable label from a DoWhy IdentifiedEstimand.

        DoWhy's identified_estimand object exposes an `estimand_type`
        attribute (string-like) that identifies the identification strategy
        (e.g., "nonparametric-ate", "backdoor"). We prefer that label;
        fall back to the class name if estimand_type is missing.
        """
        estimand_type = getattr(identified_estimand, "estimand_type", None)
        if estimand_type is not None:
            return str(estimand_type)
        return type(identified_estimand).__name__

    def validate_input(self, state: PipelineState) -> tuple[bool, str]:
        """Validate input for DoWhy analysis.

        Semantics preserved from C-1 placeholder (locked by ABC contract):
        require both `treatment_var` and `outcome_var` to be present in
        state. DataFrame presence is checked inside `execute()` (not here)
        because `validate_input` is invoked by orchestrators before data
        passthrough is necessarily populated.
        """
        if not state.get("treatment_var"):
            return False, "DoWhy requires treatment_var"
        if not state.get("outcome_var"):
            return False, "DoWhy requires outcome_var"
        return True, ""
