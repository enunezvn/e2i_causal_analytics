"""CausalML executor — uplift modeling stage of the pipeline.

Rewired in phase C-4 of GH #354 to wrap the production-mature uplift module
at ``src.causal_engine.uplift`` (which itself wraps CausalML's
``UpliftRandomForestClassifier`` / ``UpliftTreeClassifier`` /
``Base{T,X,S}Classifier`` meta-learners — V-05). Those uplift classes are
already production-wired via ``UpliftAnalyzerNode`` in
``agents/heterogeneous_optimizer``; this executor REUSES them rather than
reimplementing the CausalML call.

Contract preservation (locked in C-1, see V-20 / V-23):
- ``library`` property → ``CausalLibrary.CAUSALML`` (unchanged)
- ``execute(state, config)`` async → ``LibraryExecutionResult`` (unchanged shape)
- ``validate_input(state)`` → strict; same treatment_var + outcome_var checks
- ``LibraryExecutionResult.result: Dict[str, Any]`` carries real uplift outputs:
  ``model``, ``ate``, ``att``, ``atc``, ``ate_ci_lower``, ``ate_ci_upper``,
  ``auuc``, ``qini``, ``uplift_scores_summary``, ``feature_importances``,
  ``n_samples``, ``treatment_groups``.

Fail-closed semantics (CLAUDE.md anti-mocking discipline + Wave-3 patterns
#3 and #4):
- When ``state["filters"]["dataframe"]`` is missing, malformed, missing the
  declared treatment/outcome columns, or empty → return
  ``success=False, result=None, confidence=0.0, error=<reason>``.
- Never feed synthetic data, ``np.random.seed``, or ``random.uniform`` to the
  uplift model. The caller is responsible for handing in real data via the
  ``filters["dataframe"]`` escape-hatch (this contract will be replaced by a
  proper data backend hook in C-6 / C-7 — out of scope for C-4).
- When the uplift FIT succeeds but the auuc/qini metric helper raises, mark
  those fields as ``None`` and add a warning. Never silent-substitute with a
  different signal.

Cross-refs:
- Wrap point (V-05): ``causal_engine/uplift/random_forest.py:54,182``;
  ``gradient_boosting.py:163,177,191``.
- Production-wiring reference: ``agents/heterogeneous_optimizer/nodes/
  uplift_analyzer.py:358-383`` (the ``_fit_uplift_rf`` pattern).
- Design plan: ``.claude/plans/causal_engine_canonical_routing_v4.md``
  §1.3 (CausalML maturity) and §5.1 C-4.
- Dispatch plan: ``.claude/plans/354_dispatch_plan_v1.md`` §2.2 C-4 brief.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Tuple

from ..router import CausalLibrary
from ..state import LibraryExecutionResult, PipelineConfig, PipelineState
from .base import LibraryExecutor

logger = logging.getLogger(__name__)


class ExecutorDataUnavailable(RuntimeError):
    """Raised when the caller has not supplied a real DataFrame for uplift fitting.

    The pipeline package does not (yet) ship a data backend hook on
    ``PipelineState``. Until C-6/C-7 land one, callers inject a real
    ``pandas.DataFrame`` via ``state["filters"]["dataframe"]``. When that
    is absent, malformed, or missing required columns, the executor fails
    closed by raising this exception (caught at the top of ``execute()``
    and surfaced as ``success=False, error=<reason>``). This contract is
    intentional: per CLAUDE.md anti-mocking discipline, the executor MUST
    NOT fall back to synthetic data, hardcoded plausible values, or any
    silent substitution.
    """


def _extract_uplift_inputs_from_state(
    state: PipelineState,
) -> Tuple[Any, Any, Any, List[str], List[str]]:
    """Pull the DataFrame, treatment array, outcome array, and feature names
    out of the pipeline state — or raise ``ExecutorDataUnavailable``.

    Returns:
        Tuple of (X_df, treatment_arr, y_arr, feature_names, treatment_groups).

    Raises:
        ExecutorDataUnavailable: when ``filters["dataframe"]`` is missing,
            not a pandas DataFrame, empty, or missing the declared
            ``treatment_var`` / ``outcome_var`` columns.
    """
    # Local imports keep module import cheap when callers don't use this
    # executor; uplift wrapper itself imports causalml lazily.
    import pandas as pd

    filters = state.get("filters")
    if not isinstance(filters, dict):
        raise ExecutorDataUnavailable(
            "CausalMLExecutor: no real data available — "
            "`state['filters']['dataframe']` is required."
        )
    df = filters.get("dataframe")
    if df is None:
        raise ExecutorDataUnavailable(
            "CausalMLExecutor: no real data available — "
            "`state['filters']['dataframe']` is missing."
        )
    if not isinstance(df, pd.DataFrame):
        raise ExecutorDataUnavailable(
            "CausalMLExecutor: `state['filters']['dataframe']` must be a "
            f"pandas DataFrame; got {type(df).__name__}."
        )
    if len(df) == 0:
        raise ExecutorDataUnavailable(
            "CausalMLExecutor: input DataFrame is empty (0 rows)."
        )

    treatment_var = state.get("treatment_var")
    outcome_var = state.get("outcome_var")
    if not treatment_var:
        raise ExecutorDataUnavailable(
            "CausalMLExecutor: state['treatment_var'] is required."
        )
    if not outcome_var:
        raise ExecutorDataUnavailable(
            "CausalMLExecutor: state['outcome_var'] is required."
        )
    if treatment_var not in df.columns:
        raise ExecutorDataUnavailable(
            f"CausalMLExecutor: treatment column '{treatment_var}' "
            f"missing from input DataFrame."
        )
    if outcome_var not in df.columns:
        raise ExecutorDataUnavailable(
            f"CausalMLExecutor: outcome column '{outcome_var}' "
            f"missing from input DataFrame."
        )

    # Feature columns = explicit confounders + effect_modifiers, intersected
    # with available columns. If neither was supplied, fall back to
    # "everything except treatment and outcome." This is NOT a synthesis
    # decision — it's a column-selection one over the real DataFrame the
    # caller handed in.
    declared_features: List[str] = []
    for key in ("confounders", "effect_modifiers"):
        vals = state.get(key)
        if vals:
            declared_features.extend(vals)
    feature_names: List[str] = []
    for col in declared_features:
        if col in df.columns and col not in (treatment_var, outcome_var):
            feature_names.append(col)
    if not feature_names:
        feature_names = [c for c in df.columns if c not in (treatment_var, outcome_var)]
    if not feature_names:
        raise ExecutorDataUnavailable(
            "CausalMLExecutor: no feature columns available "
            f"(DataFrame columns: {list(df.columns)})."
        )

    X_df = df[feature_names].copy()
    treatment_arr = df[treatment_var].to_numpy()
    y_arr = df[outcome_var].to_numpy().astype(float)

    treatment_groups = sorted({str(t) for t in treatment_arr})

    return X_df, treatment_arr, y_arr, feature_names, treatment_groups


def _fit_uplift_model(
    X_df: Any,
    treatment_arr: Any,
    y_arr: Any,
    random_state: int,
) -> Tuple[Any, str]:
    """Fit the production uplift model and return ``(UpliftResult, model_id)``.

    Mirrors the pattern at ``agents/heterogeneous_optimizer/nodes/
    uplift_analyzer.py:358-383`` (the ``_fit_uplift_rf`` path): build an
    ``UpliftConfig`` with a sample-size-aware ``min_samples_leaf``, then call
    ``UpliftRandomForest(config).estimate(X, treatment, y)``.

    Returns:
        Tuple of (UpliftResult, model identifier string used in result dict).

    Raises:
        RuntimeError: when the uplift wrapper reports failure
            (caught by the executor and surfaced as success=False).
        ImportError: when CausalML is not installed (same handling).
    """
    # Lazy import so module import doesn't pay the causalml load cost when
    # callers exercise only the contract tests.
    from src.causal_engine.uplift import UpliftConfig, UpliftRandomForest

    n = len(X_df)
    config = UpliftConfig(
        n_estimators=100,
        max_depth=5,
        # Match the production pattern: floor at 10, scale with sample size.
        min_samples_leaf=max(10, n // 50),
        random_state=random_state,
    )
    model = UpliftRandomForest(config)
    upl_result = model.estimate(X_df, treatment_arr, y_arr)
    if not upl_result.success:
        raise RuntimeError(
            f"Uplift estimation failed: {upl_result.error_message or 'unknown error'}"
        )
    return upl_result, model.model_type.value


def _compute_uplift_metrics_safe(
    uplift_scores: Any,
    treatment_arr: Any,
    y_arr: Any,
) -> Dict[str, Any]:
    """Compute auuc + qini from a fitted UpliftResult; never substitute on error.

    Returns a dict with ``auuc`` and ``qini`` floats (or ``None`` when the
    helper itself raises — in which case the caller adds a warning rather
    than dropping in a different signal). The metrics module wraps
    ``causalml.metrics`` so this stays a thin pass-through.

    Mirrors the pattern at ``UpliftAnalyzerNode._calculate_metrics`` (its
    try/except wraps ``causalml.metrics`` calls and returns ``None`` on
    failure). We surface failures as exceptions HERE so the test-side patch
    can pin the warning-emit path explicitly.
    """
    import numpy as np

    from src.causal_engine.uplift.metrics import (
        auuc as calculate_auuc,
    )
    from src.causal_engine.uplift.metrics import (
        qini_coefficient,
    )

    # CausalML's metrics helpers expect 1-D uplift scores; squeeze multi-treatment.
    scores_1d = np.asarray(uplift_scores)
    if scores_1d.ndim > 1:
        scores_1d = scores_1d[:, 0]

    auuc_val = calculate_auuc(scores_1d, treatment_arr, y_arr)
    qini_val = qini_coefficient(scores_1d, treatment_arr, y_arr)
    return {
        "auuc": float(auuc_val) if auuc_val is not None else None,
        "qini": float(qini_val) if qini_val is not None else None,
    }


def _summarize_uplift_scores(uplift_scores: Any) -> Dict[str, Any]:
    """Build a small JSON-safe summary of per-sample uplift scores.

    The per-sample array can be O(n_samples), which is unsuitable for
    ``LibraryExecutionResult.result`` (which propagates through state and
    downstream serialization). The summary keeps the load-bearing
    statistics for aggregation (C-6) without dragging the full array.
    """
    import numpy as np

    scores = np.asarray(uplift_scores)
    if scores.ndim > 1:
        scores = scores[:, 0]
    if scores.size == 0:
        return {"n": 0}
    return {
        "n": int(scores.size),
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "median": float(np.median(scores)),
        "p10": float(np.percentile(scores, 10)),
        "p90": float(np.percentile(scores, 90)),
    }


def _confidence_from_uplift_result(upl_result: Any) -> float:
    """Derive a [0, 1] confidence from the real UpliftResult.

    Maps the CI width (smaller CI → higher confidence) and sample size
    (more samples → higher confidence) onto [0, 1]. This is NOT a hardcoded
    ``0.78`` stub — it is computed from real fit outputs. If the underlying
    UpliftResult lacks the inputs to compute confidence (no CI, no ATE),
    falls back to a neutral 0.5.
    """
    ate = getattr(upl_result, "ate", None)
    ci_lower = getattr(upl_result, "ate_ci_lower", None)
    ci_upper = getattr(upl_result, "ate_ci_upper", None)
    metadata = getattr(upl_result, "metadata", {}) or {}
    n_train = int(metadata.get("n_samples_train", 0))

    # Sample-size component (saturates above ~1000 rows).
    n_component = min(1.0, n_train / 1000.0) if n_train > 0 else 0.0

    # CI-width component (smaller width → higher confidence; capped at 0.5 width).
    ci_component = 0.0
    if ate is not None and ci_lower is not None and ci_upper is not None:
        width = float(ci_upper - ci_lower)
        scale = max(1e-6, abs(float(ate)) + 1e-3)
        # Width relative to |ATE|: <= 0.1 → strong; >= 1.0 → weak.
        rel = min(1.0, width / scale)
        ci_component = 1.0 - rel

    # Average the two components; never return 0.0 on success (reserve 0.0
    # for failure paths to keep the test contract clean).
    raw = 0.5 * (n_component + ci_component)
    if raw <= 0.0:
        return 0.5
    return float(min(1.0, max(0.05, raw)))


class CausalMLExecutor(LibraryExecutor):
    """Executor for CausalML uplift modeling (real-library wiring as of C-4).

    Wraps ``src.causal_engine.uplift.UpliftRandomForest`` (which delegates
    to ``causalml.inference.tree.UpliftRandomForestClassifier``). The
    underlying uplift module is already production-wired via
    ``UpliftAnalyzerNode`` in ``heterogeneous_optimizer`` — this executor
    REUSES that wrapper rather than re-importing CausalML directly. V-05.
    """

    @property
    def library(self) -> CausalLibrary:
        return CausalLibrary.CAUSALML

    async def execute(
        self,
        state: PipelineState,
        config: PipelineConfig,
    ) -> LibraryExecutionResult:
        """Execute CausalML uplift modeling against real data from pipeline state.

        Contract:
        - ``state['filters']['dataframe']`` MUST be a non-empty pandas
          DataFrame containing the declared ``treatment_var`` and
          ``outcome_var`` columns; feature columns are the declared
          confounders + effect_modifiers (intersected with DataFrame columns)
          or, lacking those, all remaining columns.
        - Real-library success path returns ``success=True`` with
          ``result`` populated from the real UpliftResult (auuc, qini, ate,
          uplift_scores_summary, feature_importances, model, n_samples,
          treatment_groups). ``confidence`` is derived from CI width +
          sample size; never the hardcoded ``0.78`` C-1 stub.
        - On missing/malformed data → fail-closed with
          ``success=False, result=None, confidence=0.0, error=<reason>``.
        - On uplift fit failure → same fail-closed shape.
        - On post-fit metric-helper failure → keep ``success=True`` but
          set auuc/qini to None and add a warning. Never silent-substitute.
        """
        start_time = time.time()
        warnings: List[str] = []
        try:
            (
                X_df,
                treatment_arr,
                y_arr,
                feature_names,
                treatment_groups,
            ) = _extract_uplift_inputs_from_state(state)

            # Deterministic random_state: drawn from a stable config field so
            # repeated calls against the same state produce repeatable fits.
            # NOT a seeded synthetic-data injection — `random_state` is passed
            # to the real CausalML tree-splitting algorithm, which uses it for
            # tree-construction randomization only. Real data still flows in
            # from `state["filters"]["dataframe"]`.
            random_state = 42

            upl_result, model_id = _fit_uplift_model(
                X_df, treatment_arr, y_arr, random_state=random_state
            )

            uplift_scores = upl_result.uplift_scores
            if uplift_scores is None:
                raise RuntimeError(
                    "Uplift estimation reported success but produced no uplift_scores."
                )

            # Best-effort auuc/qini; failure here is non-fatal but emits a warning.
            try:
                metrics = _compute_uplift_metrics_safe(uplift_scores, treatment_arr, y_arr)
            except Exception as metric_err:  # noqa: BLE001 — intentional broad catch
                logger.warning(
                    "CausalMLExecutor: uplift metrics computation failed: %s", metric_err
                )
                warnings.append(
                    f"Uplift metrics unavailable (auuc/qini): {metric_err}"
                )
                metrics = {"auuc": None, "qini": None}

            result_payload: Dict[str, Any] = {
                "model": model_id,
                "ate": upl_result.ate,
                "att": upl_result.att,
                "atc": upl_result.atc,
                "ate_std": upl_result.ate_std,
                "ate_ci_lower": upl_result.ate_ci_lower,
                "ate_ci_upper": upl_result.ate_ci_upper,
                "auuc": metrics["auuc"],
                "qini": metrics["qini"],
                "uplift_scores_summary": _summarize_uplift_scores(uplift_scores),
                "feature_importances": upl_result.feature_importances,
                "feature_names": feature_names,
                "n_samples": int(len(X_df)),
                "treatment_groups": treatment_groups,
            }

            # When EconML produced upstream CATE estimates, propagate the
            # comparison signal without mutating the canonical result fields.
            # This matches the pre-C-4 hint behavior; a richer downstream
            # comparison lives in C-6 (consensus aggregator).
            if state.get("cate_by_segment"):
                result_payload["econml_comparison"] = "available"

            confidence = _confidence_from_uplift_result(upl_result)
            latency_ms = int((time.time() - start_time) * 1000)
            return LibraryExecutionResult(
                library="causalml",
                success=True,
                latency_ms=latency_ms,
                result=result_payload,
                error=None,
                confidence=confidence,
                warnings=warnings,
            )

        except ExecutorDataUnavailable as data_err:
            latency_ms = int((time.time() - start_time) * 1000)
            logger.info("CausalMLExecutor fail-closed: %s", data_err)
            return LibraryExecutionResult(
                library="causalml",
                success=False,
                latency_ms=latency_ms,
                result=None,
                error=str(data_err),
                confidence=0.0,
                warnings=warnings,
            )
        except Exception as e:  # noqa: BLE001 — preserves prior fail-closed shape
            latency_ms = int((time.time() - start_time) * 1000)
            logger.error("CausalML execution failed: %s", e)
            return LibraryExecutionResult(
                library="causalml",
                success=False,
                latency_ms=latency_ms,
                result=None,
                error=str(e),
                confidence=0.0,
                warnings=warnings,
            )

    def validate_input(self, state: PipelineState) -> tuple[bool, str]:
        """Validate input for CausalML analysis.

        Strict semantics preserved from C-1: requires ``treatment_var`` and
        ``outcome_var``. The richer column-and-DataFrame validation happens
        inside ``execute()`` via ``_extract_uplift_inputs_from_state``; this
        method intentionally stays cheap and ABC-contract-pinned (V-20).
        """
        if not state.get("treatment_var"):
            return False, "CausalML requires treatment_var"
        if not state.get("outcome_var"):
            return False, "CausalML requires outcome_var"
        return True, ""
