"""
WS1 Model Performance KPI Calculators

Implements calculators for model performance metrics:
- ROC-AUC
- PR-AUC
- F1 Score
- Recall@Top-K
- Brier Score
- Calibration Slope Deviation
- SHAP Coverage
- Feature Drift (PSI)

Unavailability discipline (#439, F-007-PhaseB)
----------------------------------------------
This calculator does NOT return plausible-fake defaults (0.0 / 0.5 / 1.0 /
0.25 / 0.1) when MLflow is unreachable, the model is missing, the metric
is absent, or a query fails. Every `_calc_*` method propagates an
explicit `KPIResult(value=None, error="<reason>")` shape so that the
existing `_evaluate_status:120-121` fail-close primitive routes the
result through `KPIStatus.UNKNOWN`. Downstream consumers (dashboards,
alerts) can then distinguish "MLflow unreachable" from "model is random",
and "no rows in window" from "SHAP coverage is 0%".

Unavailability reasons:
  - mlflow_client_unavailable    (mlflow is ABSENT from this environment — the
    `import mlflow` itself raised ImportError. NOT "a client was passed as
    None": that argument only means "build one lazily".)
  - model_not_found:<name>       (registry returned no versions)
  - metric_not_found:<metric>    (run exists but metric key absent)
  - mlflow_exception:<Class>:<msg>  (any other MLflow-side failure, INCLUDING
    an unreachable, unroutable or silent tracking server — see the time bound
    below. A connection/timeout failure is not a new category: mlflow raises
    `MlflowException` and it lands on this existing reason, carrying the URL
    and "failed with exception" / "failed with timeout exception" in <msg>.
    Also carries CONSTRUCTION failures since #1658 — see below.)
  - db_query_failed:<reason>     (SQL execution raised)
  - db_query_returned_empty[:<note>]  (no rows / NULL value)

Construction is inside the taxonomy too (#1658)
-----------------------------------------------
The taxonomy above wraps CALLS. Until #1658 it did not wrap CONSTRUCTION: the
lazy `mlflow_client` property caught only `ImportError`, which is all the
`import mlflow` line can raise, so every way `MlflowClient()` itself can fail —
an unsupported or typo'd `MLFLOW_TRACKING_URI` scheme, a backing store that will
not open, a bad credential — escaped a plain attribute access. ABSENT was
handled and MISCONFIGURED was not, which is the one shape a working deployment
can actually regress into: prod ships `MLFLOW_TRACKING_URI=http://mlflow:5000`
and `htp://mlflow:5000` (one character) raises
`UnsupportedModelRegistryStoreURIException` at construction.

Those failures now fail closed to a `None` client, exactly like the absent case,
and report `mlflow_exception:<Class>:<msg>` — the module's existing catch-all for
"MLflow-side failure", not a new category. The two stay distinguishable because
absent keeps `mlflow_client_unavailable`. Nothing about no-fabrication changes:
the fix decides how a failure is NAMED, never that a value exists.

Time bound on the MLflow leg (#1650)
------------------------------------
Refusing to fabricate is only half the contract — the refusal also has to be
PROMPT. mlflow's REST store retries every request with urllib3 exponential
backoff, and its shipped defaults (`MLFLOW_HTTP_REQUEST_MAX_RETRIES=7`,
`BACKOFF_FACTOR=2`, `TIMEOUT=120`) are deliberately generous: mlflow's own
source comments that 7 retries "will take ~4 minutes", which is right for a
rate-limited backend and wrong for a KPI read a user's question is waiting on.
Measured pre-fix against a dead port: `_get_metric_from_mlflow` was still
running at 75s with no result. A fail-closed that takes minutes is a hang, not
a refusal — the same shape as the SHAP wedge in #1548.

`_bounded_mlflow_http` scopes those knobs to this read only, capping it at
`MLFLOW_LEG_WORST_CASE_SECONDS`. It is scoped rather than set globally because
the `src/agents/*/mlflow_tracker.py` WRITE paths legitimately want mlflow's
generous retry policy. The no-fabrication property is untouched: the bound
changes only how long "unavailable" takes to establish, never what is returned.
"""

import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Callable

import numpy as np

from src.kpi.calculator import KPICalculatorBase
from src.kpi.models import (
    KPIMetadata,
    KPIResult,
    KPIStatus,
    Workstream,
)
from src.kpi.synthetic_mode import resolve_kpi_query_id

# --------------------------------------------------------------------------- #
# #1650: time bound for the MLflow fail-closed leg.
#
# Per-attempt connect AND read timeout. The prod tracking server is
# `http://mlflow:5000` on the same compose network, so 3s is ~100x its expected
# latency; a false "unavailable" here is fail-closed (KPIStatus.UNKNOWN), never
# a wrong value.
MLFLOW_LEG_TIMEOUT_SECONDS = 3
# Retries AFTER the initial attempt. Combined with backoff_factor=0 below this
# is what kills the ~126s of urllib3 sleep per endpoint.
MLFLOW_LEG_MAX_RETRIES = 1
# Worst case for the whole leg: every attempt hangs for the full timeout.
# Measured 6.01s against both an unroutable address and a silent (accepts but
# never replies) peer; ~0.08s against a refused connection.
MLFLOW_LEG_WORST_CASE_SECONDS = (MLFLOW_LEG_MAX_RETRIES + 1) * MLFLOW_LEG_TIMEOUT_SECONDS

# mlflow reads all four of these from the environment AT CALL TIME
# (`mlflow/utils/rest_utils.py::http_request` — verified identical in the
# installed 3.11.1 and in the 3.15.1 that requirements.lock pins for CI), which
# is what makes an env scope a real seam rather than a hopeful one.
_MLFLOW_HTTP_BOUND = {
    "MLFLOW_HTTP_REQUEST_TIMEOUT": str(MLFLOW_LEG_TIMEOUT_SECONDS),
    "MLFLOW_HTTP_REQUEST_MAX_RETRIES": str(MLFLOW_LEG_MAX_RETRIES),
    "MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR": "0",
    "MLFLOW_HTTP_REQUEST_BACKOFF_JITTER": "0",
}

# `os.environ` is process-global and KPI reads can run concurrently (FastAPI
# runs sync endpoints in a threadpool). Depth-count so only the OUTERMOST scope
# restores; without it two interleaved readers can leave the bound values
# installed permanently, silently shortening the mlflow_tracker write paths.
# The lock guards only the counter, never the HTTP call, so concurrent KPI
# reads still overlap.
_MLFLOW_ENV_LOCK = threading.Lock()
_MLFLOW_ENV_SAVED: dict[str, str | None] = {}
_MLFLOW_ENV_DEPTH = 0


@contextmanager
def _bounded_mlflow_http() -> Iterator[None]:
    """Scope mlflow's HTTP retry/timeout knobs to one KPI read (#1650)."""
    global _MLFLOW_ENV_DEPTH
    with _MLFLOW_ENV_LOCK:
        if _MLFLOW_ENV_DEPTH == 0:
            _MLFLOW_ENV_SAVED.clear()
            _MLFLOW_ENV_SAVED.update({k: os.environ.get(k) for k in _MLFLOW_HTTP_BOUND})
            os.environ.update(_MLFLOW_HTTP_BOUND)
        _MLFLOW_ENV_DEPTH += 1
    try:
        yield
    finally:
        with _MLFLOW_ENV_LOCK:
            _MLFLOW_ENV_DEPTH -= 1
            if _MLFLOW_ENV_DEPTH == 0:
                for key, previous in _MLFLOW_ENV_SAVED.items():
                    if previous is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = previous
                _MLFLOW_ENV_SAVED.clear()


class ModelPerformanceCalculator(KPICalculatorBase):
    """Calculator for WS1 Model Performance KPIs."""

    def __init__(self, db_client: Any = None, mlflow_client: Any = None):
        """Initialize with database and MLflow clients.

        Args:
            db_client: Database client for executing queries.
            mlflow_client: MLflow client for retrieving model metrics.
        """
        self._db_client = db_client
        self._mlflow_client = mlflow_client
        # #1658: why the lazy client could not be built, already in this
        # module's `mlflow_exception:<Class>:<msg>` / `mlflow_client_unavailable`
        # vocabulary. `None` means "not attempted yet"; a non-None value is a
        # remembered failure and STOPS further construction attempts on this
        # instance (see the property below for why that matters).
        self._mlflow_client_error: str | None = None

    @property
    def db_client(self) -> Any:
        """Get database client, lazily initializing if needed."""
        if self._db_client is None:
            from src.repositories import get_supabase_client

            self._db_client = get_supabase_client()
        return self._db_client

    @property
    def mlflow_client(self) -> Any:
        """Get MLflow client, lazily initializing if needed. NEVER raises (#1658).

        Two distinct things can go wrong here, and before #1658 only the first
        was handled:

        - **ABSENT** — mlflow is not installed. `import mlflow` raises
          `ImportError`, the client stays `None`, and the leg reports
          `mlflow_client_unavailable`. Nothing an operator can do.
        - **MISCONFIGURED** — mlflow is installed but the client cannot be BUILT:
          an unsupported/typo'd `MLFLOW_TRACKING_URI` scheme, a backing store
          that will not open, a bad credential, a missing DB driver. The old
          `except ImportError` wrapped both statements but only the import can
          raise it, so these escaped a mere attribute access and bypassed the
          module's entire error taxonomy. Measured on mlflow 3.11.1:
          `htp://mlflow:5000` — one character off the value prod ships — raises
          `UnsupportedModelRegistryStoreURIException` right here.

        Note the two are told apart by WHICH STATEMENT failed, not by exception
        type, and that distinction is load-bearing: `MlflowClient()` can itself
        raise an `ImportError` subclass — measured, `mysql://…` raises
        `ModuleNotFoundError: No module named 'MySQLdb'` — and a missing DB
        driver is a config problem, not an absent mlflow. Collapsing the two
        `try`s into one `except ImportError` would file it under "no MLflow
        here" and send the operator looking in the wrong place.

        Both now fail CLOSED to `None`, with the reason recorded in
        `_mlflow_client_error` for `_get_metric_from_mlflow` to surface. This is
        classification, not repair: no value is ever invented.

        The failure is REMEMBERED for the life of the instance. Construction is
        not uniformly cheap — an unreachable DB-backed tracking URI costs ~102s
        per attempt (mlflow's `create_sqlalchemy_engine_with_retry` backs off
        over `MAX_RETRY_COUNT`, a module constant with no env knob, and
        `_bounded_mlflow_http` cannot reach it: those knobs are read only by the
        REST store). A WS1 model-performance grid asks this calculator for 7
        MLflow-backed KPIs, so retrying per access would turn one ~102s failure
        into ~12 minutes of "fail-closed" — the #1650 hang shape, re-entered
        through the door this fix opens. Remembering it is safe because the
        lifetime is one request: `get_kpi_calculator()` (src/api/routes/kpi.py)
        is a plain `Depends(...)` with no `lru_cache`, so a corrected config is
        picked up on the very next request without a restart.
        """
        if self._mlflow_client is None and self._mlflow_client_error is None:
            try:
                try:
                    import mlflow
                except ImportError:
                    self._mlflow_client_error = "mlflow_client_unavailable"
                else:
                    self._mlflow_client = mlflow.tracking.MlflowClient()
            except Exception as e:
                # Constructor failures (measured, exercised by the #1658 tests)
                # and — defensively, and NOT exercised by a real-config test
                # because only a corrupt install can produce it — a non-ImportError
                # failure raised while importing mlflow itself. Both are
                # environment problems rather than an absent mlflow, so both take
                # the `mlflow_exception:` family. This branch exists so the "NEVER
                # raises" promise above is true of the code and not just of the
                # cases we could reproduce.
                self._mlflow_client_error = f"mlflow_exception:{type(e).__name__}:{str(e)[:200]}"
        return self._mlflow_client

    def supports(self, kpi: KPIMetadata) -> bool:
        """Check if this calculator supports the given KPI."""
        return kpi.workstream == Workstream.WS1_MODEL_PERFORMANCE

    def calculate(self, kpi: KPIMetadata, context: dict[str, Any] | None = None) -> KPIResult:
        """Calculate a model performance KPI.

        Args:
            kpi: The KPI metadata defining what to calculate.
            context: Optional context with model_name, model_version, etc.

        Returns:
            KPIResult with calculated value and status. On unavailability,
            `value=None`, `error=<reason>`, `status=KPIStatus.UNKNOWN`.
        """
        context = context or {}

        # Route to specific calculator based on KPI ID. Each _calc_* returns
        # ``(value, error)``; a calculator MAY return a third element — a dict
        # merged into KPIResult.metadata (e.g. WS1-MP-006's per-model
        # calibration_slope_detail) so KPI-specific detail reaches the payload
        # without widening the shared 2-tuple contract for every metric.
        calculator_map: dict[str, Callable[[dict[str, Any]], tuple[Any, ...]]] = {
            "WS1-MP-001": self._calc_roc_auc,
            "WS1-MP-002": self._calc_pr_auc,
            "WS1-MP-003": self._calc_f1_score,
            "WS1-MP-004": self._calc_recall_at_k,
            "WS1-MP-005": self._calc_brier_score,
            "WS1-MP-006": self._calc_calibration_slope,
            "WS1-MP-007": self._calc_shap_coverage,
            "WS1-MP-009": self._calc_feature_drift,
        }

        calc_func = calculator_map.get(kpi.id)
        if calc_func is None:
            return KPIResult(  # type: ignore[call-arg]
                kpi_id=kpi.id,
                error=f"No calculator implemented for {kpi.id}",
            )

        try:
            outcome = calc_func(context)
            value, error = outcome[0], outcome[1]
            extra_metadata = outcome[2] if len(outcome) > 2 else None
            # Determine if lower is better
            lower_is_better = kpi.id in {"WS1-MP-005", "WS1-MP-009"}
            status = self._evaluate_status(kpi, value, lower_is_better)
            metadata: dict[str, Any] = {"context": context, "lower_is_better": lower_is_better}
            if extra_metadata:
                metadata.update(extra_metadata)
            return KPIResult(  # type: ignore[call-arg]
                kpi_id=kpi.id,
                value=value,
                status=status,
                error=error,
                metadata=metadata,
            )
        except Exception as e:
            return KPIResult(  # type: ignore[call-arg]
                kpi_id=kpi.id,
                error=str(e),
            )

    def _evaluate_status(
        self, kpi: KPIMetadata, value: float | None, lower_is_better: bool = False
    ) -> KPIStatus:
        """Evaluate KPI value against thresholds.

        The `value is None -> UNKNOWN` branch is the load-bearing fail-close
        primitive for #439. Every `_calc_*` method that hits unavailability
        returns `value=None` so that this method routes the result through
        `KPIStatus.UNKNOWN` instead of fabricating GOOD/WARNING/CRITICAL
        from a plausible default.
        """
        if value is None:
            return KPIStatus.UNKNOWN
        if kpi.threshold is None:
            # No threshold by design -> tracked for trend/context only.
            return KPIStatus.INFORMATIONAL
        return kpi.threshold.evaluate(value, lower_is_better=lower_is_better)

    # ------------------------------------------------------------------ gold-standard helper

    def _goldstd_summary(self, context: dict[str, Any]) -> dict[str, Any] | None:
        """Per-brand gold-standard holdout summary (best-effort, never raises).

        ``context['brand']`` scopes to that brand's ``*_goldstd_lr_v1`` staging
        models (absent/All -> all 12). Returns ``None`` when no gold-standard
        data is available or a read fails — never fabricates.
        """
        from src.kpi.goldstd_model_perf import summarize_sync

        try:
            summary = summarize_sync(self.db_client, context.get("brand"))
        except Exception:
            return None
        return summary or None

    def _goldstd_metric(self, context: dict[str, Any], metric_name: str) -> float | None:
        """Per-brand aggregate of the gold-standard models' holdout ``metric_name``.

        Best-effort PRIMARY source for the dashboard. Returns ``None`` (caller
        then falls back to the existing corpus/MLflow legs) when no
        gold-standard data is available or a read fails — never raises, never
        fabricates.
        """
        summary = self._goldstd_summary(context)
        if not summary:
            return None
        val = summary.get(metric_name)
        return float(val) if val is not None else None

    # ------------------------------------------------------------------ MLflow-backed metrics

    def _calc_roc_auc(self, context: dict[str, Any]) -> tuple[float | None, str | None]:
        """Calculate WS1-MP-001: ROC-AUC.

        PRIMARY (gold-standard): per-brand average of the gold-standard models'
        holdout ``auc_roc`` (brand-reactive; fixes the corpus-wide invariant
        value the dashboard previously showed for every brand).

        FALLBACK 1 (SQL): the real ``ml_predictions.model_auc`` corpus mean via
        the ``kpi_query`` allowlist (registry id ``model_performance_roc_auc``),
        the source declared by ``config/kpi_definitions.yaml`` for WS1-MP-001.

        FALLBACK 2 (MLflow): the preserved fail-closed leg. Never fabricates a
        plausible default at any leg.
        """
        gs = self._goldstd_metric(context, "auc_roc")
        if gs is not None:
            return gs, None
        result, db_error = self._execute_query("model_performance_roc_auc", [])
        if db_error is None and result:
            roc_auc = result[0].get("roc_auc")
            if roc_auc is not None:
                return float(roc_auc), None
        # SQL leg unavailable (error, empty, or NULL avg) -> MLflow fail-closed leg.
        model_name = context.get("model_name", "default_model")
        return self._get_metric_from_mlflow(model_name, "roc_auc")

    def _calc_pr_auc(self, context: dict[str, Any]) -> tuple[float | None, str | None]:
        """Calculate WS1-MP-002: PR-AUC.

        PRIMARY: per-brand average of the gold-standard models' holdout ``pr_auc``.
        FALLBACK: MLflow (fail-closed; no fabricated default).
        """
        gs = self._goldstd_metric(context, "pr_auc")
        if gs is not None:
            return gs, None
        model_name = context.get("model_name", "default_model")
        return self._get_metric_from_mlflow(model_name, "pr_auc")

    def _calc_f1_score(self, context: dict[str, Any]) -> tuple[float | None, str | None]:
        """Calculate WS1-MP-003: F1 Score.

        PRIMARY: per-brand average of the gold-standard models' holdout ``f1``.
        FALLBACK: MLflow (fail-closed; no fabricated default).
        """
        gs = self._goldstd_metric(context, "f1")
        if gs is not None:
            return gs, None
        model_name = context.get("model_name", "default_model")
        return self._get_metric_from_mlflow(model_name, "f1_score")

    def _calc_recall_at_k(self, context: dict[str, Any]) -> tuple[float | None, str | None]:
        """Calculate WS1-MP-004: Recall@Top-K."""
        model_name = context.get("model_name", "default_model")
        k = context.get("k", 100)
        metric_name = f"recall_at_{k}"
        return self._get_metric_from_mlflow(model_name, metric_name)

    def _calc_brier_score(self, context: dict[str, Any]) -> tuple[float | None, str | None]:
        """Calculate WS1-MP-005: Brier Score (lower is better).

        PRIMARY: per-brand average of the gold-standard models' holdout
        ``brier_score``. FALLBACK: MLflow (fail-closed; no fabricated default).
        """
        gs = self._goldstd_metric(context, "brier_score")
        if gs is not None:
            return gs, None
        model_name = context.get("model_name", "default_model")
        return self._get_metric_from_mlflow(model_name, "brier_score")

    def _calc_calibration_slope(
        self, context: dict[str, Any]
    ) -> tuple[float | None, str | None, dict[str, Any] | None]:
        """Calculate WS1-MP-006: Calibration Slope Deviation.

        PRIMARY: per-brand aggregate of the gold-standard models' holdout
        ``calibration_slope`` — ``1 + mean(|slope_i - 1|)``, computed by
        ``goldstd_model_perf.average_holdout`` (kills signed cancellation while
        staying in slope-band units, so the kpi_definitions band applies
        unchanged). The third tuple element carries the per-model detail
        (slope, holdout n, bootstrap CI) into ``KPIResult.metadata`` so a
        wide-CI red is visibly a small-sample red.
        FALLBACK: MLflow (fail-closed; no fabricated default).
        """
        summary = self._goldstd_summary(context)
        if summary and summary.get("calibration_slope") is not None:
            detail = summary.get("calibration_slope_detail")
            extra = {"calibration_slope_detail": detail} if detail else None
            return float(summary["calibration_slope"]), None, extra
        model_name = context.get("model_name", "default_model")
        value, error = self._get_metric_from_mlflow(model_name, "calibration_slope")
        return value, error, None

    # ------------------------------------------------------------------ SQL-backed / hybrid metrics

    def _calc_shap_coverage(self, context: dict[str, Any]) -> tuple[float | None, str | None]:
        """Calculate WS1-MP-007: SHAP Coverage.

        Percentage of predictions with SHAP explanations generated.
        Source is SQL only. Unavailability reasons:
          - db_query_failed: `_execute_query` returned None (exception raised
            during execution).
          - db_query_returned_empty: query returned no rows OR `coverage` was
            NULL (zero-denominator window).
        """
        result, db_error = self._execute_query("model_performance_shap_coverage", [])
        if db_error is not None:
            return None, f"db_query_failed:{db_error}"
        # `_execute_query` succeeded but may have returned [] or a row with
        # NULL coverage.
        if not result:
            return None, "db_query_returned_empty:no_rows_in_window"
        coverage = result[0].get("coverage")
        if coverage is None:
            return None, "db_query_returned_empty:null_coverage"
        return float(coverage), None

    def _calc_feature_drift(self, context: dict[str, Any]) -> tuple[float | None, str | None]:
        """Calculate WS1-MP-009: Feature Drift (PSI, lower is better).

        Two legs: SQL primary (drift monitoring table) + MLflow fallback.
        Returns the SQL value if it succeeds with a real (non-NULL) PSI.
        Otherwise consults MLflow. If both legs fail-close, returns
        `value=None` with a combined error.
        """
        model_name = context.get("model_name", "default_model")

        # #577 WS1-MP-009: the SQL leg is now REGISTERED + seeded (migration 053). It is a
        # CORPUS-level aggregate — `SELECT AVG(test_statistic) AS avg_psi FROM ml_drift_history
        # WHERE test_type='psi' AND drift_type='data'` — registered with max_params=0, so the
        # call binds NO params. We deliberately do NOT bind `model_name` here: the calculator's
        # model_name is a STRING but ml_drift_history keys on a UUID `model_id` (NULL in the
        # seed — ml_model_registry has 0 rows), so a filter would be a LABEL-not-functional
        # no-op falsely implying per-model scoping that does not exist (cf. the max_params=0
        # sibling model_performance_shap_coverage). A 1-element
        # `params` would make `kpi_query` RAISE "expects 0 param(s), got 1". A future PR that
        # seeds real ml_model_registry rows + per-model drift can promote this to arity 1.
        # `model_name` is still used by the MLflow fallback leg below. The two-leg fail-closed
        # contract is preserved: empty/unseeded table -> AVG over 0 rows is NULL -> MLflow ->
        # fail-closed UNKNOWN (never a fabricated PSI).
        sql_result, sql_error = self._execute_query("model_performance_feature_drift", [])

        # First leg: SQL succeeded with a real PSI.
        if sql_result:
            avg_psi = sql_result[0].get("avg_psi")
            if avg_psi is not None:
                return float(avg_psi), None
            sql_leg_error = "db_query_returned_empty:null_avg_psi"
        elif sql_error is not None:
            sql_leg_error = f"db_query_failed:{sql_error}"
        else:
            sql_leg_error = "db_query_returned_empty:no_rows_in_window"

        # Second leg: MLflow fallback. If it returns a real value, the SQL
        # leg's silence is acceptable (no rows yet in the table is not a
        # KPI failure as long as MLflow has the metric).
        mlflow_value, mlflow_error = self._get_metric_from_mlflow(model_name, "feature_drift_psi")
        if mlflow_value is not None:
            return mlflow_value, None

        # Both legs failed: surface combined unavailability honestly.
        combined_error = (
            f"feature_drift_psi unavailable: sql_leg={sql_leg_error}; mlflow_leg={mlflow_error}"
        )
        return None, combined_error

    # ------------------------------------------------------------------ infra

    def _get_metric_from_mlflow(
        self, model_name: str, metric_name: str
    ) -> tuple[float | None, str | None]:
        """Get a metric value from MLflow for the latest model version.

        Returns:
            A `(value, error)` tuple. Exactly one of the two is non-None:
              - `(<float>, None)` on success.
              - `(None, "mlflow_client_unavailable")` when no MLflow client
                is wired in (e.g., mlflow not installed in this environment).
              - `(None, "mlflow_exception:<ExceptionClass>:<msg[:200]>")` when
                mlflow IS installed but the client could not be CONSTRUCTED
                (#1658) — a misconfigured `MLFLOW_TRACKING_URI`, an unopenable
                backing store, a bad credential. Distinct from the line above on
                purpose: "no MLflow here" needs no action, "your MLflow config
                is wrong" needs an operator.
              - `(None, "model_not_found:<name>")` when the registry has no
                versions for the model.
              - `(None, "metric_not_found:<metric>")` when the latest run
                does not have the requested metric key.
              - `(None, "mlflow_exception:<ExceptionClass>:<msg[:200]>")`
                on any other MLflow-side failure — including an unreachable,
                unroutable or silent tracking server, which mlflow surfaces as
                `MlflowException`.

        This method NEVER returns a plausible-fake default like 0.5 or 1.0
        when the metric is unavailable — that was the #439 anti-pattern.

        It is also bounded in TIME (#1650): the whole leg runs inside
        `_bounded_mlflow_http`, so an unreachable tracking server resolves to
        an `mlflow_exception:` reason within `MLFLOW_LEG_WORST_CASE_SECONDS`
        instead of retrying for minutes. The scope covers the `mlflow_client`
        property too, but note what that does and does not buy (measured, #1658):
        an HTTP tracking URI touches NO network while constructing, so there is
        nothing there to bound; a DB-backed one does, via SQLAlchemy, which does
        not read `MLFLOW_HTTP_REQUEST_*` at all and so is NOT bounded by this
        scope. What keeps that case from compounding is the property's
        remembered failure, not this context manager.
        """
        with _bounded_mlflow_http():
            if self.mlflow_client is None:
                # #1658: prefer the recorded reason — absent
                # (`mlflow_client_unavailable`) and misconfigured
                # (`mlflow_exception:...`) are different problems needing
                # different fixes. The fallback covers a `mlflow_client` property
                # that was overridden to return None without going through the
                # lazy path (which the fail-closed tests do).
                return None, self._mlflow_client_error or "mlflow_client_unavailable"

            try:
                versions = self.mlflow_client.get_latest_versions(
                    model_name, stages=["Production", "Staging", "None"]
                )
                if not versions:
                    return None, f"model_not_found:{model_name}"

                run_id = versions[0].run_id
                run = self.mlflow_client.get_run(run_id)
                metrics = run.data.metrics
                if metric_name not in metrics:
                    return None, f"metric_not_found:{metric_name}"
                return float(metrics[metric_name]), None
            except Exception as e:
                msg = str(e)[:200]
                return None, f"mlflow_exception:{type(e).__name__}:{msg}"

    def _execute_query(
        self, query_id: str, params: list[Any]
    ) -> tuple[list[dict[str, Any]] | None, str | None]:
        """Run a vetted read-only KPI statement via the kpi_query allowlist RPC (#574).

        `query_id` indexes a statement in `kpi_query_registry`; `params` bind its
        $1..$N placeholders. Returns `(rows, error)` — exactly one is non-None:
          - `(<rows>, None)` on success (rows may be `[]`).
          - `(None, "<ExceptionClass>:<msg[:200]>")` on any execution failure.
        """
        try:
            # Demo/review: swap to the _include_synthetic twin under the
            # E2I_KPI_INCLUDE_SYNTHETIC flag (no-op otherwise). See synthetic_mode.py.
            query_id = resolve_kpi_query_id(query_id)
            response = self.db_client.rpc(
                "kpi_query", {"query_id": query_id, "params": params}
            ).execute()
            return response.data, None  # type: ignore[no-any-return]
        except Exception as e:
            msg = str(e)[:200]
            return None, f"{type(e).__name__}:{msg}"


def calculate_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Calculate Population Stability Index.

    Args:
        expected: Expected/reference distribution.
        actual: Actual/current distribution.
        bins: Number of bins for histogram.

    Returns:
        PSI value (0 = no drift, higher = more drift).
    """
    # Bin the distributions
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]

    # Convert to proportions with smoothing
    expected_pct = (expected_counts + 1) / (len(expected) + bins)
    actual_pct = (actual_counts + 1) / (len(actual) + bins)

    # Calculate PSI
    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)
