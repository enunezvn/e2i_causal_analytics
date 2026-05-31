"""
Causal Metrics KPI Calculators

Implements calculators for causal inference metrics:
- Average Treatment Effect (ATE)
- Conditional ATE (CATE)
- Causal Impact
- Counterfactual Outcome
- Mediation Effect
"""

from typing import Any

import numpy as np

from src.kpi.calculator import KPICalculatorBase
from src.kpi.models import (
    KPIMetadata,
    KPIResult,
    KPIStatus,
    Workstream,
)


class CausalMetricsCalculator(KPICalculatorBase):
    """Calculator for Causal Metrics KPIs."""

    def __init__(self, db_client: Any = None, causal_engine: Any = None):
        """Initialize with database and causal engine clients.

        Args:
            db_client: Database client for executing queries.
            causal_engine: Optional causal inference engine.
        """
        self._db_client = db_client
        self._causal_engine = causal_engine

    @property
    def db_client(self) -> Any:
        """Get database client, lazily initializing if needed."""
        if self._db_client is None:
            from src.repositories import get_supabase_client

            self._db_client = get_supabase_client()
        return self._db_client

    @property
    def causal_engine(self) -> Any:
        """Get causal engine, lazily initializing if needed."""
        if self._causal_engine is None:
            try:
                from src.causal_engine.energy_score.estimator_selector import (
                    EstimatorSelector,
                )

                self._causal_engine = EstimatorSelector()
            except ImportError:
                pass
        return self._causal_engine

    def supports(self, kpi: KPIMetadata) -> bool:
        """Check if this calculator supports the given KPI."""
        return kpi.workstream == Workstream.CAUSAL_METRICS

    def calculate(self, kpi: KPIMetadata, context: dict[str, Any] | None = None) -> KPIResult:
        """Calculate a causal metrics KPI.

        Args:
            kpi: The KPI metadata defining what to calculate.
            context: Optional context with treatment, outcome, covariates.

        Returns:
            KPIResult with calculated value and status.
        """
        context = context or {}

        calculator_map = {
            "CM-001": self._calc_ate,
            "CM-002": self._calc_cate,
            "CM-003": self._calc_causal_impact,
            "CM-004": self._calc_counterfactual,
            "CM-005": self._calc_mediation_effect,
        }

        calc_func = calculator_map.get(kpi.id)
        if calc_func is None:
            return KPIResult(  # type: ignore[call-arg]
                kpi_id=kpi.id,
                error=f"No calculator implemented for {kpi.id}",
            )

        try:
            result_data = calc_func(context)
            value = result_data.get("value")
            metadata = result_data.get("metadata", {})

            return KPIResult(  # type: ignore[call-arg]
                kpi_id=kpi.id,
                value=value,
                status=KPIStatus.UNKNOWN,  # Causal metrics typically don't have thresholds
                metadata={**context, **metadata},
            )
        except Exception as e:
            return KPIResult(  # type: ignore[call-arg]
                kpi_id=kpi.id,
                error=str(e),
            )

    def _calc_ate(self, context: dict[str, Any]) -> dict[str, Any]:
        """Calculate CM-001: Average Treatment Effect (ATE).

        E[Y(1) - Y(0)] - average effect of treatment on outcome.
        """
        # Try from stored predictions first
        result = self._execute_query("causal_metrics_ate", [])
        if result and result[0].get("ate") is not None:
            ate = result[0]["ate"]
            ate_std = result[0].get("ate_std", 0.0)
            n = result[0].get("n_samples", 0)

            # Calculate confidence interval
            se = ate_std / np.sqrt(n) if n > 0 else 0.0
            ci_lower = ate - 1.96 * se
            ci_upper = ate + 1.96 * se

            return {
                "value": ate,
                "metadata": {
                    "ate_std": ate_std,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "n_samples": n,
                    "source": "ml_predictions",
                },
            }

        # Fall back to causal engine if data provided
        treatment = context.get("treatment")
        outcome = context.get("outcome")
        covariates = context.get("covariates")

        if treatment is not None and outcome is not None and covariates is not None:
            try:
                result = self.causal_engine.estimate_effect(
                    treatment=treatment,
                    outcome=outcome,
                    covariates=covariates,
                )
                if result.success:
                    return {
                        "value": result.ate,
                        "metadata": {
                            "ate_std": result.ate_std,
                            "ci_lower": result.ate_ci_lower,
                            "ci_upper": result.ate_ci_upper,
                            "estimator": str(result.estimator_type),
                            "source": "causal_engine",
                        },
                    }
            except Exception:
                pass

        return {"value": None, "metadata": {"error": "No data available"}}

    def _calc_cate(self, context: dict[str, Any]) -> dict[str, Any]:
        """Calculate CM-002: Conditional ATE (CATE).

        E[Y(1) - Y(0) | X=x] - treatment effect by segment.
        """
        segment = context.get("segment")
        result = self._execute_query("causal_metrics_cate", [segment])

        if result and len(result) > 0:
            if segment:
                # Return CATE for specific segment
                row = result[0]
                return {
                    "value": row["cate"],
                    "metadata": {
                        "segment": row["segment_assignment"],
                        "cate_std": row["cate_std"],
                        "n_samples": row["n_samples"],
                    },
                }
            else:
                # Return overall and segment breakdown
                overall_cate = np.mean([r["cate"] for r in result])
                return {
                    "value": overall_cate,
                    "metadata": {
                        "segment_breakdown": [
                            {
                                "segment": r["segment_assignment"],
                                "cate": r["cate"],
                                "n_samples": r["n_samples"],
                            }
                            for r in result
                        ]
                    },
                }

        return {"value": None, "metadata": {"error": "No CATE data available"}}

    def _calc_causal_impact(self, context: dict[str, Any]) -> dict[str, Any]:
        """Calculate CM-003: Causal Impact.

        The average strength of the discovered causal effects in causal_paths:
        the path-level mean of ``causal_effect_size``. This is a DESCRIPTIVE
        aggregate over discovered pathways — NOT the effect of intervening on a
        variable. ``start_node`` is the discovered path SOURCE (where a chain
        begins), not a do()-style intervention target; it is surfaced only as a
        descriptive breakdown in metadata (#574/#577). An optional
        ``validation_status`` context filter ('' = all paths; e.g. 'validated' =
        audited only) narrows the cohort.
        """
        validation_status = context.get("validation_status", "") or ""
        rows = self._execute_query("causal_metrics_causal_impact", [validation_status])

        if rows:
            total_n = sum((r.get("n_paths") or 0) for r in rows)
            if total_n > 0:
                # Path-level (across-paths) mean: SUM(effect_i * n_i) / SUM(n_i).
                value = (
                    sum((r.get("effect") or 0.0) * (r.get("n_paths") or 0) for r in rows) / total_n
                )
                return {
                    "value": value,
                    "metadata": {
                        "n_paths": total_n,
                        "validation_status": validation_status or "all",
                        "breakdown": [
                            {
                                "start_node": r.get("start_node"),
                                "effect": r.get("effect"),
                                "n_paths": r.get("n_paths"),
                                "avg_confidence": r.get("avg_confidence"),
                            }
                            for r in rows
                        ],
                        "note": (
                            "mean causal_effect_size across discovered causal paths; "
                            "start_node is the discovered path source, NOT an intervention "
                            "target (#574)"
                        ),
                        "source": "causal_paths",
                    },
                }

        return {"value": None, "metadata": {"error": "No causal_paths data available"}}

    def _calc_counterfactual(self, context: dict[str, Any]) -> dict[str, Any]:
        """Calculate CM-004: Counterfactual Outcome.

        E[Y(a') | do(A=a), X] — the expected predicted outcome under the
        alternative arm. ``counterfactual_outcome`` is a coherent do-contrast of
        the factual ``prediction_value``: the factual minus the (additive)
        ``treatment_effect_estimate``, floored at 0 (an outcome cannot be
        negative). So the per-row contrast factual − counterfactual equals the
        treatment effect on UNCLAMPED rows, and is floor-attenuated (smaller, =
        prediction_value) where the effect exceeds the factual (#577). The VALUE
        is the counterfactual LEVEL (mean counterfactual_outcome) — distinct from
        CM-001 ATE (the contrast). An optional ``prediction_type`` context filter
        ('' = all types) narrows the cohort.
        """
        prediction_type = context.get("prediction_type", "") or ""
        rows = self._execute_query("causal_metrics_counterfactual", [prediction_type])

        if rows and rows[0].get("mean_counterfactual") is not None:
            row = rows[0]
            return {
                "value": row["mean_counterfactual"],
                "metadata": {
                    "mean_factual": row.get("mean_factual"),
                    # the TRUE realized contrast (factual − counterfactual); floor-attenuated
                    "mean_realized_contrast": row.get("mean_realized_contrast"),
                    # the NOMINAL mean treatment effect estimate (>= realized contrast)
                    "mean_effect": row.get("mean_effect"),
                    "n": row.get("n"),
                    "prediction_type": prediction_type or "all",
                    "note": (
                        "counterfactual outcome level E[Y(a')] = mean(counterfactual_outcome), "
                        "where counterfactual = max(0, prediction_value − treatment_effect_estimate). "
                        "mean_realized_contrast (= factual − counterfactual) equals the treatment "
                        "effect on unclamped rows and is floor-attenuated (<= mean_effect) on the "
                        "rows where the effect exceeds the factual (#577)"
                    ),
                    "source": "ml_predictions",
                },
            }

        return {"value": None, "metadata": {"error": "No counterfactual data available"}}

    def _calc_mediation_effect(self, context: dict[str, Any]) -> dict[str, Any]:
        """Calculate CM-005: Mediation Effect.

        Proportion mediated = mean(indirect_effect / causal_effect_size) over the
        discovered causal paths. The decomposition is coherent: total =
        causal_effect_size; indirect_effect is the serial-mediation effect through
        the identified mediators, grounded in the PRODUCT of the causal_chain edge
        magnitudes (the textbook serial path coefficient); direct_effect = total −
        indirect is a SYNTHESIZED residual allocation for the direct X→Y channel
        (which is NOT an observed edge in causal_chain), so direct + indirect =
        total exactly. Paths with no mediators contribute a proportion of 0 (no
        mediation channel) (#577).
        """
        rows = self._execute_query("causal_metrics_mediation", [])

        if rows and rows[0].get("proportion_mediated") is not None:
            row = rows[0]
            return {
                "value": row["proportion_mediated"],
                "metadata": {
                    "n_paths": row.get("n_paths"),
                    "mean_indirect": row.get("mean_indirect"),
                    "mean_direct": row.get("mean_direct"),
                    "note": (
                        "proportion mediated = mean(indirect_effect / causal_effect_size) "
                        "over discovered paths; indirect_effect is the serial-mediation effect "
                        "through the identified mediators (grounded in the product of the "
                        "causal_chain edge magnitudes); direct_effect = total − indirect is a "
                        "SYNTHESIZED residual allocation for the direct X→Y channel (not an "
                        "observed edge in causal_chain); paths with no mediators contribute 0 (#577)"
                    ),
                    "source": "causal_paths",
                },
            }

        return {"value": None, "metadata": {"error": "No mediation data available"}}

    def _execute_query(self, query_id: str, params: list[Any]) -> list[dict[str, Any]] | None:
        """Run a vetted statement from kpi_query_registry by id.

        The statement identified by ``query_id`` is looked up in the
        kpi_query_registry and executed by the ``kpi_query`` RPC; ``params``
        bind to its positional placeholders ($1..$N).
        """
        # #574: do NOT swallow RPC failures into None — callers convert None -> 0.0,
        # fabricating a zero KPI on a dead/misconfigured backend. Let exceptions propagate
        # to calculate(), which surfaces them as KPIResult(error=...). A successful query
        # with no rows still returns [] (a genuine empty, not an error).
        response = self.db_client.rpc(
            "kpi_query", {"query_id": query_id, "params": params}
        ).execute()
        return response.data  # type: ignore[no-any-return]
