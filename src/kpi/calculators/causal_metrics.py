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

    def calculate(
        self, kpi: KPIMetadata, context: dict[str, Any] | None = None
    ) -> KPIResult:
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
            return KPIResult(
                kpi_id=kpi.id,
                error=f"No calculator implemented for {kpi.id}",
            )

        try:
            result_data = calc_func(context)
            value = result_data.get("value")
            metadata = result_data.get("metadata", {})

            return KPIResult(
                kpi_id=kpi.id,
                value=value,
                status=KPIStatus.UNKNOWN,  # Causal metrics typically don't have thresholds
                metadata={**context, **metadata},
            )
        except Exception as e:
            return KPIResult(
                kpi_id=kpi.id,
                error=str(e),
            )

    def _calc_ate(self, context: dict[str, Any]) -> dict[str, Any]:
        """Calculate CM-001: Average Treatment Effect (ATE).

        E[Y(1) - Y(0)] - average effect of treatment on outcome.
        """
        # Try from stored predictions first
        query = """
            SELECT
                AVG(treatment_effect_estimate) as ate,
                STDDEV(treatment_effect_estimate) as ate_std,
                COUNT(*) as n_samples
            FROM ml_predictions
            WHERE treatment_effect_estimate IS NOT NULL
            AND prediction_timestamp >= NOW() - INTERVAL '30 days'
        """
        result = self._execute_query(query, [])
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
        segment_filter = "AND segment_assignment = $1" if segment else ""

        query = f"""
            SELECT
                segment_assignment,
                AVG(heterogeneous_effect) as cate,
                STDDEV(heterogeneous_effect) as cate_std,
                COUNT(*) as n_samples
            FROM ml_predictions
            WHERE heterogeneous_effect IS NOT NULL
            AND prediction_timestamp >= NOW() - INTERVAL '30 days'
            {segment_filter}
            GROUP BY segment_assignment
            ORDER BY AVG(heterogeneous_effect) DESC
        """
        params = [segment] if segment else []
        result = self._execute_query(query, params)

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

        Estimated causal effect size from causal paths table.
        """
        intervention = context.get("intervention")
        intervention_filter = "AND intervention_name = $1" if intervention else ""

        query = f"""
            SELECT
                intervention_name,
                AVG(causal_effect_size) as effect_size,
                AVG(confidence_level) as avg_confidence,
                COUNT(*) as n_paths
            FROM causal_paths
            WHERE validated = true
            {intervention_filter}
            GROUP BY intervention_name
            ORDER BY AVG(causal_effect_size) DESC
            LIMIT 10
        """
        params = [intervention] if intervention else []
        result = self._execute_query(query, params)

        if result and len(result) > 0:
            if intervention:
                row = result[0]
                return {
                    "value": row["effect_size"],
                    "metadata": {
                        "intervention": row["intervention_name"],
                        "confidence": row["avg_confidence"],
                        "n_paths": row["n_paths"],
                    },
                }
            else:
                # Return top interventions
                return {
                    "value": result[0]["effect_size"],
                    "metadata": {
                        "top_interventions": [
                            {
                                "intervention": r["intervention_name"],
                                "effect_size": r["effect_size"],
                                "confidence": r["avg_confidence"],
                            }
                            for r in result
                        ]
                    },
                }

        return {"value": None, "metadata": {"error": "No causal impact data available"}}

    def _calc_counterfactual(self, context: dict[str, Any]) -> dict[str, Any]:
        """Calculate CM-004: Counterfactual Outcome.

        E[Y(a') | do(A=a), X] - predicted outcome under alternative treatment.
        """
        patient_id = context.get("patient_id")
        if not patient_id:
            return {"value": None, "metadata": {"error": "patient_id required"}}

        query = """
            SELECT
                counterfactual_outcome,
                actual_outcome,
                treatment_received,
                counterfactual_treatment,
                (counterfactual_outcome - actual_outcome) as outcome_delta
            FROM ml_predictions
            WHERE patient_id = $1
            AND counterfactual_outcome IS NOT NULL
            ORDER BY prediction_timestamp DESC
            LIMIT 1
        """
        result = self._execute_query(query, [patient_id])

        if result and len(result) > 0:
            row = result[0]
            return {
                "value": row["counterfactual_outcome"],
                "metadata": {
                    "actual_outcome": row["actual_outcome"],
                    "treatment_received": row["treatment_received"],
                    "counterfactual_treatment": row["counterfactual_treatment"],
                    "outcome_delta": row["outcome_delta"],
                },
            }

        return {"value": None, "metadata": {"error": "No counterfactual data for patient"}}

    def _calc_mediation_effect(self, context: dict[str, Any]) -> dict[str, Any]:
        """Calculate CM-005: Mediation Effect.

        indirect_effect / total_effect - proportion mediated.
        """
        treatment = context.get("treatment")
        outcome = context.get("outcome")
        treatment_filter = "WHERE treatment = $1 AND outcome = $2" if treatment and outcome else ""

        query = f"""
            SELECT
                treatment,
                outcome,
                mediators_identified,
                pathway_details,
                indirect_effect,
                direct_effect,
                total_effect,
                indirect_effect / NULLIF(total_effect, 0) as proportion_mediated
            FROM causal_paths
            {treatment_filter if treatment_filter else "WHERE true"}
            AND total_effect IS NOT NULL
            ORDER BY validated DESC, created_at DESC
            LIMIT 5
        """
        params = [treatment, outcome] if treatment and outcome else []
        result = self._execute_query(query, params)

        if result and len(result) > 0:
            row = result[0]
            return {
                "value": row.get("proportion_mediated"),
                "metadata": {
                    "treatment": row["treatment"],
                    "outcome": row["outcome"],
                    "mediators": row.get("mediators_identified"),
                    "indirect_effect": row.get("indirect_effect"),
                    "direct_effect": row.get("direct_effect"),
                    "total_effect": row.get("total_effect"),
                    "pathway_details": row.get("pathway_details"),
                },
            }

        return {"value": None, "metadata": {"error": "No mediation data available"}}

    def _execute_query(
        self, query: str, params: list[Any]
    ) -> list[dict[str, Any]] | None:
        """Execute a SQL query and return results."""
        try:
            response = self.db_client.rpc("execute_sql", {"query": query}).execute()
            return response.data
        except Exception:
            return None
