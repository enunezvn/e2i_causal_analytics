"""
WS2 Trigger Performance KPI Calculators

Implements calculators for trigger performance metrics:
- Trigger Precision
- Trigger Recall
- Action Rate Uplift
- Acceptance Rate
- False Alert Rate
- Override Rate
- Lead Time
- Change-Fail Rate (CFR)
"""

from typing import Any

from src.kpi.calculator import KPICalculatorBase
from src.kpi.models import (
    KPIMetadata,
    KPIResult,
    KPIStatus,
    Workstream,
)


class TriggerPerformanceCalculator(KPICalculatorBase):
    """Calculator for WS2 Trigger Performance KPIs."""

    def __init__(self, db_client: Any = None):
        """Initialize with database client.

        Args:
            db_client: Database client for executing queries.
        """
        self._db_client = db_client

    @property
    def db_client(self) -> Any:
        """Get database client, lazily initializing if needed."""
        if self._db_client is None:
            from src.repositories.base import get_supabase_client

            self._db_client = get_supabase_client()
        return self._db_client

    def supports(self, kpi: KPIMetadata) -> bool:
        """Check if this calculator supports the given KPI."""
        return kpi.workstream == Workstream.WS2_TRIGGERS

    def calculate(
        self, kpi: KPIMetadata, context: dict[str, Any] | None = None
    ) -> KPIResult:
        """Calculate a trigger performance KPI.

        Args:
            kpi: The KPI metadata defining what to calculate.
            context: Optional context with brand, date_range, etc.

        Returns:
            KPIResult with calculated value and status.
        """
        context = context or {}

        calculator_map = {
            "WS2-TR-001": self._calc_trigger_precision,
            "WS2-TR-002": self._calc_trigger_recall,
            "WS2-TR-003": self._calc_action_rate_uplift,
            "WS2-TR-004": self._calc_acceptance_rate,
            "WS2-TR-005": self._calc_false_alert_rate,
            "WS2-TR-006": self._calc_override_rate,
            "WS2-TR-007": self._calc_lead_time,
            "WS2-TR-008": self._calc_change_fail_rate,
        }

        calc_func = calculator_map.get(kpi.id)
        if calc_func is None:
            return KPIResult(
                kpi_id=kpi.id,
                error=f"No calculator implemented for {kpi.id}",
            )

        try:
            value = calc_func(context)
            # Lead time and CFR are lower-is-better metrics
            lower_is_better = kpi.id in {"WS2-TR-005", "WS2-TR-006", "WS2-TR-007", "WS2-TR-008"}
            status = self._evaluate_status(kpi, value, lower_is_better)
            return KPIResult(
                kpi_id=kpi.id,
                value=value,
                status=status,
                metadata={"context": context, "lower_is_better": lower_is_better},
            )
        except Exception as e:
            return KPIResult(
                kpi_id=kpi.id,
                error=str(e),
            )

    def _evaluate_status(
        self, kpi: KPIMetadata, value: float | None, lower_is_better: bool = False
    ) -> KPIStatus:
        """Evaluate KPI value against thresholds."""
        if value is None or kpi.threshold is None:
            return KPIStatus.UNKNOWN
        return kpi.threshold.evaluate(value, lower_is_better=lower_is_better)

    def _calc_trigger_precision(self, context: dict[str, Any]) -> float:
        """Calculate WS2-TR-001: Trigger Precision.

        Percentage of fired triggers resulting in positive outcome.
        """
        query = """
            SELECT
                COUNT(CASE WHEN outcome_tracked AND outcome_value > 0 THEN 1 END)::float /
                NULLIF(COUNT(CASE WHEN outcome_tracked THEN 1 END), 0) as precision
            FROM triggers
            WHERE fired_at >= NOW() - INTERVAL '30 days'
        """
        result = self._execute_query(query, [])
        if result and result[0].get("precision") is not None:
            return result[0]["precision"]
        return 0.0

    def _calc_trigger_recall(self, context: dict[str, Any]) -> float:
        """Calculate WS2-TR-002: Trigger Recall.

        Percentage of positive outcomes preceded by a trigger.
        """
        query = """
            WITH positive_outcomes AS (
                SELECT DISTINCT patient_id
                FROM treatment_events
                WHERE event_type IN ('prescription', 'conversion')
                AND event_date >= NOW() - INTERVAL '30 days'
            ),
            trigger_preceded AS (
                SELECT DISTINCT po.patient_id
                FROM positive_outcomes po
                INNER JOIN triggers t ON t.patient_id = po.patient_id
                WHERE t.fired_at < (
                    SELECT MIN(event_date) FROM treatment_events te
                    WHERE te.patient_id = po.patient_id
                    AND te.event_type IN ('prescription', 'conversion')
                )
            )
            SELECT
                COUNT(DISTINCT tp.patient_id)::float /
                NULLIF(COUNT(DISTINCT po.patient_id), 0) as recall
            FROM positive_outcomes po
            LEFT JOIN trigger_preceded tp ON tp.patient_id = po.patient_id
        """
        result = self._execute_query(query, [])
        if result and result[0].get("recall") is not None:
            return result[0]["recall"]
        return 0.0

    def _calc_action_rate_uplift(self, context: dict[str, Any]) -> float:
        """Calculate WS2-TR-003: Action Rate Uplift.

        Incremental action rate vs control group.
        """
        query = """
            WITH rates AS (
                SELECT
                    control_group_flag,
                    COUNT(CASE WHEN action_taken THEN 1 END)::float /
                    NULLIF(COUNT(*), 0) as action_rate
                FROM triggers
                WHERE fired_at >= NOW() - INTERVAL '30 days'
                GROUP BY control_group_flag
            )
            SELECT
                COALESCE(
                    (
                        (SELECT action_rate FROM rates WHERE NOT control_group_flag) -
                        (SELECT action_rate FROM rates WHERE control_group_flag)
                    ) /
                    NULLIF((SELECT action_rate FROM rates WHERE control_group_flag), 0),
                    0
                ) as uplift
        """
        result = self._execute_query(query, [])
        if result and result[0].get("uplift") is not None:
            return result[0]["uplift"]
        return 0.0

    def _calc_acceptance_rate(self, context: dict[str, Any]) -> float:
        """Calculate WS2-TR-004: Acceptance Rate.

        Percentage of delivered triggers accepted by reps.
        """
        query = """
            SELECT
                COUNT(CASE WHEN acceptance_status = 'accepted' THEN 1 END)::float /
                NULLIF(COUNT(CASE WHEN acceptance_status IS NOT NULL THEN 1 END), 0)
                as acceptance_rate
            FROM triggers
            WHERE fired_at >= NOW() - INTERVAL '30 days'
        """
        result = self._execute_query(query, [])
        if result and result[0].get("acceptance_rate") is not None:
            return result[0]["acceptance_rate"]
        return 0.0

    def _calc_false_alert_rate(self, context: dict[str, Any]) -> float:
        """Calculate WS2-TR-005: False Alert Rate.

        Percentage of triggers marked as false positives.
        Lower is better.
        """
        query = """
            SELECT
                COUNT(CASE WHEN false_positive_flag THEN 1 END)::float /
                NULLIF(COUNT(*), 0) as false_alert_rate
            FROM triggers
            WHERE fired_at >= NOW() - INTERVAL '30 days'
        """
        result = self._execute_query(query, [])
        if result and result[0].get("false_alert_rate") is not None:
            return result[0]["false_alert_rate"]
        return 0.0

    def _calc_override_rate(self, context: dict[str, Any]) -> float:
        """Calculate WS2-TR-006: Override Rate.

        Percentage of triggers overridden by users.
        Lower is better.
        """
        query = """
            SELECT
                COUNT(CASE WHEN acceptance_status = 'overridden' THEN 1 END)::float /
                NULLIF(COUNT(CASE WHEN acceptance_status IS NOT NULL THEN 1 END), 0)
                as override_rate
            FROM triggers
            WHERE fired_at >= NOW() - INTERVAL '30 days'
        """
        result = self._execute_query(query, [])
        if result and result[0].get("override_rate") is not None:
            return result[0]["override_rate"]
        return 0.0

    def _calc_lead_time(self, context: dict[str, Any]) -> float:
        """Calculate WS2-TR-007: Lead Time.

        Median days between trigger and outcome.
        Lower is better.
        """
        query = """
            SELECT
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY lead_time_days)
                as median_lead_time
            FROM triggers
            WHERE lead_time_days IS NOT NULL
            AND fired_at >= NOW() - INTERVAL '30 days'
        """
        result = self._execute_query(query, [])
        if result and result[0].get("median_lead_time") is not None:
            return float(result[0]["median_lead_time"])
        return 0.0

    def _calc_change_fail_rate(self, context: dict[str, Any]) -> float:
        """Calculate WS2-TR-008: Change-Fail Rate (CFR).

        Percentage of trigger changes that resulted in worse outcomes.
        Uses v_kpi_change_fail_rate view if available.
        Lower is better.
        """
        # Try view first
        query = """
            SELECT avg_cfr as cfr
            FROM v_kpi_change_fail_rate
            WHERE calculated_at >= NOW() - INTERVAL '7 days'
            ORDER BY calculated_at DESC
            LIMIT 1
        """
        result = self._execute_query(query, [])
        if result and result[0].get("cfr") is not None:
            return result[0]["cfr"]

        # Fall back to direct calculation
        query = """
            SELECT
                COUNT(CASE WHEN change_failed THEN 1 END)::float /
                NULLIF(COUNT(CASE WHEN previous_trigger_id IS NOT NULL THEN 1 END), 0)
                as cfr
            FROM triggers
            WHERE fired_at >= NOW() - INTERVAL '30 days'
        """
        result = self._execute_query(query, [])
        if result and result[0].get("cfr") is not None:
            return result[0]["cfr"]
        return 0.0

    def _execute_query(
        self, query: str, params: list[Any]
    ) -> list[dict[str, Any]] | None:
        """Execute a SQL query and return results."""
        try:
            response = self.db_client.rpc("execute_sql", {"query": query}).execute()
            return response.data
        except Exception:
            return None
