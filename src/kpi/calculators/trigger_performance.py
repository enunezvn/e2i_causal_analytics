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
            from src.repositories import get_supabase_client

            self._db_client = get_supabase_client()
        return self._db_client

    def supports(self, kpi: KPIMetadata) -> bool:
        """Check if this calculator supports the given KPI."""
        return kpi.workstream == Workstream.WS2_TRIGGERS

    def calculate(self, kpi: KPIMetadata, context: dict[str, Any] | None = None) -> KPIResult:
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
                value=None,
                status=KPIStatus.UNKNOWN,
                cached=False,
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
                cached=False,
                error=None,
                metadata={"context": context, "lower_is_better": lower_is_better},
            )
        except Exception as e:
            return KPIResult(
                kpi_id=kpi.id,
                value=None,
                status=KPIStatus.UNKNOWN,
                cached=False,
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
        result = self._execute_query("trigger_performance_precision", [])
        if result and result[0].get("precision") is not None:
            return float(result[0]["precision"])
        return 0.0

    def _calc_trigger_recall(self, context: dict[str, Any]) -> float:
        """Calculate WS2-TR-002: Trigger Recall.

        Percentage of positive outcomes preceded by a trigger.
        """
        result = self._execute_query("trigger_performance_recall", [])
        if result and result[0].get("recall") is not None:
            return float(result[0]["recall"])
        return 0.0

    def _calc_action_rate_uplift(self, context: dict[str, Any]) -> float:
        """Calculate WS2-TR-003: Action Rate Uplift.

        Incremental action rate vs control group.
        """
        raise RuntimeError(
            "KPI WS2-TR-003 action_rate_uplift unavailable: triggers has no "
            "control_group_flag column to compute uplift vs control (#574)"
        )

    def _calc_acceptance_rate(self, context: dict[str, Any]) -> float:
        """Calculate WS2-TR-004: Acceptance Rate.

        Percentage of delivered triggers accepted by reps.
        """
        result = self._execute_query("trigger_performance_acceptance_rate", [])
        if result and result[0].get("acceptance_rate") is not None:
            return float(result[0]["acceptance_rate"])
        return 0.0

    def _calc_false_alert_rate(self, context: dict[str, Any]) -> float:
        """Calculate WS2-TR-005: False Alert Rate.

        Percentage of triggers marked as false positives.
        Lower is better.
        """
        result = self._execute_query("trigger_performance_false_alert_rate", [])
        if result and result[0].get("false_alert_rate") is not None:
            return float(result[0]["false_alert_rate"])
        return 0.0

    def _calc_override_rate(self, context: dict[str, Any]) -> float:
        """Calculate WS2-TR-006: Override Rate.

        Percentage of triggers overridden by users.
        Lower is better.
        """
        result = self._execute_query("trigger_performance_override_rate", [])
        if result and result[0].get("override_rate") is not None:
            return float(result[0]["override_rate"])
        return 0.0

    def _calc_lead_time(self, context: dict[str, Any]) -> float:
        """Calculate WS2-TR-007: Lead Time.

        Median days between trigger and outcome.
        Lower is better.
        """
        result = self._execute_query("trigger_performance_lead_time", [])
        if result and result[0].get("median_lead_time") is not None:
            return float(result[0]["median_lead_time"])
        return 0.0

    def _calc_change_fail_rate(self, context: dict[str, Any]) -> float:
        """Calculate WS2-TR-008: Change-Fail Rate (CFR).

        Percentage of trigger changes that resulted in worse outcomes.
        Lower is better.
        """
        result = self._execute_query("trigger_performance_cfr", [])
        if result and result[0].get("cfr") is not None:
            return float(result[0]["cfr"])
        return 0.0

    def _execute_query(self, query_id: str, params: list[Any]) -> list[dict[str, Any]] | None:
        """Execute a vetted KPI query via the kpi_query allowlist RPC.

        Runs the statement registered under ``query_id`` in
        ``kpi_query_registry``; ``params`` bind to ``$1..$N`` in that statement.
        """
        # #574: do NOT swallow RPC failures into None — callers convert None -> 0.0,
        # fabricating a zero KPI on a dead/misconfigured backend. Let exceptions propagate
        # to calculate(), which surfaces them as KPIResult(error=...). A successful query
        # with no rows still returns [] (a genuine empty, not an error).
        response = self.db_client.rpc(
            "kpi_query", {"query_id": query_id, "params": params}
        ).execute()
        return response.data  # type: ignore[no-any-return]
