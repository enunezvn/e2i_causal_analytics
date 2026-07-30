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
- Trigger Funnel Conversion (#1360)

#1360 (2026-07-30 ruling): the four trigger-effectiveness KPIs — precision
(WS2-TR-001), acceptance rate (WS2-TR-004), override rate (WS2-TR-006) and
funnel conversion (WS2-TR-009) — are chat-KPI-path KPIs and additionally bind
``trigger_type`` and explicit ``window`` axes through the migration-118
``trigger_effectiveness_*`` statement families (see :meth:`_effectiveness_scoped`).
"""

from typing import Any

from src.kpi.calculator import KPICalculatorBase
from src.kpi.models import (
    KPIMetadata,
    KPIResult,
    KPIStatus,
    Workstream,
)
from src.kpi.synthetic_mode import (
    brand_region_query_id,
    brand_scoped_query_id,
    region_query_id,
    resolve_kpi_query_id,
    trigger_effectiveness_query_id,
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
            "WS2-TR-009": self._calc_funnel_conversion,
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
        if value is None:
            return KPIStatus.UNKNOWN
        if kpi.threshold is None:
            # No threshold by design -> tracked for trend/context only.
            return KPIStatus.INFORMATIONAL
        return kpi.threshold.evaluate(value, lower_is_better=lower_is_better)

    @staticmethod
    def _scoped(base_query_id: str, context: dict[str, Any]) -> tuple[str, list[Any]]:
        """Route to the brand/region-scoped query variant (migrations 078 + 113)
        from the request context, else the base query.

        The certified base queries take no params; scoped variants bind their
        scope(s) positionally — region as ``$1`` on ``_region`` (078), brand as
        ``$1`` on ``_brand`` and brand ``$1`` + region ``$2`` on
        ``_brand_region`` (113). Region joins triggers.patient_id ->
        patient_journeys.geographic_region (triggers carry no region column);
        brand filters triggers.brand_id directly. Returns ``(query_id,
        params)``; no scope yields the base query + ``[]`` (so the certified
        gates are byte-identical). The synthetic_mode helpers self-suffix
        ``_include_synthetic`` under the showcase flag.
        """
        brand = context.get("brand")
        region = context.get("region")
        if brand and region:
            return brand_region_query_id(base_query_id), [brand, region]
        if brand:
            return brand_scoped_query_id(base_query_id), [brand]
        if region:
            return region_query_id(base_query_id), [region]
        return base_query_id, []

    @staticmethod
    def _effectiveness_scoped(metric: str, context: dict[str, Any]) -> tuple[str, list[Any]]:
        """Route a trigger-effectiveness ask to the migration-118 family (#1360).

        Only called when the ask carries a NEW axis (``trigger_type`` and/or
        ``window``) that the legacy 044/078/113 variants cannot bind — the
        certified legacy routing in :meth:`_scoped` stays byte-identical
        otherwise. Param order is the migration's declared contract:

        * no window  -> ``trigger_effectiveness_{metric}``,
          ``[brand, region, trigger_type]`` — all nullable (NULL = no filter).
        * window     -> ``trigger_effectiveness_{metric}_windowed``,
          ``[brand, trigger_type, start, end]`` (half-open ``[start, end)`` on
          ``trigger_timestamp``). Region can NOT also bind — the kpi_query RPC
          caps at 4 positional params — so region+window FAILS CLOSED here
          rather than silently dropping the region (the dead-'territory'-key
          lesson: a filter the response echoes but the SQL never applied).
        """
        brand = context.get("brand")
        region = context.get("region")
        trigger_type = context.get("trigger_type")
        window = context.get("window")
        if window is not None:
            if region:
                raise RuntimeError(
                    "trigger-effectiveness KPIs cannot combine a region filter "
                    "with an explicit time window (the kpi_query RPC binds at "
                    "most 4 positional params — migration 118); drop the window "
                    "or the region filter"
                )
            return (
                trigger_effectiveness_query_id(metric, windowed=True),
                [brand, trigger_type, window["start"], window["end"]],
            )
        return (
            trigger_effectiveness_query_id(metric, windowed=False),
            [brand, region, trigger_type],
        )

    @staticmethod
    def _stash_data_through(context: dict[str, Any], result: list[dict[str, Any]] | None) -> None:
        """Surface the row's ``data_through`` provenance into the per-call context.

        Mirrors ``BusinessImpactCalculator._stash_data_through``: the
        frontier-anchored migration-118 rows report the as-of date their window
        ends at; ``calculate()`` embeds the context in ``KPIResult.metadata`` so
        the chatbot cites the real period instead of implying wall-clock
        recency. Rows without the column (``*_windowed*`` forms — the window is
        explicit) leave the key absent: honest absence, never a fabricated date.
        """
        if result and isinstance(result[0], dict) and result[0].get("data_through") is not None:
            context["data_through"] = result[0]["data_through"]

    @staticmethod
    def _wants_effectiveness(context: dict[str, Any]) -> bool:
        """True when the ask carries an axis only the 118 family can bind."""
        return bool(context.get("trigger_type") or context.get("window"))

    def _calc_trigger_precision(self, context: dict[str, Any]) -> float:
        """Calculate WS2-TR-001: Trigger Precision.

        Percentage of fired triggers resulting in positive outcome. With a
        ``trigger_type`` or explicit ``window`` in the ask, routes to the
        migration-118 effectiveness family (#1360); otherwise the certified
        legacy routing is untouched.
        """
        if self._wants_effectiveness(context):
            query_id, params = self._effectiveness_scoped("precision", context)
        else:
            query_id, params = self._scoped("trigger_performance_precision", context)
        result = self._execute_query(query_id, params)
        self._stash_data_through(context, result)
        if result and result[0].get("precision") is not None:
            return float(result[0]["precision"])
        raise RuntimeError("KPI WS2-TR-001 unavailable: no data for trigger precision")

    def _calc_trigger_recall(self, context: dict[str, Any]) -> float:
        """Calculate WS2-TR-002: Trigger Recall.

        Percentage of positive outcomes preceded by a trigger.
        """
        query_id, params = self._scoped("trigger_performance_recall", context)
        result = self._execute_query(query_id, params)
        if result and result[0].get("recall") is not None:
            return float(result[0]["recall"])
        raise RuntimeError("KPI WS2-TR-002 unavailable: no data for trigger recall")

    def _calc_action_rate_uplift(self, context: dict[str, Any]) -> float:
        """Calculate WS2-TR-003: Action Rate Uplift.

        Incremental (RELATIVE) action rate of the TREATMENT arm (NBA shown,
        control_group_flag=false) over the CONTROL arm (NBA withheld,
        control_group_flag=true):
            (action_rate_treatment - action_rate_control) / action_rate_control
        where each arm's action_rate = fraction of triggers with action_taken
        IS NOT NULL. "Action" is action_taken (a rep BEHAVIOR measurable in BOTH
        arms); acceptance_status is deliberately NOT used — it is treatment-only
        (you cannot accept a withheld NBA) and is already the outcome of
        WS2-TR-004/006 (see #577 migration 051).

        #577: the per-arm aggregation and the relative division are done in SQL;
        this returns the realized RELATIVE uplift as a bare float (a fraction —
        NOT a percentage, NOT an absolute difference). Fails loud when EITHER arm
        is empty (no row, or NULL uplift) — that was the #574 fail-loud reason
        (no control_group_flag column existed). A genuine 0.0 (both arms populated
        with equal action rates → no lift) is a legitimate value and is returned,
        not raised; a negative uplift (treatment worse than control) is likewise
        returned (it reads CRITICAL via the higher-is-better bands).
        """
        query_id, params = self._scoped("trigger_performance_action_rate_uplift", context)
        result = self._execute_query(query_id, params)
        if not result or result[0].get("action_rate_uplift") is None:
            raise RuntimeError(
                "KPI WS2-TR-003 action_rate_uplift unavailable: no populated "
                "treatment/control arm to contrast (apply the #577 control-arm "
                "migration 051)"
            )
        return float(result[0]["action_rate_uplift"])

    def _calc_acceptance_rate(self, context: dict[str, Any]) -> float:
        """Calculate WS2-TR-004: Acceptance Rate.

        Percentage of delivered triggers accepted by reps. With a
        ``trigger_type`` or explicit ``window``, routes to the migration-118
        effectiveness family (#1360).
        """
        if self._wants_effectiveness(context):
            query_id, params = self._effectiveness_scoped("acceptance_rate", context)
        else:
            query_id, params = self._scoped("trigger_performance_acceptance_rate", context)
        result = self._execute_query(query_id, params)
        self._stash_data_through(context, result)
        if result and result[0].get("acceptance_rate") is not None:
            return float(result[0]["acceptance_rate"])
        raise RuntimeError("KPI WS2-TR-004 unavailable: no data for acceptance rate")

    def _calc_false_alert_rate(self, context: dict[str, Any]) -> float:
        """Calculate WS2-TR-005: False Alert Rate.

        Percentage of triggers marked as false positives.
        Lower is better.
        """
        query_id, params = self._scoped("trigger_performance_false_alert_rate", context)
        result = self._execute_query(query_id, params)
        if result and result[0].get("false_alert_rate") is not None:
            return float(result[0]["false_alert_rate"])
        raise RuntimeError("KPI WS2-TR-005 unavailable: no data for false alert rate")

    def _calc_override_rate(self, context: dict[str, Any]) -> float:
        """Calculate WS2-TR-006: Override Rate.

        Percentage of triggers overridden by users. Lower is better. With a
        ``trigger_type`` or explicit ``window``, routes to the migration-118
        effectiveness family (#1360).
        """
        if self._wants_effectiveness(context):
            query_id, params = self._effectiveness_scoped("override_rate", context)
        else:
            query_id, params = self._scoped("trigger_performance_override_rate", context)
        result = self._execute_query(query_id, params)
        self._stash_data_through(context, result)
        if result and result[0].get("override_rate") is not None:
            return float(result[0]["override_rate"])
        raise RuntimeError("KPI WS2-TR-006 unavailable: no data for override rate")

    def _calc_lead_time(self, context: dict[str, Any]) -> float:
        """Calculate WS2-TR-007: Lead Time.

        Median days between trigger and outcome.
        Lower is better.
        """
        query_id, params = self._scoped("trigger_performance_lead_time", context)
        result = self._execute_query(query_id, params)
        if result and result[0].get("median_lead_time") is not None:
            return float(result[0]["median_lead_time"])
        raise RuntimeError("KPI WS2-TR-007 unavailable: no data for lead time")

    def _calc_change_fail_rate(self, context: dict[str, Any]) -> float:
        """Calculate WS2-TR-008: Change-Fail Rate (CFR).

        Percentage of trigger changes that resulted in worse outcomes.
        Lower is better.
        """
        query_id, params = self._scoped("trigger_performance_cfr", context)
        result = self._execute_query(query_id, params)
        if result and result[0].get("cfr") is not None:
            return float(result[0]["cfr"])
        raise RuntimeError("KPI WS2-TR-008 unavailable: no data for change-fail rate")

    def _calc_funnel_conversion(self, context: dict[str, Any]) -> float:
        """Calculate WS2-TR-009: Trigger Funnel Conversion (#1360, migration 118).

        Headline = actioned share of DELIVERED triggers (delivery_status IN
        ('delivered','viewed') -> accepted -> action_taken). The full stage
        counts (delivered -> viewed -> accepted -> actioned -> outcome) are
        stashed into ``context["funnel_stages"]`` so the chat layer can surface
        the whole funnel alongside the headline. The headline deliberately
        STOPS at actioned — extending it to outcome would conflate
        outcome-TRACKING coverage with effectiveness (the v1 precision trap,
        migration 113). ``n_viewed`` is informational: delivery_status is a
        progression state, so viewed is not monotone with accepted.
        """
        query_id, params = self._effectiveness_scoped("funnel_conversion", context)
        result = self._execute_query(query_id, params)
        self._stash_data_through(context, result)
        if result and result[0].get("funnel_conversion") is not None:
            row = result[0]
            context["funnel_stages"] = {
                "delivered": row.get("n_delivered"),
                "viewed": row.get("n_viewed"),
                "accepted": row.get("n_accepted"),
                "actioned": row.get("n_actioned"),
                "outcome": row.get("n_outcome"),
            }
            return float(row["funnel_conversion"])
        raise RuntimeError(
            "KPI WS2-TR-009 unavailable: no delivered triggers in the window for funnel conversion"
        )

    def _execute_query(self, query_id: str, params: list[Any]) -> list[dict[str, Any]] | None:
        """Execute a vetted KPI query via the kpi_query allowlist RPC.

        Runs the statement registered under ``query_id`` in
        ``kpi_query_registry``; ``params`` bind to ``$1..$N`` in that statement.
        """
        # #574: do NOT swallow RPC failures into None — callers convert None -> 0.0,
        # fabricating a zero KPI on a dead/misconfigured backend. Let exceptions propagate
        # to calculate(), which surfaces them as KPIResult(error=...). A successful query
        # with no rows still returns [] (a genuine empty, not an error).
        # Demo/review: swap to the _include_synthetic twin under the
        # E2I_KPI_INCLUDE_SYNTHETIC flag (no-op otherwise). See synthetic_mode.py.
        query_id = resolve_kpi_query_id(query_id)
        response = self.db_client.rpc(
            "kpi_query", {"query_id": query_id, "params": params}
        ).execute()
        return response.data  # type: ignore[no-any-return]
