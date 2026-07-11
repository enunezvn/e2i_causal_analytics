"""Health Checker Node.

This node checks the health status of active experiments including:
1. Enrollment rates and trends
2. Data quality and freshness
3. Overall experiment health status

Performance Target: <2s per experiment
"""

import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from src.agents.experiment_monitor.state import (
    EnrollmentIssue,
    ErrorDetails,
    ExperimentMonitorState,
    ExperimentSummary,
    StaleDataIssue,
)

# Bound the interactive sweep. The synthetic generator leaves 1000+ running
# experiments heavily DUPLICATED by name (e.g. "Kisqali - Predict prescribing"
# x252). Fetch a wide newest-first window so duplicates can be collapsed and we
# still surface up to _MAX_EXPERIMENTS DISTINCT experiment names. The wide fetch
# is a single lightweight select; only the deduped subset incurs the
# per-experiment assignment/SRM/freshness checks (an unbounded check-loop blew
# past the 30s client timeout — see _get_experiments).
_RAW_FETCH_LIMIT = 1000

# The platform's analyzed brand portfolio (brand_type values). An unscoped
# check-all-active sweep is restricted to these — see _get_experiments.
_PLATFORM_BRANDS = ["Remibrutinib", "Kisqali", "Fabhalta"]
_MAX_EXPERIMENTS = 25


class HealthCheckerNode:
    """Checks experiment health and enrollment rates.

    Health Check Strategy:
    1. Query active experiments from database
    2. Calculate enrollment rates and trends
    3. Identify experiments with health issues
    4. Update state with experiment summaries and issues

    Performance Target: <2s per experiment
    """

    def __init__(self):
        """Initialize health checker node."""
        self._client = None

    async def _get_client(self):
        """Lazy load Supabase client."""
        if self._client is None:
            from src.memory.services.factories import get_async_supabase_client

            self._client = await get_async_supabase_client()
        return self._client

    async def execute(self, state: ExperimentMonitorState) -> ExperimentMonitorState:
        """Execute health check on experiments.

        Args:
            state: Current agent state

        Returns:
            Updated state with experiment health information
        """
        start_time = time.time()

        try:
            state["status"] = "checking"

            # Get client
            client = await self._get_client()
            if not client:
                # Fail closed: NO mock data in a production path. A missing client
                # is recorded as an error (surfaced via the route's `errors`),
                # never fabricated into plausible experiments.
                state["errors"] = state.get("errors", []) + [
                    {
                        "node": "health_checker",
                        "error": "database client unavailable",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                ]
                experiments: List[Dict] = []
            else:
                experiments = await self._get_experiments(client, state)

            # Process each experiment
            experiment_summaries: List[ExperimentSummary] = []
            enrollment_issues: List[EnrollmentIssue] = []
            stale_data_issues: List[StaleDataIssue] = []

            from src.repositories.provenance import coerce_provenance_flag

            include_synthetic = coerce_provenance_flag(state.get("include_synthetic"))
            for exp in experiments:
                summary = await self._check_experiment_health(exp, client, include_synthetic)
                experiment_summaries.append(summary)

                # Check for enrollment issues (skip if disabled; FE selective-check flag, #825)
                if state.get("check_enrollment", True):
                    issue = self._check_enrollment_rate(exp, summary, state)
                    if issue:
                        enrollment_issues.append(issue)

                # Check for stale data
                stale_issue = await self._check_stale_data(exp, client, state)
                if stale_issue:
                    stale_data_issues.append(stale_issue)

            # Update state
            state["experiments"] = experiment_summaries
            state["enrollment_issues"] = enrollment_issues
            state["stale_data_issues"] = stale_data_issues
            state["experiments_checked"] = len(experiment_summaries)

            # Calculate latency
            latency_ms = int((time.time() - start_time) * 1000)
            state["check_latency_ms"] = latency_ms

        except Exception as e:
            error: ErrorDetails = {
                "node": "health_checker",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            state["errors"] = state.get("errors", []) + [error]
            state["experiments"] = []
            state["enrollment_issues"] = []
            state["stale_data_issues"] = []

        return state

    async def _get_experiments(self, client: Any, state: ExperimentMonitorState) -> List[Dict]:
        """Get experiments to check from database.

        Args:
            client: Supabase client
            state: Current state with filter criteria

        Returns:
            List of experiment dictionaries
        """
        from src.repositories.provenance import apply_provenance_filter, coerce_provenance_flag

        # Provenance (#894): ml_experiments is is_synthetic-tagged (migration
        # 069) and the synthetic generator leaves 360 perpetually-"running"
        # rows — without the predicate the whole AB sweep chain (assignments,
        # SRM, interim analyses) runs against planted experiments. Strictly
        # parsed state opt-in (validation runs), real-mode default-exclude.
        include_synthetic = coerce_provenance_flag(state.get("include_synthetic"))
        # Explainability columns (2026-07-11): brand/description/channel ride to
        # the UI card; brand also scopes the sweep and keys the roster interleave.
        _SELECT = (
            "id, experiment_name, status, prediction_target, created_at, "
            "is_synthetic, brand, description, intervention_channel, "
            "target_enrollment, planned_duration_days"
        )
        brand = state.get("brand")

        try:
            if state.get("check_all_active"):
                # Fetch a wide newest-first window, then collapse same-named
                # duplicates to the most-recent row (_dedupe_by_name) and cap the
                # result so per-experiment checks stay bounded. Without the dedup
                # the newest-N slice was dominated by duplicate-named synthetic
                # rows, surfacing many identical cards in the UI.
                # count="exact" rides the same request (PostgREST Content-Range)
                # so the UI can say "N running, 25 monitored" instead of showing
                # the capped roster size as if it were the portfolio size.
                # Genuine A/B experiments only (2026-07-11): rows without an
                # intervention_channel are pipeline-lineage records (scope
                # definitions, evals, deploys), not experiments — they carry
                # zero assignments and their status is registry bookkeeping.
                # Migration 102 closed the historical ones out of 'running';
                # this predicate keeps the sweep honest against any future
                # writer that inherits the DB-default status. Same predicate
                # as the insights endpoint (insights_strategic.py).
                query = (
                    client.table("ml_experiments")
                    .select(_SELECT, count="exact")
                    .eq("status", "running")
                    .not_.is_("intervention_channel", "null")
                    .order("created_at", desc=True)
                    .limit(_RAW_FETCH_LIMIT)
                )
                if brand:
                    query = query.eq("brand", brand)
                else:
                    # Unscoped = the platform brand portfolio ("All Brands"
                    # everywhere else in the app). ml_experiments also carries
                    # scope_definer scaffolding rows (brand NULL / 'competitor',
                    # e.g. "unknown - Simple outcome") with ZERO A/B assignments
                    # — ML-scoping artifacts, not A/B experiments; they only
                    # added unexplainable stale-alert noise to the A/B page
                    # (2026-07-11 review).
                    query = query.in_("brand", _PLATFORM_BRANDS)
                result = await apply_provenance_filter(query, include_synthetic).execute()
                deduped = self._dedupe_by_name(result.data or [], cap=_RAW_FETCH_LIMIT)
                # isinstance guard: unit-test fakes leave .count as a Mock/None;
                # fall back to the fetched-window size rather than fabricating.
                total = getattr(result, "count", None)
                state["total_running"] = total if isinstance(total, int) else len(deduped)
                # Brand-balanced roster (2026-07-11): newest-first alone let one
                # generation batch monopolize the capped slice (the live top-25
                # was 25 same-batch Fabhalta rows). Interleave across brands so
                # an unscoped sweep represents the whole portfolio.
                return self._interleave_by_brand(deduped, cap=_MAX_EXPERIMENTS)
            elif state.get("experiment_ids"):
                # Get specific experiments (a synthetic id must not resolve in
                # real mode either — same semantics as BaseRepository.get_by_id)
                query = (
                    client.table("ml_experiments")
                    .select(_SELECT)
                    .in_("id", state["experiment_ids"])
                )
                result = await apply_provenance_filter(query, include_synthetic).execute()
            else:
                return []

            return result.data if result.data else []

        except Exception:
            return []

    @staticmethod
    def _dedupe_by_name(rows: List[Dict], cap: int) -> List[Dict]:
        """Collapse same-named running experiments to the most-recent row.

        The synthetic generator leaves many perpetually-"running" rows that share
        an ``experiment_name``. Rows arrive ``created_at``-desc, so the first
        occurrence per name is the most recent; later duplicates are dropped.
        Returns at most ``cap`` rows so the per-experiment checks stay bounded.

        Args:
            rows: Experiment rows, newest first.
            cap: Maximum number of distinct-named rows to return.

        Returns:
            Up to ``cap`` rows, one per experiment_name (most recent kept).
        """
        seen: set = set()
        deduped: List[Dict] = []
        for row in rows:
            # Distinct rows share names; fall back to id so an unnamed row is
            # never silently dropped as a "duplicate" of another unnamed row.
            key = row.get("experiment_name") or row.get("id")
            if key in seen:
                continue
            seen.add(key)
            deduped.append(row)
            if len(deduped) >= cap:
                break
        return deduped

    @staticmethod
    def _interleave_by_brand(rows: List[Dict], cap: int) -> List[Dict]:
        """Round-robin the newest-first roster across brands, up to ``cap``.

        Newest-first alone is batch-biased: the synthetic substrate is written
        in per-brand bursts, so the newest N rows are typically ONE brand from
        the latest burst (live incident 2026-07-11: the top-25 slice was 25
        same-instant Fabhalta rows). Grouping by brand (preserving each group's
        newest-first order, groups ordered by their newest row; rows without a
        brand form their own group) and taking one per group per round keeps
        the capped roster representative of the whole running portfolio.

        Args:
            rows: Deduped experiment rows, newest first.
            cap: Maximum number of rows to return.

        Returns:
            Up to ``cap`` rows, brands interleaved, per-brand order preserved.
        """
        groups: Dict[str, List[Dict]] = {}
        order: List[str] = []
        for row in rows:
            key = str(row.get("brand") or "__none__")
            if key not in groups:
                groups[key] = []
                order.append(key)  # first appearance = group's newest row rank
            groups[key].append(row)
        interleaved: List[Dict] = []
        round_idx = 0
        while len(interleaved) < cap:
            emitted = False
            for key in order:
                group = groups[key]
                if round_idx < len(group):
                    interleaved.append(group[round_idx])
                    emitted = True
                    if len(interleaved) >= cap:
                        break
            if not emitted:
                break
            round_idx += 1
        return interleaved

    async def _check_experiment_health(
        self, experiment: Dict, client: Optional[Any], include_synthetic: bool = False
    ) -> ExperimentSummary:
        """Check health of a single experiment.

        Args:
            experiment: Experiment dictionary
            client: Optional Supabase client

        Returns:
            ExperimentSummary with health status
        """
        exp_id = experiment["id"]
        created_at = experiment.get("created_at", datetime.now(timezone.utc).isoformat())

        # Calculate days running
        if isinstance(created_at, str):
            start_date = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        else:
            start_date = created_at
        days_running = max(1, (datetime.now(timezone.utc) - start_date).days)

        # Get enrollment data (#894: ab_experiment_assignments is tagged —
        # synthetic units must not count as enrollment)
        total_enrolled = 0
        if client:
            from src.repositories.provenance import apply_provenance_filter

            try:
                query = (
                    client.table("ab_experiment_assignments")
                    .select("id", count="exact")
                    .eq("experiment_id", exp_id)
                )
                result = await apply_provenance_filter(query, include_synthetic).execute()
                total_enrolled = result.count or 0
            except Exception:
                pass

        # Calculate metrics. The enrollment PLAN (migration 101) is per-row and
        # nullable — a row without a recorded plan gets NO information fraction
        # and NO plan-relative health judgement. (The previous code read
        # config.get("target_sample_size", 1000) from a `config` field that was
        # never selected and doesn't exist as a column: every experiment was
        # measured against a fabricated 1000-in-30-days plan, flagging the
        # whole portfolio "warning" — live incident 2026-07-11.)
        enrollment_rate = total_enrolled / days_running if days_running > 0 else 0
        target_enrollment = experiment.get("target_enrollment")
        planned_duration_days = experiment.get("planned_duration_days")
        information_fraction: Optional[float] = None
        if isinstance(target_enrollment, int) and target_enrollment > 0:
            # Cap at 1.0: past-target enrollment means "target reached", and
            # the interim milestone ladder (25/50/75/100%) expects [0, 1].
            information_fraction = min(1.0, total_enrolled / target_enrollment)

        # Determine health status
        health_status, health_reason = self._determine_health_status(
            enrollment_rate=enrollment_rate,
            days_running=days_running,
            total_enrolled=total_enrolled,
            target_enrollment=target_enrollment,
            planned_duration_days=planned_duration_days,
        )

        return ExperimentSummary(
            experiment_id=exp_id,
            # The live ml_experiments table uses `experiment_name` (not `name`);
            # the previous select referenced a non-existent `name` column.
            name=experiment.get("experiment_name", experiment.get("name", "Unknown")),
            status=experiment.get("status", "unknown"),
            health_status=health_status,
            health_reason=health_reason,
            days_running=days_running,
            total_enrolled=total_enrolled,
            enrollment_rate=round(enrollment_rate, 2),
            current_information_fraction=(
                round(information_fraction, 4) if information_fraction is not None else None
            ),
            target_enrollment=target_enrollment,
            is_synthetic=bool(experiment.get("is_synthetic", False)),
            # Explainability (2026-07-11): what the experiment tests and why —
            # None (not fabricated text) when the row predates the metadata.
            brand=experiment.get("brand"),
            description=experiment.get("description"),
            intervention_channel=experiment.get("intervention_channel"),
        )

    def _determine_health_status(
        self,
        enrollment_rate: float,
        days_running: int,
        total_enrolled: int,
        target_enrollment: Optional[int],
        planned_duration_days: Optional[int],
    ) -> tuple[Literal["healthy", "warning", "critical", "unknown"], str]:
        """Determine overall health status with a human-readable reason.

        Two tiers of checks:
        1. Absolute enrollment-pace checks — need no plan, always run.
        2. Plan-relative checks — ONLY when the row carries a real enrollment
           plan (migration 101). A missing plan is reported as such, never
           substituted with a default target (the fabricated 1000-in-30-days
           default previously flagged the entire portfolio "warning").

        The reason string is user-facing: it rides to the UI as the hover
        explanation on the health flag.

        Returns:
            (health status, reason) tuple.
        """
        # Critical: enrollment effectively stalled after significant time
        if days_running >= 14 and enrollment_rate < 2:
            return "critical", (
                f"Enrollment stalled: {enrollment_rate:.1f}/day over {days_running} days "
                f"(below 2/day after 14+ days)"
            )

        # Warning: enrollment pace below the platform floor
        if days_running >= 7 and enrollment_rate < 5:
            return "warning", (
                f"Slow enrollment: {enrollment_rate:.1f}/day over {days_running} days "
                f"(below 5/day after 7+ days)"
            )

        # Plan-relative checks (real recorded plan only)
        if target_enrollment and planned_duration_days:
            if total_enrolled >= target_enrollment:
                return "healthy", (
                    f"Enrollment target reached "
                    f"({total_enrolled:,} of {target_enrollment:,} enrolled)"
                )
            enrolled_pct = 100.0 * total_enrolled / target_enrollment
            if days_running > planned_duration_days:
                return "warning", (
                    f"Past planned duration (day {days_running} of {planned_duration_days}) "
                    f"at {enrolled_pct:.0f}% of the {target_enrollment:,} enrollment target"
                )
            expected_pct = 100.0 * days_running / planned_duration_days
            if enrolled_pct < expected_pct * 0.5:
                return "warning", (
                    f"Far behind plan: {enrolled_pct:.0f}% enrolled vs at least "
                    f"{expected_pct * 0.5:.0f}% expected by day {days_running} "
                    f"of {planned_duration_days}"
                )
            return "healthy", (
                f"On pace: {enrollment_rate:.1f}/day, {enrolled_pct:.0f}% of the "
                f"{target_enrollment:,} target by day {days_running} of {planned_duration_days}"
            )

        return "healthy", (
            f"Enrollment rate healthy ({enrollment_rate:.1f}/day); "
            f"no enrollment plan recorded to measure progress against"
        )

    def _check_enrollment_rate(
        self,
        experiment: Dict,
        summary: ExperimentSummary,
        state: ExperimentMonitorState,
    ) -> Optional[EnrollmentIssue]:
        """Check if enrollment rate is below threshold.

        Args:
            experiment: Experiment dictionary
            summary: Experiment summary
            state: Current state with thresholds

        Returns:
            EnrollmentIssue if rate is below threshold, None otherwise
        """
        threshold = state.get("enrollment_threshold", 5.0)

        if summary["enrollment_rate"] < threshold:
            # Calculate severity based on how long below threshold
            days = summary["days_running"]
            if days >= 14:
                severity = "critical"
            elif days >= 7:
                severity = "warning"
            else:
                severity = "info"

            return EnrollmentIssue(
                experiment_id=summary["experiment_id"],
                current_rate=summary["enrollment_rate"],
                expected_rate=threshold,
                days_below_threshold=days,
                severity=severity,  # type: ignore
            )

        return None

    async def _check_stale_data(
        self,
        experiment: Dict,
        client: Optional[Any],
        state: ExperimentMonitorState,
    ) -> Optional[StaleDataIssue]:
        """Check if experiment data is stale.

        Args:
            experiment: Experiment dictionary
            client: Optional Supabase client
            state: Current state with thresholds

        Returns:
            StaleDataIssue if data is stale, None otherwise
        """
        threshold_hours = state.get("stale_data_threshold_hours", 24.0)
        exp_id = experiment["id"]

        if not client:
            return None

        from src.repositories.provenance import apply_provenance_filter, coerce_provenance_flag

        include_synthetic = coerce_provenance_flag(state.get("include_synthetic"))

        try:
            # Get the most recent assignment timestamp for this experiment
            # (#894: a synthetic assignment must not mask real-data staleness)
            query = (
                client.table("ab_experiment_assignments")
                .select("assigned_at")
                .eq("experiment_id", exp_id)
            )
            result = await (
                apply_provenance_filter(query, include_synthetic)
                .order("assigned_at", desc=True)
                .limit(1)
                .execute()
            )

            if not result.data:
                # No assignments yet - might be stale or just new
                # Check experiment created_at to determine
                created_at = experiment.get("created_at")
                if created_at:
                    if isinstance(created_at, str):
                        created_time = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                    else:
                        created_time = created_at

                    hours_since_creation = (
                        datetime.now(timezone.utc) - created_time
                    ).total_seconds() / 3600

                    # If experiment is older than threshold and no data, it's stale.
                    # SAME ladder as the with-assignments branch below: the API
                    # documents one threshold-relative contract (breach = info,
                    # 1.5x = warning, 3x = critical) regardless of whether
                    # staleness is measured from the last assignment or, for a
                    # never-enrolled experiment, from creation.
                    if hours_since_creation > threshold_hours:
                        if hours_since_creation > 3 * threshold_hours:
                            severity = "critical"
                        elif hours_since_creation > 1.5 * threshold_hours:
                            severity = "warning"
                        else:
                            severity = "info"
                        return StaleDataIssue(
                            experiment_id=exp_id,
                            last_data_timestamp="N/A - No assignments",
                            hours_since_update=hours_since_creation,
                            threshold_hours=threshold_hours,
                            severity=severity,  # type: ignore
                        )
                return None

            # Get the last assignment timestamp
            last_timestamp_str = result.data[0]["assigned_at"]
            if isinstance(last_timestamp_str, str):
                last_timestamp = datetime.fromisoformat(last_timestamp_str.replace("Z", "+00:00"))
            else:
                last_timestamp = last_timestamp_str

            # Calculate hours since last update
            hours_since_update = (
                datetime.now(timezone.utc) - last_timestamp
            ).total_seconds() / 3600

            if hours_since_update > threshold_hours:
                # Severity RELATIVE to the caller's threshold (2026-07-11): the
                # old absolute tiers (48h/72h) pinned every experiment critical
                # once the substrate's refresh cadence was slower than 3 days —
                # e.g. the weekly synthetic refresh — regardless of the threshold
                # the caller chose. Breach = info, 1.5x = warning, 3x = critical.
                if hours_since_update > 3 * threshold_hours:
                    severity = "critical"
                elif hours_since_update > 1.5 * threshold_hours:
                    severity = "warning"
                else:
                    severity = "info"

                return StaleDataIssue(
                    experiment_id=exp_id,
                    last_data_timestamp=last_timestamp.isoformat(),
                    hours_since_update=round(hours_since_update, 2),
                    threshold_hours=threshold_hours,
                    severity=severity,  # type: ignore
                )

        except Exception:
            # Don't fail the whole check if stale data detection fails
            pass

        return None
