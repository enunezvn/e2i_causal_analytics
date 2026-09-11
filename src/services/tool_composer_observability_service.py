"""Aggregation for GET /admin/observability/tool-composer (spec §8).

Reads ``composer_episodes`` and ``composition_steps`` — what the learning loop recorded — and
presents them next to the per-tool verdicts the caller fetched from the SAME reader the planner
uses, so the word on the page and the word in the planning prompt cannot diverge.

Sync methods by design: the route runs them via ``asyncio.to_thread``, like its admin.py siblings.
"""

import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_PAGE = 1000
_IN_CHUNK = 100  # keep the .in_() URL length bounded, as llm_observability_service does
_RECENT_FAILURES = 10
_QUERY_PREVIEW_CHARS = 100

#: Longer than the recorder's heartbeat interval: an unfinished episode silent for this long is
#: not slow, it is abandoned — its worker went away mid-composition.
_ABANDONED_AFTER = timedelta(minutes=5)

#: Every status the composition_status enum treats as finished (ml/013). TIMEOUT belongs here:
#: a timed-out episode is over, not still running and not abandoned.
_TERMINAL = ("COMPLETED", "FAILED", "TIMEOUT")

_EPISODE_COLUMNS = (
    "episode_id, composition_id, query_text, status, outcome, failed_phase, error_type, "
    "plan_source, entry_point, total_latency_ms, last_activity_at, tools_executed, "
    "tools_succeeded, is_synthetic"
)
_STEP_COLUMNS = "episode_id, step_number, tool_name, outcome_class"


def _percentile(values: List[float], fraction: float) -> Optional[float]:
    """Continuous percentile, matching Postgres ``percentile_cont``; ``None`` when empty.

    The per-tool percentiles beside these come from ``get_tool_reliability``, which uses
    ``percentile_cont``. A nearest-rank number here would read as the same statistic while
    being a different one — on [100, 1000] it would say 100 where the database says 550.
    """
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


class ToolComposerObservabilityService:
    def __init__(self, client: Optional[Any] = None) -> None:
        if client is None:
            from src.api.dependencies.supabase_client import get_supabase

            client = get_supabase()
        if client is None:
            # Raise rather than cache a dead client, so the route's singleton getter self-heals
            # on the next request (AdminUserService / LLMObservabilityService semantics).
            raise RuntimeError("Supabase client unavailable for tool-composer observability")
        self.client = client

    # ------------------------------------------------------------- fetch ----

    def _fetch_episodes(self, since_iso: str, include_synthetic: bool) -> List[Dict[str, Any]]:
        """Episodes STARTED in the window, newest first.

        Membership is ``created_at``: a long composition belongs to the window it began in, and
        a heartbeat on an old episode must not pull it back into a recent one. Paging orders by
        ``episode_id`` as well, because ``created_at`` is neither unique nor, for the abandoned
        reading, the only timestamp that moves — without a unique tiebreaker a row can cross a
        page boundary and be counted twice or skipped.
        """
        rows: List[Dict[str, Any]] = []
        start = 0
        while True:
            query = (
                self.client.table("composer_episodes")
                .select(_EPISODE_COLUMNS)
                .gte("created_at", since_iso)
            )
            if not include_synthetic:
                # The tool rows beside these exclude synthetic runs; one response, one population.
                query = query.eq("is_synthetic", False)
            page = (
                query.order("created_at", desc=True)
                .order("episode_id")
                .range(start, start + _PAGE - 1)
                .execute()
            )
            batch = list(page.data or [])
            rows.extend(batch)
            if len(batch) < _PAGE:
                return rows
            start += _PAGE

    def _fetch_steps(self, episode_ids: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        by_episode: Dict[str, List[Dict[str, Any]]] = {}
        for chunk_start in range(0, len(episode_ids), _IN_CHUNK):
            chunk = episode_ids[chunk_start : chunk_start + _IN_CHUNK]
            if not chunk:
                continue
            start = 0
            while True:
                page = (
                    self.client.table("composition_steps")
                    .select(_STEP_COLUMNS)
                    .in_("episode_id", chunk)
                    .order("episode_id")
                    .order("step_number")
                    .range(start, start + _PAGE - 1)
                    .execute()
                )
                batch = list(page.data or [])
                for row in batch:
                    by_episode.setdefault(str(row.get("episode_id")), []).append(row)
                if len(batch) < _PAGE:
                    break
                start += _PAGE
        for steps in by_episode.values():
            steps.sort(key=lambda s: s.get("step_number") or 0)
        return by_episode

    # --------------------------------------------------------- aggregate ----

    def _compositions(self, episodes: List[Dict[str, Any]], now: datetime) -> Dict[str, Any]:
        counts = {"success": 0, "partial": 0, "failed": 0, "cancelled": 0}
        by_plan_source: Dict[str, int] = {}
        unfinished = 0
        abandoned = 0
        latencies: List[float] = []

        for episode in episodes:
            outcome = episode.get("outcome")
            if outcome in counts:
                counts[outcome] += 1
            if episode.get("status") not in _TERMINAL:
                unfinished += 1
                if self._is_abandoned(episode, now):
                    abandoned += 1
            source = episode.get("plan_source")
            if source:
                by_plan_source[source] = by_plan_source.get(source, 0) + 1
            latency = episode.get("total_latency_ms")
            if isinstance(latency, (int, float)):
                latencies.append(float(latency))

        return {
            "total": len(episodes),
            **counts,
            "unfinished": unfinished,
            "abandoned": abandoned,
            "by_plan_source": by_plan_source,
            "p50_latency_ms": _percentile(latencies, 0.50),
            "p95_latency_ms": _percentile(latencies, 0.95),
        }

    @staticmethod
    def _is_abandoned(episode: Dict[str, Any], now: datetime) -> bool:
        stamp = episode.get("last_activity_at")
        if not isinstance(stamp, str):
            return False
        try:
            seen = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
        except ValueError:
            return False
        if seen.tzinfo is None:
            seen = seen.replace(tzinfo=timezone.utc)
        return now - seen > _ABANDONED_AFTER

    @staticmethod
    def _tools(verdicts: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        rows = []
        for tool in (verdicts or {}).values():
            rows.append(
                {
                    "tool_name": tool.tool_name,
                    "verdict": tool.verdict,
                    "category": tool.category,
                    "source_agent": tool.source_agent,
                    "n_invoked": tool.n_invoked,
                    "n_succeeded": tool.n_succeeded,
                    "n_refused": tool.n_refused,
                    "n_health_failures": tool.n_health_failures,
                    "n_health": tool.n_health,
                    "n_retried": tool.n_retried,
                    "n_synthetic": tool.n_synthetic,
                    "p50_latency_ms": tool.p50_latency_ms,
                    "p95_latency_ms": tool.p95_latency_ms,
                    "declared_latency_ms": tool.declared_latency_ms,
                    "most_common_health_error": tool.most_common_health_error,
                    "last_executed_at": tool.last_executed_at,
                }
            )
        # Worst news first: a caveat is what an admin opened this page to see.
        order = {"caveat": 0, "inconclusive": 1, "reliable": 2, "too_few_runs": 3, "no_runs": 4}
        rows.sort(key=lambda r: (order.get(r["verdict"], 9), r["tool_name"]))
        return rows

    def _recent_failures(self, episodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        failed = [e for e in episodes if e.get("outcome") in ("failed", "partial", "cancelled")]
        failed = failed[:_RECENT_FAILURES]
        steps_by_episode = self._fetch_steps([str(e.get("episode_id")) for e in failed])

        rows = []
        for episode in failed:
            steps = steps_by_episode.get(str(episode.get("episode_id")), [])
            rows.append(
                {
                    "composition_id": episode.get("composition_id") or "",
                    "outcome": episode.get("outcome"),
                    "status": episode.get("status"),
                    "failed_phase": episode.get("failed_phase"),
                    "error_type": episode.get("error_type"),
                    "entry_point": episode.get("entry_point"),
                    "plan_source": episode.get("plan_source"),
                    "query_preview": str(episode.get("query_text") or "")[:_QUERY_PREVIEW_CHARS],
                    "step_classes": [
                        {
                            "step_number": s.get("step_number"),
                            "tool_name": s.get("tool_name"),
                            "outcome_class": s.get("outcome_class"),
                        }
                        for s in steps
                        if s.get("outcome_class") not in ("succeeded", "cache_hit")
                    ],
                    "last_activity_at": episode.get("last_activity_at"),
                    "total_latency_ms": episode.get("total_latency_ms"),
                }
            )
        return rows

    # -------------------------------------------------------------- read ----

    def overview(self, days: int, verdicts: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """The window's compositions, the per-tool verdicts, and what recently went wrong."""
        from src.repositories.provenance import deployment_includes_synthetic

        now = datetime.now(timezone.utc)
        since_iso = (now - timedelta(days=days)).isoformat()
        include_synthetic = deployment_includes_synthetic()
        episodes = self._fetch_episodes(since_iso, include_synthetic)

        return {
            "window_days": days,
            "include_synthetic": include_synthetic,
            "compositions": self._compositions(episodes, now),
            "tools": self._tools(verdicts),
            "recent_failures": self._recent_failures(episodes),
        }
