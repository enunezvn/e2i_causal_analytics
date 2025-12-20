"""Metrics Aggregator Node - Compute quality metrics from spans."""

from typing import Dict, Any, List
from collections import defaultdict
from datetime import datetime, timedelta


async def aggregate_metrics(state: Dict[str, Any]) -> Dict[str, Any]:
    """Aggregate quality metrics from observability spans.

    Args:
        state: Current agent state with time_window and filters

    Returns:
        State updates with aggregated metrics
    """
    try:
        time_window = state.get("time_window", "24h")
        agent_filter = state.get("agent_name_filter")
        trace_filter = state.get("trace_id_filter")

        # In production, query ml_observability_spans table
        # spans = await db.query(
        #     "ml_observability_spans",
        #     filters={
        #         "start_time": {">=": cutoff_time},
        #         "agent_name": agent_filter if agent_filter else None,
        #         "trace_id": trace_filter if trace_filter else None,
        #     }
        # )

        # For now, simulate with mock data
        spans = _get_mock_spans(time_window)

        # Compute latency by agent
        latency_by_agent = _compute_latency_stats(spans, group_by="agent_name")

        # Compute latency by tier
        latency_by_tier = _compute_latency_stats(spans, group_by="agent_tier")

        # Compute error rates by agent
        error_rate_by_agent = _compute_error_rates(spans, group_by="agent_name")

        # Compute error rates by tier
        error_rate_by_tier = _compute_error_rates(spans, group_by="agent_tier")

        # Compute token usage by agent (for Hybrid/Deep agents)
        token_usage_by_agent = _compute_token_usage(spans)

        # Compute overall metrics
        total_spans = len(spans)
        error_spans = sum(1 for s in spans if s.get("status") == "error")
        timeout_spans = sum(1 for s in spans if s.get("status") == "timeout")
        ok_spans = total_spans - error_spans - timeout_spans

        overall_success_rate = ok_spans / total_spans if total_spans > 0 else 1.0

        # Compute percentiles across all spans
        all_durations = [s.get("duration_ms", 0) for s in spans if s.get("duration_ms")]
        all_durations_sorted = sorted(all_durations)

        if all_durations_sorted:
            overall_p95 = all_durations_sorted[int(len(all_durations_sorted) * 0.95)]
            overall_p99 = all_durations_sorted[int(len(all_durations_sorted) * 0.99)]
        else:
            overall_p95 = 0.0
            overall_p99 = 0.0

        # Fallback invocation rate
        fallback_spans = sum(
            1 for s in spans if s.get("metadata", {}).get("fallback_used", False)
        )
        fallback_rate = fallback_spans / total_spans if total_spans > 0 else 0.0

        # Status distribution
        status_distribution = {
            "ok": ok_spans,
            "error": error_spans,
            "timeout": timeout_spans,
        }

        # Compute quality score (weighted combination)
        quality_score = _compute_quality_score(
            success_rate=overall_success_rate,
            p95_latency_ms=overall_p95,
            fallback_rate=fallback_rate,
        )

        return {
            "quality_metrics_computed": True,
            "latency_by_agent": latency_by_agent,
            "latency_by_tier": latency_by_tier,
            "error_rate_by_agent": error_rate_by_agent,
            "error_rate_by_tier": error_rate_by_tier,
            "token_usage_by_agent": token_usage_by_agent,
            "overall_success_rate": overall_success_rate,
            "overall_p95_latency_ms": overall_p95,
            "overall_p99_latency_ms": overall_p99,
            "total_spans_analyzed": total_spans,
            "quality_score": quality_score,
            "fallback_invocation_rate": fallback_rate,
            "status_distribution": status_distribution,
        }

    except Exception as e:
        return {
            "error": f"Metrics aggregation failed: {str(e)}",
            "error_type": "metrics_aggregation_error",
            "error_details": {"exception": str(e)},
            "quality_metrics_computed": False,
        }


def _get_mock_spans(time_window: str) -> List[Dict[str, Any]]:
    """Get mock span data for simulation.

    Args:
        time_window: Time window string

    Returns:
        List of mock span dicts
    """
    # Simulate 1000 spans across different agents
    agents = [
        ("scope_definer", 0),
        ("data_preparer", 0),
        ("model_selector", 0),
        ("model_trainer", 0),
        ("feature_analyzer", 0),
        ("model_deployer", 0),
    ]

    spans = []
    for i in range(1000):
        agent_name, tier = agents[i % len(agents)]

        # Simulate different latencies and error rates per agent
        if agent_name == "scope_definer":
            duration = 2000 + (i % 500)
            status = "error" if i % 50 == 0 else "ok"
        elif agent_name == "data_preparer":
            duration = 5000 + (i % 3000)
            status = "error" if i % 30 == 0 else "ok"
        elif agent_name == "model_selector":
            duration = 8000 + (i % 5000)
            status = "error" if i % 40 == 0 else "ok"
        elif agent_name == "model_trainer":
            duration = 60000 + (i % 20000)
            status = "error" if i % 100 == 0 else "ok"
        elif agent_name == "feature_analyzer":
            duration = 15000 + (i % 10000)
            status = "error" if i % 60 == 0 else "ok"
        else:  # model_deployer
            duration = 10000 + (i % 5000)
            status = "error" if i % 80 == 0 else "ok"

        span = {
            "span_id": f"span_{i}",
            "trace_id": f"trace_{i // 10}",
            "agent_name": agent_name,
            "agent_tier": tier,
            "duration_ms": duration,
            "status": status,
            "metadata": {"fallback_used": i % 200 == 0},
        }

        # Add token usage for feature_analyzer (Hybrid agent)
        if agent_name == "feature_analyzer":
            span["input_tokens"] = 1000 + (i % 500)
            span["output_tokens"] = 500 + (i % 300)
            span["total_tokens"] = span["input_tokens"] + span["output_tokens"]

        spans.append(span)

    return spans


def _compute_latency_stats(
    spans: List[Dict[str, Any]], group_by: str
) -> Dict[str, Dict[str, float]]:
    """Compute latency statistics grouped by field.

    Args:
        spans: List of span dicts
        group_by: Field to group by ("agent_name" or "agent_tier")

    Returns:
        Dict of latency stats by group
    """
    groups = defaultdict(list)

    for span in spans:
        key = span.get(group_by)
        duration = span.get("duration_ms", 0)
        if key is not None and duration:
            groups[key].append(duration)

    stats = {}
    for key, durations in groups.items():
        durations_sorted = sorted(durations)
        if durations_sorted:
            stats[str(key)] = {
                "p50": durations_sorted[int(len(durations_sorted) * 0.50)],
                "p95": durations_sorted[int(len(durations_sorted) * 0.95)],
                "p99": durations_sorted[int(len(durations_sorted) * 0.99)],
                "avg": sum(durations_sorted) / len(durations_sorted),
            }

    return stats


def _compute_error_rates(
    spans: List[Dict[str, Any]], group_by: str
) -> Dict[str, float]:
    """Compute error rates grouped by field.

    Args:
        spans: List of span dicts
        group_by: Field to group by ("agent_name" or "agent_tier")

    Returns:
        Dict of error rates by group
    """
    groups = defaultdict(lambda: {"total": 0, "errors": 0})

    for span in spans:
        key = span.get(group_by)
        if key is not None:
            groups[str(key)]["total"] += 1
            if span.get("status") == "error":
                groups[str(key)]["errors"] += 1

    error_rates = {}
    for key, counts in groups.items():
        if counts["total"] > 0:
            error_rates[key] = counts["errors"] / counts["total"]
        else:
            error_rates[key] = 0.0

    return error_rates


def _compute_token_usage(spans: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    """Compute token usage by agent (for Hybrid/Deep agents).

    Args:
        spans: List of span dicts

    Returns:
        Dict of token usage by agent
    """
    usage = defaultdict(lambda: {"input": 0, "output": 0, "total": 0})

    for span in spans:
        agent_name = span.get("agent_name")
        if agent_name and span.get("input_tokens"):
            usage[agent_name]["input"] += span.get("input_tokens", 0)
            usage[agent_name]["output"] += span.get("output_tokens", 0)
            usage[agent_name]["total"] += span.get("total_tokens", 0)

    return dict(usage)


def _compute_quality_score(
    success_rate: float, p95_latency_ms: float, fallback_rate: float
) -> float:
    """Compute overall quality score (0.0-1.0).

    Args:
        success_rate: Overall success rate (0.0-1.0)
        p95_latency_ms: P95 latency in milliseconds
        fallback_rate: Fallback invocation rate (0.0-1.0)

    Returns:
        Quality score (0.0-1.0)
    """
    # Success rate weight: 60%
    success_score = success_rate * 0.6

    # Latency weight: 30%
    # Target: p95 < 10000ms (10s)
    # Score degrades linearly from 1.0 at 0ms to 0.0 at 30000ms
    if p95_latency_ms <= 10000:
        latency_score = 1.0
    elif p95_latency_ms >= 30000:
        latency_score = 0.0
    else:
        latency_score = 1.0 - ((p95_latency_ms - 10000) / 20000)
    latency_score *= 0.3

    # Fallback rate weight: 10%
    # Lower fallback rate is better
    fallback_score = (1.0 - fallback_rate) * 0.1

    return success_score + latency_score + fallback_score
