#!/usr/bin/env python
"""Insights-panel side-by-side A/B for the DSPy-lane provider flip.

Generates five REAL insight panels (server-derived groundings only - no
caller-posted figures, no fabricated inputs) under each candidate LM and
emits the payloads side by side for human review. Qualitative evidence for
the flip decision; the hard gates live in scripts/run_dspy_lane_ab.py.

Run INSIDE the prod container (read-only rootfs -> pipe over stdin)::

    docker exec -i e2i_api python - < scripts/run_dspy_lane_insights_ab.py

Panels: home_kpi, executive_brief, knowledge_graph, feedback_learning,
experiments. The originally proposed hte/causal_discovery/treatment_effect
panels take frontend-posted figures or expiring persisted analyses as input;
generating them here would require inventing those inputs, so the five fully
server-derived panels stand in.

Grounding construction MIRRORS the corresponding handlers in
src/api/routes/insights_strategic.py (same reads, same arguments). The route
Redis cache is deliberately bypassed: its key hashes only the grounding, not
the LM, so cached payloads would return one model's output for all three.

The LM override uses ``dspy.context`` around ``generate_insight``, which works
because src/insights/common.py ``run_signature`` defaults to ``lm_cache=True``
(the program call inherits the context LM). ``lm_cache=False`` would force
``get_default_dspy_model()`` and ignore the override.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time

from dotenv import load_dotenv

sys.path.insert(0, "/app")

# DSPy LM config reads env at import time in parts of the app; a script that
# touches it must load .env before anything else resolves keys (issue #470).
# Inside the prod container the env is already injected and this is a no-op.
load_dotenv()

DEFAULT_MODELS = [
    "openai/gpt-5.6-terra",
    "anthropic/claude-sonnet-5",
    "anthropic/claude-haiku-4-5-20251001",
]
BRAND = "Kisqali"


async def build_groundings() -> dict:
    """Build the five panel groundings from real server data (route-faithful)."""
    from datetime import datetime, timedelta, timezone

    groundings: dict = {}

    # --- home_kpi (mirrors home_kpi_insight._load) ---
    from src.api.routes.kpi import get_kpi_calculator
    from src.insights import home_kpi

    calc = get_kpi_calculator()
    metas = calc.list_kpis()
    batch = calc.calculate_batch(
        kpi_ids=[m.id for m in metas], use_cache=True, context={"brand": BRAND}
    )
    groundings["home_kpi"] = (
        home_kpi,
        home_kpi.build_grounding(BRAND, None, metas, batch.results),
    )

    # --- executive_brief (mirrors executive_brief_insight) ---
    from src.api.routes.gaps import list_opportunities
    from src.insights import executive_brief
    from src.insights.causal_context import fetch_commercial_drivers, format_driver_names
    from src.insights.clinical_context import fetch_clinical_payload, format_clinical_context

    feed = await list_opportunities(brand=BRAND, min_roi=None, difficulty=None, limit=5)
    levers = format_driver_names(
        await fetch_commercial_drivers(BRAND, outcomes=("TRx", "NRx", "market share", "ROI"))
    )
    clinical_context = format_clinical_context(await fetch_clinical_payload(BRAND, "TRx"))
    groundings["executive_brief"] = (
        executive_brief,
        executive_brief.build_grounding(
            brand=BRAND,
            total_addressable_value=feed.total_addressable_value,
            quick_wins_count=feed.quick_wins_count,
            steady_plays_count=feed.steady_plays_count,
            strategic_bets_count=feed.strategic_bets_count,
            suppressed_count=feed.suppressed_count or 0,
            opportunities=[
                {
                    "rank": o.rank,
                    "recommended_action": o.recommended_action,
                    "expected_roi": o.roi_estimate.expected_roi,
                    "revenue_impact": o.roi_estimate.estimated_revenue_impact,
                    "gap_metric": o.gap.metric,
                    "gap_percentage": o.gap.gap_percentage,
                    "segment_value": o.gap.segment_value,
                    "implementation_difficulty": o.implementation_difficulty.value,
                }
                for o in feed.opportunities
            ],
            causal_drivers=levers,
            clinical_context=clinical_context,
        ),
    )

    # --- knowledge_graph (mirrors knowledge_graph_insight._load, brand=All) ---
    from src.insights import knowledge_graph
    from src.memory.semantic_memory import get_semantic_memory

    sm = get_semantic_memory()
    nodes = sm.list_nodes(limit=500, curated_only=True)
    rels = sm.list_relationships(limit=500, curated_only=True)
    groundings["knowledge_graph"] = (
        knowledge_graph,
        knowledge_graph.build_grounding(
            "All",
            nodes,
            rels,
            node_count=sm.count_nodes(curated_only=True),
            rel_count=len(rels),
        ),
    )

    # --- feedback_learning (mirrors feedback_learning_insight, days=7) ---
    from src.api.repositories.feedback_repository import FeedbackRepository
    from src.insights import feedback_learning
    from src.memory.services.factories import get_async_supabase_client
    from src.repositories.chatbot_feedback import get_chatbot_feedback_repository
    from src.repositories.learning_signals_feedback import get_learning_signals_feedback_store

    repo = FeedbackRepository()
    batches = await repo.count_recent_and_last()
    patterns = await repo.list_patterns()
    updates = await repo.list_updates()
    now = datetime.now(timezone.utc)
    cycles_24h = sum(1 for b in batches if (now - b.timestamp).total_seconds() < 86400)
    last_cycle_at = max((b.timestamp for b in batches), default=None)
    client = await get_async_supabase_client()
    thumbs = await get_chatbot_feedback_repository(supabase_client=client).get_feedback_summary(
        days=7
    )
    window_start = (now - timedelta(days=7)).isoformat()
    signals = await get_learning_signals_feedback_store(supabase_client=client).get_feedback(
        start_time=window_start
    )
    rewards = [float(s["metadata"]["reward"]) for s in signals]
    avg_reward = sum(rewards) / len(rewards) if rewards else None
    per_agent: dict[str, list[float]] = {}
    for s in signals:
        per_agent.setdefault(str(s["agent"]), []).append(float(s["metadata"]["reward"]))
    low_reward_agents = sorted(
        ((agent, sum(v) / len(v)) for agent, v in per_agent.items() if sum(v) / len(v) < 0.5),
        key=lambda t: t[1],
    )
    groundings["feedback_learning"] = (
        feedback_learning,
        feedback_learning.build_grounding(
            cycles_24h=cycles_24h,
            last_cycle_at=(last_cycle_at.isoformat() if last_cycle_at else None),
            thumbs_7d=int(thumbs.get("total_feedback", 0) or 0),
            signals_7d=len(signals),
            avg_reward_7d=avg_reward,
            patterns=[p.model_dump(mode="json") for p in patterns],
            updates=[u.model_dump(mode="json") for u in updates],
            low_reward_agents=low_reward_agents,
        ),
    )

    # --- experiments (mirrors experiments_portfolio_insight._load, brand=All) ---
    from src.api.dependencies.supabase_client import get_supabase
    from src.insights import experiments as experiments_insight_mod
    from src.repositories.provenance import apply_provenance_filter

    sb = get_supabase()
    if sb is None:
        raise RuntimeError("Database unavailable")
    query = (
        sb.table("ml_experiments")
        .select(
            "id, brand, intervention_channel, "
            "ab_experiment_results(effect_estimate, p_value, is_significant)"
        )
        .eq("status", "running")
        .not_.is_("intervention_channel", "null")
    )
    query = apply_provenance_filter(query, include_synthetic=True)
    rows = query.execute().data or []
    groundings["experiments"] = (
        experiments_insight_mod,
        experiments_insight_mod.build_grounding("All", rows),
    )

    return groundings


def main() -> None:
    import dspy

    models = json.loads(os.environ.get("AB_MODELS", "null")) or DEFAULT_MODELS
    groundings = asyncio.run(build_groundings())

    results: dict = {}
    for panel, (module, grounding) in groundings.items():
        results[panel] = {}
        for model in models:
            t0 = time.perf_counter()
            try:
                with dspy.context(lm=dspy.LM(model, cache=False)):
                    payload = module.generate_insight(grounding)
                results[panel][model] = {
                    "payload": payload,
                    "latency_s": time.perf_counter() - t0,
                    "error": None,
                }
            except Exception as exc:  # noqa: BLE001 - error class is the datum
                results[panel][model] = {
                    "payload": None,
                    "latency_s": time.perf_counter() - t0,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            print(f"{panel} x {model.split('/')[-1]} done", file=sys.stderr)

    print("RESULTS_JSON_BEGIN")
    print(json.dumps(results, default=str))
    print("RESULTS_JSON_END")


if __name__ == "__main__":
    main()
