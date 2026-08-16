"""Durable persistence for feedback_learner training signals (audit F5).

The finalized FeedbackLearnerTrainingSignal is written to the
`dspy_agent_training_signals` table (migration 014) so the self-improver's
own signals survive process restart and can be read back by the optimizer
(Shard 03/05). The CognitiveRAG path already writes other agents' signals to
the same table via src/rag/memory_adapters.py; this closes the gap for the
feedback_learner itself.
"""

from __future__ import annotations

import inspect
import logging
import os
from typing import Any, Dict, Optional

from .dspy_integration import FeedbackLearnerTrainingSignal

logger = logging.getLogger(__name__)

TABLE = "dspy_agent_training_signals"
RUNS_TABLE = "prompt_optimization_runs"

# --- Optimizer-gate SSOT (#1661) ---------------------------------------------
#
# The daily beat (src/tasks/dspy_optimization_tasks.py) reads signals through
# this module and then gates on their count. Both halves of that gate live here
# so the operator-visible status (GET /feedback/health) is derived from the SAME
# numbers the beat skips on, and cannot quietly disagree with it.
#
# Why the floor matters more than the threshold — measured 2026-08-16 in prod:
# 218 feedback_learner signals exist, 8 clear ``reward >= 0.5``. That is not a
# volume shortfall. ``compute_reward`` gives the coverage and actionability
# terms zero on any cycle that detected no patterns, which caps such a cycle at
# EXACTLY 0.5 (0.3 with no rubric) — so only cycles that found real quality
# defects can ever produce an eligible training example. All 8 came from the
# 2026-08-05..08-08 window, the only stretch where user-reward ratings dipped
# below the analyzer's 3.0 gate. Eligibility is therefore coupled to the
# platform behaving badly; see #1661 before touching either constant.
OPTIMIZER_MIN_REWARD = 0.5
OPTIMIZER_SIGNAL_LIMIT = 2000
DEFAULT_MIN_SIGNALS = 20
MIN_SIGNALS_ENV = "DSPY_MIN_SIGNALS"


def optimizer_min_signals() -> int:
    """Signal count the beat's trigger requires, honouring ``DSPY_MIN_SIGNALS``.

    A garbled override falls back to the default rather than raising: this is
    read on a health endpoint, and a bad env var must not take the page down.
    """
    raw = os.getenv(MIN_SIGNALS_ENV)
    if raw is None:
        return DEFAULT_MIN_SIGNALS
    try:
        return int(raw)
    except ValueError:
        logger.warning(
            "%s=%r is not an integer; using default %d", MIN_SIGNALS_ENV, raw, DEFAULT_MIN_SIGNALS
        )
        return DEFAULT_MIN_SIGNALS


def build_signal_record(signal: FeedbackLearnerTrainingSignal) -> Dict[str, Any]:
    """Map a FeedbackLearnerTrainingSignal to a migration-014 row dict.

    Pure: no I/O. Column names match database/memory/014_dspy_training_signals.sql.
    """
    d = signal.to_dict()
    quality_metrics = dict(d.get("quality_metrics") or {})
    # Fold rubric evaluation into quality_metrics so the row stays single-table.
    rubric = d.get("rubric_evaluation") or {}
    if rubric:
        quality_metrics["rubric"] = rubric
    return {
        "source_agent": "feedback_learner",
        "batch_id": signal.batch_id or None,
        "input_context": {
            **(d.get("input_context") or {}),
            # carries the bounded real content (feedback_batch, etc.) added by
            # Shard 04 so the optimizer's signal->example conversion has content.
        },
        "output": d.get("output") or {},
        "quality_metrics": quality_metrics,
        "reward": d.get("reward", 0.0),
        "latency_breakdown": d.get("latency") or {},
        "total_latency_ms": int(signal.total_latency_ms or 0),
        "model_used": signal.model_used or "deterministic",
        "llm_calls": int(signal.llm_calls or 0),
        "total_tokens": int(signal.total_tokens or 0),
        "has_cognitive_context": signal.cognitive_context is not None,
        "is_training_example": True,
    }


async def _maybe_await(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


async def persist_training_signal(
    signal: FeedbackLearnerTrainingSignal,
    client: Optional[Any] = None,
) -> bool:
    """Insert one finalized signal. Returns True on success, False otherwise.

    Never raises: a DB outage must not fail a learning cycle.
    """
    record = build_signal_record(signal)
    try:
        if client is None:
            from src.memory.services.factories import get_supabase_client

            client = await _maybe_await(get_supabase_client())
        if client is None:
            logger.warning("No Supabase client; feedback_learner signal not persisted")
            return False
        await _maybe_await(client.table(TABLE).insert(record).execute())
        logger.info("Persisted feedback_learner training signal batch=%s", signal.batch_id)
        return True
    except Exception as e:  # noqa: BLE001 - persistence is best-effort
        logger.error("Failed to persist feedback_learner training signal: %s", e)
        return False


async def get_feedback_learner_training_signals(
    client: Optional[Any] = None,
    min_reward: float = 0.5,
    limit: int = 1000,
) -> list[Dict[str, Any]]:
    """Read back persisted feedback_learner signals for optimization (Shard 03/05).

    Thin wrapper over SignalCollectorAdapter.get_signals_for_optimization filtered
    to source_agent='feedback_learner'.
    """
    from src.rag.memory_adapters import SignalCollectorAdapter

    if client is None:
        from src.memory.services.factories import get_supabase_client

        client = await _maybe_await(get_supabase_client())
    if client is None:
        logger.warning("No Supabase client; cannot read training signals")
        return []
    adapter = SignalCollectorAdapter(supabase_client=client)
    return await adapter.get_signals_for_optimization(
        source_agent="feedback_learner", min_reward=min_reward, limit=limit
    )


async def get_optimizer_gate_status(client: Optional[Any] = None) -> Dict[str, Any]:
    """Report the daily optimizer gate's own inputs, for an operator surface.

    The DSPy prompt-optimization beat returns ``{"status": "skipped"}`` whenever
    its trigger is unsatisfied. That is a legitimate return, so nothing fails
    and nothing alerts — and the loop can stay inert indefinitely while every
    signal an operator looks at stays green (#1661).

    This derives the gate's verdict from the SAME table, filters and constants
    the beat uses, so the two cannot drift:

    - ``eligible_signals``  — feedback_learner signals at ``reward >= min_reward``
    - ``total_signals``     — ALL feedback_learner signals ever. The denominator
      is the point: "8 of 218" reads as a low-yield problem, "8" alone reads as
      a volume problem and invites lowering the threshold instead.
    - ``last_eligible_signal_at`` — when supply last moved (None = never).
    - ``optimization_runs`` — rows in ``prompt_optimization_runs``; 0 means the
      loop has never once compiled anything.
    - ``would_trigger``     — whether the count gate is satisfied right now.

    Every count is ``None``, never 0, when the read fails: a fabricated zero on
    a health surface is indistinguishable from a measured one.
    """
    min_signals = optimizer_min_signals()
    unavailable: Dict[str, Any] = {
        "eligible_signals": None,
        "total_signals": None,
        "last_eligible_signal_at": None,
        "optimization_runs": None,
        "min_signals": min_signals,
        "min_reward": OPTIMIZER_MIN_REWARD,
        "would_trigger": None,
        "reason": "Optimizer gate status unavailable (no database client)",
    }

    if client is None:
        from src.memory.services.factories import get_supabase_client

        client = await _maybe_await(get_supabase_client())
    if client is None:
        logger.warning("No Supabase client; cannot read optimizer gate status")
        return unavailable

    try:
        eligible_res = await _maybe_await(
            client.table(TABLE)
            .select("created_at", count="exact")
            .eq("source_agent", "feedback_learner")
            .gte("reward", OPTIMIZER_MIN_REWARD)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        total_res = await _maybe_await(
            client.table(TABLE)
            .select("signal_id", count="exact")
            .eq("source_agent", "feedback_learner")
            .limit(1)
            .execute()
        )
        runs_res = await _maybe_await(
            client.table(RUNS_TABLE).select("run_id", count="exact").limit(1).execute()
        )
    except Exception as e:  # noqa: BLE001 - a health surface must never 500
        logger.warning("Optimizer gate status read failed: %s", e)
        return {**unavailable, "reason": f"Optimizer gate status unavailable ({e})"}

    eligible = int(getattr(eligible_res, "count", 0) or 0)
    total = int(getattr(total_res, "count", 0) or 0)
    runs = int(getattr(runs_res, "count", 0) or 0)
    eligible_rows = getattr(eligible_res, "data", None) or []
    last_eligible = eligible_rows[0].get("created_at") if eligible_rows else None

    would_trigger = eligible >= min_signals
    if would_trigger:
        reason = (
            f"{eligible} of {total} feedback_learner signals clear reward "
            f">= {OPTIMIZER_MIN_REWARD}; threshold {min_signals} met"
        )
    else:
        reason = (
            f"Optimizer inert: {eligible} of {total} feedback_learner signals clear "
            f"reward >= {OPTIMIZER_MIN_REWARD}, below the {min_signals} required. "
            "Only cycles that detected quality patterns can clear that floor "
            "(see #1661); "
            + (f"last eligible signal {last_eligible}" if last_eligible else "none ever recorded")
            + f"; {runs} optimization run(s) recorded"
        )

    return {
        "eligible_signals": eligible,
        "total_signals": total,
        "last_eligible_signal_at": last_eligible,
        "optimization_runs": runs,
        "min_signals": min_signals,
        "min_reward": OPTIMIZER_MIN_REWARD,
        "would_trigger": would_trigger,
        "reason": reason,
    }
