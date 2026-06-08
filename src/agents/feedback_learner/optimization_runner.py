"""Orchestrates a feedback_learner optimization run (audit F1/F3 closure).

Reads persisted training signals -> configures a DSPy LM -> runs
FeedbackLearnerOptimizer.optimize() per phase -> saves the optimized module
per phase under agent_name 'feedback_learner_<phase>' so the analyzer node
(Shard 06) can load each phase independently.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Sequence

logger = logging.getLogger(__name__)

DEFAULT_PHASES: tuple[str, ...] = ("pattern", "recommendation", "summary")
MIN_SIGNALS = 5


async def run_feedback_learner_optimization(
    phases: Sequence[str] = DEFAULT_PHASES,
    budget: str = "light",
    client: Optional[Any] = None,
    min_reward: float = 0.5,
    optimizer_type: str = "gepa",
) -> Dict[str, Any]:
    """Run optimization across phases and persist artifacts. Never raises."""
    from src.optimization.dspy_lm import ensure_dspy_configured
    from src.optimization.gepa import save_optimized_module

    from .dspy_integration import FeedbackLearnerOptimizer
    from .signal_store import get_feedback_learner_training_signals

    result: Dict[str, Any] = {"status": "completed", "signals_used": 0, "phases": {}}

    signals = await get_feedback_learner_training_signals(client=client, min_reward=min_reward)
    result["signals_used"] = len(signals)
    if len(signals) < MIN_SIGNALS:
        result["status"] = "skipped_insufficient_signals"
        logger.info("Optimization skipped: %d < %d signals", len(signals), MIN_SIGNALS)
        return result

    if not ensure_dspy_configured():
        result["status"] = "skipped_no_lm"
        return result

    optimizer = FeedbackLearnerOptimizer(optimizer_type=optimizer_type)  # type: ignore[arg-type]

    for phase in phases:
        try:
            module = await optimizer.optimize(phase, signals, budget=budget)  # type: ignore[arg-type]
            if module is None:
                result["phases"][phase] = {"status": "no_module"}
                continue
            info = save_optimized_module(
                module,
                agent_name=f"feedback_learner_{phase}",
                metadata={"phase": phase, "budget": budget, "optimizer": optimizer_type},
            )
            result["phases"][phase] = {
                "status": "optimized",
                "version_id": info["version_id"],
                "path": info["path"],
            }
            logger.info("Optimized + saved feedback_learner_%s: %s", phase, info["version_id"])
        except Exception as e:  # noqa: BLE001 - one phase failing must not abort the run
            logger.error("Optimization failed for phase %s: %s", phase, e)
            result["phases"][phase] = {"status": "error", "error": str(e)}

    return result
