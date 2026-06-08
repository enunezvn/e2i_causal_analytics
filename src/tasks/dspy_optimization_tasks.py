"""DSPy prompt-optimization Celery task (audit F1 keystone).

Runs the closed self-improvement loop on a schedule:
  read signals -> GEPAOptimizationTrigger gate -> optimize (Shard 05)
  -> install recipient bundles (Shard 07).
Mirrors src/tasks/feedback_loop_tasks.py conventions (run_async helper, beat
schedule). This is the ONLY production trigger for prompt optimization.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

from src.workers.celery_app import celery_app

logger = logging.getLogger(__name__)

_STATE_PATH = Path("optimized_modules") / ".trigger_state.json"


def run_async(coro):
    """Run an async coroutine from sync Celery context (mirrors feedback_loop_tasks)."""
    try:
        loop = asyncio.get_running_loop()
        import nest_asyncio

        nest_asyncio.apply()
        return loop.run_until_complete(coro)
    except RuntimeError:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)


def _load_trigger_state() -> Dict[str, Any]:
    try:
        if _STATE_PATH.exists():
            return json.loads(_STATE_PATH.read_text())
    except Exception:  # noqa: BLE001
        pass
    return {}


def _save_trigger_state(state: Dict[str, Any]) -> None:
    try:
        _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _STATE_PATH.write_text(json.dumps(state, indent=2))
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to persist trigger state: %s", e)


def _parse_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _decide_trigger(signals: List[Dict[str, Any]], state: Dict[str, Any]) -> Tuple[bool, str]:
    """Pure trigger decision over the available signals + persisted state."""
    from src.agents.feedback_learner.dspy_integration import GEPAOptimizationTrigger

    n = len(signals)
    mean_reward = (sum(float(s.get("reward", 0.0)) for s in signals) / n) if n else 0.0
    min_signals = int(os.getenv("DSPY_MIN_SIGNALS", "100"))
    trigger = GEPAOptimizationTrigger(min_signals=min_signals)
    return trigger.should_trigger(
        signal_count=n,
        current_reward=mean_reward,
        baseline_reward=float(state.get("baseline_reward", 0.0)),
        last_optimization=_parse_dt(state.get("last_optimization")),
        has_critical_patterns=False,
    )


async def _run(task_id: str, force: bool, budget: str) -> Dict[str, Any]:
    from src.agents.feedback_learner.optimization_runner import (
        run_feedback_learner_optimization,
    )
    from src.agents.feedback_learner.prompt_bundles import install_all_prompt_bundles
    from src.agents.feedback_learner.signal_store import (
        get_feedback_learner_training_signals,
    )

    signals = await get_feedback_learner_training_signals(min_reward=0.5, limit=2000)
    state = _load_trigger_state()
    should, reason = _decide_trigger(signals, state)

    if not force and not should:
        return {
            "status": "skipped",
            "reason": reason,
            "signals": len(signals),
            "task_id": task_id,
        }

    optimization = await run_feedback_learner_optimization(budget=budget)
    # Install recipient bundles (produced by the Shard 09 follow-on; no-op until then).
    installed = install_all_prompt_bundles()

    mean_reward = (
        sum(float(s.get("reward", 0.0)) for s in signals) / len(signals) if signals else 0.0
    )
    _save_trigger_state(
        {
            "last_optimization": datetime.now(timezone.utc).isoformat(),
            "baseline_reward": mean_reward,
        }
    )
    return {
        "status": "completed",
        "trigger_reason": reason,
        "signals": len(signals),
        "optimization": optimization,
        "bundles_installed": installed,
        "task_id": task_id,
    }


@celery_app.task(bind=True, name="src.tasks.run_dspy_prompt_optimization")
def run_dspy_prompt_optimization(
    self, force: bool = False, budget: str = "light"
) -> Dict[str, Any]:
    """Production trigger for the DSPy self-improvement loop (F1 keystone)."""
    logger.info("Starting DSPy prompt optimization: task %s (force=%s)", self.request.id, force)
    return cast(Dict[str, Any], run_async(_run(self.request.id, force, budget)))
