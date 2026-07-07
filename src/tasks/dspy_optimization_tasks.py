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
import uuid
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
            return cast(Dict[str, Any], json.loads(_STATE_PATH.read_text()))
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
    # ~1 signal/cycle; 20 ≈ reachable in normal operation
    min_signals = int(os.getenv("DSPY_MIN_SIGNALS", "20"))
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

    # Produce optimized recipient bundles (Shard 09), best-effort per recipient,
    # so the install step below has real bundles to install. Each recipient is
    # isolated: one failing must not abort the others or the cycle.
    from src.agents.feedback_learner.prompt_bundles import RECIPIENT_FACTORIES
    from src.agents.feedback_learner.recipient_optimizer import optimize_and_save_recipient

    recipient_bundles: Dict[str, Any] = {}
    for recipient in RECIPIENT_FACTORIES:
        try:
            path = await optimize_and_save_recipient(recipient, budget=budget)
            recipient_bundles[recipient] = path
        except Exception as e:  # noqa: BLE001 - one recipient must not abort the run
            logger.error("Recipient optimization failed for %s: %s", recipient, e)
            recipient_bundles[recipient] = None

    # Install recipient bundles into the live singletons.
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
        "recipient_bundles": recipient_bundles,
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


async def _run_learning_cycle(task_id: str, window_hours: float) -> Dict[str, Any]:
    """Execute a single feedback learning cycle via FeedbackLearnerAgent.learn()."""
    from datetime import timedelta

    from src.agents.feedback_learner.agent import FeedbackLearnerAgent

    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(hours=window_hours)

    # F15 (audit): wire the REAL feedback source (chatbot_message_feedback) so
    # the loop learns from real user ratings/comments. Previously the agent was
    # constructed with NO feedback_store -> the collector returned [] ->
    # update_effectiveness was pinned at 0.0 (documented starved state). The
    # store fails closed (empty list) if the DB/client is unavailable.
    # F15: wire the REAL feedback source (chatbot_message_feedback). #837: wire the
    # REAL knowledge_stores so applied updates durably persist (read-back confirmed)
    # and update_effectiveness becomes a real measured ratio. One shared builder
    # (used by this task, the /feedback/learn route, and process_feedback_batch)
    # fails closed to (None, None) — the honest unwired path (F15).
    from src.agents.feedback_learner.agent import build_production_feedback_stores

    # #883 deferred: the third element (shared async client) arms the rubric
    # node's learning_signals persistence on this Celery trigger too.
    feedback_store, knowledge_stores, db_client = await build_production_feedback_stores()

    # Optional scope: DSPY_LEARN_FOCUS_AGENTS="agent_a,agent_b" (default: all agents).
    _focus_env = os.environ.get("DSPY_LEARN_FOCUS_AGENTS", "").strip()
    focus_agents = [a.strip() for a in _focus_env.split(",") if a.strip()] or None

    agent = FeedbackLearnerAgent(
        feedback_store=feedback_store,
        knowledge_stores=knowledge_stores,
        db_client=db_client,
        use_llm=True,
        persist_signals=True,
    )
    output = await agent.learn(
        time_range_start=start_time.isoformat(),
        time_range_end=end_time.isoformat(),
        focus_agents=focus_agents,
    )

    # Persist the cycle's artifacts to the tables the /feedback-learning page
    # reads (feedback_learning_batches / feedback_patterns /
    # feedback_knowledge_updates). Without this the beat persisted ONLY dspy
    # training signals, so the page stayed empty (0 cycles / 0 patterns /
    # 0 updates) despite this loop running 4×/day. Non-fatal: the training
    # signal — this task's primary product — is already persisted by the
    # agent's finalize node.
    batch_id = f"beat_{(task_id or uuid.uuid4().hex)[:12]}"
    try:
        from src.api.routes.feedback import persist_learning_cycle_output

        await persist_learning_cycle_output(output, batch_id)
    except Exception as exc:  # noqa: BLE001 — artifact persistence is best-effort
        logger.warning("Learning-cycle artifact persistence failed (batch %s): %s", batch_id, exc)

    return {
        "status": output.status,
        "feedback_count": output.feedback_count,
        "training_reward": output.training_reward,
        "patterns_detected": output.pattern_count,
        "batch_id": batch_id,
        "task_id": task_id,
    }


@celery_app.task(bind=True, name="src.tasks.run_feedback_learning_cycle")
def run_feedback_learning_cycle(self) -> Dict[str, Any]:
    """GENERATES training signals consumed by run_dspy_prompt_optimization.

    Constructs a recent time window, runs FeedbackLearnerAgent.learn() to
    process user feedback and persist a training signal to
    dspy_agent_training_signals (persistence happens in the finalize node).
    The daily optimize beat then reads those signals and gates on
    GEPAOptimizationTrigger before running MIPROv2.

    Schedule: every 6 hours (beat entry "feedback-learning-cycle", queue
    "analytics").  Window size: DSPY_LEARN_WINDOW_HOURS env var (default 24).
    """
    window_hours = float(os.getenv("DSPY_LEARN_WINDOW_HOURS", "24"))
    logger.info(
        "Starting feedback learning cycle: task %s (window=%.1fh)",
        self.request.id,
        window_hours,
    )
    try:
        return cast(
            Dict[str, Any],
            run_async(_run_learning_cycle(self.request.id, window_hours)),
        )
    except Exception as exc:  # noqa: BLE001 — best-effort, never raise out of task
        logger.error("Feedback learning cycle failed: task %s — %s", self.request.id, exc)
        return {
            "status": "failed",
            "error": str(exc),
            "task_id": self.request.id,
        }
