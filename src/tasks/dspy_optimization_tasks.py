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

# --- RAG-prompt optimization leg (#1486) -------------------------------------
#
# Path to a replay records file produced by scripts/replay_golden_set.py
# (--record-out). Records are consumed as PURE JSON; this module imports no
# replay code. Needs a docker-compose x-common-env entry to reach the worker.
RAG_RECORDS_PATH_ENV = "DSPY_RAG_RECORDS_PATH"

# Judge budget, in GEPA metric calls. Measured against installed dspy 3.1.0:
# auto="light" resolves to ~384-396 metric calls almost independently of dataset
# size (5 examples -> 384, 20 -> 396) because auto_budget is driven by
# num_candidates=6, not by len(trainset). Each metric call is one RAGAS
# evaluate_sample = 4 sub-metrics, each at least one judge LLM call, so "light"
# is 1,500+ judge calls. Against #504's calibration (~96 min for a 30-sample
# RAGAS eval, where judge throughput was THE binding constraint) that is a
# many-hour job — unacceptable on a 24h beat. Capping examples does not help;
# only an explicit max_metric_calls does. Hence the conservative default below.
RAG_MAX_METRIC_CALLS_ENV = "DSPY_RAG_MAX_METRIC_CALLS"
_RAG_DEFAULT_MAX_METRIC_CALLS = 40

# Below this many usable examples the leg does nothing. #1485 measured that only
# 3 of 10 replayed turns retrieved any evidence (a turn that retrieved nothing
# records an empty contexts list ON PURPOSE), and the RAGAS metric refuses a
# no-context example — so "not enough usable examples" is the EXPECTED nightly
# outcome, not an edge case.
_RAG_MIN_USABLE_EXAMPLES = 5
_RAG_MAX_EXAMPLES = 20


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


def _rag_max_metric_calls() -> int:
    """Judge budget for the RAG leg, env-tunable. See RAG_MAX_METRIC_CALLS_ENV."""
    raw = os.environ.get(RAG_MAX_METRIC_CALLS_ENV, "").strip()
    if not raw:
        return _RAG_DEFAULT_MAX_METRIC_CALLS
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "%s=%r is not an integer; using default %d",
            RAG_MAX_METRIC_CALLS_ENV,
            raw,
            _RAG_DEFAULT_MAX_METRIC_CALLS,
        )
        return _RAG_DEFAULT_MAX_METRIC_CALLS
    return max(1, value)


def _rag_records_fingerprint(path: str, max_metric_calls: int) -> str:
    """Digest of the records file AND the budget, for run-once dedup.

    Content rather than mtime: the replay may be re-run and rewrite an identical
    file, and re-spending the judge budget on identical inputs buys nothing.

    The budget is part of the key because a run can legitimately end without
    improving on the seed (see run_rag_prompt_optimization). Keying on content
    alone would then mark those records permanently done, so a later
    DSPY_RAG_MAX_METRIC_CALLS increase — the one action that could actually find
    an improvement — would be silently skipped.
    """
    import hashlib

    digest = hashlib.sha256(Path(path).read_bytes())
    digest.update(f"|max_metric_calls={max_metric_calls}".encode())
    return digest.hexdigest()


def _instructions_of(module: Any) -> Tuple[str, ...]:
    """Instruction text of every predictor in a DSPy module, in order."""
    predictors = module.predictors() if hasattr(module, "predictors") else []
    return tuple(
        str(getattr(getattr(p, "signature", None), "instructions", "") or "") for p in predictors
    )


def load_rag_examples_from_records(path: str) -> List[Any]:
    """Build dspy Examples from replay records (#1485 shape), as PURE JSON.

    Accepts either the wrapper the replay writes (``{"records": [...]}``) or a
    bare list. Keeps only turns that can actually be judged: a query, a
    non-empty answer, at least one retrieved context, and no recorded error.
    Filtering here rather than in the metric matters — the RAGAS metric REFUSES
    an unjudgeable example, and a refusal inside GEPA is silently converted to
    failure_score 0.0, which would fabricate a bad-quality signal.

    ``retrieved_contexts`` is set alongside ``evidence_board`` on purpose: the
    signature wants an evidence string, while the metric wants the passage list
    so context_precision is computed per passage rather than over one blob.
    """
    import dspy

    raw = json.loads(Path(path).read_text())
    records = raw.get("records", []) if isinstance(raw, dict) else raw

    examples: List[Any] = []
    for rec in records:
        if not isinstance(rec, dict) or rec.get("error"):
            continue
        query = (rec.get("query") or "").strip()
        answer = (rec.get("response_text") or "").strip()
        contexts = [c for c in (rec.get("contexts") or []) if isinstance(c, str) and c.strip()]
        if not query or not answer or not contexts:
            continue
        examples.append(
            dspy.Example(
                user_query=query,
                # Records carry no investigation goal; the query stands in. This
                # is an INPUT substitution, not a score, but it does mean the
                # prompt is tuned against a slightly narrower input distribution
                # than production sees.
                investigation_goal=query,
                evidence_board=json.dumps(contexts),
                intent=rec.get("detected_intent") or "UNKNOWN",
                retrieved_contexts=contexts,
                synthesis=answer,
            ).with_inputs("user_query", "investigation_goal", "evidence_board", "intent")
        )
    return examples


async def run_rag_prompt_optimization() -> Dict[str, Any]:
    """Opportunistically optimize the cognitive-RAG synthesis prompt with RAGAS.

    Normally a no-op: the replay that produces its input is manual-only, so the
    common nightly outcome is a logged skip costing zero API calls. Every skip
    path returns BEFORE any metric or optimizer is constructed.

    On success the artifact is saved under
    :data:`src.rag.cognitive_rag_dspy.OPTIMIZED_SYNTHESIS_AGENT_NAME`, which
    ``AgentModule`` loads on its next construction — the leg->artifact->runtime
    chain.
    """
    records_path = os.environ.get(RAG_RECORDS_PATH_ENV, "").strip()
    if not records_path:
        reason = f"{RAG_RECORDS_PATH_ENV} is not set"
        logger.info(
            "RAG prompt optimization skipped: %s. This is the expected steady state — "
            "the replay is manual-only. To enable: run "
            "`.venv/bin/python scripts/replay_golden_set.py --target cognitive --record-out <path>` and set "
            "%s=<path>.",
            reason,
            RAG_RECORDS_PATH_ENV,
        )
        return {"status": "skipped", "reason": reason}

    if not Path(records_path).exists():
        reason = f"records file not found: {records_path}"
        logger.info(
            "RAG prompt optimization skipped: %s. Regenerate with "
            "`.venv/bin/python scripts/replay_golden_set.py --target cognitive --record-out %s` (%s).",
            reason,
            records_path,
            RAG_RECORDS_PATH_ENV,
        )
        return {"status": "skipped", "reason": reason}

    # Budget is resolved before the dedup check because it is part of the key.
    budget = _rag_max_metric_calls()
    fingerprint = _rag_records_fingerprint(records_path, budget)
    state = _load_trigger_state()
    if state.get("rag_records_fingerprint") == fingerprint:
        reason = "records already optimized at this budget (unchanged since last run)"
        logger.info(
            "RAG prompt optimization skipped: %s. Re-run "
            "scripts/replay_golden_set.py for fresh turns, or raise %s to search "
            "harder on the same records (%s).",
            reason,
            RAG_MAX_METRIC_CALLS_ENV,
            records_path,
        )
        return {"status": "skipped", "reason": reason, "fingerprint": fingerprint}

    raw = json.loads(Path(records_path).read_text())
    total_records = len(raw.get("records", []) if isinstance(raw, dict) else raw)
    examples = load_rag_examples_from_records(records_path)

    if len(examples) < _RAG_MIN_USABLE_EXAMPLES:
        reason = (
            f"only {len(examples)} usable example(s) of {total_records} record(s); "
            f"need >= {_RAG_MIN_USABLE_EXAMPLES}"
        )
        logger.info(
            "RAG prompt optimization skipped: %s. Expected — #1485 measured ~3 of 10 "
            "replayed turns retrieve any evidence, and a turn with no retrieved "
            "context cannot be judged. Re-run scripts/replay_golden_set.py with more "
            "queries and point %s at the new file.",
            reason,
            RAG_RECORDS_PATH_ENV,
        )
        return {
            "status": "skipped",
            "reason": reason,
            "usable_examples": len(examples),
            "total_records": total_records,
        }

    examples = examples[:_RAG_MAX_EXAMPLES]
    split = max(1, int(len(examples) * 0.8))
    trainset, valset = examples[:split], examples[split:] or examples[:1]

    # Everything below costs real judge calls. Say so before spending them.
    logger.info(
        "RAG prompt optimization starting: %d example(s) (%d train / %d val), "
        "max_metric_calls=%d -> approx %d RAGAS judge calls (4 sub-metrics each). "
        "Tune with %s.",
        len(examples),
        len(trainset),
        len(valset),
        budget,
        budget * 4,
        RAG_MAX_METRIC_CALLS_ENV,
    )

    import dspy

    from src.optimization.dspy_lm import ensure_dspy_configured
    from src.optimization.gepa import create_gepa_optimizer
    from src.optimization.gepa.metrics import get_metric_for_agent
    from src.optimization.gepa.versioning import save_optimized_module
    from src.rag.cognitive_rag_dspy import (
        OPTIMIZED_SYNTHESIS_AGENT_NAME,
        EvidenceSynthesisSignature,
    )

    if not ensure_dspy_configured():
        reason = "no DSPy LM configured"
        logger.warning("RAG prompt optimization skipped: %s", reason)
        return {"status": "skipped", "reason": reason}

    # Raises when the RAGAS judge cannot run (#1486): refusing here, before the
    # run, is deliberate — a per-example refusal is swallowed by dspy into
    # failure_score 0.0 and would optimize against fabricated signal. The
    # caller's guard turns this into a logged per-leg skip.
    metric = get_metric_for_agent("cognitive_rag")

    # auto=None + max_metric_calls: GEPA asserts exactly one budget is set, and
    # auto's ~384-396 calls is far beyond what the judge can serve nightly.
    optimizer = create_gepa_optimizer(
        metric=metric,
        trainset=trainset,
        valset=valset,
        auto=None,
        max_metric_calls=budget,
        seed=42,
    )
    module = dspy.ChainOfThought(EvidenceSynthesisSignature)
    seed_instructions = _instructions_of(module)
    optimized = await asyncio.to_thread(optimizer.compile, module, trainset=trainset, valset=valset)

    # A judge that degraded mid-run (heuristic fallback, timeout, rate-limit)
    # loses those examples to failure_score 0.0, so the winning candidate may
    # have been selected against noise. Persisting it would ship a prompt chosen
    # by an outage AND fingerprint the records as done. Neither: fail the leg and
    # let the next beat retry.
    degraded = int(getattr(metric, "degraded_examples", 0) or 0)
    if degraded:
        reason = f"judge degraded mid-run ({degraded} example(s) unscored by the judge)"
        logger.error(
            "RAG prompt optimization discarded: %s. No artifact saved and records "
            "left un-fingerprinted so the next run retries.",
            reason,
        )
        return {"status": "failed", "reason": reason, "degraded_examples": degraded}

    # GEPA seeds its candidate pool with the base program (dspy gepa.py:553 ->
    # gepa core/state.py:54) and picks argmax over val_aggregate_scores
    # (core/result.py:77), where `max` resolves ties to the FIRST index — the
    # seed. So an exhausted budget hands back the base prompt unchanged, and
    # saving that as "optimized" would be a lie.
    #
    # But it IS a completed measurement, so the fingerprint is persisted: these
    # records at this budget have been tried. Skipping that would re-run the
    # whole compile every triggered beat and re-spend the entire judge budget on
    # identical inputs at steady state. Persisting is safe precisely because the
    # budget is part of the key — raising DSPY_RAG_MAX_METRIC_CALLS produces a
    # different fingerprint and re-runs. (The degraded path above deliberately
    # does NOT fingerprint: there the judge failed transiently and a retry is
    # exactly what we want.)
    if _instructions_of(optimized) == seed_instructions:
        reason = f"budget exhausted without improvement (max_metric_calls={budget})"
        logger.info(
            "RAG prompt optimization produced no change: %s. Nothing saved; raise %s "
            "to search harder on the same records, or refresh them.",
            reason,
            RAG_MAX_METRIC_CALLS_ENV,
        )
        no_improvement_state = _load_trigger_state()
        no_improvement_state["rag_records_fingerprint"] = fingerprint
        _save_trigger_state(no_improvement_state)
        return {
            "status": "skipped",
            "reason": reason,
            "examples": len(examples),
            "max_metric_calls": budget,
            "fingerprint": fingerprint,
        }

    info = save_optimized_module(
        module=optimized,
        agent_name=OPTIMIZED_SYNTHESIS_AGENT_NAME,
        metadata={
            "source_records": records_path,
            "examples": len(examples),
            "max_metric_calls": budget,
        },
    )
    # Merge into freshly-loaded state, not the copy read before the compile: that
    # snapshot is minutes stale by now, and writing it back would revert whatever
    # another writer recorded meanwhile. Same discipline as the beat's final save.
    persisted = _load_trigger_state()
    persisted["rag_records_fingerprint"] = fingerprint
    _save_trigger_state(persisted)

    logger.info(
        "RAG prompt optimization complete: saved %s (version %s)",
        info["path"],
        info["version_id"],
    )
    return {
        "status": "completed",
        "examples": len(examples),
        "total_records": total_records,
        "max_metric_calls": budget,
        "artifact": info["path"],
        "version_id": info["version_id"],
        "agent_name": OPTIMIZED_SYNTHESIS_AGENT_NAME,
    }


async def _run_rag_leg_guarded() -> Dict[str, Any]:
    """Run the RAG leg so that no failure can abort the nightly beat.

    Mirrors the per-recipient guard below: one leg failing must not cost the
    others. This is also where the metric's deliberate construction-time raise
    on a keyless box becomes a logged skip rather than a failed run.
    """
    try:
        return await run_rag_prompt_optimization()
    except Exception as e:  # noqa: BLE001 - one leg must not abort the beat
        logger.error("RAG prompt optimization failed: %s", e, exc_info=True)
        return {"status": "failed", "reason": str(e)}


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

    # RAG-prompt leg (#1486): normally a zero-cost no-op — see
    # run_rag_prompt_optimization. Guarded so it can never abort the beat.
    rag_optimization = await _run_rag_leg_guarded()

    mean_reward = (
        sum(float(s.get("reward", 0.0)) for s in signals) / len(signals) if signals else 0.0
    )
    # Merge into FRESHLY-loaded state rather than writing a fresh dict. The file
    # has more than one writer now: the RAG leg above persists
    # rag_records_fingerprint mid-beat, and a whole-file overwrite here erased it
    # a few lines after it was written — which made the dedup dead on arrival,
    # re-spending the judge budget on identical records every triggered beat.
    # Re-load rather than reusing `state` from before the legs ran, or this
    # would restore a snapshot that predates their writes.
    final_state = _load_trigger_state()
    final_state.update(
        {
            "last_optimization": datetime.now(timezone.utc).isoformat(),
            "baseline_reward": mean_reward,
        }
    )
    _save_trigger_state(final_state)
    return {
        "status": "completed",
        "trigger_reason": reason,
        "signals": len(signals),
        "optimization": optimization,
        "recipient_bundles": recipient_bundles,
        "bundles_installed": installed,
        "rag_optimization": rag_optimization,
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
    # Post-auto_apply-gate note: this scheduled path runs propose-only — the
    # state built below carries no `auto_apply`, so KnowledgeUpdaterNode withholds
    # every apply and update_effectiveness is honestly None on this path even
    # when knowledge_stores is wired.
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
