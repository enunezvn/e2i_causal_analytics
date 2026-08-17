"""Orchestrates a feedback_learner optimization run (audit F1/F3 closure).

Reads persisted training signals -> configures a DSPy LM -> runs
FeedbackLearnerOptimizer.optimize() per phase -> saves the optimized module
per phase under agent_name 'feedback_learner_<phase>' so the analyzer node
(Shard 06) can load each phase independently.
"""

from __future__ import annotations

import logging
import traceback
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

DEFAULT_PHASES: tuple[str, ...] = ("pattern", "recommendation", "summary")
MIN_SIGNALS = 5


async def run_feedback_learner_optimization(
    phases: Sequence[str] = DEFAULT_PHASES,
    budget: str = "light",
    client: Optional[Any] = None,
    min_reward: Optional[float] = None,
    optimizer_type: str = "gepa",
    signals: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Run optimization across phases and persist artifacts. Never raises.

    ``signals`` (#1668) is the candidate pool. The beat passes the rows its gate
    already counted, so the published verdict and the trainset describe the SAME
    rows — re-reading here meant the gate's "N trainable" and the builder's input
    were two different queries seconds apart, and before #1668 they were also two
    different filters and two different limits. When omitted the pool is read
    from ``read_optimizer_signal_pool``, the one definition of it.

    ``min_reward`` overrides that pool's reward floor and exists only for tests
    and ad-hoc runs. It defaults to ``None`` — meaning "use the pool" — rather
    than to a number, because a number here is a second definition of the pool
    that can drift from the gate's. #1675 established why the floor must be 0:
    for the pattern phase the training label IS the patterns a cycle found, so a
    reward floor selects on *having found patterns* and hands the optimizer a
    100%-positive trainset (measured 2026-08-17: 8 of 223 real signals, every one with a
    non-empty label — and ``_signals_to_examples`` builds ZERO examples from
    them, because a single-class pool is an honest skip).

    This does NOT open the optimizer's trigger. The daily beat still gates on
    ``decide_optimizer_trigger`` (``src/tasks/dspy_optimization_tasks.py``);
    nothing below runs until that fires. ``MIN_SIGNALS`` here is only a cheap
    pre-check on raw rows — the binding guard is ``len(trainset) < 5`` inside
    ``_optimize_with_gepa``, over the examples actually built.
    """
    from src.optimization.dspy_lm import ensure_dspy_configured
    from src.optimization.gepa import save_optimized_module

    from .dspy_integration import (
        FeedbackLearnerOptimizer,
        recorded_set_sizes,
        trainset_examples_for_phase,
    )
    from .signal_store import (
        OPTIMIZER_SIGNAL_LIMIT,
        get_feedback_learner_training_signals,
        read_optimizer_signal_pool,
    )

    result: Dict[str, Any] = {"status": "completed", "signals_used": 0, "phases": {}}

    pool: List[Dict[str, Any]]
    if signals is not None:
        pool = list(signals)
    elif min_reward is None:
        # ``read_optimizer_signal_pool`` raises on a failed read so the gate can
        # tell an outage from an empty corpus (#1668). This function's contract
        # is "never raises", so the failure becomes a distinguishable STATUS
        # rather than the "skipped_insufficient_signals" a swallowed [] would
        # have produced — which reads as "nothing to do" for a database that is
        # down.
        try:
            pool = await read_optimizer_signal_pool(client=client)
        except Exception as e:  # noqa: BLE001 - documented never-raises contract
            logger.error("Optimization aborted: signal read failed: %s", e)
            result["status"] = "failed_signal_read"
            result["error"] = str(e)
            return result
    else:
        pool = await get_feedback_learner_training_signals(
            client=client, min_reward=min_reward, limit=OPTIMIZER_SIGNAL_LIMIT
        )
    result["signals_used"] = len(pool)
    if len(pool) < MIN_SIGNALS:
        result["status"] = "skipped_insufficient_signals"
        logger.info("Optimization skipped: %d < %d signals", len(pool), MIN_SIGNALS)
        return result

    if not ensure_dspy_configured():
        result["status"] = "skipped_no_lm"
        return result

    # Run tracking (prompt_optimization_runs / optimized_instructions, wired
    # per migration 023). Best-effort by contract: record_* never raises.
    from src.repositories.prompt_optimization import (
        record_run_completed,
        record_run_discarded,
        record_run_failed,
        record_run_started,
    )

    optimizer = FeedbackLearnerOptimizer(optimizer_type=optimizer_type)  # type: ignore[arg-type]
    # Record the optimizer that will ACTUALLY run: __init__ falls back to
    # miprov2 when GEPA/dspy is unavailable (and None means neither can run),
    # so persisting the requested type would misreport fallback runs.
    effective_optimizer = optimizer.optimizer_type

    for phase in phases:
        run_id = None
        if effective_optimizer:
            # The columns are called trainset_size / valset_size, and both
            # siblings (`recipient_optimizer`, the RAG leg) fill them with the
            # sets they hand to `compile()`. This used to receive `len(pool)` —
            # 223 for a phase that builds 30 and a phase that builds 4, so every
            # historical row claimed a trainset 7x larger than the one that ran
            # and gave two phases an order of magnitude apart the same number.
            # `recorded_set_sizes` reports what each optimizer path actually
            # passes: GEPA the 80/20 split (24/6 at 30 examples), MIPROv2 the
            # whole list with no valset. Same defect class as the gate's own
            # unit (#1668) — a name that stopped matching its quantity — and the
            # example count alone would have been the next stop along it.
            n_train, n_val = recorded_set_sizes(
                trainset_examples_for_phase(pool, phase), effective_optimizer
            )
            run_id = await record_run_started(
                agent_name=f"feedback_learner_{phase}",
                optimizer_type=effective_optimizer,
                budget_preset=budget,
                trainset_size=n_train,
                valset_size=n_val,
                created_by="run_dspy_prompt_optimization",
                client=client,
            )
        try:
            module = await optimizer.optimize(phase, pool, budget=budget)  # type: ignore[arg-type]
            if module is None:
                # optimize() returns None ONLY from pre-compile guards
                # (dspy/GEPA unavailable, <5 phase examples, unavailable
                # phase); compile failures raise into the except below. No
                # budget was spent, so discard the provisional row instead of
                # recording a failure that never ran.
                result["phases"][phase] = {"status": "no_module"}
                await record_run_discarded(run_id, client=client)
                continue
            info = save_optimized_module(
                module,
                agent_name=f"feedback_learner_{phase}",
                metadata={"phase": phase, "budget": budget, "optimizer": optimizer_type},
            )
            await record_run_completed(run_id, module=module, artifact_info=info, client=client)
            result["phases"][phase] = {
                "status": "optimized",
                "version_id": info["version_id"],
                "path": info["path"],
            }
            logger.info("Optimized + saved feedback_learner_%s: %s", phase, info["version_id"])
        except Exception as e:  # noqa: BLE001 - one phase failing must not abort the run
            logger.error("Optimization failed for phase %s: %s", phase, e)
            await record_run_failed(run_id, str(e), traceback.format_exc(), client=client)
            result["phases"][phase] = {"status": "error", "error": str(e)}

    # #1668: a run that compiled NOTHING must not report the same status as one
    # that did. This is the issue's own acceptance item ("a daily task that has
    # never run should not look identical to one that ran and found nothing to
    # do"), and the trainset fix makes the outcome more reachable: a single-class
    # signal pool is now an explicit skip rather than a silently biased trainset.
    # Leaving it as "completed" would move the silent inertness from the beat
    # down into here. The only consumer is the beat, which embeds this dict as
    # `optimization`; nothing branches on the string.
    if not any(p.get("status") == "optimized" for p in result["phases"].values()):
        result["status"] = "completed_no_modules"
        logger.info(
            "Optimization run compiled no modules across phases %s: %s",
            list(phases),
            {k: v.get("status") for k, v in result["phases"].items()},
        )

    return result
