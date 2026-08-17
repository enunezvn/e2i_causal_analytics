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
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

from .dspy_integration import (
    MIN_FEASIBLE_TRAINSET_EXAMPLES,
    FeedbackLearnerTrainingSignal,
    GEPAOptimizationTrigger,
    gate_supply,
    gate_supply_breakdown,
    gate_trainset_examples,
)

# Re-exported so the gate's SSOT stays addressable from one module (#1661): the
# beat, this module's status surface and the trainset builder all count trainable
# supply through the SAME function (#1668). Its definition lives beside
# ``_signals_to_examples`` in dspy_integration, which is what it must agree with.
__all__ = [
    "LEGACY_MIN_SIGNALS_ENV",
    "MIN_TRAINSET_EXAMPLES_ENV",
    "OPTIMIZER_POOL_MIN_REWARD",
    "OPTIMIZER_SIGNAL_LIMIT",
    "RUNS_TABLE",
    "TABLE",
    "TRIGGER_STATE_PATH",
    "build_signal_record",
    "decide_optimizer_trigger",
    "gate_supply",
    "gate_supply_breakdown",
    "get_feedback_learner_training_signals",
    "get_optimizer_gate_status",
    "gate_trainset_examples",
    "load_trigger_state",
    "optimizer_min_trainset_examples",
    "persist_training_signal",
    "read_optimizer_signal_pool",
]

logger = logging.getLogger(__name__)

TABLE = "dspy_agent_training_signals"
RUNS_TABLE = "prompt_optimization_runs"

# --- Optimizer-gate SSOT (#1661, re-pointed by #1668) ------------------------
#
# The daily beat (src/tasks/dspy_optimization_tasks.py) reads signals through
# this module and then gates on them. Both halves of that gate live here so the
# operator-visible status (GET /feedback/health) is derived from the SAME
# numbers the beat skips on, and cannot quietly disagree with it.
#
# WHAT THE GATE COUNTS, and why it changed (#1668). It used to count rows
# clearing ``reward >= 0.5`` — 8 on the production table. That is a defect-yield
# measure, not a supply measure: ``compute_reward`` gives the coverage and
# actionability terms zero on a cycle that detected no patterns, capping such a
# cycle at EXACTLY 0.5 (0.3 with no rubric), so in 223 stored rows every
# eligible one came from a cycle that found >= 2 patterns.
#
# After #1675 the trainset builder no longer selects that way — it requires each
# phase's INPUT to be non-empty and balances the two label classes — so the gate
# and the builder had come to measure different quantities. Measured
# 2026-08-17, read-only, against the 223 real feedback_learner rows:
#
#     eligible reward >= 0.5                        8   <- the ORIGINAL gate
#     informative pool (non-empty feedback_batch)  75
#     minority label class (pattern phase)         15
#     built pattern examples                       30   <- THE GATE'S UNIT
#
#     pattern examples built from the OLD gate's own 8 rows:   0
#
# The last line is the defect, not the discrepancy: those 8 rows are 100%
# positive, so the builder refuses them as single-class. Twenty of them would
# have opened the gate and compiled nothing.
#
# #1677 re-pointed the gate at the minority label class, which is the right
# CONSTRAINT but not the right UNIT: it published 15 against a threshold of 20
# while the builder produced 30 against an effective 40, so the number the gate
# gated on was half the number it bounded. The gate now counts
# ``gate_trainset_examples`` — the examples ``_signals_to_examples`` will
# actually build for the best-supplied phase — and the threshold is stated in
# the same unit. See dspy_integration section 5 for the derivation.
#
# NOTE this does NOT decouple eligibility from the platform behaving badly:
# today's trainset is bounded by the POSITIVE class (15 of 75), and a positive
# is a cycle that found defects. That coupling is inherent to the label — you
# cannot teach "when to report" without examples of reporting — and it is a
# product question (#1668), not something a gate quantity can fix. What changed
# is that the gate now measures the real constraint instead of a proxy for it.
#
# The pool carries NO reward floor. A correct abstention scores near zero by
# construction, so the negative class and a reward floor are the same set:
# filtering the pool by reward starves the class the balance needs (#1675).
OPTIMIZER_POOL_MIN_REWARD = 0.0
OPTIMIZER_SIGNAL_LIMIT = 2000

# The gate's unit is TRAINSET EXAMPLES (see dspy_integration section 5). The
# threshold has ONE definition — the dataclass default — because two constants
# that agree today are exactly how the previous one drifted out of its unit.
#
# NOT FORWARDED TO CONTAINERS, and deliberately left that way here. Neither this
# name nor the ``DSPY_MIN_SIGNALS`` it replaces appears in ``x-common-env``
# (docker/docker-compose.yml), which is a WHITELIST — so setting either in the
# host ``.env`` does nothing for the beat, and the in-code default governs every
# production run. That gap predates this change and is a KNOWN, DEFERRED
# decision, recorded verbatim in ``test_compose_rag_feedstock_env_1489.py``:
# "forwarding DSPY_MIN_SIGNALS would change when the nightly optimization
# triggers — a behavioral change that needs its own decision, not a drive-by."
# Wiring it here would be that drive-by, in a change whose whole claim is that
# it alters no behaviour. Stated rather than silently fixed so nobody reads the
# override below as effective in prod: it is effective for host-side runs
# (scripts, tests, a manual invocation), not for the container.
MIN_TRAINSET_EXAMPLES_ENV = "DSPY_MIN_TRAINSET_EXAMPLES"

# Read but deliberately NOT honoured. An operator who set DSPY_MIN_SIGNALS=20
# meant "20 signals of the scarcer class", i.e. a 40-example trainset. Reading
# that same 20 as examples would HALVE the gate silently — the identical defect
# this change exists to remove. Ignoring it and saying so fails closed.
LEGACY_MIN_SIGNALS_ENV = "DSPY_MIN_SIGNALS"


def optimizer_min_trainset_examples() -> int:
    """Trainset examples the beat's trigger requires (``DSPY_MIN_TRAINSET_EXAMPLES``).

    The override is NOT forwarded into the containers — see
    ``MIN_TRAINSET_EXAMPLES_ENV`` above — so in production this returns the
    in-code default. It is honoured for host-side runs.

    Clamped up to ``MIN_FEASIBLE_TRAINSET_EXAMPLES``: below that floor the
    production (GEPA) path rejects the trainset before any rollout, so a gate
    that opened there would authorise a run that compiles nothing while still
    stamping ``last_optimization``. Clamping UP is the fail-safe direction
    — it can only make the gate stricter than the operator asked for, never
    looser, and it is logged.

    A garbled override falls back to the default rather than raising: this is
    read on a health endpoint, and a bad env var must not take the page down.
    """
    default = GEPAOptimizationTrigger.min_trainset_examples

    if os.getenv(LEGACY_MIN_SIGNALS_ENV) is not None:
        logger.warning(
            "%s is set but IGNORED: the gate now counts trainset examples, not signals, "
            "so its value would mean half what it used to. Set %s instead; using %d.",
            LEGACY_MIN_SIGNALS_ENV,
            MIN_TRAINSET_EXAMPLES_ENV,
            default,
        )

    raw = os.getenv(MIN_TRAINSET_EXAMPLES_ENV)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "%s=%r is not an integer; using default %d",
            MIN_TRAINSET_EXAMPLES_ENV,
            raw,
            default,
        )
        return default
    if value < MIN_FEASIBLE_TRAINSET_EXAMPLES:
        logger.warning(
            "%s=%d is below the %d-example floor the production optimizer path enforces; "
            "clamping to %d so the gate cannot open on a trainset that compiles nothing.",
            MIN_TRAINSET_EXAMPLES_ENV,
            value,
            MIN_FEASIBLE_TRAINSET_EXAMPLES,
            MIN_FEASIBLE_TRAINSET_EXAMPLES,
        )
        return MIN_FEASIBLE_TRAINSET_EXAMPLES
    return value


# Persisted trigger state, written by the beat after a completed optimization.
# ``optimized_modules`` is a named volume: read-write on worker_medium (the
# producer), READ-ONLY on api (docker/docker-compose.yml) — which is what lets
# the health surface below evaluate the same trigger the beat evaluates. Only
# the beat writes; nothing here does.
TRIGGER_STATE_PATH = Path("optimized_modules") / ".trigger_state.json"


def load_trigger_state() -> Dict[str, Any]:
    """Read the beat's persisted trigger state ({} when absent or unreadable)."""
    try:
        if TRIGGER_STATE_PATH.exists():
            return cast(Dict[str, Any], json.loads(TRIGGER_STATE_PATH.read_text()))
    except Exception:  # noqa: BLE001 - absent/corrupt state is a normal cold start
        pass
    return {}


def _parse_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def decide_optimizer_trigger(
    signals: List[Dict[str, Any]], state: Dict[str, Any], scheduled: bool = False
) -> Tuple[bool, str]:
    """Pure trigger decision over the available signals + persisted state.

    ``signals`` is the WHOLE optimizer pool (see
    :func:`read_optimizer_signal_pool`), not a pre-filtered slice: the quantity
    this gates on is ``gate_trainset_examples`` — the number of EXAMPLES the
    trainset builder will produce for the best-supplied phase, computed from the
    same classifier the builder uses (#1668). Passing a reward-filtered slice
    would understate it — and would reintroduce exactly the divergence this
    function now closes.

    The unit is examples, not signals and not label-class members. The threshold
    is compared against the builder's own output count, so the two cannot drift
    apart the way ``min_signals`` drifted from what it gated.

    THE single decision function: the Celery beat calls it to decide whether to
    optimize, and ``get_optimizer_gate_status`` calls it to tell an operator
    what the beat would decide. Keeping one implementation is the point — the
    trigger checks cooldown BEFORE the trainset size, then a forced interval,
    then a reward delta, so a health surface that modelled only "count >=
    threshold" could report Ready while the beat skipped (#1661).

    ``scheduled`` (#1656) marks the wall-clock cron path, which suppresses the
    cooldown. ``last_optimization`` is stamped *after* a run completes, so on a
    ``crontab(hour=6, minute=0)`` entry any nonzero runtime leaves the next
    fire under the 24h window (a 06:35 finish gives 23.4h) and the task skips —
    a daily schedule that silently runs every OTHER day. The crontab already IS
    the rate limit on that path; a second, drifting rate limiter can only
    interfere. Event-triggered runs keep the cooldown, because nothing else
    bounds how often they fire.

    Passing ``last_optimization=None`` rather than widening the trigger keeps
    the cooldown semantics in one place, so the status surface and the beat
    stay the same function — the #1661 invariant. Callers reporting on the beat
    MUST pass the same ``scheduled`` value the beat uses, or the surface
    reports Ready while the beat skips.
    """
    _, examples = gate_trainset_examples(signals)
    total = len(signals)
    mean_reward = (sum(float(s.get("reward", 0.0)) for s in signals) / total) if total else 0.0
    trigger = GEPAOptimizationTrigger(min_trainset_examples=optimizer_min_trainset_examples())
    last_optimization = None if scheduled else _parse_dt(state.get("last_optimization"))
    return trigger.should_trigger(
        trainset_examples=examples,
        current_reward=mean_reward,
        baseline_reward=float(state.get("baseline_reward", 0.0)),
        last_optimization=last_optimization,
        has_critical_patterns=False,
    )


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
    min_reward: float = OPTIMIZER_POOL_MIN_REWARD,
    limit: int = OPTIMIZER_SIGNAL_LIMIT,
    strict: bool = False,
) -> list[Dict[str, Any]]:
    """Read back persisted feedback_learner signals for optimization (Shard 03/05).

    Thin wrapper over SignalCollectorAdapter.get_signals_for_optimization filtered
    to source_agent='feedback_learner'.

    ``strict`` (#1668) turns a read failure into a raise instead of an empty
    list. See :func:`read_optimizer_signal_pool` for why the gate needs it.
    """
    from src.rag.memory_adapters import SignalCollectorAdapter

    if client is None:
        from src.memory.services.factories import get_supabase_client

        client = await _maybe_await(get_supabase_client())
    if client is None:
        logger.warning("No Supabase client; cannot read training signals")
        if strict:
            raise RuntimeError("No Supabase client available to read training signals")
        return []
    adapter = SignalCollectorAdapter(supabase_client=client)
    return await adapter.get_signals_for_optimization(
        source_agent="feedback_learner", min_reward=min_reward, limit=limit, strict=strict
    )


async def read_optimizer_signal_pool(client: Optional[Any] = None) -> list[Dict[str, Any]]:
    """THE optimizer signal pool — one definition, three readers (#1668).

    The daily beat gates on these rows, hands these same rows to
    ``run_feedback_learner_optimization``, and ``get_optimizer_gate_status``
    reports on these rows. Before #1668 the beat read ``reward >= 0.5`` at
    ``limit=2000`` while the runner re-read at ``min_reward=0.0, limit=1000`` —
    two different row sets, so the number the gate published described a pool
    that was never trained on.

    Newest-first with a PK tiebreak (see
    ``SignalCollectorAdapter.get_signals_for_optimization``), so the slice is
    identical for every reader once the limit binds.

    Cost, measured 2026-08-17 against the production table: 223 rows, ~1.07 MiB,
    ~310 ms warm. The health surface polls this every 30 s. Selecting fewer
    columns would be cheaper but would give the gate a different row shape from
    the builder's, which is precisely the divergence being closed — so the read
    is whole-row on purpose.

    RAISES on a read failure (``strict=True``), unlike every other caller of the
    adapter. The adapter's default is to swallow and return ``[]``, which is
    right for best-effort readers — but these rows are COUNTED and the count is
    published, so an empty list here is a measurement. A swallowed outage would
    surface as ``trainset_examples: 0`` and "Insufficient trainset: 0 < 40 examples",
    which is precisely the fabricated zero the #1661 health contract forbids and
    is indistinguishable from a genuinely single-class corpus. Both callers
    handle the raise: the status returns its ``unavailable`` shape, the beat
    returns a failed status rather than a skip.
    """
    return await get_feedback_learner_training_signals(
        client=client,
        min_reward=OPTIMIZER_POOL_MIN_REWARD,
        limit=OPTIMIZER_SIGNAL_LIMIT,
        strict=True,
    )


async def get_optimizer_gate_status(client: Optional[Any] = None) -> Dict[str, Any]:
    """Report the daily optimizer gate's own inputs, for an operator surface.

    The DSPy prompt-optimization beat returns ``{"status": "skipped"}`` whenever
    its trigger is unsatisfied. That is a legitimate return, so nothing fails
    and nothing alerts — and the loop can stay inert indefinitely while every
    signal an operator looks at stays green (#1661).

    ``would_trigger``/``reason`` come from :func:`decide_optimizer_trigger` —
    the beat's OWN decision function, over the same pool
    (:func:`read_optimizer_signal_pool`) and the same persisted state — so the
    surface cannot report Ready while the beat skips. In particular the cooldown
    branch, which the trigger evaluates BEFORE the trainset size, stays visible
    once supply clears the threshold.

    The counts around it explain that verdict:

    - ``trainset_examples`` — the gate's own input, in the gate's own unit: the
      number of EXAMPLES the trainset builder produces for the best-supplied
      phase, compared directly against ``min_trainset_examples``.
      #1668 replaced ``eligible_signals`` (rows at ``reward >= 0.5``) here: that
      number was 8 while the beat's builder could use 15, and those 8 rows were
      single-class, so the trainset built from them was empty. Its replacement,
      ``trainable_signals``, was right about the constraint but published in a
      different unit from the threshold beside it — 15 against 20 while the
      builder produced 30 against an effective 40. Publishing one unit removes
      the conversion an operator had to do in their head to check the verdict.
    - ``governing_phase`` — the phase these class counts describe: the
      best-supplied one, falling back to the largest usable pool when NO phase
      has both classes. It is never null on a successful read, because the
      single-class case is exactly when the breakdown matters most (it names the
      class the loop is starved of) and a pair of counts beside a null phase
      would describe nothing. ``trainset_examples == 0`` is what says nothing is
      trainable.
    - ``positive_signals`` / ``negative_signals`` — the two classes for that
      phase. This is the actionable pair: it says WHICH class is short, and
      today it is the positive one (15 vs 60, measured 2026-08-17), i.e. supply
      is waiting on the
      platform to exhibit defects.
    - ``total_signals``     — ALL feedback_learner signals ever. The denominator
      is the point: "30 examples out of 223 signals" reads as a low-yield
      problem, "30" alone reads as a volume problem and invites lowering the
      threshold instead.
    - ``last_trainable_signal_at`` — when the SCARCER class last moved (None =
      never). Reporting the newest row of either class would show a date
      advancing daily while the gate stayed frozen.
    - ``optimization_runs`` — rows in ``prompt_optimization_runs``; 0 means the
      loop has never once compiled anything.

    Every count is ``None``, never 0, when the read fails: a fabricated zero on
    a health surface is indistinguishable from a measured one.
    """
    from .dspy_integration import classify_signal_for_phase, gate_supply_breakdown

    min_trainset_examples = optimizer_min_trainset_examples()
    unavailable: Dict[str, Any] = {
        "trainset_examples": None,
        "governing_phase": None,
        "positive_signals": None,
        "negative_signals": None,
        "total_signals": None,
        "last_trainable_signal_at": None,
        "optimization_runs": None,
        "min_trainset_examples": min_trainset_examples,
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
        # The SAME rows the beat gates on and hands to the trainset builder.
        pool = await read_optimizer_signal_pool(client)
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

    # `total` is a separate exact count on purpose: the pool is capped at
    # OPTIMIZER_SIGNAL_LIMIT, so len(pool) is what the gate saw while `total` is
    # how many exist. The gate's own numbers all come from `pool`, never from
    # the wider count, so the figures and the verdict describe the same rows.
    total = int(getattr(total_res, "count", 0) or 0)
    runs = int(getattr(runs_res, "count", 0) or 0)

    # ONE breakdown function, shared with the beat's own task result (#1668):
    # the phase is always named, so the class counts are never left describing
    # nothing. `supply == 0` is what says no phase has both classes.
    #
    # `supply` is per-class; the GATE's number is the trainset it yields, and it
    # comes from the SAME function the trigger gates on rather than from a
    # doubling written out again here. Publishing the per-class number beside an
    # examples threshold is the unit mismatch this change removes.
    phase, supply, positives, negatives = gate_supply_breakdown(pool)
    _, trainset_examples = gate_trainset_examples(pool)

    # When supply last moved. ``supply == min(positives, negatives)``, so:
    #
    #   unequal — only the SCARCER class can raise it; the newest row of that
    #     class is what last did. Reporting the newest row of either class here
    #     would advance the date on every abundant-class arrival while the gate
    #     stayed frozen, which is the false-green shape #1661 removed (negatives
    #     arrive steadily in production, positives do not).
    #   equal   — adding EITHER class leaves ``min`` unchanged, so supply last
    #     moved when the PAIR completed: the newest usable row of either class.
    #
    # `pool` is newest-first, so the first match is the newest.
    if positives == negatives:

        def _moved_supply(row: Dict[str, Any]) -> bool:
            return classify_signal_for_phase(row, phase) is not None

    else:
        scarce_label = positives < negatives

        def _moved_supply(row: Dict[str, Any]) -> bool:
            return classify_signal_for_phase(row, phase) is scarce_label

    last_trainable = next(
        (row.get("created_at") for row in pool if _moved_supply(row)),
        None,
    )

    # scheduled=True mirrors the beat's own call (#1656). This surface exists to
    # report what the beat WOULD decide, so the flag must match it — asking with
    # a different value is exactly the Ready-while-skipping defect of #1661.
    would_trigger, reason = decide_optimizer_trigger(pool, load_trigger_state(), scheduled=True)

    return {
        "trainset_examples": trainset_examples,
        "governing_phase": phase,
        "positive_signals": positives,
        "negative_signals": negatives,
        "total_signals": total,
        "last_trainable_signal_at": last_trainable,
        "optimization_runs": runs,
        "min_trainset_examples": min_trainset_examples,
        "would_trigger": would_trigger,
        "reason": reason,
    }
