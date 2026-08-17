"""#1668: the gate and the trainset builder, measured against the REAL corpus.

READ-ONLY. This test issues one bounded select against
``dspy_agent_training_signals`` and writes nothing. It is the faithful form of
the unit-level invariant: the identity
``len(trainset) == trainset_examples_for_phase(pool, phase)`` is
asserted over whatever the production table actually holds today, not over
constructed rows, so a real signal shape the synthetic fixtures do not cover
(a malformed ``output`` blob, a new key, a row the builder's ``dspy.Example``
construction rejects) shows up here rather than in a beat that promised N
trainable examples and produced fewer.

Gated behind an explicit opt-in because it needs live Supabase credentials; CI
has none, and a test that silently no-ops on a missing client would assert
nothing while looking green. Run it on the box:

    E2I_RUN_REAL_DB_READS=1 .venv/bin/pytest \\
      tests/integration/test_optimizer_gate_supply_real_1668.py -v -s

Measured 2026-08-17 on the 223 real rows (recorded here so a future reader can
tell drift from a defect — the assertions below are all relational, none pin
these numbers):

    A  eligible reward >= 0.5                        8
    B  informative pool (non-empty feedback_batch)  75
    C  minority label class (pattern phase)         15
    D  built pattern examples                       30   == 2 * C
    pattern examples from the OLD gate's 8 rows      0
    corpus span 2026-06-09..2026-08-17            68.8 days
    positives in the last 8 recorded days            0

The threshold's UNIT is pinned here too, not just the supply's: the gate is
compared against ``trainset_examples_for_phase``, and this asserts that function
equals what the builder returns on whatever production holds today.
"""

from __future__ import annotations

import json
import os

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("E2I_RUN_REAL_DB_READS") != "1",
    reason="reads the production signals table; set E2I_RUN_REAL_DB_READS=1 to run",
)


@pytest.mark.asyncio
async def test_gate_supply_matches_the_builder_on_the_real_corpus(capsys):
    from src.agents.feedback_learner.dspy_integration import (
        DSPY_AVAILABLE,
        OPTIMIZABLE_PHASES,
        FeedbackLearnerOptimizer,
        gate_trainset_examples,
        label_class_counts,
        trainset_examples_for_phase,
    )
    from src.agents.feedback_learner.signal_store import (
        optimizer_min_trainset_examples,
        read_optimizer_signal_pool,
    )

    if not DSPY_AVAILABLE:
        pytest.skip("dspy required to build the trainset half of the identity")

    pool = await read_optimizer_signal_pool()
    assert pool, "the pool came back empty — the read failed, so nothing below is a measurement"

    optimizer = FeedbackLearnerOptimizer(optimizer_type="gepa")
    for phase in OPTIMIZABLE_PHASES:
        positives, negatives = label_class_counts(pool, phase)
        built = optimizer._signals_to_examples(pool, phase)
        expected = trainset_examples_for_phase(pool, phase)
        print(
            f"{phase}: pool={len(pool)} usable={positives + negatives} "
            f"pos={positives} neg={negatives} built={len(built)} gate_unit={expected}"
        )
        # THE invariant, on real rows, and now with NO conversion factor in the
        # assertion: the quantity the gate compares against its threshold is the
        # quantity the builder produces.
        assert len(built) == expected

    phase, examples = gate_trainset_examples(pool)
    threshold = optimizer_min_trainset_examples()
    print(f"gate: phase={phase} trainset_examples={examples} threshold={threshold}")

    # The number the gate compares IS the trainset the run would compile, on the
    # phase the gate named — not a proxy for it and not half of it. Both sides
    # of the published comparison are the same unit, so a reader can check the
    # verdict without doubling anything.
    assert phase is not None, "no phase has both label classes — see the breakdown above"
    assert examples == len(optimizer._signals_to_examples(pool, phase))

    # The old gate's quantity, for contrast — and the reason it was the wrong
    # one: those rows are single-class, so they build nothing.
    eligible = [row for row in pool if float(row.get("reward") or 0.0) >= 0.5]
    print(f"reward>=0.5 rows: {len(eligible)}")
    assert len(optimizer._signals_to_examples(eligible, "pattern")) == 0 or len(eligible) == 0

    # Positive control for that null result: the SAME builder on the SAME code
    # path does produce examples from the full pool, so "0" above is a property
    # of that row set rather than of a broken builder.
    assert len(optimizer._signals_to_examples(pool, "pattern")) > 0


@pytest.mark.asyncio
async def test_the_status_surface_and_the_beat_agree_on_the_real_corpus():
    """The #1661 invariant, against production rather than a fake client."""
    from src.agents.feedback_learner.signal_store import (
        decide_optimizer_trigger,
        get_optimizer_gate_status,
        load_trigger_state,
        read_optimizer_signal_pool,
    )

    pool = await read_optimizer_signal_pool()
    assert pool, "empty pool — the read failed"

    beat_should, beat_reason = decide_optimizer_trigger(pool, load_trigger_state(), scheduled=True)
    status = await get_optimizer_gate_status()

    # Printed so `-s` gives the exact payload /feedback/health serves. A PR that
    # claims "before and after" for this surface should paste this, not a
    # reconstruction of it.
    print(json.dumps(status, indent=1, default=str))

    assert status["would_trigger"] == beat_should
    assert status["reason"] == beat_reason
    assert status["trainset_examples"] is not None
    assert f"{status['trainset_examples']}" in status["reason"] or beat_should
