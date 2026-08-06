"""answer_relevancy is confounded with the retrieval hit rate (#1489 deferral 7).

WHAT THE DEFERRAL ASKED FOR, AND WHAT WAS ALREADY DONE
------------------------------------------------------
#1489's close-out recorded "answer_relevancy has failed its 0.3 gate in both
baselines" and recommended ratcheting to the measured baseline. That gate no
longer exists: commit 6709d42e ("#1485 calibrate the real-pipeline gate from
the first honest run", 2026-08-05, an ancestor of main) had ALREADY replaced
the aspirational 0.30 with a measured-baseline floor of 0.10 and added
``MIN_RETRIEVAL_HIT_RATE``. Both recorded baselines (0.140 and 0.179) PASS the
gate that actually ships. The ratchet was not missing.

THE DEFECT THAT IS REAL
-----------------------
``answer_relevancy`` as reported is an aggregate over ALL rows, and every
zero-retrieval row scores 0.0 (ragas zeroes a noncommittal answer, and every
zero-retrieval turn in both baselines abstained). So it factorises exactly:

    aggregate_relevancy = retrieval_hit_rate x hit_conditioned_relevancy

Verified against both recorded runs, to the digit they were recorded at:

    deployed   (n=10, hit 3/10): 0.300  x 0.4667 = 0.1400   recorded 0.140
    pre-deploy (n=15, hit 5/15): 0.3333 x 0.5360 = 0.1787   recorded 0.179

One factor is already gated on its own (``MIN_RETRIEVAL_HIT_RATE`` = 0.21).
Gating the PRODUCT as well judges the generation half through a moving lens:
the 0.10 aggregate floor implied a hit-conditioned floor of ``0.10 / hit_rate``
— 0.333 at today's 0.30, but only 0.167 at 0.60. So the aggregate gate
contradicted the retrieval gate at the bottom and went slack at the top:

    hit 0.21 (legal) x 0.4667 (the deployed baseline's OWN quality) = 0.0980
        -> BLOCKED by the 0.10 floor
    hit 0.60 x 0.18 (a 65% generation collapse)                     = 0.1080
        -> PASSES the 0.10 floor, because retrieval improved

Improving the hit rate is the project's stated next move, so the second window
opens above hit rate 0.50 — precisely where the work is heading.

THE TRADEOFF THESE TESTS ALSO PIN: at today's ~0.30 hit rate the new fixed
0.20 floor is LOOSER on generation than the entangled 0.333 the aggregate
implied. That is deliberate — 0.333 was an artifact of the hit rate, not a
calibrated floor — but it is a real loss of strictness at today's operating
point, and ``test_the_new_floor_is_looser_at_todays_hit_rate`` records it so
nobody discovers it by surprise.

These tests pin the decomposition, the two measured baselines it is calibrated
from, and the arithmetic guarantee that the aggregate backstop can no longer
contradict the two factor gates.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from src.rag.real_pipeline_eval import (
    MIN_HIT_CONDITIONED_RELEVANCY,
    MIN_RETRIEVAL_HIT_RATE,
    RAGAS_MEASURED_BASELINES,
    REAL_PIPELINE_THRESHOLDS,
    check_hit_conditioned_relevancy_gate,
    check_run_gates,
    hit_conditioned_relevancy,
)


def _rows(scores: List[Optional[float]], n_contexts: List[int]) -> List[Dict[str, Any]]:
    """per_sample rows in the shape run_dspy_lane_ragas_judge.py emits."""
    return [
        {
            "query_id": f"q{i:02d}",
            "n_contexts": ctx,
            "faithfulness": 0.9 if ctx > 0 else None,
            "answer_relevancy": ar,
        }
        for i, (ar, ctx) in enumerate(zip(scores, n_contexts, strict=True), start=1)
    ]


def _block(rows: List[Dict[str, Any]], **overrides: Any) -> Dict[str, Any]:
    ctx_rows = [r for r in rows if r["n_contexts"] > 0]

    def _mean(subset: List[Dict[str, Any]], key: str) -> Optional[float]:
        vals = [r[key] for r in subset if r.get(key) is not None]
        return (sum(vals) / len(vals)) if vals else None

    block = {
        "model": "cognitive",
        "n_samples": len(rows),
        "n_faithfulness": len(ctx_rows),
        "faithfulness": _mean(ctx_rows, "faithfulness"),
        "answer_relevancy": _mean(rows, "answer_relevancy"),
        "per_sample": rows,
    }
    block.update(overrides)
    return block


def _baseline_block(baseline: Dict[str, Any]) -> Dict[str, Any]:
    """Rebuild a judge block from a recorded baseline's hit-row scores."""
    hit_scores = list(baseline["hit_row_relevancy_scores"])
    n_zero_retrieval = baseline["n_samples"] - baseline["n_hit"]
    scores = hit_scores + [0.0] * n_zero_retrieval
    contexts = [2] * len(hit_scores) + [0] * n_zero_retrieval
    return _block(_rows(scores, contexts))


# ---------------------------------------------------------------------------
# The recorded baselines — PROVENANCE checks, not validation
# ---------------------------------------------------------------------------
#
# Read these for what they are (codex iter-2): the per-row scores in
# RAGAS_MEASURED_BASELINES are DERIVED from the recorded aggregates, not from a
# re-run of the gpt-4o judge, so asserting the factors multiply back to the
# aggregate checks the derivation's arithmetic and that the recorded numbers
# have not been edited apart. They say nothing about whether the judge's scores
# were right. Re-deriving a baseline is a judge run, not a test.
# ---------------------------------------------------------------------------


def test_two_baselines_are_recorded_machine_readably():
    assert len(RAGAS_MEASURED_BASELINES) == 2
    for baseline in RAGAS_MEASURED_BASELINES:
        assert baseline["n_hit"] <= baseline["n_samples"]
        assert len(baseline["hit_row_relevancy_scores"]) == baseline["n_hit"]
        assert baseline["_method"], "a baseline without its derivation is not auditable"


@pytest.mark.parametrize("baseline", RAGAS_MEASURED_BASELINES, ids=lambda b: b["label"])
def test_recorded_aggregate_is_the_product_of_the_two_factors(baseline):
    """The confound, stated as an identity and checked against the real runs.

    Provenance, not validation: the hit-row scores were derived FROM the
    recorded aggregate, so this pins that the derivation is self-consistent and
    that no one has since edited one number without the others.
    """
    hit_rate = baseline["n_hit"] / baseline["n_samples"]
    conditioned = baseline["answer_relevancy_hit_conditioned"]
    assert hit_rate == pytest.approx(baseline["retrieval_hit_rate"], abs=5e-4)
    assert hit_rate * conditioned == pytest.approx(baseline["answer_relevancy"], abs=5e-4)


@pytest.mark.parametrize("baseline", RAGAS_MEASURED_BASELINES, ids=lambda b: b["label"])
def test_hit_conditioned_relevancy_reproduces_the_recorded_baseline(baseline):
    value, n = hit_conditioned_relevancy(_baseline_block(baseline))
    assert n == baseline["n_hit"]
    assert value == pytest.approx(baseline["answer_relevancy_hit_conditioned"], abs=5e-4)


@pytest.mark.parametrize("baseline", RAGAS_MEASURED_BASELINES, ids=lambda b: b["label"])
def test_every_recorded_baseline_passes_the_floor_it_calibrated(baseline):
    """A floor above its own measured baseline would block the run it was
    derived from — the mirror image of gate-shopping."""
    assert baseline["answer_relevancy_hit_conditioned"] >= MIN_HIT_CONDITIONED_RELEVANCY
    assert baseline["retrieval_hit_rate"] >= MIN_RETRIEVAL_HIT_RATE
    assert baseline["answer_relevancy"] >= REAL_PIPELINE_THRESHOLDS["answer_relevancy"]


# ---------------------------------------------------------------------------
# hit_conditioned_relevancy
# ---------------------------------------------------------------------------


def test_averages_only_over_context_bearing_rows():
    """Zero-retrieval rows are what the aggregate drags in; they must not count."""
    rows = _rows([0.9, 0.5, 0.0, 0.0, 0.0], [2, 2, 0, 0, 0])
    value, n = hit_conditioned_relevancy(_block(rows))
    assert n == 2
    assert value == pytest.approx(0.70)


def test_a_scored_zero_on_a_hit_row_still_counts():
    """A row that retrieved and was still judged 0.0 is a real measurement —
    that is exactly the generation failure this metric exists to see."""
    rows = _rows([0.9, 0.5, 0.0], [2, 2, 2])
    value, n = hit_conditioned_relevancy(_block(rows))
    assert n == 3
    assert value == pytest.approx(0.4667, abs=1e-4)


def test_no_hit_rows_yields_no_measurement():
    value, n = hit_conditioned_relevancy(_block(_rows([0.0, 0.0], [0, 0])))
    assert (value, n) == (None, 0)


def test_malformed_block_yields_no_measurement():
    assert hit_conditioned_relevancy({}) == (None, 0)
    assert hit_conditioned_relevancy({"per_sample": "nope"}) == (None, 0)


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


def test_gate_passes_a_run_at_the_measured_baseline():
    rows = _rows([0.9, 0.5, 0.0] + [0.0] * 7, [2, 2, 2] + [0] * 7)
    passed, failures = check_hit_conditioned_relevancy_gate(_block(rows))
    assert passed, failures


# The aggregate floor this gate replaces, kept as a literal so these tests keep
# demonstrating the miss even after REAL_PIPELINE_THRESHOLDS moves again.
_SUPERSEDED_AGGREGATE_FLOOR = 0.10


def test_gate_blocks_the_collapse_the_aggregate_waves_through():
    """The headline case: retrieval IMPROVES to 6/10 while answers on retrieved
    rows collapse 65%. The superseded aggregate floor passes it; this must not.

    6/10 rather than 5/10 is load-bearing: the miss window only opens above hit
    rate 0.50, because below it the 0.10 aggregate floor implies a conditioned
    floor above 0.20.
    """
    rows = _rows([0.18] * 6 + [0.0] * 4, [2] * 6 + [0] * 4)
    block = _block(rows)

    assert block["answer_relevancy"] == pytest.approx(0.108)
    assert block["answer_relevancy"] >= _SUPERSEDED_AGGREGATE_FLOOR, (
        "premise: the aggregate floor this replaces does not catch this"
    )
    assert block["answer_relevancy"] >= REAL_PIPELINE_THRESHOLDS["answer_relevancy"], (
        "premise: nor does the total-collapse backstop"
    )

    passed, failures = check_hit_conditioned_relevancy_gate(block)
    assert not passed
    assert any("hit-conditioned" in f for f in failures), failures


def test_the_floor_is_no_weaker_than_the_gate_it_replaces():
    """Decoupling must not BUY the stability with detection.

    The superseded aggregate floor implied a hit-conditioned floor of
    ``0.10 / hit_rate`` — strictest exactly where the hit rate is lowest. At the
    measured baseline hit rate (0.30) it implied 0.333. A fixed floor at or
    above that is no weaker at today's operating point, and strictly stronger
    everywhere above it, where the implied floor decays toward zero.

    An earlier revision of this lane set the floor to 0.20 and documented the
    resulting loss as an accepted tradeoff; codex iteration 2 flagged it HIGH
    (a regression the old gate caught that neither new gate caught), and it was
    a real loss taken for a ~13-point reduction in false-block rate that the
    shipped gate was already paying anyway.
    """
    assert MIN_HIT_CONDITIONED_RELEVANCY >= _SUPERSEDED_AGGREGATE_FLOOR / 0.30

    for hit_rate in (0.40, 0.50, 0.60, 0.70):
        assert MIN_HIT_CONDITIONED_RELEVANCY > _SUPERSEDED_AGGREGATE_FLOOR / hit_rate, (
            f"weaker than the superseded gate at hit rate {hit_rate}"
        )


def test_blocks_the_regression_the_superseded_gate_caught_at_todays_hit_rate():
    """The concrete case codex iteration 2 named: hit rate unchanged at 3/10,
    hit-conditioned relevancy down to 0.25 (a 51% fall from the pooled 0.510
    baseline). The old aggregate blocked it at 0.075 < 0.10; the decomposition
    must block it too."""
    rows = _rows([0.25] * 3 + [0.0] * 7, [2] * 3 + [0] * 7)
    block = _block(rows)
    assert block["answer_relevancy"] == pytest.approx(0.075)
    assert block["answer_relevancy"] < _SUPERSEDED_AGGREGATE_FLOOR, "old gate blocked this"

    passed, failures = check_hit_conditioned_relevancy_gate(block)
    assert not passed, "the decomposition must not lose what the aggregate caught"
    assert any("hit-conditioned" in f for f in failures), failures


def test_gate_fails_closed_when_it_cannot_measure():
    """'We could not measure it' is not 'it is fine' — the #1485 discipline."""
    for block in (
        {},
        {"per_sample": []},
        _block(_rows([0.0, 0.0], [0, 0])),
        _block(_rows([None, 0.5], [2, 2])),
    ):
        passed, failures = check_hit_conditioned_relevancy_gate(block)
        assert not passed, block
        assert failures


def test_gate_fails_closed_on_a_non_finite_score():
    rows = _rows([0.9, 0.5], [2, 2])
    rows[0]["answer_relevancy"] = float("nan")
    passed, failures = check_hit_conditioned_relevancy_gate(_block(rows))
    assert not passed
    assert failures


# ---------------------------------------------------------------------------
# The aggregate backstop can no longer contradict the factor gates
# ---------------------------------------------------------------------------


def test_aggregate_floor_cannot_block_a_run_that_passes_both_factors():
    """The old 0.10 aggregate floor blocked a run at the legal minimum hit rate
    (0.21) carrying the deployed baseline's own generation quality (0.4667 ->
    aggregate 0.0980). The backstop must sit at or below the product of the two
    factor floors so that can never happen again."""
    worst_passing_aggregate = MIN_RETRIEVAL_HIT_RATE * MIN_HIT_CONDITIONED_RELEVANCY
    assert REAL_PIPELINE_THRESHOLDS["answer_relevancy"] <= worst_passing_aggregate


def test_aggregate_backstop_still_catches_total_collapse():
    """Weaker is not the same as absent: an all-zero relevancy run still blocks."""
    assert REAL_PIPELINE_THRESHOLDS["answer_relevancy"] > 0.0


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


def test_check_run_gates_includes_the_hit_conditioned_gate():
    """A caller must not be able to gate the metrics while ignoring whether the
    answers on retrieved rows were any good."""
    rows = _rows([0.18] * 6 + [0.0] * 4, [2] * 6 + [0] * 4)
    retrieval = {
        "n_records": 10,
        "n_errors": 0,
        "n_with_contexts": 6,
        "retrieval_hit_rate": 0.6,
    }
    passed, failures = check_run_gates(_block(rows), retrieval)
    assert not passed
    assert any("hit-conditioned" in f for f in failures), failures


def test_check_run_gates_passes_a_healthy_run():
    rows = _rows([0.9, 0.8, 0.7] + [0.0] * 7, [2, 2, 2] + [0] * 7)
    retrieval = {
        "n_records": 10,
        "n_errors": 0,
        "n_with_contexts": 3,
        "retrieval_hit_rate": 0.3,
    }
    passed, failures = check_run_gates(_block(rows), retrieval)
    assert passed, failures
