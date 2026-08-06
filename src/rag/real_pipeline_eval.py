"""Real-pipeline RAGAS gate — adapter + fail-closed gates (#1485).

Why this module exists
----------------------
``scripts/run_ragas_eval.py`` calls ``RAGEvaluationPipeline.run_evaluation()``
with no ``rag_pipeline`` argument, so ``src/rag/evaluation.py:1499`` skips
``_generate_answers`` and the judge scores the golden set's *hardcoded*
answers over ``retrieved_contexts`` that are byte-identical to the reference
``contexts``. Context precision/recall are therefore 1.0-by-construction and
faithfulness/answer_relevancy score the fixture author's prose. That job is
still useful — as a judge-drift sentinel on frozen input — but it cannot see
production quality.

This module is the honest half: it turns REAL replay records (genuinely
generated answers over genuinely retrieved contexts, produced by
``scripts/replay_golden_set.py --record-out``) into judge input, and gates the
judge's verdict fail-closed.

What it deliberately reuses rather than rebuilds
------------------------------------------------
The DSPy-lane A/B harness already solved the hard parts and its gates made a
real decision (blocking the ``anthropic/claude-sonnet-5`` flip on faithfulness
0.122 vs baseline 0.690 — ``docs/reports/dspy_lane_ab_20260718.md`` §3). So:

* ``build_ragas_samples`` converts replay records to judge samples, including
  the drop-rules for errored/empty answers.
* ``_ragas_consistency_error`` reconciles a judge block's aggregates against
  its own per-sample rows.
* ``_ragas_scoreless_error`` catches rows the judge covered but never scored.
* ``GATE_RAGAS_MIN_FAITHFULNESS_N`` is the faithfulness denominator floor.

Those two underscore-named helpers are imported across module boundaries on
purpose: duplicating ~70 lines of subtle fail-closed reconciliation would let
the two copies drift, and drift in *this* logic is silent by nature. The
judging step itself is ``scripts/run_dspy_lane_ragas_judge.py``, invoked
unchanged.

The trap this module exists to keep shut
----------------------------------------
``RAGASEvaluator.evaluate_sample`` (``src/rag/evaluation.py:982``) does::

    if not sample.retrieved_contexts:
        sample.retrieved_contexts = sample.contexts

so a sample whose replay retrieved NOTHING is silently rescored against
whatever reference evidence ``contexts`` holds. Real-pipeline samples must
therefore always carry an empty ``contexts`` — otherwise a zero-retrieval turn
is judged against curated evidence and scores well, which is exactly the
tautology #1485 exists to remove. ``build_ragas_samples`` never emits
``contexts``; ``test_adapter_never_emits_reference_contexts`` pins it.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from src.optimization.dspy_lane_ab import (
    GATE_RAGAS_MIN_FAITHFULNESS_N,
    HEURISTIC_EVALUATION_METHOD,
    _is_finite_number,
    _ragas_consistency_error,
    _ragas_heuristic_contamination_error,
    _ragas_scoreless_error,
    build_ragas_samples,
)

__all__ = [
    "HEURISTIC_EVALUATION_METHOD",
    "MIN_HIT_CONDITIONED_RELEVANCY",
    "MIN_REAL_PIPELINE_SAMPLES",
    "MIN_RETRIEVAL_HIT_RATE",
    "RAGAS_MEASURED_BASELINES",
    "REAL_PIPELINE_METRICS",
    "REAL_PIPELINE_THRESHOLDS",
    "build_samples_from_replay",
    "check_hit_conditioned_relevancy_gate",
    "check_real_pipeline_gates",
    "check_retrieval_gate",
    "check_run_gates",
    "contexts_from_evidence",
    "hit_conditioned_relevancy",
    "summarize_retrieval",
]

# The only two metrics the real path can honestly report. context_precision
# and context_recall need a ground-truth reference the replay deliberately
# does not fabricate, so they are dropped rather than reported as
# 1.0-by-construction — mirroring how run_dspy_lane_ragas_judge.py already
# omits them. Note this is about what is REPORTED and GATED, not cost:
# _evaluate_with_ragas computes all four metrics per sample regardless.
REAL_PIPELINE_METRICS: Tuple[str, ...] = ("faithfulness", "answer_relevancy")

# ===========================================================================
# BASELINE — MEASURED 2026-08-05, n=15, deployed main 9784abbd
# ===========================================================================
# 15 golden questions replayed through POST /api/cognitive/rag (0 errors),
# judged by the frozen gpt-4o judge via the production RAGASEvaluator:
#
#     retrieval hit      5/15  = 0.333
#     faithfulness       0.524  (n=5 context-bearing replays; SE 0.195)
#     answer_relevancy   0.179  (n=15; SE 0.083)
#
# Judging wall time 4m25s. Compare the fixture gate's 0.804 answer_relevancy
# and 1.0-by-construction context metrics — that gap is the whole of #1485.
# The prior real measurement (2026-07-18, n=10, terra) was faithfulness 0.690
# over 3 context-bearing replays and answer_relevancy 0.401.
#
# HOW TO READ answer_relevancy HERE — it is mostly an ABSTENTION RATE.
# ragas multiplies the score by zero for any answer its judge calls
# noncommittal (ragas/metrics/_answer_relevance.py:127,
# `score = cosine_sim.mean() * int(not all_noncommittal)`). On this run 11 of
# 15 turns scored EXACTLY 0.000 because the pipeline declined to answer
# ("I could not verify X from the supplied materials"); the 4 turns that did
# commit averaged 0.670. So 0.179 ≈ (4/15) × 0.670 — it measures how often
# the pipeline commits, not how well it words a committed answer. Every one
# of the 10 zero-retrieval turns abstained, so the binding constraint is
# RETRIEVAL, not generation. Read a drop here as "the pipeline abstained more
# often", and check the retrieval hit rate alongside it.
#
# Thresholds sit ~1 standard error below the measured means. They are NOT
# aspirational quality targets and must not be read as "good" — this is the
# first honest number, deliberately recorded as a floor to regress against.
# Raising them is a product decision; lowering them to accommodate a red run
# is gate-shopping. Recalibrate only with a fresh measured run recorded here.
# ===========================================================================
# BASELINE 2 — MEASURED 2026-08-06, n=10, deployed dfd4a7a6 (#1489 close-out)
# ===========================================================================
# The same golden replay against the deployed stack, judged by the same frozen
# gpt-4o judge in the production container:
#
#     retrieval hit      3/10  = 0.300
#     faithfulness       1.000  (on the context rows the judge scored finitely)
#     answer_relevancy   0.140  (n=10)
#
# The #1488 fail-closed guard flagged q07/q08 as blocking (no finite
# faithfulness) — the hardening working on real traffic, not a regression.
#
# WHAT THE TWO BASELINES TOGETHER SHOWED, AND WHY THE GATE CHANGED (#1489)
# -----------------------------------------------------------------------
# answer_relevancy as reported is an aggregate over ALL rows, and every
# zero-retrieval row scores 0.0 — ragas zeroes a noncommittal answer, and every
# zero-retrieval turn in both baselines abstained. So it factorises exactly:
#
#     aggregate_relevancy = retrieval_hit_rate x hit_conditioned_relevancy
#
#     deployed   0.300  x 0.4667 = 0.1400   recorded 0.140
#     pre-deploy 0.3333 x 0.5360 = 0.1787   recorded 0.179
#
# One factor already has its own gate (MIN_RETRIEVAL_HIT_RATE). Gating the
# PRODUCT too means the generation half is judged through a moving lens: the
# aggregate floor 0.10 implied a hit-conditioned floor of 0.10/hit_rate, which
# is 0.333 at today's 0.30 hit rate but only 0.167 at 0.60. So:
#
#   * a run at the legal minimum hit rate (0.21) carrying the deployed
#     baseline's OWN generation quality (0.4667) computes 0.0980 and was
#     BLOCKED — the two gates that shipped together were not mutually
#     consistent; and
#   * once the hit rate exceeds 0.50 the implied floor drops below 0.20, so a
#     severe generation collapse starts passing BECAUSE retrieval improved.
#     At hit 0.30 -> 0.60 with hit-conditioned relevancy falling 0.510 -> 0.18,
#     the aggregate is 0.108 and passes the 0.10 floor while answers on rows
#     that did retrieve got 65% worse. Improving the hit rate is the project's
#     stated next move, so this window is about to open.
#
# So the gate now rides the two factors separately: MIN_RETRIEVAL_HIT_RATE for
# retrieval, MIN_HIT_CONDITIONED_RELEVANCY for generation, with the aggregate
# kept only as a total-collapse backstop pinned below their product so it can
# never contradict them.
#
# THE TRADEOFF, STATED PLAINLY: decoupling buys stability (a floor that means
# the same thing every run) and it must not be paid for with detection.
# MIN_HIT_CONDITIONED_RELEVANCY is therefore set at 0.34 — just above the
# strictest floor the aggregate ever implied at the measured hit rate
# (0.10/0.30 = 0.333) and stricter than it at every higher hit rate. The cost
# is a ~26% false-block rate at n_hit=3, which is the same cost the superseded
# gate was already paying at today's hit rate. See the constant for the full
# derivation and for why an earlier 0.20 revision of this lane was wrong.
#
# NOTE for anyone re-reading #1489's close-out: its deferral 7 asked to
# "recalibrate the 0.3 answer_relevancy gate to the measured baseline". That
# had already happened — commit 6709d42e replaced the aspirational 0.30 with
# the 0.10 measured floor and added the retrieval gate on 2026-08-05, the day
# before the close-out was written. Both recorded baselines pass the gate that
# actually ships. The confound above is the defect that remained.

# The two measured runs the floors below are calibrated from, machine-readable
# so a future recalibration argues with data rather than re-deriving it from
# prose. hit_row_relevancy_scores is what makes the aggregate reproducible.
RAGAS_MEASURED_BASELINES: Tuple[Dict[str, Any], ...] = (
    {
        "label": "pre-deploy-20260805",
        "measured_on": "2026-08-05",
        "deployed_sha": "9784abbd",
        "n_samples": 15,
        "n_hit": 5,
        "retrieval_hit_rate": 0.3333,
        "faithfulness": 0.524,
        "answer_relevancy": 0.179,
        "answer_relevancy_hit_conditioned": 0.5360,
        "hit_row_relevancy_scores": (0.670, 0.670, 0.670, 0.670, 0.0),
        "_method": (
            "11 of 15 turns scored exactly 0.000 and the 4 that committed averaged "
            "0.670; every zero-retrieval turn abstained, so all 4 committers are hit "
            "rows and the 5th hit row scored 0.0. Individual hit-row scores were not "
            "recorded separately, so the four are carried at their recorded mean — "
            "which reproduces the recorded aggregate 4 x 0.670 / 15 = 0.1787."
        ),
    },
    {
        "label": "deployed-20260806",
        "measured_on": "2026-08-06",
        "deployed_sha": "dfd4a7a6",
        "n_samples": 10,
        "n_hit": 3,
        "retrieval_hit_rate": 0.300,
        "faithfulness": 1.000,
        "answer_relevancy": 0.140,
        "answer_relevancy_hit_conditioned": 0.4667,
        "hit_row_relevancy_scores": (0.90, 0.50, 0.0),
        "_method": (
            "#1489's close-out recorded q07=0.90, q08=0.50 and aggregate 0.140 over "
            "n=10. 0.140 x 10 = 1.40 = 0.90 + 0.50, so the third hit row (q03) scored "
            "0.0 and every zero-retrieval row scored 0.0 — as the close-out states. "
            "Derived from the recorded aggregate, not from a re-run of the judge."
        ),
    },
)

REAL_PIPELINE_THRESHOLDS: Dict[str, float] = {
    # measured 0.524, SE 0.195 over only 5 context-bearing replays whose
    # values ranged 0.000-0.929 — a wide spread, so the floor is generous.
    "faithfulness": 0.35,
    # TOTAL-COLLAPSE BACKSTOP ONLY (#1489). This was 0.10, ~1 SE below the
    # pre-deploy aggregate — but the aggregate is the product of two factors
    # now gated separately, and at 0.10 it BLOCKED a run passing both of them
    # (hit 0.21 x conditioned 0.4667 = 0.0980). Pinned at or below
    # MIN_RETRIEVAL_HIT_RATE x MIN_HIT_CONDITIONED_RELEVANCY = 0.21 x 0.20 =
    # 0.042 so it can never contradict them, while still refusing a run whose
    # relevancy has gone to nothing. Detection power is deliberately delegated
    # to the two factor gates; a test pins the inequality.
    "answer_relevancy": 0.04,
}

# #1489 step 3 fixes the cadence at n≈10-15 (the CI OpenAI key throughput was
# the binding constraint in #504). Below 10 judged samples the aggregate is
# too noisy to carry a verdict, so the gate fails closed rather than reporting
# a confident number over 3 replays.
MIN_REAL_PIPELINE_SAMPLES = 10

# Retrieval-hit floor, measured baseline 5/15 = 0.333 with
# SE = sqrt(p(1-p)/n) = 0.122, so ~1 SE below is 0.211.
#
# Why retrieval needs its own gate: the metric gates cannot see a retrieval
# collapse. If only 3 of 15 turns retrieve and those 3 are judged well, then
# answer_relevancy = 3 x 0.670 / 15 = 0.134 (still above its 0.10 floor) and
# n_faithfulness = 3 (exactly at GATE_RAGAS_MIN_FAITHFULNESS_N). Retrieval
# would have dropped 40% with every block-level gate green — and since every
# zero-retrieval turn in the baseline abstained, that is the failure mode this
# pipeline actually has.
#
# 0.21 blocks 3/15 (0.200) and passes 4/15 (0.267).
MIN_RETRIEVAL_HIT_RATE = 0.21

# Generation quality on the rows that DID retrieve — the other factor of the
# aggregate, and the only one the retrieval gate cannot see.
#
# Measured: 0.4667 (n_hit=3, deployed) and 0.5360 (n_hit=5, pre-deploy);
# pooled 0.510 over 8 hit rows, SE 0.118.
#
# WHY 0.34 — it is calibrated to LOSE NO DETECTION relative to the gate it
# replaces. The superseded aggregate floor of 0.10 implied a hit-conditioned
# floor of 0.10/hit_rate: 0.333 at the measured 0.30 hit rate, 0.250 at 0.40,
# 0.167 at 0.60. 0.34 therefore sits just above the strictest value that gate
# ever implied at the measured operating point, and is strictly stricter at
# every higher hit rate — where the implied floor decayed toward zero.
# Independently, the module's own convention (~1 SE below the measured mean, as
# faithfulness 0.524/SE 0.195 -> 0.35 was set) applied to the POOLED baseline
# gives 0.510 - 0.118 = 0.392, so 0.34 is below that too. Both baselines pass
# it (0.4667 and 0.5360).
#
# DETECTION POWER, STATED HONESTLY:
#
#   * It blocks a sustained fall below 0.34 — a ~33% relative regression from
#     the pooled 0.510 baseline. It does NOT detect a 20% one (0.510 -> 0.408
#     passes).
#   * The cost is the false-block rate, and it is NOT new. At n_hit=3 the
#     run-level SE is 0.260 (from the deployed run's own 0.90/0.50/0.00), so a
#     run whose true quality is 0.510 reads below 0.34 about 26% of the time.
#     The superseded gate was already paying that: at hit 0.30 its implied
#     0.333 floor carried the same ~25%. The difference is that this floor
#     means the same thing every run instead of sliding with the hit rate.
#   * An earlier revision of this lane set 0.20 and documented the resulting
#     loss of detection as an accepted tradeoff. That was wrong: it bought a
#     ~13-point reduction in a false-block rate the shipped gate already paid,
#     with a real regression the old gate caught. Do not re-loosen it on that
#     argument.
#   * Its power still GROWS with the hit count — #1489's own conclusion that
#     the hit rate, not the judge thresholds, is the binding constraint.
#     Recalibrate when the hit rate improves, and record the run above.
#
# The eval is manual-only (since #504, CI OpenAI key throughput), so a false
# block costs a re-run and a look at the per-sample rows, not a red CI.
MIN_HIT_CONDITIONED_RELEVANCY = 0.34

# Heuristic-contamination refusal (``_ragas_heuristic_contamination_error``,
# re-exported above) lives in dspy_lane_ab beside its sibling validity checks:
# that module is stdlib-only because its source is embedded into the container
# bundle, and this module already imports from it, so the reverse direction
# would be circular. The DSPy A/B lane gates on the same predicate.


def contexts_from_evidence(evidence: Optional[Iterable[Any]]) -> List[str]:
    """Flatten a cognitive-RAG ``evidence`` list into context strings.

    Mirrors ``dspy_lane_ab.run_e2e_replays`` exactly: an evidence entry is a
    dict carrying ``content``; anything else is stringified rather than
    dropped, so an unrecognised shape shows up in the judged contexts instead
    of silently shrinking the denominator.
    """
    contexts: List[str] = []
    for item in evidence or []:
        content = item.get("content") if isinstance(item, dict) else None
        contexts.append(str(content) if content is not None else str(item))
    return contexts


def build_samples_from_replay(
    records: Sequence[Dict[str, Any]],
    model_label: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Adapt replay records into ``EvaluationSample`` dicts for the judge.

    Delegates to the proven ``build_ragas_samples`` so the drop-rules (errored
    runs and empty answers are excluded — judging an error string scores the
    failure mode, not the answer) and the empty-``contexts`` discipline live in
    exactly one place.

    ``records`` are what ``replay_golden_set.py --record-out`` writes: each
    carries ``query_id``, ``query``, ``response_text``, ``contexts`` and
    ``error``. ``model_label`` backfills provenance for records that predate
    the field.
    """
    prepared: List[Dict[str, Any]] = []
    for record in records:
        item = dict(record)
        if not item.get("model"):
            item["model"] = model_label or item.get("target") or "real-pipeline"
        prepared.append(item)
    golden_set = {"queries": [{"id": r["query_id"], "query": r["query"]} for r in prepared]}
    return build_ragas_samples({"records": prepared}, golden_set)


def summarize_retrieval(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Retrieval coverage over the replay, independent of any judge call.

    The retrieval-hit rate is the headline #1485 number (3/10 on the recorded
    2026-07-18 run): a gate that stays green through a 70% zero-retrieval rate
    is not measuring RAG quality. Errored replays count in the denominator —
    a turn that failed retrieved no evidence, and excluding it would flatter
    the rate.
    """
    total = len(records)
    errored = [r for r in records if r.get("error")]
    with_contexts = [r for r in records if not r.get("error") and (r.get("contexts") or [])]
    return {
        "n_records": total,
        "n_errors": len(errored),
        "n_with_contexts": len(with_contexts),
        "retrieval_hit_rate": (len(with_contexts) / total) if total else None,
    }


def check_retrieval_gate(
    retrieval: Optional[Dict[str, Any]],
    floor: float = MIN_RETRIEVAL_HIT_RATE,
) -> Tuple[bool, List[str]]:
    """Gate the replay's retrieval-hit rate. Returns ``(passed, failures)``.

    This reads the REPLAY summary, not the judge block: the judge only sees
    samples that survived the adapter's drop-rules, so its denominator excludes
    errored turns while ``summarize_retrieval`` counts them — a run where half
    the turns error out must not look like a healthy retrieval rate.

    Missing or malformed data fails closed: "we could not measure retrieval" is
    not "retrieval is fine".
    """
    if not isinstance(retrieval, dict) or not retrieval:
        return False, [
            "retrieval summary missing — cannot verify the pipeline retrieved anything; "
            "the gate fails closed"
        ]

    def _count(key: str) -> Optional[int]:
        value = retrieval.get(key)
        if not isinstance(value, int) or isinstance(value, bool):
            return None
        return value

    n_records = _count("n_records")
    n_with = _count("n_with_contexts")
    if n_records is None or n_records <= 0:
        return False, [
            f"n_records={retrieval.get('n_records')!r} is not a positive integer count "
            "— fails closed"
        ]
    if n_with is None or not (0 <= n_with <= n_records):
        return False, [
            f"n_with_contexts={retrieval.get('n_with_contexts')!r} is not an integer within "
            f"[0, {n_records}] — fails closed"
        ]

    rate = retrieval.get("retrieval_hit_rate")
    if not _is_finite_number(rate):
        return False, [f"retrieval_hit_rate={rate!r} is not a finite rate — fails closed"]
    if not (0.0 <= float(rate) <= 1.0):
        return False, [f"retrieval_hit_rate={float(rate)!r} is outside [0, 1] — fails closed"]
    # A summary whose rate no longer describes its own counts is stale or
    # hand-edited; the gate reads the rate, so the counts must corroborate it.
    expected = n_with / n_records
    if abs(expected - float(rate)) > 1e-6:
        return False, [
            f"retrieval_hit_rate={float(rate):.6f} but n_with_contexts/n_records = "
            f"{n_with}/{n_records} = {expected:.6f} — the summary does not describe "
            "its own counts; fails closed"
        ]

    if float(rate) < floor:
        return False, [
            f"retrieval hit rate {float(rate):.3f} < floor {floor} "
            f"({retrieval.get('n_with_contexts')}/{retrieval.get('n_records')} replays "
            "retrieved any context) — the metric gates cannot see this on their own"
        ]
    return True, []


def hit_conditioned_relevancy(
    block: Optional[Dict[str, Any]],
) -> Tuple[Optional[float], int]:
    """Mean ``answer_relevancy`` over the CONTEXT-BEARING rows, and that count.

    The reported aggregate averages over every row, and a zero-retrieval row
    scores 0.0 by construction (ragas zeroes a noncommittal answer, and every
    zero-retrieval turn in both measured baselines abstained). That makes the
    aggregate the product of two factors — see the baseline block above — and
    this is the factor the retrieval gate cannot see: how good the answers were
    on the turns that actually had evidence to work from.

    A hit row judged 0.0 is a real measurement and counts; it is precisely the
    generation failure this metric exists to expose. Returns ``(None, 0)`` when
    there is nothing to measure, so the caller decides what that means.
    """
    if not isinstance(block, dict):
        return None, 0
    rows = block.get("per_sample")
    if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows):
        return None, 0

    scores: List[float] = []
    for row in rows:
        n_ctx = row.get("n_contexts")
        if not isinstance(n_ctx, int) or isinstance(n_ctx, bool) or n_ctx <= 0:
            continue
        value = row.get("answer_relevancy")
        if not _is_finite_number(value):
            # A hit row the judge never scored would silently shrink the
            # denominator; refuse the whole measurement instead.
            return None, 0
        scores.append(float(value))

    if not scores:
        return None, 0
    return (sum(scores) / len(scores)), len(scores)


def check_hit_conditioned_relevancy_gate(
    block: Optional[Dict[str, Any]],
    floor: float = MIN_HIT_CONDITIONED_RELEVANCY,
) -> Tuple[bool, List[str]]:
    """Gate generation quality on the retrieved rows. Returns ``(passed, failures)``.

    Fails closed: a block whose hit rows cannot be read is not a passing run.
    See ``MIN_HIT_CONDITIONED_RELEVANCY`` for what this floor can and cannot
    detect — it is deliberately weak, and deliberately documented as such.
    """
    value, n_hit = hit_conditioned_relevancy(block)
    if value is None:
        return False, [
            "hit-conditioned answer_relevancy could not be measured (no context-bearing "
            "row carrying a finite score) — the gate fails closed"
        ]
    if value < floor:
        return False, [
            f"hit-conditioned answer_relevancy {value:.3f} < floor {floor} over {n_hit} "
            "context-bearing rows — answers on the turns that DID retrieve got worse; "
            "the aggregate cannot see this when the hit rate moves"
        ]
    return True, []


def check_run_gates(
    block: Optional[Dict[str, Any]],
    retrieval: Optional[Dict[str, Any]],
    thresholds: Optional[Dict[str, float]] = None,
    min_samples: int = MIN_REAL_PIPELINE_SAMPLES,
    retrieval_floor: float = MIN_RETRIEVAL_HIT_RATE,
    hit_conditioned_floor: float = MIN_HIT_CONDITIONED_RELEVANCY,
) -> Tuple[bool, List[str]]:
    """Full verdict for one run: judge-block gates, retrieval, and generation.

    This is what the driver calls. Keeping the retrieval gate out of
    ``check_real_pipeline_gates`` (which is about the judge's own output) but
    composing them here means a caller cannot accidentally gate the metrics
    while ignoring whether the pipeline retrieved anything — and, since #1489,
    cannot gate the aggregate while ignoring whether the answers on the rows it
    did retrieve were any good.
    """
    block_passed, failures = check_real_pipeline_gates(
        block, thresholds=thresholds, min_samples=min_samples
    )
    retrieval_passed, retrieval_failures = check_retrieval_gate(retrieval, floor=retrieval_floor)
    relevancy_passed, relevancy_failures = check_hit_conditioned_relevancy_gate(
        block, floor=hit_conditioned_floor
    )
    return (
        (block_passed and retrieval_passed and relevancy_passed),
        [*failures, *retrieval_failures, *relevancy_failures],
    )


def check_real_pipeline_gates(
    block: Optional[Dict[str, Any]],
    thresholds: Optional[Dict[str, float]] = None,
    min_samples: int = MIN_REAL_PIPELINE_SAMPLES,
    min_faithfulness_n: int = GATE_RAGAS_MIN_FAITHFULNESS_N,
) -> Tuple[bool, List[str]]:
    """Gate a judge-output block fail-closed. Returns ``(passed, failures)``.

    Fail-closed means every uncertainty blocks: a missing metric, a non-finite
    score, an aggregate that no longer describes its own rows, a row the judge
    never scored, too few samples, or too thin a faithfulness denominator all
    FAIL. A gate that cannot see the measurement must not report success —
    that is how the fixture eval stayed green while production sat at 0.401.

    ``block`` is the ``RESULTS_JSON`` payload emitted by
    ``scripts/run_dspy_lane_ragas_judge.py``.
    """
    active = dict(REAL_PIPELINE_THRESHOLDS if thresholds is None else thresholds)
    unsupported = sorted(m for m in active if m not in REAL_PIPELINE_METRICS)
    if unsupported:
        raise ValueError(
            f"unsupported gate metrics {unsupported}: the real-pipeline gate scores only "
            f"{list(REAL_PIPELINE_METRICS)}. context_precision/context_recall need a "
            "ground-truth reference the replay deliberately does not fabricate (#1485)."
        )

    if not isinstance(block, dict) or not block:
        return False, ["judge output is missing or empty — the gate fails closed"]

    failures: List[str] = []

    consistency = _ragas_consistency_error(block)
    if consistency:
        failures.append(f"ragas consistency: {consistency}")

    scoreless = _ragas_scoreless_error(block)
    if scoreless:
        failures.append(f"ragas scoreless rows: {scoreless}")

    contaminated = _ragas_heuristic_contamination_error(block)
    if contaminated:
        failures.append(contaminated)

    n_samples = block.get("n_samples")
    if not isinstance(n_samples, int) or isinstance(n_samples, bool) or n_samples < min_samples:
        failures.append(
            f"n_samples={n_samples!r} below the minimum {min_samples} judged samples "
            "— too few to carry a verdict"
        )

    n_faithfulness = block.get("n_faithfulness")
    if (
        not isinstance(n_faithfulness, int)
        or isinstance(n_faithfulness, bool)
        or n_faithfulness < min_faithfulness_n
    ):
        failures.append(
            f"faithfulness denominator n_faithfulness={n_faithfulness!r} below the floor "
            f"{min_faithfulness_n} — a mean over 1-2 context-bearing replays is noise"
        )

    for metric in sorted(active):
        value = block.get(metric)
        if not _is_finite_number(value):
            failures.append(f"{metric}={value!r} is not a finite score — fails closed")
            continue
        if float(value) < active[metric]:
            failures.append(f"{metric}={float(value):.3f} < threshold {active[metric]}")

    return (not failures), failures
