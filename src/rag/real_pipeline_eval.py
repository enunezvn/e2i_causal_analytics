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
    _is_finite_number,
    _ragas_consistency_error,
    _ragas_scoreless_error,
    build_ragas_samples,
)

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
REAL_PIPELINE_THRESHOLDS: Dict[str, float] = {
    # measured 0.524, SE 0.195 over only 5 context-bearing replays whose
    # values ranged 0.000-0.929 — a wide spread, so the floor is generous.
    "faithfulness": 0.35,
    # measured 0.179, SE 0.083. At this floor, abstention rising from 11/15
    # to 13/15 (AR ≈ 0.089) blocks.
    "answer_relevancy": 0.10,
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

# RAGASEvaluator._evaluate_with_ragas ends in a broad
# ``except Exception: return await self._evaluate_with_fallback(...)``
# (src/rag/evaluation.py:1188), so a quota error, rate limit, or transient API
# failure DURING judging silently swaps gpt-4o judgments for word-overlap
# heuristics on that sample. The judge process still exits 0 and the
# aggregates still reconcile, so without this marker the gate could pass on
# numbers no judge ever produced — the same masquerade class as the fixture
# tautology itself.
#
# Only the FALLBACK path is stamped (:1270). The judged path returns
# ``metadata=sample.metadata`` with no positive marker, so a judged row
# legitimately carries no key and "missing" must mean judged. Requiring a
# positive marker would be stronger, but it needs a change to the evaluator
# AND a deploy before this gate could ever pass — the container judges with
# deployed code, not this worktree. Refusing the known-bad stamp is the
# contract that works against the code that is actually running.
#
# The same vocabulary already distinguishes real from heuristic scores in
# src/agents/feedback_learner/evaluation/models.py.
HEURISTIC_EVALUATION_METHOD = "fallback_heuristic"


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


def _heuristic_contamination_error(block: Dict[str, Any]) -> Optional[str]:
    """Refuse a block whose rows were scored by the heuristic fallback.

    See ``HEURISTIC_EVALUATION_METHOD``: a mid-run judge failure degrades that
    sample to word-overlap scoring without failing the process. A row is
    accepted when its ``evaluation_method`` is absent or None (the judged path
    stamps nothing); ANY other value is refused, so a future marker we do not
    recognise blocks rather than slips through.
    """
    rows = block.get("per_sample")
    if not isinstance(rows, list):
        return None  # shape problems are already reported by the consistency gate
    contaminated = [
        row.get("query_id")
        for row in rows
        if isinstance(row, dict) and row.get("evaluation_method") not in (None, "")
    ]
    if contaminated:
        return (
            f"rows not scored by the gpt-4o judge (evaluation_method set): {contaminated} "
            f"— a mid-run judge failure degrades samples to {HEURISTIC_EVALUATION_METHOD} "
            "word-overlap scoring (src/rag/evaluation.py:1188); those numbers are not "
            "measurements"
        )
    return None


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
    rate = retrieval.get("retrieval_hit_rate")
    if not _is_finite_number(rate):
        return False, [f"retrieval_hit_rate={rate!r} is not a finite rate — fails closed"]
    if float(rate) < floor:
        return False, [
            f"retrieval hit rate {float(rate):.3f} < floor {floor} "
            f"({retrieval.get('n_with_contexts')}/{retrieval.get('n_records')} replays "
            "retrieved any context) — the metric gates cannot see this on their own"
        ]
    return True, []


def check_run_gates(
    block: Optional[Dict[str, Any]],
    retrieval: Optional[Dict[str, Any]],
    thresholds: Optional[Dict[str, float]] = None,
    min_samples: int = MIN_REAL_PIPELINE_SAMPLES,
    retrieval_floor: float = MIN_RETRIEVAL_HIT_RATE,
) -> Tuple[bool, List[str]]:
    """Full verdict for one run: judge-block gates AND the retrieval gate.

    This is what the driver calls. Keeping the retrieval gate out of
    ``check_real_pipeline_gates`` (which is about the judge's own output) but
    composing both here means a caller cannot accidentally gate the metrics
    while ignoring whether the pipeline retrieved anything.
    """
    block_passed, failures = check_real_pipeline_gates(
        block, thresholds=thresholds, min_samples=min_samples
    )
    retrieval_passed, retrieval_failures = check_retrieval_gate(retrieval, floor=retrieval_floor)
    return (block_passed and retrieval_passed), [*failures, *retrieval_failures]


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

    contaminated = _heuristic_contamination_error(block)
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
