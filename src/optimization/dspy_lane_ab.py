"""DSPy-lane provider A/B harness logic.

Compares candidate LMs against the production baseline on the two real intent
surfaces that ride ``DSPY_LM_MODEL`` (cognitive RAG intent classification and
the chatbot intent classifier), plus end-to-end latency/error telemetry, and
evaluates the pre-registered decision gates from
``.claude/plans/dspy_lane_anthropic_flip_plan.md``.

This module keeps aggregation and gate logic import-light (stdlib only) so it
is unit-testable without dspy installed. The functions that perform real LLM
calls (``run_signature_ab``) import dspy lazily; they are executed inside the
prod container via a self-contained bundle produced by
``emit_container_script`` (the container rootfs is read-only and runs deployed
code, so the bundle embeds this module's source plus the golden set and is
piped over stdin).

Scoring is case-sensitive by default because production consumers compare
intent strings exactly (``cognitive_rag_dspy.py:576``, ``chatbot_dspy.py:2062``);
a wrong-case answer would misbehave in production even if semantically right.
The chatbot comparison happens AFTER ``_normalize_intent`` at
``chatbot_dspy.py:615``, so chatbot predictions are normalized through the
same function before exact scoring (``production_chatbot_intent``); the
cognitive consumer reads ``primary_intent`` raw, so cognitive stays raw.
"""

from __future__ import annotations

import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypeGuard

# Pre-registered gate parameters (plan section 5). Changing these after seeing
# results would be gate-shopping; they are constants on purpose.
GATE_ACCURACY_MARGIN = 0.05
GATE_RAGAS_MARGIN = 0.05
GATE_E2E_LATENCY_FACTOR = 1.5
# Floor on faithfulness denominators: a mean over 1-2 context-bearing replays
# is noise, not signal. Added post-run (codex iter-2) as an add-only guard; 3
# equals the completed 20260718 run's baseline coverage, so it binds future
# reruns without re-adjudicating that verdict.
GATE_RAGAS_MIN_FAITHFULNESS_N = 3

_EPS = 1e-9

# The stamp RAGASEvaluator._evaluate_with_fallback writes into result metadata
# (src/rag/evaluation.py:1270) when a sample was scored by word-overlap
# heuristics instead of the gpt-4o judge. See
# _ragas_heuristic_contamination_error. The same vocabulary distinguishes real
# from heuristic scores in src/agents/feedback_learner/evaluation/models.py.
HEURISTIC_EVALUATION_METHOD = "fallback_heuristic"

# Float slack when recomputing a judge aggregate from its own per_sample rows
# (same values, same summation order as the judge script - anything beyond
# repr/JSON round-trip noise means the aggregate does not describe the rows).
_RAGAS_CONSISTENCY_TOL = 1e-6

# The two real intent surfaces the harness measures. The gate loop iterates
# this constant rather than whatever keys the baseline happens to carry: an
# empty or partial baseline signature block must FAIL its gates, not silently
# drop them from the verdict (codex iter-6).
GATE_SIGNATURE_TAXONOMIES = ("cognitive_rag", "chatbot")

# Golden-set label key per taxonomy: a null label means the query is excluded
# from that taxonomy, and that status is a static fixture property, never
# model-dependent.
_TAXONOMY_LABEL_KEYS = {"cognitive_rag": "expected_cognitive", "chatbot": "expected_chatbot"}


# ---------------------------------------------------------------------------
# Golden set
# ---------------------------------------------------------------------------


def load_golden_set(path: str | Path) -> Dict[str, Any]:
    """Load and minimally validate the golden-set fixture."""
    golden: Dict[str, Any] = json.loads(Path(path).read_text())
    if not golden.get("queries"):
        raise ValueError(f"golden set at {path} has no queries")
    for item in golden["queries"]:
        missing = {"id", "query", "expected_cognitive", "expected_chatbot"} - set(item)
        if missing:
            raise ValueError(f"golden item {item.get('id')} missing keys: {missing}")
    return golden


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _lenient_form(value: str) -> str:
    return value.strip().strip("\"'.` ").lower()


def is_correct(
    predicted: Optional[str],
    acceptable: Optional[List[str]],
    lenient: bool = False,
) -> Optional[bool]:
    """Score one prediction against an acceptable-set label.

    Returns None when the query is excluded from this taxonomy
    (``acceptable is None``). A missing prediction (parse failure) is wrong.
    """
    if acceptable is None:
        return None
    if predicted is None:
        return False
    if lenient:
        return _lenient_form(predicted) in {_lenient_form(a) for a in acceptable}
    return predicted.strip() in acceptable


def _error_class(error: Optional[str]) -> Optional[str]:
    if not error:
        return None
    return error.split(":", 1)[0].strip()


def production_chatbot_intent(raw: Optional[str]) -> Optional[str]:
    """Map a raw ChatbotIntentClassifier prediction to the value production
    routes on.

    The prod path (chatbot_dspy.classify_intent_dspy) passes the DSPy output
    through ``_normalize_intent`` before any consumer sees it, so scoring the
    raw string would count casing/alias variants production accepts (e.g.
    ``"KPI_QUERY"``) as wrong (codex iter-5). The cognitive taxonomy is
    scored raw on purpose: its production consumer reads ``primary_intent``
    unnormalized. Parse failures stay None.
    """
    if raw is None:
        return None
    # Deferred: keeps this module import-light; resolvable wherever the
    # signature runs actually execute (the prod container has src on path).
    from src.api.routes.chatbot_dspy import _normalize_intent

    return _normalize_intent(str(raw))


def _percentile(values: List[float], p: float) -> Optional[float]:
    """Nearest-rank percentile; None on empty input."""
    if not values:
        return None
    ordered = sorted(values)
    rank = min(len(ordered) - 1, max(0, math.ceil(p * len(ordered)) - 1))
    return ordered[rank]


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def summarize_signature_runs(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Aggregate raw signature-level records into per-model, per-taxonomy metrics.

    Excluded records (``acceptable is None``) contribute latency (the call
    really happened) but not accuracy/parse metrics.
    """
    grouped: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for rec in records:
        grouped.setdefault(rec["model"], {}).setdefault(rec["taxonomy"], []).append(rec)

    out: Dict[str, Dict[str, Any]] = {}
    for model, taxonomies in grouped.items():
        out[model] = {}
        for taxonomy, recs in taxonomies.items():
            scored = [r for r in recs if r["acceptable"] is not None]
            strict = [is_correct(r["predicted"], r["acceptable"]) for r in scored]
            lenient = [is_correct(r["predicted"], r["acceptable"], lenient=True) for r in scored]
            parse_failures = [r for r in scored if r["predicted"] is None]
            latencies = [r["latency_s"] for r in recs if r.get("latency_s") is not None]
            errors = Counter(cls for cls in (_error_class(r.get("error")) for r in recs) if cls)
            n_scored = len(scored)
            out[model][taxonomy] = {
                "n_scored": n_scored,
                "n_excluded": len(recs) - n_scored,
                # Sorted multisets so the query_set gate can prove both sides
                # measured the SAME queries - counts alone cannot (codex
                # iter-6).
                "scored_query_ids": sorted(r["query_id"] for r in scored),
                "excluded_query_ids": sorted(
                    r["query_id"] for r in recs if r["acceptable"] is None
                ),
                "accuracy_strict": (sum(1 for v in strict if v) / n_scored) if n_scored else None,
                "accuracy_lenient": (sum(1 for v in lenient if v) / n_scored) if n_scored else None,
                "parse_failure_rate": (len(parse_failures) / n_scored) if n_scored else None,
                "latency_p50": _percentile(latencies, 0.50),
                "latency_p95": _percentile(latencies, 0.95),
                "error_classes": dict(errors),
            }
    return out


def summarize_e2e_runs(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate end-to-end RAG replay records into per-model metrics."""
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for rec in records:
        grouped.setdefault(rec["model"], []).append(rec)

    out: Dict[str, Any] = {}
    for model, recs in grouped.items():
        ok = [r for r in recs if not r.get("error")]
        latencies = [r["latency_s"] for r in recs if r.get("latency_s") is not None]
        errors = Counter(cls for cls in (_error_class(r.get("error")) for r in recs) if cls)
        out[model] = {
            # The block's own identity, so the replay_provenance gate can
            # prove these latencies belong to the model being verdicted
            # (codex iter-10).
            "model": model,
            "n": len(recs),
            "n_errors": len(recs) - len(ok),
            # Sorted multiset so the replay_anchor gate can prove which
            # replays produced these latencies - a count alone cannot (codex
            # iter-9).
            "query_ids": sorted(r["query_id"] for r in recs),
            "latency_p50": _percentile(latencies, 0.50),
            "latency_p95": _percentile(latencies, 0.95),
            "mean_hops": (sum(r["hop_count"] for r in ok) / len(ok)) if ok else None,
            "mean_evidence": (sum(r["evidence_count"] for r in ok) / len(ok)) if ok else None,
            "mean_answer_chars": (sum(r["answer_chars"] for r in ok) / len(ok)) if ok else None,
            "error_classes": dict(errors),
        }
    return out


# ---------------------------------------------------------------------------
# Pre-registered decision gates (plan section 5)
# ---------------------------------------------------------------------------


def _is_finite_number(value: Any) -> TypeGuard[float]:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _ragas_consistency_error(block: Dict[str, Any]) -> Optional[str]:
    """Reconcile a RAGAS block's reported aggregates against its own
    per_sample rows, mirroring exactly how the judge script computes them
    (n_samples = all rows; n_faithfulness = rows with contexts; faithfulness =
    mean over non-None context-bearing scores; answer_relevancy = mean over
    all non-None scores). Returns the first mismatch found, or None. A stale,
    hand-edited, or partially merged block whose aggregates no longer describe
    its rows must fail closed (codex iter-3)."""
    rows = block.get("per_sample")
    if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows):
        return "per_sample rows missing or malformed"
    ctx_rows: List[Dict[str, Any]] = []
    for row in rows:
        n_ctx = row.get("n_contexts")
        if not isinstance(n_ctx, int) or isinstance(n_ctx, bool) or n_ctx < 0:
            return f"row {row.get('query_id')!r} has invalid n_contexts {n_ctx!r}"
        if n_ctx > 0:
            ctx_rows.append(row)
    if block.get("n_samples") != len(rows):
        return f"n_samples={block.get('n_samples')!r} but per_sample has {len(rows)} rows"
    if block.get("n_faithfulness") != len(ctx_rows):
        return (
            f"n_faithfulness={block.get('n_faithfulness')!r} but per_sample "
            f"has {len(ctx_rows)} context-bearing rows"
        )
    for metric, subset in (("faithfulness", ctx_rows), ("answer_relevancy", rows)):
        vals: List[float] = []
        for row in subset:
            value = row.get(metric)
            if value is None:
                continue
            if not _is_finite_number(value):
                return f"row {row.get('query_id')!r} has non-finite {metric} {value!r}"
            vals.append(float(value))
        recomputed = (sum(vals) / len(vals)) if vals else None
        reported = block.get(metric)
        if recomputed is None and reported is None:
            continue
        if recomputed is None or reported is None or not _is_finite_number(reported):
            return f"{metric}={reported!r} but per_sample recomputes to {recomputed!r}"
        if abs(recomputed - float(reported)) > _RAGAS_CONSISTENCY_TOL:
            return f"{metric}={float(reported):.6f} but per_sample recomputes to {recomputed:.6f}"
    return None


def _ragas_scoreless_error(block: Dict[str, Any]) -> Optional[str]:
    """A row the judge covered but never scored is a silent hole in the
    aggregates: n_faithfulness counts every context-bearing row while the
    faithfulness mean (and the consistency recompute that mirrors it) skips
    None scores, so a block can wear n=10 coverage over an n=3 measurement
    without any aggregate mismatch (codex iter-4). Requires a finite
    faithfulness on every context-bearing row and a finite answer_relevancy on
    every row. Rows with no retrieved contexts are legitimately scoreless for
    faithfulness and are not flagged. The real 20260718 blocks have zero
    scoreless rows, so this gate is add-only on the recorded run."""
    rows = block.get("per_sample")
    if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows):
        return "per_sample rows missing or malformed"
    faith_scoreless = [
        row.get("query_id")
        for row in rows
        if isinstance(row.get("n_contexts"), int)
        and not isinstance(row.get("n_contexts"), bool)
        and row["n_contexts"] > 0
        and not _is_finite_number(row.get("faithfulness"))
    ]
    if faith_scoreless:
        return f"context-bearing rows without a finite faithfulness score: {faith_scoreless}"
    rel_scoreless = [
        row.get("query_id") for row in rows if not _is_finite_number(row.get("answer_relevancy"))
    ]
    if rel_scoreless:
        return f"rows without a finite answer_relevancy score: {rel_scoreless}"
    return None


def _ragas_heuristic_contamination_error(block: Dict[str, Any]) -> Optional[str]:
    """A row the judge did not actually judge is not a measurement (#1485).

    ``RAGASEvaluator._evaluate_with_ragas`` ends in a broad
    ``except Exception: return await self._evaluate_with_fallback(...)``
    (src/rag/evaluation.py:1188), so a quota error, rate limit, or transient
    API failure DURING judging silently swaps gpt-4o judgments for
    word-overlap heuristics on that sample. The judge process still exits 0
    and the aggregates still reconcile against their rows, so neither
    ``_ragas_consistency_error`` nor ``_ragas_scoreless_error`` can see it.
    Confirmed against the running container: a keyless judge run returned
    faithfulness 0.125 with perfectly self-consistent aggregates.

    Only the FALLBACK path is stamped (``evaluation_method="fallback_heuristic"``
    at :1270); the judged path returns ``metadata=sample.metadata`` with no
    positive marker, so an absent or None value means judged. ANY other value
    is refused, so an unrecognised future marker blocks rather than slips
    through. Blocks recorded before the judge carried the stamp (the 20260718
    run) have no such key, making this add-only on them.

    This lives here rather than in ``src/rag/real_pipeline_eval.py`` because
    this module is deliberately stdlib-only — its source is embedded into the
    container bundle by ``emit_container_script`` — and real_pipeline_eval
    already imports from it, so the reverse direction would be circular.
    """
    rows = block.get("per_sample")
    if not isinstance(rows, list):
        return None  # shape problems are reported by the consistency gate
    contaminated = [
        row.get("query_id")
        for row in rows
        if isinstance(row, dict) and row.get("evaluation_method") is not None
    ]
    if contaminated:
        return (
            f"rows not scored by the gpt-4o judge (evaluation_method set): {contaminated} "
            f"— a mid-run judge failure degrades samples to {HEURISTIC_EVALUATION_METHOD} "
            "word-overlap scoring (src/rag/evaluation.py:1188); those numbers are not "
            "measurements"
        )
    return None


def _common_subset_faithfulness(
    b_rows: Optional[List[Dict[str, Any]]],
    c_rows: Optional[List[Dict[str, Any]]],
) -> Tuple[Optional[float], Optional[float], int]:
    """Mean faithfulness for each side over the replays BOTH sides retrieved
    contexts for - the only apples-to-apples faithfulness comparison when
    context capture differs by model. Returns (baseline_mean, candidate_mean,
    n_common); (None, None, 0) when there is no usable overlap."""

    def _by_id(rows: Optional[List[Dict[str, Any]]]) -> Dict[Any, float]:
        out: Dict[Any, float] = {}
        for row in rows or []:
            if not isinstance(row, dict):
                continue
            qid = row.get("query_id")
            n_ctx = row.get("n_contexts")
            if (
                qid is None
                or not isinstance(n_ctx, int)
                or isinstance(n_ctx, bool)
                or n_ctx <= 0
                or not _is_finite_number(row.get("faithfulness"))
            ):
                continue
            out[qid] = float(row["faithfulness"])
        return out

    b_by_id = _by_id(b_rows)
    c_by_id = _by_id(c_rows)
    common = sorted(set(b_by_id) & set(c_by_id))
    if not common:
        return None, None, 0
    return (
        sum(b_by_id[q] for q in common) / len(common),
        sum(c_by_id[q] for q in common) / len(common),
        len(common),
    )


def _gate(name: str, passed: bool, detail: str) -> Dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def _all_error_classes(bundle: Dict[str, Any]) -> set:
    classes: set = set()
    for tax_metrics in (bundle.get("signature") or {}).values():
        classes |= set((tax_metrics.get("error_classes") or {}))
    classes |= set(((bundle.get("e2e") or {}).get("error_classes") or {}))
    return classes


def rebind_acceptable_labels(
    records: List[Dict[str, Any]], golden: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Replace every record's ``acceptable`` label with the golden fixture's.

    The golden-anchor gate validates which queries were measured, but scoring
    trusted each record's own label - so a results file keeping the exact
    golden query ids while widening one hard query's acceptable set to
    include its wrong prediction passed every gate, and a bundle emitted
    from a stale golden file scored against outdated labels (codex iter-8).
    Rebinding makes the fixture the single source of truth for labels, the
    same trust move analyze already makes for rates (records over stored
    summary). A record whose (query_id, taxonomy) the fixture does not know
    fails loud. Input records are not mutated.
    """
    labels: Dict[Tuple[str, str], Optional[List[str]]] = {}
    for item in golden["queries"]:
        for taxonomy, label_key in _TAXONOMY_LABEL_KEYS.items():
            labels[(item["id"], taxonomy)] = item[label_key]
    rebound = []
    for rec in records:
        key = (rec["query_id"], rec["taxonomy"])
        if key not in labels:
            raise ValueError(
                f"record references (query_id, taxonomy) {key} that the golden set "
                "does not define - cannot rebind its acceptable label"
            )
        rebound.append({**rec, "acceptable": labels[key]})
    return rebound


def stored_summary_divergences(stored: object, recomputed: Dict[str, Any]) -> List[str]:
    """Cross-check a results file's stored summary block against the
    records-recomputed one, comparing only keys the stored block carries (so
    additive fields newer summarize versions emit never trigger on old
    files). The verdict never uses the stored block, but a stored aggregate
    contradicting its own records means tampering, a runner bug, or a
    label-set change - analyze treats any divergence as a hard failure
    (codex iter-8). An absent or malformed stored block has nothing to
    cross-check and reports no divergence.
    """
    divergences: List[str] = []
    if not isinstance(stored, dict):
        return divergences
    for model, taxonomies in stored.items():
        if not isinstance(taxonomies, dict):
            continue
        for taxonomy, block in taxonomies.items():
            if not isinstance(block, dict):
                continue
            recomputed_block = recomputed.get(model, {}).get(taxonomy, {})
            for key, value in block.items():
                if key in recomputed_block and recomputed_block[key] != value:
                    divergences.append(
                        f"{model}/{taxonomy}/{key}: stored {value!r} vs "
                        f"recomputed {recomputed_block[key]!r}"
                    )
    return divergences


def expected_signature_sets(golden: Dict[str, Any]) -> Dict[str, Dict[str, List[str]]]:
    """Derive, per taxonomy, which golden queries MUST appear scored and which
    excluded. Every side-to-side gate can be satisfied by two sides sharing
    the same truncated subset (codex iter-7); this is the external anchor the
    signature_golden_anchor gate holds both sides against."""
    out: Dict[str, Dict[str, List[str]]] = {}
    for taxonomy, label_key in _TAXONOMY_LABEL_KEYS.items():
        out[taxonomy] = {
            "scored_query_ids": sorted(
                item["id"] for item in golden["queries"] if item[label_key] is not None
            ),
            "excluded_query_ids": sorted(
                item["id"] for item in golden["queries"] if item[label_key] is None
            ),
        }
    return out


def _replay_anchor_error(side: Dict[str, Any], expected_sorted: List[str]) -> Optional[str]:
    """One side's replay-identity check against the intended replay set.

    The RAGAS ``per_sample`` rows are the score-bearing identity: their
    query-id multiset must equal the intended replay multiset exactly, so a
    duplicated easy replay hiding a dropped hard one fails even though every
    count stays intact (codex iter-9). The e2e ``query_ids`` field is checked
    strictly whenever the block carries it; blocks recorded before the field
    existed carry only ``n``, and a latency number's provenance is
    unverifiable without raw replay records either way, so its absence is
    tolerated as legacy rather than fabricating a FAIL on genuine pre-field
    data.
    """
    ragas = side.get("ragas") or {}
    rows = ragas.get("per_sample")
    if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows):
        return "per_sample rows missing or malformed"
    ids = [r.get("query_id") for r in rows]
    if not all(isinstance(q, str) for q in ids):
        return "per_sample rows carry non-string query ids"
    if sorted(ids) != expected_sorted:
        return f"per_sample query-id multiset {sorted(ids)} != expected replay set"
    e2e_ids = (side.get("e2e") or {}).get("query_ids")
    if e2e_ids is not None:
        if not isinstance(e2e_ids, list) or not all(isinstance(q, str) for q in e2e_ids):
            return "e2e query_ids malformed"
        if sorted(e2e_ids) != expected_sorted:
            return f"e2e query-id multiset {sorted(e2e_ids)} != expected replay set"
    return None


def _replay_provenance_error(
    side: Dict[str, Any], allow_legacy_e2e: bool, allow_absent_ragas_model: bool
) -> Tuple[Optional[str], Optional[str]]:
    """(error, acceptance_note) for one side's replay-block ownership.

    A present-but-mismatched identity is affirmative evidence the block
    belongs to a different model - it always fails, overrides or not. The
    two absence cases are NOT the same condition and take separate overrides
    (codex iter-11): e2e blocks recorded before the model/query_ids fields
    existed genuinely have no identity to check (legacy), but the RAGAS
    judge has ALWAYS emitted ``model``, so an absent ragas.model means the
    block was stripped or hand-assembled - accepting it is a deliberate
    attestation that ownership was verified out-of-band (run logs), never
    something the e2e legacy flag covers. Every acceptance is surfaced in
    the gate detail rather than being silent (codex iter-10).
    """
    expected = side.get("model")
    if not isinstance(expected, str) or not expected:
        return "bundle missing model identity (fail-closed)", None
    notes: List[str] = []
    ragas_model = (side.get("ragas") or {}).get("model")
    if ragas_model is None:
        if not allow_absent_ragas_model:
            return (
                "ragas.model absent - the judge always emits it; "
                "--allow-absent-ragas-model attests out-of-band ownership",
                None,
            )
        notes.append("ragas.model absence attested by override")
    elif ragas_model != expected:
        return f"ragas block belongs to {ragas_model!r}, not {expected!r}", None
    e2e = side.get("e2e") or {}
    legacy: List[str] = []
    e2e_model = e2e.get("model")
    if e2e_model is None:
        legacy.append("e2e.model")
    elif e2e_model != expected:
        return f"e2e block belongs to {e2e_model!r}, not {expected!r}", None
    if e2e.get("query_ids") is None:
        legacy.append("e2e.query_ids")
    if legacy:
        fields = ", ".join(legacy)
        if not allow_legacy_e2e:
            return (
                f"unverified legacy e2e provenance ({fields} absent; "
                "--allow-legacy-replay-provenance accepts it explicitly)",
                None,
            )
        notes.append(f"legacy e2e provenance accepted by override ({fields} absent)")
    return None, "; ".join(notes) if notes else None


def _query_sets_match(b: Dict[str, Any], c: Dict[str, Any]) -> bool:
    """True only when both signature blocks carry well-formed query-id lists
    that agree with their own counts and match each other as multisets
    (sorted-list comparison catches duplicated-plus-dropped ids that keep the
    count intact)."""
    for blk in (b, c):
        for ids_key, n_key in (
            ("scored_query_ids", "n_scored"),
            ("excluded_query_ids", "n_excluded"),
        ):
            ids = blk.get(ids_key)
            if not isinstance(ids, list) or not all(isinstance(q, str) for q in ids):
                return False
            if len(ids) != blk.get(n_key):
                return False
    return sorted(b["scored_query_ids"]) == sorted(c["scored_query_ids"]) and sorted(
        b["excluded_query_ids"]
    ) == sorted(c["excluded_query_ids"])


def evaluate_gates(
    baseline: Dict[str, Any],
    candidate: Dict[str, Any],
    expected_signature: Optional[Dict[str, Dict[str, List[str]]]] = None,
    expected_replay_ids: Optional[List[str]] = None,
    allow_legacy_replay_provenance: bool = False,
    allow_absent_ragas_model: bool = False,
) -> Dict[str, Any]:
    """Evaluate one candidate against the baseline on the five hard gates.

    Missing measurement blocks (e.g. no RAGAS scores) fail closed: a gate
    cannot pass on absent data. ``expected_signature`` is the golden-set
    anchor from ``expected_signature_sets`` and ``expected_replay_ids`` the
    intended e2e replay multiset; omitting either fails its anchor gates
    rather than skipping them (codex iter-7/iter-9).
    """
    gates: List[Dict[str, Any]] = []

    baseline_sig = baseline.get("signature") or {}
    candidate_sig = candidate.get("signature") or {}
    for taxonomy in GATE_SIGNATURE_TAXONOMIES:
        b = baseline_sig.get(taxonomy)
        c = candidate_sig.get(taxonomy)
        if b is None or c is None:
            missing = " and ".join(
                side for side, blk in (("baseline", b), ("candidate", c)) if blk is None
            )
            gates.append(
                _gate(f"signature[{taxonomy}]", False, f"missing {missing} data (fail-closed)")
            )
            continue
        # Rates are only comparable over the same golden-set denominator: a
        # truncated or partially merged summary (n_scored=1, accuracy 1.0)
        # would otherwise beat a full-set baseline on every rate gate (codex
        # iter-5). Counts must be real ints and match exactly, fail-closed.
        b_n = (b.get("n_scored"), b.get("n_excluded"))
        c_n = (c.get("n_scored"), c.get("n_excluded"))
        counts_valid = all(isinstance(v, int) and not isinstance(v, bool) for v in (*b_n, *c_n))
        gates.append(
            _gate(
                f"signature_denominator[{taxonomy}]",
                counts_valid and b_n == c_n,
                f"baseline (n_scored, n_excluded) {b_n} vs candidate {c_n} (must match exactly)",
            )
        )
        # Equal counts cannot prove equal queries: a merged or hand-edited
        # summary after a partial run can swap a hard query for an easy one
        # while preserving (n_scored, n_excluded) (codex iter-6). The scored
        # and excluded query-id multisets must match exactly, and each side's
        # lists must agree with its own counts; the analyze CLI recomputes
        # summaries from raw records so these fields cannot be forged
        # independently of the records.
        gates.append(
            _gate(
                f"signature_query_set[{taxonomy}]",
                _query_sets_match(b, c),
                "scored/excluded query-id multisets must match baseline and "
                "agree with counts (fail-closed)",
            )
        )
        # Side-to-side equality cannot catch BOTH sides sharing the same
        # truncated or cherry-picked subset (codex iter-7) - each side must
        # also match what the golden set says should have been measured,
        # where scored/excluded is a static fixture property.
        exp = (expected_signature or {}).get(taxonomy)
        if exp is None:
            gates.append(
                _gate(
                    f"signature_golden_anchor[{taxonomy}]",
                    False,
                    "missing expected golden query sets (fail-closed)",
                )
            )
        else:
            anchored = all(
                isinstance(blk.get(key), list) and sorted(blk[key]) == exp[key]
                for blk in (b, c)
                for key in ("scored_query_ids", "excluded_query_ids")
            )
            gates.append(
                _gate(
                    f"signature_golden_anchor[{taxonomy}]",
                    anchored,
                    "scored/excluded query-id multisets must equal the golden "
                    "set's expected sets, both sides (fail-closed)",
                )
            )
        b_parse, c_parse = b.get("parse_failure_rate"), c.get("parse_failure_rate")
        if _is_finite_number(b_parse) and _is_finite_number(c_parse):
            gates.append(
                _gate(
                    f"parse_failure[{taxonomy}]",
                    c_parse <= b_parse + _EPS,
                    f"candidate {c_parse:.3f} vs baseline {b_parse:.3f} (must not exceed)",
                )
            )
        else:
            gates.append(
                _gate(
                    f"parse_failure[{taxonomy}]",
                    False,
                    "missing or non-finite parse_failure_rate (fail-closed)",
                )
            )
        b_acc, c_acc = b.get("accuracy_strict"), c.get("accuracy_strict")
        if _is_finite_number(b_acc) and _is_finite_number(c_acc):
            gates.append(
                _gate(
                    f"accuracy[{taxonomy}]",
                    c_acc >= b_acc - GATE_ACCURACY_MARGIN - _EPS,
                    f"candidate {c_acc:.3f} vs baseline {b_acc:.3f} "
                    f"(margin {GATE_ACCURACY_MARGIN:.2f})",
                )
            )
        else:
            gates.append(
                _gate(
                    f"accuracy[{taxonomy}]",
                    False,
                    "missing or non-finite accuracy_strict (fail-closed)",
                )
            )

    b_ragas = baseline.get("ragas")
    c_ragas = candidate.get("ragas")
    if not b_ragas or not c_ragas:
        gates.append(_gate("ragas", False, "missing RAGAS scores (fail-closed)"))
    else:
        # Every other RAGAS gate reads the reported aggregate fields, so a
        # stale or hand-edited block could clear all of them while its own
        # per_sample rows tell a different story (codex iter-3). Reconcile
        # BOTH sides against their raw rows first - a corrupted-low baseline
        # weakens every floor, so the baseline is not exempt.
        b_err = _ragas_consistency_error(b_ragas)
        c_err = _ragas_consistency_error(c_ragas)
        gates.append(
            _gate(
                "ragas[consistency]",
                b_err is None and c_err is None,
                f"baseline: {b_err or 'OK'}; candidate: {c_err or 'OK'}",
            )
        )
        # Consistency alone cannot see covered-but-unscored rows: the judge's
        # None-skip means a candidate scored on only 3 of its 10 context-bearing
        # rows recomputes cleanly while coverage certifies all 10 (codex
        # iter-4). Every covered row must actually carry a score, both sides.
        b_sc = _ragas_scoreless_error(b_ragas)
        c_sc = _ragas_scoreless_error(c_ragas)
        gates.append(
            _gate(
                "ragas[fully_scored]",
                b_sc is None and c_sc is None,
                f"baseline: {b_sc or 'OK'}; candidate: {c_sc or 'OK'}",
            )
        )
        # Scored is not the same as JUDGED: a mid-run judge failure degrades a
        # sample to word-overlap heuristics that reconcile and are fully
        # scored, so the two gates above both pass on it (#1485). Refuse rows
        # the judge did not actually judge, both sides — a heuristic-corrupted
        # baseline weakens every floor below, exactly as for consistency.
        b_hb = _ragas_heuristic_contamination_error(b_ragas)
        c_hb = _ragas_heuristic_contamination_error(c_ragas)
        gates.append(
            _gate(
                "ragas[judged]",
                b_hb is None and c_hb is None,
                f"baseline: {b_hb or 'OK'}; candidate: {c_hb or 'OK'}",
            )
        )
        for metric in ("faithfulness", "answer_relevancy"):
            b_val = b_ragas.get(metric)
            c_val = c_ragas.get(metric)
            # The judge emits None when zero judged samples carried contexts -
            # a truthy dict with a missing metric must fail, not TypeError.
            if not _is_finite_number(b_val) or not _is_finite_number(c_val):
                gates.append(
                    _gate(
                        f"ragas[{metric}]",
                        False,
                        f"missing {metric} (baseline={b_val!r}, candidate={c_val!r}; fail-closed)",
                    )
                )
                continue
            gates.append(
                _gate(
                    f"ragas[{metric}]",
                    c_val >= b_val - GATE_RAGAS_MARGIN - _EPS,
                    f"candidate {c_val:.3f} vs baseline {b_val:.3f} "
                    f"(margin {GATE_RAGAS_MARGIN:.2f})",
                )
            )
        # Errored/empty-answer replays are excluded from judging by
        # build_ragas_samples; require every requested replay to have produced
        # a judgeable answer so RAGAS can't quietly score only a candidate's
        # easier successful subset.
        b_n = b_ragas.get("n_samples")
        c_n = c_ragas.get("n_samples")
        b_req = (baseline.get("e2e") or {}).get("n")
        c_req = (candidate.get("e2e") or {}).get("n")
        complete = (
            all(isinstance(v, int) and not isinstance(v, bool) for v in (b_n, c_n, b_req, c_req))
            and b_n == b_req
            and c_n == c_req
        )
        gates.append(
            _gate(
                "ragas[completeness]",
                complete,
                f"judged/requested replays: baseline {b_n}/{b_req}, "
                f"candidate {c_n}/{c_req} (missing counts or errored/empty "
                "answers fail-closed)",
            )
        )
        # Faithfulness averages only context-bearing replays, and how often
        # retrieval finds contexts is itself model-influenced - so equal
        # n_samples does NOT make the faithfulness means comparable (codex
        # iter-2). Two add-only guards close that hole; requiring
        # n_faithfulness == n instead was rejected because context capture is
        # legitimately query/model-dependent (the completed run's baseline was
        # 3/10) - that would be always-fail, not fail-closed.
        b_nf = b_ragas.get("n_faithfulness")
        c_nf = c_ragas.get("n_faithfulness")
        nf_ints = all(isinstance(v, int) and not isinstance(v, bool) for v in (b_nf, c_nf))
        gates.append(
            _gate(
                "ragas[faithfulness_coverage]",
                nf_ints and c_nf >= b_nf and b_nf >= GATE_RAGAS_MIN_FAITHFULNESS_N,
                f"context-bearing replays: baseline {b_nf!r}, candidate "
                f"{c_nf!r} (need candidate >= baseline >= "
                f"{GATE_RAGAS_MIN_FAITHFULNESS_N}; missing counts fail-closed)",
            )
        )
        b_common, c_common, n_common = _common_subset_faithfulness(
            b_ragas.get("per_sample"), c_ragas.get("per_sample")
        )
        if b_common is None or c_common is None or n_common < GATE_RAGAS_MIN_FAITHFULNESS_N:
            gates.append(
                _gate(
                    "ragas[faithfulness_common_subset]",
                    False,
                    f"only {n_common} common context-bearing replays (need >= "
                    f"{GATE_RAGAS_MIN_FAITHFULNESS_N}; missing per-sample rows "
                    "fail-closed)",
                )
            )
        else:
            gates.append(
                _gate(
                    "ragas[faithfulness_common_subset]",
                    c_common >= b_common - GATE_RAGAS_MARGIN - _EPS,
                    f"candidate {c_common:.3f} vs baseline {b_common:.3f} on "
                    f"{n_common} common context-bearing replays "
                    f"(margin {GATE_RAGAS_MARGIN:.2f})",
                )
            )

    new_classes = _all_error_classes(candidate) - _all_error_classes(baseline)
    gates.append(
        _gate(
            "no_new_error_class",
            not new_classes,
            f"new error classes: {sorted(new_classes) or 'none'}",
        )
    )

    # Replay identity: counts alone let a duplicated easy replay hide a
    # dropped hard one on the e2e/RAGAS side, the same measurement-identity
    # hole the signature path anchors against the fixture (codex iter-9).
    if expected_replay_ids is None:
        gates.append(_gate("replay_anchor", False, "missing expected replay ids (fail-closed)"))
    else:
        expected_sorted = sorted(expected_replay_ids)
        b_ra = _replay_anchor_error(baseline, expected_sorted)
        c_ra = _replay_anchor_error(candidate, expected_sorted)
        gates.append(
            _gate(
                "replay_anchor",
                b_ra is None and c_ra is None,
                f"baseline: {b_ra or 'OK'}; candidate: {c_ra or 'OK'}",
            )
        )

    # Replay measurements must belong to the model they verdict (codex
    # iter-10): a mis-bound block always fails; identity-less legacy blocks
    # fail unless explicitly accepted, and the acceptance is surfaced.
    b_pe, b_note = _replay_provenance_error(
        baseline, allow_legacy_replay_provenance, allow_absent_ragas_model
    )
    c_pe, c_note = _replay_provenance_error(
        candidate, allow_legacy_replay_provenance, allow_absent_ragas_model
    )
    gates.append(
        _gate(
            "replay_provenance",
            b_pe is None and c_pe is None,
            f"baseline: {b_pe or b_note or 'OK'}; candidate: {c_pe or c_note or 'OK'}",
        )
    )

    b_e2e = baseline.get("e2e") or {}
    c_e2e = candidate.get("e2e") or {}
    b_lat, c_lat = b_e2e.get("latency_p50"), c_e2e.get("latency_p50")
    # Finite on BOTH sides: an inf baseline makes the limit inf (any candidate
    # passes) and a non-numeric baseline raised instead of failing (codex
    # iter-6) - same corruption class the signature/RAGAS gates fail closed on.
    if not _is_finite_number(b_lat) or not _is_finite_number(c_lat):
        gates.append(
            _gate("e2e_latency_p50", False, "missing or non-finite e2e latency (fail-closed)")
        )
    else:
        limit = GATE_E2E_LATENCY_FACTOR * b_lat
        gates.append(
            _gate(
                "e2e_latency_p50",
                c_lat <= limit + _EPS,
                f"candidate {c_lat:.1f}s vs limit {limit:.1f}s "
                f"({GATE_E2E_LATENCY_FACTOR}x baseline {b_lat:.1f}s)",
            )
        )

    return {"gates": gates, "all_passed": all(g["passed"] for g in gates)}


# ---------------------------------------------------------------------------
# Real-call runner (executes inside the prod container via the emitted bundle)
# ---------------------------------------------------------------------------


def run_signature_ab(
    golden_set: Dict[str, Any],
    models: List[str],
    progress=print,
) -> Dict[str, Any]:
    """Run both real intent signatures for every golden query on every model.

    Interleaves models per query (round-robin) so provider-side load variation
    over the run window affects all models equally. Uses ``cache=False`` and
    the real production signature classes - no mocks, no reimplementation.
    """
    import dspy  # deferred: only available where the run actually happens

    from src.api.routes.chatbot_dspy import ChatbotIntentClassifier
    from src.rag.cognitive_rag_dspy import IntentClassificationSignature

    cognitive = dspy.ChainOfThought(IntentClassificationSignature)
    chatbot = ChatbotIntentClassifier()
    lms = {model: dspy.LM(model, cache=False) for model in models}

    records: List[Dict[str, Any]] = []
    queries = golden_set["queries"]
    for i, item in enumerate(queries):
        for model in models:
            lm = lms[model]

            t0 = time.perf_counter()
            predicted, error = None, None
            try:
                with dspy.context(lm=lm):
                    pred = cognitive(query=item["query"], extracted_entities="{}")
                predicted = getattr(pred, "primary_intent", None)
            except Exception as exc:  # noqa: BLE001 - error class is the datum
                error = f"{type(exc).__name__}: {exc}"
            records.append(
                {
                    "model": model,
                    "taxonomy": "cognitive_rag",
                    "query_id": item["id"],
                    "predicted": predicted,
                    "acceptable": item["expected_cognitive"],
                    "latency_s": time.perf_counter() - t0,
                    "error": error,
                }
            )

            t0 = time.perf_counter()
            predicted_raw, predicted, error = None, None, None
            try:
                with dspy.context(lm=lm):
                    pred = chatbot(query=item["query"])
                predicted_raw = getattr(pred, "intent", None)
                # Score what production routes on, not the raw emission
                # (codex iter-5); raw is kept in the record for audit.
                predicted = production_chatbot_intent(predicted_raw)
            except Exception as exc:  # noqa: BLE001
                error = f"{type(exc).__name__}: {exc}"
            records.append(
                {
                    "model": model,
                    "taxonomy": "chatbot",
                    "query_id": item["id"],
                    "predicted": predicted,
                    "predicted_raw": predicted_raw,
                    "acceptable": item["expected_chatbot"],
                    "latency_s": time.perf_counter() - t0,
                    "error": error,
                }
            )
        progress(f"[{i + 1}/{len(queries)}] {item['id']} done")

    return {"records": records, "summary": summarize_signature_runs(records)}


def run_e2e_replays(
    golden_set: Dict[str, Any],
    query_ids: List[str],
    conversation_prefix: str,
    progress=print,
    expected_model: Optional[str] = None,
) -> Dict[str, Any]:
    """Replay full cognitive RAG turns for the given golden queries.

    Runs against the REAL production path (``CausalRAG.cognitive_search`` with
    real memory backends and retrieval) in the current process. The LM under
    test is whatever ``DSPY_LM_MODEL`` resolves to for this process - the
    caller sets the env var per candidate, exactly how the flip itself works.
    ``expected_model`` (always set by emitted bundles) makes a wrong or unset
    env fail fast instead of silently measuring the wrong candidate.

    Replays write real Reflector-phase learning signals; the recognizable
    ``conversation_prefix`` lets the operator remove those rows afterwards.
    """
    import os

    model = os.environ.get("DSPY_LM_MODEL", "(env default)")
    if expected_model is not None and model != expected_model:
        raise RuntimeError(
            f"DSPY_LM_MODEL resolves to {model!r} but this bundle expects "
            f"{expected_model!r} - refusing to measure the wrong candidate"
        )

    import asyncio
    import re

    from src.rag.causal_rag import CausalRAG

    model_slug = re.sub(r"[^a-zA-Z0-9]+", "-", model.split("/")[-1])
    by_id = {q["id"]: q for q in golden_set["queries"]}
    records: List[Dict[str, Any]] = []

    async def _run_all() -> None:
        rag = CausalRAG()
        for qid in query_ids:
            item = by_id[qid]
            t0 = time.perf_counter()
            record: Dict[str, Any] = {
                "model": model,
                "query_id": qid,
                # model-specific so concurrent candidates never share a
                # LangGraph thread or a Reflector signal attribution
                "conversation_id": f"{conversation_prefix}-{model_slug}-{qid}",
            }
            try:
                result = await rag.cognitive_search(
                    query=item["query"],
                    conversation_id=record["conversation_id"],
                )
                # cognitive_search reports failures as error-as-data
                error = result.get("error")
                contexts = []
                for ev in result.get("evidence") or []:
                    content = ev.get("content") if isinstance(ev, dict) else None
                    contexts.append(str(content) if content is not None else str(ev))
                record.update(
                    {
                        "latency_s": time.perf_counter() - t0,
                        "hop_count": result.get("hop_count", 0),
                        "evidence_count": len(result.get("evidence") or []),
                        "answer_chars": len(result.get("response") or ""),
                        "response_text": result.get("response") or "",
                        "contexts": contexts,
                        "detected_intent": result.get("intent"),
                        "routed_agents": result.get("routed_agents") or [],
                        "error": f"CognitiveSearchError: {error}" if error else None,
                    }
                )
            except Exception as exc:  # noqa: BLE001 - error class is the datum
                record.update(
                    {
                        "latency_s": time.perf_counter() - t0,
                        "hop_count": 0,
                        "evidence_count": 0,
                        "answer_chars": 0,
                        "response_text": "",
                        "contexts": [],
                        "detected_intent": None,
                        "routed_agents": [],
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            records.append(record)
            progress(f"[{len(records)}/{len(query_ids)}] e2e {qid} done")

    asyncio.run(_run_all())
    return {"model": model, "records": records, "summary": summarize_e2e_runs(records)}


def build_ragas_samples(
    e2e_results: Dict[str, Any], golden_set: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Convert e2e replay records into RAGAS ``EvaluationSample`` dicts.

    The samples carry each candidate's REAL generated answer and the contexts
    its run actually retrieved; the frozen gpt-4o judge then scores
    faithfulness (answer vs contexts) and answer_relevancy (answer vs query).
    ``ground_truth`` stays empty - neither gated metric uses it, and inventing
    one would fabricate a reference. Errored or empty-answer runs are dropped:
    judging an error string would score the failure mode, not the answer
    (those failures are already counted by the error-class and latency gates).
    """
    by_id = {q["id"]: q for q in golden_set["queries"]}
    samples = []
    for rec in e2e_results["records"]:
        if rec.get("error") or not rec.get("response_text"):
            continue
        samples.append(
            {
                "query": by_id[rec["query_id"]]["query"],
                "ground_truth": "",
                "answer": rec["response_text"],
                "retrieved_contexts": rec.get("contexts") or [],
                "metadata": {"query_id": rec["query_id"], "model": rec["model"]},
            }
        )
    return samples


# ---------------------------------------------------------------------------
# Container bundle emission
# ---------------------------------------------------------------------------

_BUNDLE_DRIVER = """

# --- bundle driver (appended by emit_container_script) ---
if __name__ == "__main__":
    import sys

    sys.path.insert(0, "/app")
    GOLDEN_SET = json.loads(GOLDEN_SET_JSON)
    if BUNDLE_MODE == "signature":
        results = run_signature_ab(GOLDEN_SET, BUNDLE_MODELS)
    else:
        results = run_e2e_replays(
            GOLDEN_SET,
            BUNDLE_E2E_IDS,
            BUNDLE_CONVERSATION_PREFIX,
            expected_model=BUNDLE_EXPECTED_MODEL,
        )
    print("RESULTS_JSON_BEGIN")
    print(json.dumps(results))
    print("RESULTS_JSON_END")
"""


def emit_container_script(
    golden_set: Dict[str, Any],
    models: List[str],
    mode: str = "signature",
    e2e_query_ids: Optional[List[str]] = None,
    conversation_prefix: str = "dspy-ab",
) -> str:
    """Produce a self-contained script for stdin-piping into the prod container.

    The container runs deployed code on a read-only rootfs, so the bundle
    embeds this module's full source plus the golden set - it must not import
    this module from the repo.

    ``mode="signature"`` runs every model in ``models`` over the golden set.
    ``mode="e2e"`` replays ``e2e_query_ids`` through the full cognitive RAG
    path; the model under test comes from the process's ``DSPY_LM_MODEL``.
    """
    if mode not in ("signature", "e2e"):
        raise ValueError(f"unknown mode: {mode!r}")
    e2e_query_ids = e2e_query_ids or []
    expected_model: Optional[str] = None
    if mode == "e2e":
        known = {q["id"] for q in golden_set["queries"]}
        unknown = [qid for qid in e2e_query_ids if qid not in known]
        if unknown:
            raise ValueError(f"unknown e2e query ids: {unknown}")
        if not e2e_query_ids:
            raise ValueError("e2e mode requires e2e_query_ids")
        # One bundle == one candidate: the intended model is pinned into the
        # bundle so a wrong/unset DSPY_LM_MODEL env fails fast at launch
        # instead of silently measuring the wrong candidate.
        if len(models) != 1:
            raise ValueError("e2e mode requires exactly one model (the candidate under test)")
        expected_model = models[0]

    golden_json = json.dumps(golden_set)
    if '"""' in golden_json:
        raise ValueError("golden set must not contain triple quotes")
    module_source = Path(__file__).read_text()
    return (
        module_source
        + f'\nGOLDEN_SET_JSON = r"""{golden_json}"""\n'
        + f"BUNDLE_MODE = {mode!r}\n"
        + f"BUNDLE_MODELS = {models!r}\n"
        + f"BUNDLE_E2E_IDS = {e2e_query_ids!r}\n"
        + f"BUNDLE_CONVERSATION_PREFIX = {conversation_prefix!r}\n"
        + f"BUNDLE_EXPECTED_MODEL = {expected_model!r}\n"
        + _BUNDLE_DRIVER
    )
