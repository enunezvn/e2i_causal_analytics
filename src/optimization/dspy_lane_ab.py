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
"""

from __future__ import annotations

import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

# Pre-registered gate parameters (plan section 5). Changing these after seeing
# results would be gate-shopping; they are constants on purpose.
GATE_ACCURACY_MARGIN = 0.05
GATE_RAGAS_MARGIN = 0.05
GATE_E2E_LATENCY_FACTOR = 1.5

_EPS = 1e-9


# ---------------------------------------------------------------------------
# Golden set
# ---------------------------------------------------------------------------


def load_golden_set(path: str | Path) -> Dict[str, Any]:
    """Load and minimally validate the golden-set fixture."""
    golden = json.loads(Path(path).read_text())
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
                "accuracy_strict": (sum(strict) / n_scored) if n_scored else None,
                "accuracy_lenient": (sum(lenient) / n_scored) if n_scored else None,
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
            "n": len(recs),
            "n_errors": len(recs) - len(ok),
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


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _gate(name: str, passed: bool, detail: str) -> Dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def _all_error_classes(bundle: Dict[str, Any]) -> set:
    classes: set = set()
    for tax_metrics in (bundle.get("signature") or {}).values():
        classes |= set((tax_metrics.get("error_classes") or {}))
    classes |= set(((bundle.get("e2e") or {}).get("error_classes") or {}))
    return classes


def evaluate_gates(baseline: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate one candidate against the baseline on the five hard gates.

    Missing measurement blocks (e.g. no RAGAS scores) fail closed: a gate
    cannot pass on absent data.
    """
    gates: List[Dict[str, Any]] = []

    for taxonomy, b in (baseline.get("signature") or {}).items():
        c = (candidate.get("signature") or {}).get(taxonomy)
        if c is None:
            gates.append(_gate(f"signature[{taxonomy}]", False, "missing candidate data"))
            continue
        gates.append(
            _gate(
                f"parse_failure[{taxonomy}]",
                c["parse_failure_rate"] <= b["parse_failure_rate"] + _EPS,
                f"candidate {c['parse_failure_rate']:.3f} vs baseline "
                f"{b['parse_failure_rate']:.3f} (must not exceed)",
            )
        )
        gates.append(
            _gate(
                f"accuracy[{taxonomy}]",
                c["accuracy_strict"] >= b["accuracy_strict"] - GATE_ACCURACY_MARGIN - _EPS,
                f"candidate {c['accuracy_strict']:.3f} vs baseline {b['accuracy_strict']:.3f} "
                f"(margin {GATE_ACCURACY_MARGIN:.2f})",
            )
        )

    b_ragas = baseline.get("ragas")
    c_ragas = candidate.get("ragas")
    if not b_ragas or not c_ragas:
        gates.append(_gate("ragas", False, "missing RAGAS scores (fail-closed)"))
    else:
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
                        f"missing {metric} (baseline={b_val!r}, "
                        f"candidate={c_val!r}; fail-closed)",
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

    new_classes = _all_error_classes(candidate) - _all_error_classes(baseline)
    gates.append(
        _gate(
            "no_new_error_class",
            not new_classes,
            f"new error classes: {sorted(new_classes) or 'none'}",
        )
    )

    b_e2e = baseline.get("e2e") or {}
    c_e2e = candidate.get("e2e") or {}
    if b_e2e.get("latency_p50") is None or c_e2e.get("latency_p50") is None:
        gates.append(_gate("e2e_latency_p50", False, "missing e2e latency (fail-closed)"))
    else:
        limit = GATE_E2E_LATENCY_FACTOR * b_e2e["latency_p50"]
        gates.append(
            _gate(
                "e2e_latency_p50",
                c_e2e["latency_p50"] <= limit + _EPS,
                f"candidate {c_e2e['latency_p50']:.1f}s vs limit {limit:.1f}s "
                f"({GATE_E2E_LATENCY_FACTOR}x baseline {b_e2e['latency_p50']:.1f}s)",
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
            predicted, error = None, None
            try:
                with dspy.context(lm=lm):
                    pred = chatbot(query=item["query"])
                predicted = getattr(pred, "intent", None)
            except Exception as exc:  # noqa: BLE001
                error = f"{type(exc).__name__}: {exc}"
            records.append(
                {
                    "model": model,
                    "taxonomy": "chatbot",
                    "query_id": item["id"],
                    "predicted": predicted,
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
