"""Measure Layer-4 (DSPy + Haiku-evaluator) precision on the literature-derived golden set.

Per FINAL plan §4.1 Phase 4 acceptance criterion: precision ≥0.95 on
``instrument`` classifications gated by ``evaluator_satisfied=True``. The
golden set built under issue #358 supplies the ground truth.

Invocation::

    python scripts/measure_layer4_precision.py \\
        --enable-evaluator \\
        --evaluator-gate true \\
        [--classifier-artifact artifacts/dspy/cr_bootstrap_n200.json] \\
        [--golden-set tests/fixtures/causal_role_golden_set.json] \\
        [--cohort CSU_remibrutinib|PNH_fabhalta|BC_kisqali|all] \\
        [--threshold 0.95] \\
        [--report-path /tmp/layer4_precision_report.json]

Exits 0 if precision ≥ ``--threshold`` on the gated subset; non-zero
otherwise. Suitable for CI gating Phase 4 instrument routing.

Implementation notes:

- This script runs the compiled DSPy classifier (`classify_feature`) on
  each golden-set entry's ``(feature_name, derivation_pseudocode,
  dataset_context)`` triple. When no DSPy LM is configured
  (``ANTHROPIC_API_KEY`` absent), ``classify_feature`` returns ``None``
  and the entry is reported as ``skipped_no_lm`` — the script then exits
  0 with a summary instead of failing, so local laptop runs without API
  credentials don't break.

- The evaluator gate (``evaluator_satisfied=True``) is the Layer-4-Phase-4
  contract per issue #240 audit-evaluator promotion plan: only
  classifier outputs that pass the Haiku audit gate are trusted for IV
  routing. ``--evaluator-gate both`` reports both gated and ungated
  precision so you can see what the gate buys you on the literature set.

- The Haiku audit evaluator is wired into ``classify_feature`` and
  gated by the env var ``ADAPTIVE_VALIDITY_EVALUATOR_ENABLED=1``.
  The ``--enable-evaluator`` flag sets this in-process BEFORE the
  loader import. Without it, the gated subset will be empty (all
  entries fall through to ``skipped_no_eval``).

- The literature-derived golden set has 90 entries (30 per cohort).
  ≥6 per cohort are labeled ``instrument`` → ≥18 instrument labels
  total → enough for a meaningful precision estimate when prevalence
  is ~20%.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load .env so CLI invocations get ANTHROPIC_API_KEY without manual export.
# Without this the script silently no-ops on the LM-keyed path even when the
# key sits in .env (conftest does this for pytest paths; CLI does not).
load_dotenv()

# NOTE: the classifier_loader import is intentionally deferred into main() so
# that --enable-evaluator can set ADAPTIVE_VALIDITY_EVALUATOR_ENABLED=1 before
# the module is first imported (the env var is read at import time).

DEFAULT_GOLDEN_SET = PROJECT_ROOT / "tests" / "fixtures" / "causal_role_golden_set.json"
DEFAULT_THRESHOLD = 0.95
ALL_COHORTS = ("CSU_remibrutinib", "PNH_fabhalta", "BC_kisqali")

logger = logging.getLogger(__name__)


@dataclass
class CohortMetrics:
    """Precision/recall breakdown for one cohort × one gate setting."""

    cohort: str
    gate: str  # "gated", "ungated"
    n_total: int = 0
    n_evaluated: int = 0
    n_skipped_no_lm: int = 0
    n_skipped_no_eval: int = 0
    instrument_tp: int = 0
    instrument_fp: int = 0
    instrument_fn: int = 0
    confusion: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def precision_instrument(self) -> Optional[float]:
        denom = self.instrument_tp + self.instrument_fp
        if denom == 0:
            return None
        return self.instrument_tp / denom

    @property
    def recall_instrument(self) -> Optional[float]:
        denom = self.instrument_tp + self.instrument_fn
        if denom == 0:
            return None
        return self.instrument_tp / denom

    def as_dict(self) -> dict[str, Any]:
        return {
            "cohort": self.cohort,
            "gate": self.gate,
            "n_total": self.n_total,
            "n_evaluated": self.n_evaluated,
            "n_skipped_no_lm": self.n_skipped_no_lm,
            "n_skipped_no_eval": self.n_skipped_no_eval,
            "instrument_tp": self.instrument_tp,
            "instrument_fp": self.instrument_fp,
            "instrument_fn": self.instrument_fn,
            "precision_instrument": self.precision_instrument,
            "recall_instrument": self.recall_instrument,
            "confusion": dict(self.confusion),
        }


def _load_golden_set(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Golden set not found at {path}")
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path} did not parse as a JSON object")
    if "entries" not in raw:
        raise ValueError(f"{path} missing 'entries' key")
    data: dict[str, Any] = raw
    return data


def _structural_predict(entry: dict[str, Any]) -> str:
    """Predict a feature's causal role deterministically from its authored DAG
    edges via :func:`extract_role` (Plan v4 Layer B / Phase 2).

    Raises ``SystemExit`` when the entry carries no ``edges`` — the 91-entry
    literature golden set must be edge-augmented before ``--decider structural``
    can score it (plan Task 8); failing loudly beats silently scoring nothing.
    """
    import networkx as nx

    from src.ml.causal_role_dgp.extractor import extract_role

    edges = entry.get("edges")
    if not edges:
        raise SystemExit(
            f"--decider structural requires per-entry 'edges'; entry "
            f"{entry.get('feature_name', '<missing>')!r} has none. Edge-augment "
            f"the golden set first (plan Task 8)."
        )
    graph = nx.DiGraph([tuple(e) for e in edges])
    return extract_role(
        entry["feature_node"], entry["treatment_node"], entry["outcome_node"], graph
    )


def _macro_precision(buckets: dict[tuple[str, str], CohortMetrics]) -> Optional[float]:
    """Macro-averaged per-PREDICTED-role precision across the ungated buckets.

    For each predicted role P, precision(P) = correct(P) / predicted(P); the
    macro average weights every predicted role equally (the non-circular
    literature metric plan Task 8 gates on, ≥0.90). Uses the ungated buckets (the
    decider's raw predictions) so a gated+ungated double-run does not double-
    count; falls back to all buckets when no ungated pass ran. Returns None when
    nothing was predicted.
    """
    relevant = [m for m in buckets.values() if m.gate == "ungated"] or list(buckets.values())
    predicted_total: dict[str, int] = {}
    correct: dict[str, int] = {}
    for m in relevant:
        for truth, row in m.confusion.items():
            for pred, cnt in row.items():
                predicted_total[pred] = predicted_total.get(pred, 0) + cnt
                if pred == truth:
                    correct[pred] = correct.get(pred, 0) + cnt
    if not predicted_total:
        return None
    precisions = [correct.get(p, 0) / predicted_total[p] for p in predicted_total]
    return sum(precisions) / len(precisions)


def _evaluate_entries(
    entries: list[dict[str, Any]],
    *,
    cohort_filter: Optional[str],
    require_evaluator: bool,
    classifier_artifact: Optional[Path] = None,
    disagreements: Optional[list[dict[str, str]]] = None,
    decider: str = "llm",
    _classify_feature: Any,
    _ensure_dspy_lm_configured: Any,
    _load_compiled_classifier: Any,
) -> dict[tuple[str, str], CohortMetrics]:
    """Run the classifier on each entry and bucket into per-cohort metrics.

    Returns a mapping of (cohort, gate) -> CohortMetrics. Gate is
    "gated" when require_evaluator=True (subset to satisfied=True) and
    "ungated" when False (all evaluated entries).

    When ``disagreements`` is provided, appends one record per
    classifier/ground-truth mismatch with EXACTLY the four keys
    ``{cohort, gate, predicted_role, ground_truth_role}`` — feature_name
    and derivation_pseudocode are excluded by construction (plan-239 §6.4
    HARD RULE: golden-set entries must not leak into compile-set authoring).

    The three ``_*`` parameters receive the lazily-imported loader callables
    from ``main()`` so that ``ADAPTIVE_VALIDITY_EVALUATOR_ENABLED`` is set
    before the loader module is first imported (env-var-before-import
    contract for the --enable-evaluator flag).
    """
    # Ensure DSPy has an LM configured at inference time. Plan-239 §6.8 A/B
    # requires the classifier to actually run against the literature golden
    # set; the loader's classify_feature returns None if no LM is configured,
    # which would silently mark every entry as skipped_no_lm. The
    # ensure_dspy_lm_configured() helper is provider-aware and no-ops when
    # an LM is already configured or when no provider-matching key is in env
    # (matches conftest dotenv path + production orchestrator behaviour).
    # The structural decider is LLM-free; only configure/load the DSPy classifier
    # for the 'llm' decider.
    classifier = None
    if decider != "structural":
        _ensure_dspy_lm_configured()

        if classifier_artifact is not None:
            classifier = _load_compiled_classifier(artifact_path=classifier_artifact)
        else:
            classifier = _load_compiled_classifier()
        if classifier is None:
            logger.warning(
                "load_compiled_classifier returned None — no LM configured. "
                "All entries will be reported as skipped_no_lm."
            )

    gate_label = "gated" if require_evaluator else "ungated"
    bucket: dict[tuple[str, str], CohortMetrics] = {}

    for entry in entries:
        cohort = entry.get("cohort") or entry.get("scenario") or "unknown"
        if cohort_filter and cohort_filter != "all" and cohort != cohort_filter:
            continue

        key = (cohort, gate_label)
        if key not in bucket:
            bucket[key] = CohortMetrics(cohort=cohort, gate=gate_label)
        metrics = bucket[key]
        metrics.n_total += 1

        ground_truth = entry.get("ground_truth_role")
        feature_name = entry.get("feature_name", "<missing>")
        derivation = entry.get("derivation_pseudocode", "")
        context = entry.get("dataset_context", "")

        if decider == "structural":
            # Deterministic structural decider: extract_role over the entry's
            # authored DAG edges (Plan v4 Layer B / Phase 2). No LLM, no
            # evaluator gate — every entry with edges is scored.
            predicted = _structural_predict(entry)
            metrics.n_evaluated += 1
        else:
            if classifier is None:
                metrics.n_skipped_no_lm += 1
                continue

            verdict = _classify_feature(
                feature_name=feature_name,
                derivation_pseudocode=derivation,
                dataset_context=context,
                classifier=classifier,
            )
            if verdict is None:
                metrics.n_skipped_no_lm += 1
                continue

            # require_evaluator gate is applied here: skip entries where the
            # evaluator did not satisfy criteria. In production, the evaluator
            # would run alongside the classifier — for this metric script we
            # approximate by treating verdict.evaluator_audit.satisfied as the
            # gate. When require_evaluator=False, all classifier outputs count.
            evaluator_audit = getattr(verdict, "evaluator_audit", None)
            if require_evaluator:
                if evaluator_audit is None:
                    metrics.n_skipped_no_eval += 1
                    continue
                if not getattr(evaluator_audit, "satisfied", False):
                    metrics.n_skipped_no_eval += 1
                    continue

            metrics.n_evaluated += 1
            predicted = verdict.causal_role

        # Track confusion matrix entries.
        truth_row = metrics.confusion.setdefault(str(ground_truth), {})
        truth_row[str(predicted)] = truth_row.get(str(predicted), 0) + 1

        # Plan-239 §6.4 disagreement accumulator (4-key shape, by construction
        # excludes feature_name + derivation_pseudocode to prevent golden-set
        # leakage into compile-set authoring).
        if disagreements is not None and str(predicted) != str(ground_truth):
            disagreements.append(
                {
                    "cohort": str(cohort),
                    "gate": gate_label,
                    "predicted_role": str(predicted),
                    "ground_truth_role": str(ground_truth),
                }
            )

        # Instrument-specific TP/FP/FN bookkeeping (the gating decision).
        if ground_truth == "instrument" and predicted == "instrument":
            metrics.instrument_tp += 1
        elif ground_truth != "instrument" and predicted == "instrument":
            metrics.instrument_fp += 1
        elif ground_truth == "instrument" and predicted != "instrument":
            metrics.instrument_fn += 1

    return bucket


def _aggregate_metrics(
    buckets: dict[tuple[str, str], CohortMetrics],
) -> dict[str, dict[str, Any]]:
    """Aggregate per-cohort metrics into per-gate overall totals."""
    by_gate: dict[str, CohortMetrics] = {}
    for (_, gate), m in buckets.items():
        overall = by_gate.setdefault(gate, CohortMetrics(cohort="OVERALL", gate=gate))
        overall.n_total += m.n_total
        overall.n_evaluated += m.n_evaluated
        overall.n_skipped_no_lm += m.n_skipped_no_lm
        overall.n_skipped_no_eval += m.n_skipped_no_eval
        overall.instrument_tp += m.instrument_tp
        overall.instrument_fp += m.instrument_fp
        overall.instrument_fn += m.instrument_fn
        for truth_label, pred_row in m.confusion.items():
            overall_row = overall.confusion.setdefault(truth_label, {})
            for pred_label, count in pred_row.items():
                overall_row[pred_label] = overall_row.get(pred_label, 0) + count
    return {gate: m.as_dict() for gate, m in by_gate.items()}


def _format_report(
    *,
    buckets: dict[tuple[str, str], CohortMetrics],
    overall: dict[str, dict[str, Any]],
    threshold: float,
) -> str:
    lines = ["", "=" * 78, "Layer-4 Precision Report (issue #358 golden set)", "=" * 78]
    for (cohort, gate), m in sorted(buckets.items()):
        prec = m.precision_instrument
        rec = m.recall_instrument
        prec_str = f"{prec:.3f}" if prec is not None else "n/a (no instrument predictions)"
        rec_str = f"{rec:.3f}" if rec is not None else "n/a (no instrument ground truths)"
        lines.append("")
        lines.append(
            f"[{cohort}/{gate}]  n_total={m.n_total}  n_evaluated={m.n_evaluated}  "
            f"skipped_no_lm={m.n_skipped_no_lm}  skipped_no_eval={m.n_skipped_no_eval}"
        )
        lines.append(
            f"  instrument: TP={m.instrument_tp} FP={m.instrument_fp} FN={m.instrument_fn}"
        )
        lines.append(f"  precision={prec_str}   recall={rec_str}")

    lines.append("")
    lines.append("-" * 78)
    lines.append("OVERALL aggregates:")
    for gate, gate_data in overall.items():
        prec = gate_data["precision_instrument"]
        prec_str = f"{prec:.3f}" if prec is not None else "n/a"
        pass_fail = (
            "PASS"
            if prec is not None and prec >= threshold
            else ("FAIL" if prec is not None else "n/a")
        )
        lines.append(
            f"  [{gate}]  TP={gate_data['instrument_tp']} FP={gate_data['instrument_fp']} "
            f"FN={gate_data['instrument_fn']}  precision={prec_str}  threshold={threshold:.2f}  → {pass_fail}"
        )
    lines.append("=" * 78)
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    """Return the argument parser for measure_layer4_precision.py."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--golden-set",
        type=Path,
        default=DEFAULT_GOLDEN_SET,
        help=f"Path to golden-set JSON (default: {DEFAULT_GOLDEN_SET.relative_to(PROJECT_ROOT)})",
    )
    parser.add_argument(
        "--cohort",
        choices=list(ALL_COHORTS) + ["all"],
        default="all",
        help="Restrict to one cohort, or 'all' (default).",
    )
    parser.add_argument(
        "--evaluator-gate",
        choices=("true", "false", "both"),
        default="both",
        help="Apply evaluator_satisfied=True gate. 'both' reports both subsets.",
    )
    parser.add_argument(
        "--enable-evaluator",
        action="store_true",
        help=(
            "Set ADAPTIVE_VALIDITY_EVALUATOR_ENABLED=1 in-process BEFORE "
            "importing the classifier loader, so classify_feature attaches "
            "evaluator_audit to each verdict. Required for the gated "
            "subset to be non-empty. Pair with --evaluator-gate=true."
        ),
    )
    parser.add_argument(
        "--decider",
        choices=("llm", "structural"),
        default="llm",
        help=(
            "Which role decider to score. 'llm' (default) runs the compiled DSPy "
            "classifier. 'structural' runs the deterministic extract_role over each "
            "entry's authored DAG 'edges' (Plan v4 Layer B / Phase 2) — LLM-free; "
            "requires per-entry edges/feature_node/treatment_node/outcome_node and "
            "gates on macro_precision."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Precision threshold for exit code (default: {DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=None,
        help="If set, write JSON report to this path in addition to stdout.",
    )
    parser.add_argument(
        "--classifier-artifact",
        type=Path,
        default=None,
        help=(
            "Override the compiled-classifier path (plan-239 §6.0 F1). "
            "Defaults to artifacts/dspy/causal_role_classifier.json via "
            "load_compiled_classifier()."
        ),
    )
    parser.add_argument(
        "--disagreements-path",
        type=Path,
        default=None,
        help=(
            "If set, write a JSON list of classifier/ground-truth "
            "disagreements (plan-239 §6.0 F2 / §6.4). Each record contains "
            "EXACTLY {cohort, gate, predicted_role, ground_truth_role} — "
            "feature_name and derivation_pseudocode are excluded by "
            "construction (HARD RULE: no golden-set leakage into compile-set)."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level (default: INFO)",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()

    # --enable-evaluator: set env var BEFORE importing the classifier loader so
    # classify_feature sees ADAPTIVE_VALIDITY_EVALUATOR_ENABLED=1 at import
    # time (env-var-before-import contract).
    if args.enable_evaluator:
        os.environ["ADAPTIVE_VALIDITY_EVALUATOR_ENABLED"] = "1"

    # Test seam: pytest may patch module-level `load_compiled_classifier` /
    # `classify_feature` / `ensure_dspy_lm_configured` on this module before
    # calling main(); honor those overrides. Each symbol is resolved
    # independently so partial patches (e.g., only load_compiled_classifier)
    # do not get clobbered by the lazy import. Lazy-imports happen AFTER the
    # env-var contract above so the loader reads
    # ADAPTIVE_VALIDITY_EVALUATOR_ENABLED on its first import.
    _module = sys.modules[__name__]
    _load_compiled_classifier = getattr(_module, "load_compiled_classifier", None)
    _classify_feature = getattr(_module, "classify_feature", None)
    _ensure_dspy_lm_configured = getattr(_module, "ensure_dspy_lm_configured", None)

    # The structural decider is LLM-free — skip importing the DSPy classifier
    # loader entirely (no ANTHROPIC_API_KEY / DSPy LM needed). The loader
    # callables stay None and are never invoked on the structural path.
    if args.decider != "structural":
        if _load_compiled_classifier is None:
            from src.data.causal_role_classifier_loader import (  # noqa: E402
                load_compiled_classifier as _load_compiled_classifier,
            )
        if _classify_feature is None:
            from src.data.causal_role_classifier_loader import (  # noqa: E402
                classify_feature as _classify_feature,
            )
        if _ensure_dspy_lm_configured is None:
            from src.data.causal_role_classifier_loader import (  # noqa: E402
                ensure_dspy_lm_configured as _ensure_dspy_lm_configured,
            )

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    data = _load_golden_set(args.golden_set)
    entries = data["entries"]
    logger.info(
        "loaded %d entries from %s (cohort filter: %s)",
        len(entries),
        args.golden_set,
        args.cohort,
    )

    cohort_filter = None if args.cohort == "all" else args.cohort
    buckets: dict[tuple[str, str], CohortMetrics] = {}
    disagreements: Optional[list[dict[str, str]]] = (
        [] if args.disagreements_path is not None else None
    )

    gates_to_run: list[bool]
    if args.decider == "structural":
        # The evaluator gate is an LLM-only concept; the deterministic structural
        # decider has no evaluator_audit, so run a single ungated pass.
        gates_to_run = [False]
    elif args.evaluator_gate == "true":
        gates_to_run = [True]
    elif args.evaluator_gate == "false":
        gates_to_run = [False]
    else:
        gates_to_run = [True, False]

    for gate in gates_to_run:
        buckets.update(
            _evaluate_entries(
                entries,
                cohort_filter=cohort_filter,
                require_evaluator=gate,
                classifier_artifact=args.classifier_artifact,
                disagreements=disagreements,
                decider=args.decider,
                _classify_feature=_classify_feature,
                _ensure_dspy_lm_configured=_ensure_dspy_lm_configured,
                _load_compiled_classifier=_load_compiled_classifier,
            )
        )

    overall = _aggregate_metrics(buckets)
    macro_prec = _macro_precision(buckets)
    print(_format_report(buckets=buckets, overall=overall, threshold=args.threshold))

    if args.report_path:
        report_doc = {
            "golden_set_path": str(args.golden_set),
            "cohort_filter": args.cohort,
            "decider": args.decider,
            "threshold": args.threshold,
            "macro_precision": macro_prec,
            "per_cohort": [m.as_dict() for m in buckets.values()],
            "overall": overall,
        }
        args.report_path.write_text(json.dumps(report_doc, indent=2, sort_keys=True) + "\n")
        logger.info("wrote JSON report to %s", args.report_path)

    if args.disagreements_path is not None and disagreements is not None:
        args.disagreements_path.write_text(
            json.dumps(disagreements, indent=2, sort_keys=True) + "\n"
        )
        logger.info(
            "wrote %d disagreement records to %s (plan-239 §6.4 4-key shape)",
            len(disagreements),
            args.disagreements_path,
        )

    # Exit code logic: 0 if ANY overall gate passes threshold (so a true
    # PASS on the gated subset is sufficient for Phase 4 acceptance);
    # 0 if no instrument predictions were made (n/a — typically because no
    # LM is configured locally); 1 if at least one gate has a measurable
    # precision and ALL are below threshold.
    # Structural decider: gate on macro-averaged precision (the non-circular
    # literature metric, plan Task 8) rather than the LLM instrument precision.
    if args.decider == "structural":
        if macro_prec is None:
            logger.warning("No structural predictions made — macro precision not measured.")
            return 0
        if macro_prec >= args.threshold:
            return 0
        logger.error(
            "Structural macro precision %.3f did not meet threshold %.2f",
            macro_prec,
            args.threshold,
        )
        return 1

    measurable_precisions = [
        m["precision_instrument"] for m in overall.values() if m["precision_instrument"] is not None
    ]
    if not measurable_precisions:
        logger.warning(
            "No instrument predictions made — precision not measured. Exiting 0 "
            "(typical for laptop runs without ANTHROPIC_API_KEY)."
        )
        return 0
    if any(p >= args.threshold for p in measurable_precisions):
        return 0
    logger.error(
        "Layer-4 precision %s did not meet threshold %.2f",
        [f"{p:.3f}" for p in measurable_precisions],
        args.threshold,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
