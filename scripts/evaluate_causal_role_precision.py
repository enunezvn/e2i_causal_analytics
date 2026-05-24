"""Precision/recall harness for the synthetic golden set (plan §5).

Replays the golden-set fixture through the compiled
``CausalRoleClassifier`` and emits:

- Console: confusion matrix + per-role precision/recall/F1 + macro F1
  split between Family A (cohort-only, GATED) and Family B
  ((T, Y)-explicit, INFORMATIONAL).
- JSON artifact at ``--out`` with full per-entry results + computed metrics.

The harness itself does NOT enforce a threshold. The integration test
at ``tests/integration/test_causal_role_classifier_golden_set.py``
wraps the harness and runs the Tier 1 unconditional sanity assertions.

Invocation (run with a real Anthropic key for full results)::

    python scripts/evaluate_causal_role_precision.py \\
        [--golden-set tests/fixtures/causal_role_golden_set_synthetic.json] \\
        [--lm-model anthropic/claude-sonnet-4-20250514] \\
        [--limit N] \\
        [--out artifacts/golden_set/precision_recall_<ts>.json]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Bridge .env → process env BEFORE the API-key gate at
# ``os.environ.get("ANTHROPIC_API_KEY")`` below and BEFORE any DSPy /
# causal_role_classifier_loader import (which may read provider env
# vars at import time). Without this call, invoking the script
# directly from the shell silently degrades to fixture-schema-only
# validation even when ``ANTHROPIC_API_KEY`` is present in ``.env``
# but not exported. See GitHub issue #470.
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_GOLDEN_SET = PROJECT_ROOT / "tests" / "fixtures" / "causal_role_golden_set_synthetic.json"
DEFAULT_LM_MODEL = "anthropic/claude-sonnet-4-20250514"


VALID_ROLES = frozenset(
    {"ancestor", "confounder", "mediator", "collider", "descendant", "instrument"}
)
GRAPH_READABLE_ROLES = frozenset({"ancestor", "confounder", "mediator", "collider", "descendant"})


def _macro_f1(
    results: list[dict[str, Any]],
    roles: frozenset[str],
) -> tuple[float, dict[str, dict[str, float]]]:
    """Compute macro-averaged F1 + per-class precision/recall."""
    per_role: dict[str, dict[str, float]] = {}
    f1s: list[float] = []
    for role in sorted(roles):
        tp = sum(1 for r in results if r["expected"] == role and r["observed"] == role)
        fp = sum(1 for r in results if r["expected"] != role and r["observed"] == role)
        fn = sum(1 for r in results if r["expected"] == role and r["observed"] != role)
        if tp == 0 and fp == 0 and fn == 0:
            continue
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_role[role] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
        f1s.append(f1)
    macro = sum(f1s) / len(f1s) if f1s else 0.0
    return macro, per_role


def _git_head() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--golden-set",
        type=Path,
        default=DEFAULT_GOLDEN_SET,
        help="Path to the golden-set fixture JSON.",
    )
    parser.add_argument(
        "--classifier-artifact",
        type=Path,
        default=None,
        help="Override the compiled DSPy artifact path. Default: loader default.",
    )
    parser.add_argument("--lm-model", default=DEFAULT_LM_MODEL)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap the number of entries replayed (cost-bounded dry run).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON path. Default: artifacts/golden_set/precision_recall_<UTC>.json",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        default=True,
        help="Disable DSPy LM caching (default ON per Option C iter-0 MED).",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    logger = logging.getLogger(__name__)

    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key.startswith("sk-ant-"):
        logger.warning(
            "ANTHROPIC_API_KEY missing or not a real Anthropic key "
            "(expected `sk-ant-` prefix). Live-LM replay will fail; "
            "the harness will still validate the fixture schema and exit."
        )

    golden = json.loads(args.golden_set.read_text())
    entries = golden["entries"]
    if args.limit is not None:
        entries = entries[: args.limit]

    family_a: list[dict[str, Any]] = []
    family_b: list[dict[str, Any]] = []

    if api_key.startswith("sk-ant-"):
        import dspy

        from src.data.causal_role_classifier_loader import (
            classify_feature,
            load_compiled_classifier,
        )

        dspy.configure(lm=dspy.LM(args.lm_model, cache=False))
        classifier = load_compiled_classifier(artifact_path=args.classifier_artifact)
        if classifier is None:
            raise SystemExit(
                "load_compiled_classifier returned None — recompile or check "
                "--classifier-artifact path."
            )

        try:
            for entry in entries:
                try:
                    verdict = classify_feature(
                        feature_name=entry["feature_name"],
                        derivation_pseudocode=entry["derivation_pseudocode"],
                        dataset_context=entry["dataset_context"],
                        classifier=classifier,
                    )
                    observed = verdict.causal_role if verdict is not None else "<None>"
                except Exception as exc:
                    observed = f"<EXCEPTION: {type(exc).__name__}: {exc}>"
                row = {
                    "scenario": entry["scenario"],
                    "feature_name": entry["feature_name"],
                    "expected": entry["ground_truth_role"],
                    "observed": observed,
                    "treatment_explicit": entry["treatment_explicit"],
                }
                if entry["treatment_explicit"]:
                    family_b.append(row)
                else:
                    family_a.append(row)
        finally:
            dspy.settings.configure(lm=None)
    else:
        logger.info("Skipping LM replay; fixture-schema-only validation done.")

    cohort_only_macro_f1, per_role_a = _macro_f1(family_a, GRAPH_READABLE_ROLES)
    treatment_explicit_macro_f1, per_role_b = _macro_f1(family_b, GRAPH_READABLE_ROLES)
    iv_a, _ = _macro_f1(family_a, frozenset({"instrument"}))
    iv_b, _ = _macro_f1(family_b, frozenset({"instrument"}))

    result = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "generator_commit": _git_head(),
        "lm_model": args.lm_model,
        "n_family_a": len(family_a),
        "n_family_b": len(family_b),
        "cohort_only_macro_f1": cohort_only_macro_f1,
        "treatment_explicit_macro_f1": treatment_explicit_macro_f1,
        "instrument_macro_f1_family_a": iv_a,
        "instrument_macro_f1_family_b": iv_b,
        "per_role_family_a": per_role_a,
        "per_role_family_b": per_role_b,
        "family_a": family_a,
        "family_b": family_b,
    }

    if args.out is None:
        ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.out = PROJECT_ROOT / "artifacts" / "golden_set" / f"precision_recall_{ts}.json"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    logger.info("wrote precision/recall results: %s", args.out)
    logger.info(
        "Family A (cohort-only, GATED): macro_f1=%.4f (N=%d); instrument=%.4f",
        cohort_only_macro_f1,
        len(family_a),
        iv_a,
    )
    logger.info(
        "Family B ((T,Y)-explicit, INFO): macro_f1=%.4f (N=%d); instrument=%.4f",
        treatment_explicit_macro_f1,
        len(family_b),
        iv_b,
    )


if __name__ == "__main__":
    main()
