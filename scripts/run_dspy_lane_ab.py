#!/usr/bin/env python
"""CLI for the DSPy-lane provider A/B harness.

Subcommands:

emit
    Produce a self-contained bundle (module source + golden set + driver) for
    stdin-piping into the prod container, which runs deployed code on a
    read-only rootfs::

        python scripts/run_dspy_lane_ab.py emit \
            --golden-set tests/fixtures/dspy_lane_golden_queries.json \
            --models openai/gpt-5.6-terra,anthropic/claude-haiku-4-5-20251001 \
            --out /tmp/bundle.py
        docker exec -i e2i_api python - < /tmp/bundle.py

analyze
    Evaluate the pre-registered decision gates for each candidate against the
    baseline from per-model measurement JSONs (signature summaries plus
    optional ragas/e2e blocks) and print a markdown results table::

        python scripts/run_dspy_lane_ab.py analyze \
            --baseline-model openai/gpt-5.6-terra \
            --signature-results results.json \
            --extra extra_measurements.json

    ``--signature-results`` is the JSON captured between RESULTS_JSON_BEGIN /
    RESULTS_JSON_END markers of a bundle run. ``--extra`` maps model name to
    ``{"ragas": {...}, "e2e": {...}}`` blocks gathered by the e2e/RAGAS runs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.optimization.dspy_lane_ab import (  # noqa: E402
    emit_container_script,
    evaluate_gates,
    load_golden_set,
)


def _cmd_emit(args: argparse.Namespace) -> int:
    golden = load_golden_set(args.golden_set)
    script = emit_container_script(
        golden,
        models=args.models.split(",") if args.models else [],
        mode=args.mode,
        e2e_query_ids=args.e2e_ids.split(",") if args.e2e_ids else None,
        conversation_prefix=args.conversation_prefix,
    )
    Path(args.out).write_text(script)
    print(
        f"bundle written: {args.out} ({len(script)} chars, "
        f"{len(golden['queries'])} queries, mode={args.mode})"
    )
    return 0


def _bundle_for(model: str, signature_summary: dict, extra: dict) -> dict:
    block = extra.get(model, {})
    return {
        "signature": signature_summary.get(model, {}),
        "ragas": block.get("ragas"),
        "e2e": block.get("e2e"),
    }


def _cmd_analyze(args: argparse.Namespace) -> int:
    results = json.loads(Path(args.signature_results).read_text())
    summary = results["summary"]
    extra = json.loads(Path(args.extra).read_text()) if args.extra else {}

    baseline = _bundle_for(args.baseline_model, summary, extra)
    candidates = [m for m in summary if m != args.baseline_model]

    print(f"\n## DSPy-lane A/B gates (baseline: {args.baseline_model})\n")
    verdicts = {}
    for model in candidates:
        candidate = _bundle_for(model, summary, extra)
        verdict = evaluate_gates(baseline, candidate)
        verdicts[model] = verdict
        print(f"### {model} - {'ALL GATES PASS' if verdict['all_passed'] else 'FAIL'}\n")
        print("| gate | result | detail |")
        print("|------|--------|--------|")
        for gate in verdict["gates"]:
            print(f"| {gate['name']} | {'PASS' if gate['passed'] else 'FAIL'} | {gate['detail']} |")
        print()

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(verdicts, indent=2))
        print(f"verdicts written: {args.json_out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    emit = sub.add_parser("emit", help="emit self-contained container bundle")
    emit.add_argument("--golden-set", required=True)
    emit.add_argument(
        "--models",
        default="",
        help="comma-separated litellm model ids "
        "(signature mode; e2e mode uses the process DSPY_LM_MODEL)",
    )
    emit.add_argument("--mode", default="signature", choices=["signature", "e2e"])
    emit.add_argument(
        "--e2e-ids", default="", help="comma-separated golden query ids to replay end-to-end"
    )
    emit.add_argument(
        "--conversation-prefix",
        default="dspy-ab",
        help="conversation_id prefix so replay-written learning "
        "signals can be identified and removed",
    )
    emit.add_argument("--out", required=True)
    emit.set_defaults(func=_cmd_emit)

    analyze = sub.add_parser("analyze", help="evaluate decision gates")
    analyze.add_argument("--baseline-model", required=True)
    analyze.add_argument("--signature-results", required=True)
    analyze.add_argument("--extra", default=None)
    analyze.add_argument("--json-out", default=None)
    analyze.set_defaults(func=_cmd_analyze)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
