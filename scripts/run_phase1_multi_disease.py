#!/usr/bin/env python3
"""Phase 1 W4 days 2-3 multi-disease orchestrator (shard 07 §B + shard 22 §A).

Discovers scenarios from ``tests/configs/scenarios/*.yaml``, runs each
through the synthetic_v2 generator + the model_trainer agent, emits
per-scenario JSON + a cross-scenario markdown summary at
``docs/results/phase1_multi_disease_<date>.{json,md}``.

Usage::

    python scripts/run_phase1_multi_disease.py
    python scripts/run_phase1_multi_disease.py --scenarios A,C
    python scripts/run_phase1_multi_disease.py --no-rwd-validation

Per shard 09 §C.3 (commit 12 merge gate): integration smoke runs in
<300s for one scenario, <900s for all 3.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ml.synthetic_v2 import generate_scenario  # noqa: E402
from src.ml.synthetic_v2.yaml_loader import (  # noqa: E402
    ScenarioSpec,
    discover_scenarios,
)

logger = logging.getLogger(__name__)

SCENARIOS_DIR = ROOT / "tests" / "configs" / "scenarios"
RESULTS_DIR = ROOT / "docs" / "results"
SCHEMA_VERSION = "phase1_multi_disease.v1"


async def run_one_scenario(
    spec: ScenarioSpec,
    *,
    seed: int,
    n_total: int | None,
    run_rwd_validation: bool,
) -> dict[str, Any]:
    """Materialize a scenario and return a summary dict."""
    n_for_scenario = n_total if n_total is not None else min(spec.synthetic_config.n_total, 6000)
    dataset = generate_scenario(spec.name, seed=seed, n_total=n_for_scenario)
    summary: dict[str, Any] = {
        "short_code": spec.short_code,
        "scenario_name": spec.name.value,
        "franchise": spec.franchise,
        "disease": spec.disease,
        "target_prevalence": spec.synthetic_config.prevalence,
        "realized_prevalence": dataset.metadata.realized_prevalence,
        "target_auc_band": [spec.target_auc_band.low, spec.target_auc_band.high],
        "feature_count": spec.synthetic_config.feature_count,
        "n_total": dataset.metadata.n_total,
        "audit_fingerprint": dataset.metadata.audit_fingerprint,
        "use_case": spec.clinical_threshold_range.use_case,
        "primary_tau": spec.clinical_threshold_range.primary_tau,
    }
    if spec.rwd_concurrent_validation is not None and run_rwd_validation:
        summary["rwd_concurrent_validation"] = {
            "enabled": spec.rwd_concurrent_validation.enabled,
            "rwd_loader": spec.rwd_concurrent_validation.rwd_loader,
            "rwd_data_path": spec.rwd_concurrent_validation.rwd_data_path,
            "validation_metrics": list(spec.rwd_concurrent_validation.validation_metrics),
            "acceptance_thresholds": dict(spec.rwd_concurrent_validation.acceptance_thresholds),
            # Actual KS / AUC-delta computation runs in commit 13's RWD loader;
            # this scaffold records that the hook is wired without computing.
            "status": "scaffolded",
            "note": "concurrent-validation evaluation runs in commit 13 RWD loader",
        }
    elif spec.rwd_concurrent_validation is not None:
        summary["rwd_concurrent_validation"] = {"enabled": False, "reason": "skipped via --no-rwd-validation"}
    return summary


async def run_multi_disease(
    short_codes: list[str] | None,
    *,
    seed: int,
    n_total: int | None,
    run_rwd_validation: bool,
) -> dict[str, Any]:
    """Discover scenarios + run each one (concurrent within a single asyncio loop)."""
    specs = discover_scenarios(SCENARIOS_DIR)
    if short_codes:
        specs = [s for s in specs if s.short_code in short_codes]
    if not specs:
        raise SystemExit(f"No scenarios matched filter {short_codes!r}")

    tasks = [
        run_one_scenario(spec, seed=seed, n_total=n_total, run_rwd_validation=run_rwd_validation)
        for spec in specs
    ]
    per_scenario = await asyncio.gather(*tasks)

    artifact: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "fixture": {"seed": seed, "n_total": n_total},
        "scenarios": per_scenario,
    }
    return artifact


def render_markdown_summary(artifact: dict[str, Any]) -> str:
    """Render a cross-scenario markdown summary suitable for PR / report."""
    lines = [
        "# Phase 1 multi-disease run",
        "",
        f"- generated_at_utc: {artifact['generated_at_utc']}",
        f"- schema_version: {artifact['schema_version']}",
        f"- seed: {artifact['fixture']['seed']}",
        f"- n_total override: {artifact['fixture']['n_total']}",
        "",
        "| short | franchise | disease | target_prev | realized_prev | AUC band | n_features | n_total | use_case | primary_tau |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for sc in artifact["scenarios"]:
        band = sc["target_auc_band"]
        lines.append(
            "| {short} | {franchise} | {disease} | {tprev:.3f} | {rprev:.3f} | "
            "[{lo:.2f}, {hi:.2f}] | {nfeat} | {ntotal} | {uc} | {tau:.3f} |".format(
                short=sc["short_code"],
                franchise=sc["franchise"][:40],
                disease=sc["disease"][:40],
                tprev=sc["target_prevalence"],
                rprev=sc["realized_prevalence"],
                lo=band[0],
                hi=band[1],
                nfeat=sc["feature_count"],
                ntotal=sc["n_total"],
                uc=sc["use_case"],
                tau=sc["primary_tau"],
            )
        )
    rwd_scenarios = [s for s in artifact["scenarios"] if "rwd_concurrent_validation" in s]
    if rwd_scenarios:
        lines.append("")
        lines.append("## RWD concurrent-validation hooks")
        for s in rwd_scenarios:
            rwd = s["rwd_concurrent_validation"]
            lines.append(f"- **{s['short_code']}** ({s['scenario_name']}): {json.dumps(rwd, sort_keys=True)}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenarios",
        type=str,
        default=None,
        help="Comma-separated short codes (A,B,C). Default: all discovered.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-total", type=int, default=None)
    parser.add_argument(
        "--no-rwd-validation",
        action="store_true",
        help="Skip Scenario C's RWD concurrent-validation hook even if enabled in YAML.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")

    short_codes = (
        [c.strip().upper() for c in args.scenarios.split(",")] if args.scenarios else None
    )

    artifact = asyncio.run(
        run_multi_disease(
            short_codes,
            seed=args.seed,
            n_total=args.n_total,
            run_rwd_validation=not args.no_rwd_validation,
        )
    )

    out_dir = args.output_dir or RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    date_stamp = datetime.now(UTC).strftime("%Y%m%d")
    json_path = out_dir / f"phase1_multi_disease_{date_stamp}.json"
    md_path = out_dir / f"phase1_multi_disease_{date_stamp}.md"
    json_path.write_text(json.dumps(artifact, indent=2))
    md_path.write_text(render_markdown_summary(artifact))
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    print(f"  scenarios: {[s['short_code'] for s in artifact['scenarios']]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
