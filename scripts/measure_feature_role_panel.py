"""Measure the feature-role panel on a causal frame (Lane E item 4).

Runs ``src.causal_engine.feature_role_panel.build_feature_role_panel`` over a
parquet frame's covariates for ``(manifest source, T, Y)`` with Layer 1
(contracts), Layer 3 (adversarial probe) and Layer 2 (the committed KG cache)
live, and Layer 4 on the LM you choose, then writes ``panel.json`` (the
serialised panel: which layers fired, how many features each decided, the
abstain rate — a null is a finding) and ``summary.md`` next to it.

Layer 4 is a PAID call (``anthropic/claude-sonnet-4-6`` by default; the compiled
classifier ships 192 few-shot demos, measured ~273k prompt chars ≈ 68k tokens
per call — ``docs/demos/results/2026-09-22_lane_e_feature_role_voters/
layer4_prompt_size_probe.txt``). ``--layer4`` therefore defaults to ``fake``:

* ``fake`` — a ``DummyLM`` answering ``confounder`` for every feature Layer 4
  would fire on. Measures WHICH features fire and how many calls the real run
  would make, at zero cost. Provider keys are blanked in-process so nothing
  paid can happen by accident.
* ``off`` — Layer 4 disabled in the per-run profile (three live layers only).
* ``real`` — the paid run. Refuses unless ``--i-accept-cost`` is given, and
  prints the estimate first. This is an OWNER decision (spec §3 Lane E item 4 /
  the lane brief's cost gate).

Real Dupixent-vs-Xolair persistence frame::

    python scripts/measure_feature_role_panel.py \\
        --parquet data/rwd/mart/persistence_causal/e2i_causal_v1_biologic_persistence.parquet \\
        --manifest-source optum_mart \\
        --treatment treatment_dupixent --outcome persistent_at_180d_g28 \\
        --covariates mart-safe \\
        --out docs/demos/results/<date>_<slug>/panel_layer4_fake \\
        --layer4 fake
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Issue #470 discipline: a DSPy-touching CLI script loads .env at import time so
# a `--layer4 real` invocation sees the provider key without a manual export.
# `--layer4 fake` / `off` blank the provider keys in-process afterwards.
load_dotenv()

logger = logging.getLogger(__name__)

# Sonnet 4.6 list prices (USD per million tokens) used ONLY for the estimate the
# owner reads before a real run; the measured prompt size is the input.
_SONNET_INPUT_USD_PER_MTOK = 3.0
_SONNET_OUTPUT_USD_PER_MTOK = 15.0
_MEASURED_PROMPT_TOKENS = 68_261  # layer4_prompt_size_probe.txt (273,047 chars / 4)
_ASSUMED_OUTPUT_TOKENS = 400

_PROVIDER_KEYS = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "AZURE_API_KEY", "AZURE_OPENAI_API_KEY")


def _estimate_cost_usd(n_calls: int) -> float:
    per_call = (
        _MEASURED_PROMPT_TOKENS * _SONNET_INPUT_USD_PER_MTOK
        + _ASSUMED_OUTPUT_TOKENS * _SONNET_OUTPUT_USD_PER_MTOK
    ) / 1_000_000
    return n_calls * per_call


# DummyLM consumes its answer list ONCE (dspy.utils.dummies: a list is popped per
# call, then the call fails to parse) — measured on the first real-frame run:
# 19 Layer-4 attempts, 1 answered, 18 "Layer 4 skipped". The fake must answer
# every call or the firing count is under-reported.
_FAKE_LM_ANSWERS = 8192


def _configure_fake_lm() -> Any:
    import dspy
    from dspy.utils.dummies import DummyLM

    answer = {
        "reasoning": "fake LM (measurement run): no reasoning",
        "causal_role": "confounder",
        "mechanism": "fake LM (measurement run): no mechanism; records which features fire",
        "recommended_remediation": "keep_with_caveat",
    }
    lm = DummyLM([dict(answer) for _ in range(_FAKE_LM_ANSWERS)])
    dspy.configure(lm=lm)
    return lm


def _resolve_covariates(arg: Optional[str], covariates_file: Optional[Path]) -> Optional[list[str]]:
    if covariates_file is not None:
        return [ln.strip() for ln in covariates_file.read_text().splitlines() if ln.strip()]
    if arg is None:
        return None
    if arg == "mart-safe":
        from src.data.manifests import MART_SAFE_FEATURES

        return list(MART_SAFE_FEATURES)
    return [c.strip() for c in arg.split(",") if c.strip()]


def _write_summary(path: Path, panel: dict[str, Any], *, lm_label: str, cost_note: str) -> None:
    la = panel["layer_activity"]
    recs = panel["records"]
    lines = [
        f"# Feature-role panel — {panel['manifest_source']} · {panel['treatment']} → {panel['outcome']}",
        "",
        f"built_at: {panel['built_at']}  ·  n_rows: {panel['n_rows']}  ·  covariates: {len(panel['features'])}",
        f"activation_profile: `{json.dumps(panel['activation_profile'], sort_keys=True)}`",
        f"Layer 4 LM: **{lm_label}**. {cost_note}",
        "",
        "## Which layers fired",
        "",
        "| Layer | Activity |",
        "|---|---|",
        f"| Layer 1 contracts | consulted {la['layer_1']['consulted']}, contracted {la['layer_1']['contracted']}, declared-safe {la['layer_1']['declared_safe']}, post-index {la['layer_1']['post_index']} |",
        f"| Layer 2 KG ({la['layer_2']['mode']}) | cache bound {la['layer_2']['cache_bound']}, with cached edges {la['layer_2']['with_cached_edges']}, signalled {la['layer_2']['signalled']} `{json.dumps(la['layer_2']['signals'], sort_keys=True)}` |",
        f"| Layer 3 adversarial | scored {la['layer_3']['scored']}, pre-joint severities `{json.dumps(la['layer_3']['severity_pre_joint_check'], sort_keys=True)}`, FDR-confident {la['layer_3']['fdr_confident']}, declared-safe immunity applied {la['layer_3']['declared_safe_immunity_applied']}, fdr `{json.dumps(la['layer_3']['fdr'], sort_keys=True)}` |",
        f"| Layer 4 LLM | enabled {la['layer_4']['enabled']}, classifier loaded {la['layer_4']['classifier_loaded']}, fired {la['layer_4']['fired']}, roles `{json.dumps(la['layer_4']['roles'], sort_keys=True)}` |",
        f"| Ensemble | decided_by `{json.dumps(la['ensemble']['decided_by'], sort_keys=True)}`, abstain rate {la['ensemble']['abstain_rate']:.3f}, leak verdicts {la['ensemble']['leak_verdicts']} `{json.dumps(la['ensemble']['leak_sources'], sort_keys=True)}` |",
        "",
        f"promotion_eligibility: `{json.dumps(panel['promotion_eligibility'], sort_keys=True)}` (KG promotion is an owner decision, spec §7)",
        "",
        "## Per feature",
        "",
        "| Feature | L1 (temporal status) | L2 signal | L3 z / pre-joint sev | L4 role | decided_by | final_role | conf | excluded (why) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name in panel["features"]:
        r = recs[name]
        z = r["layer_3"].get("z_score")
        z_s = f"{z:.2f}" if isinstance(z, (int, float)) else "—"
        conf = r["ensemble"].get("confidence")
        conf_s = f"{conf:.2f}" if isinstance(conf, (int, float)) else "—"
        if r["leak_verdict"]:
            excluded = (
                "proven post-index leakage"
                if r["leak_source"] == "layer_1_post_index"
                else "excluded per 3(b), pending temporal review"
            )
        else:
            excluded = ""
        lines.append(
            f"| `{name}` | {r['layer_1']['verdict']} ({r['layer_1'].get('temporal_status', '?')}) | "
            f"{r['layer_2']['signal']} | {z_s} / {r['layer_3'].get('severity_pre_joint_check') or '—'} | "
            f"{r['layer_4'].get('role') or '—'} | {r['ensemble'].get('decided_by') or '—'} | "
            f"{r['ensemble'].get('final_role') or '—'} | {conf_s} | {excluded} |"
        )
    fired = [n for n in panel["features"] if recs[n]["layer_4"]["fired"]]
    proven = [n for n in panel["features"] if recs[n]["leak_source"] == "layer_1_post_index"]
    review = [n for n in panel["features"] if recs[n].get("review_required")]
    lines += [
        "",
        f"Layer 4 fired on {len(fired)} feature(s): {', '.join(f'`{n}`' for n in fired) or '—'}",
        "",
        f"Excluded from the adjustment set — proven post-index leakage ({len(proven)}): "
        + (", ".join(f"`{n}`" for n in proven) or "none"),
        "",
        f"Excluded per spec 3(b), pending temporal review ({len(review)}; Layer-3 high on a column "
        "with no manifest contract — predictiveness of Y, NOT proven timing): "
        + (", ".join(f"`{n}`" for n in review) or "none"),
    ]
    path.write_text("\n".join(lines) + "\n")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--parquet", type=Path, required=True, help="Causal frame (parquet).")
    parser.add_argument("--manifest-source", required=True, help="Registered MANIFEST_SOURCES key.")
    parser.add_argument("--treatment", required=True)
    parser.add_argument("--outcome", required=True)
    parser.add_argument(
        "--covariates",
        default=None,
        help="Comma-separated covariate list, or 'mart-safe' for MART_SAFE_FEATURES (default: every other column).",
    )
    parser.add_argument(
        "--covariates-file", type=Path, default=None, help="One covariate per line."
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="Output directory (panel.json + summary.md)."
    )
    parser.add_argument("--layer4", choices=("fake", "off", "real"), default="fake")
    parser.add_argument("--i-accept-cost", action="store_true", help="Required with --layer4 real.")
    parser.add_argument("--n-permutations", type=int, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(levelname)s %(name)s: %(message)s")

    covariates = _resolve_covariates(args.covariates, args.covariates_file)
    n_candidates = len(covariates) if covariates is not None else None
    cost_note = ""
    if args.layer4 == "real":
        n_calls = n_candidates if n_candidates is not None else 64
        est = _estimate_cost_usd(n_calls)
        cost_note = (
            f"REAL Layer 4: up to {n_calls} paid calls × ~{_MEASURED_PROMPT_TOKENS:,} prompt tokens "
            f"≈ US${est:,.2f} at Sonnet list prices (upper bound: Layer 4 fires only on "
            "moderate / high-and-declared-safe features)."
        )
        print(cost_note, file=sys.stderr)
        if not args.i_accept_cost:
            print(
                "Refusing the paid run without --i-accept-cost (owner decision).", file=sys.stderr
            )
            sys.exit(3)
        lm_label = "real"
    elif args.layer4 == "fake":
        for var in _PROVIDER_KEYS:
            os.environ.pop(var, None)
        _configure_fake_lm()
        lm_label = "fake"
        cost_note = "Fake LM (DummyLM): zero paid calls; measures which features fire."
    else:
        for var in _PROVIDER_KEYS:
            os.environ.pop(var, None)
        lm_label = "off"
        cost_note = "Layer 4 disabled in the per-run profile."

    import pandas as pd

    from src.causal_engine.feature_role_panel import (
        CAUSAL_ACTIVATION_PROFILE,
        build_feature_role_panel_sync,
    )

    profile = dict(CAUSAL_ACTIVATION_PROFILE)
    if args.layer4 == "off":
        profile["adaptive_layer4_enabled"] = False

    frame = pd.read_parquet(args.parquet)
    panel = build_feature_role_panel_sync(
        frame,
        manifest_source=args.manifest_source,
        treatment=args.treatment,
        outcome=args.outcome,
        covariates=covariates,
        activation_profile=profile,
        n_permutations=args.n_permutations,
        seed=args.seed,
        lm_label=lm_label,
    )
    payload = panel.to_dict()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "panel.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _write_summary(args.out / "summary.md", payload, lm_label=lm_label, cost_note=cost_note)

    la = payload["layer_activity"]
    print(
        f"panel: {len(payload['features'])} covariates; L1 post-index {la['layer_1']['post_index']}; "
        f"L2 signalled {la['layer_2']['signalled']}; L3 scored {la['layer_3']['scored']}; "
        f"L4 fired {la['layer_4']['fired']} ({lm_label}); decided_by {Counter(la['ensemble']['decided_by'])}; "
        f"abstain rate {la['ensemble']['abstain_rate']:.3f}; leak verdicts {la['ensemble']['leak_verdicts']}",
        file=sys.stderr,
    )
    print(f"Wrote {args.out / 'panel.json'} and {args.out / 'summary.md'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
