"""Offline A/B: multi-model ensemble vs single-Sonnet over the golden set (#242).

Runs the COMPILED causal-role classifier (what production single-Sonnet uses)
under all three ensemble members (Sonnet 4.6 + Opus 4.7 + GPT-5) on the
``causal_role_golden_set.json`` entries whose ground-truth role is leak-relevant
(``descendant`` = leaks, plus the subtle ``collider`` / ``mediator`` boundaries),
and reports a precision-style comparison plus the #242 AC5 signal:

* AC5 case        — single-Sonnet WRONG and the ensemble {right via majority | escalated via split}
* leak FN         — gt=descendant but single-Sonnet says benign (ancestor/confounder/instrument)
* ensemble regression — single-Sonnet right but the ensemble wrong (harm check)
* correlated failure  — all three models (incl GPT-5) wrong together (multi-vendor independence failed)

This is the reproducible harness behind ``docs/plans/242-p8-ab-findings.md``.
Requires live API + credits for all three providers (preflight fails loudly if a
key is missing). Re-run after an Anthropic credit top-up to get an uncontaminated
full dataset:

    python scripts/measure_ensemble_ab.py --roles descendant,collider,mediator

Spend: ~3 calls/entry (~$0.06/entry at current list prices).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(override=True)  # find_dotenv walks up to the repo .env (works from a worktree)

from src.data import causal_role_classifier_ensemble as ens  # noqa: E402
from src.data.causal_role_classifier_loader import load_compiled_classifier  # noqa: E402

GOLDEN = REPO_ROOT / "tests/fixtures/causal_role_golden_set.json"
BENIGN = {"ancestor", "confounder", "instrument"}  # for leak-false-negative detection


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--roles",
        default="descendant,collider,mediator",
        help="Comma-separated ground-truth roles to include (default: leak-relevant).",
    )
    parser.add_argument("--limit", type=int, default=0, help="Cap entries (0 = all).")
    parser.add_argument("--out", type=Path, default=None, help="Write per-entry JSON here.")
    args = parser.parse_args()

    roles = {r.strip() for r in args.roles.split(",") if r.strip()}
    entries = json.loads(GOLDEN.read_text())["entries"]
    sample = [e for e in entries if e.get("ground_truth_role") in roles]
    if args.limit:
        sample = sample[: args.limit]
    print(
        f"golden={len(entries)} roles={sorted(roles)} sample={len(sample)} (~{len(sample) * 3} calls)"
    )

    models = ens._resolve_models()
    ens._preflight_models(models)  # loud if any provider key absent
    classifier = load_compiled_classifier()
    print("models:", models, "| compiled classifier:", classifier is not None)

    rows = []
    n = s_ok = e_ok = splits = misses = ac5 = regress = correlated = leak_fn = leak_fn_caught = 0
    for e in sample:
        gt = e["ground_truth_role"]
        clf = ens.run_ensemble_classification(
            feature_name=e["feature_name"],
            derivation_pseudocode=e.get("derivation_pseudocode", ""),
            dataset_context=e.get("dataset_context", ""),
            models=models,
            classifier=classifier,
        )
        s = next((v.causal_role for v in clf.votes if "sonnet" in v.model), None)
        o = next((v.causal_role for v in clf.votes if "opus" in v.model), None)
        g = next((v.causal_role for v in clf.votes if "gpt" in v.model), None)
        contaminated = s is None or o is None  # provider outage (e.g. credit exhaustion)
        n += 1
        s_correct = s == gt
        e_correct = clf.fused_role == gt
        is_split = clf.agreement == "split"
        s_ok += s_correct
        e_ok += e_correct
        splits += is_split
        s_wrong = s is not None and not s_correct
        misses += s_wrong
        if s_wrong and (e_correct or is_split):
            ac5 += 1
        if s_correct and not e_correct:
            regress += 1
        if all(v is not None and v != gt for v in (s, o, g)):
            correlated += 1
        is_leak_fn = gt == "descendant" and s in BENIGN
        leak_fn += is_leak_fn
        if is_leak_fn and (e_correct or is_split):
            leak_fn_caught += 1
        print(
            f"{e['feature_name'][:40]:40s} gt={gt:10s} S={str(s):10s} O={str(o):10s} "
            f"G={str(g):10s} -> {clf.agreement}/{clf.fused_role}"
            f"{' [CONTAMINATED]' if contaminated else ''}"
        )
        rows.append(
            {
                "feature_name": e["feature_name"],
                "cohort": e.get("cohort"),
                "gt": gt,
                "sonnet": s,
                "opus": o,
                "gpt5": g,
                "agreement": clf.agreement,
                "fused_role": clf.fused_role,
                "contaminated": contaminated,
            }
        )

    clean = [r for r in rows if not r["contaminated"]]
    print(f"\n=== A/B (n={n}, clean={len(clean)}) ===")
    print(f"single-Sonnet correct : {s_ok}/{n}")
    print(f"ensemble role-correct : {e_ok}/{n}  (splits/escalated={splits})")
    print(f"sonnet misses={misses}  AC5(caught|escalated)={ac5}  regressions={regress}")
    print(f"correlated all-3-wrong={correlated}  leak-FN={leak_fn} (caught={leak_fn_caught})")
    if len(clean) != n:
        print(
            f"WARNING: {n - len(clean)} entries CONTAMINATED (provider outage) — exclude from conclusions."
        )
    if args.out:
        args.out.write_text(json.dumps({"rows": rows}, indent=2))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
