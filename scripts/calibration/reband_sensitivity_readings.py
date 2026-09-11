#!/usr/bin/env python3
"""Re-band every live agent run under the 2026-09-10 sensitivity reading (spec §7).

For each ``causal_validations`` estimate with ``estimate_source = causal_impact_query``:
read the five stored test rows (statuses, effect, e_value_ci, n_rows), pull the
frame the way the route pulls it (``_load_agent_estimation_frame`` /
``_load_hcp_adoption_join_frame``, brand-scoped default covariates, the stored row
cap), compute the benchmark inputs on it, classify, and recompute the band with the
runner's own ``_calculate_confidence_score`` / ``_determine_gate_decision``.
Writes a markdown table of today's band vs the new band per pair.

The CI bound is recovered EXACTLY from the stored ``e_value_ci`` and the stored
``outcome_std`` (algebraic inverse of the old runner's formula); the reading needs
only that bound and whether the CI includes zero (stored as ``e_value_ci == 1.0``).
Baseline risk, naive contrast and covariate factors are not persisted, so they come
from a re-pulled frame; a ``limit N`` pull has no guaranteed row order, so every run
is classified TWICE — on the capped pull and on the full brand table — and the output
flags any pair whose reading flips (``frame_sensitive``). Pairs without a current
dataset mapping are listed as unmapped, never guessed.

Run (from the repo root, venv active, the live stack reachable):

    .venv/bin/python scripts/calibration/reband_sensitivity_readings.py \
        --out docs/demos/results/$(date +%F)_sensitivity_calibration/reband.md
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.causal_engine import evalue  # noqa: E402
from src.causal_engine.refutation_runner import (  # noqa: E402
    RefutationResult,
    RefutationRunner,
    RefutationStatus,
    RefutationTestType,
)

# (treatment, outcome) -> dataset, for the pairs the live table holds. Anything
# absent is reported as unmapped.
DATASET_BY_PAIR: Dict[Tuple[str, str], str] = {
    **{
        (t, o): "patient_journeys"
        for t in (
            "treatment_arm",
            "copay_support",
            "psp_enrolled",
            "rep_detailing_high",
            "sample_dropped",
            "trigger_accepted",
            "treatment_initiated",
            "urticaria_severity_uas7",
            "disease_stage",
        )
        for o in ("treatment_initiated", "adherent_180d", "low_gap_180d", "persistent_180d")
    },
    ("peer_influence_score", "adopted"): "hcp_adoption",
    ("treatment_arm", "adopted"): "hcp_adoption",
    ("control_group_flag", "action_taken"): "nba_triggers",
    ("acceptance_status", "conversion_flag"): "nba_triggers",
}


async def _rows(client) -> List[Dict[str, Any]]:
    res = await (
        client.table("causal_validations")
        .select(
            "estimate_id,test_type,status,original_effect,brand,treatment_variable,"
            "outcome_variable,details_json,gate_decision,confidence_score"
        )
        .eq("estimate_source", "causal_impact_query")
        .limit(5000)
        .execute()
    )
    return res.data or []


def _details(raw: Any) -> Dict[str, Any]:
    """``details_json`` is a jsonb object on every live sensitivity row (measured
    2026-09-10: 124 object / 0 string), but an earlier lane found the agent path
    double-encoding it as a JSON *string*; decode that shape too, never guess keys."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return {}
    return raw if isinstance(raw, dict) else {}


async def _frame(dataset: str, treatment: str, outcome: str, brand: Optional[str], limit: int):
    from src.api.routes.causal import (
        _CAUSAL_DATASET_SPECS,
        _brand_scoped_covariates,
        _load_agent_estimation_frame,
    )

    spec = _CAUSAL_DATASET_SPECS[dataset]
    # The submit route brand-scopes the curated default for EVERY brand value —
    # brand=None (all-brands) keeps only the universals among the clinical set.
    covariates = [
        c
        for c in _brand_scoped_covariates(list(spec["covariate"]), brand or None)
        if c not in (treatment, outcome)
    ]
    df, select_cols = await _load_agent_estimation_frame(
        dataset=dataset,
        treatment_var=treatment,
        outcome_var=outcome,
        covariates=covariates,
        limit=limit,
        brand=brand or None,
    )
    covs = [c for c in select_cols if c not in (treatment, outcome) and c in df.columns]
    return df, covs


FULL_TABLE_LIMIT = (
    20000  # the route's own ceiling for a whole-table read (routes/causal.py ``limit(20000)``)
)


async def _cached_frame(
    cache: Dict[Any, Any], dataset: str, t: str, o: str, brand: str, limit: int
):
    key = (dataset, brand, t, o, limit)
    if key not in cache:
        try:
            cache[key] = await _frame(dataset, t, o, brand, limit)
        except Exception as exc:  # noqa: BLE001 - report, never guess
            cache[key] = exc
    return cache[key]


def _classify_on(got, t: str, o: str, ate: float, ci: Tuple[float, float], sd: float):
    df, covs = got
    inputs = evalue.benchmark_inputs_from_frame(df, t, o, covs)
    return evalue.classify(
        ate,
        ci,
        randomized=False,
        baseline_risk=inputs.baseline_risk,
        outcome_std=sd,
        naive_effect=inputs.naive_effect,
        covariate_factors=inputs.covariate_bias_factors,
        n_rows=len(df),
        covariates_measured=inputs.covariates_measured,
    )


def _recover_ci(ate: float, e_value_ci: float, outcome_sd: float) -> Tuple[float, float]:
    """The stored row keeps ``e_value_ci`` (old SMD formula on the bound nearest the
    null), not the CI. Invert it exactly. The old runner computed
    ``RR = exp(0.91 * bound / sd)`` and ``E = RR + sqrt(RR * (RR - 1))``, whose inverse
    is ``RR = E^2 / (2E - 1)`` (verified against the forward formula to 1e-6), then
    ``d = ln(RR) / 0.91``, ``bound = d * sd``. Only two facts matter to ``classify``:
    whether the CI includes zero (E <= 1.0) and the bound nearest the null; the far
    bound is set symmetric. The near bound is clamped at ``|ate|`` so float noise in
    the round trip can never push the point estimate outside its own interval."""
    if e_value_ci <= 1.0:
        return (min(ate, 0.0) - 1e-9, max(ate, 0.0) + 1e-9)  # includes zero
    rr = (e_value_ci * e_value_ci) / (2.0 * e_value_ci - 1.0)
    bound = min((math.log(rr) / 0.91) * outcome_sd, abs(ate))
    near, far = bound, max(bound, 2.0 * abs(ate) - bound)
    return (near, far) if ate >= 0 else (-far, -near)


def _band(runner: RefutationRunner, statuses: Dict[str, str]) -> Tuple[str, float]:
    tests = [
        RefutationResult(RefutationTestType(name), RefutationStatus(st), 0.0, 0.0)
        for name, st in statuses.items()
        if name in RefutationTestType._value2member_map_
    ]
    conf = runner._calculate_confidence_score(tests)
    return runner._determine_gate_decision(tests, conf).value, conf


async def main(out: Path) -> int:
    from dotenv import load_dotenv

    # A worktree (``<main>/.worktrees/<lane>``) carries no ``.env``; fall back to the
    # main checkout's. Existing process env always wins (``override=False``).
    env_candidates = [REPO / ".env"]
    if REPO.parent.name == ".worktrees":
        env_candidates.append(REPO.parents[1] / ".env")
    for env_file in env_candidates:
        if env_file.is_file():
            load_dotenv(env_file)
            break

    from src.memory.services.factories import get_async_supabase_client

    client = await get_async_supabase_client()
    if client is None:
        print("no supabase client", file=sys.stderr)
        return 2
    runner = RefutationRunner()
    by_estimate: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"tests": {}})
    for r in await _rows(client):
        e = by_estimate[r["estimate_id"]]
        e["tests"][r["test_type"]] = r["status"]
        e.update(
            brand=r.get("brand") or "",
            t=r["treatment_variable"],
            o=r["outcome_variable"],
            old_gate=r["gate_decision"],
        )
        if r["test_type"] == "sensitivity_e_value":
            d = _details(r.get("details_json"))
            e.update(
                ate=float(r["original_effect"]),
                e_ci=float(d.get("e_value_ci") or 1.0),
                sd_stored=(float(d["outcome_std"]) if d.get("outcome_std") else None),
                randomized=(r["status"] == "skipped"),
                n=int(d.get("refutation_n_rows_total") or 1500),
            )

    readings: Counter = Counter()
    # (reading, benchmark basis) — tells the two unbenchmarked sub-cases apart
    bases: Counter = Counter()
    moves: Counter = Counter()
    per_pair: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    frame_cache: Dict[Tuple[str, str, str, str, int], Any] = {}
    for eid, e in by_estimate.items():
        key = (e["brand"] or "<all>", e["t"], e["o"])
        dataset = DATASET_BY_PAIR.get((e["t"], e["o"]))
        if e.get("randomized"):
            reading = evalue.READING_RANDOMIZED
            new_sens = "skipped"
        elif dataset is None:
            reading, new_sens = "unmapped", None
        else:
            got_capped = await _cached_frame(
                frame_cache, dataset, e["t"], e["o"], e["brand"], e["n"]
            )
            got_full = await _cached_frame(
                frame_cache, dataset, e["t"], e["o"], e["brand"], FULL_TABLE_LIMIT
            )
            if isinstance(got_capped, Exception):
                reading, new_sens = f"frame_error: {type(got_capped).__name__}", None
                print(f"frame_error {key}: {got_capped!r}", file=sys.stderr)
            else:
                # Exact inversion of the stored E-value with the STORED outcome SD (the one
                # the old runner used); the re-pulled frame's SD is only a fallback.
                ate = e["ate"]
                sd = e["sd_stored"] or evalue.outcome_std_from_frame(
                    got_capped[0], e["o"], treatment=e["t"]
                )
                ci = _recover_ci(ate, e["e_ci"], sd)
                rd = _classify_on(got_capped, e["t"], e["o"], ate, ci, sd)
                reading, new_sens = rd.reading, rd.status
                e.update(rr_point=rd.rr_point, benchmark=rd.benchmark, basis=rd.benchmark_basis)
                bases[(rd.reading, rd.benchmark_basis)] += 1
                # Perturbation check: same run, full brand table instead of the capped pull.
                if not isinstance(got_full, Exception):
                    rd_full = _classify_on(got_full, e["t"], e["o"], ate, ci, sd)
                    e["frame_sensitive"] = rd_full.reading != rd.reading
                    e["reading_full_table"] = rd_full.reading
        readings[reading] += 1
        e["reading"] = reading
        if new_sens is None:
            e["new_gate"] = "?"
        else:
            statuses = dict(e["tests"])
            statuses["sensitivity_e_value"] = new_sens
            e["new_gate"], e["new_conf"] = _band(runner, statuses)
            moves[(e["old_gate"], e["new_gate"])] += 1
        per_pair[key].append(e)

    lines = [
        f"# Live re-band under the 2026-09-10 sensitivity reading ({date.today().isoformat()})",
        "",
        f"Estimates: {len(by_estimate)} (`estimate_source = causal_impact_query`).",
        "",
        "## Readings",
        "",
        "| reading | runs |",
        "|---|---|",
    ]
    lines += [f"| {k} | {v} |" for k, v in readings.most_common()]
    lines += ["", "| reading / basis | runs |", "|---|---|"]
    lines += [f"| {r} / {b} | {v} |" for (r, b), v in bases.most_common()]
    lines += ["", "## Gate moves (today → new)", "", "| move | runs |", "|---|---|"]
    lines += [f"| {a} → {b} | {n} |" for (a, b), n in sorted(moves.items())]
    flips = [(k, x) for k, es in per_pair.items() for x in es if x.get("frame_sensitive")]
    lines += [
        "",
        "## Frame perturbation check",
        "",
        f"Runs whose reading differs between the capped pull and the full brand table: **{len(flips)}**"
        + (
            " — re-run these live in Task 12 step 5: "
            + ", ".join(f"{k[0]} {k[1]}→{k[2]}" for k, _ in flips)
            if flips
            else " — the row-order caveat is retired by measurement."
        ),
    ]
    lines += [
        "",
        "## Per pair",
        "",
        "| brand | treatment → outcome | runs | today | new | readings | frame-sensitive | median rr_point | median benchmark |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for key in sorted(per_pair):
        es = per_pair[key]

        def med(k: str, es=es) -> str:
            vals = [x[k] for x in es if k in x and x[k] is not None]
            return f"{statistics.median(vals):.2f}" if vals else "-"

        lines.append(
            f"| {key[0]} | {key[1]} → {key[2]} | {len(es)} | {dict(Counter(x['old_gate'] for x in es))} | "
            f"{dict(Counter(x['new_gate'] for x in es))} | {dict(Counter(x['reading'] for x in es))} | "
            f"{sum(1 for x in es if x.get('frame_sensitive'))} | {med('rr_point')} | {med('benchmark')} |"
        )
    lines += [
        "",
        "## Caveats",
        "",
        "- The CI bound is recovered EXACTLY from the stored `e_value_ci` and the stored `outcome_std` (algebraic inverse of the old formula); the reading needs only that bound and whether the CI includes zero.",
        "- Baseline risk, naive contrast and covariate factors come from a re-pulled frame (not persisted). A `limit N` pull has no guaranteed row order, so each run was classified on the capped pull AND on the full brand table; the frame-sensitive column counts runs whose reading differs.",
        f"- The full-table pull is capped at the route's own whole-table ceiling ({FULL_TABLE_LIMIT} rows); every single-brand table fits under it, the all-brands patient table does not.",
        "- Pairs listed as `unmapped` have no current dataset mapping and were not guessed.",
        "- Runs whose sensitivity row was SKIPPED (randomized design) keep SKIPPED.",
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:20]))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, required=True)
    sys.exit(asyncio.run(main(ap.parse_args().out)))
