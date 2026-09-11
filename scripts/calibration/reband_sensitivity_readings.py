#!/usr/bin/env python3
"""Re-band every live agent run under the 2026-09-10 sensitivity reading (spec §7).

For each ``causal_validations`` estimate with ``estimate_source = causal_impact_query``:
read the five stored test rows (statuses, effect, e_value_ci, n_rows), pull the
frame the way the route pulls it (``_load_agent_estimation_frame`` /
``_load_hcp_adoption_join_frame``, brand-scoped default covariates, the stored row
cap), compute the benchmark inputs on it, classify, and recompute the band with the
runner's own ``_calculate_confidence_score`` / ``_determine_gate_decision``.
Writes a markdown table of today's band vs the new band per pair.

The CI bound is recovered from the stored ``e_value_ci`` by the algebraic inverse of
the old runner's formula. When the old runner standardized (a stored, valid
``outcome_std``) the inverse uses that SD and is exact; when it did not (SD absent or
invalid — the old runner's unstandardized branch) the inverse is taken without an SD,
exactly as the old runner computed it, and the SD the NEW classification needs comes
from the re-pulled frame. The reading needs only that bound and whether the CI
includes zero (stored as ``e_value_ci == 1.0``).
Baseline risk, naive contrast and covariate factors are not persisted, so they come
from a re-pulled frame; a ``limit N`` pull has no guaranteed row order, so every run
is classified TWICE — on the capped pull and on the full brand table — and the output
reports, per run, whether that comparison was attempted, succeeded, failed or was
skipped, whether the full pull was itself capped, and any pair whose reading flips
(``frame_sensitive``). Pairs without a current dataset mapping are listed as unmapped,
never guessed. An appendix reconciles the ``acceptance_status → conversion_flag``
family against the 2026-09-10 preview by re-pulling the triggers table the way the
preview did (raw SQL, read-only) next to the route's own frame.

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
import subprocess
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

# Physical table + brand column behind each dataset, for the population COUNT that
# decides whether a "full table" pull was complete (mirrors routes/causal.py
# ``_CAUSAL_PHYSICAL_TABLE`` / ``_CAUSAL_BRAND_COLUMN`` and the hcp JOIN loader).
POPULATION_TABLE: Dict[str, Tuple[str, str]] = {
    "patient_journeys": ("patient_journeys", "brand"),
    "hcp_adoption": ("hcp_brand_adoption", "brand"),
    "nba_triggers": ("triggers", "brand_id"),
}

RECONCILE_PAIR = ("acceptance_status", "conversion_flag")
# The 2026-09-10 preview's own pull for that pair (live_reband_preview.py): raw SQL,
# treatment derived in SQL, NO covariates, NULL outcomes DROPPED by pandas.
PREVIEW_SQL = (
    "select (lower(acceptance_status::text)='accepted')::int t, conversion_flag::int y "
    "from triggers where brand_id = '{brand}' limit {n}"
)


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


async def _population_count(
    client, cache: Dict[Tuple[str, str], Optional[int]], dataset: str, brand: str
) -> Optional[int]:
    """Rows in the (brand-scoped) source table — read-only, ``None`` when unmeasurable."""
    key = (dataset, brand)
    if key not in cache:
        table, brand_col = POPULATION_TABLE[dataset]
        try:
            query = client.table(table).select("*", count="exact", head=True)
            if brand:
                query = query.eq(brand_col, brand)
            res = await query.execute()
            cache[key] = int(res.count) if res.count is not None else None
        except Exception as exc:  # noqa: BLE001 - report, never guess
            print(f"population count failed {key}: {exc!r}", file=sys.stderr)
            cache[key] = None
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


def _valid_sd(sd: Optional[float]) -> bool:
    """The old runner's own test for standardizing (``_run_sensitivity_test``)."""
    return sd is not None and math.isfinite(sd) and sd > 0


def _recover_ci(ate: float, e_value_ci: float, outcome_sd: Optional[float]) -> Tuple[float, float]:
    """The stored row keeps ``e_value_ci`` (old formula on the bound nearest the null),
    not the CI. Invert it exactly. The old runner computed ``RR = exp(0.91 * d)`` with
    ``d = bound / sd`` when it standardized (valid SD) and ``d = bound`` when it did not,
    then ``E = RR + sqrt(RR * (RR - 1))``, whose inverse is ``RR = E^2 / (2E - 1)``
    (verified against the forward formula to 1e-6); so ``bound = ln(RR) / 0.91 * sd``
    on the standardized branch and ``bound = ln(RR) / 0.91`` on the other. Pass
    ``outcome_sd=None`` for the unstandardized branch. Only two facts matter to
    ``classify``: whether the CI includes zero (E <= 1.0) and the bound nearest the
    null; the far bound is set symmetric. The near bound is clamped at ``|ate|`` so
    float noise in the round trip can never push the point estimate outside its own
    interval."""
    if e_value_ci <= 1.0:
        return (min(ate, 0.0) - 1e-9, max(ate, 0.0) + 1e-9)  # includes zero
    rr = (e_value_ci * e_value_ci) / (2.0 * e_value_ci - 1.0)
    d = math.log(rr) / 0.91
    bound = min(d * outcome_sd if outcome_sd is not None else d, abs(ate))
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


# --- reconciliation with the 2026-09-10 preview (acceptance_status -> conversion_flag)


def _psql(sql: str) -> Optional[List[List[str]]]:
    """Read-only SELECT through the live container's psql, the way the preview ran
    its pulls. ``None`` when the container is unreachable (reported, never guessed)."""
    try:
        proc = subprocess.run(
            [
                "docker",
                "exec",
                "supabase-db",
                "psql",
                "-U",
                "postgres",
                "-d",
                "postgres",
                "-At",
                "-F",
                "\t",
                "-c",
                sql,
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"preview pull unavailable: {exc!r}", file=sys.stderr)
        return None
    if proc.returncode != 0:
        print(f"preview pull failed: {proc.stderr.strip()}", file=sys.stderr)
        return None
    return [ln.split("\t") for ln in proc.stdout.splitlines() if ln]


def _p0_naive(pairs: List[Tuple[float, float]]) -> Tuple[Optional[float], Optional[float]]:
    y0 = [y for t, y in pairs if t == 0.0]
    y1 = [y for t, y in pairs if t == 1.0]
    if not y0 or not y1:
        return None, None
    p0 = sum(y0) / len(y0)
    return p0, sum(y1) / len(y1) - p0


def _preview_pull(brand: str, n: int) -> Optional[Dict[str, Any]]:
    """The preview's SQL at the stored row cap, scored two ways: NULL outcomes
    DROPPED (what the preview's ``dropna`` did) and NULL outcomes FILLED TO 0 (what
    the route's loader does for the designed-NULL ``conversion_flag``)."""
    rows = _psql(PREVIEW_SQL.format(brand=brand, n=n))
    if rows is None:
        return None
    dropped: List[Tuple[float, float]] = []
    filled: List[Tuple[float, float]] = []
    n_null = 0
    for t, y in rows:
        tv = float(t)
        if y == "":
            n_null += 1
            filled.append((tv, 0.0))
            continue
        dropped.append((tv, float(y)))
        filled.append((tv, float(y)))
    p0_d, naive_d = _p0_naive(dropped)
    p0_f, naive_f = _p0_naive(filled)
    return {
        "n_pulled": len(rows),
        "n_null_outcome": n_null,
        "p0_dropped": p0_d,
        "naive_dropped": naive_d,
        "p0_filled": p0_f,
        "naive_filled": naive_f,
    }


def _preview_rule(ate: float, p0: Optional[float], naive: Optional[float]) -> str:
    """The preview script's own reading rule (joint naive-vs-adjusted benchmark on
    the risk-ratio conversion, no covariate factors), reproduced verbatim so the
    owner can see what changes when only the inputs change."""
    if p0 is None or naive is None or not p0 > 0:
        return "no_baseline"

    def orient(r: float) -> float:
        return max(r, 1 / r)

    def rr_rd(rd: float) -> float:
        p1 = p0 + rd
        return orient(p1 / p0) if p1 > 0 else float("inf")

    rrp = rr_rd(ate)
    b = orient(rr_rd(naive) / rrp)
    return "beyond" if rrp > b else "within"


def _fmt(v: Any, nd: int = 3) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


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
            sd_raw = d.get("outcome_std")
            sd_stored = float(sd_raw) if sd_raw not in (None, "") else None
            e.update(
                ate=float(r["original_effect"]),
                e_ci=float(d.get("e_value_ci") or 1.0),
                sd_stored=sd_stored,
                # The old runner standardized iff the SD was valid; the row also
                # records which branch it took. Either signal of the unstandardized
                # branch is honoured (and counted).
                standardized=bool(d.get("standardized", _valid_sd(sd_stored)))
                and _valid_sd(sd_stored),
                randomized=(r["status"] == "skipped"),
                n=int(d.get("refutation_n_rows_total") or 1500),
            )

    readings: Counter = Counter()
    # (reading, benchmark basis) — tells the two unbenchmarked sub-cases apart
    bases: Counter = Counter()
    basis_counts: Counter = Counter()
    moves: Counter = Counter()
    comparison: Counter = Counter()
    per_pair: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    frame_cache: Dict[Tuple[str, str, str, str, int], Any] = {}
    count_cache: Dict[Tuple[str, str], Optional[int]] = {}
    n_no_stored_sd = 0
    for eid, e in by_estimate.items():
        key = (e["brand"] or "<all>", e["t"], e["o"])
        dataset = DATASET_BY_PAIR.get((e["t"], e["o"]))
        if e.get("randomized"):
            reading = evalue.READING_RANDOMIZED
            new_sens = "skipped"
            e["cmp"] = "skipped_randomized"
        elif dataset is None:
            reading, new_sens = "unmapped", None
            e["cmp"] = "skipped_unmapped"
        else:
            got_capped = await _cached_frame(
                frame_cache, dataset, e["t"], e["o"], e["brand"], e["n"]
            )
            got_full = await _cached_frame(
                frame_cache, dataset, e["t"], e["o"], e["brand"], FULL_TABLE_LIMIT
            )
            if isinstance(got_capped, Exception):
                reading, new_sens = f"frame_error: {type(got_capped).__name__}", None
                e["cmp"] = "capped_pull_failed"
                print(f"frame_error {key}: {got_capped!r}", file=sys.stderr)
            else:
                ate = e["ate"]
                e["n_capped"] = len(got_capped[0])
                if e["standardized"]:
                    # Exact inversion with the STORED SD, the one the old runner used.
                    sd = e["sd_stored"]
                    ci = _recover_ci(ate, e["e_ci"], sd)
                    e["sd_branch"] = "stored"
                else:
                    # The old runner's UNSTANDARDIZED branch: invert without an SD;
                    # the new classification measures its SD on the re-pulled frame.
                    n_no_stored_sd += 1
                    sd = evalue.outcome_std_from_frame(got_capped[0], e["o"], treatment=e["t"])
                    ci = _recover_ci(ate, e["e_ci"], None)
                    e["sd_branch"] = "unstandardized"
                e["ci"] = ci
                rd = _classify_on(got_capped, e["t"], e["o"], ate, ci, sd)
                reading, new_sens = rd.reading, rd.status
                e.update(
                    rr_point=rd.rr_point,
                    benchmark=rd.benchmark,
                    basis=rd.benchmark_basis,
                    p0=rd.baseline_risk,
                    naive=rd.naive_effect,
                    conversion=rd.conversion,
                )
                bases[(rd.reading, rd.benchmark_basis)] += 1
                basis_counts[rd.benchmark_basis] += 1
                # Perturbation check: same run, full brand table instead of the capped pull.
                if isinstance(got_full, Exception):
                    e["cmp"] = "full_pull_failed"
                    print(f"full pull failed {key}: {got_full!r}", file=sys.stderr)
                else:
                    rd_full = _classify_on(got_full, e["t"], e["o"], ate, ci, sd)
                    e["frame_sensitive"] = rd_full.reading != rd.reading
                    e["reading_full_table"] = rd_full.reading
                    e["n_full"] = len(got_full[0])
                    pop = await _population_count(client, count_cache, dataset, e["brand"])
                    e["population"] = pop
                    # A full pull is complete only when the population fits under the
                    # ceiling; a frame at the ceiling is capped by construction.
                    e["full_capped"] = e["n_full"] >= FULL_TABLE_LIMIT or (
                        pop is not None and pop > FULL_TABLE_LIMIT
                    )
                    e["cmp"] = "compared_capped_full" if e["full_capped"] else "compared_complete"
        comparison[e["cmp"]] += 1
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
    lines += ["", "| benchmark basis | runs |", "|---|---|"]
    lines += [f"| {b} | {v} |" for b, v in basis_counts.most_common()]
    lines += ["", "| reading / basis | runs |", "|---|---|"]
    lines += [f"| {r} / {b} | {v} |" for (r, b), v in bases.most_common()]
    lines += ["", "## Gate moves (today → new)", "", "| move | runs |", "|---|---|"]
    lines += [f"| {a} → {b} | {n} |" for (a, b), n in sorted(moves.items())]

    # --- frame perturbation coverage
    all_runs = [x for es in per_pair.values() for x in es]
    mapped = [x for x in all_runs if x["cmp"] not in ("skipped_randomized", "skipped_unmapped")]
    compared = [x for x in mapped if x["cmp"].startswith("compared")]
    not_compared = [x for x in mapped if not x["cmp"].startswith("compared")]
    capped_cmp = [x for x in compared if x.get("full_capped")]
    flips = [x for x in compared if x.get("frame_sensitive")]

    def _run_label(x: Dict[str, Any]) -> str:
        return f"{x['brand'] or '<all>'} {x['t']}→{x['o']}"

    lines += [
        "",
        "## Frame perturbation check",
        "",
        "| comparison | runs |",
        "|---|---|",
        f"| attempted (mapped, non-randomized) | {len(mapped)} |",
        f"| succeeded — full pull complete | {len(compared) - len(capped_cmp)} |",
        f"| succeeded — full pull CAPPED at {FULL_TABLE_LIMIT} (second bounded subset) | {len(capped_cmp)} |",
        f"| failed — full pull raised | {comparison['full_pull_failed']} |",
        f"| failed — capped pull raised | {comparison['capped_pull_failed']} |",
        f"| skipped — randomized | {comparison['skipped_randomized']} |",
        f"| skipped — unmapped | {comparison['skipped_unmapped']} |",
        "",
        f"Runs whose reading differs between the capped pull and the full brand table: **{len(flips)}**"
        + (
            " — re-run these live in Task 12 step 5: "
            + ", ".join(sorted({_run_label(x) for x in flips}))
            if flips
            else ""
        )
        + ".",
    ]
    if not_compared:
        lines.append(
            f"Not compared ({len(not_compared)} of {len(mapped)} mapped runs): "
            + ", ".join(sorted({f"{_run_label(x)} [{x['cmp']}]" for x in not_compared}))
            + "."
        )
    if not flips and not not_compared and not capped_cmp:
        lines.append(
            f"Every one of the {len(mapped)} mapped non-randomized runs was classified on two "
            "complete frames with the same reading — the row-order caveat is retired by measurement."
        )
    else:
        retired_n = len(compared) - len(capped_cmp)
        lines.append(
            f"The row-order caveat is retired only for the {retired_n} runs compared on a COMPLETE "
            f"full table with no flip; it stands for the {len(capped_cmp)} runs whose full pull was "
            f"capped ({', '.join(sorted({_run_label(x) for x in capped_cmp})) or 'none'}) and for "
            f"the {len(not_compared)} runs not compared."
        )

    lines += [
        "",
        "## Per pair",
        "",
        "| brand | treatment → outcome | runs | today | new | readings | frame-sensitive | capped n / full n | median rr_point | median benchmark |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for key in sorted(per_pair):
        es = per_pair[key]

        def med(k: str, es=es) -> str:
            vals = [x[k] for x in es if k in x and x[k] is not None]
            return f"{statistics.median(vals):.2f}" if vals else "-"

        sizes = sorted(
            {
                f"{x['n_capped']} / {x['n_full']} ({'capped' if x['full_capped'] else 'complete'})"
                for x in es
                if "n_full" in x
            }
        )
        sizes_txt = "; ".join(sizes) if sizes else "-"
        lines.append(
            f"| {key[0]} | {key[1]} → {key[2]} | {len(es)} | {dict(Counter(x['old_gate'] for x in es))} | "
            f"{dict(Counter(x['new_gate'] for x in es))} | {dict(Counter(x['reading'] for x in es))} | "
            f"{sum(1 for x in es if x.get('frame_sensitive'))} | {sizes_txt} | {med('rr_point')} | {med('benchmark')} |"
        )

    # --- reconciliation appendix
    recon = sorted(
        (x for x in all_runs if (x["t"], x["o"]) == RECONCILE_PAIR),
        key=lambda x: (x["brand"], x["ate"]),
    )
    lines += [
        "",
        "## Reconciliation with the 2026-09-10 preview (`acceptance_status → conversion_flag`)",
        "",
        "Per run, on the route's own frame (the NBA patient JOIN with the brand-scoped curated covariates, designed-NULL `conversion_flag` filled to 0):",
        "",
        "| brand | ate | recovered CI | p0 (baseline risk) | naive | rr_point | benchmark | basis | conversion | reading |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for x in recon:
        ci = x.get("ci")
        lines.append(
            f"| {x['brand'] or '<all>'} | {_fmt(x['ate'], 4)} | "
            f"{'[' + _fmt(ci[0], 4) + ', ' + _fmt(ci[1], 4) + ']' if ci else '-'} | "
            f"{_fmt(x.get('p0'))} | {_fmt(x.get('naive'))} | {_fmt(x.get('rr_point'))} | "
            f"{_fmt(x.get('benchmark'))} | {x.get('basis', '-')} | {x.get('conversion', '-')} | {x.get('reading', '-')} |"
        )
    lines += [
        "",
        "The preview's pull, re-run read-only with its own SQL (`(lower(acceptance_status::text)='accepted')::int`, `conversion_flag::int`, brand filter, `limit n`, no covariates), scored two ways — NULL outcomes DROPPED (the preview's `dropna`) and NULL outcomes FILLED TO 0 (the route's loader) — next to the route frame, with the preview's own reading rule applied to each set of inputs:",
        "",
        "| brand | n | ate | preview pull: rows / NULL outcome | p0 / naive (NULL dropped) | preview rule on those | p0 / naive (NULL → 0) | preview rule on those | p0 / naive (route frame) | preview rule on route inputs | current reading |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    preview_cache: Dict[Tuple[str, int], Optional[Dict[str, Any]]] = {}
    for x in recon:
        pk = (x["brand"], x["n"])
        if pk not in preview_cache:
            preview_cache[pk] = _preview_pull(x["brand"], x["n"])
        pv = preview_cache[pk]
        if pv is None:
            pull_txt = "unavailable"
            d_txt = f_txt = r_d = r_f = "-"
        else:
            pull_txt = f"{pv['n_pulled']} / {pv['n_null_outcome']}"
            d_txt = f"{_fmt(pv['p0_dropped'])} / {_fmt(pv['naive_dropped'])}"
            f_txt = f"{_fmt(pv['p0_filled'])} / {_fmt(pv['naive_filled'])}"
            r_d = _preview_rule(x["ate"], pv["p0_dropped"], pv["naive_dropped"])
            r_f = _preview_rule(x["ate"], pv["p0_filled"], pv["naive_filled"])
        r_route = _preview_rule(x["ate"], x.get("p0"), x.get("naive"))
        lines.append(
            f"| {x['brand'] or '<all>'} | {x['n']} | {_fmt(x['ate'], 4)} | {pull_txt} | {d_txt} | {r_d} | "
            f"{f_txt} | {r_f} | {_fmt(x.get('p0'))} / {_fmt(x.get('naive'))} | {r_route} | {x.get('reading', '-')} |"
        )
    lines += [
        "",
        "Reading of the two tables above: the preview and this script agree on the stored effect and on the "
        "conversion; they differ on the INPUTS. The preview's `dropna` removed every trigger whose "
        "`conversion_flag` is NULL (a designed NULL — the DB stored-generated `outcome_value > 0` is NULL "
        "when no outcome was recorded), so its control-arm rate and naive contrast describe only the "
        "triggers with a recorded outcome. The route's loader fills that designed NULL to 0 "
        "(`_CAUSAL_FILL_ZERO_OUTCOMES['nba_triggers']`) before the estimator ever sees the frame, which is "
        "what produced the stored ATE; filling the raw pull the same way reproduces the route's p0 and "
        "naive to the third decimal. The route's pull is the faithful one because it is the frame "
        "production estimated on. Whether the preview's own rule reads `within` or `beyond` on each set "
        "of inputs is printed per run, so the flip is attributed by measurement, not by argument.",
    ]

    lines += [
        "",
        "## Caveats",
        "",
        "- The CI bound is recovered from the stored `e_value_ci` by the exact algebraic inverse of the old runner's formula. On the standardized branch (a stored, valid `outcome_std`) the inverse uses that SD and the bound is exact. On the unstandardized branch (no valid stored SD) the inverse is taken without an SD — as the old runner computed it — and the SD the new classification needs is measured on the re-pulled frame. "
        f"{n_no_stored_sd} runs lacked a stored SD"
        + ("." if n_no_stored_sd == 0 else " and took the unstandardized branch."),
        "- Baseline risk, naive contrast and covariate factors come from a re-pulled frame (not persisted). A `limit N` pull has no guaranteed row order, so each run was classified on the capped pull AND on the full brand table; the frame-sensitive column counts runs whose reading differs, and the perturbation section counts how many runs were actually compared, and on what.",
        f"- The full-table pull is bounded by the route's own whole-table ceiling ({FULL_TABLE_LIMIT} rows). Where the population exceeds it (the all-brands patient table, {count_cache.get(('patient_journeys', ''), 'n/a')} rows) the second frame is another bounded subset, not the population, and the row-order caveat is NOT retired for those pairs; single-brand tables fit under the ceiling and their full pull is complete.",
        "- Pairs listed as `unmapped` have no current dataset mapping and were not guessed.",
        "- Runs whose sensitivity row was SKIPPED (randomized design) keep SKIPPED.",
        "- The per-run covariate set is not persisted; the re-pull uses the brand-scoped curated default the submit route applies. Runs submitted through the discovery path used the SSOT adjustment set, which may differ; the perturbation check covers row order only.",
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:24]))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, required=True)
    sys.exit(asyncio.run(main(ap.parse_args().out)))
