#!/usr/bin/env python3
"""Re-band every live agent ``random_common_cause`` row under the scale-free rule (#2005).

The stored rows were scored by the retired rule, ``delta_percent = |refuted -
original| / max(|original|, 1e-10) * 100`` with PASSED <= 20 %, WARNING <= 30 %,
else FAILED (``PASS_THRESHOLDS["common_cause_delta"]``, deleted 2026-09-11). The
denominator was the effect itself, so a null-ish effect failed on perturbation
noise that a larger effect absorbs. The runner now scores the SAME shift against
the reported interval's standard error, and this script re-bands every stored row
through the runner's OWN scoring helper, ``refutation_runner._score_common_cause_shift``
(the cutoffs are read from ``PASS_THRESHOLDS["common_cause_shift_se"]``, never
restated here)::

    reported_se  = (ci_upper - ci_lower) / (2 * 1.959964)   # from the REPORTED interval
    scale        = sqrt(n / refit_n)  when the refutation ran on a #1419 SUBSAMPLE
                                      (refit_n < n), else 1.0
    shift_se     = |refuted - original| / (reported_se * scale)
    PASSED <= thresholds["pass"], WARNING <= thresholds["warning"], else FAILED
    # thresholds = PASS_THRESHOLDS["common_cause_shift_se"], read at run time

``n`` is the stored ``refutation_n_rows_total`` (the estimation frame the reported
interval came from) and ``refit_n`` the stored ``refutation_n_rows`` (the frame the
refits ran on) -- the same two counts the node now passes to the runner. The
UNSCALED shift is reported beside the scaled one so the disproof re-band (commit
792490184, which scored ``|delta| / reported_se`` without the scale) stays
comparable.

The reported interval is what ``original_ci`` was on the runner call — on the agent
path the estimator's own ``ate_ci_lower / ate_ci_upper`` (``nodes/refutation.py``).
It is not persisted as such, but two stored fields invert to it exactly and are
cross-checked against each other here:

* ``reported``  — the ``bootstrap`` row's ``ci_ratio`` is ``bootstrap_ci_width /
  original_ci_width`` (``_run_bootstrap_test``), so ``original_ci_width =
  bootstrap_ci_width / ci_ratio``. Present on the rows that carry a bootstrap
  interval (29 of the 136 estimates on the agent path, 2026-09-11).
* ``evalue_inv`` — the ``sensitivity_e_value`` row's ``e_value_ci`` is the E-value at
  the reported CI bound nearest the null. Old-format rows (pre Lane D') invert
  through ``reband_sensitivity_readings._recover_ci`` (the 0.91-SMD conversion, with
  the stored SD on the standardized branch, without one otherwise). Lane D' rows
  (key ``rr_ci``) used the risk-ratio conversion ``rr = (p0 + bound) / p0`` on the
  stored ``baseline_risk`` and invert as ``bound = p0 * (rr_ci - 1)`` (a negative
  effect reverses the coding: ``bound = p0 * (1 / rr_ci - 1)``); ``_recover_ci``
  does NOT apply to them. Then ``se = (|ate| - |bound|) / 1.959964``. Rows whose
  CI includes zero store ``e_value_ci == 1.0`` and carry no bound.

Lower-fidelity fallbacks, reported alongside for calibration, never silently:

* ``boot_sd`` / ``boot_ci`` — stdev of the stored ``bootstrap_effects`` (20 row
  resamples) and ``(bci_hi - bci_lo) / 3.92`` — a resampling SE, not the reported one.
* ``naive_frame`` — for rows with no stored bound at all: the frame is re-pulled the
  way the route pulls it (``_load_agent_estimation_frame`` at the stored row cap,
  brand-scoped default covariates, exactly as ``reband_sensitivity_readings._frame``)
  and a covariate-free SE is computed: the two-proportion formula
  ``sqrt(p1(1-p1)/n1 + p0(1-p0)/n0)`` for a binary treatment and binary outcome,
  the classical OLS slope SE of ``y ~ t`` otherwise. Its ratio to the higher-fidelity
  sources is reported wherever both exist so the proxy is calibrated, not assumed.

The perturbation noise itself is also recovered: DoWhy scores the refutation with a
normal test at 20 simulations (``causal_refuter.py`` ``perform_normal_distribution_test``),
``z = (original - mean_refits) / std_refits`` and ``refuted_effect`` IS ``mean_refits``,
so ``std_refits = |delta| / |z|`` with ``z`` from the stored p-value (5 decimals;
``p == 0`` means ``p < 5e-6``, ``z > 4.42``, and only an upper bound on the noise is
known).

The gate is recomputed with the runner's own ``_calculate_confidence_score`` /
``_determine_gate_decision`` (``reband_sensitivity_readings._band``) swapping ONLY
the rcc status. Seeded rows (``estimate_source = causal_paths``) are counted in a
footnote and excluded. Nothing here writes to the database or touches ``src/``.

Run (from the repo root — or a worktree, with the MAIN venv and PYTHONPATH pointed
at the checkout under test — with the live stack reachable):

    set -a; . .env; set +a
    .venv/bin/python scripts/calibration/reband_rcc_rule.py \
        --out docs/demos/results/$(date +%F)_rcc_scale_free/reband.md
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

import src  # noqa: E402

_SRC_FILE = Path(src.__file__).resolve()
print(f"src.__file__ = {_SRC_FILE}")
assert _SRC_FILE.is_relative_to(REPO), f"src resolves outside the checkout under test: {_SRC_FILE}"

import reband_sensitivity_readings as pattern  # noqa: E402  (same directory)

from src.causal_engine.refutation_runner import (  # noqa: E402
    RefutationRunner,
    _score_common_cause_shift,
)

Z975 = 1.959964  # the rule's constant; the agent CI was built by the estimator
P_DECIMALS = 5  # causal_validations.p_value is numeric(6,5)
P_FLOOR = 0.5 * 10 ** (-P_DECIMALS)  # a stored 0.00000 means p < this
# The runner's cutoffs, read from the source of truth (asserted below, never restated).
SE_THRESHOLDS = RefutationRunner.PASS_THRESHOLDS["common_cause_shift_se"]
PASS_SE, WARN_SE = float(SE_THRESHOLDS["pass"]), float(SE_THRESHOLDS["warning"])
PASS_PCT, WARN_PCT = 20.0, 30.0  # the RETIRED rule's cutoffs (the stored statuses)

TESTS = (
    "random_common_cause",
    "bootstrap",
    "sensitivity_e_value",
    "placebo_treatment",
    "data_subset",
)


# --- stored rows (read-only psql, the way the pattern script runs its pulls)


def _rows() -> List[Dict[str, Any]]:
    cols = (
        "estimate_id::text, test_type, status, original_effect::text, refuted_effect::text, "
        "p_value::text, delta_percent::text, gate_decision, confidence_score::text, "
        "coalesce(brand,''), treatment_variable, outcome_variable, details_json::text, "
        "created_at::text"
    )
    got = pattern._psql(
        f"select {cols} from causal_validations where estimate_source='causal_impact_query' "
        "order by created_at"
    )
    if got is None:
        raise SystemExit("psql pull failed — nothing to re-band")
    names = [
        "estimate_id",
        "test_type",
        "status",
        "original_effect",
        "refuted_effect",
        "p_value",
        "delta_percent",
        "gate_decision",
        "confidence_score",
        "brand",
        "t",
        "o",
        "details",
        "created_at",
    ]
    out = []
    for r in got:
        d = dict(zip(names, r, strict=True))
        d["details"] = pattern._details(json.loads(d["details"])) if d["details"] else {}
        for k in (
            "original_effect",
            "refuted_effect",
            "p_value",
            "delta_percent",
            "confidence_score",
        ):
            d[k] = float(d[k]) if d[k] not in (None, "") else None
        out.append(d)
    return out


def _seeded_counts() -> Tuple[int, Dict[str, int]]:
    got = pattern._psql(
        "select status, count(*) from causal_validations where test_type='random_common_cause' "
        "and estimate_source<>'causal_impact_query' group by 1"
    )
    if got is None:
        return -1, {}
    by = {st: int(n) for st, n in got}
    return sum(by.values()), by


# --- standard-error sources


def _se_reported(boot: Optional[Dict[str, Any]]) -> Optional[float]:
    """Reported interval width from ``bootstrap_ci`` and ``ci_ratio`` (exact inversion)."""
    if not boot:
        return None
    d = boot["details"]
    bci, ratio = d.get("bootstrap_ci"), d.get("ci_ratio")
    if not (isinstance(bci, list) and len(bci) == 2) or ratio in (None, 0):
        return None
    width = (float(bci[1]) - float(bci[0])) / float(ratio)
    return width / (2 * Z975) if width > 0 else None


def _se_boot(boot: Optional[Dict[str, Any]]) -> Tuple[Optional[float], Optional[float]]:
    if not boot:
        return None, None
    d = boot["details"]
    effs = d.get("bootstrap_effects")
    sd = (
        statistics.stdev([float(e) for e in effs])
        if isinstance(effs, list) and len(effs) > 1
        else None
    )
    bci = d.get("bootstrap_ci")
    ci_se = (
        (float(bci[1]) - float(bci[0])) / (2 * Z975)
        if isinstance(bci, list) and len(bci) == 2
        else None
    )
    return sd, ci_se


def _se_evalue_inv(ate: float, sens: Optional[Dict[str, Any]]) -> Tuple[Optional[float], str]:
    """(se, branch). ``branch`` names which inversion applied, or why none did."""
    if not sens:
        return None, "no_sensitivity_row"
    d = sens["details"]
    e_ci = d.get("e_value_ci")
    if e_ci is None:
        return None, "no_e_value_ci"
    e_ci = float(e_ci)
    if e_ci <= 1.0:
        return None, "ci_includes_zero"
    if "rr_ci" in d:  # Lane D' format
        conv = d.get("conversion")
        rr = float(d["rr_ci"])
        if conv == "risk_ratio":
            p0 = float(d["baseline_risk"])
            bound = p0 * (rr - 1.0) if ate >= 0 else p0 * (1.0 / rr - 1.0)
            branch = "laneD_risk_ratio"
        elif conv == "standardized_difference":
            sd = d.get("outcome_std")
            near, far = pattern._recover_ci(ate, e_ci, float(sd) if sd not in (None, "") else None)
            bound = near if ate >= 0 else far
            branch = "laneD_smd"
        else:
            return None, f"laneD_unknown_conversion:{conv}"
    else:
        sd_raw = d.get("outcome_std")
        sd = float(sd_raw) if sd_raw not in (None, "") else None
        standardized = bool(d.get("standardized", pattern._valid_sd(sd))) and pattern._valid_sd(sd)
        near, far = pattern._recover_ci(ate, e_ci, sd if standardized else None)
        bound = near if ate >= 0 else far  # the bound nearest the null
        branch = "old_standardized" if standardized else "old_unstandardized"
    se = (abs(ate) - abs(bound)) / Z975
    return (se if se > 0 else None), branch


def _is_binary(values: List[float]) -> bool:
    return set(values) <= {0.0, 1.0}


def _naive_se(df, t: str, o: str) -> Tuple[Optional[float], str]:
    tv = [float(x) for x in df[t].tolist()]
    yv = [float(x) for x in df[o].tolist()]
    n = len(tv)
    if n < 3:
        return None, "naive_too_few_rows"
    if _is_binary(tv) and _is_binary(yv):
        y1 = [y for tt, y in zip(tv, yv, strict=True) if tt == 1.0]
        y0 = [y for tt, y in zip(tv, yv, strict=True) if tt == 0.0]
        if not y1 or not y0:
            return None, "naive_one_arm_empty"
        p1, p0 = sum(y1) / len(y1), sum(y0) / len(y0)
        return math.sqrt(p1 * (1 - p1) / len(y1) + p0 * (1 - p0) / len(y0)), "naive_2prop"
    # classical OLS slope SE of y ~ t (covariate-free)
    mt, my = sum(tv) / n, sum(yv) / n
    sxx = sum((x - mt) ** 2 for x in tv)
    if sxx == 0:
        return None, "naive_treatment_constant"
    sxy = sum((x - mt) * (y - my) for x, y in zip(tv, yv, strict=True))
    b = sxy / sxx
    rss = sum((y - my - b * (x - mt)) ** 2 for x, y in zip(tv, yv, strict=True))
    return math.sqrt(rss / (n - 2) / sxx), "naive_ols"


async def _cached_frame(
    cache: Dict[Any, Any], dataset: str, t: str, o: str, brand: str, limit: int
):
    """One pull per (dataset, brand, treatment, outcome, limit), the pattern script's
    key: the loader returns ONLY the requested columns, so a frame cannot be shared
    across pairs. ``_frame`` is the pattern script's loader call, unchanged."""
    key = (dataset, brand, t, o, limit)
    if key not in cache:
        try:
            cache[key] = await pattern._frame(dataset, t, o, brand, limit)
        except Exception as exc:  # noqa: BLE001 - report, never guess
            cache[key] = exc
    return cache[key]


# --- scoring


def _z_from_p(p: Optional[float]) -> Tuple[Optional[float], bool]:
    """(|z|, is_lower_bound). DoWhy's one-tailed normal p at 20 simulations."""
    if p is None:
        return None, False
    nd = statistics.NormalDist()
    if p <= 0.0:
        return nd.inv_cdf(1.0 - P_FLOOR), True
    if p >= 1.0:
        return 0.0, False
    return abs(nd.inv_cdf(1.0 - p)), False


def _status_today(delta_percent: float) -> str:
    if delta_percent <= PASS_PCT:
        return "passed"
    if delta_percent <= WARN_PCT:
        return "warning"
    return "failed"


def _score_new(
    ate: float, ref: float, se: float, n: int, n_ref: int
) -> Tuple[str, float, float, float]:
    """(status, shift_se scaled, shift_se unscaled, scale) through the runner's own
    ``_score_common_cause_shift``. The helper takes the reported INTERVAL; the stored
    sources give its SE, so the interval is rebuilt symmetric about the effect
    (``ate +/- 1.959964 * se``) -- the helper's ``(hi - lo) / (2 * 1.959964)`` then
    returns ``se`` within floating-point rounding (relative ~1e-15; e.g. ate 1.0,
    se 0.01 comes back as 0.010000000000000014). A stored count of 0 means unknown
    and is passed as None."""
    status, d = _score_common_cause_shift(
        original_effect=ate,
        refuted_effect=ref,
        original_ci=(ate - Z975 * se, ate + Z975 * se),
        reference_n=n or None,
        refit_n=n_ref or None,
        thresholds=SE_THRESHOLDS,
    )
    return (
        status.value,
        float(d["shift_se_units"]),
        abs(ref - ate) / se,
        float(d["reference_se_scale"]),
    )


def _fmt(v: Any, nd: int = 4) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _ratio_stats(pairs: List[Tuple[float, float]]) -> str:
    vals = [a / b for a, b in pairs if b]
    if not vals:
        return "n=0"
    return f"n={len(vals)}, median {statistics.median(vals):.3f}, min {min(vals):.3f}, max {max(vals):.3f}"


def _pct(vals: List[float], q: float) -> Optional[float]:
    if not vals:
        return None
    s = sorted(vals)
    k = (len(s) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


async def main(out: Path) -> int:
    from dotenv import load_dotenv

    env_candidates = [REPO / ".env"]
    if REPO.parent.name == ".worktrees":
        env_candidates.append(REPO.parents[1] / ".env")
    for env_file in env_candidates:
        if env_file.is_file():
            load_dotenv(env_file)
            break

    rows = _rows()
    by_est: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for r in rows:
        by_est[r["estimate_id"]][r["test_type"]] = r
    runner = RefutationRunner()
    frame_cache: Dict[Any, Any] = {}

    runs: List[Dict[str, Any]] = []
    for eid, tests in by_est.items():
        rcc = tests.get("random_common_cause")
        if rcc is None:
            continue
        ate, ref = rcc["original_effect"], rcc["refuted_effect"]
        delta = abs(ref - ate)
        e: Dict[str, Any] = {
            "eid": eid,
            "brand": rcc["brand"],
            "t": rcc["t"],
            "o": rcc["o"],
            "n": int(rcc["details"].get("refutation_n_rows_total") or 0),
            # rows the refutation refits actually ran on (a subsample when the
            # estimation frame exceeded the runner's cap: ``refutation_subsampled``)
            "n_ref": int(rcc["details"].get("refutation_n_rows") or 0),
            "subsampled": str(rcc["details"].get("refutation_subsampled")).lower() == "true",
            "ate": ate,
            "ref": ref,
            "delta": delta,
            "pct": rcc["delta_percent"],
            "status_today": rcc["status"],
            "status_today_recomputed": _status_today(rcc["delta_percent"]),
            "gate_stored": rcc["gate_decision"],
            "p": rcc["p_value"],
            "created": rcc["created_at"][:10],
            "statuses": {name: tr["status"] for name, tr in tests.items()},
        }
        boot, sens = tests.get("bootstrap"), tests.get("sensitivity_e_value")
        e["se_reported"] = _se_reported(boot)
        e["se_boot_sd"], e["se_boot_ci"] = _se_boot(boot)
        e["se_inv"], e["inv_branch"] = _se_evalue_inv(ate, sens)
        # naive SE from a re-pulled frame, for every mapped pair (calibration needs it
        # on rows that ALSO have a stored source)
        dataset = pattern.DATASET_BY_PAIR.get((e["t"], e["o"]))
        e["dataset"] = dataset
        e["se_naive"], e["naive_kind"] = None, "unmapped_pair" if dataset is None else None
        if dataset is not None and e["n"] > 0:
            got = await _cached_frame(frame_cache, dataset, e["t"], e["o"], e["brand"], e["n"])
            if isinstance(got, Exception):
                e["naive_kind"] = f"frame_error:{type(got).__name__}"
                print(
                    f"frame_error {eid[:8]} {e['brand']} {e['t']}->{e['o']}: {got!r}",
                    file=sys.stderr,
                )
            else:
                df, _covs = got
                e["frame_n"] = len(df)
                e["se_naive"], e["naive_kind"] = _naive_se(df, e["t"], e["o"])
        # primary SE by fidelity: the reported interval (two exact inversions,
        # cross-checked), then the bootstrap resampling SE, then the frame proxy
        for src_name, key in (
            ("reported", "se_reported"),
            ("evalue_inv", "se_inv"),
            ("boot_sd", "se_boot_sd"),
            ("naive", "se_naive"),
        ):
            if e.get(key):
                e["se_source"], e["se"] = src_name, e[key]
                break
        else:
            e["se_source"], e["se"] = "NONE", None
        if e["se"]:
            e["status_new"], e["shift_se"], e["shift_se_unscaled"], e["se_scale"] = _score_new(
                ate, ref, e["se"], e["n"], e["n_ref"]
            )
        else:
            e["status_new"], e["shift_se"], e["shift_se_unscaled"], e["se_scale"] = (
                None,
                None,
                None,
                None,
            )
        z, z_is_bound = _z_from_p(e["p"])
        e["std_refits"] = (delta / z) if z else None
        e["std_refits_is_upper_bound"] = z_is_bound
        # gates: today's recomputed from the stored statuses, new with rcc swapped
        e["gate_today"], e["conf_today"] = pattern._band(runner, e["statuses"])
        if e["status_new"] is not None:
            swapped = dict(e["statuses"])
            swapped["random_common_cause"] = e["status_new"]
            e["gate_new"], e["conf_new"] = pattern._band(runner, swapped)
        else:
            e["gate_new"], e["conf_new"] = "?", None
        runs.append(e)

    runs.sort(
        key=lambda x: (x["status_today"] != "failed", x["brand"], x["t"], x["o"], x["created"])
    )

    # --- counts
    moves = Counter((x["status_today"], x["status_new"]) for x in runs)
    gate_moves = Counter((x["gate_today"], x["gate_new"]) for x in runs)
    passed_to_failed = [
        x for x in runs if x["status_today"] == "passed" and x["status_new"] == "failed"
    ]
    failed_rows = [x for x in runs if x["status_today"] == "failed"]
    no_se = [x for x in runs if x["se"] is None]
    stored_mismatch = [x for x in runs if x["status_today"] != x["status_today_recomputed"]]
    gate_mismatch = [x for x in runs if x["gate_stored"] != x["gate_today"]]
    # Stored gates predate Lane D' (PR #2002, 2026-09-11), when sensitivity FAILED was a
    # critical BLOCK; the current runner scores it non-critical. A mismatch with that
    # signature is explained; anything else is listed.
    gate_mismatch_unexplained = [
        x
        for x in gate_mismatch
        if not (
            x["gate_stored"] == "block" and x["statuses"].get("sensitivity_e_value") == "failed"
        )
    ]
    # rows whose PRIMARY se is the proxy kind the calibration shows is off
    ols_primary = [x for x in runs if x["se_source"] == "naive" and x["naive_kind"] == "naive_ols"]
    subsampled = [x for x in runs if x["subsampled"] or (x["n_ref"] and x["n_ref"] < x["n"])]
    shifts_passed = [
        x["shift_se"] for x in runs if x["status_today"] == "passed" and x["shift_se"] is not None
    ]
    shifts_passed_unscaled = [
        x["shift_se_unscaled"]
        for x in runs
        if x["status_today"] == "passed" and x["shift_se_unscaled"] is not None
    ]
    scaled_rows = [x for x in runs if x["se_scale"] not in (None, 1.0)]

    # --- calibration: cross-checks between sources on rows carrying both
    cal = {
        "evalue_inv / reported": [
            (x["se_inv"], x["se_reported"]) for x in runs if x["se_inv"] and x["se_reported"]
        ],
        "boot_sd / reported": [
            (x["se_boot_sd"], x["se_reported"])
            for x in runs
            if x["se_boot_sd"] and x["se_reported"]
        ],
        "boot_ci / reported": [
            (x["se_boot_ci"], x["se_reported"])
            for x in runs
            if x["se_boot_ci"] and x["se_reported"]
        ],
        "naive / reported": [
            (x["se_naive"], x["se_reported"]) for x in runs if x["se_naive"] and x["se_reported"]
        ],
        "naive / evalue_inv": [
            (x["se_naive"], x["se_inv"]) for x in runs if x["se_naive"] and x["se_inv"]
        ],
        "naive_2prop / evalue_inv (binary treatment & outcome)": [
            (x["se_naive"], x["se_inv"])
            for x in runs
            if x["se_naive"] and x["se_inv"] and x["naive_kind"] == "naive_2prop"
        ],
        "naive_ols / evalue_inv (continuous treatment)": [
            (x["se_naive"], x["se_inv"])
            for x in runs
            if x["se_naive"] and x["se_inv"] and x["naive_kind"] == "naive_ols"
        ],
        "naive / boot_sd": [
            (x["se_naive"], x["se_boot_sd"]) for x in runs if x["se_naive"] and x["se_boot_sd"]
        ],
        "std_refits / se": [
            (x["std_refits"], x["se"])
            for x in runs
            if x["std_refits"] and x["se"] and not x["std_refits_is_upper_bound"]
        ],
    }
    n_seeded, seeded_by = _seeded_counts()
    sources = Counter(x["se_source"] for x in runs)
    branches = Counter(x["inv_branch"] for x in runs)
    naive_kinds = Counter(x["naive_kind"] for x in runs if x["naive_kind"])

    lines = [
        f"# Live re-band of `random_common_cause` under the shift-vs-reported-SE rule as implemented ({date.today().isoformat()})",
        "",
        f"Agent runs: {len(runs)} (`estimate_source = causal_impact_query`, one rcc row each). "
        f"Stored (the retired |Δ|/|ATE| rule): {dict(Counter(x['status_today'] for x in runs))}. "
        f"Rule as implemented, scored through the runner's own `_score_common_cause_shift`: "
        f"`shift_se = |refuted − original| / (se × scale)`, PASSED ≤ {PASS_SE:g}, WARNING ≤ {WARN_SE:g}, else FAILED "
        f'(`PASS_THRESHOLDS["common_cause_shift_se"]`); `se = (ci_hi − ci_lo) / (2 × {Z975})` from the reported interval; '
        f"`scale = sqrt(n / refit n)` when the refutation ran on a #1419 subsample (refit n < n), else 1.0. "
        f"Rows scaled: {len(scaled_rows)}"
        + (
            " — "
            + ", ".join(
                f"`{x['eid'][:8]}` n={x['n']} refit n={x['n_ref']} scale {x['se_scale']:.2f} (unscaled {x['shift_se_unscaled']:.2f} → {x['shift_se']:.2f} SE, {x['status_new']})"
                for x in scaled_rows
            )
            if scaled_rows
            else ""
        )
        + ".",
        "",
        "## Status moves (today → new)",
        "",
        "| today → new | runs |",
        "|---|---|",
    ]
    lines += [
        f"| {a} → {b} | {n} |"
        for (a, b), n in sorted(moves.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1])))
    ]
    lines += [
        "",
        "## Gate moves (today, recomputed from stored statuses → new, rcc swapped)",
        "",
        "| today → new | runs |",
        "|---|---|",
    ]
    lines += [f"| {a} → {b} | {n} |" for (a, b), n in sorted(gate_moves.items())]
    lines += [
        "",
        "## Headline counts",
        "",
        f"- FAILED today → new: {dict(Counter(x['status_new'] for x in failed_rows))} ({len(failed_rows)} rows).",
        f"- PASSED today → FAILED new: **{len(passed_to_failed)}**"
        + (
            " — "
            + ", ".join(
                f"`{x['eid'][:8]}` {x['brand'] or '<all>'} {x['t']}→{x['o']} (Δ {x['delta']:.4f}, se {x['se']:.4f} [{x['se_source']}], shift {x['shift_se']:.2f})"
                for x in passed_to_failed
            )
            if passed_to_failed
            else "."
        ),
        f"- shift_se on today's PASSED rows (n={len(shifts_passed)}): max {_fmt(max(shifts_passed) if shifts_passed else None, 3)}, "
        f"p95 {_fmt(_pct(shifts_passed, 0.95), 3)}, p50 {_fmt(_pct(shifts_passed, 0.50), 3)} "
        f"(unscaled, as in the disproof re-band: max {_fmt(max(shifts_passed_unscaled) if shifts_passed_unscaled else None, 3)}, "
        f"p95 {_fmt(_pct(shifts_passed_unscaled, 0.95), 3)}).",
        f"- SE source used, by fidelity: {dict(sources)}. E-value inversion branches: {dict(branches)}. Naive proxy kinds: {dict(naive_kinds)}.",
        f"- Rows with NO SE source: {len(no_se)}"
        + (
            " — "
            + ", ".join(
                f"`{x['eid'][:8]}` {x['brand'] or '<all>'} {x['t']}→{x['o']}" for x in no_se
            )
            if no_se
            else "."
        ),
        f"- Sanity: stored rcc status ≠ status recomputed from stored delta_percent on {len(stored_mismatch)} rows. "
        f"Stored gate ≠ gate recomputed by the CURRENT runner from the stored statuses on {len(gate_mismatch)} rows, "
        f"of which {len(gate_mismatch) - len(gate_mismatch_unexplained)} are stored BLOCK with sensitivity FAILED (gated before Lane D′ made "
        f"sensitivity non-critical — the criticality change, not this rule)"
        + (
            "; unexplained: "
            + ", ".join(
                f"`{x['eid'][:8]}` stored {x['gate_stored']} / recomputed {x['gate_today']}"
                for x in gate_mismatch_unexplained
            )
            if gate_mismatch_unexplained
            else "; none unexplained"
        )
        + ". The gate columns below hold every stored status fixed except rcc (the old-format sensitivity statuses included, which the Lane D′ live re-band has since re-read), so the gate delta is attributable to rcc alone.",
        f"- Rows whose primary SE is the continuous-treatment OLS proxy: {len(ols_primary)}"
        + (
            " — "
            + ", ".join(
                f"`{x['eid'][:8]}` {x['brand'] or '<all>'} {x['t']}→{x['o']} (shift {x['shift_se']:.2f}, {x['status_new']})"
                for x in ols_primary
            )
            if ols_primary
            else ""
        )
        + f". Rows whose refutation refits ran on a SUBSAMPLE of the estimation frame (`refutation_subsampled`): {len(subsampled)}"
        + (
            " — "
            + ", ".join(
                f"`{x['eid'][:8]}` refit n={x['n_ref']} of n={x['n']} (shift {_fmt(x['shift_se'], 2)}, {x['status_new'] or '?'})"
                for x in subsampled
            )
            if subsampled
            else ""
        )
        + ".",
        "",
        "## Proxy calibration (ratio of SE sources on rows carrying both)",
        "",
        "| ratio | stats |",
        "|---|---|",
    ]
    lines += [f"| {k} | {_ratio_stats(v)} |" for k, v in cal.items()]
    lines += [
        "",
        "`reported` and `evalue_inv` are two independent exact inversions of the SAME reported interval "
        "(`bootstrap_ci / ci_ratio` and the stored `e_value_ci` bound); their ratio is the cross-check. "
        "`boot_sd` is a 20-resample SE of the estimator, `naive` a covariate-free SE on the re-pulled frame, "
        "`std_refits` the spread of the 20 random-common-cause refits (from the stored p-value; rows with p stored as 0 excluded).",
        "",
        "## Per run",
        "",
        "| estimate | brand | pair | n (refit n if subsampled) | original | refuted | Δ | today % / status | se source | se | scale | shift_se (unscaled) | new status | gate today → new | std_refits (rcc noise) | p | reported / inv / boot_sd / naive |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for x in runs:
        sr = x["std_refits"]
        sr_txt = (
            ("≤ " if x["std_refits_is_upper_bound"] else "") + _fmt(sr) if sr is not None else "-"
        )
        lines.append(
            f"| `{x['eid'][:8]}` | {x['brand'] or '<all>'} | {x['t']}→{x['o']} | {x['n']}{' (' + str(x['n_ref']) + ')' if x['subsampled'] else ''} | {_fmt(x['ate'])} | {_fmt(x['ref'])} | "
            f"{_fmt(x['delta'])} | {x['pct']:.1f} / {x['status_today']} | {x['se_source']} | {_fmt(x['se'])} | "
            f"{_fmt(x['se_scale'], 2)} | {_fmt(x['shift_se'], 2)} ({_fmt(x['shift_se_unscaled'], 2)}) | {x['status_new'] or '?'} | {x['gate_today']} → {x['gate_new']} | {sr_txt} | "
            f"{_fmt(x['p'], 5)} | {_fmt(x['se_reported'])} / {_fmt(x['se_inv'])} / {_fmt(x['se_boot_sd'])} / {_fmt(x['se_naive'])} |"
        )

    # --- per pair
    per_pair: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for x in runs:
        per_pair[(x["brand"] or "<all>", x["t"], x["o"])].append(x)
    lines += [
        "",
        "## Per pair",
        "",
        "| brand | pair | runs | today status | new status | gate today | gate new | median |ATE| | median shift_se | max shift_se | se sources |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for key in sorted(per_pair):
        es = per_pair[key]
        shifts = [y["shift_se"] for y in es if y["shift_se"] is not None]
        lines.append(
            f"| {key[0]} | {key[1]}→{key[2]} | {len(es)} | {dict(Counter(y['status_today'] for y in es))} | "
            f"{dict(Counter(y['status_new'] for y in es))} | {dict(Counter(y['gate_today'] for y in es))} | "
            f"{dict(Counter(y['gate_new'] for y in es))} | {statistics.median(abs(y['ate']) for y in es):.4f} | "
            f"{_fmt(statistics.median(shifts) if shifts else None, 2)} | {_fmt(max(shifts) if shifts else None, 2)} | "
            f"{dict(Counter(y['se_source'] for y in es))} |"
        )

    # --- reading
    seven_moved = all(x["status_new"] in ("passed", "warning") for x in failed_rows)
    stuck = [x for x in failed_rows if x["status_new"] not in ("passed", "warning")]
    stuck_on_proxy = [
        x for x in stuck if x["se_source"] == "naive" and x["naive_kind"] == "naive_ols"
    ]
    premise = seven_moved and not passed_to_failed and not no_se
    passed_to_warning = [
        x for x in runs if x["status_today"] == "passed" and x["status_new"] == "warning"
    ]
    ols_cal = cal["naive_ols / evalue_inv (continuous treatment)"]
    ols_ratio_txt = _ratio_stats(ols_cal)
    lines += [
        "",
        "## Footnotes",
        "",
        f"- Seeded rows excluded: {n_seeded} rcc rows with `estimate_source <> causal_impact_query` ({seeded_by}); they are not agent runs.",
        "- Pairs without a dataset mapping (no re-pulled frame, stored sources only): "
        + (
            ", ".join(
                sorted(
                    {
                        f"{x['brand'] or '<all>'} {x['t']}→{x['o']}"
                        for x in runs
                        if x["dataset"] is None
                    }
                )
            )
            or "none"
        )
        + ".",
        "- The `<all>` brand rows carry a NULL brand on every test row; their frame is the all-brands pull (brand=None), as in the sensitivity re-band.",
        f"- SE constants: the rule's {Z975}. The reported interval on the agent path is the estimator's own `ate_ci_lower / ate_ci_upper`; on the pipeline path it is `effect ± 1.96·se`, a 0.002 % difference from the rule's constant.",
        "- `std_refits` is recovered from DoWhy's one-tailed normal p (`z = |Δ| / std_refits`, `np.std` ddof=0 over 20 refits). A stored `p = 0.00000` means p < 5e-6 (z > 4.42) and the column shows an upper bound.",
        "- `std_refits` is a ratio of two small numbers when p is near 0.5 (z near 0); read it as an order of magnitude there.",
        f"- Nothing is restated: the cutoffs {PASS_SE:g} / {WARN_SE:g} are read from the runner's `PASS_THRESHOLDS` and the verdict comes from its `_score_common_cause_shift`; the refit-frame scale is the runner's, applied from the stored `refutation_n_rows_total` / `refutation_n_rows`.",
        "",
        "## Reading",
        "",
    ]
    if premise:
        lines.append(
            f"The premise survives on the live rows. All {len(failed_rows)} rows FAILED under today's |Δ|/|ATE| rule move to "
            f"{dict(Counter(x['status_new'] for x in failed_rows))} under the shift-vs-SE rule, no PASSED row becomes FAILED, and the "
            f"largest shift on any PASSED row is {_fmt(max(shifts_passed) if shifts_passed else None, 2)} SE (p95 {_fmt(_pct(shifts_passed, 0.95), 2)}). "
            f"What would reverse it: a PASSED row whose shift exceeded {WARN_SE:g} refit-scaled reference SEs (none here), a reported SE that the "
            "stored inversions mis-recover (the two exact inversions above agree wherever both exist), or a naive-proxy row "
            "whose true SE is far below the proxy (the calibration ratios bound that)."
        )
    else:
        why = []
        if not seven_moved:
            why.append(
                "not every FAILED row moves to PASSED/WARNING: "
                + ", ".join(
                    f"`{x['eid'][:8]}` {x['brand'] or '<all>'} {x['t']}→{x['o']} → {x['status_new'] or 'UNSCORED (no SE source)'} (se source {x['se_source']}/{x['naive_kind'] or '-'}, shift {_fmt(x['shift_se'], 2)})"
                    for x in stuck
                )
            )
        if passed_to_failed:
            why.append(f"{len(passed_to_failed)} PASSED row(s) become FAILED (listed above)")
        if no_se:
            why.append(f"{len(no_se)} row(s) have no SE source")
        lines.append(
            f"The premise does NOT fully survive as measured: {'; '.join(why)}. "
            f"{len(failed_rows) - len(stuck)} of the {len(failed_rows)} FAILED rows move to {dict(Counter(x['status_new'] for x in failed_rows if x['status_new'] in ('passed', 'warning')))}; "
            f"no PASSED row becomes FAILED (max shift on a PASSED row {_fmt(max(shifts_passed) if shifts_passed else None, 2)} SE, p95 {_fmt(_pct(shifts_passed, 0.95), 2)}, p50 {_fmt(_pct(shifts_passed, 0.50), 2)})."
        )
        for x in stuck_on_proxy:
            # Same pair, same brand, rows carrying an exact inversion: scale their
            # reported SE by sqrt(n_i / n_row). A HYPOTHESIS for the reader, not a
            # verdict — printed with its inputs, never fed back into the status.
            peers = [
                y
                for y in runs
                if (y["brand"], y["t"], y["o"]) == (x["brand"], x["t"], x["o"])
                and y["se_inv"]
                and y["n"] > 0
            ]
            scaled = [y["se_inv"] * math.sqrt(y["n"] / x["n"]) for y in peers] if x["n"] > 0 else []
            hyp = (
                f"the same pair's {len(peers)} exact inversions at n={sorted({y['n'] for y in peers})} scale by √(n_i/{x['n']}) to "
                f"se ≈ {statistics.median(scaled):.4f}, i.e. shift ≈ {x['delta'] / statistics.median(scaled):.2f} SE "
                f"({_score_new(x['ate'], x['ref'], statistics.median(scaled), x['n'], x['n_ref'])[0].upper()})"
                if scaled
                else "no same-pair exact inversion exists to scale from"
            )
            noise = (
                f"std_refits {x['std_refits']:.4f} is {x['std_refits'] / x['se']:.1f}× the proxy SE"
                if x["std_refits"] and x["se"]
                else "std_refits unavailable"
            )
            lines.append(
                f"`{x['eid'][:8]}` stays FAILED on the least faithful SE source — the covariate-free OLS proxy for a continuous treatment — "
                f"which the calibration measures at {ols_ratio_txt} of the reported SE on the same pair (naive_ols / evalue_inv). "
                f"Its verdict is therefore UNRESOLVED by this measurement, not a counter-example. Hypothesis, not tuned in: {hyp}. "
                f"Its own DoWhy p-value ({x['p']:.3f}) puts the shift inside the refit spread ({noise}): a null effect (ATE {x['ate']:.4f}) perturbed by noise."
            )
        if passed_to_warning:
            lines.append(
                f"{len(passed_to_warning)} PASSED row(s) become WARNING: "
                + ", ".join(
                    f"`{x['eid'][:8]}` {x['brand'] or '<all>'} {x['t']}→{x['o']} n={x['n']} (Δ {x['delta']:.4f}, se {x['se']:.4f} [{x['se_source']}], shift {x['shift_se']:.2f})"
                    for x in passed_to_warning
                )
                + (
                    ". All of them are SUBSAMPLED refutations even after the refit-frame scale: the estimate and its reported SE come from the full frame (n="
                    + "/".join(sorted({str(y["n"]) for y in passed_to_warning}))
                    + ") while the random-common-cause refits ran on "
                    + "/".join(sorted({str(y["n_ref"]) for y in passed_to_warning}))
                    + " rows, whose own SE is ≈ √(n/refit n) = "
                    + "/".join(
                        f"{math.sqrt(y['n'] / y['n_ref']):.1f}"
                        for y in sorted(passed_to_warning, key=lambda y: y["n"])
                    )
                    + "× the reported one. Against the refit-scale SE (the scale the runner applies, already in `shift_se`) the shifts are "
                    + ", ".join(
                        f"{y['shift_se']:.2f} (unscaled {y['shift_se_unscaled']:.2f})"
                        for y in passed_to_warning
                    )
                    + " SE: still WARNING after the scale, so the scale did not explain them."
                    if all(
                        y["subsampled"] and y["n_ref"] and y["n_ref"] < y["n"]
                        for y in passed_to_warning
                    )
                    else ". Not all of them are subsampled refutations — read each."
                )
            )
        inv_dev = [abs(a / b - 1.0) for a, b in cal["evalue_inv / reported"] if b]
        lines.append(
            f"What would reverse the reading: a PASSED row whose shift exceeds {WARN_SE:g} refit-scaled reference SE (none), or the two exact inversions of the reported interval disagreeing "
            + (
                f"(max |ratio − 1| = {max(inv_dev):.2e} over {len(inv_dev)} rows carrying both)."
                if inv_dev
                else "(no row carries both)."
            )
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[: lines.index("## Per run")]))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, required=True)
    sys.exit(asyncio.run(main(ap.parse_args().out)))
