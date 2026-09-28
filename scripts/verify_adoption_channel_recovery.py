#!/usr/bin/env python3
"""Recovery probe for the planted channel effects on ``hcp_brand_adoption.adopted`` (lane T1).

READ-ONLY acceptance gate the owner runs after ``scripts/backfill_hcp_treatment_arm.py
--execute`` (``--live``), and the lane runs on the dry-run frame before the PR (``--frame``).
It builds the collapsed per-(hcp, brand) exposure frame joined to ``adopted``, runs the twin's
REAL estimator (``estimate_cohort_effect``, CausalForestDML, outcome ``adopted``, seed 42) on
every brand x channel cell -- one fit at a time, single-threaded -- and gates the result:

  point gate, per brand:  |ATE - planted| <= 0.06                8/8 (the null at 0 included)
                          Spearman(ATE, planted) >= 0.8 over the 8 channels
  significance (lane T2): the focus channels (engagement, speaker, peer, PSP) CI excludes 0
                          in 3/3 brands
  null calibration:       the null channel refit on the SAME frame under fresh DGP seeds
    (family level)        (``NULL_CALIBRATION_SEEDS``; ``adopted`` re-drawn, design fixed),
                          pooled over brands x seeds: false-positive rate (CI excludes 0)
                          <= 0.10, and mean reported SE / empirical SD of the null ATEs >= 0.9
  reported, not gated:    CI covers ADOPTION_CHANNEL_PLANTED_RD (n/8 per brand); a realised
                          null CI excluding 0 (printed as a draw, with the family FP rate)

plus the treatment_arm ATE (naive treated-minus-control and the DGP-true mean CATE). Verdict
word first; exit 1 on any failure.

The CI is the twin estimator's own (lane T2): the causal forest's doubly-robust
``ate_ +- z * ate_stderr_``, calibrated at 1.06-1.52x the seed Monte-Carlo SD (cert.md section
3). Lane T1 wrote this gate against ``CausalForestDML.ate_interval``, econml's conservative
+-0.17 bound, under which no planted channel could be significant, so its gate was point-based
and "CI covers planted 8/8" was nearly free. Against a calibrated 95% interval, 24 cells all
covering happens ~0.95**24 = 29% of the time, so coverage is reported, not gated.

WHY THE NULL'S INTERVAL IS GATED AT THE FAMILY LEVEL
----------------------------------------------------
History: 900a8318b (lane T1) required the null CI to cover 0 in EVERY brand, against the
conservative ``ate_interval``. 2a71bf1e4 (lane T2, calibrated DR interval) relaxed it to
"covers 0 in >= 2/3 brands" with a named Remibrutinib exception, after the live null read
+0.047, CI (+0.008, +0.086). Neither contract is what a calibrated test can promise: a correct
95% interval excludes 0 on a true null 5% of the time per brand, so "every brand covers" fails
a correct estimator ~14% of the time over three brands, and "2 of 3" certifies whatever one
brand happened to draw. The Remibrutinib reading was measured to be a draw, not a defect
(docs/demos/results/2026-09-28_remi_null_caveat/): its adjusted contrast is +0.0021 in the
structural adoption probability and +0.0445 in the final Bernoulli coin flips of the live
seed; placebo permutation excluded 0 in 6/100 fits and 100 fresh DGP seeds in 3/100. The owner
chose (option A, 2026-09-28) to certify the property that IS promised: the interval's
operating characteristics on the null, measured on the live design by re-drawing ``adopted``
with fixed, replayable seeds. The live draw's own exclusion is printed, never gated.

Reversal condition: FP rate > 0.10 means the null is systematically picked up (bias or an
anti-conservative interval); SE / empirical SD < 0.9 means the DR interval understates the
null's sampling spread, and the plan's reversal clause applies -- fall back to LinearDML's
``ate_inference`` interval. The redraw is certifying only if the live seed reproduces the live
labels exactly on the gated frame (otherwise it calibrates a different process).

``--via-twin-loader`` (with ``--live``) builds the frame with the twin's own
``cohort_loader.load_cohort_frame`` -- the frame ``/simulate`` estimates on (synthetic rows,
channel-carrying rollups only, paged adoption join) -- instead of re-deriving the plant.
``frame_from_live`` refuses rollup rows with NULL channels by design: it re-derives the DGP from
the rollups, and an un-planted row would move the planting medians. The daily per-HCP ETL
adds such rows after the plant, so Task 3's acceptance uses ``--via-twin-loader``. The null
calibration re-derives from the rollup rows that carry a channel (the loader's own row rule).
``--frame`` has no centrality to re-derive from, so it never certifies (calibration not run).

USAGE
-----
    # on the dry-run frame written by the backfill (--frame-out):
    python scripts/verify_adoption_channel_recovery.py --frame /tmp/adoption_frame.parquet

    # on the LIVE table after --execute (reads hcp_brand_adoption + hcp_profiles +
    # business_metrics; writes nothing):
    python scripts/verify_adoption_channel_recovery.py --live [--fits-out fits.csv]

    # the same, through the twin's own loader (what /simulate estimates on):
    python scripts/verify_adoption_channel_recovery.py --live --via-twin-loader
"""

from __future__ import annotations

import os

# Single-threaded BEFORE numpy/sklearn import: this runs on the prod box.
for _var in ("LOKY_MAX_CPU_COUNT", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import logging  # noqa: E402
import resource  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from datetime import date, timedelta  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, List, Optional, Sequence  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.per_hcp_cohort_collapse import (  # noqa: E402
    CHANNEL_COLUMNS,
    collapse_per_hcp_brand,
)
from src.data.per_hcp_cohort_columns import (  # noqa: E402
    ADOPTION_CHANNEL_PLANTED_RD,
    ADOPTION_NULL_CHANNEL,
    INTERVENTION_TREATMENT_MAP,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BRANDS = ("Remibrutinib", "Fabhalta", "Kisqali")
NULL_COLUMN = INTERVENTION_TREATMENT_MAP[ADOPTION_NULL_CHANNEL]
DEFAULT_TOL = 0.06
DEFAULT_MIN_SPEARMAN = 0.8
DEFAULT_SEED = 42
#: Lane T2's significance clause: these channels' CIs must exclude 0 in every brand.
FOCUS_COLUMNS: tuple[str, ...] = tuple(
    INTERVENTION_TREATMENT_MAP[k]
    for k in (
        "digital_engagement",
        "speaker_program_invitation",
        "peer_influence_activation",
        "patient_support_program",
    )
)
#: The DGP seed of the live labels (``scripts/backfill_hcp_treatment_arm.DEFAULT_SEED``; checked
#: at run time, not imported here: that module loads .env on import).
LIVE_DGP_SEED = 427
#: Fresh DGP seeds for the family-level null calibration: fixed so a run is replayable, and
#: disjoint from the live seed. 100 seeds x 3 brands = 300 null fits (~2.5 s each, ~13 min
#: sequential at ~0.6 GiB); at a 4% FP rate the Wilson 95% interval is ~(0.02, 0.07), clear of
#: the 0.10 ceiling, and the pooled SD ratio carries ~4% relative error.
NULL_CALIBRATION_SEEDS: tuple[int, ...] = tuple(range(2001, 2101))
#: Certifying thresholds of the null calibration (owner option A, 2026-09-28). Constants, not
#: flags (codex r2): an overridable threshold would certify anything.
MAX_NULL_FP_RATE = 0.10
MIN_SE_RATIO = 0.9
#: The live seed must reproduce the gated frame's labels exactly for the redraw to be the DGP.
MIN_REDRAW_REPRODUCTION = 1.0
_Z95 = 1.959963984540054
_FRAME_COLUMNS = (
    "hcp_id",
    "brand",
    "adopted",
    "region",
    "market_share",
    "triggers_total_count",
    *CHANNEL_COLUMNS,
)


def planted_rd_by_column() -> Dict[str, float]:
    """``{planted column: realised RD}`` from the intervention-keyed table."""
    return {INTERVENTION_TREATMENT_MAP[k]: float(v) for k, v in ADOPTION_CHANNEL_PLANTED_RD.items()}


# ---------------------------------------------------------------------------
# The gate (pure; tested on synthetic fits tables)
# ---------------------------------------------------------------------------


@dataclass
class BrandGate:
    brand: str
    n_fits: int = 0
    covers: int = 0
    within_tol: int = 0
    max_abs_err: float = float("nan")
    spearman: float = float("nan")
    null_ate: float = float("nan")
    null_lo: float = float("nan")
    null_hi: float = float("nan")
    null_ok: bool = False
    null_covers_zero: bool = False
    focus_significant: int = 0
    failures: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failures


def wilson_interval(k: int, n: int, z: float = _Z95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion ``k / n``."""
    if n <= 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * float(np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


@dataclass
class NullCalibration:
    """Family-level operating characteristics of the null channel's interval."""

    n_expected: int = 0
    n_fits: int = 0
    n_fp: int = 0
    fp_rate: float = float("nan")
    wilson_lo: float = float("nan")
    wilson_hi: float = float("nan")
    se_ratio: float = float("nan")
    reproduction: Optional[float] = None
    per_brand: Dict[str, Dict[str, float]] = field(default_factory=dict)
    failures: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failures

    def lines(self) -> List[str]:
        repro = "not measured" if self.reproduction is None else f"{self.reproduction:.4f}"
        out = [
            f"  null calibration {'PASS' if self.passed else 'FAIL'}: {NULL_COLUMN} refit under "
            f"{self.n_fits}/{self.n_expected} fresh-seed redraws (brands x seeds): "
            f"FP rate {self.n_fp}/{self.n_fits} = {self.fp_rate:.3f} (Wilson 95% "
            f"{self.wilson_lo:.3f}..{self.wilson_hi:.3f}; gate <= {MAX_NULL_FP_RATE}), "
            f"SE/empirical SD {self.se_ratio:.2f} (gate >= {MIN_SE_RATIO}), live-seed "
            f"reproduction {repro}"
        ]
        for brand, b in self.per_brand.items():
            out.append(
                f"    {brand:14s} n={int(b['n'])} FP {int(b['n_fp'])} mean ATE {b['mean_ate']:+.4f} "
                f"mean SE {b['mean_se']:.4f} empirical SD {b['emp_sd']:.4f} ratio {b['ratio']:.2f}"
            )
        for f in self.failures:
            out.append(f"      - {f}")
        return out


def evaluate_null_calibration(
    null_fits: pd.DataFrame,
    *,
    reproduction: Optional[float],
    required_brands: Sequence[str] = BRANDS,
    seeds: Sequence[int] = NULL_CALIBRATION_SEEDS,
) -> NullCalibration:
    """Gate a null-calibration fits table (``brand, seed, ate, ci_lower, ci_upper, stderr,
    error``; one row per brand x seed). Pooled over every required brand x seed cell: the FP
    rate (CI excludes 0) must be <= ``MAX_NULL_FP_RATE`` and the mean reported SE / the
    empirical SD of the null ATEs (within brand, pooled) >= ``MIN_SE_RATIO``. A missing,
    duplicated or errored cell fails -- the FP rate of a subset is not the family's -- and so
    does a ``reproduction`` (share of the gated frame's labels the live seed reproduces) that
    is unmeasured or below ``MIN_REDRAW_REPRODUCTION``."""
    cal = NullCalibration(n_expected=len(required_brands) * len(seeds), reproduction=reproduction)
    df = null_fits[
        null_fits["brand"].isin(list(required_brands)) & null_fits["seed"].isin(list(seeds))
    ]
    n_dup = int(df.duplicated(["brand", "seed"]).sum())
    if n_dup:
        cal.failures.append(f"{n_dup} duplicated brand x seed cells")
    present = set(zip(df["brand"], df["seed"], strict=True))
    n_missing = sum((b, s) not in present for b in required_brands for s in seeds)
    if n_missing:
        cal.failures.append(
            f"missing {n_missing} of {cal.n_expected} brand x seed cells (a subset never certifies)"
        )
    err = df["error"].notna() & df["error"].astype(str).ne("")
    if err.any():
        cal.failures.append(
            f"estimator error in {int(err.sum())} cells: {sorted(set(df.loc[err, 'error'].astype(str)))[:3]}"
        )
    ok = df[~err].copy()
    for c in ("ate", "ci_lower", "ci_upper", "stderr"):
        ok[c] = pd.to_numeric(ok[c], errors="coerce")
    ok = ok.dropna(subset=["ate", "ci_lower", "ci_upper", "stderr"])
    cal.n_fits = int(len(ok))
    if cal.n_fits:
        fp = (ok["ci_lower"] > 0.0) | (ok["ci_upper"] < 0.0)
        cal.n_fp = int(fp.sum())
        cal.fp_rate = cal.n_fp / cal.n_fits
        cal.wilson_lo, cal.wilson_hi = wilson_interval(cal.n_fp, cal.n_fits)
        ss, dof = 0.0, 0
        for brand in required_brands:
            b = ok[ok["brand"] == brand]
            if len(b) < 2:
                continue
            sd = float(b["ate"].std(ddof=1))
            ss += float(((b["ate"] - b["ate"].mean()) ** 2).sum())
            dof += len(b) - 1
            cal.per_brand[brand] = {
                "n": len(b),
                "n_fp": int(fp[b.index].sum()),
                "mean_ate": float(b["ate"].mean()),
                "mean_se": float(b["stderr"].mean()),
                "emp_sd": sd,
                "ratio": float(b["stderr"].mean()) / sd if sd > 0 else float("inf"),
            }
        pooled_sd = float(np.sqrt(ss / dof)) if dof else float("nan")
        cal.se_ratio = (
            float(ok["stderr"].mean()) / pooled_sd
            if dof and pooled_sd > 0
            else float("inf")
            if dof
            else float("nan")
        )
        if cal.fp_rate > MAX_NULL_FP_RATE:
            cal.failures.append(
                f"null false-positive rate {cal.n_fp}/{cal.n_fits} = {cal.fp_rate:.3f} > "
                f"{MAX_NULL_FP_RATE}: the null is picked up as an effect across redraws"
            )
        if not (cal.se_ratio >= MIN_SE_RATIO):
            cal.failures.append(
                f"SE/empirical SD {cal.se_ratio:.2f} < {MIN_SE_RATIO}: the DR interval "
                "understates the null's sampling spread -- fall back to LinearDML (the plan's "
                "reversal clause)"
            )
    else:
        cal.failures.append("no usable null-calibration fits")
    if reproduction is None:
        cal.failures.append(
            "live-seed reproduction not measured: the redraw cannot be shown to be the live DGP"
        )
    elif reproduction < MIN_REDRAW_REPRODUCTION:
        cal.failures.append(
            f"derive(seed {LIVE_DGP_SEED}) does not reproduce the live labels on the gated "
            f"frame ({reproduction:.4f} < {MIN_REDRAW_REPRODUCTION}); the redraw would "
            "calibrate a different process"
        )
    return cal


@dataclass
class GateResult:
    passed: bool
    per_brand: Dict[str, BrandGate]
    tol: float
    min_spearman: float
    null_calibration: Optional[NullCalibration] = None

    def verdict(self) -> str:
        cal = self.null_calibration
        fam = (
            "not run" if cal is None else f"FP {cal.n_fp}/{cal.n_fits}, SE/empSD {cal.se_ratio:.2f}"
        )
        lines = [
            f"{'PASS' if self.passed else 'FAIL'}: recovery of the planted channel effects on "
            f"adopted in {sum(g.passed for g in self.per_brand.values())}/{len(self.per_brand)} brands, "
            f"null calibration {fam} "
            f"(gate: |ATE-planted| <= {self.tol} 8/8 incl. the null, Spearman >= "
            f"{self.min_spearman}, focus channels' CI excludes 0 in every brand, null FP rate "
            f"<= {MAX_NULL_FP_RATE} and SE/empirical SD >= {MIN_SE_RATIO} over fresh-seed "
            "redraws; coverage of the planted RD reported, not gated)"
        ]
        for brand, g in self.per_brand.items():
            lines.append(
                f"  {brand:14s} {'PASS' if g.passed else 'FAIL'}  focus sig {g.focus_significant}/"
                f"{len(FOCUS_COLUMNS)}  covers {g.covers}/8  within_tol {g.within_tol}/8  "
                f"max|err| {g.max_abs_err:.3f}  spearman {g.spearman:.2f}  null ATE "
                f"{g.null_ate:+.3f} CI ({g.null_lo:+.3f}, {g.null_hi:+.3f}) "
                f"({'ok' if g.null_ok else 'NOT ok'})"
            )
            for f in g.failures:
                lines.append(f"      - {f}")
        family = "not measured" if cal is None else f"{cal.n_fp}/{cal.n_fits}"
        for brand, g in self.per_brand.items():
            if g.n_fits and not g.null_covers_zero and not np.isnan(g.null_ate):
                lines.append(
                    f"  NOTE: {brand} null CI excludes 0 in the realised draw (seed "
                    f"{LIVE_DGP_SEED}): ATE {g.null_ate:+.3f}, CI ({g.null_lo:+.3f}, "
                    f"{g.null_hi:+.3f}); family FP rate {family}"
                )
        if cal is None:
            lines.append(
                "  FAIL: null calibration not run (it needs --live over all three brands); "
                "non-certifying"
            )
        else:
            lines.extend(cal.lines())
        return "\n".join(lines)


def _spearman(a: Sequence[float], b: Sequence[float]) -> float:
    ra = pd.Series(list(a)).rank()
    rb = pd.Series(list(b)).rank()
    if ra.nunique() < 2 or rb.nunique() < 2:
        return float("nan")
    return float(ra.corr(rb))


def evaluate_recovery_gate(
    fits: pd.DataFrame,
    *,
    tol: float = DEFAULT_TOL,
    min_spearman: float = DEFAULT_MIN_SPEARMAN,
    required_brands: Sequence[str] = BRANDS,
    null_calibration: Optional[NullCalibration] = None,
) -> GateResult:
    """Gate a fits table with columns ``brand, channel, planted_rd, ate, ci_lower, ci_upper,
    error`` (one row per brand x channel). The gate iterates ``required_brands`` (all three by
    default), not the brands present: a brand with no fits FAILS, so a ``--brands`` subset run
    can never certify (codex r1). A missing or errored cell fails its brand. The null's
    interval is certified by ``null_calibration`` (:func:`evaluate_null_calibration`); without
    one the gate never passes."""
    planted = planted_rd_by_column()
    per_brand: Dict[str, BrandGate] = {}
    for brand in required_brands:
        sub = fits[fits["brand"] == brand].set_index("channel")
        g = BrandGate(brand=brand, n_fits=int(len(sub)))
        if sub.empty:
            g.failures.append(
                "missing: no fits for this brand (a --brands subset is non-certifying)"
            )
            per_brand[brand] = g
            continue
        missing = [c for c in CHANNEL_COLUMNS if c not in sub.index]
        if missing:
            g.failures.append(f"missing fits for {missing}")
        errored = [
            c for c in sub.index if pd.notna(sub.loc[c, "error"]) and str(sub.loc[c, "error"])
        ]
        if errored:
            g.failures.append(
                f"estimator error in {errored}: {[str(sub.loc[c, 'error']) for c in errored]}"
            )
        usable = [c for c in CHANNEL_COLUMNS if c in sub.index and c not in errored]
        ates = {c: float(sub.loc[c, "ate"]) for c in usable}
        errs = {c: abs(ates[c] - planted[c]) for c in usable}
        covers = {
            c: bool(float(sub.loc[c, "ci_lower"]) <= planted[c] <= float(sub.loc[c, "ci_upper"]))
            for c in usable
        }
        g.covers = sum(covers.values())
        g.within_tol = sum(e <= tol for e in errs.values())
        g.max_abs_err = max(errs.values()) if errs else float("nan")
        # Coverage of the planted RD is reported in the verdict line, not gated (see module doc).
        focus = [c for c in FOCUS_COLUMNS if c in usable]
        significant = {
            c: float(sub.loc[c, "ci_lower"]) > 0.0 or float(sub.loc[c, "ci_upper"]) < 0.0
            for c in focus
        }
        g.focus_significant = sum(significant.values())
        if g.focus_significant < len(FOCUS_COLUMNS):
            g.failures.append(
                f"focus channels whose CI covers 0: "
                f"{[c for c in FOCUS_COLUMNS if not significant.get(c, False)]}"
            )
        if g.within_tol < len(usable):
            g.failures.append(
                f"|ATE - planted| > {tol} for "
                + str({c: round(e, 3) for c, e in errs.items() if e > tol})
            )
        if len(usable) == len(CHANNEL_COLUMNS):
            g.spearman = _spearman(
                [ates[c] for c in CHANNEL_COLUMNS], [planted[c] for c in CHANNEL_COLUMNS]
            )
            if not (g.spearman >= min_spearman):
                g.failures.append(f"Spearman(ATE, planted) {g.spearman:.2f} < {min_spearman}")
        if NULL_COLUMN in usable:
            g.null_ate = ates[NULL_COLUMN]
            g.null_lo = float(sub.loc[NULL_COLUMN, "ci_lower"])
            g.null_hi = float(sub.loc[NULL_COLUMN, "ci_upper"])
            g.null_covers_zero = g.null_lo <= 0.0 <= g.null_hi
            # Per brand the null clause is the point tolerance; its interval is certified at the
            # family level (evaluate_null_calibration), and a realised exclusion is printed.
            g.null_ok = abs(g.null_ate) <= tol
            if not g.null_ok:
                g.failures.append(
                    f"null channel {NULL_COLUMN}: ATE {g.null_ate:+.3f} CI "
                    f"({g.null_lo:+.3f}, {g.null_hi:+.3f}) must satisfy |ATE| <= {tol}"
                )
        per_brand[brand] = g
    # Certification is over the full brand set: a subset's per-brand verdicts are diagnostic
    # (codex r6), and the null's interval certifies only through the family calibration.
    passed = (
        set(BRANDS) <= set(per_brand)
        and all(g.passed for g in per_brand.values())
        and null_calibration is not None
        and null_calibration.passed
    )
    return GateResult(
        passed=passed,
        per_brand=per_brand,
        tol=tol,
        min_spearman=min_spearman,
        null_calibration=null_calibration,
    )


# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------


def frame_from_parquet(path: Path) -> pd.DataFrame:
    """The backfill's ``--frame-out`` output: keep the joined rows (they carry the collapsed
    exposure); ``adopted`` is the generated label."""
    df = pd.read_parquet(path)
    if "joined" in df.columns:
        df = df[df["joined"].astype(bool)]
    missing = [c for c in _FRAME_COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(f"frame {path} lacks columns {missing}")
    return df.reset_index(drop=True)


def frame_from_live(client: Any) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(joined estimator frame, live hcp_brand_adoption, re-derived frame) from the live tables.

    The re-derivation (seed 427, same rollups) is what proves the live table IS the DGP's
    output after ``--execute`` (arm and label match printed) and supplies the DGP-true
    treatment_arm ATE, which the table does not persist.
    """
    from scripts.backfill_hcp_treatment_arm import (
        DEFAULT_SEED,
        derive,
        fetch_centrality,
        fetch_channel_rollups,
        fetch_live_adoption,
    )

    centrality = fetch_centrality(client)
    if centrality is None or centrality.empty:
        raise SystemExit("hcp_profiles has no synthetic centrality rows")
    rollups = fetch_channel_rollups(client)
    if rollups.empty:
        raise SystemExit("business_metrics has no planted per_hcp_rollup rows")
    live = fetch_live_adoption(client)
    if live is None or live.empty:
        raise SystemExit("hcp_brand_adoption has no synthetic rows")
    live["adopted"] = pd.to_numeric(live["adopted"], errors="coerce")
    live["treatment_arm"] = pd.to_numeric(live["treatment_arm"], errors="coerce")
    # The date rule is irrelevant here; give derive() a run date after every exposure.
    run_date = max(date.today(), rollups["metric_date"].max().date() + timedelta(days=1))
    derived = derive(centrality, seed=DEFAULT_SEED, channel_rollups=rollups, run_date=run_date)

    collapsed = collapse_per_hcp_brand(rollups)
    frame = collapsed.merge(
        live[["hcp_id", "brand", "adopted", "treatment_arm"]], on=["hcp_id", "brand"], how="inner"
    )
    if "specialty" in centrality.columns:
        frame = frame.merge(centrality[["hcp_id", "specialty"]], on="hcp_id", how="left")
    return frame.reset_index(drop=True), live, derived


async def frame_from_twin_loader(client: Any, brands: Sequence[str] = BRANDS) -> pd.DataFrame:
    """The frame ``/simulate`` estimates on, one brand at a time, through the twin's own
    ``cohort_loader.load_cohort_frame`` (``client`` is an async Supabase client)."""
    from src.digital_twin.effect.cohort_loader import load_cohort_frame

    parts = [await load_cohort_frame(client, brand) for brand in brands]
    parts = [p for p in parts if not p.empty]
    if not parts:
        raise SystemExit("the twin loader returned no cohort rows for any brand")
    return pd.concat(parts, ignore_index=True)


def _frame_via_twin_loader(brands: Sequence[str]) -> pd.DataFrame:
    import asyncio

    from src.memory.services.factories import loop_scoped_async_supabase_client

    async def _load() -> pd.DataFrame:
        async with loop_scoped_async_supabase_client() as client:
            return await frame_from_twin_loader(client, brands)

    return asyncio.run(_load())


# ---------------------------------------------------------------------------
# Fits
# ---------------------------------------------------------------------------


def _rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2)


def _fit_one(sub: pd.DataFrame, col: str, *, seed: int) -> Dict[str, Any]:
    """The twin's REAL estimator on one brand's frame and one channel: ATE, DR CI, stderr."""
    from src.digital_twin.effect.cohort_causal_estimator import (
        DEFAULT_CONFOUNDERS,
        estimate_cohort_effect,
    )
    from src.digital_twin.effect.errors import EffectDataUnavailable

    try:
        eff = estimate_cohort_effect(
            sub, col, outcome_col="adopted", confounders=DEFAULT_CONFOUNDERS, seed=seed
        )
    except EffectDataUnavailable as e:
        return {"error": f"{e.cause}: {e}"}
    return {
        "n": eff.n,
        "ate": eff.ate,
        "ci_lower": eff.ate_ci_lower,
        "ci_upper": eff.ate_ci_upper,
        "stderr": eff.ate_stderr,
    }


def run_fits(
    frame: pd.DataFrame, *, brands: Sequence[str] = BRANDS, seed: int = 42
) -> pd.DataFrame:
    """One REAL ``estimate_cohort_effect`` per brand x channel, sequentially."""
    planted = planted_rd_by_column()
    rows: List[dict] = []
    for brand in brands:
        sub = frame[frame["brand"] == brand].reset_index(drop=True)
        for col in CHANNEL_COLUMNS:
            t0 = time.time()
            rec: Dict[str, Any] = {
                "brand": brand,
                "channel": col,
                "planted_rd": planted[col],
                "n": int(len(sub)),
                "ate": np.nan,
                "ci_lower": np.nan,
                "ci_upper": np.nan,
                "error": None,
            }
            rec.update(_fit_one(sub, col, seed=seed))
            rec["seconds"] = round(time.time() - t0, 1)
            rec["peak_rss_gib"] = round(_rss_gib(), 3)
            rows.append(rec)
            logger.info(
                "  %-14s %-28s n=%-5s planted=%+.3f ate=%s ci=(%s, %s) covers=%s %ss rss=%.2fGiB%s",
                brand,
                col,
                rec["n"],
                planted[col],
                "nan" if pd.isna(rec["ate"]) else f"{rec['ate']:+.3f}",
                "nan" if pd.isna(rec["ci_lower"]) else f"{rec['ci_lower']:+.3f}",
                "nan" if pd.isna(rec["ci_upper"]) else f"{rec['ci_upper']:+.3f}",
                None
                if pd.isna(rec["ate"])
                else bool(rec["ci_lower"] <= planted[col] <= rec["ci_upper"]),
                rec["seconds"],
                rec["peak_rss_gib"],
                f"  ERR {rec['error']}" if rec["error"] else "",
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Family-level null calibration (fresh DGP seeds on the gated frame)
# ---------------------------------------------------------------------------

_KEY = ["hcp_id", "brand"]


def redraw_adopted(frame: pd.DataFrame, derived: pd.DataFrame) -> pd.DataFrame:
    """``frame`` with ``adopted`` replaced by ``derived``'s label on the same (hcp_id, brand).

    Everything else -- rows, treatments, confounders, and which rows carry a label (the
    estimator's usable set) -- is held fixed: the redraw moves only the outcome."""
    labels = derived.drop_duplicates(_KEY).set_index(_KEY)["adopted"].astype(float)
    new = labels.reindex(pd.MultiIndex.from_frame(frame[_KEY])).to_numpy()
    out = frame.copy()
    out["adopted"] = np.where(frame["adopted"].notna().to_numpy(), new, np.nan)
    return out


def label_reproduction(frame: pd.DataFrame, derived: pd.DataFrame) -> float:
    """Share of ``frame``'s labelled rows whose label ``derived`` reproduces exactly."""
    labelled = frame[frame["adopted"].notna()]
    if labelled.empty:
        return float("nan")
    redrawn = redraw_adopted(labelled, derived)["adopted"].to_numpy()
    return float(np.mean(redrawn == labelled["adopted"].astype(float).to_numpy()))


def run_null_calibration(
    client: Any,
    frame: pd.DataFrame,
    *,
    seeds: Sequence[int] = NULL_CALIBRATION_SEEDS,
    brands: Sequence[str] = BRANDS,
    fit_seed: int = DEFAULT_SEED,
) -> tuple[pd.DataFrame, float]:
    """Re-draw ``adopted`` with each fresh DGP seed (``derive`` on the live centrality,
    specialty and planted rollups -- the design fixed) and refit the null channel per brand on
    the gated ``frame``. Returns (null fits table, live-seed label reproduction on ``frame``).
    Read-only: ``client`` is the sync Supabase client, used only for ``.select()`` reads."""
    from scripts.backfill_hcp_treatment_arm import (
        DEFAULT_SEED as BACKFILL_SEED,
    )
    from scripts.backfill_hcp_treatment_arm import (
        derive,
        fetch_centrality,
        fetch_channel_rollups,
    )

    if BACKFILL_SEED != LIVE_DGP_SEED:
        raise SystemExit(
            f"the backfill's live seed is {BACKFILL_SEED}, this gate assumes {LIVE_DGP_SEED}"
        )
    centrality = fetch_centrality(client)
    if centrality is None or centrality.empty:
        raise SystemExit("hcp_profiles has no synthetic centrality rows")
    rollups = fetch_channel_rollups(client)
    # The loader's row rule: rows with no channel are the per-HCP ETL's post-plant rows, which
    # the executed re-plant never saw (derive() refuses a partially null row, loudly).
    rollups = rollups[rollups[list(CHANNEL_COLUMNS)].notna().any(axis=1)]
    if rollups.empty:
        raise SystemExit("business_metrics has no planted per_hcp_rollup rows")
    run_date = max(date.today(), rollups["metric_date"].max().date() + timedelta(days=1))

    live = derive(centrality, seed=LIVE_DGP_SEED, channel_rollups=rollups, run_date=run_date)
    reproduction = label_reproduction(frame, live)
    logger.info(
        "null calibration: derive(seed %d) reproduces %.4f of the gated frame's labels; "
        "%d seeds x %d brands",
        LIVE_DGP_SEED,
        reproduction,
        len(seeds),
        len(brands),
    )
    rows: List[dict] = []
    t_start = time.time()
    for i, seed in enumerate(seeds):
        redrawn = redraw_adopted(
            frame, derive(centrality, seed=seed, channel_rollups=rollups, run_date=run_date)
        )
        for brand in brands:
            sub = redrawn[redrawn["brand"] == brand].reset_index(drop=True)
            t0 = time.time()
            rec: Dict[str, Any] = {
                "brand": brand,
                "seed": seed,
                "ate": np.nan,
                "ci_lower": np.nan,
                "ci_upper": np.nan,
                "stderr": np.nan,
                "error": None,
            }
            rec.update(_fit_one(sub, NULL_COLUMN, seed=fit_seed))
            rec["seconds"] = round(time.time() - t0, 1)
            rows.append(rec)
        done = pd.DataFrame(rows)
        fp = int(((done["ci_lower"] > 0) | (done["ci_upper"] < 0)).sum())
        logger.info(
            "  null seed %d (%d/%d): FP so far %d/%d  %.0f s  rss=%.2fGiB",
            seed,
            i + 1,
            len(seeds),
            fp,
            len(done),
            time.time() - t_start,
            _rss_gib(),
        )
    return pd.DataFrame(rows), reproduction


def treatment_arm_summary(
    all_rows: pd.DataFrame, *, true_source: Optional[pd.DataFrame] = None
) -> List[str]:
    """Per brand: naive treated-minus-control on ``adopted`` and the DGP-true mean CATE."""
    lines = []
    for brand in BRANDS:
        sub = all_rows[all_rows["brand"] == brand]
        if sub.empty or "treatment_arm" not in sub.columns:
            continue
        y, t = sub["adopted"].astype(float), sub["treatment_arm"]
        naive = float(y[t == 1].mean() - y[t == 0].mean())
        src = true_source if true_source is not None else all_rows
        s2 = src[src["brand"] == brand]
        true_ate = (
            float(s2["cate_estimate"].mean()) if "cate_estimate" in s2.columns else float("nan")
        )
        lines.append(
            f"  {brand:14s} treatment_arm ATE: DGP-true {true_ate:+.4f} (mean prob CATE)  naive {naive:+.4f}"
        )
    return lines


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--frame", type=Path, help="dry-run frame parquet from the backfill's --frame-out"
    )
    src.add_argument("--live", action="store_true", help="read the LIVE tables (after --execute)")
    parser.add_argument(
        "--via-twin-loader",
        action="store_true",
        help="with --live: build the frame with the twin's own load_cohort_frame (the frame "
        "/simulate estimates on) instead of re-deriving the plant",
    )
    parser.add_argument(
        "--brands",
        nargs="*",
        default=list(BRANDS),
        help="brands to FIT (diagnostic); the gate always requires all three, so a subset "
        "run reports FAIL for the brands it skipped and is non-certifying",
    )
    # The gate's tolerance, Spearman floor and seed are constants (DEFAULT_TOL,
    # DEFAULT_MIN_SPEARMAN, DEFAULT_SEED), deliberately NOT flags: `--tol 1` would certify a
    # structural null (codex r2).
    parser.add_argument(
        "--fits-out", type=Path, default=None, help="write the fits table to this CSV"
    )
    parser.add_argument(
        "--null-fits-out",
        type=Path,
        default=None,
        help="write the null-calibration fits (brand x fresh seed) to this CSV",
    )
    # The null calibration's seeds and thresholds are constants too (NULL_CALIBRATION_SEEDS,
    # MAX_NULL_FP_RATE, MIN_SE_RATIO), for the same reason.
    args = parser.parse_args(argv)
    if args.via_twin_loader and not args.live:
        parser.error("--via-twin-loader requires --live")

    if args.frame is not None:
        all_rows = pd.read_parquet(args.frame)
        frame = frame_from_parquet(args.frame)
        true_source = all_rows
        logger.info("frame %s: %d rows, %d joined", args.frame, len(all_rows), len(frame))
    else:
        from dotenv import load_dotenv

        load_dotenv(_PROJECT_ROOT / ".env")
        from src.memory.services.factories import get_supabase_client

        if args.via_twin_loader:
            frame = _frame_via_twin_loader(BRANDS)
            logger.info(
                "twin loader: %d joined (hcp, brand) rows", int(frame["adopted"].notna().sum())
            )
            fits = run_fits(frame, brands=args.brands, seed=DEFAULT_SEED)
            if args.fits_out:
                fits.to_csv(args.fits_out, index=False)
                logger.info("fits written to %s", args.fits_out)
            client = get_supabase_client()
            calibration, null_fits = _null_calibration(client, frame, args)
            result = evaluate_recovery_gate(fits, null_calibration=calibration)
            print(result.verdict())
            print(_cost_line(fits, null_fits))
            return 0 if result.passed else 1
        client = get_supabase_client()
        frame, live, derived = frame_from_live(client)
        m = live.merge(derived, on=["hcp_id", "brand"], suffixes=("_live", "_dgp"))
        for brand in args.brands:
            mb = m[m["brand"] == brand]
            logger.info(
                "  %-14s live vs DGP(seed 427 + live channels): treatment_arm match %.4f  adopted match %.4f  (n=%d)",
                brand,
                float((mb["treatment_arm_live"] == mb["treatment_arm_dgp"]).mean()),
                float((mb["adopted_live"] == mb["adopted_dgp"]).mean()),
                len(mb),
            )
        all_rows = live
        true_source = derived
        logger.info(
            "live: %d hcp_brand_adoption rows, %d joined to the collapsed exposure",
            len(live),
            len(frame),
        )

    fits = run_fits(frame, brands=args.brands, seed=DEFAULT_SEED)
    if args.fits_out:
        fits.to_csv(args.fits_out, index=False)
        logger.info("fits written to %s", args.fits_out)
    calibration: Optional[NullCalibration] = None
    null_fits: Optional[pd.DataFrame] = None
    if args.live:
        calibration, null_fits = _null_calibration(client, frame, args)
    result = evaluate_recovery_gate(fits, null_calibration=calibration)
    print(result.verdict())
    print("\n".join(treatment_arm_summary(all_rows, true_source=true_source)))
    print(_cost_line(fits, null_fits))
    return 0 if result.passed else 1


def _null_calibration(
    client: Any, frame: pd.DataFrame, args: argparse.Namespace
) -> tuple[Optional[NullCalibration], Optional[pd.DataFrame]]:
    """Run and evaluate the family-level null calibration, unless the fit was a brand subset
    (non-certifying anyway: skip the ~13 minutes)."""
    if not set(BRANDS) <= set(args.brands):
        logger.info("null calibration skipped: --brands %s is a non-certifying subset", args.brands)
        return None, None
    null_fits, reproduction = run_null_calibration(client, frame)
    if args.null_fits_out:
        null_fits.to_csv(args.null_fits_out, index=False)
        logger.info("null-calibration fits written to %s", args.null_fits_out)
    return evaluate_null_calibration(null_fits, reproduction=reproduction), null_fits


def _cost_line(fits: pd.DataFrame, null_fits: Optional[pd.DataFrame]) -> str:
    n_null = 0 if null_fits is None else len(null_fits)
    s_null = 0 if null_fits is None else int(null_fits["seconds"].sum())
    return (
        f"  peak RSS {_rss_gib():.2f} GiB; {len(fits)} gate fits {int(fits['seconds'].sum())} s, "
        f"{n_null} null-calibration fits {s_null} s (K = {len(NULL_CALIBRATION_SEEDS)} seeds)"
    )


if __name__ == "__main__":
    sys.exit(main())
