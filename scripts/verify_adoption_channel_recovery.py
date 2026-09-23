#!/usr/bin/env python3
"""Recovery probe for the planted channel effects on ``hcp_brand_adoption.adopted`` (lane T1).

READ-ONLY acceptance gate the owner runs after ``scripts/backfill_hcp_treatment_arm.py
--execute`` (``--live``), and the lane runs on the dry-run frame before the PR (``--frame``).
It builds the collapsed per-(hcp, brand) exposure frame joined to ``adopted``, runs the twin's
REAL estimator (``estimate_cohort_effect``, CausalForestDML, outcome ``adopted``, seed 42) on
every brand x channel cell -- one fit at a time, single-threaded -- and gates the result:

  per brand: CI covers ADOPTION_CHANNEL_PLANTED_RD  8/8
             |ATE - planted| <= 0.06                8/8
             Spearman(ATE, planted) >= 0.8 over the 8 channels
             null channel (rep_training_score): |ATE| <= 0.06 and CI covers 0

plus the treatment_arm ATE (naive treated-minus-control and the DGP-true mean CATE). Verdict
word first; exit 1 on any failure.

The gate is POINT-based on purpose: ``CausalForestDML.ate_interval`` returns econml's
conservative +-0.17 bound on this outcome at n ~ 3.4k, so no planted channel can be
"significant" with the estimator as shipped -- that is lane T2's population-ATE interval. The
null clause is a tolerance, not 0: Remibrutinib's null read +0.05 BEFORE any planting (a seed
artefact measured in twinad_q3_fits.csv, 2026-09-23); the clause accepts it and would still
catch a planted-by-mistake null (0.08+).

USAGE
-----
    # on the dry-run frame written by the backfill (--frame-out):
    python scripts/verify_adoption_channel_recovery.py --frame /tmp/adoption_frame.parquet

    # on the LIVE table after --execute (reads hcp_brand_adoption + hcp_profiles +
    # business_metrics; writes nothing):
    python scripts/verify_adoption_channel_recovery.py --live [--fits-out fits.csv]
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
    null_ok: bool = False
    failures: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failures


@dataclass
class GateResult:
    passed: bool
    per_brand: Dict[str, BrandGate]
    tol: float
    min_spearman: float

    def verdict(self) -> str:
        lines = [
            f"{'PASS' if self.passed else 'FAIL'}: point recovery of the planted channel effects on "
            f"adopted in {sum(g.passed for g in self.per_brand.values())}/{len(self.per_brand)} brands "
            f"(gate: CI covers planted 8/8, |ATE-planted| <= {self.tol}, Spearman >= {self.min_spearman}, "
            f"null |ATE| <= {self.tol} with CI covering 0)"
        ]
        for brand, g in self.per_brand.items():
            lines.append(
                f"  {brand:14s} {'PASS' if g.passed else 'FAIL'}  covers {g.covers}/8  within_tol "
                f"{g.within_tol}/8  max|err| {g.max_abs_err:.3f}  spearman {g.spearman:.2f}  "
                f"null ATE {g.null_ate:+.3f} ({'ok' if g.null_ok else 'NOT ok'})"
            )
            for f in g.failures:
                lines.append(f"      - {f}")
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
) -> GateResult:
    """Gate a fits table with columns ``brand, channel, planted_rd, ate, ci_lower, ci_upper,
    error`` (one row per brand x channel). The gate iterates ``required_brands`` (all three by
    default), not the brands present: a brand with no fits FAILS, so a ``--brands`` subset run
    can never certify (codex r1). A missing or errored cell fails its brand."""
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
        if not all(covers.values()):
            g.failures.append(
                f"covers {g.covers}/{len(usable)}: CI misses the planted RD for "
                f"{[c for c, ok in covers.items() if not ok]}"
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
            lo, hi = (
                float(sub.loc[NULL_COLUMN, "ci_lower"]),
                float(sub.loc[NULL_COLUMN, "ci_upper"]),
            )
            g.null_ok = abs(g.null_ate) <= tol and lo <= 0.0 <= hi
            if not g.null_ok:
                g.failures.append(
                    f"null channel {NULL_COLUMN}: ATE {g.null_ate:+.3f} CI ({lo:+.3f}, {hi:+.3f}) "
                    f"must satisfy |ATE| <= {tol} and cover 0"
                )
        per_brand[brand] = g
    passed = bool(per_brand) and all(g.passed for g in per_brand.values())
    return GateResult(passed=passed, per_brand=per_brand, tol=tol, min_spearman=min_spearman)


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


# ---------------------------------------------------------------------------
# Fits
# ---------------------------------------------------------------------------


def _rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2)


def run_fits(
    frame: pd.DataFrame, *, brands: Sequence[str] = BRANDS, seed: int = 42
) -> pd.DataFrame:
    """One REAL ``estimate_cohort_effect`` per brand x channel, sequentially."""
    from src.digital_twin.effect.cohort_causal_estimator import (
        DEFAULT_CONFOUNDERS,
        estimate_cohort_effect,
    )
    from src.digital_twin.effect.errors import EffectDataUnavailable

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
            try:
                eff = estimate_cohort_effect(
                    sub, col, outcome_col="adopted", confounders=DEFAULT_CONFOUNDERS, seed=seed
                )
                rec.update(
                    {
                        "n": eff.n,
                        "ate": eff.ate,
                        "ci_lower": eff.ate_ci_lower,
                        "ci_upper": eff.ate_ci_upper,
                    }
                )
            except EffectDataUnavailable as e:
                rec["error"] = f"{e.cause}: {e}"
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
    args = parser.parse_args(argv)

    if args.frame is not None:
        all_rows = pd.read_parquet(args.frame)
        frame = frame_from_parquet(args.frame)
        true_source = all_rows
        logger.info("frame %s: %d rows, %d joined", args.frame, len(all_rows), len(frame))
    else:
        from dotenv import load_dotenv

        load_dotenv(_PROJECT_ROOT / ".env")
        from src.memory.services.factories import get_supabase_client

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
    result = evaluate_recovery_gate(fits)
    print(result.verdict())
    print("\n".join(treatment_arm_summary(all_rows, true_source=true_source)))
    print(f"  peak RSS {_rss_gib():.2f} GiB; {len(fits)} fits, {int(fits['seconds'].sum())} s")
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
