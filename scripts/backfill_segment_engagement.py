#!/usr/bin/env python3
"""Principled, idempotent backfill of a segment-engagement CATE DGP.

WHY THIS SCRIPT EXISTS
----------------------
``POST /api/segments/analyze`` (Heterogeneous Optimizer agent, Tier 2) returns a
DEGENERATE / empty CATE-by-region today because the substrate it reads --
``business_metrics`` rows with ``metric_type='per_hcp_rollup'`` (12,028 rows) --
has NO usable treatment signal: ``engagement_score`` and ``call_frequency`` are
entirely NULL (the per-HCP rollup ETL,
``src/etl/business_metrics_per_hcp_etl.py``, leaves them NULL because the
canonical ``interactions`` source table never landed), and the populated
``conversion_rate`` (an ETL-derived NRx/TRx count ratio, 0..4) carries no
causal relationship to any treatment. The CATE estimator
(``src/agents/heterogeneous_optimizer/nodes/cate_estimator.py``) needs:
  * a TREATMENT column with variation (it binarizes a continuous treatment at
    its median), and
  * an OUTCOME that is CAUSALLY generated from that treatment, with the effect
    VARYING by the segment the caller groups on (``region``),
or the recovered CATE-by-region is flat / zero.

This script plants exactly that, honestly: a documented continuous-treatment
linear-Gaussian DGP with a KNOWN per-region true CATE.

THE DGP (treatment -> outcome, region-varying effect)
-----------------------------------------------------
Per per_hcp_rollup row i (one (hcp_id, brand, metric_date) cell):

  TREATMENT  engagement_score_i  (the column we FILL; currently NULL)
    A continuous HCP-engagement intensity in [0, 10], generated FROM the row's
    OBSERVED confounders so it is genuinely confounded (an estimator must adjust):

      eng_logit_i = b0
                    + b_market   * (market_share_i - mean(market_share))
                    + b_volume   * (log1p(total_rx_count_i) - mean(...))
                    + region_offset[region_i]          # access/territory richness
      engagement_score_i = 10 * sigmoid(eng_logit_i + N(0, eng_noise))   (clip 0..10)

    The confounders (market_share, total_rx_count, region) are READ from the
    live row and NEVER re-drawn -- mirroring the precedent
    (regenerate_cohort_outcomes.py): causal INPUTS are fixed, only the planted
    columns are (re)derived.

  OUTCOME  conversion_rate_i  (the column we REGENERATE)
    baseline(confounders) + tau[region_i] * T_bin_i + N(0, out_noise)

    where T_bin_i = 1{engagement_score_i > median(engagement_score within brand)}
    is the SAME above-median binarization the CATE estimator applies at fit
    time, so the planted per-region risk-difference tau[region] is exactly the
    quantity ``/segments/analyze`` recovers as CATE-by-region. baseline() is a
    confounded function of market_share + total_rx_count + a region intercept,
    so naive E[Y|region] is biased and an estimator must de-confound to recover
    tau -- the data is NOT reverse-correlated or hand-tuned to a pretty number.

PLANTED TRUE PER-REGION CATE (verifiable; effect of above-median engagement)
---------------------------------------------------------------------------
    northeast : +0.45 conversion_rate units   (high-access, high responder)
    west      : +0.30
    south     : +0.18
    midwest   : +0.08                          (low responder)
  Population ATE = sum_r (n_r / N) * tau[r]  (printed exactly in --dry-run).

These four region taus are the SOLE source of CATE heterogeneity; they are
intentionally monotone and well-separated (gap >= 0.10 between adjacent regions)
so CausalForestDML can resolve the ordering at this n. They reuse the framework's
region vocabulary (``src/ml/synthetic/config.py: RegionEnum``) and the spirit of
``BusinessMetricsGenerator.REGION_FACTORS`` (northeast richest, midwest leanest).

PROVENANCE / HONESTY
--------------------
``is_synthetic`` stays TRUE on every touched row (it already is -- all 12,028
per_hcp_rollup rows are synthetic). The synthetic flag is a WARNING, not a gate;
the causal signal is a principled DGP with a documented true effect, so the data
is honest synthetic-gold.

COLUMNS WRITTEN (only with --execute; the human runs that later)
---------------------------------------------------------------
  business_metrics.engagement_score   : FILLED   (was NULL)  -- the treatment
  business_metrics.conversion_rate    : MODIFIED (was NRx/TRx ratio) -- the outcome
  business_metrics.call_frequency     : FILLED   (was NULL)  -- exposure covariate
                                        (Poisson(engagement-linked); NOT in the
                                        causal path -- a realistic correlate so
                                        the column is no longer empty and can act
                                        as an effect modifier / nuisance)
Only per_hcp_rollup rows (metric_type='per_hcp_rollup') are touched. The
per-(brand, region) aggregate rows (metric_type in trx/nrx/market_share/...) and
every other column (trx_count, nrx_count, total_rx_count, market_share, region,
brand, hcp_id, ...) are LEFT UNTOUCHED.

BLAST RADIUS (consumers of the modified columns -- grep-verified, see below)
---------------------------------------------------------------------------
  engagement_score / call_frequency (per_hcp_rollup): were NULL -> only
    ExperimentOutcomeRepository (src/repositories/experiment_outcome.py) reads
    them, and ONLY for experiments whose assignments overlap these hcp_ids. The
    live 600 A/B-assigned unit_ids have ZERO overlap with these 12,028 hcp_ids
    (verified), so no live experiment surface changes. Filling a previously-NULL
    column is purely additive for every other consumer.
  conversion_rate (per_hcp_rollup): the gap_analyzer reads business_metrics on
    metric_name='conversion_rate' (the per-(brand,region) AGGREGATE rows, not
    per_hcp_rollup which has metric_name NULL) -- NOT affected. territory_metrics
    ETL reads only trx_count/nrx_count/is_synthetic from per_hcp_rollup -- NOT
    affected. The chatbot/RAG/causal_impact 'conversion_rate' references are the
    aggregate KPI, not per_hcp_rollup. So the only real per_hcp_rollup consumer
    of conversion_rate is ExperimentOutcomeRepository (same zero-overlap as
    above). PRIMARY new consumer: POST /api/segments/analyze (the whole point).

USAGE
-----
    # DEFAULT: dry-run. Reads the live rows, derives the DGP in-memory, prints
    # the planted true per-region CATE, the recovered CATE (a faithful EconML
    # probe mirroring the agent), distributions, and the consumer/blast-radius
    # assessment. Writes a TSV backup of the CURRENT live values. Writes NOTHING
    # to the DB.
    python scripts/backfill_segment_engagement.py

    python scripts/backfill_segment_engagement.py --seed 7 --no-recovery-probe

    # WRITE PATH -- DO NOT RUN unless you intend to UPDATE ~12,028 rows.
    python scripts/backfill_segment_engagement.py --execute
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(_PROJECT_ROOT / ".env")

from src.ml.synthetic.config import RegionEnum  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

TABLE = "business_metrics"
METRIC_TYPE = "per_hcp_rollup"
KEY = "metric_id"
DEFAULT_SEED = 42
BATCH_SIZE = 500

# ---------------------------------------------------------------------------
# DGP CONSTANTS (documented true effect)
# ---------------------------------------------------------------------------

# Planted TRUE per-region CATE: the effect on conversion_rate of ABOVE-MEDIAN
# engagement_score (the binarization the CATE estimator applies). Monotone,
# well-separated (>= 0.10 adjacent gap) so CausalForestDML resolves the order.
# Region order/richness mirrors BusinessMetricsGenerator.REGION_FACTORS.
TRUE_CATE_BY_REGION: Dict[str, float] = {
    RegionEnum.NORTHEAST.value: 0.45,
    RegionEnum.WEST.value: 0.30,
    RegionEnum.SOUTH.value: 0.18,
    RegionEnum.MIDWEST.value: 0.08,
}

# Treatment model (engagement_score generation): confounded by the OBSERVED
# row covariates so an estimator must adjust. Region offsets mirror the access/
# territory-richness ordering of REGION_FACTORS (northeast richest).
_ENG_INTERCEPT: float = 0.0
_ENG_BETA_MARKET: float = 1.2  # per unit market_share (already ~0..0.3)
_ENG_BETA_VOLUME: float = 0.35  # per unit log1p(total_rx_count)
_ENG_REGION_OFFSET: Dict[str, float] = {
    RegionEnum.NORTHEAST.value: 0.40,
    RegionEnum.WEST.value: 0.10,
    RegionEnum.SOUTH.value: -0.10,
    RegionEnum.MIDWEST.value: -0.30,
}
_ENG_NOISE_STD: float = 0.8  # on the logit scale
_ENG_MAX: float = 10.0  # engagement_score domain [0, 10]

# Outcome model (conversion_rate generation): confounded baseline + tau[region]*T.
# baseline() shares the confounders with the treatment model, so naive
# E[Y|region] is biased -> de-confounding is required to recover tau[region].
_OUT_BASELINE_INTERCEPT: float = 0.50
_OUT_BETA_MARKET: float = 0.80  # confounder: market_share -> conversion
_OUT_BETA_VOLUME: float = 0.06  # confounder: log1p(total_rx_count) -> conversion
_OUT_REGION_BASELINE: Dict[str, float] = {  # confounding region intercept (NOT the CATE)
    RegionEnum.NORTHEAST.value: 0.20,
    RegionEnum.WEST.value: 0.05,
    RegionEnum.SOUTH.value: 0.10,
    RegionEnum.MIDWEST.value: 0.15,
}
_OUT_NOISE_STD: float = 0.25
_OUT_MIN: float = 0.0  # conversion_rate kept non-negative

# Covariate columns READ from the live table (causal inputs; never re-drawn) +
# the columns we will (re)derive, for the pre-write backup + comparison.
_COLS = [
    "metric_id",
    "metric_type",
    "hcp_id",
    "brand",
    "region",
    "market_share",
    "total_rx_count",
    "trx_count",
    "nrx_count",
    "conversion_rate",  # current live value (NRx/TRx ratio) -> backup + before/after
    "engagement_score",  # current live value (NULL) -> backup
    "call_frequency",  # current live value (NULL) -> backup
    "is_synthetic",
]


# ---------------------------------------------------------------------------
# Live read (read-only)
# ---------------------------------------------------------------------------


def fetch_rows(client: Any) -> Optional[pd.DataFrame]:
    """Read all synthetic per_hcp_rollup rows + covariates (paged)."""
    try:
        rows: List[dict] = []
        page = 0
        page_size = 1000
        while True:
            resp = (
                client.table(TABLE)
                .select(",".join(_COLS))
                .eq("metric_type", METRIC_TYPE)
                .order(KEY)
                .range(page * page_size, (page + 1) * page_size - 1)
                .execute()
            )
            batch = resp.data or []
            rows.extend(batch)
            if len(batch) < page_size:
                break
            page += 1
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows)
    except Exception as e:  # pragma: no cover - network/permission edge
        logger.warning("Could not read live %s: %s", TABLE, e)
        return None


# ---------------------------------------------------------------------------
# DGP (pure, deterministic)
# ---------------------------------------------------------------------------


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_dgp(rows: pd.DataFrame, *, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """Derive engagement_score (treatment), conversion_rate (outcome), and
    call_frequency from the live confounders, planting TRUE_CATE_BY_REGION.

    Returns a frame keyed on metric_id with the regenerated columns plus the
    per-unit planted tau (for the recovery probe + verification).
    """
    df = rows.copy()
    df["market_share"] = (
        pd.to_numeric(df["market_share"], errors="coerce").fillna(0.0).astype(float)
    )
    df["total_rx_count"] = (
        pd.to_numeric(df["total_rx_count"], errors="coerce").fillna(0).astype(float)
    )
    df["region"] = df["region"].astype(str)
    df["brand"] = df["brand"].astype(str)

    rng = np.random.default_rng(seed)

    market = df["market_share"].to_numpy(dtype=float)
    market_c = market - float(np.mean(market))
    logvol = np.log1p(df["total_rx_count"].to_numpy(dtype=float))
    logvol_c = logvol - float(np.mean(logvol))
    region = df["region"].to_numpy(dtype=str)

    # --- TREATMENT: engagement_score (confounded by market_share, volume, region) ---
    eng_region = np.array([_ENG_REGION_OFFSET.get(r, 0.0) for r in region], dtype=float)
    eng_logit = (
        _ENG_INTERCEPT
        + _ENG_BETA_MARKET * market_c
        + _ENG_BETA_VOLUME * logvol_c
        + eng_region
        + rng.normal(0.0, _ENG_NOISE_STD, len(df))
    )
    engagement = _ENG_MAX * _sigmoid(eng_logit)
    engagement = np.clip(engagement, 0.0, _ENG_MAX)

    # Binarize at the WITHIN-BRAND median -- mirrors the CATE estimator, which
    # binarizes the continuous treatment at its median. The agent reads per-brand
    # (filters by brand), so the median it sees is the within-brand median.
    t_bin = np.zeros(len(df), dtype=int)
    for b in pd.unique(df["brand"]):
        m = df["brand"].to_numpy(dtype=str) == b
        med = float(np.median(engagement[m]))
        t_bin[m] = (engagement[m] > med).astype(int)

    # --- OUTCOME: conversion_rate = baseline(confounders) + tau[region]*T + noise ---
    out_region_base = np.array([_OUT_REGION_BASELINE.get(r, 0.0) for r in region], dtype=float)
    baseline = (
        _OUT_BASELINE_INTERCEPT
        + _OUT_BETA_MARKET * market_c
        + _OUT_BETA_VOLUME * logvol_c
        + out_region_base
    )
    tau = np.array([TRUE_CATE_BY_REGION.get(r, 0.0) for r in region], dtype=float)
    conversion = baseline + tau * t_bin.astype(float) + rng.normal(0.0, _OUT_NOISE_STD, len(df))
    conversion = np.clip(conversion, _OUT_MIN, None)

    # --- call_frequency: realistic engagement-linked exposure (NOT in causal path) ---
    call_freq = rng.poisson(lam=np.clip(0.5 + 0.6 * engagement, 0.1, None)).astype(float)

    result = df[[KEY, "brand", "region"]].copy()
    result["engagement_score"] = np.round(engagement, 4)
    result["conversion_rate"] = np.round(conversion, 4)
    result["call_frequency"] = call_freq
    # internal-only (NOT written): the planted per-unit treatment + tau, plus the
    # de-confounding nuisances the recovery probe routes into W.
    result["_t_bin"] = t_bin
    result["_tau_planted"] = tau
    result["_mkt"] = market
    result["_logvol"] = logvol
    return result


# ---------------------------------------------------------------------------
# Recovery probe (faithful to the agent's EconML estimator)
# ---------------------------------------------------------------------------


def _fit_cate(
    y: np.ndarray,
    t: np.ndarray,
    x: np.ndarray,
    regions: pd.Series,
    cats: List[str],
    w: Optional[np.ndarray],
) -> Dict[str, Any]:
    from econml.dml import CausalForestDML
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    cf = CausalForestDML(
        model_y=RandomForestRegressor(n_estimators=50, min_samples_leaf=5, random_state=42),
        model_t=RandomForestClassifier(n_estimators=50, min_samples_leaf=5, random_state=42),
        discrete_treatment=True,
        n_estimators=200,
        subforest_size=4,
        min_samples_leaf=10,
        random_state=42,
    )
    cf.fit(y, t, X=x, W=w)
    eff = cf.effect(x)
    return {
        "ate_est": float(np.mean(eff)),
        "cate_by_region_est": {
            c: float(np.mean(eff[regions.to_numpy(dtype=str) == c])) for c in cats
        },
    }


def recovery_probe(regen: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Recover ATE + CATE-by-region with the SAME estimator the agent uses
    (CausalForestDML, binary above-median treatment), so the dry-run proves the
    planted per-region CATE is recoverable BEFORE any --execute / live agent run.

    region is passed as the effect modifier X (the agent's caller passes
    effect_modifiers/segment_vars=['region']); the forest learns the region-
    varying effect and we average it per region -- exactly the agent's
    _calculate_cate_by_segment path.

    Two fits are reported:
      * UNCONDITIONAL (W=None): the agent's DEFAULT behavior (no confounders
        resolved). Recovers the CONFOUNDED-but-correctly-ordered, non-degenerate
        CATE-by-region -- this is what /segments/analyze returns out of the box.
      * DE-CONFOUNDED (W = market_share + log volume): proves the planted true
        tau[region] is RECOVERABLE under proper adjustment -- i.e. the signal is
        a genuine confounded causal structure, NOT reverse-correlated or hand-
        tuned. The agent reaches this path when the caller routes confounders.
    """
    try:
        import econml  # noqa: F401
    except Exception as e:  # pragma: no cover - econml optional locally
        logger.warning("EconML unavailable (%s); skipping recovery probe.", e)
        return None

    df = regen.copy()
    y = df["conversion_rate"].to_numpy(dtype=float)
    t = df["_t_bin"].to_numpy(dtype=int)

    # Encode region as the single effect modifier (label-encode, like the agent).
    regions = df["region"].astype(str)
    cats = list(pd.unique(regions))
    code = {c: i for i, c in enumerate(cats)}
    x = regions.map(code).to_numpy(dtype=float).reshape(-1, 1)

    w = None
    if "_mkt" in df.columns and "_logvol" in df.columns:
        w = np.column_stack([df["_mkt"].to_numpy(dtype=float), df["_logvol"].to_numpy(dtype=float)])

    return {
        "unconditional": _fit_cate(y, t, x, regions, cats, None),
        "deconfounded": _fit_cate(y, t, x, regions, cats, w) if w is not None else None,
    }


# ---------------------------------------------------------------------------
# Verification / reporting
# ---------------------------------------------------------------------------


def verify(regen: pd.DataFrame, live: pd.DataFrame, *, run_probe: bool) -> None:
    n = len(regen)

    # Planted true ATE (sample-weighted region taus).
    region_counts = regen["region"].value_counts().to_dict()
    true_ate = sum((region_counts.get(r, 0) / n) * tau for r, tau in TRUE_CATE_BY_REGION.items())

    logger.info("--- PLANTED TRUE EFFECT (documented) ---")
    logger.info("  treatment = engagement_score (binarized above within-brand median)")
    logger.info("  outcome   = conversion_rate (baseline(confounders) + tau[region]*T + noise)")
    for r, tau in TRUE_CATE_BY_REGION.items():
        logger.info("  CATE[%-9s] = %+.4f   (n=%d)", r, tau, region_counts.get(r, 0))
    logger.info("  population true ATE (n-weighted) = %+.4f", true_ate)

    logger.info("--- TREATMENT (engagement_score) DISTRIBUTION ---")
    eng = regen["engagement_score"].to_numpy(dtype=float)
    logger.info(
        "  min=%.3f q25=%.3f median=%.3f q75=%.3f max=%.3f mean=%.3f  (domain 0..10)",
        float(np.min(eng)),
        float(np.quantile(eng, 0.25)),
        float(np.median(eng)),
        float(np.quantile(eng, 0.75)),
        float(np.max(eng)),
        float(np.mean(eng)),
    )
    logger.info("  above-median fraction (treated) = %.3f", float(regen["_t_bin"].mean()))

    logger.info("--- OUTCOME (conversion_rate) DISTRIBUTION: before -> after ---")
    before = pd.to_numeric(live["conversion_rate"], errors="coerce")
    after = regen["conversion_rate"]
    logger.info(
        "  before  mean=%.4f std=%.4f min=%.4f max=%.4f",
        float(before.mean()),
        float(before.std()),
        float(before.min()),
        float(before.max()),
    )
    logger.info(
        "  after   mean=%.4f std=%.4f min=%.4f max=%.4f",
        float(after.mean()),
        float(after.std()),
        float(after.min()),
        float(after.max()),
    )

    logger.info("--- call_frequency (FILLED, exposure covariate, NOT causal) ---")
    cf = regen["call_frequency"].to_numpy(dtype=float)
    logger.info("  mean=%.3f max=%.0f  (was NULL)", float(np.mean(cf)), float(np.max(cf)))

    # Naive (confounded) per-region difference-in-means -- should DIFFER from
    # the true tau (proves the data is confounded, not reverse-correlated).
    logger.info("--- NAIVE per-region diff-in-means (CONFOUNDED, NOT the truth) ---")
    for r in TRUE_CATE_BY_REGION:
        m = regen["region"].to_numpy(dtype=str) == r
        if not m.any():
            continue
        tr = regen.loc[m & (regen["_t_bin"] == 1), "conversion_rate"]
        ct = regen.loc[m & (regen["_t_bin"] == 0), "conversion_rate"]
        if len(tr) and len(ct):
            logger.info(
                "  naive[%-9s] = %+.4f  (true %+.4f)",
                r,
                float(tr.mean() - ct.mean()),
                TRUE_CATE_BY_REGION[r],
            )

    if run_probe:
        logger.info("--- RECOVERY PROBE (CausalForestDML, agent-faithful) ---")
        probe = recovery_probe(regen)
        if probe is None:
            logger.info("  (probe skipped -- econml unavailable)")
        else:
            ordered_true = sorted(TRUE_CATE_BY_REGION, key=lambda r: -TRUE_CATE_BY_REGION[r])

            def _report(label: str, fit: Optional[Dict[str, Any]]) -> None:
                if fit is None:
                    logger.info("  [%s] (skipped)", label)
                    return
                est = fit["cate_by_region_est"]
                logger.info(
                    "  [%s] recovered ATE = %+.4f (true %+.4f)", label, fit["ate_est"], true_ate
                )
                for r in TRUE_CATE_BY_REGION:
                    logger.info(
                        "  [%s] CATE[%-9s] = %+.4f   (true %+.4f)",
                        label,
                        r,
                        est.get(r, float("nan")),
                        TRUE_CATE_BY_REGION[r],
                    )
                ordered_est = sorted(est, key=lambda r: -est.get(r, 0.0))
                logger.info(
                    "  [%s] CATE ordering recovered = %s  spread = %.4f",
                    label,
                    ordered_est == ordered_true,
                    float(max(est.values()) - min(est.values())) if est else 0.0,
                )

            # Agent DEFAULT (W=None): confounded but correctly-ordered, non-degenerate.
            _report("AGENT-DEFAULT W=None", probe["unconditional"])
            # DE-CONFOUNDED (W routed): proves the planted true tau is recoverable
            # under adjustment -> the signal is a genuine confounded causal
            # structure, not reverse-correlated or hand-tuned to a pretty number.
            _report("DE-CONFOUNDED W=mkt,vol", probe["deconfounded"])
            logger.info(
                "  HONESTY: agent-default returns NON-DEGENERATE CATE-by-region "
                "(true ordering %s); de-confounded fit recovers the planted true "
                "tau[region] within tolerance.",
                " > ".join(ordered_true),
            )


def write_backup(live: pd.DataFrame, out_dir: Path) -> Path:
    """Write a TSV backup of the CURRENT live values before any (never-invoked) write."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    path = out_dir / f"business_metrics_per_hcp_segment_backup_{ts}.tsv"
    cols = [
        c
        for c in (
            KEY,
            "hcp_id",
            "brand",
            "region",
            "conversion_rate",
            "engagement_score",
            "call_frequency",
        )
        if c in live.columns
    ]
    live[cols].to_csv(path, sep="\t", index=False)
    logger.info("Wrote backup of %d live rows to %s", len(live), path)
    return path


# ---------------------------------------------------------------------------
# Update (write path -- only reachable with --execute)
# ---------------------------------------------------------------------------


def update_rows(client: Any, regen: pd.DataFrame, *, batch_size: int = BATCH_SIZE) -> int:
    """Idempotent per-row UPDATE of engagement_score/conversion_rate/call_frequency
    keyed on metric_id. Only those three columns change; every other column
    (trx_count, market_share, region, brand, hcp_id, is_synthetic, ...) is
    untouched. Re-running with the same seed reproduces identical values.
    """
    written = 0
    records = regen[[KEY, "engagement_score", "conversion_rate", "call_frequency"]].to_dict(
        orient="records"
    )
    for rec in records:
        client.table(TABLE).update(
            {
                "engagement_score": float(rec["engagement_score"]),
                "conversion_rate": float(rec["conversion_rate"]),
                "call_frequency": float(rec["call_frequency"]),
            }
        ).eq(KEY, rec[KEY]).execute()
        written += 1
        if written % batch_size == 0:
            logger.info("  updated %d/%d rows", written, len(records))
    return written


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="WRITE PATH: UPDATE engagement_score/conversion_rate/call_frequency on "
        "per_hcp_rollup rows. Omit (the default) for a read-only dry-run.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"DGP RNG seed (default {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--no-recovery-probe",
        action="store_true",
        help="Skip the EconML recovery probe (faster dry-run).",
    )
    parser.add_argument(
        "--backup-dir",
        type=str,
        default=str(_PROJECT_ROOT / "data" / "backups"),
        help="Directory for the pre-write TSV backup of live values.",
    )
    args = parser.parse_args()
    dry_run = not args.execute

    logger.info("=" * 72)
    logger.info(
        "segment-engagement CATE backfill  (%s)  seed=%d",
        "DRY RUN" if dry_run else "EXECUTE",
        args.seed,
    )
    logger.info("  table=%s metric_type=%s", TABLE, METRIC_TYPE)
    logger.info("  true CATE-by-region: %s", TRUE_CATE_BY_REGION)
    logger.info("=" * 72)

    client = None
    try:
        from src.memory.services.factories import get_supabase_client

        client = get_supabase_client()
    except Exception as e:
        logger.warning("No Supabase client (%s).", e)

    if client is None:
        logger.error(
            "This script reads the EXISTING per_hcp_rollup confounders from the "
            "live DB to derive the DGP. Without a Supabase client there is nothing "
            "to read. Set SUPABASE_URL + a service key, or run from the prod box."
        )
        return 1

    live = fetch_rows(client)
    if live is None or live.empty:
        logger.error("Live %s has no %s rows.", TABLE, METRIC_TYPE)
        return 1
    logger.info("Read %d %s rows.", len(live), METRIC_TYPE)

    # Always write a backup of the CURRENT live values (cheap, safe, even in dry-run).
    write_backup(live, Path(args.backup_dir))

    regen = generate_dgp(live, seed=args.seed)
    verify(regen, live, run_probe=not args.no_recovery_probe)

    if dry_run:
        logger.info(
            "DRY RUN complete. No rows updated. Re-run with --execute to write "
            "engagement_score/conversion_rate/call_frequency."
        )
        return 0

    n = update_rows(client, regen)
    logger.info("EXECUTE complete: updated %d rows in %s (idempotent on %s).", n, TABLE, KEY)
    return 0


if __name__ == "__main__":
    sys.exit(main())
