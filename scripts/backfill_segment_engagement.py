#!/usr/bin/env python3
"""Principled, idempotent backfill of a MULTI-CHANNEL intervention-effect DGP.

WHY THIS SCRIPT EXISTS
----------------------
``POST /api/segments/analyze`` (Heterogeneous Optimizer agent, Tier 2) and the
Digital Twin ``/simulate`` cohort path both estimate causal effects from the
``business_metrics`` rows with ``metric_type='per_hcp_rollup'`` (~12k rows).
The substrate originally had NO usable treatment signal (engagement_score /
call_frequency entirely NULL); revision 1 of this script planted a single
engagement-score treatment with a known per-region CATE. Revision 2 (this
version, 2026-07-08, user-approved taxonomy decision) extends the DGP to EIGHT
treatment channels — one per canonical Digital Twin intervention — so the
intervention dropdown can expose every catalog entry with an HONESTLY
identified, per-channel, per-region causal effect (PR #1050 identification
gate; no fabricated uplifts).

THE DGP (multi-channel treatments -> one outcome, region-varying effects)
-------------------------------------------------------------------------
Per per_hcp_rollup row i (one (hcp_id, brand, metric_date) cell):

  TREATMENTS  (columns we FILL; migration 099 adds the six new ones)
    Every channel k is generated FROM the row's OBSERVED confounders so it is
    genuinely confounded (an estimator must adjust):

      latent_k_i = intercept_k
                   + b_market_k * (market_share_i - mean(market_share))
                   + b_volume_k * (log1p(total_rx_count_i) - mean(...))
                   + region_offset_k[region_i]     # access/territory richness
                   + N(0, noise_k)
      T_k_i = domain_transform_k(latent_k_i)       # sigmoid*10 | Poisson | share

    Channels are generated with INDEPENDENT noise (per-channel child RNGs), so
    T_j and T_k are correlated ONLY through the shared observed confounders:
    adjusting for (market_share, log-volume, region) identifies every channel's
    effect; the other channels contribute conditionally-independent outcome
    variance (wider CIs, no bias). engagement_score keeps the EXACT revision-1
    generation stream (same seed -> identical values), so the /segments/analyze
    treatment assignment is unchanged.

  OUTCOME  conversion_rate_i  (the column we REGENERATE)
    baseline(confounders) + SUM_k tau_k[region_i] * Tbin_k_i + N(0, out_noise)

    where Tbin_k_i = 1{T_k_i > median(T_k within brand)} is the SAME
    above-median binarization the estimators apply at fit time (the digital-twin
    estimator loads a per-brand cohort, so its median IS the within-brand
    median). baseline() is a confounded function of market_share +
    total_rx_count + a region intercept, so naive contrasts are biased and an
    estimator must de-confound to recover tau — the data is NOT
    reverse-correlated or hand-tuned to a pretty number.

PLANTED TRUE PER-REGION CATE PER CHANNEL (verifiable; effect of above-median T)
-------------------------------------------------------------------------------
  column                       intervention                 NE     W      S      MW
  engagement_score             digital_engagement           +0.45  +0.30  +0.18  +0.08
  speaker_program_count        speaker_program_invitation   +0.36  +0.25  +0.15  +0.06
  peer_influence_score         peer_influence_activation    +0.30  +0.21  +0.13  +0.05
  patient_support_enrollment   patient_support_program      +0.26  +0.18  +0.11  +0.05
  email_campaign_count         email_campaign               +0.24  +0.16  +0.09  +0.03
  call_frequency               call_frequency_increase      +0.20  +0.14  +0.08  +0.03
  sample_volume                sample_distribution          +0.18  +0.12  +0.07  +0.02
  rep_training_score           rep_training_quality         +0.16  +0.10  +0.05  +0.01

engagement_score taus are UNCHANGED from revision 1 (/segments/analyze
coherence). All channels are monotone in the region-richness ordering
(northeast > west > south > midwest, mirroring
BusinessMetricsGenerator.REGION_FACTORS — the brand-independent MARKET-SIZE
factor, unchanged by #1833; the #1833 brand x region execution matrix and
step events are value-only terms in business_metrics and do not enter this
ordering) with distinct headline magnitudes so
the Digital Twin page differentiates channels. Adjacent-region gaps are sized
for recoverability at per-brand n (~4k rows); the dry-run recovery probe is the
acceptance arbiter for every channel.

call_frequency NOTE: revision 1 deliberately planted call_frequency as a
NON-CAUSAL engagement-linked exposure correlate. Revision 2 re-plants it as a
genuine treatment channel (confounded draw from the shared confounders + its
own tau), consistent with the user-approved decision to make every catalog
intervention honestly simulable.

PROVENANCE / HONESTY
--------------------
``is_synthetic`` stays TRUE on every touched row. The causal signal is a
principled DGP with documented true effects, so the data is honest
synthetic-gold ("showcase capabilities before real data is connected").

COLUMNS WRITTEN (only with --execute; requires migration 099 applied)
---------------------------------------------------------------------
  business_metrics.engagement_score            : treatment (revision-1 values preserved)
  business_metrics.call_frequency              : treatment (RE-PLANTED as causal)
  business_metrics.email_campaign_count        : treatment (NEW, mig 099)
  business_metrics.speaker_program_count       : treatment (NEW, mig 099)
  business_metrics.sample_volume               : treatment (NEW, mig 099)
  business_metrics.peer_influence_score        : treatment (NEW, mig 099)
  business_metrics.patient_support_enrollment  : treatment (NEW, mig 099)
  business_metrics.rep_training_score          : treatment (NEW, mig 099)
  business_metrics.conversion_rate             : outcome  (REGENERATED)
Only per_hcp_rollup rows (metric_type='per_hcp_rollup') are touched. The
per-(brand, region) aggregate rows (metric_type in trx/nrx/market_share/...)
and every other column are LEFT UNTOUCHED.

BLAST RADIUS (consumers of the modified columns — verified for revision 1;
re-checked 2026-07-08)
-----------------------------------------------------------------------------
  engagement_score / call_frequency / new columns (per_hcp_rollup): read by
    ExperimentOutcomeRepository only for experiments whose assignments overlap
    these hcp_ids (zero overlap, verified rev 1); the new columns are read only
    by the digital-twin cohort loader (the whole point).
  conversion_rate (per_hcp_rollup): consumers unchanged from rev 1 — the
    gap_analyzer/chatbot read the AGGREGATE conversion_rate rows, not
    per_hcp_rollup. Regeneration shifts /segments/analyze recovered CATEs
    slightly (added channel variance) while preserving the planted engagement
    taus — re-proven by the recovery probe before any --execute.

USAGE
-----
    # DEFAULT: dry-run. Reads live rows, derives the DGP in-memory, prints per-
    # channel planted vs naive vs recovered effects (EconML probe mirroring the
    # agents), and writes a TSV backup of CURRENT live values. Writes NOTHING.
    python scripts/backfill_segment_engagement.py

    # Probe only selected channels (faster iteration):
    python scripts/backfill_segment_engagement.py --channels engagement_score,call_frequency

    # WRITE PATH — DO NOT RUN unless you intend to UPDATE ~12k rows.
    python scripts/backfill_segment_engagement.py --execute
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
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

_NE = RegionEnum.NORTHEAST.value
_W = RegionEnum.WEST.value
_S = RegionEnum.SOUTH.value
_MW = RegionEnum.MIDWEST.value

# ---------------------------------------------------------------------------
# DGP CONSTANTS (documented true effects)
# ---------------------------------------------------------------------------

# LEGACY (revision-1) engagement channel — UNCHANGED so engagement_score values
# and the /segments/analyze planted CATE are preserved for the same seed.
TRUE_CATE_BY_REGION: Dict[str, float] = {_NE: 0.45, _W: 0.30, _S: 0.18, _MW: 0.08}

_ENG_INTERCEPT: float = 0.0
_ENG_BETA_MARKET: float = 1.2  # per unit market_share (already ~0..0.3)
_ENG_BETA_VOLUME: float = 0.35  # per unit log1p(total_rx_count)
_ENG_REGION_OFFSET: Dict[str, float] = {_NE: 0.40, _W: 0.10, _S: -0.10, _MW: -0.30}
_ENG_NOISE_STD: float = 0.8  # on the logit scale
_ENG_MAX: float = 10.0  # engagement_score domain [0, 10]

# Outcome model: confounded baseline + SUM_k tau_k[region]*Tbin_k. baseline()
# shares the confounders with every treatment model, so naive contrasts are
# biased -> de-confounding is required to recover each tau_k[region].
_OUT_BASELINE_INTERCEPT: float = 0.50
_OUT_BETA_MARKET: float = 0.80  # confounder: market_share -> conversion
_OUT_BETA_VOLUME: float = 0.06  # confounder: log1p(total_rx_count) -> conversion
_OUT_REGION_BASELINE: Dict[str, float] = {  # confounding region intercept (NOT the CATE)
    _NE: 0.20,
    _W: 0.05,
    _S: 0.10,
    _MW: 0.15,
}
_OUT_NOISE_STD: float = 0.25
_OUT_MIN: float = 0.0  # conversion_rate kept non-negative


@dataclass(frozen=True)
class ChannelSpec:
    """A confounded treatment channel with a planted per-region true CATE."""

    column: str
    intervention: str  # canonical intervention value (docs/reporting only)
    kind: str  # "sigmoid10" | "poisson" | "share01"
    intercept: float
    beta_market: float
    beta_volume: float
    region_offset: Dict[str, float]
    noise_std: float
    tau_by_region: Dict[str, float]
    lam_base: float = 0.0  # poisson only
    lam_scale: float = 0.0  # poisson only


# Revision-2 channels (engagement_score is handled by the legacy stream above).
# Region offsets mirror the access/territory-richness ordering; betas vary per
# channel for realism. Taus: monotone NE > W > S > MW, distinct magnitudes.
CHANNEL_SPECS: tuple[ChannelSpec, ...] = (
    ChannelSpec(
        column="call_frequency",
        intervention="call_frequency_increase",
        kind="poisson",
        intercept=0.0,
        beta_market=1.0,
        beta_volume=0.30,
        region_offset={_NE: 0.35, _W: 0.10, _S: -0.10, _MW: -0.30},
        noise_std=0.8,
        lam_base=1.0,
        lam_scale=8.0,
        tau_by_region={_NE: 0.20, _W: 0.14, _S: 0.08, _MW: 0.03},
    ),
    ChannelSpec(
        column="email_campaign_count",
        intervention="email_campaign",
        kind="poisson",
        intercept=0.0,
        beta_market=0.8,
        beta_volume=0.45,
        region_offset={_NE: 0.30, _W: 0.05, _S: -0.05, _MW: -0.25},
        noise_std=0.8,
        lam_base=2.0,
        lam_scale=10.0,
        tau_by_region={_NE: 0.24, _W: 0.16, _S: 0.09, _MW: 0.03},
    ),
    ChannelSpec(
        column="speaker_program_count",
        intervention="speaker_program_invitation",
        kind="poisson",
        intercept=0.0,
        beta_market=1.4,
        beta_volume=0.25,
        region_offset={_NE: 0.45, _W: 0.15, _S: -0.10, _MW: -0.35},
        noise_std=0.8,
        lam_base=0.3,
        lam_scale=3.0,
        tau_by_region={_NE: 0.36, _W: 0.25, _S: 0.15, _MW: 0.06},
    ),
    ChannelSpec(
        column="sample_volume",
        intervention="sample_distribution",
        kind="poisson",
        intercept=0.0,
        beta_market=0.9,
        beta_volume=0.50,
        region_offset={_NE: 0.25, _W: 0.10, _S: -0.05, _MW: -0.20},
        noise_std=0.8,
        lam_base=5.0,
        lam_scale=25.0,
        tau_by_region={_NE: 0.18, _W: 0.12, _S: 0.07, _MW: 0.02},
    ),
    ChannelSpec(
        column="peer_influence_score",
        intervention="peer_influence_activation",
        kind="sigmoid10",
        intercept=0.0,
        beta_market=1.1,
        beta_volume=0.30,
        region_offset={_NE: 0.40, _W: 0.10, _S: -0.10, _MW: -0.30},
        noise_std=0.8,
        tau_by_region={_NE: 0.30, _W: 0.21, _S: 0.13, _MW: 0.05},
    ),
    ChannelSpec(
        column="patient_support_enrollment",
        intervention="patient_support_program",
        kind="share01",
        intercept=-0.2,
        beta_market=0.9,
        beta_volume=0.35,
        region_offset={_NE: 0.30, _W: 0.10, _S: -0.10, _MW: -0.25},
        noise_std=0.8,
        tau_by_region={_NE: 0.26, _W: 0.18, _S: 0.11, _MW: 0.05},
    ),
    ChannelSpec(
        column="rep_training_score",
        intervention="rep_training_quality",
        kind="sigmoid10",
        intercept=0.2,
        beta_market=0.7,
        beta_volume=0.20,
        region_offset={_NE: 0.30, _W: 0.10, _S: -0.05, _MW: -0.20},
        noise_std=0.8,
        tau_by_region={_NE: 0.16, _W: 0.10, _S: 0.05, _MW: 0.01},
    ),
)

# Every planted channel, engagement first (legacy), for reporting/probing.
ALL_CHANNEL_TAUS: Dict[str, Dict[str, float]] = {
    "engagement_score": TRUE_CATE_BY_REGION,
    **{spec.column: spec.tau_by_region for spec in CHANNEL_SPECS},
}
_TREATMENT_COLUMNS: tuple[str, ...] = tuple(ALL_CHANNEL_TAUS)

# Covariate columns READ from the live table (causal inputs; never re-drawn) +
# the columns we will (re)derive, for the pre-write backup + comparison.
# NOTE: the six migration-099 columns must exist (apply migration 099 first).
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
    "conversion_rate",  # current live value -> backup + before/after
    *_TREATMENT_COLUMNS,  # current live values -> backup
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
        logger.warning(
            "Could not read live %s: %s (if a migration-099 column is missing, "
            "apply database/migrations/099_business_metrics_intervention_treatments.sql "
            "first)",
            TABLE,
            e,
        )
        return None


# ---------------------------------------------------------------------------
# DGP (pure, deterministic)
# ---------------------------------------------------------------------------


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _within_brand_tbin(values: np.ndarray, brands: np.ndarray) -> np.ndarray:
    """Above-median binarization WITHIN brand — mirrors the estimators, which
    load per-brand cohorts and binarize the treatment at its (per-brand) median."""
    t_bin = np.zeros(len(values), dtype=int)
    for b in pd.unique(brands):
        m = brands == b
        med = float(np.median(values[m]))
        t_bin[m] = (values[m] > med).astype(int)
    return t_bin


def generate_dgp(rows: pd.DataFrame, *, seed: int = DEFAULT_SEED) -> pd.DataFrame:
    """Derive every treatment channel + the outcome from the live confounders,
    planting the documented per-channel, per-region true CATEs.

    Returns a frame keyed on metric_id with the regenerated columns plus the
    per-channel planted bins/taus (for the recovery probe + verification).
    engagement_score and the outcome noise reuse the revision-1 main-stream draw
    order, so for the same seed engagement values are IDENTICAL to revision 1.
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
    n = len(df)

    market = df["market_share"].to_numpy(dtype=float)
    market_c = market - float(np.mean(market))
    logvol = np.log1p(df["total_rx_count"].to_numpy(dtype=float))
    logvol_c = logvol - float(np.mean(logvol))
    region = df["region"].to_numpy(dtype=str)
    brands = df["brand"].to_numpy(dtype=str)

    # --- LEGACY main-stream draws (ORDER PRESERVED vs revision 1) ---
    # Draw 1: engagement noise -> engagement_score identical to revision 1.
    eng_region = np.array([_ENG_REGION_OFFSET.get(r, 0.0) for r in region], dtype=float)
    eng_logit = (
        _ENG_INTERCEPT
        + _ENG_BETA_MARKET * market_c
        + _ENG_BETA_VOLUME * logvol_c
        + eng_region
        + rng.normal(0.0, _ENG_NOISE_STD, n)
    )
    engagement = np.clip(_ENG_MAX * _sigmoid(eng_logit), 0.0, _ENG_MAX)
    # Draw 2: outcome noise (same stream position as revision 1).
    out_noise = rng.normal(0.0, _OUT_NOISE_STD, n)

    result = df[[KEY, "brand", "region"]].copy()
    result["engagement_score"] = np.round(engagement, 4)

    tbin_by_col: Dict[str, np.ndarray] = {
        "engagement_score": _within_brand_tbin(engagement, brands)
    }

    # --- Revision-2 channels: per-channel child RNGs (order-independent,
    # idempotent for the same seed regardless of channel iteration order). ---
    for idx, spec in enumerate(CHANNEL_SPECS):
        rng_k = np.random.default_rng([seed, 1000 + idx])
        latent = (
            spec.intercept
            + spec.beta_market * market_c
            + spec.beta_volume * logvol_c
            + np.array([spec.region_offset.get(r, 0.0) for r in region], dtype=float)
            + rng_k.normal(0.0, spec.noise_std, n)
        )
        if spec.kind == "sigmoid10":
            values = np.clip(10.0 * _sigmoid(latent), 0.0, 10.0)
            result[spec.column] = np.round(values, 4)
        elif spec.kind == "share01":
            values = np.clip(_sigmoid(latent), 0.0, 1.0)
            result[spec.column] = np.round(values, 4)
        elif spec.kind == "poisson":
            lam = np.clip(spec.lam_base + spec.lam_scale * _sigmoid(latent), 0.05, None)
            values = rng_k.poisson(lam=lam).astype(float)
            result[spec.column] = values
        else:  # pragma: no cover - spec typo guard
            raise ValueError(f"unknown channel kind '{spec.kind}' for {spec.column}")
        tbin_by_col[spec.column] = _within_brand_tbin(
            result[spec.column].to_numpy(dtype=float), brands
        )

    # --- OUTCOME: baseline(confounders) + SUM_k tau_k[region]*Tbin_k + noise ---
    out_region_base = np.array([_OUT_REGION_BASELINE.get(r, 0.0) for r in region], dtype=float)
    baseline = (
        _OUT_BASELINE_INTERCEPT
        + _OUT_BETA_MARKET * market_c
        + _OUT_BETA_VOLUME * logvol_c
        + out_region_base
    )
    conversion = baseline + out_noise
    for col, taus in ALL_CHANNEL_TAUS.items():
        tau = np.array([taus.get(r, 0.0) for r in region], dtype=float)
        conversion = conversion + tau * tbin_by_col[col].astype(float)
        # internal-only (NOT written): planted bins + per-unit taus for the probe.
        result[f"_tbin_{col}"] = tbin_by_col[col]
        result[f"_tau_{col}"] = tau
    result["conversion_rate"] = np.round(np.clip(conversion, _OUT_MIN, None), 4)

    # De-confounding nuisances the recovery probe routes into W.
    result["_mkt"] = market
    result["_logvol"] = logvol
    return result


# ---------------------------------------------------------------------------
# Recovery probe (faithful to the agents' EconML estimator)
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


def recovery_probe(regen: pd.DataFrame, channel_col: str) -> Optional[Dict[str, Any]]:
    """Recover one channel's ATE + CATE-by-region with the SAME estimator the
    agents use (CausalForestDML, binary above-median treatment), so the dry-run
    proves the planted per-region CATE is recoverable BEFORE any --execute.

    Two fits are reported per channel:
      * UNCONDITIONAL (W=None): the segments agent's DEFAULT behavior. Recovers
        a CONFOUNDED-but-correctly-ordered, non-degenerate CATE-by-region.
      * DE-CONFOUNDED (W = market_share + log volume): proves the planted true
        tau[region] is RECOVERABLE under proper adjustment — i.e. the signal is
        a genuine confounded causal structure. This mirrors the digital-twin
        CohortCausalEstimator (which adjusts for the same confounders).
    """
    try:
        import econml  # noqa: F401
    except Exception as e:  # pragma: no cover - econml optional locally
        logger.warning("EconML unavailable (%s); skipping recovery probe.", e)
        return None

    df = regen
    y = df["conversion_rate"].to_numpy(dtype=float)
    t = df[f"_tbin_{channel_col}"].to_numpy(dtype=int)

    # Encode region as the single effect modifier (label-encode, like the agent).
    regions = df["region"].astype(str)
    cats = list(pd.unique(regions))
    code = {c: i for i, c in enumerate(cats)}
    x = regions.map(code).to_numpy(dtype=float).reshape(-1, 1)

    w = np.column_stack([df["_mkt"].to_numpy(dtype=float), df["_logvol"].to_numpy(dtype=float)])

    return {
        "unconditional": _fit_cate(y, t, x, regions, cats, None),
        "deconfounded": _fit_cate(y, t, x, regions, cats, w),
    }


# ---------------------------------------------------------------------------
# Verification / reporting
# ---------------------------------------------------------------------------


def _report_channel_fit(
    label: str, fit: Optional[Dict[str, Any]], taus: Dict[str, float], true_ate: float
) -> bool:
    """Log one probe fit; return True when the region ORDERING is recovered."""
    if fit is None:
        logger.info("  [%s] (skipped)", label)
        return False
    est = fit["cate_by_region_est"]
    logger.info("  [%s] recovered ATE = %+.4f (true %+.4f)", label, fit["ate_est"], true_ate)
    for r in taus:
        logger.info(
            "  [%s] CATE[%-9s] = %+.4f   (true %+.4f)",
            label,
            r,
            est.get(r, float("nan")),
            taus[r],
        )
    ordered_true = sorted(taus, key=lambda r: -taus[r])
    ordered_est = sorted(est, key=lambda r: -est.get(r, 0.0))
    ok = ordered_est == ordered_true
    logger.info(
        "  [%s] CATE ordering recovered = %s  spread = %.4f",
        label,
        ok,
        float(max(est.values()) - min(est.values())) if est else 0.0,
    )
    return ok


def verify(
    regen: pd.DataFrame,
    live: pd.DataFrame,
    *,
    run_probe: bool,
    probe_channels: List[str],
) -> bool:
    """Report distributions + planted-vs-naive-vs-recovered effects per channel.

    Returns True when every probed channel recovers its region ordering under
    de-confounding (the acceptance gate for --execute)."""
    n = len(regen)
    region_counts = regen["region"].value_counts().to_dict()

    logger.info("--- PLANTED TRUE EFFECTS (documented; effect of above-median T) ---")
    true_ates: Dict[str, float] = {}
    for col, taus in ALL_CHANNEL_TAUS.items():
        true_ates[col] = sum((region_counts.get(r, 0) / n) * tau for r, tau in taus.items())
        logger.info(
            "  %-27s taus NE/W/S/MW = %+.2f/%+.2f/%+.2f/%+.2f  pop-ATE = %+.4f",
            col,
            taus.get(_NE, 0.0),
            taus.get(_W, 0.0),
            taus.get(_S, 0.0),
            taus.get(_MW, 0.0),
            true_ates[col],
        )

    logger.info("--- TREATMENT DISTRIBUTIONS ---")
    for col in ALL_CHANNEL_TAUS:
        v = regen[col].to_numpy(dtype=float)
        logger.info(
            "  %-27s min=%.3f median=%.3f max=%.3f mean=%.3f treated-frac=%.3f",
            col,
            float(np.min(v)),
            float(np.median(v)),
            float(np.max(v)),
            float(np.mean(v)),
            float(regen[f"_tbin_{col}"].mean()),
        )

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

    # Naive (confounded) per-region difference-in-means — should DIFFER from
    # the true tau (proves the data is confounded, not reverse-correlated).
    logger.info("--- NAIVE per-region diff-in-means (CONFOUNDED, NOT the truth) ---")
    for col in probe_channels:
        taus = ALL_CHANNEL_TAUS[col]
        for r in taus:
            m = regen["region"].to_numpy(dtype=str) == r
            if not m.any():
                continue
            tr = regen.loc[m & (regen[f"_tbin_{col}"] == 1), "conversion_rate"]
            ct = regen.loc[m & (regen[f"_tbin_{col}"] == 0), "conversion_rate"]
            if len(tr) and len(ct):
                logger.info(
                    "  %-27s naive[%-9s] = %+.4f  (true %+.4f)",
                    col,
                    r,
                    float(tr.mean() - ct.mean()),
                    taus[r],
                )

    if not run_probe:
        return True

    all_ok = True
    for col in probe_channels:
        taus = ALL_CHANNEL_TAUS[col]
        logger.info("--- RECOVERY PROBE: %s (CausalForestDML, agent-faithful) ---", col)
        probe = recovery_probe(regen, col)
        if probe is None:
            logger.info("  (probe skipped — econml unavailable)")
            all_ok = False
            continue
        # Agent DEFAULT (W=None): confounded but non-degenerate (advisory).
        _report_channel_fit("AGENT-DEFAULT W=None", probe["unconditional"], taus, true_ates[col])
        # DE-CONFOUNDED (W routed): the acceptance gate — the planted true
        # tau[region] must be recoverable under adjustment.
        ok = _report_channel_fit(
            "DE-CONFOUNDED W=mkt,vol", probe["deconfounded"], taus, true_ates[col]
        )
        all_ok = all_ok and ok
    logger.info(
        "PROBE %s: de-confounded region ordering recovered for %s",
        "PASSED" if all_ok else "FAILED",
        ", ".join(probe_channels),
    )
    return all_ok


def write_backup(live: pd.DataFrame, out_dir: Path) -> Path:
    """Write a TSV backup of the CURRENT live values before any write."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    path = out_dir / f"business_metrics_per_hcp_segment_backup_{ts}.tsv"
    cols = [
        c
        for c in (KEY, "hcp_id", "brand", "region", "conversion_rate", *_TREATMENT_COLUMNS)
        if c in live.columns
    ]
    live[cols].to_csv(path, sep="\t", index=False)
    logger.info("Wrote backup of %d live rows to %s", len(live), path)
    return path


# ---------------------------------------------------------------------------
# Update (write path — only reachable with --execute)
# ---------------------------------------------------------------------------


def update_rows(client: Any, regen: pd.DataFrame, *, batch_size: int = BATCH_SIZE) -> int:
    """Idempotent per-row UPDATE of every planted column keyed on metric_id.
    Only the planted columns change; every other column (trx_count,
    market_share, region, brand, hcp_id, is_synthetic, ...) is untouched.
    Re-running with the same seed reproduces identical values.
    """
    written = 0
    write_cols = ["conversion_rate", *_TREATMENT_COLUMNS]
    records = regen[[KEY, *write_cols]].to_dict(orient="records")
    for rec in records:
        client.table(TABLE).update({c: float(rec[c]) for c in write_cols}).eq(
            KEY, rec[KEY]
        ).execute()
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
        help="WRITE PATH: UPDATE every planted treatment column + conversion_rate on "
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
        "--channels",
        type=str,
        default="all",
        help="Comma-separated treatment columns to probe/report (default: all). "
        f"Known: {','.join(_TREATMENT_COLUMNS)}",
    )
    parser.add_argument(
        "--backup-dir",
        type=str,
        default=str(_PROJECT_ROOT / "data" / "backups"),
        help="Directory for the pre-write TSV backup of live values.",
    )
    args = parser.parse_args()
    dry_run = not args.execute

    if args.channels.strip().lower() == "all":
        probe_channels = list(_TREATMENT_COLUMNS)
    else:
        probe_channels = [c.strip() for c in args.channels.split(",") if c.strip()]
        unknown = [c for c in probe_channels if c not in ALL_CHANNEL_TAUS]
        if unknown:
            logger.error("Unknown channel(s) %s. Known: %s", unknown, list(_TREATMENT_COLUMNS))
            return 1

    logger.info("=" * 72)
    logger.info(
        "multi-channel intervention DGP backfill  (%s)  seed=%d",
        "DRY RUN" if dry_run else "EXECUTE",
        args.seed,
    )
    logger.info("  table=%s metric_type=%s channels=%d", TABLE, METRIC_TYPE, len(ALL_CHANNEL_TAUS))
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
    probe_ok = verify(
        regen, live, run_probe=not args.no_recovery_probe, probe_channels=probe_channels
    )

    if dry_run:
        logger.info(
            "DRY RUN complete. No rows updated. Re-run with --execute to write "
            "the planted treatment channels + conversion_rate."
        )
        return 0 if probe_ok else 2

    if not args.no_recovery_probe and not probe_ok:
        logger.error(
            "REFUSING --execute: the de-confounded recovery probe failed for at "
            "least one channel (planted effect not recoverable). Fix the DGP "
            "constants (tau gaps/noise) and re-run."
        )
        return 2

    n = update_rows(client, regen)
    logger.info("EXECUTE complete: updated %d rows in %s (idempotent on %s).", n, TABLE, KEY)
    return 0


if __name__ == "__main__":
    sys.exit(main())
