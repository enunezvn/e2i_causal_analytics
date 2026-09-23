#!/usr/bin/env python3
"""Backfill a confounded synthetic ``treatment_arm`` onto ``hcp_brand_adoption``
and re-derive its causally-linked ``adopted`` label from the committed DGP.

WHY THIS SCRIPT EXISTS
----------------------
``hcp_brand_adoption`` (15,000 rows = 5,000 synthetic HCPs x 3 brands; migration
076) carries the gold-standard HCP-adoption cohort's label ``adopted`` but has NO
``treatment_arm`` column. The HCP-adoption "Treatment Effects" surface therefore
has nothing to estimate an ATE from. The user wants a synthetic treatment arm so
the per-brand HCP-adoption ATE is real and recoverable.

The committed DGP ``_compute_adoption`` (src/ml/synthetic/generators/
hcp_adoption_artifact.py) ALREADY generates exactly this structure:

    centrality_z      ~ EXOGENOUS (stored peer_influence_score, NEVER re-drawn)
    hcp_segment       = centrality tier {high,medium,low}_influence (effect modifier)
    treatment_arm     ~ Bernoulli(sigmoid(0.8*centrality_z + noise))   # CONFOUNDED
    affinity          = per-(brand, specialty) DETERMINISTIC logit shift (#1551;
                        stored hcp_profiles.specialty, consumes NO rng draws)
    adoption_logit    = a + b*centrality_z + affinity
                        + scale*tau(segment)*treatment_arm + noise
    adopted           ~ Bernoulli(sigmoid(adoption_logit))             # T -> Y
    cate_estimate     = sigmoid(base + scale*tau) - sigmoid(base)      # per-HCP RD
                        (base includes the affinity term)

i.e. ``adopted`` is CAUSALLY generated FROM ``treatment_arm`` (treatment -> outcome),
heterogeneous by HCP influence segment, scaled per brand by ``_BRAND_ADOPT_SCALE``
(Fabhalta 1.2 > Remibrutinib 1.0 > Kisqali 0.8). The original ad-hoc load
(scripts/load_hcp_brand_adoption.py) called this DGP but persisted ONLY ``adopted``
-- it threw away the ``treatment_arm`` and ``cate_estimate`` it had just drawn.

WHAT IT DOES
------------
Reads the EXISTING per-HCP covariates ``peer_influence_score`` + ``specialty``
from the live ``hcp_profiles`` table (the causal inputs -- NEVER re-drawn),
standardizes the centrality exactly as the generator does, and per brand re-runs
``_compute_adoption`` (specialty threaded through for the #1551 affinity) to draw:
  * ``treatment_arm`` (0/1, confounded by centrality)  -- NEW column
  * ``adopted``       (0/1, CAUSALLY from treatment_arm) -- re-derived
keyed back to ``hcp_brand_adoption`` rows on ``(hcp_id, brand)``. The per-brand
TRUE ATE (mean prob-scale per-HCP ``cate_estimate``) is computed and printed so it
is verifiable.

WHY ``adopted`` IS RE-DERIVED (the honest choice; not a free lunch)
------------------------------------------------------------------
The live ``adopted`` labels were drawn by THIS SAME DGP at the original (uncommitted)
seed -- but the ``treatment_arm`` that produced them was discarded and the seed/RNG
threading was not committed (the load script's sweep found only ~63% per-row
agreement). So we CANNOT recover the exact arm that generated the live labels.

Two options were considered:
  * ADD treatment_arm WITHOUT touching ``adopted``. REJECTED: the new arm would
    NOT be the arm that generated the live label, so any "ATE" recovered against
    the live ``adopted`` would be a SPURIOUS reverse-correlation, exactly the
    anti-pattern the honesty directive forbids. The causal link must be real.
  * RE-DERIVE both arm and label together from the committed DGP on the stored
    centrality + specialty (this script). The treatment -> outcome link is
    INTACT and the TRUE ATE is documented. Cost: ``adopted`` moves, so the 3
    deployed HCP gold-standard models must be RE-TRAINED on the new labels
    (one-time). The aggregate adoption rate stays in the live band (~0.40-0.43)
    so the models' base rate and AUC ceiling are preserved -- see the dry-run
    prevalence print and the BLAST RADIUS section.

    #1551 NOTE: "the committed DGP" now INCLUDES the per-(brand, specialty)
    adoption affinity (_BRAND_SPECIALTY_AFFINITY), so this script threads the
    stored hcp_profiles.specialty into _compute_adoption. A specialty-blind
    re-derivation would silently UPDATE the live labels back to the
    specialty-free distribution and undo the served-propensity fix. The
    affinity is a deterministic logit shift with ZERO extra rng draws, so the
    treatment_arm stream this script faithfully reproduces is unchanged.

BLAST RADIUS (must read before --execute)
-----------------------------------------
* SCHEMA: ``hcp_brand_adoption`` needs a new ``treatment_arm INTEGER`` column.
  ``--execute`` runs ``ALTER TABLE ... ADD COLUMN IF NOT EXISTS`` first (additive,
  idempotent). A sibling migration is the cleaner home; the ALTER here is a
  convenience so the backfill is self-contained. The column is nullable with no
  default so existing/real (is_synthetic=false) rows are unaffected.
* MODELS: the 3 staging models ``hcp_adoption_{brand}_goldstd_lr_v1`` (AUC
  0.74-0.76, training_samples 4000) use ``adopted`` as their TRAINING LABEL. Their
  FEATURES come from ``hcp_profiles`` (peer_influence_score, influence_network_size,
  years_experience, specialty, geographic_region) -- NOT from this table -- and
  ``treatment_arm`` is NOT in that covariate set, so adding the column does NOT
  change the feature space. BUT re-deriving ``adopted`` changes the label, so the
  models need a one-time RE-TRAIN (run_hcp_cohorts.py) to stay consistent. Until
  retrained they predict against the OLD label distribution; AUC is expected to
  hold (~same prevalence, same DGP family) but should be re-verified.
* TREATMENT-EFFECTS SURFACE: the new ``treatment_arm`` + the documented per-brand
  TRUE ATE are what the HCP-adoption Treatment Effects view consumes (treatment =
  treatment_arm, outcome = adopted, confounders = centrality covariates).

CHANNEL RE-PLANT (lane T1, owner decision 2026-09-23: "approve DGP extension, we
need to recover statistical, not structural effects")
---------------------------------------------------------------------------------
This script is THE re-plant path for the planted channel effects on ``adopted``:
seed 427 reproduces the live table (treatment_arm 100 %, adopted 99.9 %; the
loader script's regenerated frame does not, ~63 %). On every run it now also

  (a) reads the planted ``business_metrics`` ``per_hcp_rollup`` rows of the three
      brands (``is_synthetic``; paged to exhaustion with ``count="exact"``, failing
      loud on a short page),
  (b) collapses them to ONE exposure per (hcp_id, brand) with
      ``src.data.per_hcp_cohort_collapse`` (SUM counts, MEAN scores, FIRST labels),
      bins each channel above its within-brand median (the estimator's own
      contrast) and builds the per-HCP logit shift
      ``sum_k beta_k * (tbin_k - 0.5)`` from ``ADOPTION_CHANNEL_LOGIT_BETA``
      (``src.data.per_hcp_cohort_columns``); HCPs with no rollup row for a brand
      (~1,600 per brand) are NOT joined: shift 0, labels untouched by the term;
  (c) passes that shift into ``_compute_adoption`` -- a deterministic term beside
      the #1551 affinity, consuming NO rng draws, so ``treatment_arm`` stays
      byte-identical to the live column (the dry-run prints the match; it must
      read 1.0000) and ``adopted`` moves only where the shifted sigmoid crosses
      the SAME uniform (~9.5 % of rows);
  (d) re-dates JOINED rows: ``consideration_date = max(metric_date) + lag``, lag
      ~ integers(1, 61) from a SPAWNED stream ``default_rng([brand_seed, 1])`` (the
      four DGP draws are untouched), CLAMPED to the run date so no row is dated in
      the future. CONSEQUENCE, not hidden: the Aug/Sep-exposed pairs (< 10 %) can
      land past the run date and are clamped onto it (the dry-run prints the count
      per brand), and the gold-standard walk-forward axis (``feature_builder``
      aliases ``consideration_date -> journey_start_date``) will see joined rows
      clumped in Jul-Sep 2026 instead of 37 even months. A 90-day lag was measured
      and rejected: it clamped ~1/3 of joined rows onto one date. Non-joined rows
      keep their live dates (the writer omits the column for them);
  (e) stamps ``updated_at = now()`` on every written row (the table has no
      trigger; until now the script left it at 2026-06-14).

The dry-run report prints, per brand: rows, joined/non-joined, arm match vs live
(must be 1.0000), prevalence before/after, labels flipped, the realised risk
difference per channel (stratified on the exposure bit, and the DGP-true
counterfactual contrast), the null channel's RD, that every joined row is dated
after its last exposure, and the future-date count (must be 0). ``--frame-out``
writes the generated frame (with the collapsed exposure) to parquet for
``scripts/verify_adoption_channel_recovery.py --frame``.

OWNER ACTIONS, in order (after the PR merges):
  1. ``python scripts/backfill_hcp_treatment_arm.py`` (dry-run; read the report)
  2. ``python scripts/backfill_hcp_treatment_arm.py --execute``
  3. ``python scripts/verify_adoption_channel_recovery.py --live`` -> must PASS
  4. ``scripts/retrain_goldstd.sh`` (or wait for Monday's cron) to re-train the
     three HCP champions on the new labels; AUC expected ~0.78 (band 0.70-0.86)
  5. lane T2 re-points the twin's endpoint at ``adopted`` (loader collapse +
     population-ATE interval).

USAGE
-----
    # DEFAULT: dry-run. Reads live centrality + planted rollups, derives arm+label
    # (+ channel shift + consideration_date) in memory, prints the report above,
    # and writes a TSV backup of the CURRENT live labels. Writes NOTHING to the DB.
    python scripts/backfill_hcp_treatment_arm.py

    python scripts/backfill_hcp_treatment_arm.py --seed 427 --run-date 2026-09-23 \
        --frame-out /tmp/adoption_frame.parquet

    # WRITE PATH -- DO NOT RUN against prod unless you intend to ALTER the table and
    # UPDATE 15k rows (treatment_arm + adopted + adoption_category + consideration_date
    # for joined rows + updated_at). The human runs this after review.
    python scripts/backfill_hcp_treatment_arm.py --execute
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(_PROJECT_ROOT / ".env")

from src.data.per_hcp_cohort_collapse import (  # noqa: E402
    CHANNEL_COLUMNS,
    adoption_channel_shift,
    align_channel_shift,
    channel_tbin,
    collapse_per_hcp_brand,
    dgp_true_channel_rd,
    stratified_channel_rd,
    tbin_column,
)
from src.data.per_hcp_cohort_columns import (  # noqa: E402
    ADOPTION_NULL_CHANNEL,
    INTERVENTION_TREATMENT_MAP,
)
from src.ml.synthetic.generators.hcp_adoption_artifact import (  # noqa: E402
    _BRAND_ADOPT_SCALE,
    ADOPTER_VALUE,
    _compute_adoption,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

ADOPTION_TABLE = "hcp_brand_adoption"
PROFILES_TABLE = "hcp_profiles"
ROLLUP_TABLE = "business_metrics"
ROLLUP_METRIC_TYPE = "per_hcp_rollup"
_NON_ADOPTER_VALUE = "NON_ADOPTER"
NULL_CHANNEL_COLUMN = INTERVENTION_TREATMENT_MAP[ADOPTION_NULL_CHANNEL]
# Lag (days) from a joined pair's last planted exposure to its consideration_date:
# integers(LAG_MIN, LAG_MAX_EXCLUSIVE) from the spawned stream default_rng([brand_seed, 1]).
# 1..60, not 1..90 (lead decision 2026-09-23, measured): the plant's 07-21 write date is
# the last exposure for 9,166 of 10,136 pairs, so a 90-day lag clamped ~1/3 of joined rows
# (3,253) onto the run date -- one same-date clump on the goldstd walk-forward axis. With
# 60 days the 07-21 mass lands 07-22..09-19, before the run date; only the Aug/Sep-exposed
# pairs (< 10 %) can hit the clamp.
LAG_MIN = 1
LAG_MAX_EXCLUSIVE = 61
# The planted exposure columns read from the rollups, besides the eight channels.
_ROLLUP_CONTEXT_COLUMNS = (
    "hcp_id",
    "brand",
    "metric_date",
    "region",
    "market_share",
    "triggers_total_count",
)
_ROLLUP_SELECT = ",".join((*_ROLLUP_CONTEXT_COLUMNS, *CHANNEL_COLUMNS))
_ROLLUP_NUMERIC = ("market_share", "triggers_total_count", *CHANNEL_COLUMNS)

# Default seed for the per-brand re-derivation RNG. EMPIRICAL: the ad-hoc load's
# seed was not committed; seed=427 is the load script's best AGGREGATE match to the
# live per-brand prevalence (0.407/0.417/0.399). Overridable via --seed. Per-row
# labels will NOT match the live table bit-for-bit (the discarded arm/seed cannot
# be recovered) -- --dry-run quantifies the exact movement so the verdict is
# data-driven. The brand RNG is derived from a master_rng exactly as
# generate_hcp_brand_adoption_frame does, so this mirrors the original load.
DEFAULT_SEED = 427

# Brands in the same order the original load iterated them (master_rng stream
# order matters for the per-brand sub-seed derivation -> faithful reproduction).
BRANDS = ("Remibrutinib", "Fabhalta", "Kisqali")
KEY = ["hcp_id", "brand"]


# ---------------------------------------------------------------------------
# Live read (read-only)
# ---------------------------------------------------------------------------


def fetch_centrality(client: Any) -> Optional[pd.DataFrame]:
    """Read the EXISTING per-HCP causal inputs from hcp_profiles (paged).

    Returns hcp_id + peer_influence_score + specialty for the synthetic HCPs.
    These are the causal inputs -- NEVER re-drawn; treatment_arm and adopted are
    derived FROM them, exactly as the generator does (specialty feeds the #1551
    per-(brand, specialty) adoption affinity).
    """
    try:
        rows: List[dict] = []
        page = 0
        page_size = 1000
        while True:
            resp = (
                client.table(PROFILES_TABLE)
                .select("hcp_id,peer_influence_score,specialty")
                .eq("is_synthetic", True)
                .order("hcp_id")
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
        df = pd.DataFrame(rows)
        df["peer_influence_score"] = df["peer_influence_score"].astype(float)
        return df.sort_values("hcp_id").reset_index(drop=True)
    except Exception as e:  # pragma: no cover - network/permission edge
        logger.warning("Could not read live %s: %s", PROFILES_TABLE, e)
        return None


def fetch_live_adoption(client: Any) -> Optional[pd.DataFrame]:
    """Read the CURRENT live hcp_brand_adoption rows (for backup + comparison)."""
    try:
        rows: List[dict] = []
        page = 0
        page_size = 1000
        cols = "hcp_id,brand,adopted,adoption_category,treatment_arm,consideration_date"
        while True:
            resp = (
                client.table(ADOPTION_TABLE)
                .select(cols)
                .eq("is_synthetic", True)
                # Total order: hcp_id repeats once per brand, and a page boundary inside
                # one hcp_id's rows under ORDER BY hcp_id alone duplicated one row and
                # dropped another (5001/5000/4999 per brand, measured 2026-09-23).
                .order("hcp_id")
                .order("brand")
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
    except Exception as e:  # pragma: no cover
        logger.warning("Could not read live %s: %s", ADOPTION_TABLE, e)
        return None


def fetch_channel_rollups(
    client: Any, *, brands: Sequence[str] = BRANDS, page_size: int = 1000
) -> pd.DataFrame:
    """Read the planted ``per_hcp_rollup`` rows of ``brands`` (``is_synthetic``) with the
    eight channel columns and their context (metric_date, region, market_share,
    triggers_total_count), paged to exhaustion.

    Exact-count idiom: ``count="exact"`` rides the first page (PostgREST Content-Range),
    pages continue until that many rows are in hand, and a page shorter than
    ``page_size`` BEFORE the count is reached raises -- a silent partial read would plant
    the shift on a subset and call the rest non-joined. An empty table returns an empty
    frame. Raises on any read error (this is the re-plant's input; nothing to fall back to).
    """
    rows: List[dict] = []
    total: Optional[int] = None
    page = 0
    while True:
        resp = (
            client.table(ROLLUP_TABLE)
            .select(_ROLLUP_SELECT, count="exact")
            .eq("metric_type", ROLLUP_METRIC_TYPE)
            .eq("is_synthetic", True)
            .in_("brand", list(brands))
            .order("metric_id")
            .range(page * page_size, (page + 1) * page_size - 1)
            .execute()
        )
        batch = resp.data or []
        if total is None:
            total = getattr(resp, "count", None)
            if total is None:
                raise RuntimeError(
                    f"{ROLLUP_TABLE} read returned no exact count; refusing an unbounded page loop"
                )
        rows.extend(batch)
        if not batch or len(rows) >= total:
            break
        if len(batch) < page_size:
            raise RuntimeError(
                f"{ROLLUP_TABLE} short page: got {len(batch)} < {page_size} rows on page {page} "
                f"with {len(rows)}/{total} rows read"
            )
        page += 1
    if len(rows) != total:
        raise RuntimeError(f"{ROLLUP_TABLE}: read {len(rows)} rows but the server counted {total}")
    if not rows:
        return pd.DataFrame(columns=[*_ROLLUP_CONTEXT_COLUMNS, *CHANNEL_COLUMNS])
    df = pd.DataFrame(rows)
    df["metric_date"] = pd.to_datetime(df["metric_date"])
    for col in _ROLLUP_NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# Derivation (pure, deterministic)
# ---------------------------------------------------------------------------


def derive(
    centrality: pd.DataFrame,
    *,
    seed: int = DEFAULT_SEED,
    channel_rollups: Optional[pd.DataFrame] = None,
    run_date: Optional[date] = None,
) -> pd.DataFrame:
    """Re-derive treatment_arm + adopted per brand from the committed DGP.

    Standardizes the STORED peer_influence_score exactly as
    generate_hcp_brand_adoption_frame does (z-score over the synthetic HCP
    population), then for each brand draws a sub-RNG from the master_rng stream
    (mirroring the original load's brand-seed derivation) and calls
    ``_compute_adoption``. Returns one row per (hcp_id, brand) with treatment_arm,
    adopted, adoption_category, and the per-HCP cate_estimate (for the TRUE ATE).

    #1551: when the frame carries a ``specialty`` column (fetch_centrality now
    selects it), it is threaded into ``_compute_adoption`` so the re-derived
    labels carry the committed per-(brand, specialty) affinity — a
    specialty-blind re-derivation would clobber the fixed labels on --execute.
    NULL/missing specialty values naturally take the brand's default shift via
    ``_specialty_affinity``'s ``.get(s, default)`` — no special casing. The
    affinity consumes NO rng draws, so treatment_arm is byte-identical with or
    without the column (faithful arm reproduction preserved).

    Lane T1: with ``channel_rollups`` (the planted ``per_hcp_rollup`` rows from
    ``fetch_channel_rollups``) the rows are collapsed per (hcp_id, brand), binned
    above the within-brand median, turned into the per-HCP logit shift and passed
    to ``_compute_adoption`` (deterministic; the arm stream is untouched). Joined
    rows get ``consideration_date = max_metric_date + lag`` (lag ~ integers(1, 61)
    from the spawned stream ``default_rng([brand_seed, 1])``, clamped to
    ``run_date``); non-joined
    rows get NaT (the writer leaves their live date). The output then carries, per
    row: ``joined``, ``channel_shift``, ``adoption_logit``, ``max_metric_date``,
    ``n_metric_rows``, ``consideration_date``, ``specialty``, the collapsed exposure
    (region, market_share, triggers_total_count, the eight channels) and the eight
    ``tbin_*`` bits. Raises if any joined pair's last exposure is on/after
    ``run_date`` (no date after the exposure and not in the future exists).
    Without ``channel_rollups`` the legacy six-column output is bit-identical to
    the pre-T1 script.
    """
    hcp_ids = centrality["hcp_id"].tolist()
    n = len(hcp_ids)
    pis = centrality["peer_influence_score"].to_numpy(dtype=float)
    pis_std = pis.std()
    centrality_z = (pis - pis.mean()) / (pis_std if pis_std > 0 else 1.0)
    specialty: Optional[List[str]] = (
        centrality["specialty"].tolist() if "specialty" in centrality.columns else None
    )

    with_channels = channel_rollups is not None
    collapsed = tbin = None
    if with_channels:
        assert channel_rollups is not None
        if run_date is None:
            run_date = date.today()
        # The planted contrast must be the estimator's contrast, and the estimator only sees
        # rows joined to the cohort with every model input non-null (codex r1): restrict the
        # rollups to the cohort's HCPs BEFORE the medians, and refuse nulls (the plant writes
        # every column; a null is corruption, not a value to average around).
        cohort_rollups = channel_rollups[channel_rollups["hcp_id"].isin(hcp_ids)]
        n_orphan = int(len(channel_rollups) - len(cohort_rollups))
        if n_orphan:
            logger.warning(
                "%d rollup rows belong to HCPs outside the cohort (orphans); excluded from "
                "the planting medians and from the join.",
                n_orphan,
            )
        required = ["region", "market_share", "triggers_total_count", *CHANNEL_COLUMNS]
        if len(cohort_rollups):
            # Row level, before the collapse: pandas' sum/mean would silently skip a null
            # inside a multi-row pair and the corruption would vanish into an average.
            null_counts = {c: int(cohort_rollups[c].isna().sum()) for c in required}
            if any(null_counts.values()):
                raise ValueError(
                    "planted rollups carry null model inputs (the estimator would drop these "
                    f"rows while they sat in the planting medians): {null_counts}"
                )
        collapsed = collapse_per_hcp_brand(cohort_rollups)
        if not collapsed.empty:
            tbin = channel_tbin(collapsed)
            collapsed = collapsed.merge(tbin, on=KEY, how="left")
            collapsed["channel_shift"] = adoption_channel_shift(tbin).to_numpy()
            latest = pd.to_datetime(collapsed["max_metric_date"]).max()
            if latest.date() >= run_date:
                raise ValueError(
                    f"a joined pair's last exposure ({latest.date()}) is on/after the run date "
                    f"({run_date}); no consideration_date after the exposure and not in the "
                    "future exists -- pass a later --run-date"
                )

    master_rng = np.random.default_rng(seed)
    frames: List[pd.DataFrame] = []
    for brand in BRANDS:
        # Same per-brand sub-seed derivation as the original load -> faithful.
        brand_seed = int(master_rng.integers(0, 2**32))
        brand_rng = np.random.default_rng(brand_seed)
        shift: Optional[np.ndarray] = None
        brand_coll = None
        if with_channels and collapsed is not None and not collapsed.empty:
            brand_coll = collapsed[collapsed["brand"] == brand].set_index("hcp_id")
            shift = align_channel_shift(hcp_ids, brand_coll["channel_shift"])
        dgp = _compute_adoption(
            brand_rng, centrality_z, brand, specialty=specialty, channel_shift=shift
        )
        adopted = dgp["adopted"].astype(int)
        frame = pd.DataFrame(
            {
                "hcp_id": hcp_ids,
                "brand": brand,
                "treatment_arm": dgp["treatment_arm"].astype(int),
                "adopted": adopted,
                "adoption_category": np.where(adopted == 1, ADOPTER_VALUE, _NON_ADOPTER_VALUE),
                "cate_estimate": dgp["cate_estimate"].astype(float),
            }
        )
        if with_channels:
            assert run_date is not None
            # SPAWNED lag stream: the brand's four DGP draws above are untouched.
            lags = np.random.default_rng([brand_seed, 1]).integers(
                LAG_MIN, LAG_MAX_EXCLUSIVE, size=n
            )
            frame["joined"] = False
            frame["channel_shift"] = dgp["channel_shift"].astype(float)
            frame["adoption_logit"] = dgp["adoption_logit"].astype(float)
            frame["max_metric_date"] = pd.NaT
            frame["n_metric_rows"] = 0
            frame["consideration_date"] = pd.NaT
            if specialty is not None:
                frame["specialty"] = specialty
            for col in ("region", "market_share", "triggers_total_count", *CHANNEL_COLUMNS):
                frame[col] = np.nan
            for col in CHANNEL_COLUMNS:
                frame[tbin_column(col)] = np.nan
            if brand_coll is not None and len(brand_coll):
                joined_mask = frame["hcp_id"].isin(brand_coll.index).to_numpy()
                aligned = brand_coll.reindex(frame["hcp_id"])
                frame["joined"] = joined_mask
                frame["max_metric_date"] = pd.to_datetime(aligned["max_metric_date"]).to_numpy()
                frame["n_metric_rows"] = aligned["n_metric_rows"].fillna(0).astype(int).to_numpy()
                for col in ("region", "market_share", "triggers_total_count", *CHANNEL_COLUMNS):
                    frame[col] = aligned[col].to_numpy()
                for col in CHANNEL_COLUMNS:
                    frame[tbin_column(col)] = aligned[tbin_column(col)].to_numpy(dtype=float)
                proposed = pd.to_datetime(frame["max_metric_date"]) + pd.to_timedelta(
                    lags, unit="D"
                )
                clamped = proposed.where(proposed <= pd.Timestamp(run_date), pd.Timestamp(run_date))
                frame["consideration_date"] = clamped.where(joined_mask, pd.NaT)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def verify(
    derived: pd.DataFrame,
    live: Optional[pd.DataFrame],
    *,
    run_date: Optional[date] = None,
) -> Dict[str, Dict[str, Any]]:
    """Print the per-brand report and return it: ``{brand: {field: value}}``.

    Fields (the owner reads these before ``--execute``): ``n_rows``, ``n_joined``,
    ``n_non_joined``, ``true_ate_arm`` (mean prob-scale CATE = the documented planted
    treatment_arm effect), ``naive_diff_arm``, ``arm_match_vs_live`` (must be 1.0 --
    the seed reproduces the live arm), ``prevalence_live`` / ``prevalence_new``,
    ``labels_flipped``, ``realised_rd_stratified`` (the UNADJUSTED tbin contrast; the
    plant's shared latent confounds it, so it overstates every channel incl. the null)
    / ``realised_rd_dgp_true`` (the planted counterfactual effect) per channel over the
    joined rows, ``null_channel_rd``, ``n_joined_dated_after_last_exposure``
    (must equal ``n_joined``), ``n_future_dates`` (must be 0).
    """
    has_channels = "joined" in derived.columns
    if run_date is None:
        run_date = date.today()
    report: Dict[str, Dict[str, Any]] = {}
    logger.info("--- PER-BRAND DGP SUMMARY (treatment -> outcome) ---")
    logger.info(
        "  %-14s %-7s %-11s %-11s %-22s",
        "brand",
        "scale",
        "treat_rate",
        "adopt_rate",
        "TRUE_ATE (mean prob CATE)",
    )
    for brand in BRANDS:
        sub = derived[derived["brand"] == brand]
        scale = _BRAND_ADOPT_SCALE.get(brand, 1.0)
        ate = float(sub["cate_estimate"].mean())
        t1 = float(sub.loc[sub["treatment_arm"] == 1, "adopted"].mean())
        t0 = float(sub.loc[sub["treatment_arm"] == 0, "adopted"].mean())
        r: Dict[str, Any] = {
            "n_rows": int(len(sub)),
            "n_joined": int(sub["joined"].sum()) if has_channels else 0,
            "n_non_joined": int((~sub["joined"]).sum()) if has_channels else int(len(sub)),
            "treat_rate": round(float(sub["treatment_arm"].mean()), 4),
            "prevalence_new": round(float(sub["adopted"].mean()), 4),
            "true_ate_arm": round(ate, 4),
            "naive_diff_arm": round(t1 - t0, 4),
            "arm_match_vs_live": None,
            "prevalence_live": None,
            "labels_flipped": None,
            "realised_rd_stratified": {},
            "realised_rd_dgp_true": {},
            "null_channel_rd": {},
            "n_joined_dated_after_last_exposure": 0,
            "n_future_dates": 0,
        }
        report[brand] = r
        logger.info(
            "  %-14s %-7.1f %-11.4f %-11.4f %-22.4f",
            brand,
            scale,
            r["treat_rate"],
            r["prevalence_new"],
            ate,
        )
    logger.info(
        "  ATE ordering (prob scale): Fabhalta(1.2) > Remibrutinib(1.0) > Kisqali(0.8) "
        "-- matches _BRAND_ADOPT_SCALE; each ATE = E[P(adopt|do T=1) - P(adopt|do T=0)]."
    )

    # Confounding sanity: adoption rate should be higher among treated (positive,
    # confounded effect) -- a quick directional check the surface will reflect.
    logger.info("--- NAIVE (confounded) vs treated/untreated adoption ---")
    for brand in BRANDS:
        sub = derived[derived["brand"] == brand]
        t1 = float(sub.loc[sub["treatment_arm"] == 1, "adopted"].mean())
        t0 = float(sub.loc[sub["treatment_arm"] == 0, "adopted"].mean())
        logger.info(
            "  %-14s treated_adopt=%.4f untreated_adopt=%.4f naive_diff=%.4f "
            "(>= TRUE_ATE: confounded by centrality, as designed)",
            brand,
            t1,
            t0,
            t1 - t0,
        )

    if live is not None and len(live):
        live_cols = [
            c for c in ("hcp_id", "brand", "adopted", "treatment_arm") if c in live.columns
        ]
        merged = derived[["hcp_id", "brand", "adopted", "treatment_arm"]].merge(
            live[live_cols], on=KEY, how="inner", suffixes=("_new", "_live")
        )
        if len(merged):
            logger.info("--- LABEL MOVEMENT vs live (n=%d) ---", len(merged))
            for brand in BRANDS:
                m = merged[merged["brand"] == brand]
                if not len(m):
                    continue
                r = report[brand]
                r["prevalence_live"] = round(float(m["adopted_live"].mean()), 4)
                r["labels_flipped"] = int((m["adopted_new"] != m["adopted_live"]).sum())
                if "treatment_arm_live" in m.columns and m["treatment_arm_live"].notna().any():
                    arm_live = pd.to_numeric(m["treatment_arm_live"], errors="coerce")
                    r["arm_match_vs_live"] = round(
                        float((m["treatment_arm_new"] == arm_live).mean()), 4
                    )
                logger.info(
                    "  %-14s arm_match=%s  prevalence live=%.4f -> new=%.4f (%+.4f)  "
                    "labels flipped=%d/%d (%.1f%%)  in live band ~0.38-0.45: %s",
                    brand,
                    r["arm_match_vs_live"],
                    r["prevalence_live"],
                    r["prevalence_new"],
                    r["prevalence_new"] - r["prevalence_live"],
                    r["labels_flipped"],
                    len(m),
                    100.0 * r["labels_flipped"] / len(m),
                    "YES" if 0.38 <= r["prevalence_new"] <= 0.45 else "CHECK",
                )
            logger.info(
                "  VERDICT: re-derived label is the HONEST causally-linked outcome "
                "(treatment_arm -> adopted via committed DGP, now WITH the planted channel "
                "term). arm_match must read 1.0000 (seed 427 reproduces the live arm); "
                "labels move only where the shifted sigmoid crosses the same uniform, so "
                "the 3 HCP gold-standard models need a one-time RE-TRAIN on the new labels. "
                "Aggregate prevalence stays in-band so the AUC ceiling is preserved."
            )

    if has_channels:
        logger.info(
            "--- PLANTED CHANNEL EFFECTS on adopted (joined rows; RD in probability points) ---"
        )
        logger.info(
            "  true_RD is the planted effect (DGP counterfactual, everything else held). naive_RD "
            "is the UNADJUSTED mean(adopted | tbin=1) - mean(adopted | tbin=0): the plant drives "
            "every channel from one latent (market share + log trigger volume + region), so each "
            "channel's naive contrast carries the others' effects and the null reads positive. "
            "The estimator adjusts for exactly that latent; compare true_RD to the recovery "
            "probe's ATE (scripts/verify_adoption_channel_recovery.py), not to naive_RD."
        )
        logger.info(
            "  %-14s %-28s %-9s %-9s %-9s", "brand", "channel", "naive_RD", "true_RD", "P(tbin=1)"
        )
        for brand in BRANDS:
            sub = derived[(derived["brand"] == brand) & derived["joined"]]
            r = report[brand]
            if not len(sub):
                logger.info("  %-14s (no joined rows)", brand)
                continue
            r["realised_rd_stratified"] = {
                k: round(v, 4)
                for k, v in stratified_channel_rd(sub["adopted"].to_numpy(), sub).items()
            }
            r["realised_rd_dgp_true"] = {
                k: round(v, 4)
                for k, v in dgp_true_channel_rd(sub["adoption_logit"].to_numpy(), sub).items()
            }
            r["null_channel_rd"] = {
                "channel": NULL_CHANNEL_COLUMN,
                "stratified": r["realised_rd_stratified"][NULL_CHANNEL_COLUMN],
                "dgp_true": r["realised_rd_dgp_true"][NULL_CHANNEL_COLUMN],
            }
            for col in CHANNEL_COLUMNS:
                logger.info(
                    "  %-14s %-28s %+9.4f %+9.4f %9.4f%s",
                    brand,
                    col,
                    r["realised_rd_stratified"][col],
                    r["realised_rd_dgp_true"][col],
                    float(sub[tbin_column(col)].mean()),
                    "   <- NULL" if col == NULL_CHANNEL_COLUMN else "",
                )
            cd = pd.to_datetime(sub["consideration_date"])
            mx = pd.to_datetime(sub["max_metric_date"])
            r["n_joined_dated_after_last_exposure"] = int((cd > mx).sum())
            r["n_future_dates"] = int((cd > pd.Timestamp(run_date)).sum())
            n_clamped = int((cd == pd.Timestamp(run_date)).sum())
            logger.info(
                "  %-14s consideration_date: joined rows dated after last exposure %d/%d "
                "(must equal), future dates %d (must be 0), clamped to run date %s: %d; "
                "non-joined rows (%d) keep their live dates; span %s..%s",
                brand,
                r["n_joined_dated_after_last_exposure"],
                r["n_joined"],
                r["n_future_dates"],
                run_date,
                n_clamped,
                r["n_non_joined"],
                cd.min().date(),
                cd.max().date(),
            )
    return report


def write_backup(live: pd.DataFrame, out_dir: Path) -> Path:
    """Write a TSV backup of the CURRENT live adopted labels before any write."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    path = out_dir / f"hcp_brand_adoption_label_backup_{ts}.tsv"
    cols = [c for c in ("hcp_id", "brand", "adopted", "adoption_category") if c in live.columns]
    live[cols].to_csv(path, sep="\t", index=False)
    logger.info("Wrote backup of %d live label rows to %s", len(live), path)
    return path


# ---------------------------------------------------------------------------
# Write path (only reachable with --execute)
# ---------------------------------------------------------------------------

_ADD_COLUMN_SQL = "ALTER TABLE hcp_brand_adoption ADD COLUMN IF NOT EXISTS treatment_arm INTEGER;"
# Postgres has no `ADD CONSTRAINT IF NOT EXISTS`; guard with a DO block so the
# DDL is idempotent (re-runnable) on --execute.
_ADD_CHECK_SQL = (
    "DO $$ BEGIN "
    "IF NOT EXISTS (SELECT 1 FROM pg_constraint "
    "WHERE conname = 'ck_hcp_brand_adoption_treatment_arm') THEN "
    "ALTER TABLE hcp_brand_adoption "
    "ADD CONSTRAINT ck_hcp_brand_adoption_treatment_arm "
    "CHECK (treatment_arm IS NULL OR treatment_arm IN (0, 1)); "
    "END IF; END $$;"
)


def ensure_schema(client: Any) -> None:
    """Additively add the nullable treatment_arm column (idempotent).

    Tries a SQL RPC if the project exposes one; otherwise logs the DDL for the
    human to run as a sibling migration. The column is nullable / no default so
    real (is_synthetic=false) rows are untouched.
    """
    for sql in (_ADD_COLUMN_SQL, _ADD_CHECK_SQL):
        try:
            client.postgrest.rpc("exec_sql", {"sql": sql}).execute()  # type: ignore[attr-defined]
            logger.info("Applied DDL: %s", sql)
        except Exception as e:  # pragma: no cover - depends on RPC availability
            logger.warning(
                "Could not apply DDL via RPC (%s). Run this as a sibling migration "
                "BEFORE --execute writes:\n    %s",
                e,
                sql,
            )


def write_rows(client: Any, derived: pd.DataFrame, *, batch_size: int = 500) -> int:
    """Idempotent per-row UPDATE keyed on (hcp_id, brand).

    Writes treatment_arm + adopted + adoption_category on every row, ``updated_at =
    now()`` on every row (the table has no trigger), and ``consideration_date`` ONLY
    for joined rows (non-joined rows keep their live date). data_split / is_synthetic /
    created_at are untouched.
    """
    written = 0
    has_dates = "consideration_date" in derived.columns and "joined" in derived.columns
    stamp = datetime.now(timezone.utc).isoformat()
    for rec in derived.to_dict(orient="records"):
        payload: Dict[str, Any] = {
            "treatment_arm": int(rec["treatment_arm"]),
            "adopted": int(rec["adopted"]),
            "adoption_category": str(rec["adoption_category"]),
            "updated_at": stamp,
        }
        if has_dates and bool(rec["joined"]) and pd.notna(rec["consideration_date"]):
            payload["consideration_date"] = (
                pd.Timestamp(rec["consideration_date"]).date().isoformat()
            )
        resp = (
            client.table(ADOPTION_TABLE)
            .update(payload)
            .eq("hcp_id", rec["hcp_id"])
            .eq("brand", rec["brand"])
            .eq("is_synthetic", True)
            .execute()
        )
        # PostgREST returns the updated rows (return=representation): exactly one per key,
        # else the count below would be of ATTEMPTS, not writes (codex r1).
        matched = len(getattr(resp, "data", None) or [])
        if matched != 1:
            raise RuntimeError(
                f"UPDATE matched {matched} rows for ({rec['hcp_id']}, {rec['brand']}); "
                f"expected exactly 1 -- aborting after {written} rows written"
            )
        written += 1
        if written % batch_size == 0:
            logger.info("  updated %d/%d rows", written, len(derived))
    return written


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None, *, client: Any = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="WRITE PATH: ALTER TABLE (add treatment_arm) then UPDATE 15k rows "
        "(treatment_arm + adopted + adoption_category + consideration_date for joined "
        "rows + updated_at). Omit (the default) for a read-only dry-run.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Re-derivation seed (default {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--backup-dir",
        type=str,
        default=str(_PROJECT_ROOT / "data" / "backups"),
        help="Directory for the pre-write TSV backup of live labels.",
    )
    parser.add_argument(
        "--run-date",
        type=date.fromisoformat,
        default=None,
        help="Clamp for joined rows' consideration_date (default: today).",
    )
    parser.add_argument(
        "--frame-out",
        type=str,
        default=None,
        help="Write the generated frame (with the collapsed exposure) to this parquet path; "
        "input for scripts/verify_adoption_channel_recovery.py --frame.",
    )
    args = parser.parse_args(argv)
    dry_run = not args.execute
    run_date: date = args.run_date or date.today()

    logger.info("=" * 72)
    logger.info(
        "hcp_brand_adoption treatment-arm + channel re-plant backfill  (%s)  seed=%d  run_date=%s",
        "DRY RUN" if dry_run else "EXECUTE",
        args.seed,
        run_date,
    )
    logger.info("  brand adoption scales: %s", _BRAND_ADOPT_SCALE)
    logger.info("=" * 72)

    if client is None:
        try:
            from src.memory.services.factories import get_supabase_client

            client = get_supabase_client()
        except Exception as e:
            logger.warning("No Supabase client (%s).", e)

    if client is None:
        logger.error(
            "This script reads the EXISTING hcp_profiles centrality to derive the "
            "treatment arm + label. Without a Supabase client there is nothing to "
            "derive from. Set SUPABASE_URL + a service key, or run from the prod box."
        )
        return 1

    centrality = fetch_centrality(client)
    if centrality is None or centrality.empty:
        logger.error("Live %s has no synthetic centrality to derive from.", PROFILES_TABLE)
        return 1
    logger.info("Read %d synthetic HCP centrality rows from %s.", len(centrality), PROFILES_TABLE)

    rollups = fetch_channel_rollups(client)
    if rollups.empty:
        logger.error(
            "Live %s has no synthetic %s rows for %s: nothing to plant the channel term from. "
            "Re-run scripts/backfill_segment_engagement.py first.",
            ROLLUP_TABLE,
            ROLLUP_METRIC_TYPE,
            BRANDS,
        )
        return 1
    logger.info(
        "Read %d planted %s rows over %d (hcp, brand) pairs from %s (metric_date %s..%s).",
        len(rollups),
        ROLLUP_METRIC_TYPE,
        int(rollups.groupby(KEY).ngroups),
        ROLLUP_TABLE,
        rollups["metric_date"].min().date(),
        rollups["metric_date"].max().date(),
    )

    live = fetch_live_adoption(client)
    if live is not None and len(live):
        # Always write a backup of the CURRENT live labels (cheap, safe, even dry-run).
        write_backup(live, Path(args.backup_dir))

    derived = derive(centrality, seed=args.seed, channel_rollups=rollups, run_date=run_date)

    # Every brand must carry a planted population (codex r1): a brand with no rollups would
    # silently get zero shifts and be written as if planted.
    joined_by_brand = derived.groupby("brand")["joined"].sum().to_dict()
    unplanted = [b for b in BRANDS if int(joined_by_brand.get(b, 0)) == 0]
    if unplanted:
        logger.error(
            "REFUSING to proceed: no planted %s rows join the cohort for %s (joined per brand: %s).",
            ROLLUP_METRIC_TYPE,
            unplanted,
            joined_by_brand,
        )
        return 1
    logger.info(
        "Derived %d rows (%d HCPs x %d brands): treatment_arm + adopted + channel term; "
        "joined %d / non-joined %d.",
        len(derived),
        derived["hcp_id"].nunique(),
        len(BRANDS),
        int(derived["joined"].sum()),
        int((~derived["joined"]).sum()),
    )

    report = verify(derived, live, run_date=run_date)
    logger.info(
        "DOCUMENTED PER-BRAND TRUE ATE of treatment_arm (prob scale): %s",
        {b: r["true_ate_arm"] for b, r in report.items()},
    )
    bad_arm = [b for b, r in report.items() if r["arm_match_vs_live"] not in (None, 1.0)]
    bad_dates = [
        b
        for b, r in report.items()
        if r["n_future_dates"] != 0 or r["n_joined_dated_after_last_exposure"] != r["n_joined"]
    ]
    if bad_arm or bad_dates:
        logger.error(
            "REFUSING to proceed: arm mismatch vs live in %s; date invariant broken in %s.",
            bad_arm,
            bad_dates,
        )
        return 1
    if not dry_run:
        # --execute preconditions (codex r1: the write path was fail-open when the live read
        # failed): a complete live snapshot whose keys are exactly the derived keys, backed
        # up above, with the arm reproduced 1.0000 in EVERY brand.
        if live is None or not len(live):
            logger.error(
                "REFUSING to --execute: the live %s snapshot could not be read.", ADOPTION_TABLE
            )
            return 1
        live_keys = set(zip(live["hcp_id"], live["brand"], strict=True))
        derived_keys = set(zip(derived["hcp_id"], derived["brand"], strict=True))
        if len(live_keys) != len(live) or live_keys != derived_keys:
            logger.error(
                "REFUSING to --execute: live keys (%d rows, %d unique) != derived keys (%d); "
                "%d derived keys absent from live, %d live keys not derived.",
                len(live),
                len(live_keys),
                len(derived_keys),
                len(derived_keys - live_keys),
                len(live_keys - derived_keys),
            )
            return 1
        unmatched = [b for b, r in report.items() if r["arm_match_vs_live"] != 1.0]
        if unmatched:
            logger.error(
                "REFUSING to --execute: treatment_arm not reproduced 1.0000 vs live in %s "
                "(%s) -- the seed does not reproduce the live arm; do not overwrite.",
                unmatched,
                {b: report[b]["arm_match_vs_live"] for b in unmatched},
            )
            return 1

    if args.frame_out:
        out = Path(args.frame_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        derived.to_parquet(out, index=False)
        logger.info("Wrote the generated frame (%d rows) to %s", len(derived), out)

    if dry_run:
        logger.info("--- WOULD WRITE (on --execute) ---")
        logger.info("  1. %s", _ADD_COLUMN_SQL)
        logger.info("  2. %s", _ADD_CHECK_SQL)
        logger.info(
            "  3. UPDATE %d rows: treatment_arm + adopted + adoption_category + updated_at "
            "(keyed on (hcp_id, brand)); consideration_date on the %d joined rows only; "
            "data_split / is_synthetic / created_at untouched.",
            len(derived),
            int(derived["joined"].sum()),
        )
        logger.info(
            "  4. THEN scripts/verify_adoption_channel_recovery.py --live (must PASS) and "
            "re-train the 3 hcp_adoption_*_goldstd_lr_v1 models (scripts/retrain_goldstd.sh)."
        )
        logger.info("DRY RUN complete. No schema/rows changed. Re-run with --execute to write.")
        return 0

    ensure_schema(client)
    n = write_rows(client, derived)
    logger.info(
        "EXECUTE complete: updated %d rows in %s (idempotent on (hcp_id, brand)).",
        n,
        ADOPTION_TABLE,
    )
    logger.info(
        "NEXT: python scripts/verify_adoption_channel_recovery.py --live (must PASS), then "
        "re-train the 3 hcp_adoption_*_goldstd_lr_v1 models on the new labels and re-verify "
        "AUC before promoting."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
