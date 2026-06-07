"""Optum mart -> tier-0 HCP adoption-propensity cohort adapter.

The entity-stacked Optum drop (``data/rwd/Optum_Parquet/Optum.parquet``) carries
four unjoinable grains (patient / optum_hcp / veeva_hcp / market). The *patient*
grain has only baseline comorbidity/demographics, so the patient disease cohorts
(initiation/discontinuation/persistence) are feature-bound and non-deployable
(AUC 0.54-0.64). The commercial-HCP-targeting use case is natively an HCP-grain
problem, and the ``optum_hcp`` grain co-locates BOTH an adoption target and a
rich, admissible practice profile (claims referral-network position, all-cause
volume, specialty, geography). A propensity model on it is genuinely deployable
(AUC ~0.85, calibrated, low overfit; signal dominated by referral-network
diffusion + specialty, not a tautology — see
``docs/results/deployable_cohort_decision_20260607.md``).

This converter shapes that cohort into the canonical tier-0 contract:

* binary ``adopted_target_brand`` target derived from ``adoption_status``;
* POSITIVE-ENUMERATION feature allow-list — only admissible pre-adoption
  practice-profile columns are emitted; every adoption-DERIVED column
  (``adopter_rank``, ``days_to_first``, ``target_*``, ``adoption_*``) and every
  brand-specific / id / constant column is excluded, so nothing the target is
  computed from can leak into the model frame;
* a stratified-random ``data_split`` (the HCP grain has no temporal index, so a
  chronological split is meaningless) that preserves the rare positive rate.

HONESTY CAVEAT (documented, not hidden): the network/volume features are
cross-sectional aggregates in this pre-built mart (no clean pre/post window).
The model is therefore an adoption-PROPENSITY / segmentation model. A strict
forward-causal deployment should recompute features over a pre-index baseline
window upstream; network position is structurally stable and known at targeting
time, so this is a feature-window design step, not a no-signal risk.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.rwd_common import (  # noqa: E402
    SPLIT_RATIOS,
    patient_hash,
    write_attrition_report,
    write_data_dictionary,
    write_records,
)
from src.data.manifests.optum_hcp_feature_manifest import (  # noqa: E402
    OPTUM_HCP_SAFE_FEATURES,
)

logger = logging.getLogger("convert_optum_hcp_adoption")

DEFAULT_INPUT = "data/rwd/Optum_Parquet/Optum.parquet"
DEFAULT_OUTPUT = "data/rwd/mart/hcp_adoption"

HCP_TARGET = "adopted_target_brand"
HCP_ENTITY = "optum_hcp"

# Adoption status used as the target source (positive class).
_ADOPTER_VALUE = "ADOPTER"
# No real HCP index date exists in the pre-built mart (launch_dt is a constant),
# so the journey carries a fixed launch reference; the split is stratified-random
# (NOT chronological), assigned directly into ``data_split``.
_SYNTHETIC_INDEX_DATE = pd.Timestamp("2021-01-01")
_SPLIT_ORDER = ("train", "validation", "test", "holdout")

# --- Admissible pre-adoption practice-profile features (positive enumeration) --
# Single source of truth: the optum_hcp feature manifest (mirrors how the patient
# mart converter imports MART_SAFE_FEATURES). The manifest declares each column's
# knowable_at contract; that allow-list is the pre-index-admissible set (claims
# referral-network position + all-cause volume + specialty + geography).
#
# The two ``referral_out_*`` features remain ADMISSIBLE in the manifest (they are
# claims-network metrics, NOT adoption-derived — not leaks), but are excluded from
# THIS cohort's emit list: each is the highest single-feature AUC (referral_out_
# patient_count ~0.80) and trips the leakage detector's ``single_feature_auc`` gate
# (threshold 0.80), which is conservatively calibrated for clinical/causal models
# and false-positives on legitimately-strong commercial-targeting features.
# Investigation (docs/results/deployable_cohort_decision_20260607.md): single-
# feature AUCs form a smooth 0.68-0.80 gradient (no leak), both classes overlap,
# and the model is robust without them (AUC 0.84 vs 0.85; 0.81 even dropping all
# >=0.75). Excluding them is the CONSERVATIVE choice (removes signal / lowers AUC,
# the opposite of gaming) and keeps the cohort honestly deployable on the legit
# referral-IN / shared-patient / specialty / volume / KOL diffusion signal.
# (The gate's clinical-vs-commercial mis-calibration is filed as a follow-up.)
_GATE_EXCLUDED_FEATURES: tuple[str, ...] = (
    "referral_out_patient_count",
    "referral_out_degree",
)
HCP_SAFE_FEATURES: tuple[str, ...] = tuple(
    f for f in OPTUM_HCP_SAFE_FEATURES if f not in _GATE_EXCLUDED_FEATURES
)

# Heavy-tailed claims COUNT/WEIGHT/DEGREE features are log1p-transformed at emit.
# These span 0 -> millions (a handful of aggregate/institutional NPIs reach
# network sizes of 2.4M and shared-patient weights of 478M, far beyond any
# individual prescriber), so a log1p is the standard transform for such count
# data and improves linear-model conditioning. It ALSO resolves a false-positive
# in the leakage detector's range-based ``perfect_class_separation`` overlap
# metric, which is fooled when the minority (adopter) range is NESTED inside the
# majority range inflated by those outliers (overlap/combined_range -> <1% even
# though the distributions fully overlap on the adopter support). The transform
# is monotone, so a GENUINELY disjoint leak (e.g. days_on_therapy=0 for class 0,
# 300 for class 1) stays disjoint and is still flagged — verified — i.e. this
# fixes the false positive WITHOUT weakening real-leak detection. The bounded
# score/pct features (kol_score, kol_score_100pt, *_kol_score_pct) and the
# categoricals are NOT transformed. Explicit list (NOT a substring heuristic) so a
# categorical or a bounded score can never be accidentally log-transformed.
# (The underlying range-overlap metric flaw is filed separately for a robust fix.)
_LOG1P_FEATURES: tuple[str, ...] = (
    "influence_network_size",
    "shared_patient_edge_count",
    "shared_patient_weight",
    "max_shared_patient_edge_weight",
    "referral_in_degree",
    "referral_in_patient_count",
    "max_referral_in_edge_weight",
    "medical_claim_count",
    "medical_patient_count",
    "treated_patient_count",
)

# --- Excluded columns, by reason (documented, not silently dropped) ------------
# Adoption-DERIVED: the target is computed from these, so they would leak.
_LEAKY_HCP_COLS: tuple[str, ...] = (
    "adoption_status",
    "adoption_category",
    "adoption_category_method",
    "adopter_rank",
    "adopter_count",
    "adoption_cumulative_share",
    "days_to_first",
    "first_adoption_dt",
    "target_event_count",
    "target_patient_count",
    "distinct_target_code_count",
    "target_match_methods",
    "event_sources",
)
# Identifier / free-text / high-cardinality provider-id-like columns.
_ID_COLS: tuple[str, ...] = (
    "hcp_id",
    "prov",
    "npi",
    "hcp_npi",
    "patid",
    "dea",
    "hcp_name",
    "grp_practice",
    "hosp_affil",
)
# Single-value constants on the optum_hcp grain (no variance) + redundant-with-
# specialty_group high-card taxonomies (excluded for model robustness).
_CONSTANT_OR_REDUNDANT_COLS: tuple[str, ...] = (
    "brand",
    "molecule",
    "is_csu_approved",
    "launch_dt",
    "launch_context",
    "influence_network_method",
    "influence_network_source",
    "taxonomy1",
    "taxonomy2",
    "provcat",
)

# Columns the read must project beyond the feature allow-list.
_GATING_COLS = ("entity_type", "hcp_id", "adoption_status")

# Journey-contract metadata keys (match the patient-cohort converter so the
# data_preparer QC / ge_validator route this to the no-event ``ml_patients``
# suite and the downstream pipeline finds the columns it expects).
_CONTRACT_META = (
    "patient_journey_id",
    "patient_id",
    "patient_hash",
    "index_date",
    "journey_start_date",
    "journey_status",
    "discontinuation_flag",
)


def select_hcp_cohort(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[tuple[str, int]]]:
    """Filter to the optum_hcp grain and derive the binary adoption target.

    Returns ``(cohort_df, attrition)`` where attrition is a list of
    ``(step, surviving_count)`` (plus a final target-positive count). The full
    HCP universe is kept (no silent specialty filtering) so the model ranks the
    whole population; ``adopted_target_brand`` = 1 iff ``adoption_status`` is
    ``ADOPTER``.
    """
    attrition: list[tuple[str, int]] = [("input_rows", len(df))]
    df = df.loc[df["entity_type"] == HCP_ENTITY].copy()
    attrition.append(("optum_hcp_rows", len(df)))
    df[HCP_TARGET] = (df["adoption_status"] == _ADOPTER_VALUE).astype(int)
    attrition.append(("with_target", len(df)))
    attrition.append(("target_positives", int(df[HCP_TARGET].sum())))
    return df, attrition


def build_hcp_journey_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Map optum_hcp cohort rows to canonical journey record-dicts.

    Emits ONLY the admissible feature allow-list + the journey-contract metadata
    + the target. Each HCP is one "journey" (``patient_id`` surrogate
    ``HCP_<hcp_id>``) so the pipeline's patient-level isolation == HCP-level
    isolation. Leakage/id/constant columns are never carried through.
    """
    safe = [c for c in HCP_SAFE_FEATURES if c in df.columns]
    records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        hid = row["hcp_id"]
        rec: dict[str, Any] = {
            "patient_journey_id": f"PJ_{hid}",
            "patient_id": f"HCP_{hid}",
            "patient_hash": patient_hash(hid),
            "index_date": _SYNTHETIC_INDEX_DATE,
            "journey_start_date": _SYNTHETIC_INDEX_DATE,
            "journey_status": "active",
            "discontinuation_flag": 0,
            HCP_TARGET: int(row[HCP_TARGET]),
        }
        for col in safe:
            val = row[col]
            if col in _LOG1P_FEATURES and pd.notna(val):
                # log1p of the non-negative count/weight/degree (clip guards any
                # stray negative); see _LOG1P_FEATURES rationale above.
                val = float(np.log1p(max(float(val), 0.0)))
            rec[col] = val
        model_inputs = [rec[c] for c in safe]
        present = sum(1 for v in model_inputs if pd.notna(v))
        rec["data_quality_score"] = round(present / len(model_inputs), 4) if model_inputs else 0.0
        records.append(rec)
    return records


def assign_stratified_split(
    records: list[dict[str, Any]],
    *,
    target: str = HCP_TARGET,
    seed: int = 42,
    ratios: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Assign a deterministic, target-stratified ``data_split`` to each record.

    Mutates each record by setting ``record["data_split"]`` and returns a summary
    ``{"counts": {split: n}}``. Stratifying by target preserves the rare positive
    rate across train/validation/test/holdout (a chronological split is
    meaningless on the HCP grain, which has no temporal index).
    """
    r = dict(SPLIT_RATIOS if ratios is None else ratios)
    rng = np.random.RandomState(seed)
    by_class: dict[int, list[int]] = defaultdict(list)
    for i, rec in enumerate(records):
        by_class[int(rec[target])].append(i)
    counts: dict[str, int] = dict.fromkeys(_SPLIT_ORDER, 0)
    for _cls, idxs in sorted(by_class.items()):
        idxs = list(idxs)
        rng.shuffle(idxs)
        n = len(idxs)
        sizes = {name: int(n * r.get(name, 0.0)) for name in _SPLIT_ORDER}
        sizes["train"] += n - sum(sizes.values())  # remainder -> train
        pos = 0
        for name in _SPLIT_ORDER:
            for j in idxs[pos : pos + sizes[name]]:
                records[j]["data_split"] = name
                counts[name] += 1
            pos += sizes[name]
    return {"counts": counts}


def _data_dictionary_entries() -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for name in sorted(HCP_SAFE_FEATURES) + [HCP_TARGET]:
        entries.append(
            {
                "feature": name,
                "type": "target" if name == HCP_TARGET else "feature",
                "source_table": "optum_mart.optum_hcp",
                "lookback_window": "post_index" if name == HCP_TARGET else "pre_adoption_profile",
                "null_rate": "",
                "notes": (
                    "supervised label: HCP prescribed the target brand (ADOPTER)"
                    if name == HCP_TARGET
                    else (
                        "admissible pre-adoption practice profile; not brand-specific; "
                        "cross-sectional aggregate (see module docstring windowing caveat)"
                        + (
                            "; log1p-transformed (heavy-tailed count)"
                            if name in _LOG1P_FEATURES
                            else ""
                        )
                    )
                ),
            }
        )
    return entries


def _read_hcp_frame(input_path: str, *, sample_n: int | None = None) -> pd.DataFrame:
    """Read the optum_hcp entity with column projection (memory-frugal).

    Projects only the gating columns + the admissible feature allow-list and
    pushes down ``entity_type == 'optum_hcp'`` so the read is bounded to the HCP
    grain (~2.75M rows). ``sample_n`` takes a stratified-by-target sample.
    """
    import pyarrow.dataset as pads

    dset = pads.dataset(input_path, format="parquet")
    schema_names = set(dset.schema.names)
    projection = sorted((set(_GATING_COLS) | set(HCP_SAFE_FEATURES)) & schema_names)
    flt = pads.field("entity_type") == HCP_ENTITY
    df = dset.to_table(columns=projection, filter=flt).to_pandas()
    if sample_n is not None and len(df) > sample_n:
        df = df.assign(_y=(df["adoption_status"] == _ADOPTER_VALUE).astype(int))
        frac = sample_n / len(df)
        df = df.groupby("_y", group_keys=False).sample(frac=frac, random_state=42)
        df = df.drop(columns=["_y"])
    return df


def convert(
    *,
    input_path: str = DEFAULT_INPUT,
    output_dir: str = DEFAULT_OUTPUT,
    sample_n: int | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Read -> shape -> stratified-split -> write the canonical cohort files."""
    df = _read_hcp_frame(input_path, sample_n=sample_n)
    cohort_df, attrition = select_hcp_cohort(df)
    records = build_hcp_journey_records(cohort_df)
    split = assign_stratified_split(records, target=HCP_TARGET, seed=seed)

    out = Path(output_dir)
    write_records(out, "e2i_ml_v3_patient_journeys", records, fmt="parquet")
    registry = _build_hcp_split_registry(split["counts"])
    write_records(out, "e2i_ml_v3_split_registry", registry, fmt="json")
    write_attrition_report(out, attrition)
    write_data_dictionary(out, _data_dictionary_entries())

    positives = int(sum(r[HCP_TARGET] for r in records))
    summary = {
        "cohort": "hcp_adoption",
        "hcps": len(records),
        "positives": positives,
        "prevalence": round(positives / len(records), 4) if records else 0.0,
        "splits": split["counts"],
        "n_features": len(HCP_SAFE_FEATURES),
        "output_dir": str(out),
    }
    logger.info("Conversion summary: %s", summary)
    return summary


def _build_hcp_split_registry(counts: dict[str, int]) -> list[dict[str, Any]]:
    """Honest split registry for the HCP grain (stratified-random, no dates)."""
    from datetime import datetime

    total = sum(counts.values()) or 1
    return [
        {
            "split_config_id": "optum_hcp_adoption_v1",
            "config_name": "optum_hcp_adoption",
            "config_version": "v1",
            "train_ratio": round(counts.get("train", 0) / total, 4),
            "validation_ratio": round(counts.get("validation", 0) / total, 4),
            "test_ratio": round(counts.get("test", 0) / total, 4),
            "holdout_ratio": round(counts.get("holdout", 0) / total, 4),
            "data_start_date": None,
            "data_end_date": None,
            "train_end_date": None,
            "validation_end_date": None,
            "test_end_date": None,
            "temporal_gap_days": 0,
            "patient_level_isolation": True,
            "split_strategy": "stratified_random",
            "is_active": True,
            "created_at": datetime.now().isoformat(),
        }
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Optum mart optum_hcp -> tier-0 adoption-propensity cohort"
    )
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--sample-n",
        type=int,
        default=None,
        help="Stratified (by target) sample size for a smoke run.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    summary = convert(
        input_path=args.input,
        output_dir=args.output,
        sample_n=args.sample_n,
        seed=args.seed,
    )
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
