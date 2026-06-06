#!/usr/bin/env python
"""Convert the Optum MART into a tier-0-ready INITIATION cohort (split-and-map).

The new Optum drop ``data/rwd/Optum_Parquet/Optum.parquet`` is an entity-stacked,
pre-engineered mart (252 cols x 3.76M rows) — NOT the 6 raw claims files
``scripts/convert_optum_rwd.py`` consumes. Every feature is already a precomputed
aggregate, so this is a SPLIT-AND-MAP adapter: select the patient entity, shape
the leakage-safe initiation cohort, and emit the canonical tier-0 contract,
reusing the pure helpers in ``scripts/rwd_common.py``.

Leakage governance is positive-enumeration: only the owner-approved 64-column
pre-index allow-list (``src.data.manifests.MART_SAFE_FEATURES``) + the supervised
target survive into ``patient_journeys``. The mart leakage manifest
(``optum_mart`` source) grants declared-safe immunity to the sparse comorbidity
flags at tier-0 time.

Design: ``.claude/plans/optum-initiation-adapter/IMPLEMENTATION-PLAN.md``.

Usage (smoke on a stratified sample):
    python scripts/convert_optum_mart.py --sample-n 50000
    # then: python scripts/run_tier0_test.py --data-dir data/rwd/mart/initiation \
    #         --feature-manifest-source optum_mart --target-outcome initiated_biologic_180d
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.rwd_common import (  # noqa: E402
    apply_chronological_split,
    build_split_registry,
    map_zipcode_to_region,
    patient_hash,
    write_attrition_report,
    write_data_dictionary,
    write_records,
)
from src.data.manifests import MART_SAFE_FEATURES  # noqa: E402
from src.data.manifests.optum_mart_feature_manifest import optum_mart_contract_for  # noqa: E402

logger = logging.getLogger("convert_optum_mart")

TARGET = "initiated_biologic_180d"
DEFAULT_INPUT = "data/rwd/Optum_Parquet/Optum.parquet"
DEFAULT_OUTPUT = "data/rwd/mart/initiation"
_DERIVED = ("geographic_region", "enrollment_duration_days")
# Raw mart columns the cohort logic needs beyond the allow-list features.
_GATING_COLS = (
    "patid", "entity_type", "index_biologic_brand", "treatment_start_date",
    "index_date", "claim_record_count", "elig_start_date", "zipcode_5",
)


def select_initiation_cohort(
    df: pd.DataFrame, *, window_days: int = 180, min_claim_count: int = 2
) -> tuple[pd.DataFrame, list[tuple[str, int]]]:
    """Shape the leakage-safe initiation cohort from patient-entity mart rows.

    - naive-at-index: drop patients whose first biologic predates index.
    - transparent quality filter: claim_record_count >= ``min_claim_count``.
    - target ``initiated_biologic_180d`` = treated within [index, index+window].
    Returns ``(cohort_df, attrition_steps)`` where attrition is a list of
    ``(step, surviving_count)`` (plus a final target-positive count).
    """
    attrition: list[tuple[str, int]] = [("input_patients", len(df))]

    idx = pd.to_datetime(df["index_date"])
    ts = pd.to_datetime(df["treatment_start_date"])
    treated = df["index_biologic_brand"].ne("no_treatment") & ts.notna()
    gap = (ts - idx).dt.days
    pre_index_treated = treated & (gap < 0)
    df = df.loc[~pre_index_treated].copy()
    attrition.append(("naive_at_index", len(df)))

    keep = df["claim_record_count"].fillna(0) >= min_claim_count
    df = df.loc[keep].copy()
    attrition.append(("quality_filter", len(df)))

    idx = pd.to_datetime(df["index_date"])
    ts = pd.to_datetime(df["treatment_start_date"])
    treated = df["index_biologic_brand"].ne("no_treatment") & ts.notna()
    gap = (ts - idx).dt.days
    df[TARGET] = (treated & (gap >= 0) & (gap <= window_days)).astype("int64")
    attrition.append(("target_positives", int(df[TARGET].sum())))
    return df, attrition


def build_journey_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Map cohort rows to canonical journey record-dicts.

    Emits ONLY the pre-index allow-list (raw + derived geographic_region /
    enrollment_duration_days) + ids + target. Leakage columns present in the
    input are NOT carried through (positive enumeration).
    """
    raw_features = [
        c for c in MART_SAFE_FEATURES if c not in _DERIVED and c in df.columns
    ]
    records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        index_date = pd.to_datetime(row["index_date"])
        rec: dict[str, Any] = {
            "patient_journey_id": f"PJ_{row['patid']}",
            "patient_id": f"PAT_{row['patid']}",
            "patient_hash": patient_hash(row["patid"]),
            "index_date": index_date,
            "journey_start_date": index_date,
            # Journey metadata the data_preparer QC/GE contract expects (the
            # standard converter emits these on every cohort). ``ge_validator``
            # routes a patient frame carrying ``discontinuation_flag`` (and no
            # ``event_type``) to the no-event ``ml_patients`` GE suite; the QC
            # checker requires ``journey_status``. Neither is in
            # MART_SAFE_FEATURES, so the optum_mart manifest excludes them from
            # the model feature set (initiation does not model discontinuation).
            "journey_status": "active",
            "discontinuation_flag": 0,
            TARGET: int(row[TARGET]),
        }
        for col in raw_features:
            rec[col] = row[col]
        elig_raw = row.get("elig_start_date")
        elig = pd.to_datetime(elig_raw) if elig_raw is not None else pd.NaT
        rec["enrollment_duration_days"] = (
            int((index_date - elig).days) if pd.notna(elig) else None
        )
        zip5 = row.get("zipcode_5")
        rec["geographic_region"] = map_zipcode_to_region(zip5) if isinstance(zip5, str) else None
        # Transparent data-quality score: the populated fraction of THIS
        # patient's emitted model-input features (raw allow-list + the 2
        # derived columns). Honest completeness metadata — every existing
        # converter emits a ``data_quality_score`` and the harness cohort gate
        # filters on it. It is NOT a model feature: the manifest excludes it
        # (not in MART_SAFE_FEATURES) and the runner's exclude-set drops it.
        model_inputs = [rec[c] for c in raw_features]
        model_inputs.append(rec["enrollment_duration_days"])
        model_inputs.append(rec["geographic_region"])
        present = sum(1 for v in model_inputs if pd.notna(v))
        rec["data_quality_score"] = (
            round(present / len(model_inputs), 4) if model_inputs else 0.0
        )
        records.append(rec)
    return records


def _data_dictionary_entries() -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for name in sorted(MART_SAFE_FEATURES) + [TARGET]:
        contract = optum_mart_contract_for(name)
        ref = contract.knowable_at.reference if contract else "derived"
        entries.append(
            {
                "feature": name,
                "type": "target" if name == TARGET else "feature",
                "source_table": "optum_mart.patient",
                "lookback_window": ref,
                "null_rate": "",
                "notes": (
                    "supervised label (post-index)" if name == TARGET
                    else "pre-index admissible; data_quality_band is upstream-opaque, NOT used as a gate"
                ),
            }
        )
    return entries


def _read_patient_frame(input_path: str, sample_n: int | None) -> pd.DataFrame:
    """Read the patient entity with column projection (memory-frugal)."""
    import pyarrow.dataset as pads

    dset = pads.dataset(input_path, format="parquet")
    schema_names = set(dset.schema.names)
    projection = sorted(
        (set(_GATING_COLS) | {c for c in MART_SAFE_FEATURES if c not in _DERIVED}) & schema_names
    )
    flt = pads.field("entity_type") == "patient"
    if sample_n is not None:
        # Frugal path: read only the gating columns for all patients, decide the
        # eligible/sampled patids, then read the full projection for those only.
        gate = dset.to_table(columns=sorted(set(_GATING_COLS) & schema_names), filter=flt).to_pandas()
        cohort, _ = select_initiation_cohort(gate)
        if len(cohort) > sample_n:
            frac = sample_n / len(cohort)
            # stratified-by-target sample (preserves the positive rate)
            cohort = cohort.groupby(TARGET, group_keys=False).sample(
                frac=frac, random_state=42
            )
        keep_ids = cohort["patid"].tolist()
        sampled_flt = flt & pads.field("patid").isin(keep_ids)
        return dset.to_table(columns=projection, filter=sampled_flt).to_pandas()
    return dset.to_table(columns=projection, filter=flt).to_pandas()


def convert(
    *, input_path: str, output_dir: str, window_days: int = 180,
    min_claim_count: int = 2, sample_n: int | None = None,
) -> dict[str, Any]:
    """Run the full conversion: read -> shape -> split -> write canonical files."""
    df = _read_patient_frame(input_path, sample_n)
    cohort, attrition = select_initiation_cohort(
        df, window_days=window_days, min_claim_count=min_claim_count
    )
    records = build_journey_records(cohort)
    split = apply_chronological_split(records, date_key="journey_start_date", id_key="patient_id")

    out = Path(output_dir)
    write_records(out, "e2i_ml_v3_patient_journeys", records, fmt="parquet")
    registry = build_split_registry(
        split_config_id="optum_mart_initiation_v1",
        config_name="optum_mart_initiation",
        config_version="v1",
        split_dates=split["split_dates"],
    )
    write_records(out, "e2i_ml_v3_split_registry", registry, fmt="json")
    write_attrition_report(out, attrition)
    write_data_dictionary(out, _data_dictionary_entries())

    positives = int(sum(r[TARGET] for r in records))
    summary = {
        "patients": len(records),
        "positives": positives,
        "prevalence": round(positives / len(records), 4) if records else 0.0,
        "splits": split["counts"],
        "output_dir": str(out),
    }
    logger.info("Conversion summary: %s", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Optum mart -> initiation cohort adapter")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--target-window-days", type=int, default=180)
    parser.add_argument("--min-claim-count", type=int, default=2)
    parser.add_argument("--sample-n", type=int, default=None,
                        help="Stratified (by target) sample size for a smoke run.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    summary = convert(
        input_path=args.input, output_dir=args.output,
        window_days=args.target_window_days, min_claim_count=args.min_claim_count,
        sample_n=args.sample_n,
    )
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
