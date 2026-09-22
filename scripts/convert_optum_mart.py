#!/usr/bin/env python
"""Convert the Optum MART into a tier-0-ready INITIATION cohort (split-and-map).

Default input is the enriched Optum drop ``data/rwd/Optum_Parquet/Optum_enriched.parquet``
— a patient-grain, pre-engineered mart (205 cols x 814,587 rows, one row per
``patid``, all ``entity_type=patient``) — NOT the 6 raw claims files
``scripts/convert_optum_rwd.py`` consumes. (The legacy entity-stacked drop
``Optum.parquet`` (252 cols x 3.76M rows, patient + optum_hcp + veeva_hcp + market
grains) still works via ``--input``; this adapter selects the patient entity, so
both inputs yield the same 814,587-patient panel.) Every feature is already a
precomputed aggregate, so this is a SPLIT-AND-MAP adapter: select the patient
entity, shape the leakage-safe initiation cohort, and emit the canonical tier-0
contract, reusing the pure helpers in ``scripts/rwd_common.py``.

Leakage governance is positive-enumeration: only the owner-approved 64-column
pre-index allow-list (``src.data.manifests.MART_SAFE_FEATURES``) + the supervised
target survive into ``patient_journeys``. The mart leakage manifest
(``optum_mart`` source) grants declared-safe immunity to the sparse comorbidity
flags at tier-0 time.

Design: ``.claude/plans/optum-initiation-adapter/IMPLEMENTATION-PLAN.md``.

Usage (smoke on a stratified sample):
    python scripts/convert_optum_mart.py --cohort initiation --sample-n 50000
    # then run tier-0 via the Optum wrapper (sets target + manifest + AUC bar):
    #   python scripts/run_optum_tier0_test.py --cohort initiation_mart \
    #     --feature-manifest-source optum_mart --single-model
    python scripts/convert_optum_mart.py --cohort persistence_causal   # Lane A causal export (keeps the treatment)
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
TARGET_DISCONTINUED = "discontinued_180d"
TARGET_PERSISTENT = "persistent_at_180d"
# Coverage-gap thresholds for the treatment-anchored cohorts (legacy parity:
# BIOLOGIC_DISCONT_GAP_DAYS=90, BIOLOGIC_PERSISTENCE_GAP_DAYS=60 in
# convert_optum_rwd.py). Option B derives the 180d targets from the mart's
# aggregated coverage/gap columns; disc180 validated 98.2% vs discontinued_90d_flag.
DISCONT_GAP_DAYS = 90
PERSIST_GAP_DAYS = 60
# --- Lane A (spec 2026-09-22 §3A.1): the CAUSAL cohort ------------------------
# The persistence outcome is days-supply sensitive per brand (14-d Dupixent fills
# vs 28-45-d Xolair fills): the shipped ``persistent_at_180d`` gap of -17.6 pp
# collapses to 3.5 pp with a 14-day grace and INVERTS from 28 d on. The primary
# causal outcome is therefore covered-through-day-(180-28) AND no internal gap
# > 60 d — brand-invariant across 28-60 d grace. Owner decision, spec §7;
# evidence docs/demos/results/2026-09-22_persistence_definition_disproof/.
TARGET_PERSISTENT_G28 = "persistent_at_180d_g28"
PERSIST_GRACE_DAYS = 28
# Binary treatment: 1 = DUPIXENT, 0 = XOLAIR — the only contrast the drop
# observes (remibrutinib is absent; FDA approval 2025-09-30 is the drop's last
# observed day). Any other brand is EXCLUDED with an attrition step rather than
# silently coded as the reference arm.
TREATMENT_COL = "treatment_dupixent"
CAUSAL_ARMS = ("XOLAIR", "DUPIXENT")
# Precomputed switch flag carried from the raw drop (int32, both arms observed).
SWITCH_FLAG = "biologic_switch_180d_flag"
DEFAULT_INPUT = "data/rwd/Optum_Parquet/Optum_enriched.parquet"
DEFAULT_OUTPUT = "data/rwd/mart/initiation"
_DERIVED = ("geographic_region", "enrollment_duration_days")
# Raw mart columns the cohort logic needs beyond the allow-list features.
_GATING_COLS = (
    "patid",
    "entity_type",
    "index_biologic_brand",
    "treatment_start_date",
    "index_date",
    "claim_record_count",
    "elig_start_date",
    "zipcode_5",
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


def _initiator_eligible(
    df: pd.DataFrame, *, window_days: int, min_claim_count: int
) -> tuple[pd.DataFrame, list[tuple[str, int]]]:
    """Shared prelude for the TREATMENT-anchored cohorts (discontinuation/persistence).

    The temporal frame shifts vs initiation: the index is the first biologic fill
    (``treatment_start_date``), so the denominator is INITIATORS only and a 180d
    OUTCOME requires 180d of observed follow-up (initiators without it are
    right-censored — dropped). Returns ``(df, attrition)`` with steps
    input_patients/initiators/quality_filter/followup_observable.
    """
    attrition: list[tuple[str, int]] = [("input_patients", len(df))]

    ts = pd.to_datetime(df["treatment_start_date"])
    initiated = df["index_biologic_brand"].ne("no_treatment") & ts.notna()
    df = df.loc[initiated].copy()
    attrition.append(("initiators", len(df)))

    keep = df["claim_record_count"].fillna(0) >= min_claim_count
    df = df.loc[keep].copy()
    attrition.append(("quality_filter", len(df)))

    ts = pd.to_datetime(df["treatment_start_date"])
    lo = pd.to_datetime(df["last_observed_date"])
    observable = (lo - ts).dt.days >= window_days
    df = df.loc[observable].copy()
    attrition.append(("followup_observable", len(df)))

    # Require a recorded coverage end: the targets derive cov_to_end from
    # last_coverage_end, so a NaT here would silently collapse to disc=0/persist=0
    # (NaN < window and NaN >= window are both False). Right-censor those rows
    # with an explicit attrition step rather than fabricating a negative label.
    has_coverage_end = pd.to_datetime(df["last_coverage_end"]).notna()
    df = df.loc[has_coverage_end].copy()
    attrition.append(("coverage_end_observable", len(df)))
    return df, attrition


def select_discontinuation_cohort(
    df: pd.DataFrame, *, window_days: int = 180, min_claim_count: int = 2
) -> tuple[pd.DataFrame, list[tuple[str, int]]]:
    """Treatment-anchored discontinuation cohort (Option B, strict 90d gap).

    ``discontinued_180d`` = NOT covered through day ``window_days`` AND a coverage
    gap of at least ``DISCONT_GAP_DAYS`` (internal OR terminal). Denominator =
    observable initiators (see ``_initiator_eligible``).
    """
    df, attrition = _initiator_eligible(
        df, window_days=window_days, min_claim_count=min_claim_count
    )
    ts = pd.to_datetime(df["treatment_start_date"])
    lce = pd.to_datetime(df["last_coverage_end"])
    cov_to_end = (lce - ts).dt.days
    gap = df["max_internal_gap_days"].fillna(0)
    term = df["terminal_gap_days"].fillna(0)
    df[TARGET_DISCONTINUED] = (
        (cov_to_end < window_days) & ((gap >= DISCONT_GAP_DAYS) | (term >= DISCONT_GAP_DAYS))
    ).astype("int64")
    attrition.append(("target_positives", int(df[TARGET_DISCONTINUED].sum())))
    return df, attrition


def select_persistence_cohort(
    df: pd.DataFrame, *, window_days: int = 180, min_claim_count: int = 2
) -> tuple[pd.DataFrame, list[tuple[str, int]]]:
    """Treatment-anchored persistence cohort (Option B, strict).

    ``persistent_at_180d`` = covered THROUGH day ``window_days`` AND no internal
    coverage gap exceeding ``PERSIST_GAP_DAYS``. Denominator = observable
    initiators (see ``_initiator_eligible``).
    """
    df, attrition = _initiator_eligible(
        df, window_days=window_days, min_claim_count=min_claim_count
    )
    ts = pd.to_datetime(df["treatment_start_date"])
    lce = pd.to_datetime(df["last_coverage_end"])
    cov_to_end = (lce - ts).dt.days
    gap = df["max_internal_gap_days"].fillna(0)
    df[TARGET_PERSISTENT] = ((cov_to_end >= window_days) & (gap <= PERSIST_GAP_DAYS)).astype(
        "int64"
    )
    attrition.append(("target_positives", int(df[TARGET_PERSISTENT].sum())))
    return df, attrition


def select_persistence_causal_cohort(
    df: pd.DataFrame, *, window_days: int = 180, min_claim_count: int = 2
) -> tuple[pd.DataFrame, list[tuple[str, int]]]:
    """Treatment-anchored CAUSAL cohort: Dupixent vs Xolair initiators with four
    outcomes and the binary treatment (Lane A, spec 2026-09-22 §3A.1).

    Same denominator as the prediction persistence cohort (``_initiator_eligible``)
    restricted to the observed two-arm contrast. Emits:

    - ``treatment_dupixent``     1 = DUPIXENT, 0 = XOLAIR
    - ``persistent_at_180d_g28`` PRIMARY: covered through day window-28 AND no
                                 internal gap > PERSIST_GAP_DAYS (brand-invariant)
    - ``discontinued_180d``      secondary, as shipped (brand-robust)
    - ``biologic_switch_180d_flag`` as shipped in the raw drop (NULL -> 0)
    - ``persistent_at_180d``     the shipped definition, reported only alongside
                                 the days-supply sweep (a measurement artefact)
    """
    df, attrition = _initiator_eligible(
        df, window_days=window_days, min_claim_count=min_claim_count
    )
    if SWITCH_FLAG not in df.columns:
        raise KeyError(
            f"{SWITCH_FLAG} is not in the input frame; the causal cohort carries it as "
            "an outcome and will not fabricate zeros for a missing column"
        )
    in_contrast = df["index_biologic_brand"].isin(CAUSAL_ARMS)
    df = df.loc[in_contrast].copy()
    attrition.append(("two_arm_contrast", len(df)))

    ts = pd.to_datetime(df["treatment_start_date"])
    lce = pd.to_datetime(df["last_coverage_end"])
    cov_to_end = (lce - ts).dt.days
    gap = df["max_internal_gap_days"].fillna(0)
    term = df["terminal_gap_days"].fillna(0)

    df[TREATMENT_COL] = df["index_biologic_brand"].eq("DUPIXENT").astype("int64")
    df[TARGET_PERSISTENT] = ((cov_to_end >= window_days) & (gap <= PERSIST_GAP_DAYS)).astype(
        "int64"
    )
    df[TARGET_PERSISTENT_G28] = (
        (cov_to_end >= window_days - PERSIST_GRACE_DAYS) & (gap <= PERSIST_GAP_DAYS)
    ).astype("int64")
    df[TARGET_DISCONTINUED] = (
        (cov_to_end < window_days) & ((gap >= DISCONT_GAP_DAYS) | (term >= DISCONT_GAP_DAYS))
    ).astype("int64")
    df[SWITCH_FLAG] = df[SWITCH_FLAG].fillna(0).astype("int64")

    attrition.append(("target_positives", int(df[TARGET_PERSISTENT_G28].sum())))
    attrition.append(("arm_dupixent", int(df[TREATMENT_COL].sum())))
    return df, attrition


def build_journey_records(
    df: pd.DataFrame,
    *,
    target: str = TARGET,
    anchor_col: str = "index_date",
    extra_cols: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    """Map cohort rows to canonical journey record-dicts.

    Emits ONLY the pre-index allow-list (raw + derived geographic_region /
    enrollment_duration_days) + ids + the cohort ``target``. Leakage columns
    present in the input are NOT carried through (positive enumeration).

    ``anchor_col`` is the temporal index for the journey: ``index_date`` for the
    initiation cohort, ``treatment_start_date`` for the treatment-anchored
    discontinuation/persistence cohorts (the 64 baseline features are measured at
    the dx index, which is <= treatment-start, so they remain pre-index there).

    ``extra_cols`` (Lane A) is the CAUSAL cohort's positive enumeration of the
    columns the prediction cohorts must never carry: the treatment, its brand
    label, the treatment start and the secondary outcomes. Dates are emitted as
    Timestamps, text as-is, everything else as a plain int (0/1 flags). The
    default (empty) keeps every prediction cohort byte-identical.
    """
    raw_features = [c for c in MART_SAFE_FEATURES if c not in _DERIVED and c in df.columns]
    records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        index_date = pd.to_datetime(row[anchor_col])
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
            target: int(row[target]),
        }
        for col in raw_features:
            rec[col] = row[col]
        elig_raw = row.get("elig_start_date")
        elig = pd.to_datetime(elig_raw) if elig_raw is not None else pd.NaT
        rec["enrollment_duration_days"] = int((index_date - elig).days) if pd.notna(elig) else None
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
        rec["data_quality_score"] = round(present / len(model_inputs), 4) if model_inputs else 0.0
        for col in extra_cols:
            value = row[col]
            if col.endswith("_date"):
                rec[col] = pd.to_datetime(value)
            elif isinstance(value, str):
                rec[col] = value
            else:
                rec[col] = int(value)
        records.append(rec)
    return records


_CAUSAL_DICTIONARY_TYPE = {
    "index_biologic_brand": "treatment",
    TREATMENT_COL: "treatment",
    "treatment_start_date": "anchor",
    TARGET_DISCONTINUED: "outcome",
    SWITCH_FLAG: "outcome",
    TARGET_PERSISTENT: "outcome",
}


def _data_dictionary_entries(
    target: str = TARGET, *, extra_cols: tuple[str, ...] = ()
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for name in sorted(MART_SAFE_FEATURES) + [target]:
        contract = optum_mart_contract_for(name)
        ref = contract.knowable_at.reference if contract else "derived"
        entries.append(
            {
                "feature": name,
                "type": "target" if name == target else "feature",
                "source_table": "optum_mart.patient",
                "lookback_window": ref,
                "null_rate": "",
                "notes": (
                    "supervised label (post-index)"
                    if name == target
                    else "pre-index admissible; data_quality_band is upstream-opaque, NOT used as a gate"
                ),
            }
        )
    for name in extra_cols:
        entries.append(
            {
                "feature": name,
                "type": _CAUSAL_DICTIONARY_TYPE[name],
                "source_table": "optum_mart.patient",
                "lookback_window": "post-index (mart_treatment)",
                "null_rate": "",
                "notes": (
                    "CAUSAL export only — the treatment / its anchor / a secondary outcome; "
                    "never a prediction feature (manifest forbids it as such)"
                ),
            }
        )
    return entries


# --- Cohort registry (initiation + the treatment-anchored disc/persistence
# prediction cohorts + the Lane A causal cohort) ---
CAUSAL_COHORT = "persistence_causal"
PREDICTION_COHORTS = ("initiation", "discontinuation", "persistence")
# The causal export's positive enumeration of non-feature columns (order = the
# parquet / table column order after the allow-list features).
CAUSAL_EXTRA_COLS: tuple[str, ...] = (
    "index_biologic_brand",
    TREATMENT_COL,
    "treatment_start_date",
    TARGET_DISCONTINUED,
    SWITCH_FLAG,
    TARGET_PERSISTENT,
)
CAUSAL_RECORDS_NAME = "e2i_causal_v1_biologic_persistence"
COHORT_TARGETS: dict[str, str] = {
    "initiation": TARGET,
    "discontinuation": TARGET_DISCONTINUED,
    "persistence": TARGET_PERSISTENT,
    CAUSAL_COHORT: TARGET_PERSISTENT_G28,
}
_SELECTOR_BY_COHORT = {
    "initiation": select_initiation_cohort,
    "discontinuation": select_discontinuation_cohort,
    "persistence": select_persistence_cohort,
    CAUSAL_COHORT: select_persistence_causal_cohort,
}
# Journey temporal anchor: dx-index for initiation; first biologic fill for the
# treatment-anchored cohorts (the 64 baseline features are knowable at dx-index,
# which is <= treatment-start, so they remain pre-index in that frame).
_ANCHOR_BY_COHORT = {
    "initiation": "index_date",
    "discontinuation": "treatment_start_date",
    "persistence": "treatment_start_date",
    CAUSAL_COHORT: "treatment_start_date",
}
_SPLIT_CONFIG_BY_COHORT = {
    "initiation": ("optum_mart_initiation_v1", "optum_mart_initiation"),
    "discontinuation": ("optum_mart_discontinuation_v1", "optum_mart_discontinuation"),
    "persistence": ("optum_mart_persistence_v1", "optum_mart_persistence"),
    CAUSAL_COHORT: ("optum_mart_persistence_causal_v1", "optum_mart_persistence_causal"),
}
_OUTPUT_BY_COHORT = {
    "initiation": DEFAULT_OUTPUT,
    "discontinuation": "data/rwd/mart/discontinuation",
    "persistence": "data/rwd/mart/persistence",
    CAUSAL_COHORT: "data/rwd/mart/persistence_causal",
}
_TREATMENT_ANCHORED = ("discontinuation", "persistence", CAUSAL_COHORT)
_EXTRA_COLS_BY_COHORT: dict[str, tuple[str, ...]] = {CAUSAL_COHORT: CAUSAL_EXTRA_COLS}
_RECORDS_NAME_BY_COHORT: dict[str, str] = {CAUSAL_COHORT: CAUSAL_RECORDS_NAME}
# Coverage/gap columns the treatment-anchored cohorts need beyond the allow-list
# (the switch flag is projected for every treatment-anchored read; only the
# causal selector consumes it).
_OUTCOME_COLS = (
    "last_observed_date",
    "last_coverage_end",
    "max_internal_gap_days",
    "terminal_gap_days",
    SWITCH_FLAG,
)


def _count_patient_panel(input_path: str) -> int:
    """Cheap count of the full patient panel (entity_type == 'patient').

    The treatment-anchored reads push down to initiators, so this records the true
    funnel top (full patient denominator) in the attrition report for transparency
    without materializing the panel (count-only scan of one column).
    """
    import pyarrow.dataset as pads

    dset = pads.dataset(input_path, format="parquet")
    return int(dset.count_rows(filter=pads.field("entity_type") == "patient"))


def _read_patient_frame(
    input_path: str, *, cohort: str = "initiation", sample_n: int | None = None
) -> pd.DataFrame:
    """Read the patient entity with column projection (memory-frugal).

    The treatment-anchored cohorts (discontinuation/persistence) additionally
    project the coverage/gap columns and push down an initiators-only filter
    (``index_biologic_brand != 'no_treatment'``) so the read is ~24K rows, not
    ~814K. Each cohort reads independently (sequential reads bound peak memory —
    no need to hold all three frames at once).
    """
    import pyarrow.dataset as pads

    dset = pads.dataset(input_path, format="parquet")
    schema_names = set(dset.schema.names)
    gating = set(_GATING_COLS)
    if cohort in _TREATMENT_ANCHORED:
        gating |= set(_OUTCOME_COLS)
    projection = sorted(
        (gating | {c for c in MART_SAFE_FEATURES if c not in _DERIVED}) & schema_names
    )
    flt = pads.field("entity_type") == "patient"
    if cohort in _TREATMENT_ANCHORED:
        flt = flt & (pads.field("index_biologic_brand") != "no_treatment")
    selector = _SELECTOR_BY_COHORT[cohort]
    target = COHORT_TARGETS[cohort]
    if sample_n is not None:
        # Frugal path: read only the gating columns, decide the eligible/sampled
        # patids via the cohort's own selector, then read the full projection for
        # those only (stratified-by-target sample preserves the positive rate).
        gate = dset.to_table(columns=sorted(gating & schema_names), filter=flt).to_pandas()
        cohort_df, _ = selector(gate)
        if len(cohort_df) > sample_n:
            frac = sample_n / len(cohort_df)
            cohort_df = cohort_df.groupby(target, group_keys=False).sample(
                frac=frac, random_state=42
            )
        keep_ids = cohort_df["patid"].tolist()
        sampled_flt = flt & pads.field("patid").isin(keep_ids)
        return dset.to_table(columns=projection, filter=sampled_flt).to_pandas()
    return dset.to_table(columns=projection, filter=flt).to_pandas()


def convert(
    *,
    input_path: str,
    output_dir: str,
    cohort: str = "initiation",
    window_days: int = 180,
    min_claim_count: int = 2,
    sample_n: int | None = None,
) -> dict[str, Any]:
    """Run ONE cohort's conversion: read -> shape -> split -> write canonical files."""
    if cohort not in COHORT_TARGETS:
        raise ValueError(f"unknown cohort {cohort!r}; expected one of {sorted(COHORT_TARGETS)}")
    target = COHORT_TARGETS[cohort]
    selector = _SELECTOR_BY_COHORT[cohort]
    anchor = _ANCHOR_BY_COHORT[cohort]

    df = _read_patient_frame(input_path, cohort=cohort, sample_n=sample_n)
    cohort_df, attrition = selector(df, window_days=window_days, min_claim_count=min_claim_count)
    if cohort in _TREATMENT_ANCHORED and sample_n is None:
        # The read pushed down to initiators; record the full patient denominator
        # as the funnel top so the attrition report stays transparent.
        attrition = [("patient_panel", _count_patient_panel(input_path))] + attrition
    extra_cols = _EXTRA_COLS_BY_COHORT.get(cohort, ())
    records = build_journey_records(
        cohort_df, target=target, anchor_col=anchor, extra_cols=extra_cols
    )
    if cohort == CAUSAL_COHORT:
        # Real rows, tagged so the real-mode provenance filter keeps them and a
        # synthetic-gold plant can never masquerade as claims data.
        for rec in records:
            rec["is_synthetic"] = False
    split = apply_chronological_split(records, date_key="journey_start_date", id_key="patient_id")

    out = Path(output_dir)
    write_records(
        out,
        _RECORDS_NAME_BY_COHORT.get(cohort, "e2i_ml_v3_patient_journeys"),
        records,
        fmt="parquet",
    )
    cfg_id, cfg_name = _SPLIT_CONFIG_BY_COHORT[cohort]
    registry = build_split_registry(
        split_config_id=cfg_id,
        config_name=cfg_name,
        config_version="v1",
        split_dates=split["split_dates"],
    )
    write_records(out, "e2i_ml_v3_split_registry", registry, fmt="json")
    write_attrition_report(out, attrition)
    write_data_dictionary(out, _data_dictionary_entries(target, extra_cols=extra_cols))

    positives = int(sum(r[target] for r in records))
    summary: dict[str, Any] = {
        "cohort": cohort,
        "patients": len(records),
        "positives": positives,
        "prevalence": round(positives / len(records), 4) if records else 0.0,
        "splits": split["counts"],
        "output_dir": str(out),
    }
    if cohort == CAUSAL_COHORT:
        arms: dict[str, int] = {}
        for rec in records:
            arms[rec["index_biologic_brand"]] = arms.get(rec["index_biologic_brand"], 0) + 1
        summary["arms"] = arms
    logger.info("Conversion summary: %s", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Optum mart -> tier-0 cohort adapter")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument(
        "--cohort",
        default="initiation",
        choices=(*PREDICTION_COHORTS, CAUSAL_COHORT, "all"),
        help=(
            "Which cohort to build. 'all' builds the three PREDICTION cohorts; the "
            f"causal cohort '{CAUSAL_COHORT}' (keeps the treatment) is always explicit."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output dir. Single cohort: the exact dir (default "
            "data/rwd/mart/<cohort>). With --cohort all: a BASE dir whose "
            "per-cohort subdirs are written (default data/rwd/mart)."
        ),
    )
    parser.add_argument("--target-window-days", type=int, default=180)
    parser.add_argument("--min-claim-count", type=int, default=2)
    parser.add_argument(
        "--sample-n",
        type=int,
        default=None,
        help="Stratified (by target) sample size for a smoke run.",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    cohorts = list(PREDICTION_COHORTS) if args.cohort == "all" else [args.cohort]
    for cohort in cohorts:
        if args.cohort == "all":
            output_dir = str(Path(args.output or "data/rwd/mart") / cohort)
        else:
            output_dir = args.output or _OUTPUT_BY_COHORT[cohort]
        summary = convert(
            input_path=args.input,
            output_dir=output_dir,
            cohort=cohort,
            window_days=args.target_window_days,
            min_claim_count=args.min_claim_count,
            sample_n=args.sample_n,
        )
        print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
