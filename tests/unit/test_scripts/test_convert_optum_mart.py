"""Tests for the Optum mart -> initiation cohort adapter (scripts/convert_optum_mart.py).

Verifies the leakage-safe cohort shaping against the owner-approved design
(.claude/plans/optum-initiation-adapter/IMPLEMENTATION-PLAN.md):
- naive-at-index gate (drop patients treated BEFORE index)
- target initiated_biologic_180d = treatment within [index, index+180d]
- transparent quality filter from concrete record-count signals
- journey records carry ONLY the 64-column pre-index allow-list + target + ids,
  with derived geographic_region / enrollment_duration_days and NO leakers.
"""

import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.convert_optum_mart import (  # noqa: E402
    build_journey_records,
    convert,
    select_initiation_cohort,
)
from src.data.manifests import MART_SAFE_FEATURES  # noqa: E402


def _patients() -> pd.DataFrame:
    idx = pd.Timestamp("2020-01-01")
    day = pd.Timedelta(days=1)
    return pd.DataFrame(
        [
            # untreated, good DQ -> kept, target 0
            {
                "patid": 1,
                "index_biologic_brand": "no_treatment",
                "treatment_start_date": pd.NaT,
                "index_date": idx,
                "claim_record_count": 10,
            },
            # treated within 180d -> kept, target 1
            {
                "patid": 2,
                "index_biologic_brand": "XOLAIR",
                "treatment_start_date": idx + 30 * day,
                "index_date": idx,
                "claim_record_count": 10,
            },
            # treated at >180d -> kept, target 0 (later initiator becomes a negative)
            {
                "patid": 3,
                "index_biologic_brand": "XOLAIR",
                "treatment_start_date": idx + 200 * day,
                "index_date": idx,
                "claim_record_count": 10,
            },
            # pre-index treated -> DROPPED (not naive at index)
            {
                "patid": 4,
                "index_biologic_brand": "XOLAIR",
                "treatment_start_date": idx - 10 * day,
                "index_date": idx,
                "claim_record_count": 10,
            },
            # low claim count -> DROPPED (transparent quality filter)
            {
                "patid": 5,
                "index_biologic_brand": "no_treatment",
                "treatment_start_date": pd.NaT,
                "index_date": idx,
                "claim_record_count": 1,
            },
        ]
    )


def test_select_initiation_cohort_naive_target_and_quality():
    cohort, attrition = select_initiation_cohort(_patients(), window_days=180, min_claim_count=2)
    assert set(cohort["patid"]) == {1, 2, 3}  # p4 pre-index, p5 low-claims dropped
    targets = dict(zip(cohort["patid"], cohort["initiated_biologic_180d"], strict=True))
    assert targets == {1: 0, 2: 1, 3: 0}
    assert cohort["initiated_biologic_180d"].dtype.kind in "iu"  # int target
    steps = dict(attrition)
    assert steps["input_patients"] == 5
    assert steps["naive_at_index"] == 4  # p4 removed
    assert steps["quality_filter"] == 3  # p5 removed


def test_select_initiation_cohort_gap_zero_is_positive():
    """Same-day initiation (gap==0) counts as a positive (>=0 boundary)."""
    idx = pd.Timestamp("2021-06-01")
    df = pd.DataFrame(
        [
            {
                "patid": 9,
                "index_biologic_brand": "DUPIXENT",
                "treatment_start_date": idx,
                "index_date": idx,
                "claim_record_count": 5,
            },
        ]
    )
    cohort, _ = select_initiation_cohort(df, window_days=180, min_claim_count=2)
    assert int(cohort.iloc[0]["initiated_biologic_180d"]) == 1


def test_build_journey_records_emits_allowlist_plus_target_no_leakers():
    idx = pd.Timestamp("2019-03-15")
    cohort = pd.DataFrame(
        [
            {
                "patid": 42,
                "index_date": idx,
                "elig_start_date": idx - pd.Timedelta(days=400),
                "zipcode_5": "10001",
                "age_at_index": 55.0,
                "gdr_cd": "M",
                "payer_category": "commercial",
                "charlson_score": 3,
                "cci_hiv": 0,
                "elx_depression": 1,
                "high_comorbidity_burden_flag": 0,
                "initiated_biologic_180d": 1,
                # leakers that MUST NOT survive into the journey record:
                "index_biologic_brand": "XOLAIR",
                "pdc": 0.8,
                "treatment_response": "controlled",
            }
        ]
    )
    records = build_journey_records(cohort)
    assert len(records) == 1
    rec = records[0]
    # identity + anchor
    assert rec["patient_id"] == "PAT_42"
    assert rec["patient_journey_id"] == "PJ_42"
    assert rec["journey_start_date"] == idx
    # derived pre-index features
    assert rec["geographic_region"] is not None  # 10001 -> Northeast
    assert rec["enrollment_duration_days"] == 400
    # target preserved
    assert rec["initiated_biologic_180d"] == 1
    # leakers dropped
    for leak in ("index_biologic_brand", "pdc", "treatment_response", "zipcode_5"):
        assert leak not in rec, f"{leak} leaked into journey record"
    # every non-id/non-target emitted feature is in the approved allow-list.
    # journey_status / discontinuation_flag are intentional QC/GE-contract metadata
    # (not model features — excluded by the optum_mart manifest).
    structural = {
        "patient_id",
        "patient_journey_id",
        "patient_hash",
        "journey_start_date",
        "index_date",
        "initiated_biologic_180d",
        "data_split",
        "journey_status",
        "discontinuation_flag",
        "data_quality_score",
    }
    for key in rec:
        if key in structural:
            continue
        assert key in MART_SAFE_FEATURES, f"emitted non-allowlisted feature: {key}"


def test_build_journey_records_emits_transparent_data_quality_score():
    """Each record carries a transparent completeness-based data_quality_score
    in [0,1]; a fully-populated row scores 1.0; the score is metadata, NOT a
    model feature (it is excluded from the manifest allow-list)."""
    idx = pd.Timestamp("2019-03-15")
    cohort = pd.DataFrame(
        [
            {
                "patid": 7,
                "index_date": idx,
                "elig_start_date": idx - pd.Timedelta(days=300),
                "zipcode_5": "10001",
                "age_at_index": 55.0,
                "gdr_cd": "M",
                "payer_category": "commercial",
                "charlson_score": 3,
                "cci_hiv": 0,
                "elx_depression": 1,
                "high_comorbidity_burden_flag": 0,
                "initiated_biologic_180d": 0,
            }
        ]
    )
    rec = build_journey_records(cohort)[0]
    assert "data_quality_score" in rec
    dqs = rec["data_quality_score"]
    assert 0.0 <= dqs <= 1.0
    assert dqs == 1.0  # every emitted model-input feature populated
    # transparent metadata, never a model feature
    assert "data_quality_score" not in MART_SAFE_FEATURES


def test_data_quality_score_reflects_missing_model_inputs():
    """A row missing the geographic + enrollment derivations scores < 1.0 —
    the score transparently IS the populated fraction of the model inputs."""
    idx = pd.Timestamp("2019-03-15")
    cohort = pd.DataFrame(
        [
            {
                "patid": 8,
                "index_date": idx,
                "elig_start_date": pd.NaT,  # -> enrollment None
                "zipcode_5": None,  # -> geographic_region None
                "age_at_index": 55.0,
                "gdr_cd": "M",
                "payer_category": "commercial",
                "charlson_score": 3,
                "cci_hiv": 0,
                "elx_depression": 1,
                "high_comorbidity_burden_flag": 0,
                "initiated_biologic_180d": 0,
            }
        ]
    )
    rec = build_journey_records(cohort)[0]
    assert rec["geographic_region"] is None
    assert rec["enrollment_duration_days"] is None
    assert rec["data_quality_score"] < 1.0


def test_convert_end_to_end_synthetic_fixture(tmp_path):
    """Full convert() against a tiny entity-stacked synthetic mart (CI has no real RWD)."""
    idx = pd.Timestamp("2020-01-01")
    day = pd.Timedelta(days=1)
    rows = [
        {
            "entity_type": "patient",
            "patid": 1,
            "index_biologic_brand": "no_treatment",
            "treatment_start_date": pd.NaT,
            "index_date": idx,
            "claim_record_count": 10,
            "elig_start_date": idx - 300 * day,
            "zipcode_5": "10001",
            "age_at_index": 50.0,
            "gdr_cd": "M",
            "payer_category": "commercial",
            "charlson_score": 2,
            "cci_hiv": 0,
            "pdc": None,
        },
        {
            "entity_type": "patient",
            "patid": 2,
            "index_biologic_brand": "XOLAIR",
            "treatment_start_date": idx + 30 * day,
            "index_date": idx,
            "claim_record_count": 8,
            "elig_start_date": idx - 365 * day,
            "zipcode_5": "90001",
            "age_at_index": 60.0,
            "gdr_cd": "F",
            "payer_category": "medicare",
            "charlson_score": 5,
            "cci_hiv": 0,
            "pdc": 0.7,
        },
        # a non-patient row that MUST be ignored by the entity filter
        {
            "entity_type": "optum_hcp",
            "patid": 999,
            "index_biologic_brand": None,
            "treatment_start_date": pd.NaT,
            "index_date": pd.NaT,
            "claim_record_count": None,
            "elig_start_date": pd.NaT,
            "zipcode_5": None,
            "age_at_index": None,
            "gdr_cd": None,
            "payer_category": None,
            "charlson_score": None,
            "cci_hiv": None,
            "pdc": None,
        },
    ]
    mart = tmp_path / "mart.parquet"
    pd.DataFrame(rows).to_parquet(mart)
    out = tmp_path / "out"

    summary = convert(input_path=str(mart), output_dir=str(out), window_days=180, min_claim_count=2)

    assert summary["patients"] == 2  # the optum_hcp row is excluded
    assert summary["positives"] == 1  # patient 2 initiated within 180d

    journeys = pd.read_parquet(out / "e2i_ml_v3_patient_journeys.parquet")
    assert set(journeys["patient_id"]) == {"PAT_1", "PAT_2"}
    assert "initiated_biologic_180d" in journeys.columns
    assert "data_split" in journeys.columns
    # leakage barrier: no forbidden columns survive
    for leak in ("pdc", "index_biologic_brand", "treatment_start_date", "zipcode_5"):
        assert leak not in journeys.columns
    # canonical files present
    for fname in (
        "e2i_ml_v3_patient_journeys.parquet",
        "e2i_ml_v3_split_registry.json",
        "attrition_report.csv",
        "data_dictionary.csv",
    ):
        assert (out / fname).exists()
