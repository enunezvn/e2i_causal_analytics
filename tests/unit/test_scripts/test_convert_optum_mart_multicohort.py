"""Tests for the multi-cohort (discontinuation/persistence) Optum mart adapter.

Option B (validated 2026-06-06 on the source mart): derive TRUE 180d targets from
the mart's aggregated coverage/gap columns rather than the precomputed 60/90d
flags. disc180 agrees 98.2% with the precomputed discontinued_90d_flag.

Shared cohort frame for discontinuation & persistence:
- Denominator = INITIATORS only (index_biologic_brand != 'no_treatment').
- Index re-anchored to treatment_start_date (the first biologic fill).
- Require >=180d follow-up (last_observed_date - treatment_start >= window);
  right-censor (drop) initiators without it.
- Transparent quality filter: claim_record_count >= min_claim_count.

Strict definitions:
- discontinued_180d  = NOT covered to day 180 AND (max_internal_gap >= 90 OR terminal_gap >= 90)
- persistent_at_180d = covered through day 180 AND max_internal_gap <= 60
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
    select_discontinuation_cohort,
    select_persistence_cohort,
)
from scripts.convert_optum_mart import main as convert_main  # noqa: E402


def _initiators() -> pd.DataFrame:
    ts = pd.Timestamp("2020-01-01")
    day = pd.Timedelta(days=1)
    return pd.DataFrame(
        [
            # p1: untreated -> dropped (not an initiator)
            {
                "patid": 1,
                "index_biologic_brand": "no_treatment",
                "treatment_start_date": pd.NaT,
                "claim_record_count": 10,
                "last_observed_date": ts + 400 * day,
                "last_coverage_end": pd.NaT,
                "max_internal_gap_days": 0,
                "terminal_gap_days": 0,
            },
            # p2: initiator, low claims -> dropped (quality filter)
            {
                "patid": 2,
                "index_biologic_brand": "XOLAIR",
                "treatment_start_date": ts,
                "claim_record_count": 1,
                "last_observed_date": ts + 400 * day,
                "last_coverage_end": ts + 300 * day,
                "max_internal_gap_days": 0,
                "terminal_gap_days": 0,
            },
            # p3: initiator, <180d follow-up -> dropped (right-censored, not observable)
            {
                "patid": 3,
                "index_biologic_brand": "XOLAIR",
                "treatment_start_date": ts,
                "claim_record_count": 10,
                "last_observed_date": ts + 100 * day,
                "last_coverage_end": ts + 90 * day,
                "max_internal_gap_days": 0,
                "terminal_gap_days": 10,
            },
            # p4: observable, covered through day 180, small gap -> disc 0 / persist 1
            {
                "patid": 4,
                "index_biologic_brand": "XOLAIR",
                "treatment_start_date": ts,
                "claim_record_count": 10,
                "last_observed_date": ts + 400 * day,
                "last_coverage_end": ts + 220 * day,
                "max_internal_gap_days": 30,
                "terminal_gap_days": 0,
            },
            # p5: observable, coverage ends early + 90d+ internal gap -> disc 1 / persist 0
            {
                "patid": 5,
                "index_biologic_brand": "DUPIXENT",
                "treatment_start_date": ts,
                "claim_record_count": 10,
                "last_observed_date": ts + 400 * day,
                "last_coverage_end": ts + 100 * day,
                "max_internal_gap_days": 120,
                "terminal_gap_days": 0,
            },
            # p6: observable, covered to 180 but a big (>60) internal gap -> disc 0 / persist 0
            {
                "patid": 6,
                "index_biologic_brand": "XOLAIR",
                "treatment_start_date": ts,
                "claim_record_count": 10,
                "last_observed_date": ts + 400 * day,
                "last_coverage_end": ts + 220 * day,
                "max_internal_gap_days": 90,
                "terminal_gap_days": 0,
            },
        ]
    )


def test_select_discontinuation_cohort_strict_90d_gap():
    cohort, attrition = select_discontinuation_cohort(
        _initiators(), window_days=180, min_claim_count=2
    )
    steps = dict(attrition)
    assert steps["input_patients"] == 6
    assert steps["initiators"] == 5  # p1 untreated dropped
    assert steps["quality_filter"] == 4  # p2 low-claims dropped
    assert steps["followup_observable"] == 3  # p3 (<180d) right-censored
    assert set(cohort["patid"]) == {4, 5, 6}
    targets = dict(zip(cohort["patid"], cohort["discontinued_180d"], strict=True))
    assert targets == {4: 0, 5: 1, 6: 0}
    assert cohort["discontinued_180d"].dtype.kind in "iu"
    assert steps["target_positives"] == 1


def test_select_persistence_cohort_strict_coverage_and_gap():
    cohort, attrition = select_persistence_cohort(_initiators(), window_days=180, min_claim_count=2)
    steps = dict(attrition)
    assert set(cohort["patid"]) == {4, 5, 6}
    targets = dict(zip(cohort["patid"], cohort["persistent_at_180d"], strict=True))
    assert targets == {4: 1, 5: 0, 6: 0}
    assert cohort["persistent_at_180d"].dtype.kind in "iu"
    assert steps["target_positives"] == 1


def test_discontinuation_denominator_excludes_untreated_only_initiators():
    """The disc/persistence denominator is initiators (treated), never the full panel."""
    cohort, _ = select_discontinuation_cohort(_initiators(), window_days=180, min_claim_count=2)
    assert 1 not in set(cohort["patid"])  # untreated never enters the cohort


def test_followup_eligibility_boundary_is_inclusive():
    """Exactly window_days of follow-up is observable (>= boundary)."""
    ts = pd.Timestamp("2021-01-01")
    day = pd.Timedelta(days=1)
    df = pd.DataFrame(
        [
            {
                "patid": 10,
                "index_biologic_brand": "XOLAIR",
                "treatment_start_date": ts,
                "claim_record_count": 5,
                "last_observed_date": ts + 180 * day,
                "last_coverage_end": ts + 180 * day,
                "max_internal_gap_days": 0,
                "terminal_gap_days": 0,
            },
        ]
    )
    cohort, attrition = select_persistence_cohort(df, window_days=180, min_claim_count=2)
    assert dict(attrition)["followup_observable"] == 1  # exactly 180d counts
    assert int(cohort.iloc[0]["persistent_at_180d"]) == 1  # covered exactly to 180d


def test_build_journey_records_target_and_anchor_for_discontinuation():
    """Cohort-aware: emit the cohort's target, anchor the journey at treatment_start.

    For disc/persistence the index re-anchors to the first biologic fill, so the
    journey's index_date/journey_start_date must be treatment_start_date (not the
    dx-anchored index_date), and enrollment is measured to that anchor. The 64
    baseline features remain valid (measured at dx-index <= treatment-start).
    """
    dx = pd.Timestamp("2020-01-01")
    tstart = pd.Timestamp("2020-03-01")  # re-anchored index = first biologic fill
    elig = dx - pd.Timedelta(days=100)
    cohort = pd.DataFrame(
        [
            {
                "patid": 77,
                "index_date": dx,
                "treatment_start_date": tstart,
                "elig_start_date": elig,
                "zipcode_5": "10001",
                "age_at_index": 50.0,
                "charlson_score": 2,
                "cci_hiv": 0,
                "discontinued_180d": 1,
                # leakers / non-allowlist columns that MUST NOT survive
                "pdc": 0.5,
                "terminal_gap_days": 120,
                "last_coverage_end": tstart + pd.Timedelta(days=50),
            }
        ]
    )
    rec = build_journey_records(
        cohort, target="discontinued_180d", anchor_col="treatment_start_date"
    )[0]
    assert rec["discontinued_180d"] == 1
    assert "initiated_biologic_180d" not in rec
    assert rec["index_date"] == tstart
    assert rec["journey_start_date"] == tstart
    assert rec["enrollment_duration_days"] == (tstart - elig).days
    for leak in ("pdc", "terminal_gap_days", "last_coverage_end", "treatment_start_date"):
        assert leak not in rec, f"{leak} leaked into journey record"


def test_build_journey_records_emits_only_cataloged_columns():
    """Coverage guard (mart-shaped): every emitted FEATURE column is in the
    MART_SAFE_FEATURES allow-list; the only non-feature keys are an enumerated
    journey-metadata set and the cohort target. A select-from-allow-list adapter
    structurally cannot emit an uncataloged feature, so this contract test is the
    appropriate coverage guard — the AST-based check_manifest_coverage.py is built
    for dict-literal converters and cannot resolve build_journey_records'
    variable-keyed writes (``rec[col] = ...`` / ``rec[target] = ...``), which it
    would treat as unsupported writes and fail discovery."""
    from src.data.manifests import MART_SAFE_FEATURES

    dx = pd.Timestamp("2020-01-01")
    cohort = pd.DataFrame(
        [
            {
                "patid": 1,
                "index_date": dx,
                "treatment_start_date": dx,
                "elig_start_date": dx - pd.Timedelta(days=100),
                "zipcode_5": "10001",
                "age_at_index": 50.0,
                "charlson_score": 2,
                "cci_hiv": 0,
                "discontinued_180d": 1,
                # uncataloged / leak columns present in the input frame
                "pdc": 0.5,
                "terminal_gap_days": 120,
                "last_coverage_end": dx + pd.Timedelta(days=50),
            }
        ]
    )
    rec = build_journey_records(
        cohort, target="discontinued_180d", anchor_col="treatment_start_date"
    )[0]
    # Enumerated journey-metadata / audit keys (NOT in MART_SAFE_FEATURES).
    metadata = {
        "patient_journey_id",
        "patient_id",
        "patient_hash",
        "index_date",
        "journey_start_date",
        "journey_status",
        "discontinuation_flag",
        "data_quality_score",
    }
    allowed = set(MART_SAFE_FEATURES) | metadata | {"discontinued_180d"}
    extra = set(rec) - allowed
    assert not extra, f"uncataloged columns emitted by build_journey_records: {extra}"
    # every non-metadata, non-target key MUST be an allow-list feature
    feature_keys = set(rec) - metadata - {"discontinued_180d"}
    assert feature_keys <= set(MART_SAFE_FEATURES), (
        f"emitted feature columns outside the allow-list: {feature_keys - set(MART_SAFE_FEATURES)}"
    )


def test_build_journey_records_defaults_initiation_backward_compat():
    """Defaults (target=initiated_biologic_180d, anchor=index_date) preserve the
    initiation behavior — existing callers are unaffected."""
    idx = pd.Timestamp("2019-05-01")
    cohort = pd.DataFrame(
        [
            {
                "patid": 5,
                "index_date": idx,
                "elig_start_date": idx - pd.Timedelta(days=200),
                "zipcode_5": "10001",
                "age_at_index": 40.0,
                "charlson_score": 1,
                "cci_hiv": 0,
                "initiated_biologic_180d": 0,
            }
        ]
    )
    rec = build_journey_records(cohort)[0]
    assert rec["initiated_biologic_180d"] == 0
    assert rec["index_date"] == idx
    assert rec["journey_start_date"] == idx


def _entity_mart_rows() -> list[dict]:
    """A tiny entity-stacked mart supporting all 3 cohorts (CI has no real RWD)."""
    idx = pd.Timestamp("2020-01-01")
    day = pd.Timedelta(days=1)
    safe = {"age_at_index": 50.0, "charlson_score": 2, "cci_hiv": 0}
    return [
        # p1: untreated -> initiation negative; excluded from disc/persistence
        {
            "entity_type": "patient",
            "patid": 1,
            "index_biologic_brand": "no_treatment",
            "treatment_start_date": pd.NaT,
            "index_date": idx,
            "claim_record_count": 10,
            "elig_start_date": idx - 300 * day,
            "zipcode_5": "10001",
            "last_observed_date": idx + 400 * day,
            "last_coverage_end": pd.NaT,
            "max_internal_gap_days": 0,
            "terminal_gap_days": 0,
            **safe,
        },
        # p2: initiator, discontinued (coverage ends early + 120d gap), observable
        {
            "entity_type": "patient",
            "patid": 2,
            "index_biologic_brand": "XOLAIR",
            "treatment_start_date": idx + 10 * day,
            "index_date": idx,
            "claim_record_count": 8,
            "elig_start_date": idx - 365 * day,
            "zipcode_5": "90001",
            "last_observed_date": idx + 410 * day,
            "last_coverage_end": idx + 110 * day,
            "max_internal_gap_days": 120,
            "terminal_gap_days": 0,
            **safe,
        },
        # p3: initiator, persistent (covered through 180, small gap), observable
        {
            "entity_type": "patient",
            "patid": 3,
            "index_biologic_brand": "DUPIXENT",
            "treatment_start_date": idx + 20 * day,
            "index_date": idx,
            "claim_record_count": 12,
            "elig_start_date": idx - 200 * day,
            "zipcode_5": "60601",
            "last_observed_date": idx + 420 * day,
            "last_coverage_end": idx + 240 * day,
            "max_internal_gap_days": 30,
            "terminal_gap_days": 0,
            **safe,
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
            "last_observed_date": pd.NaT,
            "last_coverage_end": pd.NaT,
            "max_internal_gap_days": None,
            "terminal_gap_days": None,
            "age_at_index": None,
            "charlson_score": None,
            "cci_hiv": None,
        },
    ]


def test_convert_discontinuation_end_to_end_synthetic(tmp_path):
    mart = tmp_path / "mart.parquet"
    pd.DataFrame(_entity_mart_rows()).to_parquet(mart)
    out = tmp_path / "disc"
    summary = convert(
        input_path=str(mart),
        output_dir=str(out),
        cohort="discontinuation",
        window_days=180,
        min_claim_count=2,
    )
    assert summary["cohort"] == "discontinuation"
    assert summary["patients"] == 2  # initiators only (p2,p3); untreated p1 + hcp excluded
    assert summary["positives"] == 1  # p2 discontinued
    j = pd.read_parquet(out / "e2i_ml_v3_patient_journeys.parquet")
    assert "discontinued_180d" in j.columns
    assert "initiated_biologic_180d" not in j.columns
    for leak in (
        "pdc",
        "terminal_gap_days",
        "last_coverage_end",
        "treatment_start_date",
        "index_biologic_brand",
    ):
        assert leak not in j.columns, f"{leak} leaked into discontinuation journeys"
    for fname in (
        "e2i_ml_v3_patient_journeys.parquet",
        "e2i_ml_v3_split_registry.json",
        "attrition_report.csv",
        "data_dictionary.csv",
    ):
        assert (out / fname).exists()


def test_main_cohort_all_writes_three_cohort_dirs(tmp_path):
    mart = tmp_path / "mart.parquet"
    pd.DataFrame(_entity_mart_rows()).to_parquet(mart)
    base = tmp_path / "marts"
    rc = convert_main(["--cohort", "all", "--input", str(mart), "--output", str(base)])
    assert rc == 0
    expected = {
        "initiation": "initiated_biologic_180d",
        "discontinuation": "discontinued_180d",
        "persistence": "persistent_at_180d",
    }
    for cohort, target in expected.items():
        journeys = base / cohort / "e2i_ml_v3_patient_journeys.parquet"
        assert journeys.exists(), f"missing cohort dir: {cohort}"
        cols = pd.read_parquet(journeys).columns
        assert target in cols, f"{cohort} missing target {target}"


def test_observable_initiator_with_nat_coverage_end_is_right_censored():
    """A NaT last_coverage_end for an observable initiator must be DROPPED
    (right-censored, with an explicit attrition step) — NOT silently labeled
    disc=0/persist=0 via a NaN comparison short-circuit (review finding, MED)."""
    ts = pd.Timestamp("2020-01-01")
    day = pd.Timedelta(days=1)
    df = pd.DataFrame(
        [
            {
                "patid": 51,
                "index_biologic_brand": "XOLAIR",
                "treatment_start_date": ts,
                "claim_record_count": 10,
                "last_observed_date": ts + 400 * day,
                "last_coverage_end": pd.NaT,
                "max_internal_gap_days": 200,
                "terminal_gap_days": 200,
            },
        ]
    )
    dcoh, datt = select_discontinuation_cohort(df, window_days=180, min_claim_count=2)
    pcoh, patt = select_persistence_cohort(df, window_days=180, min_claim_count=2)
    assert len(dcoh) == 0 and len(pcoh) == 0  # dropped, not labeled
    assert dict(datt)["followup_observable"] == 1  # passed follow-up gate
    assert dict(datt)["coverage_end_observable"] == 0  # then right-censored (NaT lce)
    assert dict(datt)["target_positives"] == 0


def test_convert_discontinuation_attrition_records_full_patient_panel(tmp_path):
    """The treatment-anchored read pushes down to initiators; the attrition report
    must still record the full patient denominator (patient_panel) for transparency."""
    mart = tmp_path / "mart.parquet"
    pd.DataFrame(_entity_mart_rows()).to_parquet(mart)
    out = tmp_path / "disc"
    convert(input_path=str(mart), output_dir=str(out), cohort="discontinuation")
    attr = pd.read_csv(out / "attrition_report.csv")
    steps = dict(zip(attr["step"], attr["count"], strict=True))
    assert steps["patient_panel"] == 3  # p1,p2,p3 patients (full denominator)
    assert steps["initiators"] == 2  # p2,p3 only


def test_convert_empty_cohort_writes_files_without_error(tmp_path):
    """A cohort resolving to 0 records must not raise (prevalence guard + split n==0)."""
    rows = [r for r in _entity_mart_rows() if r["patid"] in (1, 999)]  # untreated + hcp only
    mart = tmp_path / "m.parquet"
    pd.DataFrame(rows).to_parquet(mart)
    out = tmp_path / "empty"
    summary = convert(input_path=str(mart), output_dir=str(out), cohort="discontinuation")
    assert summary["patients"] == 0
    assert summary["positives"] == 0
    assert summary["prevalence"] == 0.0
    assert (out / "e2i_ml_v3_patient_journeys.parquet").exists()


def test_convert_zero_positive_cohort_no_error(tmp_path):
    """A rows-but-zero-positive cohort must not raise a ZeroDivision in prevalence."""
    ts = pd.Timestamp("2020-01-01")
    day = pd.Timedelta(days=1)
    rows = [
        {
            "entity_type": "patient",
            "patid": 2,
            "index_biologic_brand": "XOLAIR",
            "treatment_start_date": ts,
            "index_date": ts,
            "claim_record_count": 10,
            "elig_start_date": ts - 200 * day,
            "zipcode_5": "10001",
            "last_observed_date": ts + 400 * day,
            "last_coverage_end": ts + 220 * day,
            "max_internal_gap_days": 10,
            "terminal_gap_days": 0,
            "age_at_index": 50.0,
            "charlson_score": 1,
            "cci_hiv": 0,
        },
    ]
    mart = tmp_path / "m.parquet"
    pd.DataFrame(rows).to_parquet(mart)
    out = tmp_path / "zp"
    summary = convert(input_path=str(mart), output_dir=str(out), cohort="discontinuation")
    assert summary["patients"] == 1  # persistent initiator -> disc negative
    assert summary["positives"] == 0
    assert summary["prevalence"] == 0.0
