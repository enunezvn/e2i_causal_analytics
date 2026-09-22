"""Lane A (spec 2026-09-22 §3A.1): the CAUSAL cohort export of the Optum mart.

The prediction cohorts drop ``index_biologic_brand`` on purpose (the manifest
declares it post-index ``mart_treatment``). For causal estimation that column
IS the treatment, so ``persistence_causal`` is a SEPARATE cohort that keeps it
and adds the days-supply-robust primary outcome ``persistent_at_180d_g28``
(``docs/demos/results/2026-09-22_persistence_definition_disproof/``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.convert_optum_mart import (  # noqa: E402
    _ANCHOR_BY_COHORT,
    _OUTPUT_BY_COHORT,
    _SELECTOR_BY_COHORT,
    _SPLIT_CONFIG_BY_COHORT,
    _TREATMENT_ANCHORED,
    CAUSAL_ARMS,
    CAUSAL_COHORT,
    CAUSAL_EXTRA_COLS,
    CAUSAL_RECORDS_NAME,
    COHORT_TARGETS,
    PERSIST_GRACE_DAYS,
    PREDICTION_COHORTS,
    SWITCH_FLAG,
    TARGET_PERSISTENT,
    TARGET_PERSISTENT_G28,
    TREATMENT_COL,
    build_journey_records,
    convert,
    select_persistence_causal_cohort,
)
from scripts.convert_optum_mart import main as convert_main  # noqa: E402


def _initiators_two_arms() -> pd.DataFrame:
    ts = pd.Timestamp("2020-01-01")
    day = pd.Timedelta(days=1)
    base = {
        "claim_record_count": 10,
        "last_observed_date": ts + 400 * day,
        "terminal_gap_days": 0,
    }
    return pd.DataFrame(
        [
            # p4: XOLAIR, covered through day 220, gap 30 -> persist 1 / g28 1 / disc 0
            {
                "patid": 4,
                "index_biologic_brand": "XOLAIR",
                "treatment_start_date": ts,
                "last_coverage_end": ts + 220 * day,
                "max_internal_gap_days": 30,
                SWITCH_FLAG: 0,
                **base,
            },
            # p5: DUPIXENT, coverage ends day 100 + 120d gap -> persist 0 / g28 0 / disc 1
            {
                "patid": 5,
                "index_biologic_brand": "DUPIXENT",
                "treatment_start_date": ts,
                "last_coverage_end": ts + 100 * day,
                "max_internal_gap_days": 120,
                SWITCH_FLAG: 1,
                **base,
            },
            # p6: XOLAIR, covered to 220 but a >60d gap -> persist 0 / g28 0 / disc 0
            {
                "patid": 6,
                "index_biologic_brand": "XOLAIR",
                "treatment_start_date": ts,
                "last_coverage_end": ts + 220 * day,
                "max_internal_gap_days": 90,
                SWITCH_FLAG: 0,
                **base,
            },
            # p7: DUPIXENT, coverage ends day 160 (a 14-day-supply last fill), no gap
            #     -> shipped persist 0 (160 < 180) / g28 1 (160 >= 152) / disc 0
            {
                "patid": 7,
                "index_biologic_brand": "DUPIXENT",
                "treatment_start_date": ts,
                "last_coverage_end": ts + 160 * day,
                "max_internal_gap_days": 10,
                SWITCH_FLAG: None,  # NULL flag reads as 0, never drops the row
                **base,
            },
            # p8: a brand outside the observed contrast (a future drop) -> excluded,
            #     never coded 0 (= XOLAIR) silently
            {
                "patid": 8,
                "index_biologic_brand": "RHAPSIDO",
                "treatment_start_date": ts,
                "last_coverage_end": ts + 220 * day,
                "max_internal_gap_days": 0,
                SWITCH_FLAG: 0,
                **base,
            },
            # p9: XOLAIR, cov_to_end=151 (one day BELOW the g28 boundary
            #     window_days-PERSIST_GRACE_DAYS=152), gap 0 -> g28 0
            {
                "patid": 9,
                "index_biologic_brand": "XOLAIR",
                "treatment_start_date": ts,
                "last_coverage_end": ts + 151 * day,
                "max_internal_gap_days": 0,
                SWITCH_FLAG: 0,
                **base,
            },
            # p10: XOLAIR, cov_to_end=152 (AT the g28 boundary) -> g28 1; a
            #      >=-to-> mutation on the cov_to_end predicate flips ONLY this row
            {
                "patid": 10,
                "index_biologic_brand": "XOLAIR",
                "treatment_start_date": ts,
                "last_coverage_end": ts + 152 * day,
                "max_internal_gap_days": 0,
                SWITCH_FLAG: 0,
                **base,
            },
            # p11: DUPIXENT, cov_to_end=220 (well past any window), gap=60 (AT the
            #      PERSIST_GAP_DAYS boundary) -> persist/g28 1; a <=-to-< mutation
            #      on the gap predicate flips ONLY this row
            {
                "patid": 11,
                "index_biologic_brand": "DUPIXENT",
                "treatment_start_date": ts,
                "last_coverage_end": ts + 220 * day,
                "max_internal_gap_days": 60,
                SWITCH_FLAG: 0,
                **base,
            },
            # p12: DUPIXENT, cov_to_end=220, gap=61 (one day PAST the gap
            #      boundary) -> persist/g28 0
            {
                "patid": 12,
                "index_biologic_brand": "DUPIXENT",
                "treatment_start_date": ts,
                "last_coverage_end": ts + 220 * day,
                "max_internal_gap_days": 61,
                SWITCH_FLAG: 0,
                **base,
            },
            # p13: XOLAIR, cov_to_end=100 (< window), gap=90 (AT the
            #      DISCONT_GAP_DAYS boundary) -> discontinued 1; a >=-to->
            #      mutation on the gap predicate in _discontinued flips ONLY
            #      this row
            {
                "patid": 13,
                "index_biologic_brand": "XOLAIR",
                "treatment_start_date": ts,
                "last_coverage_end": ts + 100 * day,
                "max_internal_gap_days": 90,
                SWITCH_FLAG: 0,
                **base,
            },
            # p14: DUPIXENT, cov_to_end=100, gap=89 (one day BELOW the
            #      discontinuation gap boundary) -> discontinued 0
            {
                "patid": 14,
                "index_biologic_brand": "DUPIXENT",
                "treatment_start_date": ts,
                "last_coverage_end": ts + 100 * day,
                "max_internal_gap_days": 89,
                SWITCH_FLAG: 0,
                **base,
            },
        ]
    )


def test_causal_selector_truth_table_and_treatment_coding():
    cohort, attrition = select_persistence_causal_cohort(
        _initiators_two_arms(), window_days=180, min_claim_count=2
    )
    steps = dict(attrition)
    assert steps["excluded_arm:RHAPSIDO"] == 1  # p8, loud not silent
    assert steps["two_arm_contrast"] == 10  # p8 excluded; p4-p7, p9-p14 kept
    assert set(cohort["patid"]) == {4, 5, 6, 7, 9, 10, 11, 12, 13, 14}
    by = cohort.set_index("patid")
    assert by[TREATMENT_COL].to_dict() == {
        4: 0,
        5: 1,
        6: 0,
        7: 1,
        9: 0,
        10: 0,
        11: 1,
        12: 1,
        13: 0,
        14: 1,
    }
    assert by["persistent_at_180d"].to_dict() == {
        4: 1,
        5: 0,
        6: 0,
        7: 0,
        9: 0,
        10: 0,
        11: 1,
        12: 0,
        13: 0,
        14: 0,
    }
    # p10 (cov_to_end==152) and p11 (gap==60) are the boundary cases: a
    # >=-to-> mutation on cov_to_end flips ONLY p10; a <=-to-< mutation on gap
    # flips ONLY p11. p9 (151) and p12 (61) are the just-below-boundary
    # complements that must NOT flip.
    assert by[TARGET_PERSISTENT_G28].to_dict() == {
        4: 1,
        5: 0,
        6: 0,
        7: 1,
        9: 0,
        10: 1,
        11: 1,
        12: 0,
        13: 0,
        14: 0,
    }
    # p13 (gap==90, the DISCONT_GAP_DAYS boundary) is the mutation-catching
    # case: a >=-to-> mutation on the gap predicate in _discontinued flips
    # ONLY p13. p14 (gap==89) is the just-below-boundary complement that must
    # NOT flip.
    assert by["discontinued_180d"].to_dict() == {
        4: 0,
        5: 1,
        6: 0,
        7: 0,
        9: 0,
        10: 0,
        11: 0,
        12: 0,
        13: 1,
        14: 0,
    }
    assert by[SWITCH_FLAG].to_dict() == {
        4: 0,
        5: 1,
        6: 0,
        7: 0,
        9: 0,
        10: 0,
        11: 0,
        12: 0,
        13: 0,
        14: 0,
    }
    for col in (
        TREATMENT_COL,
        "persistent_at_180d",
        TARGET_PERSISTENT_G28,
        "discontinued_180d",
        SWITCH_FLAG,
    ):
        assert cohort[col].dtype.kind in "iu", col
    assert steps["target_positives"] == 4  # g28 positives: p4, p7, p10, p11
    assert steps["arm_dupixent"] == 5  # p5, p7, p11, p12, p14


def test_causal_selector_grace_is_28_days_and_arms_are_the_observed_pair():
    assert PERSIST_GRACE_DAYS == 28
    assert CAUSAL_ARMS == ("XOLAIR", "DUPIXENT")
    assert TREATMENT_COL == "treatment_dupixent"
    assert TARGET_PERSISTENT_G28 == "persistent_at_180d_g28"


def test_causal_selector_fails_loud_without_the_switch_flag():
    df = _initiators_two_arms().drop(columns=[SWITCH_FLAG])
    with pytest.raises(KeyError, match=SWITCH_FLAG):
        select_persistence_causal_cohort(df, window_days=180, min_claim_count=2)


def _entity_mart_rows_two_arms() -> list[dict]:
    """A tiny entity-stacked mart with both arms + an untreated patient + an HCP row."""
    idx = pd.Timestamp("2020-01-01")
    day = pd.Timedelta(days=1)
    safe = {"age_at_index": 50.0, "charlson_score": 2, "cci_hiv": 0, "payer_category": "commercial"}
    return [
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
            SWITCH_FLAG: 0,
            **safe,
        },
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
            SWITCH_FLAG: 1,
            **safe,
        },
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
            SWITCH_FLAG: 0,
            **safe,
        },
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
            SWITCH_FLAG: None,
            "age_at_index": None,
            "charlson_score": None,
            "cci_hiv": None,
            "payer_category": None,
        },
    ]


def test_build_journey_records_extra_cols_are_emitted_and_default_is_unchanged():
    tstart = pd.Timestamp("2020-03-01")
    cohort = pd.DataFrame(
        [
            {
                "patid": 77,
                "index_date": pd.Timestamp("2020-01-01"),
                "treatment_start_date": tstart,
                "elig_start_date": pd.Timestamp("2019-09-01"),
                "zipcode_5": "10001",
                "age_at_index": 50.0,
                "charlson_score": 2,
                "cci_hiv": 0,
                "index_biologic_brand": "DUPIXENT",
                TREATMENT_COL: 1,
                TARGET_PERSISTENT_G28: 1,
                "discontinued_180d": 0,
                SWITCH_FLAG: 0,
                "persistent_at_180d": 0,
            }
        ]
    )
    rec = build_journey_records(
        cohort,
        target=TARGET_PERSISTENT_G28,
        anchor_col="treatment_start_date",
        extra_cols=CAUSAL_EXTRA_COLS,
    )[0]
    assert rec["index_biologic_brand"] == "DUPIXENT"
    assert rec[TREATMENT_COL] == 1 and isinstance(rec[TREATMENT_COL], int)
    assert rec["treatment_start_date"] == tstart
    assert (
        rec["persistent_at_180d"] == 0 and rec["discontinued_180d"] == 0 and rec[SWITCH_FLAG] == 0
    )
    assert rec[TARGET_PERSISTENT_G28] == 1
    # default call (prediction cohorts) still drops every one of them
    plain = build_journey_records(
        cohort, target=TARGET_PERSISTENT_G28, anchor_col="treatment_start_date"
    )[0]
    for col in CAUSAL_EXTRA_COLS:
        assert col not in plain, f"{col} leaked into a prediction record"


def test_build_journey_records_extra_cols_reject_non_binary_flags():
    """A flag-kind extra column (e.g. TREATMENT_COL/SWITCH_FLAG) must be exactly
    0 or 1: a fractional value, a NaN, or a pd.NA raises ValueError naming the
    column, never a silent truncation/coercion and never
    'boolean value of NA is ambiguous'."""
    base_row = {
        "patid": 77,
        "index_date": pd.Timestamp("2020-01-01"),
        "treatment_start_date": pd.Timestamp("2020-03-01"),
        "elig_start_date": pd.Timestamp("2019-09-01"),
        "zipcode_5": "10001",
        "age_at_index": 50.0,
        "charlson_score": 2,
        "cci_hiv": 0,
        "index_biologic_brand": "DUPIXENT",
        TREATMENT_COL: 1,
        TARGET_PERSISTENT_G28: 1,
        "discontinued_180d": 0,
        SWITCH_FLAG: 0,
        "persistent_at_180d": 0,
    }
    for bad_value in (0.5, float("nan"), pd.NA):
        cohort = pd.DataFrame([{**base_row, SWITCH_FLAG: bad_value}])
        with pytest.raises(ValueError, match=SWITCH_FLAG):
            build_journey_records(
                cohort,
                target=TARGET_PERSISTENT_G28,
                anchor_col="treatment_start_date",
                extra_cols=CAUSAL_EXTRA_COLS,
            )


def test_build_journey_records_extra_cols_text_kind_refuses_non_strings():
    """A text-kind extra column (index_biologic_brand) must be a real string:
    None must raise ValueError naming the column, never stringify to the
    literal text 'None'."""
    base_row = {
        "patid": 77,
        "index_date": pd.Timestamp("2020-01-01"),
        "treatment_start_date": pd.Timestamp("2020-03-01"),
        "elig_start_date": pd.Timestamp("2019-09-01"),
        "zipcode_5": "10001",
        "age_at_index": 50.0,
        "charlson_score": 2,
        "cci_hiv": 0,
        "index_biologic_brand": None,
        TREATMENT_COL: 1,
        TARGET_PERSISTENT_G28: 1,
        "discontinued_180d": 0,
        SWITCH_FLAG: 0,
        "persistent_at_180d": 0,
    }
    cohort = pd.DataFrame([base_row])
    with pytest.raises(ValueError, match="index_biologic_brand"):
        build_journey_records(
            cohort,
            target=TARGET_PERSISTENT_G28,
            anchor_col="treatment_start_date",
            extra_cols=CAUSAL_EXTRA_COLS,
        )


def test_build_journey_records_extra_cols_are_exactly_the_causal_columns():
    """Allow-list guard extended over extra_cols (mirrors the multicohort guard
    test_build_journey_records_emits_only_cataloged_columns): with
    CAUSAL_EXTRA_COLS passed, the only keys beyond MART_SAFE_FEATURES + journey
    metadata + target are EXACTLY the causal extra columns."""
    from src.data.manifests import MART_SAFE_FEATURES

    tstart = pd.Timestamp("2020-03-01")
    cohort = pd.DataFrame(
        [
            {
                "patid": 77,
                "index_date": pd.Timestamp("2020-01-01"),
                "treatment_start_date": tstart,
                "elig_start_date": pd.Timestamp("2019-09-01"),
                "zipcode_5": "10001",
                "age_at_index": 50.0,
                "charlson_score": 2,
                "cci_hiv": 0,
                "index_biologic_brand": "DUPIXENT",
                TREATMENT_COL: 1,
                TARGET_PERSISTENT_G28: 1,
                "discontinued_180d": 0,
                SWITCH_FLAG: 0,
                "persistent_at_180d": 0,
            }
        ]
    )
    rec = build_journey_records(
        cohort,
        target=TARGET_PERSISTENT_G28,
        anchor_col="treatment_start_date",
        extra_cols=CAUSAL_EXTRA_COLS,
    )[0]
    # Enumerated journey-metadata / audit keys (NOT in MART_SAFE_FEATURES) --
    # identical set to the multicohort guard (enrollment_duration_days and
    # geographic_region are themselves MART_SAFE_FEATURES members).
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
    allowed = set(MART_SAFE_FEATURES) | metadata | {TARGET_PERSISTENT_G28}
    assert set(rec) - allowed == set(CAUSAL_EXTRA_COLS)


def test_causal_registry_entries():
    assert COHORT_TARGETS[CAUSAL_COHORT] == TARGET_PERSISTENT_G28
    assert CAUSAL_COHORT == "persistence_causal"
    assert CAUSAL_RECORDS_NAME == "e2i_causal_v1_biologic_persistence"
    assert CAUSAL_EXTRA_COLS == (
        "index_biologic_brand",
        TREATMENT_COL,
        "treatment_start_date",
        "discontinued_180d",
        SWITCH_FLAG,
        "persistent_at_180d",
    )
    assert CAUSAL_COHORT not in PREDICTION_COHORTS
    assert PREDICTION_COHORTS == ("initiation", "discontinuation", "persistence")


def test_cohort_registries_share_one_key_set():
    """The five per-cohort registries must never drift apart: adding a cohort to
    one and forgetting another is a KeyError waiting to happen at call time."""
    key_set = set(COHORT_TARGETS)
    for registry in (
        _SELECTOR_BY_COHORT,
        _ANCHOR_BY_COHORT,
        _SPLIT_CONFIG_BY_COHORT,
        _OUTPUT_BY_COHORT,
    ):
        assert set(registry) == key_set, registry
    assert set(_TREATMENT_ANCHORED) <= key_set


def test_convert_persistence_causal_end_to_end(tmp_path):
    mart = tmp_path / "mart.parquet"
    pd.DataFrame(_entity_mart_rows_two_arms()).to_parquet(mart)
    out = tmp_path / "causal"
    summary = convert(
        input_path=str(mart),
        output_dir=str(out),
        cohort=CAUSAL_COHORT,
        window_days=180,
        min_claim_count=2,
    )
    assert summary["cohort"] == CAUSAL_COHORT
    assert summary["patients"] == 2
    assert summary["positives"] == 1  # p3 g28-persistent
    assert summary["arms"] == {"XOLAIR": 1, "DUPIXENT": 1}
    frame = pd.read_parquet(out / f"{CAUSAL_RECORDS_NAME}.parquet")
    assert not (out / "e2i_ml_v3_patient_journeys.parquet").exists()
    for col in (*CAUSAL_EXTRA_COLS, TARGET_PERSISTENT_G28, "is_synthetic", "payer_category"):
        assert col in frame.columns, col
    assert frame["is_synthetic"].dtype == bool and not frame["is_synthetic"].any()
    by = frame.set_index("patient_id")
    assert by.loc["PAT_2", TREATMENT_COL] == 0 and by.loc["PAT_3", TREATMENT_COL] == 1
    assert by.loc["PAT_2", "discontinued_180d"] == 1 and by.loc["PAT_3", TARGET_PERSISTENT_G28] == 1
    # journeys anchor at the first biologic fill
    assert pd.Timestamp(by.loc["PAT_3", "index_date"]) == pd.Timestamp("2020-01-21")
    attrition = pd.read_csv(out / "attrition_report.csv")
    assert "two_arm_contrast" in set(attrition["step"])
    dictionary = pd.read_csv(out / "data_dictionary.csv")
    assert set(CAUSAL_EXTRA_COLS) <= set(dictionary["feature"])
    assert set(dictionary.loc[dictionary["feature"] == TREATMENT_COL, "type"]) == {"treatment"}
    assert set(dictionary.loc[dictionary["feature"] == TARGET_PERSISTENT_G28, "type"]) == {"target"}


def test_prediction_cohorts_still_drop_the_treatment(tmp_path):
    """Spec §3A.1: the causal frame carries the treatment; the prediction frame does not."""
    mart = tmp_path / "mart.parquet"
    pd.DataFrame(_entity_mart_rows_two_arms()).to_parquet(mart)
    out = tmp_path / "persistence"
    convert(
        input_path=str(mart),
        output_dir=str(out),
        cohort="persistence",
        window_days=180,
        min_claim_count=2,
    )
    frame = pd.read_parquet(out / "e2i_ml_v3_patient_journeys.parquet")
    # TARGET_PERSISTENT ("persistent_at_180d") is a member of CAUSAL_EXTRA_COLS
    # (the causal export carries it as a secondary/legacy outcome) AND is this
    # PREDICTION cohort's own pre-existing supervised target -- it belongs in
    # this frame for a reason unrelated to Lane A, so it is excluded from the
    # leak-check and asserted present instead, just below.
    for col in {*CAUSAL_EXTRA_COLS, TARGET_PERSISTENT_G28, "is_synthetic"} - {TARGET_PERSISTENT}:
        assert col not in frame.columns, f"{col} leaked into the prediction persistence frame"
    assert "persistent_at_180d" in frame.columns


def test_main_all_builds_prediction_cohorts_only(tmp_path):
    mart = tmp_path / "mart.parquet"
    pd.DataFrame(_entity_mart_rows_two_arms()).to_parquet(mart)
    base = tmp_path / "marts"
    assert convert_main(["--cohort", "all", "--input", str(mart), "--output", str(base)]) == 0
    assert {p.name for p in base.iterdir()} == set(PREDICTION_COHORTS)
    assert (
        convert_main(
            ["--cohort", CAUSAL_COHORT, "--input", str(mart), "--output", str(base / CAUSAL_COHORT)]
        )
        == 0
    )
    assert (base / CAUSAL_COHORT / f"{CAUSAL_RECORDS_NAME}.parquet").exists()
