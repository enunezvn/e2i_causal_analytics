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
    CAUSAL_ARMS,
    PERSIST_GRACE_DAYS,
    SWITCH_FLAG,
    TARGET_PERSISTENT_G28,
    TREATMENT_COL,
    select_persistence_causal_cohort,
)


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
        ]
    )


def test_causal_selector_truth_table_and_treatment_coding():
    cohort, attrition = select_persistence_causal_cohort(
        _initiators_two_arms(), window_days=180, min_claim_count=2
    )
    steps = dict(attrition)
    assert steps["two_arm_contrast"] == 4  # p8 excluded
    assert set(cohort["patid"]) == {4, 5, 6, 7}
    by = cohort.set_index("patid")
    assert by[TREATMENT_COL].to_dict() == {4: 0, 5: 1, 6: 0, 7: 1}
    assert by["persistent_at_180d"].to_dict() == {4: 1, 5: 0, 6: 0, 7: 0}
    assert by[TARGET_PERSISTENT_G28].to_dict() == {4: 1, 5: 0, 6: 0, 7: 1}
    assert by["discontinued_180d"].to_dict() == {4: 0, 5: 1, 6: 0, 7: 0}
    assert by[SWITCH_FLAG].to_dict() == {4: 0, 5: 1, 6: 0, 7: 0}
    for col in (
        TREATMENT_COL,
        "persistent_at_180d",
        TARGET_PERSISTENT_G28,
        "discontinued_180d",
        SWITCH_FLAG,
    ):
        assert cohort[col].dtype.kind in "iu", col
    assert steps["target_positives"] == 2  # g28 positives: p4, p7
    assert steps["arm_dupixent"] == 2


def test_causal_selector_grace_is_28_days_and_arms_are_the_observed_pair():
    assert PERSIST_GRACE_DAYS == 28
    assert CAUSAL_ARMS == ("XOLAIR", "DUPIXENT")
    assert TREATMENT_COL == "treatment_dupixent"
    assert TARGET_PERSISTENT_G28 == "persistent_at_180d_g28"


def test_causal_selector_fails_loud_without_the_switch_flag():
    df = _initiators_two_arms().drop(columns=[SWITCH_FLAG])
    with pytest.raises(KeyError, match=SWITCH_FLAG):
        select_persistence_causal_cohort(df, window_days=180, min_claim_count=2)
