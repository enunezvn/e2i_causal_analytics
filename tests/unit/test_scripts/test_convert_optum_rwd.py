"""Unit tests for the smart-index fallback added to ``scripts/convert_optum_rwd.py``
to close backlog #19 (Optum cohort-build attrition over-filtering).

The fallback only fires for cohort A (initiation). Cohorts B/C re-anchor on
first biologic fill — re-anchoring there changes cohort semantics
(disc/pers measure outcomes 180d post-FIRST-fill).
"""

from __future__ import annotations

from datetime import timedelta

import pandas as pd

from scripts.convert_optum_rwd import (
    ENROLLMENT_POST_DAYS,
    ENROLLMENT_PRE_DAYS,
    OptumDataConverter,
)

# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _make_converter() -> OptumDataConverter:
    """Construct a converter shell. We don't run the parquet read — tests
    populate the per-patient index maps directly."""
    return OptumDataConverter(
        parquet_dir=".",  # never accessed in unit tests
        output_dir=".",
        cohorts=("initiation",),
    )


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s)


def _demo_row(eligeff: str | None, eligend: str | None) -> pd.Series:
    return pd.Series(
        {
            "patid": 1,
            "eligeff": _ts(eligeff) if eligeff else pd.NaT,
            "eligend": _ts(eligend) if eligend else pd.NaT,
            "diagcode_raw": "L509",
        }
    )


# --------------------------------------------------------------------------- #
# Smart-index helper unit tests                                               #
# --------------------------------------------------------------------------- #


class TestDeriveIndexDateFeasibilityAware:
    def test_returns_none_when_eligeff_missing(self) -> None:
        cv = _make_converter()
        demo_row = _demo_row(None, "2022-01-01")
        assert cv._derive_index_date_feasibility_aware(1, demo_row) is None

    def test_returns_none_when_eligend_missing(self) -> None:
        cv = _make_converter()
        demo_row = _demo_row("2018-01-01", None)
        assert cv._derive_index_date_feasibility_aware(1, demo_row) is None

    def test_returns_none_when_feasible_window_empty(self) -> None:
        """eligeff + PRE > eligend - POST (enrollment too short for any
        feasible index)."""
        cv = _make_converter()
        # 540d total enrollment but PRE+POST=540 makes feasibility band a
        # single point; one day shorter and the band is empty.
        eligeff = _ts("2020-01-01")
        eligend = eligeff + timedelta(days=ENROLLMENT_PRE_DAYS + ENROLLMENT_POST_DAYS - 1)
        demo_row = pd.Series(
            {
                "patid": 1,
                "eligeff": eligeff,
                "eligend": eligend,
                "diagcode_raw": "L509",
            }
        )
        assert cv._derive_index_date_feasibility_aware(1, demo_row) is None

    def test_picks_second_inpatient_l50_within_feasible_band(self) -> None:
        """Priority 1: ≥2 distinct inpatient L50.x admit dates → 2nd."""
        cv = _make_converter()
        # Feasible band: [2019-12-26, 2022-07-04] for eligeff=2019-01-01,
        # eligend=2023-01-01, PRE=360, POST=180.
        cv._inpatient_by_pat[1] = pd.DataFrame(
            {
                "diag1": ["L509", "L508", "L509"],
                "diag2": [None, None, None],
                "diag3": [None, None, None],
                "diag4": [None, None, None],
                "diag5": [None, None, None],
                "admit_date": [
                    _ts("2018-06-01"),  # before feasibility band
                    _ts("2020-03-15"),  # in band
                    _ts("2021-06-15"),  # in band
                ],
            }
        )
        demo_row = _demo_row("2019-01-01", "2023-01-01")
        result = cv._derive_index_date_feasibility_aware(1, demo_row)
        assert result == _ts("2021-06-15")  # 2nd of those WITHIN the band

    def test_picks_first_inpatient_l50_when_only_one_in_band(self) -> None:
        cv = _make_converter()
        cv._inpatient_by_pat[1] = pd.DataFrame(
            {
                "diag1": ["L509", "L509"],
                "diag2": [None, None],
                "diag3": [None, None],
                "diag4": [None, None],
                "diag5": [None, None],
                "admit_date": [
                    _ts("2018-06-01"),  # outside band
                    _ts("2020-03-15"),  # in band — single qualifying
                ],
            }
        )
        demo_row = _demo_row("2019-01-01", "2023-01-01")
        result = cv._derive_index_date_feasibility_aware(1, demo_row)
        assert result == _ts("2020-03-15")

    def test_falls_through_to_earliest_claim_when_no_inpatient_l50(
        self,
    ) -> None:
        """Priority 3: no inpatient L50.x at all → earliest med/proc/lab."""
        cv = _make_converter()
        cv._med_by_pat[1] = pd.DataFrame(
            {"medication_date": [_ts("2018-12-01"), _ts("2020-08-01")]}
        )
        cv._proc_by_pat[1] = pd.DataFrame({"proc_date": [_ts("2020-07-15"), _ts("2021-01-01")]})
        cv._lab_by_pat[1] = pd.DataFrame({"fst_dt": [_ts("2020-09-01")]})
        demo_row = _demo_row("2019-01-01", "2023-01-01")
        result = cv._derive_index_date_feasibility_aware(1, demo_row)
        # Earliest in band [2019-12-26, 2022-07-04]: proc 2020-07-15
        assert result == _ts("2020-07-15")

    def test_returns_none_when_no_anchor_in_feasible_band(self) -> None:
        cv = _make_converter()
        cv._med_by_pat[1] = pd.DataFrame(
            {"medication_date": [_ts("2018-06-01"), _ts("2024-01-01")]}
        )
        demo_row = _demo_row("2019-01-01", "2023-01-01")
        # Feasibility band [2019-12-26, 2022-07-04]; both meds outside it.
        assert cv._derive_index_date_feasibility_aware(1, demo_row) is None

    def test_handles_missing_per_patient_groups_gracefully(self) -> None:
        cv = _make_converter()
        # No entries in any per-patient index map for patid=999.
        demo_row = _demo_row("2019-01-01", "2023-01-01")
        assert cv._derive_index_date_feasibility_aware(999, demo_row) is None

    def test_inpatient_priority_beats_claim_date_priority_within_band(
        self,
    ) -> None:
        """When both are in-band, inpatient L50.x wins over claim fallback."""
        cv = _make_converter()
        cv._inpatient_by_pat[1] = pd.DataFrame(
            {
                "diag1": ["L509"],
                "diag2": [None],
                "diag3": [None],
                "diag4": [None],
                "diag5": [None],
                "admit_date": [_ts("2021-06-15")],
            }
        )
        cv._med_by_pat[1] = pd.DataFrame(
            {"medication_date": [_ts("2020-03-15")]}  # earlier in band
        )
        demo_row = _demo_row("2019-01-01", "2023-01-01")
        # Even though med is earlier, inpatient L50.x has priority.
        result = cv._derive_index_date_feasibility_aware(1, demo_row)
        assert result == _ts("2021-06-15")

    def test_boundary_date_at_feasible_start_is_inclusive(self) -> None:
        """A date exactly at feasible_start should be accepted."""
        cv = _make_converter()
        eligeff = _ts("2019-01-01")
        feasible_start = eligeff + timedelta(days=ENROLLMENT_PRE_DAYS)
        cv._med_by_pat[1] = pd.DataFrame({"medication_date": [feasible_start]})
        demo_row = _demo_row("2019-01-01", "2023-01-01")
        result = cv._derive_index_date_feasibility_aware(1, demo_row)
        assert result == feasible_start

    def test_boundary_date_at_feasible_end_is_inclusive(self) -> None:
        cv = _make_converter()
        eligend = _ts("2023-01-01")
        feasible_end = eligend - timedelta(days=ENROLLMENT_POST_DAYS)
        cv._med_by_pat[1] = pd.DataFrame({"medication_date": [feasible_end]})
        demo_row = _demo_row("2019-01-01", "2023-01-01")
        result = cv._derive_index_date_feasibility_aware(1, demo_row)
        assert result == feasible_end

    def test_dates_outside_band_excluded_inclusively(self) -> None:
        """A date one day before feasible_start is rejected."""
        cv = _make_converter()
        eligeff = _ts("2019-01-01")
        feasible_start = eligeff + timedelta(days=ENROLLMENT_PRE_DAYS)
        cv._med_by_pat[1] = pd.DataFrame({"medication_date": [feasible_start - timedelta(days=1)]})
        demo_row = _demo_row("2019-01-01", "2023-01-01")
        assert cv._derive_index_date_feasibility_aware(1, demo_row) is None


# --------------------------------------------------------------------------- #
# Integration: smart-index fallback in _build_cohort                          #
# --------------------------------------------------------------------------- #


class TestSmartIndexFallbackIntegration:
    """End-to-end integration tests over a tiny synthetic fixture covering
    each fallback path. These exercise the wiring in ``_build_cohort`` so a
    regression in either direction (fallback never fires / fallback fires
    when it shouldn't) is caught."""

    def _build_fixture_converter(self) -> OptumDataConverter:
        cv = _make_converter()

        # 4 patients:
        # pat 1: existing cohort (Pass 1 finds in-band L50.x admit)
        # pat 2: smart-index target (Pass 1 picks early date that fails
        #        enrollment, fallback finds later in-band claim)
        # pat 3: smart-index target via second inpatient L50.x in band
        # pat 4: still drops (no anchor anywhere in feasible band)
        cv.demo = pd.DataFrame(
            [
                {
                    "patid": 1,
                    "age": 35,
                    "continuous_enrollment": 1,
                    "eligeff": _ts("2019-01-01"),
                    "eligend": _ts("2023-01-01"),
                    "diagcode_raw": "L509",
                    "diagcode": "L509",
                    "gdr_cd": "F",
                    "zipcode_5": "10001",
                },
                {
                    "patid": 2,
                    "age": 50,
                    "continuous_enrollment": 1,
                    "eligeff": _ts("2019-01-01"),
                    "eligend": _ts("2023-01-01"),
                    "diagcode_raw": "L509",
                    "diagcode": "L509",
                    "gdr_cd": "M",
                    "zipcode_5": "10002",
                },
                {
                    "patid": 3,
                    "age": 60,
                    "continuous_enrollment": 1,
                    "eligeff": _ts("2019-01-01"),
                    "eligend": _ts("2023-01-01"),
                    "diagcode_raw": "L509",
                    "diagcode": "L509",
                    "gdr_cd": "F",
                    "zipcode_5": "10003",
                },
                {
                    "patid": 4,
                    "age": 25,
                    "continuous_enrollment": 1,
                    "eligeff": _ts("2019-01-01"),
                    "eligend": _ts("2023-01-01"),
                    "diagcode_raw": "L509",
                    "diagcode": "L509",
                    "gdr_cd": "M",
                    "zipcode_5": "10004",
                },
            ]
        )

        # Patient 1 — in-band single inpatient L50.x: Pass 1 succeeds.
        cv._inpatient_by_pat[1] = pd.DataFrame(
            {
                "diag1": ["L509"],
                "diag2": [None],
                "diag3": [None],
                "diag4": [None],
                "diag5": [None],
                "admit_date": [_ts("2021-06-15")],
            }
        )
        # Patient 2 — early med pre-band (Pass 1) + late med in-band (fallback).
        # Pass 1 picks the earliest claim date (2018-06-01 — but eligeff filter
        # excludes it; the fallback chooses 2020-08-15).
        cv._med_by_pat[2] = pd.DataFrame(
            {
                "medication_date": [_ts("2019-01-15"), _ts("2020-08-15")],
            }
        )
        # Patient 3 — three inpatient L50.x: 1st+2nd pre-band, 3rd in-band.
        # Pass 1 picks the 2nd (priority-1 rule), which fails enrollment
        # window. Fallback restricts to in-band dates → only 2020-12-01
        # remains, returned as the single feasible inpatient anchor.
        cv._inpatient_by_pat[3] = pd.DataFrame(
            {
                "diag1": ["L509", "L509", "L509"],
                "diag2": [None, None, None],
                "diag3": [None, None, None],
                "diag4": [None, None, None],
                "diag5": [None, None, None],
                "admit_date": [
                    _ts("2018-06-01"),
                    _ts("2018-09-01"),
                    _ts("2020-12-01"),
                ],
            }
        )
        # Patient 4 — only out-of-band activity.
        cv._med_by_pat[4] = pd.DataFrame(
            {"medication_date": [_ts("2018-06-01"), _ts("2024-06-01")]}
        )

        return cv

    def test_existing_cohort_index_dates_unchanged(self) -> None:
        """Patient 1 (Pass-1-passes) keeps the original index date — fallback
        never fires for them."""
        cv = self._build_fixture_converter()
        journeys, _, _, _ = cv._build_cohort("initiation")
        pat1 = next((j for j in journeys if j["patient_id"] == "PAT_000000000001"), None)
        assert pat1 is not None
        # journey_start_date is an ISO string (rwd_common.safe_date).
        assert pat1["journey_start_date"] == "2021-06-15"

    def test_smart_index_grows_cohort_for_pass1_failures(self) -> None:
        cv = self._build_fixture_converter()
        journeys, _, _, _ = cv._build_cohort("initiation")
        ids = sorted(j["patient_id"] for j in journeys)
        # Patients 1, 2, 3 should pass; patient 4 still drops.
        assert ids == [
            "PAT_000000000001",
            "PAT_000000000002",
            "PAT_000000000003",
        ]

    def test_smart_index_fallback_attrition_row_emitted(self) -> None:
        cv = self._build_fixture_converter()
        cv._build_cohort("initiation")
        steps = dict(cv._attrition)
        assert "initiation: smart-index fallback hits" in steps
        # Patients 2 and 3 are the fallback targets.
        assert steps["initiation: smart-index fallback hits"] == 2

    def test_smart_index_picks_feasibility_aware_dates(self) -> None:
        cv = self._build_fixture_converter()
        journeys, _, _, _ = cv._build_cohort("initiation")
        by_id = {j["patient_id"]: j for j in journeys}
        # Patient 2: fallback picks 2020-08-15 (earliest med in band).
        assert by_id["PAT_000000000002"]["journey_start_date"] == "2020-08-15"
        # Patient 3: fallback picks 2020-12-01 (only L50.x in band).
        assert by_id["PAT_000000000003"]["journey_start_date"] == "2020-12-01"

    def test_smart_index_does_not_fire_for_discontinuation(self) -> None:
        cv = self._build_fixture_converter()
        # Need a biologic fill for B/C cohorts. Re-anchoring must NOT happen.
        cv._med_by_pat[2] = pd.DataFrame(
            {
                "medication_date": [_ts("2019-01-15"), _ts("2020-08-15")],
                "code": ["50242021501", "50242021501"],  # Xolair NDC
                "Brand_Name": ["XOLAIR", "XOLAIR"],
                "Generic_Name": ["omalizumab", "omalizumab"],
            }
        )
        cv._build_cohort("discontinuation")
        steps = dict(cv._attrition)
        # Disc/pers must NOT emit the smart-index fallback row.
        assert "discontinuation: smart-index fallback hits" not in steps

    def test_smart_index_does_not_fire_for_persistence(self) -> None:
        cv = self._build_fixture_converter()
        cv._med_by_pat[2] = pd.DataFrame(
            {
                "medication_date": [_ts("2020-08-15")],
                "code": ["50242021501"],
                "Brand_Name": ["XOLAIR"],
                "Generic_Name": ["omalizumab"],
            }
        )
        cv._build_cohort("persistence")
        steps = dict(cv._attrition)
        assert "persistence: smart-index fallback hits" not in steps
