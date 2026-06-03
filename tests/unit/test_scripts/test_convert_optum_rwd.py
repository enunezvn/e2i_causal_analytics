"""Unit tests for the smart-index fallback added to ``scripts/convert_optum_rwd.py``
to close backlog #19 (Optum cohort-build attrition over-filtering).

The fallback only fires for cohort A (initiation). Cohorts B/C re-anchor on
first biologic fill — re-anchoring there changes cohort semantics
(disc/pers measure outcomes 180d post-FIRST-fill).
"""

from __future__ import annotations

from datetime import timedelta

import pandas as pd
import pytest

from scripts.convert_optum_rwd import (
    CSU_LABS_LOINC,
    DEFAULT_ENROLLMENT_REGIME,
    ENROLLMENT_POST_DAYS,
    ENROLLMENT_PRE_DAYS,
    ENROLLMENT_REGIMES,
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
        # 539d total enrollment: feasible_start = eligeff + 360 = D+360,
        # feasible_end = eligend - 180 = D+359 → band is empty and the
        # method returns None. At exactly 540d (PRE+POST), feasible_start
        # equals feasible_end, producing a valid single-point band — the
        # `feasible_start > feasible_end` guard uses strict > and does not
        # block that case (covered by a separate test).
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

    def test_single_day_feasibility_band_is_valid(self) -> None:
        """Codex pass-1 LOW-4: at exactly PRE+POST days enrollment,
        feasible_start == feasible_end. The strict ``>`` guard treats this
        as a valid single-point band; an event landing exactly on that day
        must qualify."""
        cv = _make_converter()
        eligeff = _ts("2020-01-01")
        eligend = eligeff + timedelta(days=ENROLLMENT_PRE_DAYS + ENROLLMENT_POST_DAYS)
        feasible_point = eligeff + timedelta(days=ENROLLMENT_PRE_DAYS)
        assert feasible_point == eligend - timedelta(days=ENROLLMENT_POST_DAYS)
        cv._med_by_pat[1] = pd.DataFrame({"medication_date": [feasible_point]})
        demo_row = pd.Series(
            {
                "patid": 1,
                "eligeff": eligeff,
                "eligend": eligend,
                "diagcode_raw": "L509",
            }
        )
        result = cv._derive_index_date_feasibility_aware(1, demo_row)
        assert result == feasible_point

    def test_three_in_band_inpatient_dates_picks_second_not_third(
        self,
    ) -> None:
        """Codex pass-1 LOW-5: with 3+ in-band inpatient L50.x dates, the
        rule must select ``ip_feasible[1]`` (chronologically 2nd), not the
        last or any other. Guards against an off-by-one regression that
        would silently pass when there are exactly 2 in-band dates."""
        cv = _make_converter()
        cv._inpatient_by_pat[1] = pd.DataFrame(
            {
                "diag1": ["L509", "L509", "L509"],
                "diag2": [None, None, None],
                "diag3": [None, None, None],
                "diag4": [None, None, None],
                "diag5": [None, None, None],
                "admit_date": [
                    _ts("2020-03-15"),  # 1st in-band
                    _ts("2020-09-15"),  # 2nd in-band — must be selected
                    _ts("2021-06-15"),  # 3rd in-band — must NOT be selected
                ],
            }
        )
        demo_row = _demo_row("2019-01-01", "2023-01-01")
        result = cv._derive_index_date_feasibility_aware(1, demo_row)
        assert result == _ts("2020-09-15")
        assert result != _ts("2021-06-15")


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

    def test_smart_index_counter_excludes_washout_dropouts(self) -> None:
        """Codex pass-1 LOW-3 + MEDIUM-1: when a fallback-rescued patient
        later trips ``_had_biologic_pre_index`` (30d washout), the
        ``smart-index fallback hits`` row must NOT count them. The counter
        and ``records_pass`` should share the same gate.

        Setup: patient 2 gets re-fixtured so that
          - Pass 1 picks the 2nd of 2 out-of-band inpatient L50.x dates
            (fails enrollment_window),
          - Smart-index picks the 3rd inpatient L50.x date which lies in-
            band (priority 1, NOT the claim fallback — so a biologic fill
            within 30d of the smart index does not become the smart index
            itself),
          - A Xolair fill 14 days before the smart index trips
            ``_had_biologic_pre_index`` and drops the patient.
        """
        cv = self._build_fixture_converter()
        # Patient 2: rebuild as inpatient-anchored to keep biologic-fill
        # out of the index-date pool.
        cv._inpatient_by_pat[2] = pd.DataFrame(
            {
                "diag1": ["L509", "L509", "L509"],
                "diag2": [None, None, None],
                "diag3": [None, None, None],
                "diag4": [None, None, None],
                "diag5": [None, None, None],
                "admit_date": [
                    _ts("2018-06-01"),  # out-of-band — Pass 1's 1st
                    _ts("2018-09-01"),  # out-of-band — Pass 1's 2nd, fails
                    _ts("2020-12-15"),  # in-band — smart-index's pick
                ],
            }
        )
        cv._med_by_pat[2] = pd.DataFrame(
            {
                "medication_date": [_ts("2020-12-01")],  # 14d pre-index
                "code": ["50242021501"],  # Xolair NDC
                "Brand_Name": ["XOLAIR"],
                "Generic_Name": ["omalizumab"],
            }
        )

        journeys, _, _, _ = cv._build_cohort("initiation")
        steps = dict(cv._attrition)

        ids = sorted(j["patient_id"] for j in journeys)
        # Patient 2 is rescued by smart-index but then dropped by washout.
        assert "PAT_000000000002" not in ids
        # Patient 3 still passes via smart-index → counter == 1 (not 2).
        assert steps["initiation: smart-index fallback hits"] == 1
        # And the neighbor row reflects the same drop (3 fixture passers
        # minus pat 2 washout = 2).
        assert steps["initiation: after index + enrollment + exclusions"] == 2

    def test_attrition_log_pins_enrollment_regime_label(self) -> None:
        """Plan v3 §3 Tier 1A: the first attrition row of every cohort build
        must encode which enrollment regime produced the artifact, so a
        downstream reader of attrition_report.csv can audit the regime
        without rerunning the converter."""
        cv = self._build_fixture_converter()
        cv._build_cohort("initiation")
        steps = dict(cv._attrition)
        regime_key = (
            f"initiation: enrollment_regime={cv.enrollment_regime} "
            f"(pre={cv.enrollment_pre_days}d, post={cv.enrollment_post_days}d)"
        )
        assert regime_key in steps
        # Regime row count must equal cohort start size (the row pins the
        # all-patids count, before any filter has been applied).
        assert steps[regime_key] == steps["initiation: start"]


# --------------------------------------------------------------------------- #
# Tier 1A enrollment-regime bifurcation (plan v3 §3)                          #
# --------------------------------------------------------------------------- #


class TestEnrollmentRegimeBifurcation:
    """The OptumDataConverter exposes a two-regime enrollment-window switch
    (production=360/180, research=180/90) per plan v3 §3 Tier 1A. The default
    is production; research is opt-in and must be flagged for downstream
    consumers via converter attribute + attrition log."""

    def test_default_regime_constant_is_production(self) -> None:
        assert DEFAULT_ENROLLMENT_REGIME == "production"

    def test_module_constants_match_production_regime(self) -> None:
        """Module-level ENROLLMENT_PRE_DAYS / ENROLLMENT_POST_DAYS preserve
        the production-regime values for any external caller importing them."""
        assert ENROLLMENT_PRE_DAYS == ENROLLMENT_REGIMES["production"]["pre_days"]
        assert ENROLLMENT_POST_DAYS == ENROLLMENT_REGIMES["production"]["post_days"]
        assert ENROLLMENT_PRE_DAYS == 360
        assert ENROLLMENT_POST_DAYS == 180

    def test_regime_table_pins_research_to_180_90(self) -> None:
        """Plan v3 §3 Tier 1A 'MAYBE' branch literal numbers."""
        assert ENROLLMENT_REGIMES["research"]["pre_days"] == 180
        assert ENROLLMENT_REGIMES["research"]["post_days"] == 90

    def test_default_constructor_uses_production(self) -> None:
        cv = _make_converter()
        assert cv.enrollment_regime == "production"
        assert cv.enrollment_pre_days == 360
        assert cv.enrollment_post_days == 180

    def test_explicit_production_regime(self) -> None:
        cv = OptumDataConverter(
            parquet_dir=".",
            output_dir=".",
            cohorts=("initiation",),
            enrollment_regime="production",
        )
        assert cv.enrollment_pre_days == 360
        assert cv.enrollment_post_days == 180

    def test_research_regime(self) -> None:
        cv = OptumDataConverter(
            parquet_dir=".",
            output_dir=".",
            cohorts=("initiation",),
            enrollment_regime="research",
        )
        assert cv.enrollment_regime == "research"
        assert cv.enrollment_pre_days == 180
        assert cv.enrollment_post_days == 90

    def test_invalid_regime_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="enrollment_regime"):
            OptumDataConverter(
                parquet_dir=".",
                output_dir=".",
                cohorts=("initiation",),
                enrollment_regime="aggressive",
            )

    def test_research_regime_check_enrollment_window_accepts_shorter_continuous_enrollment(
        self,
    ) -> None:
        """A patient with exactly 270 days continuous enrollment around index
        passes the research regime (180+90) but fails production (360+180)."""
        cv_prod = OptumDataConverter(
            parquet_dir=".",
            output_dir=".",
            cohorts=("initiation",),
            enrollment_regime="production",
        )
        cv_rsch = OptumDataConverter(
            parquet_dir=".",
            output_dir=".",
            cohorts=("initiation",),
            enrollment_regime="research",
        )
        index_date = _ts("2021-06-01")
        # 270d centered on index_date: eligeff = idx-180, eligend = idx+90.
        demo_row = pd.Series(
            {
                "patid": 1,
                "eligeff": index_date - timedelta(days=180),
                "eligend": index_date + timedelta(days=90),
                "diagcode_raw": "L509",
            }
        )
        # Production needs 360d pre + 180d post = 540d total → fails.
        assert cv_prod._check_enrollment_window(demo_row, index_date) is False
        # Research needs 180d pre + 90d post = 270d total → passes (boundary).
        assert cv_rsch._check_enrollment_window(demo_row, index_date) is True

    def test_research_regime_smart_index_fallback_uses_relaxed_band(self) -> None:
        """A med date that falls inside the research-regime feasibility band
        but outside the production-regime band must be selected by the
        research-regime fallback only."""
        # Feasibility band depends on (eligeff + pre_days, eligend - post_days).
        # eligeff=2020-01-01, eligend=2020-12-31 (366d):
        #   production: [2020-12-26, 2020-07-04] → empty (start > end) → None.
        #   research:   [2020-06-29, 2020-10-02] → valid 95d band.
        eligeff = _ts("2020-01-01")
        eligend = _ts("2020-12-31")
        in_research_band_only = _ts("2020-08-15")  # inside 180/90, outside 360/180

        cv_prod = OptumDataConverter(
            parquet_dir=".",
            output_dir=".",
            cohorts=("initiation",),
            enrollment_regime="production",
        )
        cv_rsch = OptumDataConverter(
            parquet_dir=".",
            output_dir=".",
            cohorts=("initiation",),
            enrollment_regime="research",
        )
        for cv in (cv_prod, cv_rsch):
            cv._med_by_pat[1] = pd.DataFrame({"medication_date": [in_research_band_only]})

        demo_row = pd.Series(
            {
                "patid": 1,
                "eligeff": eligeff,
                "eligend": eligend,
                "diagcode_raw": "L509",
            }
        )
        # Production: feasibility band is empty (start > end), fallback returns None.
        assert cv_prod._derive_index_date_feasibility_aware(1, demo_row) is None
        # Research: the date qualifies.
        assert cv_rsch._derive_index_date_feasibility_aware(1, demo_row) == in_research_band_only

    def test_research_regime_attrition_log_pins_research_label(self) -> None:
        """The regime label row at the head of every cohort build encodes
        the active regime so an attrition_report.csv reader can audit."""
        cv = OptumDataConverter(
            parquet_dir=".",
            output_dir=".",
            cohorts=("initiation",),
            enrollment_regime="research",
        )
        # Stub the demographics input so _build_cohort emits the regime row.
        cv.demo = pd.DataFrame(
            [
                {
                    "patid": 999,
                    "age": 30,
                    "continuous_enrollment": 0,  # filter drops everyone, but
                    "eligeff": _ts("2020-01-01"),  # the regime row still emits.
                    "eligend": _ts("2020-12-31"),
                    "diagcode_raw": "L509",
                    "diagcode": "L509",
                    "gdr_cd": "F",
                    "zipcode_5": "10001",
                }
            ]
        )
        cv._build_cohort("initiation")
        steps = dict(cv._attrition)
        regime_key = "initiation: enrollment_regime=research (pre=180d, post=90d)"
        assert regime_key in steps


class TestCsuLabsLoincMapping:
    """Regression guard for the ``CSU_LABS_LOINC`` analyte -> LOINC map.

    A 2026-06-03 forensic trace of the Optum extract found three entries pointing
    at the WRONG analyte (verified against the raw ``lab.parquet`` ``tst_desc``):

      - ``eosinophil`` -> ``6206-7``  is actually Peanut IgE ('F013-IGE PEANUT')
      - ``tpo_ab``     -> ``3051-0``/``3053-6`` are actually Free / Total T3
      - ``cbc``        -> ``26453-1`` is actually RBC

    plus ``ana`` -> ``14741-9`` which has zero rows in the extract. The fix
    repoints each name at codes that genuinely identify that analyte (canonical
    LOINC + the variants observed in the Optum drop). These tests exercise the
    real ``_lab_features`` code path, not just the constant.
    """

    @staticmethod
    def _lab_row(loinc: str, day: str, rslt: float = 1.0, abnl: str = "") -> dict:
        return {"loinc_cd": loinc, "fst_dt": _ts(day), "rslt_nbr": rslt, "abnl_cd": abnl}

    def test_eosinophil_matches_true_count_not_peanut_ige(self) -> None:
        cv = _make_converter()
        lab_w = pd.DataFrame(
            [
                self._lab_row("711-2", "2020-01-01", rslt=0.4),  # absolute eosinophils
                self._lab_row("6206-7", "2020-02-01", rslt=99.0),  # peanut IgE — NOT eosinophil
            ]
        )
        tested, last, _ = cv._lab_features(lab_w, CSU_LABS_LOINC["eosinophil"])
        assert tested is True
        assert last == pytest.approx(0.4)  # the 711-2 value, not the peanut-IgE 99.0
        only_peanut = pd.DataFrame([self._lab_row("6206-7", "2020-01-01")])
        assert cv._lab_features(only_peanut, CSU_LABS_LOINC["eosinophil"])[0] is False

    def test_tpo_ab_matches_thyroid_peroxidase_not_t3(self) -> None:
        cv = _make_converter()
        lab_w = pd.DataFrame(
            [
                self._lab_row("8099-4", "2020-01-01", rslt=12.0),  # thyroid peroxidase Ab
                self._lab_row("3051-0", "2020-02-01", rslt=3.1),  # free T3 — NOT tpo_ab
            ]
        )
        tested, last, _ = cv._lab_features(lab_w, CSU_LABS_LOINC["tpo_ab"])
        assert tested is True
        assert last == pytest.approx(12.0)
        only_t3 = pd.DataFrame(
            [self._lab_row("3051-0", "2020-01-01"), self._lab_row("3053-6", "2020-01-02")]
        )
        assert cv._lab_features(only_t3, CSU_LABS_LOINC["tpo_ab"])[0] is False

    def test_cbc_matches_panel_not_rbc(self) -> None:
        cv = _make_converter()
        lab_w = pd.DataFrame(
            [
                self._lab_row("58410-2", "2020-01-01"),  # CBC panel
                self._lab_row("26453-1", "2020-02-01"),  # RBC — NOT cbc
            ]
        )
        assert cv._lab_features(lab_w, CSU_LABS_LOINC["cbc"])[0] is True
        only_rbc = pd.DataFrame([self._lab_row("26453-1", "2020-01-01")])
        assert cv._lab_features(only_rbc, CSU_LABS_LOINC["cbc"])[0] is False

    def test_ana_matches_real_antinuclear_codes(self) -> None:
        cv = _make_converter()
        lab_w = pd.DataFrame([self._lab_row("42254-3", "2020-01-01")])  # 'ANA SCREEN, IFA'
        assert cv._lab_features(lab_w, CSU_LABS_LOINC["ana"])[0] is True
        only_absent = pd.DataFrame([self._lab_row("14741-9", "2020-01-01")])
        assert cv._lab_features(only_absent, CSU_LABS_LOINC["ana"])[0] is False

    def test_verified_correct_entries_unchanged(self) -> None:
        # free_t4 / tsh / crp / ige_total were verified correct — must stay intact
        assert CSU_LABS_LOINC["free_t4"] == ("3024-7",)
        assert CSU_LABS_LOINC["tsh"] == ("3016-3",)
        assert "1988-5" in CSU_LABS_LOINC["crp"]
        assert "19113-0" in CSU_LABS_LOINC["ige_total"]

    def test_known_mislabeled_codes_purged_and_codes_unique(self) -> None:
        assert "6206-7" not in CSU_LABS_LOINC["eosinophil"]
        assert "3051-0" not in CSU_LABS_LOINC["tpo_ab"]
        assert "3053-6" not in CSU_LABS_LOINC["tpo_ab"]
        assert "26453-1" not in CSU_LABS_LOINC["cbc"]
        assert "14741-9" not in CSU_LABS_LOINC["ana"]
        # no analyte may claim a code that belongs to another analyte
        all_codes = [c for codes in CSU_LABS_LOINC.values() for c in codes]
        assert len(all_codes) == len(set(all_codes)), "LOINC codes must be unique across analytes"
