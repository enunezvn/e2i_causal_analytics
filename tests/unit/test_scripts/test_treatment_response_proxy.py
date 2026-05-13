"""Unit tests for issue #157 PR C (Sub-PR-A): treatment_response CSU
claim-pattern proxies.

Covers `_derive_treatment_response` + supporting helpers
(`_classify_biologic_brand`, `_coverage_days`,
`_has_rescue_steroid_burst`, `_has_urticaria_ed_visit`) on the
OptumDataConverter class.

Each rule's boundary is exercised:
  * no biologic fill                       → NULL
  * single fill, days_sup < 60             → NULL (coverage pre-condition)
  * single fill, days_sup >= 60, no events → controlled
  * gap > 90d between fills                → discontinued
  * Xolair → Dupixent switch within 180d   → refractory
  * immunosuppressant addition             → refractory
  * rescue steroid burst (>=5 days)        → inadequate
  * urticaria ED visit (POS=23, L50.x)     → inadequate
  * first-match-wins ordering              → discontinued precedes refractory
  * outcome_indicator mapping              → improved/worsened/stable
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.convert_optum_rwd import (
    ED_CSU_DX_PREFIXES,
    ED_POS_CODE,
    RESCUE_STEROID_GENERICS,
    RESCUE_STEROID_MIN_DAYS_SUP,
    TREATMENT_RESPONSE_MIN_COVERAGE_DAYS,
    TREATMENT_RESPONSE_MIN_FOLLOWUP_DAYS,
    TREATMENT_RESPONSE_TO_OUTCOME,
    TREATMENT_RESPONSE_VOCAB,
    TREATMENT_RESPONSE_WINDOW_DAYS,
    OptumDataConverter,
)


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s)


def _converter() -> OptumDataConverter:
    return OptumDataConverter(
        parquet_dir=Path("."),
        output_dir=Path("."),
        cohorts=("discontinuation",),
    )


# --------------------------------------------------------------------------- #
# Constant invariants                                                         #
# --------------------------------------------------------------------------- #


class TestVocabInvariants:
    def test_vocab_matches_check_constraint(self) -> None:
        """Vocab must mirror migration 037 CHECK constraint values."""
        assert TREATMENT_RESPONSE_VOCAB == frozenset(
            {"controlled", "inadequate", "uncontrolled", "refractory", "discontinued"}
        )

    def test_outcome_mapping_covers_vocab(self) -> None:
        # `uncontrolled` is reserved for non-Optum cohorts (UAS7) but is
        # in the schema vocabulary, so the mapping must include it.
        for v in TREATMENT_RESPONSE_VOCAB:
            assert v in TREATMENT_RESPONSE_TO_OUTCOME, v

    def test_window_constants(self) -> None:
        assert TREATMENT_RESPONSE_MIN_COVERAGE_DAYS == 60
        assert TREATMENT_RESPONSE_MIN_FOLLOWUP_DAYS == 90
        assert TREATMENT_RESPONSE_WINDOW_DAYS == 180

    def test_rescue_constants(self) -> None:
        assert RESCUE_STEROID_GENERICS == ("prednisone", "methylprednisolone")
        assert RESCUE_STEROID_MIN_DAYS_SUP == 5
        assert ED_POS_CODE == "23"
        # L50 (urticaria) + T783 (angioedema, dot-stripped form of T78.3).
        assert "L50" in ED_CSU_DX_PREFIXES
        assert "T783" in ED_CSU_DX_PREFIXES


# --------------------------------------------------------------------------- #
# _classify_biologic_brand                                                    #
# --------------------------------------------------------------------------- #


class TestClassifyBiologicBrand:
    def test_xolair_via_brand_name(self) -> None:
        row = pd.Series({"Brand_Name": "Xolair", "code": None, "Generic_Name": None})
        assert OptumDataConverter._classify_biologic_brand(row) == "xolair"

    def test_dupixent_via_brand_name(self) -> None:
        row = pd.Series({"Brand_Name": "DUPIXENT", "code": None, "Generic_Name": None})
        assert OptumDataConverter._classify_biologic_brand(row) == "dupixent"

    def test_omalizumab_via_generic(self) -> None:
        row = pd.Series({"Brand_Name": None, "Generic_Name": "omalizumab", "code": None})
        assert OptumDataConverter._classify_biologic_brand(row) == "xolair"

    def test_dupilumab_via_generic(self) -> None:
        row = pd.Series({"Brand_Name": None, "Generic_Name": "dupilumab", "code": None})
        assert OptumDataConverter._classify_biologic_brand(row) == "dupixent"

    def test_xolair_via_ndc_prefix(self) -> None:
        row = pd.Series({"Brand_Name": None, "Generic_Name": None, "code": "50242-04-061"})
        assert OptumDataConverter._classify_biologic_brand(row) == "xolair"

    def test_xolair_via_hcpcs(self) -> None:
        row = pd.Series({"Brand_Name": None, "Generic_Name": None, "code": "J2357"})
        assert OptumDataConverter._classify_biologic_brand(row) == "xolair"

    def test_dupixent_via_ndc_prefix(self) -> None:
        row = pd.Series({"Brand_Name": None, "Generic_Name": None, "code": "00024-5912-02"})
        assert OptumDataConverter._classify_biologic_brand(row) == "dupixent"

    def test_dupixent_via_hcpcs_j0517(self) -> None:
        """J0517 is canonically eculizumab but the analyst spec assigns it
        to Dupixent — we follow the spec for consistency with
        `_csu_biologic_mask`."""
        row = pd.Series({"Brand_Name": None, "Generic_Name": None, "code": "J0517"})
        assert OptumDataConverter._classify_biologic_brand(row) == "dupixent"

    def test_unknown_returns_none(self) -> None:
        row = pd.Series({"Brand_Name": "Aspirin", "Generic_Name": None, "code": None})
        assert OptumDataConverter._classify_biologic_brand(row) is None


# --------------------------------------------------------------------------- #
# _coverage_days                                                              #
# --------------------------------------------------------------------------- #


class TestCoverageDays:
    def test_empty(self) -> None:
        conv = _converter()
        assert conv._coverage_days(pd.DataFrame(columns=["medication_date", "days_sup"])) == 0

    def test_single_fill(self) -> None:
        conv = _converter()
        df = pd.DataFrame([{"medication_date": _ts("2024-01-01"), "days_sup": 30}])
        assert conv._coverage_days(df) == 30

    def test_back_to_back_no_overlap(self) -> None:
        conv = _converter()
        df = pd.DataFrame(
            [
                {"medication_date": _ts("2024-01-01"), "days_sup": 30},
                {"medication_date": _ts("2024-02-01"), "days_sup": 30},
            ]
        )
        # Day-1 → Day-31, Day-32 → Day-62: 60 (overlap-free union has tiny
        # gap on the boundary but the merge keeps unioned segments contiguous
        # for adjacent fills since cur_end = start exactly).
        # Implementation: intervals [Jan1, Jan31), [Feb1, Mar2). gap = 1d
        # ⇒ no overlap ⇒ total = 30 + 30 = 60.
        assert conv._coverage_days(df) == 60

    def test_overlapping_fills_union(self) -> None:
        conv = _converter()
        df = pd.DataFrame(
            [
                {"medication_date": _ts("2024-01-01"), "days_sup": 60},
                {"medication_date": _ts("2024-02-01"), "days_sup": 60},
            ]
        )
        # Union = [Jan1, Apr1). 31 + 29 + 31 = 91 days.
        assert conv._coverage_days(df) == 91

    def test_zero_days_sup_skipped(self) -> None:
        conv = _converter()
        df = pd.DataFrame(
            [
                {"medication_date": _ts("2024-01-01"), "days_sup": 0},
                {"medication_date": _ts("2024-02-01"), "days_sup": 30},
            ]
        )
        assert conv._coverage_days(df) == 30


# --------------------------------------------------------------------------- #
# _has_rescue_steroid_burst                                                   #
# --------------------------------------------------------------------------- #


class TestRescueSteroidBurst:
    def test_no_meds(self) -> None:
        conv = _converter()
        conv._med_by_pat = {}
        assert (
            conv._has_rescue_steroid_burst(1, _ts("2024-01-01"), _ts("2024-06-30")) is False
        )

    def test_prednisone_long_course_in_window(self) -> None:
        conv = _converter()
        conv._med_by_pat = {
            1: pd.DataFrame(
                [
                    {
                        "medication_date": _ts("2024-02-01"),
                        "Generic_Name": "PREDNISONE",
                        "days_sup": 7,
                    }
                ]
            )
        }
        assert conv._has_rescue_steroid_burst(1, _ts("2024-01-01"), _ts("2024-06-30")) is True

    def test_prednisone_short_course_excluded(self) -> None:
        """days_sup=3 < RESCUE_STEROID_MIN_DAYS_SUP=5 → NOT a burst."""
        conv = _converter()
        conv._med_by_pat = {
            1: pd.DataFrame(
                [
                    {
                        "medication_date": _ts("2024-02-01"),
                        "Generic_Name": "prednisone",
                        "days_sup": 3,
                    }
                ]
            )
        }
        assert conv._has_rescue_steroid_burst(1, _ts("2024-01-01"), _ts("2024-06-30")) is False

    def test_methylprednisolone_counts(self) -> None:
        conv = _converter()
        conv._med_by_pat = {
            1: pd.DataFrame(
                [
                    {
                        "medication_date": _ts("2024-02-01"),
                        "Generic_Name": "methylprednisolone",
                        "days_sup": 5,
                    }
                ]
            )
        }
        assert conv._has_rescue_steroid_burst(1, _ts("2024-01-01"), _ts("2024-06-30")) is True

    def test_outside_window_excluded(self) -> None:
        conv = _converter()
        conv._med_by_pat = {
            1: pd.DataFrame(
                [
                    {
                        "medication_date": _ts("2023-12-01"),
                        "Generic_Name": "prednisone",
                        "days_sup": 7,
                    }
                ]
            )
        }
        assert conv._has_rescue_steroid_burst(1, _ts("2024-01-01"), _ts("2024-06-30")) is False

    def test_dexamethasone_not_a_burst(self) -> None:
        """Spec restricts burst signal to prednisone/methylprednisolone."""
        conv = _converter()
        conv._med_by_pat = {
            1: pd.DataFrame(
                [
                    {
                        "medication_date": _ts("2024-02-01"),
                        "Generic_Name": "dexamethasone",
                        "days_sup": 7,
                    }
                ]
            )
        }
        assert conv._has_rescue_steroid_burst(1, _ts("2024-01-01"), _ts("2024-06-30")) is False


# --------------------------------------------------------------------------- #
# _has_urticaria_ed_visit                                                     #
# --------------------------------------------------------------------------- #


class TestUrticariaEDVisit:
    def test_no_inpatient(self) -> None:
        conv = _converter()
        conv._inpatient_by_pat = {}
        assert conv._has_urticaria_ed_visit(1, _ts("2024-01-01"), _ts("2024-06-30")) is False

    def test_ed_pos_23_with_l50_dx_returns_true(self) -> None:
        conv = _converter()
        conv._inpatient_by_pat = {
            1: pd.DataFrame(
                [
                    {
                        "admit_date": _ts("2024-03-15"),
                        "pos": "23",
                        "diag1": "L509",
                        "diag2": None,
                        "diag3": None,
                        "diag4": None,
                        "diag5": None,
                    }
                ]
            )
        }
        assert conv._has_urticaria_ed_visit(1, _ts("2024-01-01"), _ts("2024-06-30")) is True

    def test_ed_pos_23_with_t783_dx_returns_true(self) -> None:
        conv = _converter()
        conv._inpatient_by_pat = {
            1: pd.DataFrame(
                [
                    {
                        "admit_date": _ts("2024-03-15"),
                        "pos": "23",
                        "diag1": "T783",
                        "diag2": None,
                        "diag3": None,
                        "diag4": None,
                        "diag5": None,
                    }
                ]
            )
        }
        assert conv._has_urticaria_ed_visit(1, _ts("2024-01-01"), _ts("2024-06-30")) is True

    def test_pos_21_inpatient_not_ed(self) -> None:
        """POS=21 (inpatient hospital) is NOT an ED — only POS=23 counts."""
        conv = _converter()
        conv._inpatient_by_pat = {
            1: pd.DataFrame(
                [
                    {
                        "admit_date": _ts("2024-03-15"),
                        "pos": "21",
                        "diag1": "L509",
                        "diag2": None,
                        "diag3": None,
                        "diag4": None,
                        "diag5": None,
                    }
                ]
            )
        }
        assert conv._has_urticaria_ed_visit(1, _ts("2024-01-01"), _ts("2024-06-30")) is False

    def test_ed_visit_without_csu_dx_excluded(self) -> None:
        conv = _converter()
        conv._inpatient_by_pat = {
            1: pd.DataFrame(
                [
                    {
                        "admit_date": _ts("2024-03-15"),
                        "pos": "23",
                        "diag1": "I10",  # hypertension — irrelevant
                        "diag2": None,
                        "diag3": None,
                        "diag4": None,
                        "diag5": None,
                    }
                ]
            )
        }
        assert conv._has_urticaria_ed_visit(1, _ts("2024-01-01"), _ts("2024-06-30")) is False


# --------------------------------------------------------------------------- #
# _derive_treatment_response — end-to-end rules                               #
# --------------------------------------------------------------------------- #


class TestDeriveTreatmentResponse:
    def _setup(
        self, med_rows: list[dict] | None, inpatient_rows: list[dict] | None = None
    ) -> OptumDataConverter:
        conv = _converter()
        conv._med_by_pat = (
            {1: pd.DataFrame(med_rows)} if med_rows else {}
        )
        conv._inpatient_by_pat = (
            {1: pd.DataFrame(inpatient_rows)} if inpatient_rows else {}
        )
        return conv

    # --- pre-condition: no biologic ---
    def test_no_biologic_fill_returns_null(self) -> None:
        conv = self._setup(None)
        assert conv._derive_treatment_response(1, _ts("2024-01-01")) == (None, None)

    def test_med_grp_present_but_no_biologic(self) -> None:
        conv = self._setup(
            [{"medication_date": _ts("2024-02-01"), "Generic_Name": "cetirizine", "days_sup": 30,
              "Brand_Name": None, "code": None}]
        )
        assert conv._derive_treatment_response(1, _ts("2024-01-01")) == (None, None)

    # --- pre-condition: insufficient coverage ---
    def test_single_short_fill_under_60d_returns_null(self) -> None:
        """A single fill with days_sup=28 cannot satisfy the 60d coverage
        pre-condition → NULL."""
        conv = self._setup(
            [
                {
                    "medication_date": _ts("2024-02-01"),
                    "Brand_Name": "Xolair",
                    "Generic_Name": "omalizumab",
                    "code": "50242040",
                    "days_sup": 28,
                }
            ]
        )
        assert conv._derive_treatment_response(1, _ts("2024-02-01")) == (None, None)

    # --- discontinued ---
    def test_gap_over_90d_returns_discontinued(self) -> None:
        conv = self._setup(
            [
                {
                    "medication_date": _ts("2024-02-01"),
                    "Brand_Name": "Xolair",
                    "Generic_Name": "omalizumab",
                    "code": "50242040",
                    "days_sup": 60,
                },
                # Next fill 5 months later — gap > 90d.
                {
                    "medication_date": _ts("2024-07-01"),
                    "Brand_Name": "Xolair",
                    "Generic_Name": "omalizumab",
                    "code": "50242040",
                    "days_sup": 60,
                },
            ]
        )
        resp, oc = conv._derive_treatment_response(1, _ts("2024-02-01"))
        assert resp == "discontinued"
        # No subsequent fill outside [init, init+180] (both are inside)
        # → outcome = 'worsened'.
        assert oc == "worsened"

    def test_discontinued_with_subsequent_fill_outside_window_stable(self) -> None:
        conv = self._setup(
            [
                {
                    "medication_date": _ts("2024-02-01"),
                    "Brand_Name": "Xolair",
                    "Generic_Name": "omalizumab",
                    "code": "50242040",
                    "days_sup": 60,
                },
                {
                    "medication_date": _ts("2024-07-01"),
                    "Brand_Name": "Xolair",
                    "Generic_Name": "omalizumab",
                    "code": "50242040",
                    "days_sup": 60,
                },
                # > 180d after init → outside window, signals re-engagement.
                {
                    "medication_date": _ts("2024-12-01"),
                    "Brand_Name": "Xolair",
                    "Generic_Name": "omalizumab",
                    "code": "50242040",
                    "days_sup": 60,
                },
            ]
        )
        resp, oc = conv._derive_treatment_response(1, _ts("2024-02-01"))
        assert resp == "discontinued"
        assert oc == "stable"

    # --- refractory: switch ---
    def test_switch_xolair_to_dupixent_returns_refractory(self) -> None:
        conv = self._setup(
            [
                {
                    "medication_date": _ts("2024-02-01"),
                    "Brand_Name": "Xolair",
                    "Generic_Name": "omalizumab",
                    "code": "50242040",
                    "days_sup": 60,
                },
                {
                    "medication_date": _ts("2024-03-15"),
                    "Brand_Name": "Dupixent",
                    "Generic_Name": "dupilumab",
                    "code": "0024591",
                    "days_sup": 30,
                },
            ]
        )
        resp, oc = conv._derive_treatment_response(1, _ts("2024-02-01"))
        assert resp == "refractory"
        assert oc == "worsened"

    # --- refractory: immunosuppressant addition ---
    def test_immunosuppressant_addition_returns_refractory(self) -> None:
        conv = self._setup(
            [
                {
                    "medication_date": _ts("2024-02-01"),
                    "Brand_Name": "Xolair",
                    "Generic_Name": "omalizumab",
                    "code": "50242040",
                    "days_sup": 60,
                },
                {
                    "medication_date": _ts("2024-02-15"),
                    "Brand_Name": None,
                    "Generic_Name": "cyclosporine",
                    "code": None,
                    "days_sup": 30,
                },
            ]
        )
        resp, oc = conv._derive_treatment_response(1, _ts("2024-02-01"))
        assert resp == "refractory"
        assert oc == "worsened"

    # --- inadequate: steroid burst ---
    def test_steroid_burst_within_window_returns_inadequate(self) -> None:
        conv = self._setup(
            [
                {
                    "medication_date": _ts("2024-02-01"),
                    "Brand_Name": "Xolair",
                    "Generic_Name": "omalizumab",
                    "code": "50242040",
                    "days_sup": 60,
                },
                {
                    "medication_date": _ts("2024-03-01"),
                    "Brand_Name": None,
                    "Generic_Name": "prednisone",
                    "code": None,
                    "days_sup": 7,
                },
            ]
        )
        resp, oc = conv._derive_treatment_response(1, _ts("2024-02-01"))
        assert resp == "inadequate"
        assert oc == "worsened"

    # --- inadequate: urticaria ED visit ---
    def test_urticaria_ed_visit_returns_inadequate(self) -> None:
        conv = self._setup(
            [
                {
                    "medication_date": _ts("2024-02-01"),
                    "Brand_Name": "Xolair",
                    "Generic_Name": "omalizumab",
                    "code": "50242040",
                    "days_sup": 60,
                }
            ],
            inpatient_rows=[
                {
                    "admit_date": _ts("2024-03-15"),
                    "pos": "23",
                    "diag1": "L509",
                    "diag2": None,
                    "diag3": None,
                    "diag4": None,
                    "diag5": None,
                }
            ],
        )
        resp, oc = conv._derive_treatment_response(1, _ts("2024-02-01"))
        assert resp == "inadequate"

    # --- controlled ---
    def test_persistence_no_rescue_returns_controlled(self) -> None:
        conv = self._setup(
            [
                {
                    "medication_date": _ts("2024-02-01"),
                    "Brand_Name": "Xolair",
                    "Generic_Name": "omalizumab",
                    "code": "50242040",
                    "days_sup": 60,
                },
                {
                    "medication_date": _ts("2024-04-01"),
                    "Brand_Name": "Xolair",
                    "Generic_Name": "omalizumab",
                    "code": "50242040",
                    "days_sup": 60,
                },
            ]
        )
        resp, oc = conv._derive_treatment_response(1, _ts("2024-02-01"))
        assert resp == "controlled"
        assert oc == "improved"

    # --- first-match-wins ordering: discontinued precedes refractory ---
    def test_first_match_discontinued_beats_refractory(self) -> None:
        """If a patient has BOTH a >90d gap AND a switch within the window,
        the discontinuation rule fires first (rule order)."""
        conv = self._setup(
            [
                {
                    "medication_date": _ts("2024-02-01"),
                    "Brand_Name": "Xolair",
                    "Generic_Name": "omalizumab",
                    "code": "50242040",
                    "days_sup": 60,
                },
                # 100d gap → discontinued.
                {
                    "medication_date": _ts("2024-07-15"),
                    "Brand_Name": "Dupixent",  # ALSO a switch
                    "Generic_Name": "dupilumab",
                    "code": "0024591",
                    "days_sup": 30,
                },
            ]
        )
        resp, oc = conv._derive_treatment_response(1, _ts("2024-02-01"))
        assert resp == "discontinued"

    # --- first-match-wins ordering: refractory precedes inadequate ---
    def test_first_match_refractory_beats_inadequate(self) -> None:
        """If a patient has BOTH a switch AND a steroid burst within
        coverage, the refractory rule fires first."""
        conv = self._setup(
            [
                {
                    "medication_date": _ts("2024-02-01"),
                    "Brand_Name": "Xolair",
                    "Generic_Name": "omalizumab",
                    "code": "50242040",
                    "days_sup": 60,
                },
                {
                    "medication_date": _ts("2024-03-01"),
                    "Brand_Name": "Dupixent",
                    "Generic_Name": "dupilumab",
                    "code": "0024591",
                    "days_sup": 30,
                },
                {
                    "medication_date": _ts("2024-03-15"),
                    "Brand_Name": None,
                    "Generic_Name": "prednisone",
                    "code": None,
                    "days_sup": 7,
                },
            ]
        )
        resp, oc = conv._derive_treatment_response(1, _ts("2024-02-01"))
        assert resp == "refractory"
