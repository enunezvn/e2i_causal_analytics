"""Unit tests for issue #156 PR-1: items 4 (weighted DQS + cost fields),
5 (soft enrollment filter), 6 (payer_category 8-vocabulary + raw fields).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.convert_optum_rwd import (
    DEFAULT_MIN_DATA_QUALITY_SCORE,
    DQS_COST_FIELDS_FALLBACK,
    DQS_COST_FIELDS_PRIMARY,
    DQS_WEIGHT_COST,
    DQS_WEIGHT_DX,
    DQS_WEIGHT_ENROLL,
    DQS_WEIGHT_PROC,
    OptumDataConverter,
)
from scripts.rwd_common import (
    MEDICARE_ADVANTAGE_PRODUCT_CODES,
    PAYER_CATEGORY_VOCABULARY,
    derive_payer_category,
    is_truthy_flag,
)


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s)


def _converter(**kwargs: object) -> OptumDataConverter:
    return OptumDataConverter(
        parquet_dir=Path("."),
        output_dir=Path("."),
        cohorts=("initiation",),
        **kwargs,  # type: ignore[arg-type]
    )


# --------------------------------------------------------------------------- #
# Item 4: weighted DQS                                                        #
# --------------------------------------------------------------------------- #


class TestDqsWeightsInvariant:
    def test_weights_sum_to_one(self) -> None:
        s = DQS_WEIGHT_DX + DQS_WEIGHT_PROC + DQS_WEIGHT_COST + DQS_WEIGHT_ENROLL
        assert abs(s - 1.0) < 1e-9, f"weights sum to {s}, expected 1.0"

    def test_weights_match_issue_body(self) -> None:
        # Issue #156 item 4: 0.40 dx + 0.25 proc + 0.20 cost + 0.15 enroll
        assert DQS_WEIGHT_DX == 0.40
        assert DQS_WEIGHT_PROC == 0.25
        assert DQS_WEIGHT_COST == 0.20
        assert DQS_WEIGHT_ENROLL == 0.15


class TestDxComplete:
    def test_unk_treated_as_missing(self) -> None:
        row = pd.Series({"diag1": "UNK", "diag2": None})
        assert OptumDataConverter._dx_complete(row) == 0.0

    def test_non_null_dx_scores_one(self) -> None:
        row = pd.Series({"diag1": "I21", "diag2": None})
        assert OptumDataConverter._dx_complete(row) == 1.0

    def test_diagcode_satisfies(self) -> None:
        row = pd.Series({"diagcode": "L509"})
        assert OptumDataConverter._dx_complete(row) == 1.0

    def test_all_null_scores_zero(self) -> None:
        row = pd.Series({"diag1": None, "diag2": None, "diag3": None})
        assert OptumDataConverter._dx_complete(row) == 0.0

    def test_empty_string_treated_as_missing(self) -> None:
        row = pd.Series({"diag1": "", "diag2": "   "})
        assert OptumDataConverter._dx_complete(row) == 0.0


class TestProcComplete:
    def test_proc_code_present(self) -> None:
        row = pd.Series({"proc_code": "99213"})
        assert OptumDataConverter._proc_complete(row) == 1.0

    def test_proc_code_null(self) -> None:
        row = pd.Series({"proc_code": None})
        assert OptumDataConverter._proc_complete(row) == 0.0

    def test_proc_code_empty(self) -> None:
        row = pd.Series({"proc_code": ""})
        assert OptumDataConverter._proc_complete(row) == 0.0


class TestCostComplete:
    def test_std_cost_present_scores_one(self) -> None:
        row = pd.Series({"std_cost": 123.45})
        assert OptumDataConverter._cost_complete(row, is_pharmacy=False) == 1.0

    def test_fallback_field_scores_half(self) -> None:
        row = pd.Series({"std_cost": None, "copay": 10.0})
        assert OptumDataConverter._cost_complete(row, is_pharmacy=False) == 0.5

    def test_no_cost_fields_scores_zero(self) -> None:
        row = pd.Series({"diag1": "I21"})
        assert OptumDataConverter._cost_complete(row, is_pharmacy=False) == 0.0

    def test_charge_or_coins_or_deduct_also_fallback(self) -> None:
        for field in DQS_COST_FIELDS_FALLBACK:
            row = pd.Series({field: 50.0})
            assert OptumDataConverter._cost_complete(row, is_pharmacy=False) == 0.5, field

    def test_primary_field_is_std_cost(self) -> None:
        assert DQS_COST_FIELDS_PRIMARY == ("std_cost",)


class TestEnrollComplete:
    def test_fully_enrolled_scores_one(self) -> None:
        row = pd.Series(
            {"eligeff": _ts("2024-01-01"), "eligend": _ts("2024-12-31"), "continuous_enrollment": 1}
        )
        assert OptumDataConverter._enroll_complete(row) == 1.0

    def test_partial_enrollment_scores_half(self) -> None:
        row = pd.Series(
            {"eligeff": _ts("2024-01-01"), "eligend": _ts("2024-12-31"), "continuous_enrollment": 0}
        )
        assert OptumDataConverter._enroll_complete(row) == 0.5

    def test_missing_date_scores_zero(self) -> None:
        row = pd.Series({"eligeff": None, "eligend": _ts("2024-12-31"), "continuous_enrollment": 1})
        assert OptumDataConverter._enroll_complete(row) == 0.0


class TestComputeDataQualityScore:
    """End-to-end claim-level DQS rollup."""

    def test_empty_window_falls_back_to_feature_completeness(self) -> None:
        conv = _converter()
        conv._inpatient_by_pat = {}
        conv._proc_by_pat = {}
        conv._med_by_pat = {}
        demo_row = pd.Series(
            {"eligeff": _ts("2024-01-01"), "eligend": _ts("2024-12-31"), "continuous_enrollment": 1}
        )
        feats = {"a": 1, "b": 2, "c": None, "d": None}  # 50% complete
        score = conv._compute_data_quality_score(
            patid=1,
            lb_start=_ts("2024-01-01"),
            lb_end=_ts("2024-06-30"),
            demo_row=demo_row,
            feats=feats,
        )
        assert score == 0.5

    def test_single_inpatient_claim_with_dx_cost_enroll(self) -> None:
        """1 inpatient claim with dx=1, no proc, std_cost present, full enroll.
        Expected: 0.40*1 + 0.25*0 + 0.20*1 + 0.15*1 = 0.75."""
        conv = _converter()
        conv._inpatient_by_pat = {
            1: pd.DataFrame(
                [
                    {
                        "admit_date": _ts("2024-03-15"),
                        "diag1": "I21",
                        "std_cost": 5000.0,
                    }
                ]
            )
        }
        conv._proc_by_pat = {}
        conv._med_by_pat = {}
        demo_row = pd.Series(
            {"eligeff": _ts("2024-01-01"), "eligend": _ts("2024-12-31"), "continuous_enrollment": 1}
        )
        score = conv._compute_data_quality_score(
            patid=1,
            lb_start=_ts("2024-01-01"),
            lb_end=_ts("2024-06-30"),
            demo_row=demo_row,
            feats={},
        )
        assert score == 0.75


# --------------------------------------------------------------------------- #
# Item 5: soft enrollment filter                                              #
# --------------------------------------------------------------------------- #


class TestSoftEnrollmentFilter:
    def test_default_is_hard_filter(self) -> None:
        """Backwards-compat: default behavior preserves the hard filter."""
        conv = _converter()
        assert conv.soft_enrollment_filter is False

    def test_soft_filter_opt_in(self) -> None:
        conv = _converter(soft_enrollment_filter=True)
        assert conv.soft_enrollment_filter is True

    def test_min_data_quality_score_default(self) -> None:
        conv = _converter()
        assert conv.min_data_quality_score == DEFAULT_MIN_DATA_QUALITY_SCORE
        assert DEFAULT_MIN_DATA_QUALITY_SCORE == 0.50

    def test_min_data_quality_score_override(self) -> None:
        conv = _converter(min_data_quality_score=0.75)
        assert conv.min_data_quality_score == 0.75

    def test_min_data_quality_score_invalid_raises(self) -> None:
        with pytest.raises(ValueError, match="min_data_quality_score"):
            _converter(min_data_quality_score=-0.1)
        with pytest.raises(ValueError, match="min_data_quality_score"):
            _converter(min_data_quality_score=1.5)

    def test_strict_check_enrollment_window_requires_full_coverage(self) -> None:
        """Codex pass-1 HIGH-1 fix: in strict (default) mode, partial
        coverage MUST fail the enrollment-window check."""
        conv = _converter(soft_enrollment_filter=False, enrollment_regime="research")
        # Need: eligeff <= index - 180d AND eligend >= index + 90d
        idx = _ts("2024-06-01")
        # Eligibility only covers index-30d through index+10d — too narrow
        demo_row = pd.Series({"eligeff": _ts("2024-05-01"), "eligend": _ts("2024-06-11")})
        assert conv._check_enrollment_window(demo_row, idx) is False

    def test_soft_check_enrollment_window_accepts_partial(self) -> None:
        """Codex pass-1 HIGH-1 fix: in soft mode, partial coverage MUST
        pass the enrollment-window check (DQS gates downstream instead)."""
        conv = _converter(soft_enrollment_filter=True, enrollment_regime="research")
        idx = _ts("2024-06-01")
        # Same narrow window — soft mode accepts it
        demo_row = pd.Series({"eligeff": _ts("2024-05-01"), "eligend": _ts("2024-06-11")})
        assert conv._check_enrollment_window(demo_row, idx) is True

    def test_soft_check_enrollment_window_rejects_null_dates(self) -> None:
        """Soft mode still requires SOME eligibility signal; null dates fail."""
        conv = _converter(soft_enrollment_filter=True)
        idx = _ts("2024-06-01")
        for elig in [
            {"eligeff": None, "eligend": _ts("2024-12-31")},
            {"eligeff": _ts("2024-01-01"), "eligend": None},
            {"eligeff": None, "eligend": None},
        ]:
            demo_row = pd.Series(elig)
            assert conv._check_enrollment_window(demo_row, idx) is False, elig


class TestCliFlagWiring:
    """Codex pass-2 MEDIUM: item 5 must be reachable from the converter CLI."""

    def test_main_parser_accepts_soft_enrollment_filter_flag(self) -> None:
        """Argparse must register --soft-enrollment-filter as a store_true flag."""
        from scripts import convert_optum_rwd as mod

        # Inspect the source as a regression guard — the CLI wires through the
        # `--soft-enrollment-filter` flag and passes it to OptumDataConverter.
        src = Path(mod.__file__).read_text()
        assert "--soft-enrollment-filter" in src
        assert "--min-data-quality-score" in src
        assert "--comorbidity-method" in src
        assert "soft_enrollment_filter=args.soft_enrollment_filter" in src
        assert "min_data_quality_score=args.min_data_quality_score" in src
        assert "comorbidity_method=args.comorbidity_method" in src


# --------------------------------------------------------------------------- #
# Item 6: payer_category 8-vocabulary                                         #
# --------------------------------------------------------------------------- #


class TestPayerCategoryVocabulary:
    def test_exactly_eight_values(self) -> None:
        assert len(PAYER_CATEGORY_VOCABULARY) == 8

    def test_contains_expected_set(self) -> None:
        assert set(PAYER_CATEGORY_VOCABULARY) == {
            "commercial",
            "commercial_exchange",
            "medicare",
            "medicare_advantage",
            "medicare_lis_dual",
            "medicaid",
            "cash",
            "other",
        }

    def test_medicare_advantage_product_codes(self) -> None:
        # MA + MAPD per Optum + CMS convention
        assert "MAPD" in MEDICARE_ADVANTAGE_PRODUCT_CODES
        assert "MA" in MEDICARE_ADVANTAGE_PRODUCT_CODES


class TestDerivePayerCategory:
    """Each row in the priority table is asserted."""

    def test_com_health_exch(self) -> None:
        assert derive_payer_category("COM", None, "Y", None) == "commercial_exchange"
        assert derive_payer_category("COM", None, 1, None) == "commercial_exchange"
        assert derive_payer_category("COM", None, True, None) == "commercial_exchange"

    def test_com_default(self) -> None:
        assert derive_payer_category("COM", None, None, None) == "commercial"
        assert derive_payer_category("COM", None, "N", None) == "commercial"
        assert derive_payer_category("COM", None, 0, None) == "commercial"

    def test_mcr_lis_dual(self) -> None:
        # LIS-dual takes priority over MA in the issue's spec.
        assert derive_payer_category("MCR", "MAPD", None, 1) == "medicare_lis_dual"
        assert derive_payer_category("MCR", None, None, "Y") == "medicare_lis_dual"

    def test_mcr_medicare_advantage(self) -> None:
        assert derive_payer_category("MCR", "MAPD", None, None) == "medicare_advantage"
        assert derive_payer_category("MCR", "MA", None, None) == "medicare_advantage"
        # Not a recognized MA product code
        assert derive_payer_category("MCR", "PPO", None, None) == "medicare"

    def test_mcd(self) -> None:
        assert derive_payer_category("MCD", None, None, None) == "medicaid"

    def test_cash(self) -> None:
        assert derive_payer_category("CASH", None, None, None) == "cash"

    def test_other_for_unknown(self) -> None:
        assert derive_payer_category("XYZ", None, None, None) == "other"
        assert derive_payer_category(None, None, None, None) == "other"
        assert derive_payer_category("", None, None, None) == "other"

    def test_case_insensitive_bus(self) -> None:
        assert derive_payer_category("com", None, None, None) == "commercial"
        assert derive_payer_category("mcr", "mapd", None, None) == "medicare_advantage"

    def test_priority_order_short_circuits(self) -> None:
        """Per issue body: priority is COM+exch > COM > MCR+LIS > MCR+MA > MCR.
        Health_exch on a MCR row must NOT push it to commercial_exchange.
        """
        # MCR + health_exch=Y → still medicare (not commercial_exchange)
        assert derive_payer_category("MCR", None, "Y", None) == "medicare"

    def test_all_eight_values_reachable(self) -> None:
        """Every vocabulary value must be reachable via a valid derivation."""
        cases = [
            ("COM", None, "Y", None, "commercial_exchange"),
            ("COM", None, None, None, "commercial"),
            ("MCR", None, None, "Y", "medicare_lis_dual"),
            ("MCR", "MAPD", None, None, "medicare_advantage"),
            ("MCR", "PPO", None, None, "medicare"),
            ("MCD", None, None, None, "medicaid"),
            ("CASH", None, None, None, "cash"),
            ("XYZ", None, None, None, "other"),
        ]
        observed = {derive_payer_category(*case[:-1]) for case in cases}
        assert observed == set(PAYER_CATEGORY_VOCABULARY)


class TestIsTruthyFlag:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (None, False),
            (True, True),
            (False, False),
            (1, True),
            (0, False),
            (1.0, True),
            ("Y", True),
            ("y", True),
            ("YES", True),
            ("1", True),
            ("true", True),
            ("True", True),
            ("T", True),
            ("N", False),
            ("0", False),
            ("", False),
        ],
    )
    def test_coercion(self, value: object, expected: bool) -> None:
        assert is_truthy_flag(value) is expected

    def test_pandas_nan_is_false(self) -> None:
        assert is_truthy_flag(float("nan")) is False
