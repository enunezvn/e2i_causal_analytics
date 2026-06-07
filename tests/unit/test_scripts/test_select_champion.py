"""Unit tests for the #633 calibration-aware champion tiebreak.

``_select_champion`` (scripts/run_tier0_test.py) is the decision-critical
Step 5b seam: among the tier0 candidate models, it picks the champion that
gets deployed (and whose ``success_criteria_met`` the v3 gates judge). The
policy is two-stage:

1. never sacrifice discrimination — only the highest-AUC band wins; a
   candidate more than ``_AUC_TIE_BAND`` below the best AUC can never be
   selected over it;
2. among a genuine discrimination tie (within ``_AUC_TIE_BAND``), prefer the
   best deploy-calibrated calibration (lowest ``calibration_slope_deviation``).

These tests pin those invariants directly (the integration e2e exercises the
seam end-to-end but is expensive and does not construct an AUC-tied,
better-calibrated alternative). codex MEDIUM finding on PR #640.
"""

from __future__ import annotations

import math

import pytest

from scripts.run_tier0_test import (
    _AUC_TIE_BAND,
    _calibration_slope_deviation_of,
    _select_champion,
)


def _cand(name: str, auc: float, slope_dev: float | None) -> dict:
    return {"algorithm": name, "auc_roc": auc, "calibration_slope_deviation": slope_dev}


class TestSelectChampion:
    def test_large_auc_gap_ignores_calibration(self) -> None:
        """A clearly-higher-AUC model wins even with worse calibration: the gap
        exceeds the tie band, so it is NOT a discrimination tie."""
        hist = [
            _cand("high_auc_worse_cal", 0.90, 0.50),
            _cand("low_auc_better_cal", 0.80, 0.01),
        ]
        assert _select_champion(hist)["algorithm"] == "high_auc_worse_cal"

    def test_within_tie_band_prefers_better_calibration(self) -> None:
        """Among candidates within ``_AUC_TIE_BAND`` of the best AUC, the
        best-calibrated (lowest slope_deviation) wins — the #633 fix."""
        hist = [
            _cand("auc_max_worse_cal", 0.900, 0.21),
            _cand("auc_tied_best_cal", 0.896, 0.07),  # within 0.01 band, better cal
            _cand("auc_tied_mid_cal", 0.897, 0.18),
        ]
        assert _select_champion(hist)["algorithm"] == "auc_tied_best_cal"

    def test_tie_band_boundary_excludes_just_outside(self) -> None:
        """A candidate more than the band below best AUC is not in the tie set,
        so its superior calibration cannot win it the champion slot."""
        hist = [
            _cand("best_auc", 0.90, 0.40),
            _cand("just_outside_band", 0.90 - _AUC_TIE_BAND - 0.005, 0.001),
        ]
        assert _select_champion(hist)["algorithm"] == "best_auc"

    def test_all_missing_calibration_degrades_to_auc_argmax(self) -> None:
        """With no usable calibration on any candidate, the tiebreak degrades
        to the legacy AUC argmax (highest AUC wins via the -auc secondary key)."""
        hist = [
            {"algorithm": "a", "auc_roc": 0.900},  # no slope_dev key
            {"algorithm": "b", "auc_roc": 0.895},
        ]
        assert _select_champion(hist)["algorithm"] == "a"

    def test_single_candidate_returned(self) -> None:
        hist = [_cand("only", 0.83, 0.5)]
        assert _select_champion(hist)["algorithm"] == "only"

    # --- Deployability-aware selection (owner-ratified 2026-06-07) --------- #

    def test_deployable_lower_auc_beats_overfit_higher_auc(self) -> None:
        """An overfit/miscalibrated higher-AUC candidate is BLOCKED by the
        quality gates; a deployable (well-calibrated, not-overfit) lower-AUC
        candidate is preferred so the cohort actually ships. Mirrors the disc
        case: overfit LightGBM (0.635) vs deployable LR (0.610)."""
        hist = [
            {
                "algorithm": "overfit_gbm",
                "auc_roc": 0.635,
                "calibration_slope_deviation": 0.41,  # fails calibration gate
                "overfitting_severity": "moderate",  # fails overfit gate
            },
            {
                "algorithm": "deployable_lr",
                "auc_roc": 0.610,
                "calibration_slope_deviation": 0.06,  # passes
                "overfitting_severity": "none",  # passes
            },
        ]
        assert _select_champion(hist)["algorithm"] == "deployable_lr"

    def test_falls_back_to_auc_when_none_deployable(self) -> None:
        """If NO candidate passes the quality gates, the pool is all candidates
        and the legacy max-AUC (calibration-tiebroken) policy applies."""
        hist = [
            {
                "algorithm": "overfit_gbm",
                "auc_roc": 0.635,
                "calibration_slope_deviation": 0.41,
                "overfitting_severity": "severe",
            },
            {
                "algorithm": "miscal_lr",
                "auc_roc": 0.610,
                "calibration_slope_deviation": 0.40,
                "overfitting_severity": "mild",
            },
        ]
        assert _select_champion(hist)["algorithm"] == "overfit_gbm"

    def test_among_deployable_highest_auc_wins(self) -> None:
        """When several candidates are deployable, discrimination still decides."""
        hist = [
            {
                "algorithm": "deployable_hi",
                "auc_roc": 0.66,
                "calibration_slope_deviation": 0.10,
                "overfitting_severity": "none",
            },
            {
                "algorithm": "deployable_lo",
                "auc_roc": 0.61,
                "calibration_slope_deviation": 0.05,
                "overfitting_severity": "none",
            },
        ]
        assert _select_champion(hist)["algorithm"] == "deployable_hi"

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            _select_champion([])


class TestCalibrationSlopeDeviationOf:
    def test_valid_value_returned_abs(self) -> None:
        assert _calibration_slope_deviation_of(
            {"test_metrics": {"calibration_slope_deviation": 0.07}}
        ) == pytest.approx(0.07)

    @pytest.mark.parametrize(
        "result",
        [
            None,
            {},  # no test_metrics
            {"test_metrics": {}},  # no slope_dev key
            {"test_metrics": {"calibration_slope_deviation": None}},
            {"test_metrics": {"calibration_slope_deviation": float("nan")}},
            {"test_metrics": {"calibration_slope_deviation": float("inf")}},
            {"test_metrics": {"calibration_slope_deviation": "not-a-number"}},
            {"test_metrics": "not-a-dict"},
        ],
    )
    def test_missing_or_nonfinite_sorts_last(self, result: object) -> None:
        assert math.isinf(_calibration_slope_deviation_of(result))  # type: ignore[arg-type]
