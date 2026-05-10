"""Unit tests for ``scripts/run_tier1b_b2_experiment.py``.

Pins the threshold constants (load-bearing per pre-spec memo §1) and
exercises the metric / aggregation / threshold-evaluation logic on
synthetic SeedResult fixtures so the harness's pass/fail decision is
testable without invoking the full FileIngestor + sklearn pipeline.

The end-to-end run on real cohort data is exercised by the CI
workflow ``tier1b_b2_experiment.yml`` against a tagged run; that path
is not duplicated here. These tests cover the deterministic helpers.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import run_tier1b_b2_experiment as H  # noqa: E402

# ---------------------------------------------------------------------------
# Threshold + lifecycle drift detector
# ---------------------------------------------------------------------------


class TestG2PreSpecConstantsDriftDetector:
    """If these constants drift without a refreshed pre-spec memo, the
    threshold-shopping invariant has been violated. v3 §8 forbids
    in-place edits to load-bearing thresholds."""

    def test_t1_delta_auc_min(self) -> None:
        assert H.G2_DELTA_AUC_MIN == 0.03, (
            "G2_DELTA_AUC_MIN drifted from the pre-spec memo's T1 "
            "value of 0.03. Per v3 §8, threshold edits require a NEW "
            "tier1b_b2_prespec_<date>.md memo at a fresh date."
        )

    def test_t2_ece_ratio_max(self) -> None:
        assert H.G2_ECE_RATIO_MAX == 0.5, (
            "G2_ECE_RATIO_MAX drifted from the pre-spec memo's T2 "
            "value of 0.5. See test_t1 for the resolution protocol."
        )

    def test_t3_cv_stability_ratio_max(self) -> None:
        assert H.G2_CV_STABILITY_RATIO_MAX == 0.7, (
            "G2_CV_STABILITY_RATIO_MAX drifted from the pre-spec "
            "memo's T3 value of 0.7. See test_t1 for the resolution "
            "protocol."
        )

    def test_seeds_locked(self) -> None:
        assert H.G2_SEEDS == (42, 43, 44, 45, 46), (
            "G2_SEEDS drifted from the pre-spec memo's locked list. "
            "See test_t1 for the resolution protocol."
        )

    def test_cv_folds_default(self) -> None:
        assert H.G2_CV_FOLDS == 5

    def test_ece_bins_default(self) -> None:
        assert H.G2_ECE_BINS == 10


def test_lifecycle_state_is_advisory() -> None:
    """G2 ships in ADVISORY until first green run lands."""
    from src.lifecycle import GateLifecycleState

    assert H.LIFECYCLE_STATE_G2 == GateLifecycleState.ADVISORY


def test_lifecycle_metadata_has_gate_name() -> None:
    """The metadata block must declare gate_name=G2 so the lifecycle
    scanner can attribute the declaration."""
    assert H.LIFECYCLE_METADATA_G2.get("gate_name") == "G2"


# ---------------------------------------------------------------------------
# Cohort registry
# ---------------------------------------------------------------------------


class TestCohortRegistry:
    def test_default_cohort_present(self) -> None:
        assert "optum_initiation_default" in H.COHORTS
        cohort = H.COHORTS["optum_initiation_default"]
        assert cohort.data_snooped is False

    def test_relaxed_cohort_marked_data_snooped(self) -> None:
        """Per pre-spec §2.2, the relaxed cohort is marked
        data_snooped=true (cannot be load-bearing)."""
        assert "optum_initiation_relaxed" in H.COHORTS
        cohort = H.COHORTS["optum_initiation_relaxed"]
        assert cohort.data_snooped is True

    def test_cohort_paths(self) -> None:
        assert H.COHORTS["optum_initiation_default"].data_dir == "data/rwd/optum/initiation"

    def test_cohort_target_columns(self) -> None:
        assert H.COHORTS["optum_initiation_default"].target == "treatment_initiated"


# ---------------------------------------------------------------------------
# _seed_mean — aggregator
# ---------------------------------------------------------------------------


class TestSeedMean:
    def test_returns_none_when_all_none(self) -> None:
        assert H._seed_mean([None, None, None]) is None

    def test_returns_mean_of_finite_values(self) -> None:
        assert H._seed_mean([0.5, 0.7, 0.6]) == pytest.approx(0.6, abs=1e-9)

    def test_skips_none_values(self) -> None:
        assert H._seed_mean([0.5, None, 0.7]) == pytest.approx(0.6, abs=1e-9)

    def test_skips_non_finite_values(self) -> None:
        assert H._seed_mean([0.5, float("nan"), 0.7]) == pytest.approx(0.6, abs=1e-9)

    def test_skips_inf_values(self) -> None:
        assert H._seed_mean([0.5, float("inf"), 0.7]) == pytest.approx(0.6, abs=1e-9)


# ---------------------------------------------------------------------------
# evaluate_t1 — held-out AUC lift
# ---------------------------------------------------------------------------


class TestEvaluateT1:
    def test_passes_when_lift_meets_threshold(self) -> None:
        result = H.evaluate_t1(
            [0.60, 0.61, 0.59, 0.60, 0.60],
            [0.65, 0.64, 0.66, 0.63, 0.65],
        )
        assert result.passes is True
        assert result.delta is not None and result.delta >= H.G2_DELTA_AUC_MIN

    def test_fails_when_lift_below_threshold(self) -> None:
        result = H.evaluate_t1(
            [0.60, 0.61, 0.59, 0.60, 0.60],
            [0.61, 0.62, 0.60, 0.61, 0.61],
        )
        assert result.passes is False
        assert result.delta is not None and result.delta < H.G2_DELTA_AUC_MIN

    def test_fails_on_negative_lift(self) -> None:
        result = H.evaluate_t1(
            [0.65, 0.64, 0.66, 0.63, 0.65],
            [0.60, 0.61, 0.59, 0.60, 0.60],
        )
        assert result.passes is False

    def test_fails_when_pre_or_post_missing(self) -> None:
        result = H.evaluate_t1([None, None], [0.60, 0.65])
        assert result.passes is False
        assert result.delta is None
        assert "missing pre or post AUC" in result.rationale

    def test_threshold_value_in_result(self) -> None:
        result = H.evaluate_t1([0.6], [0.65])
        assert result.threshold == H.G2_DELTA_AUC_MIN


# ---------------------------------------------------------------------------
# evaluate_t2 — ECE ratio
# ---------------------------------------------------------------------------


class TestEvaluateT2:
    def test_passes_when_ratio_below_threshold(self) -> None:
        # ECE post is 40% of pre → passes (< 50%)
        result = H.evaluate_t2([0.10, 0.10, 0.10], [0.04, 0.04, 0.04])
        assert result.passes is True
        assert result.delta is not None and result.delta < H.G2_ECE_RATIO_MAX

    def test_fails_when_ratio_above_threshold(self) -> None:
        result = H.evaluate_t2([0.10, 0.10, 0.10], [0.06, 0.06, 0.06])
        assert result.passes is False

    def test_fails_when_post_equals_pre(self) -> None:
        result = H.evaluate_t2([0.10, 0.10], [0.10, 0.10])
        assert result.passes is False
        assert result.delta == pytest.approx(1.0)

    def test_fails_on_degenerate_pre(self) -> None:
        """When baseline ECE is essentially zero, the ratio is
        undefined and T2 fails (not silently passes)."""
        result = H.evaluate_t2([0.0, 0.0], [0.001, 0.001])
        assert result.passes is False
        assert "degenerate baseline" in result.rationale

    def test_fails_when_pre_or_post_missing(self) -> None:
        result = H.evaluate_t2([None, None], [0.04, 0.04])
        assert result.passes is False


# ---------------------------------------------------------------------------
# evaluate_t3 — CV stability ratio
# ---------------------------------------------------------------------------


class TestEvaluateT3:
    def test_passes_when_ratio_below_threshold(self) -> None:
        # CV-stability post is 60% of pre → passes (< 70%)
        result = H.evaluate_t3([0.10, 0.10, 0.10], [0.06, 0.06, 0.06])
        assert result.passes is True

    def test_fails_when_ratio_above_threshold(self) -> None:
        result = H.evaluate_t3([0.10, 0.10, 0.10], [0.08, 0.08, 0.08])
        assert result.passes is False

    def test_fails_on_degenerate_pre(self) -> None:
        result = H.evaluate_t3([0.0, 0.0], [0.05, 0.05])
        assert result.passes is False
        assert "degenerate baseline" in result.rationale


# ---------------------------------------------------------------------------
# build_manifest — aggregation + manifest construction
# ---------------------------------------------------------------------------


def _seed_result(
    seed: int,
    *,
    baseline_auc: float = 0.60,
    hblp_auc: float = 0.65,
    baseline_ece: float = 0.10,
    hblp_ece: float = 0.04,
    baseline_cv: float = 0.10,
    hblp_cv: float = 0.06,
) -> H.SeedResult:
    return H.SeedResult(
        seed=seed,
        baseline_auc=baseline_auc,
        hblp_auc=hblp_auc,
        baseline_ece=baseline_ece,
        hblp_ece=hblp_ece,
        baseline_cv_stability=baseline_cv,
        hblp_cv_stability=hblp_cv,
    )


class TestBuildManifest:
    def test_passes_when_all_three_thresholds_met(self) -> None:
        cohort = H.COHORTS["optum_initiation_default"]
        seed_results = [_seed_result(s) for s in H.G2_SEEDS]

        manifest = H.build_manifest(
            cohort=cohort,
            seed_results=seed_results,
            experiment_commit_sha="abc123",
        )
        assert manifest.g2_passes_pre_spec is True
        assert len(manifest.thresholds) == 3
        assert all(t["passes"] for t in manifest.thresholds)

    def test_fails_on_t1_violation(self) -> None:
        cohort = H.COHORTS["optum_initiation_default"]
        # Give a tiny AUC lift below 0.03
        seed_results = [_seed_result(s, baseline_auc=0.65, hblp_auc=0.66) for s in H.G2_SEEDS]
        manifest = H.build_manifest(
            cohort=cohort,
            seed_results=seed_results,
            experiment_commit_sha="abc123",
        )
        assert manifest.g2_passes_pre_spec is False
        # Find the T1 entry
        t1 = next(t for t in manifest.thresholds if t["name"] == "T1")
        assert t1["passes"] is False

    def test_fails_on_t2_violation(self) -> None:
        cohort = H.COHORTS["optum_initiation_default"]
        # ECE doesn't halve
        seed_results = [_seed_result(s, baseline_ece=0.10, hblp_ece=0.07) for s in H.G2_SEEDS]
        manifest = H.build_manifest(
            cohort=cohort,
            seed_results=seed_results,
            experiment_commit_sha="abc123",
        )
        assert manifest.g2_passes_pre_spec is False
        t2 = next(t for t in manifest.thresholds if t["name"] == "T2")
        assert t2["passes"] is False

    def test_fails_on_t3_violation(self) -> None:
        cohort = H.COHORTS["optum_initiation_default"]
        # CV-stability doesn't shrink to 70%
        seed_results = [_seed_result(s, baseline_cv=0.10, hblp_cv=0.08) for s in H.G2_SEEDS]
        manifest = H.build_manifest(
            cohort=cohort,
            seed_results=seed_results,
            experiment_commit_sha="abc123",
        )
        assert manifest.g2_passes_pre_spec is False
        t3 = next(t for t in manifest.thresholds if t["name"] == "T3")
        assert t3["passes"] is False

    def test_manifest_records_data_snooped_flag(self) -> None:
        cohort = H.COHORTS["optum_initiation_relaxed"]
        seed_results = [_seed_result(s) for s in H.G2_SEEDS]
        manifest = H.build_manifest(
            cohort=cohort,
            seed_results=seed_results,
            experiment_commit_sha="abc123",
        )
        assert manifest.cohort_data_snooped is True

    def test_manifest_serializes_to_json(self) -> None:
        cohort = H.COHORTS["optum_initiation_default"]
        seed_results = [_seed_result(s) for s in H.G2_SEEDS]
        manifest = H.build_manifest(
            cohort=cohort,
            seed_results=seed_results,
            experiment_commit_sha="abc123",
        )
        # Round-trip through JSON to catch unserializable types.
        payload = json.dumps(manifest.to_dict())
        parsed = json.loads(payload)
        assert parsed["g2_passes_pre_spec"] is True
        assert parsed["cohort_label"] == "optum_initiation_default"

    def test_manifest_contains_lifecycle_state(self) -> None:
        cohort = H.COHORTS["optum_initiation_default"]
        manifest = H.build_manifest(
            cohort=cohort,
            seed_results=[],
            experiment_commit_sha="abc",
        )
        assert manifest.lifecycle_state == "advisory"


# ---------------------------------------------------------------------------
# _build_features_and_target — feature/target extraction
# ---------------------------------------------------------------------------


class TestBuildFeaturesAndTarget:
    def test_drops_excluded_columns(self) -> None:
        df = pd.DataFrame(
            {
                "patient_id": [1, 2, 3, 4],
                "patient_journey_id": [10, 20, 30, 40],
                "data_split": ["train"] * 4,
                "treatment_initiated": [0, 1, 0, 1],
                "feature_a": [0.1, 0.2, 0.3, 0.4],
                "feature_b": [1.0, 2.0, 3.0, 4.0],
            }
        )
        X, y = H._build_features_and_target(df, "treatment_initiated")
        assert "patient_id" not in X.columns
        assert "patient_journey_id" not in X.columns
        assert "data_split" not in X.columns
        assert "treatment_initiated" not in X.columns
        assert set(X.columns) == {"feature_a", "feature_b"}
        assert y.tolist() == [0, 1, 0, 1]

    def test_drops_bool_columns(self) -> None:
        df = pd.DataFrame(
            {
                "feature_a": [0.1, 0.2, 0.3],
                "feature_bool": [True, False, True],
                "treatment_initiated": [0, 1, 0],
            }
        )
        X, _ = H._build_features_and_target(df, "treatment_initiated")
        assert "feature_bool" not in X.columns

    def test_raises_on_missing_target(self) -> None:
        df = pd.DataFrame({"feature_a": [0.1, 0.2]})
        with pytest.raises(KeyError, match="not present"):
            H._build_features_and_target(df, "treatment_initiated")

    def test_raises_when_no_numeric_columns(self) -> None:
        df = pd.DataFrame(
            {
                "label_a": ["x", "y", "z"],
                "treatment_initiated": [0, 1, 0],
            }
        )
        with pytest.raises(ValueError, match="no numeric feature columns"):
            H._build_features_and_target(df, "treatment_initiated")

    def test_y_is_int64_binary(self) -> None:
        df = pd.DataFrame(
            {
                "feature_a": [0.1, 0.2, 0.3, 0.4],
                "treatment_initiated": [0.0, 1.0, 0.0, 1.0],
            }
        )
        _, y = H._build_features_and_target(df, "treatment_initiated")
        assert y.dtype == np.int64
        assert set(y.unique()) <= {0, 1}


# ---------------------------------------------------------------------------
# run_seed — exercised on synthetic features
# ---------------------------------------------------------------------------


def _make_synthetic_xy(n: int = 400, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """Make a small synthetic XY with non-trivial signal so the
    classifier produces meaningful AUCs."""
    rng = np.random.default_rng(seed)
    n_features = 6
    X = rng.normal(size=(n, n_features))
    # Linear true signal in features 0..2
    logits = X[:, 0] * 0.8 + X[:, 1] * 0.5 - X[:, 2] * 0.4
    p = 1.0 / (1.0 + np.exp(-logits))
    y = (rng.uniform(size=n) < p).astype(np.int64)
    df_X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(n_features)])
    return df_X, pd.Series(y, name="treatment_initiated")


class TestRunSeed:
    def test_run_seed_produces_finite_metrics(self) -> None:
        X, y = _make_synthetic_xy(n=400, seed=0)
        result = H.run_seed(X, y, seed=42)
        assert result.error is None
        assert result.baseline_auc is not None
        assert result.hblp_auc is not None
        # Baseline and HBLP feature surfaces are identical in the
        # harness's contrast layer, so the metrics agree exactly per
        # seed (the test pins the equivalence — when a future PR
        # refines the contrast layer, this test gets updated to
        # reflect the new behaviour).
        assert result.baseline_auc == pytest.approx(result.hblp_auc, abs=1e-6)

    def test_run_seed_handles_degenerate_target(self) -> None:
        X = pd.DataFrame({"f": np.arange(20.0)})
        y = pd.Series(np.zeros(20, dtype=np.int64), name="treatment_initiated")
        result = H.run_seed(X, y, seed=42)
        # Either we get an error (because stratification can't split
        # a single-class target) or we get None metrics. Both are
        # acceptable degenerate handling.
        assert result.error is not None or result.baseline_auc is None


# ---------------------------------------------------------------------------
# main() — argument plumbing + data_snooped guard
# ---------------------------------------------------------------------------


class TestMain:
    def test_main_refuses_data_snooped_without_flag(self, capsys: pytest.CaptureFixture) -> None:
        rc = H.main(["--cohort-label", "optum_initiation_relaxed"])
        assert rc == 2
        captured = capsys.readouterr()
        assert "REFUSED" in captured.err
        assert "data_snooped=true" in captured.err

    def test_main_unknown_cohort_label_raises(self) -> None:
        with pytest.raises(SystemExit):
            H.main(["--cohort-label", "made_up_cohort"])
