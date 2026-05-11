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
# HIGH-7 — exact cohort identity enforcement
# ---------------------------------------------------------------------------


class TestCohortIdentityEnforcement:
    """Per HIGH-7: cohort spec's expected_n_exact pins the cohort
    identity; the harness must refuse to run on data that does not
    match the declared row count."""

    def test_default_cohort_expects_n_1294(self) -> None:
        """Pin the pre-spec memo's load-bearing default cohort size."""
        assert H.COHORTS["optum_initiation_default"].expected_n_exact == 1294

    def test_relaxed_cohort_expects_n_1697(self) -> None:
        """Pin the pre-spec memo's sensitivity-cohort size."""
        assert H.COHORTS["optum_initiation_relaxed"].expected_n_exact == 1697

    def test_cohort_specs_carry_expected_n_exact_field(self) -> None:
        """The CohortSpec dataclass field name is the load-bearing one."""
        cohort = H.COHORTS["optum_initiation_default"]
        assert hasattr(cohort, "expected_n_exact")
        assert isinstance(cohort.expected_n_exact, int)

    def test_run_experiment_raises_on_cohort_n_mismatch(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """End-to-end HIGH-7: when the loaded patient_journeys frame
        has a different row count than the cohort spec's expected_n_exact,
        run_experiment raises ValueError."""
        # Plant a fake patient_journeys frame with the WRONG row count.
        cohort_dir = tmp_path / "data" / "rwd" / "optum" / "initiation"
        cohort_dir.mkdir(parents=True)
        wrong_n = 100  # not 1294 or 1697
        df = pd.DataFrame(
            {
                "patient_id": list(range(wrong_n)),
                "treatment_initiated": [i % 2 for i in range(wrong_n)],
                "feature_a": [float(i) for i in range(wrong_n)],
            }
        )
        df.to_parquet(cohort_dir / "e2i_ml_v3_patient_journeys.parquet")

        # Monkey-patch the FileIngestor so we don't need the full
        # ingestion machinery — return our wrong-n frame directly.
        def fake_load_patient_journeys(directory: Path) -> pd.DataFrame:
            return df

        monkeypatch.setattr(H, "_load_patient_journeys", fake_load_patient_journeys)

        with pytest.raises(ValueError, match="HIGH-7 cohort identity mismatch"):
            H.run_experiment(
                "optum_initiation_default",
                project_root=tmp_path,
            )


# ---------------------------------------------------------------------------
# HIGH-8 — all-seeds + all-metrics required before threshold eval
# ---------------------------------------------------------------------------


class TestHigh8AllSeedsRequired:
    def test_missing_seeds_fail_manifest(self) -> None:
        """Only 3 of 5 seeds produced metrics → manifest fails hard."""
        cohort = H.COHORTS["optum_initiation_default"]
        seed_results = [_seed_result(s) for s in (42, 43, 44)]  # missing 45, 46
        manifest = H.build_manifest(
            cohort=cohort,
            seed_results=seed_results,
            experiment_commit_sha="abc",
        )
        assert manifest.g2_passes_pre_spec is False
        assert manifest.thresholds == []
        comp = manifest.seed_completeness
        assert comp["all_seeds_complete"] is False
        assert comp["missing_seeds"] == [45, 46]
        assert "HIGH-8" in manifest.notes

    def test_incomplete_seed_metrics_fail_manifest(self) -> None:
        """All 5 seeds present, but ONE has a None metric → fail hard."""
        cohort = H.COHORTS["optum_initiation_default"]
        seed_results = [_seed_result(s) for s in H.G2_SEEDS]
        # Knock out one metric on seed=43.
        for r in seed_results:
            if r.seed == 43:
                r.baseline_ece = None
        manifest = H.build_manifest(
            cohort=cohort,
            seed_results=seed_results,
            experiment_commit_sha="abc",
        )
        assert manifest.g2_passes_pre_spec is False
        assert manifest.thresholds == []
        comp = manifest.seed_completeness
        assert comp["all_seeds_complete"] is False
        assert 43 in comp["incomplete_seeds"]

    def test_seed_with_error_fails_manifest(self) -> None:
        """A seed with non-None ``error`` field counts as incomplete."""
        cohort = H.COHORTS["optum_initiation_default"]
        seed_results = [_seed_result(s) for s in H.G2_SEEDS]
        seed_results[2].error = "RuntimeError: simulated"
        manifest = H.build_manifest(
            cohort=cohort,
            seed_results=seed_results,
            experiment_commit_sha="abc",
        )
        assert manifest.g2_passes_pre_spec is False
        assert manifest.thresholds == []

    def test_all_seeds_complete_proceeds_to_threshold_eval(self) -> None:
        """Happy path: all 5 seeds with all 6 metrics → thresholds eval."""
        cohort = H.COHORTS["optum_initiation_default"]
        seed_results = [_seed_result(s) for s in H.G2_SEEDS]
        manifest = H.build_manifest(
            cohort=cohort,
            seed_results=seed_results,
            experiment_commit_sha="abc",
        )
        assert manifest.seed_completeness["all_seeds_complete"] is True
        assert len(manifest.thresholds) == 3

    def test_seed_completeness_diagnostic_helper(self) -> None:
        """The diagnostic helper returns the expected shape."""
        seed_results = [_seed_result(s) for s in H.G2_SEEDS]
        comp = H._seed_completeness_diagnostic(seed_results)
        assert comp["all_seeds_complete"] is True
        assert comp["observed_seeds"] == list(H.G2_SEEDS)
        assert comp["missing_seeds"] == []
        assert comp["incomplete_seeds"] == []
        assert len(comp["per_seed_diagnostic"]) == 5


# ---------------------------------------------------------------------------
# LOW-11 — ECE method metadata in manifest
# ---------------------------------------------------------------------------


class TestEceMethodMetadata:
    """The manifest must record HOW ECE was computed (helper, bin
    count, binning strategy) so a future helper change is audit-visible."""

    def test_manifest_has_ece_method_metadata(self) -> None:
        cohort = H.COHORTS["optum_initiation_default"]
        seed_results = [_seed_result(s) for s in H.G2_SEEDS]
        manifest = H.build_manifest(
            cohort=cohort,
            seed_results=seed_results,
            experiment_commit_sha="abc",
        )
        meta = manifest.ece_method_metadata
        assert (
            meta["helper_module"]
            == "src.agents.ml_foundation.model_trainer.nodes.advanced_validation"
        )
        assert meta["helper_function"] == "compute_calibration_analysis"
        assert meta["n_bins"] == H.G2_ECE_BINS
        assert meta["binning_strategy"] == "equal-width"

    def test_ece_method_metadata_serializable(self) -> None:
        cohort = H.COHORTS["optum_initiation_default"]
        seed_results = [_seed_result(s) for s in H.G2_SEEDS]
        manifest = H.build_manifest(
            cohort=cohort,
            seed_results=seed_results,
            experiment_commit_sha="abc",
        )
        # JSON round-trip catches non-stdlib types.
        parsed = json.loads(json.dumps(manifest.to_dict()))
        assert "ece_method_metadata" in parsed
        assert parsed["ece_method_metadata"]["n_bins"] == H.G2_ECE_BINS


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
        # Both arms produce finite metrics. On synthetic data with
        # only well-behaved (low-z) features, the baseline and HBLP
        # arms have IDENTICAL retention sets (no feature crosses the
        # 5σ legacy drop threshold), so the metrics agree exactly. The
        # real-contrast test below pins the divergent case where HBLP
        # retains a feature the baseline drops.
        assert result.baseline_n_features_retained == result.hblp_n_features_retained, (
            "synthetic XY has no high-z leak features → arms agree on retention"
        )

    def test_run_seed_handles_degenerate_target(self) -> None:
        X = pd.DataFrame({"f": np.arange(20.0)})
        y = pd.Series(np.zeros(20, dtype=np.int64), name="treatment_initiated")
        result = H.run_seed(X, y, seed=42)
        # Either we get an error (because stratification can't split
        # a single-class target) or we get None metrics. Both are
        # acceptable degenerate handling.
        assert result.error is not None or result.baseline_auc is None


# ---------------------------------------------------------------------------
# HIGH-1 — REAL baseline-vs-HBLP contrast at feature retention.
#
# The contrast must materialize at retention time: a feature with high
# marginal z-score that the baseline drops must be retained by HBLP when
# layer_1_declared_safe=True (and/or low n_train_pos triggers
# variance-inflation). This test pins the divergence so a regression to
# "no contrast" (the codex HIGH-1 finding) is caught loudly.
# ---------------------------------------------------------------------------


class TestRealBaselineVsHblpContrast:
    """Pins the HIGH-1 fix: real per-arm feature retention divergence."""

    def _make_xy_with_high_z_feature(
        self,
        n: int = 400,
        seed: int = 0,
        leak_mean_shift: float = 0.8,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Construct an XY where ONE feature has a marginal z just
        above ``HIGH_Z=5σ`` but below the HBLP-relaxed threshold of
        ``5σ × 1.5 = 7.5σ`` (when ``layer_1_declared_safe=True`` and
        ``n_train_pos`` is large enough that variance-inflation = 1.0).

        The mean-shift of 0.8 produces a Welch z ≈ 5.8 in the training
        split, which is dropped by baseline (>5) but retained by HBLP
        (5.8 < 7.5). The remaining features are the same noisy signal
        as the harness's synthetic baseline.
        """
        rng = np.random.default_rng(seed)
        n_signal = 4
        X_signal = rng.normal(size=(n, n_signal))
        logits = X_signal[:, 0] * 0.8 + X_signal[:, 1] * 0.5 - X_signal[:, 2] * 0.4
        p = 1.0 / (1.0 + np.exp(-logits))
        y = (rng.uniform(size=n) < p).astype(np.int64)
        leak = np.where(
            y > 0,
            rng.normal(loc=leak_mean_shift, scale=1.0, size=n),
            rng.normal(loc=0.0, scale=1.0, size=n),
        )
        df_X = pd.DataFrame(X_signal, columns=[f"feature_{i}" for i in range(n_signal)])
        df_X["leakage_proxy"] = leak
        return df_X, pd.Series(y, name="treatment_initiated")

    def test_hblp_retains_feature_baseline_drops_when_declared_safe(self) -> None:
        """The load-bearing HIGH-1 contract: a high-z feature with
        ``layer_1_declared_safe=True`` is dropped by baseline (legacy
        ``z > HIGH_Z``) but retained by HBLP (``hblp_classify``
        severity != 'high').

        This proves the two arms diverge on retention, which proves
        the metric arrays diverge, which proves G2's ΔAUC / ECE-ratio
        / CV-stability-ratio are genuine contrasts.
        """
        X, y = self._make_xy_with_high_z_feature(n=400, seed=0)
        # Construct lookup so the leak feature is "declared safe" by
        # Layer 1 (it has a manifest contract claiming knowable_at <=
        # index_date). All other features default to declared_safe=False.
        explicit_lookup = {"leakage_proxy": True}

        result = H.run_seed(
            X,
            y,
            seed=42,
            layer_1_declared_safe_lookup=explicit_lookup,
        )
        assert result.error is None, f"unexpected error: {result.error}"
        # Sanity: the leak feature has z > HIGH_Z (legacy strict
        # would drop it).
        # Sanity: baseline dropped the leak; HBLP retained it.
        assert "leakage_proxy" in result.baseline_features_dropped, (
            "baseline should drop the leak feature (legacy strict z > HIGH_Z), but it was kept"
        )
        assert "leakage_proxy" not in result.hblp_features_dropped, (
            "HBLP should retain the leak feature when "
            "layer_1_declared_safe=True (variance-inflation + 1.5x "
            "prior pushes effective threshold above the leak's z); "
            "but HBLP dropped it"
        )
        # Divergence set is non-empty.
        assert "leakage_proxy" in result.features_diverged
        # Retained-feature counts differ → metric arrays must differ.
        assert result.hblp_n_features_retained > result.baseline_n_features_retained, (
            f"HBLP retention ({result.hblp_n_features_retained}) "
            f"should exceed baseline retention "
            f"({result.baseline_n_features_retained})"
        )

    def test_hblp_drops_with_baseline_when_declared_safe_false(self) -> None:
        """Symmetric pinning: when ``layer_1_declared_safe=False`` and
        ``n_train_pos`` is large enough that variance-inflation = 1.0,
        HBLP and baseline agree (both drop the high-z feature)."""
        X, y = self._make_xy_with_high_z_feature(n=400, seed=0)
        # No Layer-1 information → declared_safe=False for all → HBLP
        # threshold collapses to base HIGH_Z, matching baseline.
        result = H.run_seed(
            X,
            y,
            seed=42,
            layer_1_declared_safe_lookup={},  # all False
        )
        assert result.error is None
        # With n_train_pos in the hundreds (variance-inflation = 1.0)
        # and declared_safe=False everywhere, the HBLP effective
        # threshold equals the legacy HIGH_Z, so the leak should be
        # dropped by BOTH arms.
        assert "leakage_proxy" in result.baseline_features_dropped
        assert "leakage_proxy" in result.hblp_features_dropped
        assert result.features_diverged == []

    def test_compute_marginal_z_scores_returns_finite(self) -> None:
        """Sanity: the z-score helper produces finite values per feature.

        v5 Gate A1: switched to the production permutation-null probe
        (``compute_adversarial_score``). Unlike the pre-A1 Welch's-t
        surrogate which used ``abs(diff)/se`` (non-negative by
        construction), the production probe returns
        ``(actual_auc - null_mean) / null_std`` which can be small and
        negative for noise features when the actual AUC happens to land
        slightly below the folded permutation null's mean. Negative
        noise z is correct production semantics; the leak threshold
        ``z > HIGH_Z`` still fires.
        """
        X, y = self._make_xy_with_high_z_feature(n=400, seed=0)
        z = H._compute_marginal_z_scores(X, y)
        assert set(z.keys()) == set(X.columns)
        for col, val in z.items():
            assert np.isfinite(val)
        # Leak feature has z > 5 (the legacy HIGH_Z drop threshold).
        from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
            HIGH_Z,
        )

        assert z["leakage_proxy"] > HIGH_Z


class TestV5GateA1ProductionParity:
    """v5 Gate A1 regression pin: harness Layer 3 z-score MUST match the
    production permutation-null probe, not the pre-A1 Welch's-t surrogate.

    The load-bearing pin: a binary perfect-correlation feature (r=1.0 with
    target) MUST return z >> 5σ under the harness's z-score helper. The
    pre-A1 Welch implementation returned z=0.0 on this case (within-group
    variance=0 → SE=0 → fall-through), letting the leak pass both baseline
    AND HBLP arms — silently inflating AUC. The production
    ``compute_adversarial_score`` correctly returns z >> 5σ because the
    permutation-shuffled-label null distribution has non-zero spread
    (folded AUC clusters around 0.5-0.6 instead of 1.0).
    """

    def test_binary_perfect_correlation_returns_z_above_high_z(self) -> None:
        """The HARDCODED regression: a binary y-copy feature must trip
        the legacy 5σ drop threshold. If this test ever fails because
        someone reverts to a Welch surrogate, the harness silently
        regresses to retaining perfect-correlation leaks."""
        from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
            HIGH_Z,
        )

        rng = np.random.default_rng(0)
        y = pd.Series(rng.integers(0, 2, size=300).astype(np.int64))
        X = pd.DataFrame(
            {
                "leak_perfect": y.astype(float),
                "leak_anti": (1 - y).astype(float),
                "noise": rng.normal(size=300),
            }
        )
        z = H._compute_marginal_z_scores(X, y, n_permutations=200, adversarial_seed=7)
        # The two perfectly-correlated features (r=+1 and r=-1) both fold
        # to AUC=1.0 vs a null clustered near 0.5-0.6 → z very large.
        assert z["leak_perfect"] > HIGH_Z, (
            f"v5 A1 regression: binary perfect-correlation feature returned "
            f"z={z['leak_perfect']:.3f}, expected z > HIGH_Z={HIGH_Z}. Pre-A1 "
            f"Welch returned z=0.0 here — DO NOT revert to a Welch surrogate."
        )
        assert z["leak_anti"] > HIGH_Z
        # The noise feature must NOT trip the drop threshold (sanity).
        assert z["noise"] < HIGH_Z

    def test_binary_perfect_correlation_baseline_drops_via_legacy_strict(self) -> None:
        """Round-trip: with the production probe wired in, the legacy
        strict policy actually drops the perfect-correlation feature.
        Pre-A1 it did NOT (Welch z=0 fell below HIGH_Z, retained)."""
        rng = np.random.default_rng(0)
        y = pd.Series(rng.integers(0, 2, size=300).astype(np.int64))
        X = pd.DataFrame(
            {
                "leak_perfect": y.astype(float),
                "noise_1": rng.normal(size=300),
                "noise_2": rng.normal(size=300),
            }
        )
        retained, _, dropped = H._build_baseline_feature_surface(
            X, y, n_permutations=200, adversarial_seed=7
        )
        assert "leak_perfect" in dropped, (
            "v5 A1 regression: baseline (legacy strict) should drop "
            "the perfect-correlation feature, but retained it. This "
            "is the silent-leak signature the pre-A1 Welch surrogate "
            "produced."
        )
        assert "leak_perfect" not in retained

    def test_harness_constants_match_production(self) -> None:
        """v5 A1: harness defaults must match production
        ``DEFAULT_PERMUTATIONS=200`` and shouldn't drift silently."""
        from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
            DEFAULT_PERMUTATIONS,
        )

        assert H.DEFAULT_HARNESS_N_PERMUTATIONS == DEFAULT_PERMUTATIONS, (
            "Harness permutation default drifted from production "
            "(adaptive_validity_check.DEFAULT_PERMUTATIONS). Update both "
            "or document the divergence."
        )

    def test_min_layer3_samples_guard_matches_production(self) -> None:
        """v5 A1 codex HIGH-1: ``_HARNESS_MIN_LAYER3_SAMPLES`` must
        equal production ``MIN_LAYER3_SAMPLES`` (= 30). Drift detector."""
        from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
            MIN_LAYER3_SAMPLES,
        )

        assert H._HARNESS_MIN_LAYER3_SAMPLES == MIN_LAYER3_SAMPLES, (
            "Harness MIN_LAYER3_SAMPLES drifted from production. "
            "Sparse features (rows < 30) would diverge silently."
        )

    def test_below_min_samples_emits_zero_z(self) -> None:
        """v5 A1 codex HIGH-1: a feature with fewer than
        ``_HARNESS_MIN_LAYER3_SAMPLES`` non-null rows must short-circuit
        to z=0.0 (= retain), mirroring production's severity=info
        short-circuit for sparse features."""
        rng = np.random.default_rng(0)
        n = 100
        y = pd.Series(rng.integers(0, 2, size=n).astype(np.int64))
        # Build a feature with only 5 non-null rows — well below the
        # 30-sample guard.
        sparse_values = np.full(n, np.nan, dtype=np.float64)
        sparse_values[:5] = rng.normal(size=5)
        X = pd.DataFrame(
            {
                "sparse_feature": sparse_values,
                "dense_noise": rng.normal(size=n),
            }
        )
        z = H._compute_marginal_z_scores(X, y, n_permutations=50, adversarial_seed=7)
        assert z["sparse_feature"] == 0.0, (
            "v5 A1 regression: sparse feature should short-circuit to "
            "z=0.0 below MIN_LAYER3_SAMPLES, but got "
            f"z={z['sparse_feature']:.3f}. Production parity broken."
        )

    def test_infinity_values_routed_to_degenerate_path(self) -> None:
        """v5 A1 codex HIGH-2: ``+inf`` / ``-inf`` rows must flow
        through to ``compute_adversarial_score``'s exception path
        (not be silently filtered out), matching production's mask
        behavior. The downstream call returns NaN z → harness emits
        z=0.0 (= retain)."""
        rng = np.random.default_rng(0)
        n = 200
        y = pd.Series(rng.integers(0, 2, size=n).astype(np.int64))
        inf_values = rng.normal(size=n)
        # Inject a few +inf entries — they should NOT be filtered out
        # before the probe sees them. Production passes them through.
        inf_values[0] = np.inf
        inf_values[1] = -np.inf
        X = pd.DataFrame({"inf_feature": inf_values})
        z = H._compute_marginal_z_scores(X, y, n_permutations=50, adversarial_seed=7)
        # The presence of +/-inf in the feature column causes
        # roc_auc_score to raise a ValueError; the probe catches it
        # and returns NaN z; the harness floors that to 0.0.
        assert z["inf_feature"] == 0.0, (
            "v5 A1 regression: +/-inf entries should route through the "
            "production exception path (returning NaN-z → 0.0), but got "
            f"z={z['inf_feature']:.3f}."
        )

    def test_single_class_after_mask_emits_zero_z(self) -> None:
        """v5 A1: feature mask that leaves only one target class after
        filtering must short-circuit to z=0.0 (cannot fit ROC AUC
        with a single class). Production's ValueError catch produces
        the same effective outcome."""
        n = 100
        # All-zero target → impossible to compute AUC
        y = pd.Series(np.zeros(n, dtype=np.int64))
        rng = np.random.default_rng(0)
        X = pd.DataFrame({"any_feature": rng.normal(size=n)})
        z = H._compute_marginal_z_scores(X, y, n_permutations=50, adversarial_seed=7)
        assert z["any_feature"] == 0.0

    def test_constant_feature_emits_zero_z(self) -> None:
        """v5 A1: a constant-value feature has zero discriminative
        signal. Production's ``compute_adversarial_score`` returns
        z=0.0 (null_std=0, actual_auc=null_mean fallthrough) — harness
        must preserve this."""
        rng = np.random.default_rng(0)
        n = 200
        y = pd.Series(rng.integers(0, 2, size=n).astype(np.int64))
        X = pd.DataFrame(
            {
                "constant_zero": np.zeros(n, dtype=np.float64),
                "constant_seven": np.full(n, 7.0, dtype=np.float64),
            }
        )
        z = H._compute_marginal_z_scores(X, y, n_permutations=50, adversarial_seed=7)
        assert z["constant_zero"] == 0.0
        assert z["constant_seven"] == 0.0

    def test_legacy_strict_drop_uses_high_z_constant(self) -> None:
        """Pin that the baseline arm drops EXACTLY at z > HIGH_Z."""
        from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
            HIGH_Z,
        )

        z = {
            "below": HIGH_Z - 0.1,
            "at": HIGH_Z,  # not strictly greater → kept
            "above": HIGH_Z + 0.1,  # strictly greater → dropped
        }
        dropped = H._legacy_strict_drop(z)
        assert dropped == ["above"]

    def test_hblp_drop_relaxes_threshold_when_declared_safe(self) -> None:
        """Pin the HBLP retention contract directly on the helper:
        a feature with z just above HIGH_Z but ``declared_safe=True``
        is retained by HBLP (severity is ``"moderate"`` or ``"info"``,
        not ``"high"``)."""
        from src.agents.ml_foundation.data_preparer.nodes.adaptive_validity_check import (
            HIGH_Z,
        )

        z = {"borderline_safe": HIGH_Z + 0.5, "borderline_unsafe": HIGH_Z + 0.5}
        # n_train_pos large enough that variance-inflation = 1.0;
        # only the layer-1 prior matters.
        dropped = H._hblp_drop(
            z,
            n_train_pos=300,
            layer_1_declared_safe_lookup={
                "borderline_safe": True,
                "borderline_unsafe": False,
            },
        )
        # Unsafe: HBLP threshold = HIGH_Z (no relaxation) → z=HIGH_Z+0.5 > 5.0 → drop.
        assert "borderline_unsafe" in dropped
        # Safe: HBLP threshold = HIGH_Z * 1.5 = 7.5σ → z=5.5 < 7.5 → keep.
        assert "borderline_safe" not in dropped


class TestResolveLayerOneDeclaredSafeLookup:
    def test_explicit_lookup_takes_precedence(self) -> None:
        out = H._resolve_layer_1_declared_safe_lookup(
            ["a", "b", "c"],
            manifest_source="anything",
            explicit_lookup={"a": True, "c": True},
        )
        assert out == {"a": True, "b": False, "c": True}

    def test_no_manifest_returns_all_false(self) -> None:
        out = H._resolve_layer_1_declared_safe_lookup(["a", "b"])
        assert out == {"a": False, "b": False}

    def test_unknown_manifest_source_returns_all_false(self) -> None:
        out = H._resolve_layer_1_declared_safe_lookup(
            ["a", "b"],
            manifest_source="not_a_manifest_source_anywhere",
        )
        assert out == {"a": False, "b": False}


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


# ---------------------------------------------------------------------------
# HIGH-1 (iter-3) — production wiring: cohort.manifest_source threads into
# run_experiment, the resolved Layer-1 lookup is non-trivial for Optum, and
# load-bearing runs FAIL LOUDLY when no declared-safe features resolve.
# ---------------------------------------------------------------------------


class TestHigh1ProductionManifestWiring:
    """Production-path tests — without test-only fixture injection — that
    pin: (a) the cohort spec carries the correct manifest_source, (b) the
    resolved lookup against the real Optum manifest is non-trivial, and
    (c) run_experiment refuses to proceed on a load-bearing run with an
    empty declared-safe lookup."""

    def test_optum_default_cohort_declares_optum_manifest_source(self) -> None:
        """Default cohort routes to the Optum manifest registry."""
        assert H.COHORTS["optum_initiation_default"].manifest_source == "optum"

    def test_optum_relaxed_cohort_declares_optum_manifest_source(self) -> None:
        """Relaxed cohort also routes to Optum (same data dir)."""
        assert H.COHORTS["optum_initiation_relaxed"].manifest_source == "optum"

    def test_optum_manifest_resolves_nontrivial_declared_safe_lookup(self) -> None:
        """Production-path resolution against the Optum manifest must
        return a non-empty declared-safe map for typical numeric features
        (charlson_score, elixhauser_score, months_since_first_dx, etc.).
        These have ``knowable_at.reference="index_date"`` which is in
        the pre-anchor refs set the harness uses.
        """
        feature_names = [
            "charlson_score",
            "elixhauser_score",
            "months_since_first_dx",
            "office_visits_total",
            "unique_providers",
        ]
        out = H._resolve_layer_1_declared_safe_lookup(
            feature_names,
            manifest_source="optum",
        )
        # Every feature in this list has a manifest contract with
        # knowable_at.reference="index_date", which IS in the pre-anchor
        # refs set, so all should resolve declared_safe=True.
        for f in feature_names:
            assert out[f] is True, (
                f"Optum manifest feature {f!r} expected declared_safe=True; "
                f"got {out[f]!r}. The contract's knowable_at.reference must "
                f"be one of {{index_date, lookback_start_date, eligeff}}."
            )

    def test_run_experiment_fails_loud_on_empty_declared_safe_lookup(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """End-to-end HIGH-1: when the resolved Layer-1 declared-safe
        lookup is empty (zero features declared safe), a load-bearing
        run RAISES instead of silently proceeding with all-False (which
        would collapse the HBLP arm to baseline)."""
        cohort_dir = tmp_path / "data" / "rwd" / "optum" / "initiation"
        cohort_dir.mkdir(parents=True)
        n = 1294  # match cohort.expected_n_exact for the default cohort
        df = pd.DataFrame(
            {
                "patient_id": list(range(n)),
                "treatment_initiated": [i % 2 for i in range(n)],
                # All feature names are NOT in the Optum manifest, so the
                # resolved lookup will be all False.
                "feature_unknown_a": [float(i) for i in range(n)],
                "feature_unknown_b": [float(i) % 7 for i in range(n)],
            }
        )
        df.to_parquet(cohort_dir / "e2i_ml_v3_patient_journeys.parquet")

        def fake_load_patient_journeys(directory: Path) -> pd.DataFrame:
            return df

        monkeypatch.setattr(H, "_load_patient_journeys", fake_load_patient_journeys)

        with pytest.raises(RuntimeError, match="HIGH-1 hard fail"):
            H.run_experiment(
                "optum_initiation_default",
                project_root=tmp_path,
                load_bearing=True,
            )

    def test_run_experiment_proceeds_when_optum_manifest_resolves_safe_features(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """End-to-end HIGH-1: when the patient_journeys frame includes
        features that ARE in the Optum manifest with pre-anchor
        ``knowable_at.reference``, the load-bearing run resolves a
        non-trivial lookup and proceeds (does not raise HIGH-1 hard fail).

        This is the production-path proof that test-only injection is
        NOT required for the harness to see a non-trivial lookup.
        """
        cohort_dir = tmp_path / "data" / "rwd" / "optum" / "initiation"
        cohort_dir.mkdir(parents=True)
        n = 1294
        rng = np.random.default_rng(0)
        # Use feature names that ARE in the Optum manifest with
        # knowable_at.reference="index_date".
        df = pd.DataFrame(
            {
                "patient_id": list(range(n)),
                "treatment_initiated": [i % 2 for i in range(n)],
                "charlson_score": rng.normal(size=n),
                "elixhauser_score": rng.normal(size=n),
                "months_since_first_dx": rng.normal(size=n),
                "office_visits_total": rng.normal(size=n),
                "unique_providers": rng.normal(size=n),
            }
        )
        df.to_parquet(cohort_dir / "e2i_ml_v3_patient_journeys.parquet")

        def fake_load_patient_journeys(directory: Path) -> pd.DataFrame:
            return df

        monkeypatch.setattr(H, "_load_patient_journeys", fake_load_patient_journeys)

        # Should not raise HIGH-1 — the resolved lookup is non-trivial.
        manifest = H.run_experiment(
            "optum_initiation_default",
            project_root=tmp_path,
            seeds=(42,),  # one seed is enough to prove the wiring path
            load_bearing=True,
        )
        # Manifest is built; whether thresholds pass or not is not the
        # point — what we're testing is the wiring did not raise.
        assert manifest.cohort_label == "optum_initiation_default"
        assert manifest.cohort_observed_n == n

    def test_run_experiment_load_bearing_off_skips_high1_check(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """When load_bearing=False (diagnostic run), the empty-lookup
        check is skipped — the harness still emits a manifest but the
        contrast is NOT load-bearing for G2."""
        cohort_dir = tmp_path / "data" / "rwd" / "optum" / "initiation"
        cohort_dir.mkdir(parents=True)
        n = 1294
        df = pd.DataFrame(
            {
                "patient_id": list(range(n)),
                "treatment_initiated": [i % 2 for i in range(n)],
                "feature_unknown_a": [float(i) for i in range(n)],
                "feature_unknown_b": [float(i) % 7 for i in range(n)],
            }
        )
        df.to_parquet(cohort_dir / "e2i_ml_v3_patient_journeys.parquet")

        def fake_load_patient_journeys(directory: Path) -> pd.DataFrame:
            return df

        monkeypatch.setattr(H, "_load_patient_journeys", fake_load_patient_journeys)

        # load_bearing=False: do NOT raise even though declared-safe is empty.
        manifest = H.run_experiment(
            "optum_initiation_default",
            project_root=tmp_path,
            seeds=(42,),
            load_bearing=False,
        )
        assert manifest.cohort_label == "optum_initiation_default"

    def test_run_experiment_explicit_lookup_bypasses_high1_check(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """When the caller passes layer_1_declared_safe_lookup
        explicitly (test-only injection), the manifest-source
        resolution is bypassed AND the HIGH-1 empty-lookup gate is
        bypassed (the caller has taken responsibility for the lookup)."""
        cohort_dir = tmp_path / "data" / "rwd" / "optum" / "initiation"
        cohort_dir.mkdir(parents=True)
        n = 1294
        df = pd.DataFrame(
            {
                "patient_id": list(range(n)),
                "treatment_initiated": [i % 2 for i in range(n)],
                # No features in manifest, BUT the caller supplied an
                # explicit lookup (even if it is an empty dict, the
                # bypass is intentional).
                "feature_unknown": [float(i) for i in range(n)],
            }
        )
        df.to_parquet(cohort_dir / "e2i_ml_v3_patient_journeys.parquet")

        def fake_load_patient_journeys(directory: Path) -> pd.DataFrame:
            return df

        monkeypatch.setattr(H, "_load_patient_journeys", fake_load_patient_journeys)

        manifest = H.run_experiment(
            "optum_initiation_default",
            project_root=tmp_path,
            seeds=(42,),
            layer_1_declared_safe_lookup={},
            load_bearing=True,
        )
        # Should not raise — explicit (even-if-empty) lookup bypasses
        # the production-path emptiness check.
        assert manifest.cohort_label == "optum_initiation_default"
