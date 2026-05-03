"""Tests for Scenario A — HR+/HER2- early breast cancer 5-yr iDFS (Kisqali).

Per shard 03 Section D acceptance criteria.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.ml.synthetic_v2 import ScenarioName, generate_scenario
from src.ml.synthetic_v2.scenarios import SCENARIO_REGISTRY
from src.ml.synthetic_v2.scenarios.scenario_a import (
    SCENARIO_A_CORRELATION_BLOCKS,
    SCENARIO_A_MANIFEST,
    SLOPE_MULTIPLIER,
    ScenarioABuilder,
)


class TestScenarioARegistration:
    def test_builder_in_registry(self) -> None:
        assert ScenarioName.A_DIAGNOSTIC_BC_IDFS in SCENARIO_REGISTRY
        assert SCENARIO_REGISTRY[ScenarioName.A_DIAGNOSTIC_BC_IDFS] is ScenarioABuilder


class TestScenarioAManifestAlignment:
    def test_manifest_has_40_features(self) -> None:
        assert len(SCENARIO_A_MANIFEST) == 40

    def test_n_features_matches_manifest(self) -> None:
        builder = ScenarioABuilder()
        assert builder.n_features == len(SCENARIO_A_MANIFEST)
        builder.validate_manifest_alignment()

    def test_all_feature_names_unique(self) -> None:
        names = [m.name for m in SCENARIO_A_MANIFEST]
        assert len(set(names)) == 40

    def test_target_prevalence_locked(self) -> None:
        assert ScenarioABuilder().target_prevalence == 0.20

    def test_target_auc_band_locked(self) -> None:
        assert ScenarioABuilder().target_auc_band == (0.78, 0.83)

    def test_default_n_total_locked(self) -> None:
        assert ScenarioABuilder().default_n_total == 6000

    def test_correlation_strength_locked(self) -> None:
        assert ScenarioABuilder().correlation_strength == 0.30

    def test_slope_multiplier_locked(self) -> None:
        assert ScenarioABuilder().slope_multiplier == SLOPE_MULTIPLIER
        assert SLOPE_MULTIPLIER == pytest.approx(0.062)

    def test_correlation_blocks_within_n_features_range(self) -> None:
        for cols, _ in SCENARIO_A_CORRELATION_BLOCKS:
            for c in cols:
                assert 0 <= c < 40

    def test_kisqali_anchor_signed_negative(self) -> None:
        """Franchise-narrative invariant per shard 03 §D acceptance test:
        the Kisqali franchise anchor (received_cdk46_inhibitor_adjuvant)
        must protect against the iDFS event (negative coefficient).
        """
        anchor = next(
            m for m in SCENARIO_A_MANIFEST if m.name == "received_cdk46_inhibitor_adjuvant"
        )
        assert anchor.coefficient < 0
        assert anchor.monotone_direction == -1
        assert "KISQALI" in anchor.clinical_justification.upper()


class TestScenarioAGeneration:
    def test_generates_dataset_with_expected_shape(self) -> None:
        ds = generate_scenario(ScenarioName.A_DIAGNOSTIC_BC_IDFS, seed=42, n_total=2000)
        assert ds.X_train.shape[1] == 40
        assert ds.X_train.shape[0] + ds.X_val.shape[0] + ds.X_test.shape[0] == 2000

    def test_realized_prevalence_in_band(self) -> None:
        ds = generate_scenario(ScenarioName.A_DIAGNOSTIC_BC_IDFS, seed=42, n_total=6000)
        assert abs(ds.metadata.realized_prevalence - 0.20) < 0.02

    @pytest.mark.parametrize("seed", [0, 1, 5, 9])
    def test_realized_prevalence_in_band_across_seeds(self, seed: int) -> None:
        ds = generate_scenario(ScenarioName.A_DIAGNOSTIC_BC_IDFS, seed=seed, n_total=6000)
        assert abs(ds.metadata.realized_prevalence - 0.20) < 0.025

    def test_byte_identical_reruns(self) -> None:
        ds1 = generate_scenario(ScenarioName.A_DIAGNOSTIC_BC_IDFS, seed=42, n_total=2000)
        ds2 = generate_scenario(ScenarioName.A_DIAGNOSTIC_BC_IDFS, seed=42, n_total=2000)
        np.testing.assert_array_equal(ds1.X_train, ds2.X_train)
        np.testing.assert_array_equal(ds1.y_train, ds2.y_train)
        assert ds1.metadata.audit_fingerprint == ds2.metadata.audit_fingerprint

    def test_metadata_carries_full_manifest(self) -> None:
        ds = generate_scenario(ScenarioName.A_DIAGNOSTIC_BC_IDFS, seed=42, n_total=2000)
        assert len(ds.metadata.feature_manifest) == 40
        assert ds.metadata.feature_names[0] == "age_at_diagnosis_years"
        assert ds.metadata.feature_names[-1] == "noise_admin_4"

    def test_correlation_blocks_psd(self) -> None:
        """Acceptance test per shard 03 §D — all correlation blocks
        Cholesky-decomposable. Implicitly verified by successful generation.
        """
        # If this generates without ValueError on PSD, the blocks are PSD
        ds = generate_scenario(ScenarioName.A_DIAGNOSTIC_BC_IDFS, seed=42, n_total=1000)
        assert ds.X_train.shape[1] == 40


@pytest.mark.slow
class TestScenarioAAUCBandRegression:
    """Slow regression test — AUC band acceptance per shard 03 §D + shard 09 §A.2.

    Asserts that 9 of 10 seeds land in the [0.78, 0.83] band with a logistic
    regression baseline on standardized X_train. Calibration anchored by
    SLOPE_MULTIPLIER = 0.062 (locked 2026-05-03; see scenario_a.py header).
    """

    def test_lr_auc_band_9_of_10_seeds(self) -> None:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score

        aucs = []
        for seed in range(10):
            ds = generate_scenario(ScenarioName.A_DIAGNOSTIC_BC_IDFS, seed=seed, n_total=6000)
            clf = LogisticRegression(max_iter=2000, C=1.0)
            clf.fit(ds.X_train, ds.y_train)
            prob = clf.predict_proba(ds.X_test)[:, 1]
            aucs.append(roc_auc_score(ds.y_test, prob))
        in_band = sum(1 for a in aucs if 0.78 <= a <= 0.83)
        assert in_band >= 9, f"Only {in_band}/10 seeds in [0.78, 0.83]; aucs={aucs}"
