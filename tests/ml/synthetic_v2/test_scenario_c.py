"""Tests for Scenario C — CSU 12-wk UAS7=0 response on remibrutinib (Rhapsido).

Per shard 05 Section D acceptance criteria. **MAIN RWD COHORT**: the RWD
concurrent-validation tests live in test_rwd_csu_loader.py (commit 13).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.ml.synthetic_v2 import ScenarioName, generate_scenario
from src.ml.synthetic_v2.scenarios import SCENARIO_REGISTRY
from src.ml.synthetic_v2.scenarios.scenario_c import (
    SCENARIO_C_CORRELATION_BLOCKS,
    SCENARIO_C_MANIFEST,
    SLOPE_MULTIPLIER,
    ScenarioCBuilder,
)


class TestScenarioCRegistration:
    def test_builder_in_registry(self) -> None:
        assert ScenarioName.C_TREATMENT_CSU_RESPONSE in SCENARIO_REGISTRY
        assert SCENARIO_REGISTRY[ScenarioName.C_TREATMENT_CSU_RESPONSE] is ScenarioCBuilder


class TestScenarioCManifestAlignment:
    def test_manifest_has_60_features(self) -> None:
        assert len(SCENARIO_C_MANIFEST) == 60

    def test_n_features_matches_manifest(self) -> None:
        builder = ScenarioCBuilder()
        assert builder.n_features == len(SCENARIO_C_MANIFEST)
        builder.validate_manifest_alignment()

    def test_all_feature_names_unique(self) -> None:
        names = [m.name for m in SCENARIO_C_MANIFEST]
        assert len(set(names)) == 60

    def test_target_prevalence_locked(self) -> None:
        assert ScenarioCBuilder().target_prevalence == 0.40

    def test_target_auc_band_locked(self) -> None:
        assert ScenarioCBuilder().target_auc_band == (0.82, 0.88)

    def test_correlation_strength_locked(self) -> None:
        assert ScenarioCBuilder().correlation_strength == 0.50

    def test_slope_multiplier_locked(self) -> None:
        assert SLOPE_MULTIPLIER == pytest.approx(1.25)

    def test_correlation_blocks_within_n_features_range(self) -> None:
        for cols, _ in SCENARIO_C_CORRELATION_BLOCKS:
            for c in cols:
                assert 0 <= c < 60

    def test_basophil_signed_negative_per_codex_i3(self) -> None:
        """Codex I-3 closure (2026-05-03) flipped basophil sign + monotone direction
        from prior +0.10/+1 typo. Per clinical rationale (basopenia marks active
        disease -> higher failure -> negative coefficient on raw count)."""
        basophil = next(m for m in SCENARIO_C_MANIFEST if m.name == "basophil_count_cells_ul")
        assert basophil.coefficient < 0
        assert basophil.monotone_direction == -1

    def test_h1_inclusion_features_marked_noise(self) -> None:
        """prior_h1_antihistamine_* features are inclusion-criterion (all
        patients have value=1 by definition); coefficient=0 + is_noise=True
        per FeatureManifest invariant."""
        for name in (
            "prior_h1_antihistamine_standard_dose",
            "prior_h1_antihistamine_4x_dose_failed",
        ):
            feat = next(m for m in SCENARIO_C_MANIFEST if m.name == name)
            assert feat.coefficient == 0.0
            assert feat.is_noise is True


class TestScenarioCGeneration:
    def test_generates_dataset_with_expected_shape(self) -> None:
        ds = generate_scenario(ScenarioName.C_TREATMENT_CSU_RESPONSE, seed=42, n_total=2000)
        assert ds.X_train.shape[1] == 60

    def test_realized_prevalence_in_band(self) -> None:
        ds = generate_scenario(ScenarioName.C_TREATMENT_CSU_RESPONSE, seed=42, n_total=6000)
        assert abs(ds.metadata.realized_prevalence - 0.40) < 0.025

    def test_byte_identical_reruns(self) -> None:
        ds1 = generate_scenario(ScenarioName.C_TREATMENT_CSU_RESPONSE, seed=42, n_total=2000)
        ds2 = generate_scenario(ScenarioName.C_TREATMENT_CSU_RESPONSE, seed=42, n_total=2000)
        np.testing.assert_array_equal(ds1.X_train, ds2.X_train)
        np.testing.assert_array_equal(ds1.y_train, ds2.y_train)


@pytest.mark.slow
class TestScenarioCAUCBandRegression:
    """AUC band acceptance with 9/10 threshold per shard 05 §D + shard 09 §A.2."""

    def test_lr_auc_band_9_of_10_seeds(self) -> None:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score

        aucs = []
        for seed in range(10):
            ds = generate_scenario(ScenarioName.C_TREATMENT_CSU_RESPONSE, seed=seed, n_total=6000)
            clf = LogisticRegression(max_iter=3000, C=1.0)
            clf.fit(ds.X_train, ds.y_train)
            prob = clf.predict_proba(ds.X_test)[:, 1]
            aucs.append(roc_auc_score(ds.y_test, prob))
        in_band = sum(1 for a in aucs if 0.82 <= a <= 0.88)
        assert in_band >= 9, f"Only {in_band}/10 seeds in [0.82, 0.88]; aucs={aucs}"
