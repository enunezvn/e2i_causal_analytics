"""Tests for Scenario B — IgAN 5y ESKD progression (Fabhalta).

Per shard 04 Section D acceptance criteria. Note: the AUC band acceptance
is relaxed to 8/10 (not 9/10) per shard 04 §B.4 risk note about
low-prevalence variance.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.ml.synthetic_v2 import ScenarioName, generate_scenario
from src.ml.synthetic_v2.scenarios import SCENARIO_REGISTRY
from src.ml.synthetic_v2.scenarios.scenario_b import (
    SCENARIO_B_CORRELATION_BLOCKS,
    SCENARIO_B_MANIFEST,
    SLOPE_MULTIPLIER,
    ScenarioBBuilder,
)


class TestScenarioBRegistration:
    def test_builder_in_registry(self) -> None:
        assert ScenarioName.B_SCREENING_IGAN_ESKD in SCENARIO_REGISTRY
        assert SCENARIO_REGISTRY[ScenarioName.B_SCREENING_IGAN_ESKD] is ScenarioBBuilder


class TestScenarioBManifestAlignment:
    def test_manifest_has_25_features(self) -> None:
        assert len(SCENARIO_B_MANIFEST) == 25

    def test_n_features_matches_manifest(self) -> None:
        builder = ScenarioBBuilder()
        assert builder.n_features == len(SCENARIO_B_MANIFEST)
        builder.validate_manifest_alignment()

    def test_all_feature_names_unique(self) -> None:
        names = [m.name for m in SCENARIO_B_MANIFEST]
        assert len(set(names)) == 25

    def test_target_prevalence_locked(self) -> None:
        assert ScenarioBBuilder().target_prevalence == 0.05

    def test_target_auc_band_locked(self) -> None:
        assert ScenarioBBuilder().target_auc_band == (0.72, 0.78)

    def test_slope_multiplier_locked(self) -> None:
        assert SLOPE_MULTIPLIER == pytest.approx(0.060)

    def test_correlation_blocks_within_n_features_range(self) -> None:
        for cols, _ in SCENARIO_B_CORRELATION_BLOCKS:
            for c in cols:
                assert 0 <= c < 25

    def test_fabhalta_anchor_signed_negative(self) -> None:
        """ACEi/ARB and SGLT2i must protect against ESKD progression
        (Fabhalta-franchise narrative: iptacopan layered on top of these).
        """
        anchor_acei = next(m for m in SCENARIO_B_MANIFEST if m.name == "on_acei_or_arb")
        anchor_sglt2 = next(m for m in SCENARIO_B_MANIFEST if m.name == "on_sglt2_inhibitor")
        assert anchor_acei.coefficient < 0
        assert anchor_acei.monotone_direction == -1
        assert "FABHALTA" in anchor_acei.clinical_justification.upper()
        assert anchor_sglt2.coefficient < 0
        assert anchor_sglt2.monotone_direction == -1


class TestScenarioBGeneration:
    def test_generates_dataset_with_expected_shape(self) -> None:
        ds = generate_scenario(ScenarioName.B_SCREENING_IGAN_ESKD, seed=42, n_total=2000)
        assert ds.X_train.shape[1] == 25

    def test_realized_prevalence_in_band_at_low_prev(self) -> None:
        """At prev=0.05 the band is ±0.02 absolute (40% relative)."""
        ds = generate_scenario(ScenarioName.B_SCREENING_IGAN_ESKD, seed=42, n_total=6000)
        assert abs(ds.metadata.realized_prevalence - 0.05) < 0.02

    def test_byte_identical_reruns(self) -> None:
        ds1 = generate_scenario(ScenarioName.B_SCREENING_IGAN_ESKD, seed=42, n_total=2000)
        ds2 = generate_scenario(ScenarioName.B_SCREENING_IGAN_ESKD, seed=42, n_total=2000)
        np.testing.assert_array_equal(ds1.X_train, ds2.X_train)
        np.testing.assert_array_equal(ds1.y_train, ds2.y_train)


@pytest.mark.slow
class TestScenarioBAUCBandRegression:
    """AUC band acceptance with 8/10 threshold per shard 04 §B.4 + shard 09 R-1."""

    def test_lr_auc_band_8_of_10_seeds(self) -> None:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score

        aucs = []
        for seed in range(10):
            ds = generate_scenario(ScenarioName.B_SCREENING_IGAN_ESKD, seed=seed, n_total=6000)
            clf = LogisticRegression(max_iter=2000, C=1.0)
            clf.fit(ds.X_train, ds.y_train)
            prob = clf.predict_proba(ds.X_test)[:, 1]
            aucs.append(roc_auc_score(ds.y_test, prob))
        in_band = sum(1 for a in aucs if 0.72 <= a <= 0.78)
        # 8/10 acceptance per shard 04 §B.4 low-prevalence variance note
        assert in_band >= 8, f"Only {in_band}/10 seeds in [0.72, 0.78]; aucs={aucs}"
