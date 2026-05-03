"""Tests for ``api.generate_scenario`` end-to-end (shard 01 §B.3 + shard 02 §E)."""

from __future__ import annotations

import numpy as np
import pytest

from src.ml.synthetic_v2 import (
    ScenarioMetadata,
    ScenarioName,
    SyntheticDataset,
    generate_scenario,
)
from src.ml.synthetic_v2.api import _fingerprint
from src.ml.synthetic_v2.scenarios import SCENARIO_REGISTRY


class TestRegistryLookup:
    def test_unknown_scenario_raises_keyerror(self) -> None:
        # SCENARIO_REGISTRY is empty by default at this commit
        with pytest.raises(KeyError, match="not registered"):
            generate_scenario(ScenarioName.A_DIAGNOSTIC_BC_IDFS, seed=42)

    def test_keyerror_lists_available_scenarios(self) -> None:
        with pytest.raises(KeyError) as excinfo:
            generate_scenario(ScenarioName.B_SCREENING_IGAN_ESKD, seed=42)
        assert "available:" in str(excinfo.value)


class TestGenerateScenarioReturnsFrozenDataset:
    def test_returns_synthetic_dataset(self, dummy_scenario_registered: ScenarioName) -> None:
        ds = generate_scenario(dummy_scenario_registered, seed=42)
        assert isinstance(ds, SyntheticDataset)
        assert isinstance(ds.metadata, ScenarioMetadata)

    def test_dataset_arrays_have_expected_shapes(
        self, dummy_scenario_registered: ScenarioName
    ) -> None:
        ds = generate_scenario(dummy_scenario_registered, seed=42, n_total=1000)
        assert ds.X_train.shape[1] == 8
        assert ds.X_val.shape[1] == 8
        assert ds.X_test.shape[1] == 8
        assert ds.X_train.shape[0] + ds.X_val.shape[0] + ds.X_test.shape[0] == 1000
        assert ds.y_train.shape[0] == ds.X_train.shape[0]
        assert ds.y_val.shape[0] == ds.X_val.shape[0]
        assert ds.y_test.shape[0] == ds.X_test.shape[0]
        assert ds.stratify.shape == (1000,)

    def test_dataset_y_arrays_are_binary_int64(
        self, dummy_scenario_registered: ScenarioName
    ) -> None:
        ds = generate_scenario(dummy_scenario_registered, seed=42)
        for y in (ds.y_train, ds.y_val, ds.y_test):
            assert y.dtype == np.int64
            assert set(np.unique(y)) <= {0, 1}

    def test_dataset_metadata_populated(
        self, dummy_scenario_registered: ScenarioName
    ) -> None:
        ds = generate_scenario(dummy_scenario_registered, seed=42, n_total=1500)
        m = ds.metadata
        assert m.scenario == dummy_scenario_registered
        assert m.seed == 42
        assert m.n_total == 1500
        assert m.n_train + m.n_val + m.n_test == 1500
        assert m.target_prevalence == 0.20
        assert m.target_auc_band == (0.65, 0.85)
        assert m.slope_multiplier == 1.0
        assert m.correlation_strength == 0.3
        assert len(m.feature_names) == 8
        assert len(m.monotone_vector) == 8
        assert len(m.feature_manifest) == 8
        assert m.feature_names[0] == "signal_a"
        assert m.feature_names[-1] == "noise_h"
        assert isinstance(m.audit_fingerprint, str)
        assert len(m.audit_fingerprint) == 64  # SHA-256 hex


class TestPrevalenceCalibration:
    def test_realized_prevalence_within_band(
        self, dummy_scenario_registered: ScenarioName
    ) -> None:
        ds = generate_scenario(dummy_scenario_registered, seed=42, n_total=3000)
        # ±0.02 band per shard 01 §C.2 acceptance contract
        assert abs(ds.metadata.realized_prevalence - 0.20) < 0.02

    @pytest.mark.parametrize("seed", [1, 7, 17, 42, 99])
    def test_realized_prevalence_within_band_across_seeds(
        self, dummy_scenario_registered: ScenarioName, seed: int
    ) -> None:
        ds = generate_scenario(dummy_scenario_registered, seed=seed, n_total=3000)
        assert abs(ds.metadata.realized_prevalence - 0.20) < 0.025


class TestDeterminism:
    def test_byte_identical_reruns(
        self, dummy_scenario_registered: ScenarioName
    ) -> None:
        ds1 = generate_scenario(dummy_scenario_registered, seed=42, n_total=1500)
        ds2 = generate_scenario(dummy_scenario_registered, seed=42, n_total=1500)
        np.testing.assert_array_equal(ds1.X_train, ds2.X_train)
        np.testing.assert_array_equal(ds1.y_train, ds2.y_train)
        np.testing.assert_array_equal(ds1.X_val, ds2.X_val)
        np.testing.assert_array_equal(ds1.y_val, ds2.y_val)
        np.testing.assert_array_equal(ds1.X_test, ds2.X_test)
        np.testing.assert_array_equal(ds1.y_test, ds2.y_test)
        np.testing.assert_array_equal(ds1.stratify, ds2.stratify)

    def test_metadata_byte_identical(
        self, dummy_scenario_registered: ScenarioName
    ) -> None:
        ds1 = generate_scenario(dummy_scenario_registered, seed=42)
        ds2 = generate_scenario(dummy_scenario_registered, seed=42)
        assert ds1.metadata == ds2.metadata

    def test_different_seeds_differ(
        self, dummy_scenario_registered: ScenarioName
    ) -> None:
        ds1 = generate_scenario(dummy_scenario_registered, seed=42)
        ds2 = generate_scenario(dummy_scenario_registered, seed=43)
        assert not np.array_equal(ds1.X_train, ds2.X_train)


class TestArgumentValidation:
    def test_ratios_must_sum_to_one(
        self, dummy_scenario_registered: ScenarioName
    ) -> None:
        with pytest.raises(ValueError, match="must sum to 1.0"):
            generate_scenario(
                dummy_scenario_registered,
                seed=42,
                train_ratio=0.5,
                val_ratio=0.2,
                test_ratio=0.2,
            )

    def test_n_total_below_floor_raises(
        self, dummy_scenario_registered: ScenarioName
    ) -> None:
        with pytest.raises(ValueError, match="below the safety floor"):
            generate_scenario(dummy_scenario_registered, seed=42, n_total=50)

    def test_n_total_default_uses_builder_default(
        self, dummy_scenario_registered: ScenarioName
    ) -> None:
        ds = generate_scenario(dummy_scenario_registered, seed=42)  # no n_total
        assert ds.metadata.n_total == 1500  # _DummyScenario.default_n_total


class TestFingerprint:
    def test_fingerprint_is_sha256_hex(
        self, dummy_scenario_registered: ScenarioName
    ) -> None:
        ds = generate_scenario(dummy_scenario_registered, seed=42)
        fp = ds.metadata.audit_fingerprint
        assert len(fp) == 64
        # All chars are valid hex
        assert all(c in "0123456789abcdef" for c in fp)

    def test_fingerprint_stable_across_processes(
        self, dummy_scenario_registered: ScenarioName
    ) -> None:
        ds1 = generate_scenario(dummy_scenario_registered, seed=42, n_total=1500)
        ds2 = generate_scenario(dummy_scenario_registered, seed=42, n_total=1500)
        assert ds1.metadata.audit_fingerprint == ds2.metadata.audit_fingerprint

    def test_fingerprint_changes_with_seed(
        self, dummy_scenario_registered: ScenarioName
    ) -> None:
        ds1 = generate_scenario(dummy_scenario_registered, seed=42)
        ds2 = generate_scenario(dummy_scenario_registered, seed=43)
        assert ds1.metadata.audit_fingerprint != ds2.metadata.audit_fingerprint

    def test_fingerprint_changes_with_n_total(
        self, dummy_scenario_registered: ScenarioName
    ) -> None:
        ds1 = generate_scenario(dummy_scenario_registered, seed=42, n_total=1500)
        ds2 = generate_scenario(dummy_scenario_registered, seed=42, n_total=2000)
        assert ds1.metadata.audit_fingerprint != ds2.metadata.audit_fingerprint

    def test_fingerprint_helper_directly(
        self, dummy_scenario_registered: ScenarioName
    ) -> None:
        builder = SCENARIO_REGISTRY[dummy_scenario_registered]()
        fp1 = _fingerprint(dummy_scenario_registered, 7, 1500, builder.feature_manifest)
        fp2 = _fingerprint(dummy_scenario_registered, 7, 1500, builder.feature_manifest)
        assert fp1 == fp2
        fp3 = _fingerprint(dummy_scenario_registered, 8, 1500, builder.feature_manifest)
        assert fp1 != fp3


class TestStandardizationContract:
    def test_x_train_zero_mean_unit_std(
        self, dummy_scenario_registered: ScenarioName
    ) -> None:
        ds = generate_scenario(dummy_scenario_registered, seed=42, n_total=2000)
        np.testing.assert_allclose(ds.X_train.mean(axis=0), 0.0, atol=1e-12)
        np.testing.assert_allclose(ds.X_train.std(axis=0, ddof=0), 1.0, atol=1e-12)

    def test_val_test_use_train_stats(
        self, dummy_scenario_registered: ScenarioName
    ) -> None:
        ds = generate_scenario(dummy_scenario_registered, seed=42, n_total=2000)
        # Val/test will not be exactly zero-mean (they use TRAIN stats),
        # but should be close since the cohort is i.i.d.
        assert np.all(np.abs(ds.X_val.mean(axis=0)) < 0.2)
        assert np.all(np.abs(ds.X_test.mean(axis=0)) < 0.2)


class TestStratifyKey:
    def test_stratify_is_full_cohort_outcome(
        self, dummy_scenario_registered: ScenarioName
    ) -> None:
        ds = generate_scenario(dummy_scenario_registered, seed=42, n_total=1500)
        assert ds.stratify.dtype == np.int64
        assert ds.stratify.shape == (1500,)
        assert set(np.unique(ds.stratify)) <= {0, 1}
        # Stratify key matches realized prevalence
        assert abs(ds.stratify.mean() - ds.metadata.realized_prevalence) < 1e-9
