"""Tests for ``ScenarioBuilder`` ABC (shard 01 §B.4)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from src.ml.synthetic_v2.manifest import FeatureManifest
from src.ml.synthetic_v2.scenarios._base import ScenarioBuilder


def _two_feature_manifest() -> tuple[FeatureManifest, ...]:
    return (
        FeatureManifest(
            name="age",
            distribution="normal",
            distribution_params={"loc": 50.0, "scale": 10.0},
            coefficient=0.3,
            monotone_direction=1,
            is_noise=False,
            clinical_justification="age signal placeholder",
            citation_strength="moderate",
        ),
        FeatureManifest(
            name="bp",
            distribution="bernoulli",
            distribution_params={"p": 0.4},
            coefficient=-0.2,
            monotone_direction=-1,
            is_noise=False,
            clinical_justification="bp signal placeholder",
            citation_strength="moderate",
        ),
    )


class _StubScenario(ScenarioBuilder):
    """Minimal concrete subclass for ABC behavior testing.

    Uses string for `name` (not real ScenarioName) so the test file doesn't
    need commit-05 enum.
    """

    def __init__(
        self,
        *,
        manifest: tuple[FeatureManifest, ...] | None = None,
        n_features: int | None = None,
        slope_multiplier: float = 1.0,
    ) -> None:
        self._manifest = manifest if manifest is not None else _two_feature_manifest()
        self._n_features = n_features if n_features is not None else len(self._manifest)
        self._slope = slope_multiplier

    @property
    def name(self) -> Any:  # type: ignore[override]
        return "stub_scenario"

    @property
    def target_prevalence(self) -> float:
        return 0.20

    @property
    def target_auc_band(self) -> tuple[float, float]:
        return (0.70, 0.80)

    @property
    def n_features(self) -> int:
        return self._n_features

    @property
    def correlation_strength(self) -> float:
        return 0.3

    @property
    def slope_multiplier(self) -> float:
        return self._slope

    @property
    def feature_manifest(self) -> tuple[FeatureManifest, ...]:
        return self._manifest

    @property
    def default_n_total(self) -> int:
        return 1000

    @property
    def correlation_blocks(self) -> list[tuple[list[int], float]]:
        return []

    def sample_features(self, rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.normal(size=(n, self._n_features))


class TestABCEnforcement:
    def test_cannot_instantiate_abc_directly(self) -> None:
        with pytest.raises(TypeError, match="abstract"):
            ScenarioBuilder()  # type: ignore[abstract]

    def test_subclass_missing_required_property_cannot_instantiate(self) -> None:
        class _Incomplete(ScenarioBuilder):
            @property
            def name(self) -> Any:  # type: ignore[override]
                return "incomplete"

        with pytest.raises(TypeError, match="abstract"):
            _Incomplete()  # type: ignore[abstract]

    def test_full_subclass_can_instantiate(self) -> None:
        s = _StubScenario()
        assert s.name == "stub_scenario"
        assert s.target_prevalence == 0.20
        assert s.target_auc_band == (0.70, 0.80)
        assert s.n_features == 2
        assert s.correlation_strength == 0.3
        assert s.slope_multiplier == 1.0
        assert s.default_n_total == 1000
        assert s.correlation_blocks == []
        assert len(s.feature_manifest) == 2


class TestComputeLogitsDefault:
    def test_default_returns_x_at_coefs_plus_intercept(self) -> None:
        s = _StubScenario()
        rng = np.random.default_rng(42)
        X = rng.normal(size=(50, 2))
        intercept = 0.5
        logits = s.compute_logits(X, intercept)
        expected = X @ np.array([0.3, -0.2]) + 0.5
        np.testing.assert_allclose(logits, expected, atol=1e-12)

    def test_slope_multiplier_scales_coefficients(self) -> None:
        s = _StubScenario(slope_multiplier=2.0)
        rng = np.random.default_rng(42)
        X = rng.normal(size=(50, 2))
        logits = s.compute_logits(X, intercept=0.0)
        expected = X @ (np.array([0.3, -0.2]) * 2.0)
        np.testing.assert_allclose(logits, expected, atol=1e-12)

    def test_compute_logits_rejects_wrong_shape(self) -> None:
        s = _StubScenario()
        with pytest.raises(ValueError, match="X must be 2-D"):
            s.compute_logits(np.zeros(5), intercept=0.0)
        with pytest.raises(ValueError, match="does not match n_features"):
            s.compute_logits(np.zeros((10, 3)), intercept=0.0)

    def test_compute_logits_returns_one_d_array(self) -> None:
        s = _StubScenario()
        X = np.zeros((10, 2))
        logits = s.compute_logits(X, intercept=0.7)
        assert logits.shape == (10,)
        np.testing.assert_allclose(logits, 0.7, atol=1e-12)


class TestSubclassMayOverrideComputeLogits:
    def test_override_takes_precedence(self) -> None:
        class _NonlinearScenario(_StubScenario):
            def compute_logits(self, X: np.ndarray, intercept: float) -> np.ndarray:
                # Quadratic interaction: x1 * x2
                return X[:, 0] * X[:, 1] + intercept

        s = _NonlinearScenario()
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        logits = s.compute_logits(X, intercept=1.0)
        np.testing.assert_allclose(logits, [3.0, 13.0])


class TestValidateManifestAlignment:
    def test_aligned_manifest_passes(self) -> None:
        s = _StubScenario()
        s.validate_manifest_alignment()

    def test_length_mismatch_raises(self) -> None:
        # 2-feature manifest but n_features lies as 3
        s = _StubScenario(n_features=3)
        with pytest.raises(ValueError, match="does not match n_features"):
            s.validate_manifest_alignment()

    def test_duplicate_names_raises(self) -> None:
        manifest = (
            FeatureManifest(
                name="age",
                distribution="normal",
                distribution_params={"loc": 50.0, "scale": 10.0},
                coefficient=0.3,
                monotone_direction=1,
                is_noise=False,
                clinical_justification="placeholder",
                citation_strength="moderate",
            ),
            FeatureManifest(
                name="age",  # duplicate
                distribution="normal",
                distribution_params={"loc": 60.0, "scale": 10.0},
                coefficient=0.4,
                monotone_direction=1,
                is_noise=False,
                clinical_justification="placeholder duplicate",
                citation_strength="moderate",
            ),
        )
        s = _StubScenario(manifest=manifest)
        with pytest.raises(ValueError, match="duplicate names"):
            s.validate_manifest_alignment()


class TestSampleFeaturesConsumesGenerator:
    def test_sample_features_returns_correct_shape(self) -> None:
        s = _StubScenario()
        rng = np.random.default_rng(42)
        X = s.sample_features(rng, n=100)
        assert X.shape == (100, 2)

    def test_sample_features_determinism(self) -> None:
        s = _StubScenario()
        X1 = s.sample_features(np.random.default_rng(7), n=50)
        X2 = s.sample_features(np.random.default_rng(7), n=50)
        np.testing.assert_array_equal(X1, X2)
