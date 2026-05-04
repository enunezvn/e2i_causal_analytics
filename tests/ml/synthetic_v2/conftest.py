"""Shared fixtures for synthetic_v2 tests."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest

from src.ml.synthetic_v2.manifest import FeatureManifest
from src.ml.synthetic_v2.scenarios import SCENARIO_REGISTRY, ScenarioName
from src.ml.synthetic_v2.scenarios._base import ScenarioBuilder


class _DummyScenario(ScenarioBuilder):
    """Lightweight scenario for end-to-end api / determinism tests.

    8 i.i.d. normal features with mixed signal/noise mix; one 3-col
    correlation block (r=0.5) to exercise the correlation-injection path.
    """

    _MANIFEST: tuple[FeatureManifest, ...] = (
        FeatureManifest(
            name="signal_a",
            distribution="normal",
            distribution_params={"loc": 0.0, "scale": 1.0},
            coefficient=0.6,
            monotone_direction=1,
            is_noise=False,
            clinical_justification="dummy signal a",
            citation_strength="weak",
        ),
        FeatureManifest(
            name="signal_b",
            distribution="normal",
            distribution_params={"loc": 0.0, "scale": 1.0},
            coefficient=-0.4,
            monotone_direction=-1,
            is_noise=False,
            clinical_justification="dummy signal b",
            citation_strength="weak",
        ),
        FeatureManifest(
            name="signal_c",
            distribution="normal",
            distribution_params={"loc": 0.0, "scale": 1.0},
            coefficient=0.3,
            monotone_direction=1,
            is_noise=False,
            clinical_justification="dummy signal c",
            citation_strength="weak",
        ),
        FeatureManifest(
            name="signal_d",
            distribution="normal",
            distribution_params={"loc": 0.0, "scale": 1.0},
            coefficient=0.2,
            monotone_direction=1,
            is_noise=False,
            clinical_justification="dummy signal d",
            citation_strength="weak",
        ),
        FeatureManifest(
            name="noise_e",
            distribution="normal",
            distribution_params={"loc": 0.0, "scale": 1.0},
            coefficient=0.0,
            monotone_direction=0,
            is_noise=True,
            clinical_justification="dummy noise e",
            citation_strength="weak",
        ),
        FeatureManifest(
            name="noise_f",
            distribution="normal",
            distribution_params={"loc": 0.0, "scale": 1.0},
            coefficient=0.0,
            monotone_direction=0,
            is_noise=True,
            clinical_justification="dummy noise f",
            citation_strength="weak",
        ),
        FeatureManifest(
            name="noise_g",
            distribution="normal",
            distribution_params={"loc": 0.0, "scale": 1.0},
            coefficient=0.0,
            monotone_direction=0,
            is_noise=True,
            clinical_justification="dummy noise g",
            citation_strength="weak",
        ),
        FeatureManifest(
            name="noise_h",
            distribution="normal",
            distribution_params={"loc": 0.0, "scale": 1.0},
            coefficient=0.0,
            monotone_direction=0,
            is_noise=True,
            clinical_justification="dummy noise h",
            citation_strength="weak",
        ),
    )

    @property
    def name(self) -> ScenarioName:
        return ScenarioName.A_DIAGNOSTIC_BC_IDFS

    @property
    def target_prevalence(self) -> float:
        return 0.20

    @property
    def target_auc_band(self) -> tuple[float, float]:
        return (0.65, 0.85)

    @property
    def n_features(self) -> int:
        return 8

    @property
    def correlation_strength(self) -> float:
        return 0.3

    @property
    def slope_multiplier(self) -> float:
        return 1.0

    @property
    def feature_manifest(self) -> tuple[FeatureManifest, ...]:
        return self._MANIFEST

    @property
    def default_n_total(self) -> int:
        return 1500

    @property
    def correlation_blocks(self) -> list[tuple[list[int], float]]:
        return [([0, 1, 2], 0.5)]

    def sample_features(self, rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.normal(loc=0.0, scale=1.0, size=(n, self.n_features))


@pytest.fixture
def dummy_scenario_registered() -> Iterator[ScenarioName]:
    """Register ``_DummyScenario`` under ScenarioName.A_DIAGNOSTIC_BC_IDFS for the test."""
    name = ScenarioName.A_DIAGNOSTIC_BC_IDFS
    saved = SCENARIO_REGISTRY.get(name)
    SCENARIO_REGISTRY[name] = _DummyScenario
    try:
        yield name
    finally:
        if saved is None:
            SCENARIO_REGISTRY.pop(name, None)
        else:
            SCENARIO_REGISTRY[name] = saved
