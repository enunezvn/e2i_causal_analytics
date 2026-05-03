"""``ScenarioBuilder`` ABC (shard 01 §B.4).

Per-scenario subclasses (shards 03 / 04 / 05) fill in disease-specific
feature surface; the prevalence-calibration loop, correlation injection,
and dataset assembly are shared in ``dgp.py``.

Two of the abstract properties — ``default_n_total`` and ``correlation_blocks``
— were added per Codex I-2 closure 2026-05-03 (cycle-1 plan review): they
were called by ``dgp.py`` but originally undeclared, leaving mypy strict
blind to missing implementations in new scenario authors' code.

``ScenarioName`` is imported from ``scenarios/__init__.py``; it lands in the
same commit as this module per the bundled commits-04+05 ordering.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from src.ml.synthetic_v2.manifest import FeatureManifest
from src.ml.synthetic_v2.scenarios import ScenarioName


class ScenarioBuilder(ABC):
    """Per-scenario DGP base class (shard 01 §B.4).

    Subclasses must declare:

    - ``name``: canonical ``ScenarioName`` enum value.
    - ``target_prevalence``: float in (0, 1) — outcome rate the intercept
      solver targets.
    - ``target_auc_band``: ``(low, high)`` tuple — regression test asserts
      realized AUC ∈ band on at least 9/10 seeds (shard 01 §C.3).
    - ``n_features``: int — total feature count (must equal
      ``len(feature_manifest)``).
    - ``correlation_strength``: float in [0, 1] — informational; carried
      into ``ScenarioMetadata`` for audit.
    - ``slope_multiplier``: float — multiplies manifest coefficients in
      ``compute_logits`` so the AUC band can be calibrated independently
      of correlation/prevalence (shard 02 §C).
    - ``feature_manifest``: tuple of ``FeatureManifest`` records, ordered
      identically to ``X.shape[1]``.
    - ``default_n_total``: int — canonical cohort size if caller doesn't
      override (shard 02 §E).
    - ``correlation_blocks``: list of ``(column_indices, target_pearson_r)``
      pairs — signed ``r`` accepted (shard 02 §A.2; subject to PSD
      constraint validated at correlation-injection time).
    - ``sample_features(rng, n)``: returns ``(n, n_features)`` raw
      feature matrix (pre-correlation, pre-standardization), with each
      column independently distributed per its manifest entry.

    Subclasses *may* override ``compute_logits`` for non-additive
    interactions; the default takes ``X @ (coefs * slope_multiplier) +
    intercept``.
    """

    @property
    @abstractmethod
    def name(self) -> ScenarioName:
        """Canonical scenario identifier (see ``ScenarioName`` enum)."""

    @property
    @abstractmethod
    def target_prevalence(self) -> float:
        """Outcome rate the intercept solver targets (must be in (0, 1))."""

    @property
    @abstractmethod
    def target_auc_band(self) -> tuple[float, float]:
        """``(low, high)`` tuple for AUC regression test (shard 01 §C.3)."""

    @property
    @abstractmethod
    def n_features(self) -> int:
        """Total feature count; must equal ``len(self.feature_manifest)``."""

    @property
    @abstractmethod
    def correlation_strength(self) -> float:
        """Informational ``[0, 1]`` summary carried into ``ScenarioMetadata``."""

    @property
    @abstractmethod
    def slope_multiplier(self) -> float:
        """Multiplier applied to manifest coefficients in ``compute_logits``."""

    @property
    @abstractmethod
    def feature_manifest(self) -> tuple[FeatureManifest, ...]:
        """Per-feature audit records, ordered identically to ``X.shape[1]``."""

    @property
    @abstractmethod
    def default_n_total(self) -> int:
        """Canonical cohort size if caller does not override (shard 02 §E)."""

    @property
    @abstractmethod
    def correlation_blocks(self) -> list[tuple[list[int], float]]:
        """Per-scenario block-correlation specification (shard 02 §A.2)."""

    @abstractmethod
    def sample_features(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """Return raw feature matrix ``(n, n_features)`` — pre-correlation,
        pre-standardization. Each column independently distributed per its
        manifest entry.
        """

    def compute_logits(self, X: np.ndarray, intercept: float) -> np.ndarray:
        """Default linear-additive logits with slope multiplier (shard 02 §C.1).

        ``coefs[i] = manifest[i].coefficient * slope_multiplier``. Returns
        ``X @ coefs + intercept``, shape ``(n,)``.

        Subclasses may override for multiplicative interactions, threshold
        nonlinearities, etc.
        """
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D; got shape {X.shape}")
        if X.shape[1] != self.n_features:
            raise ValueError(
                f"X.shape[1]={X.shape[1]} does not match n_features={self.n_features}"
            )
        coefs = np.array(
            [m.coefficient for m in self.feature_manifest],
            dtype=np.float64,
        ) * self.slope_multiplier
        return np.asarray(X @ coefs + intercept, dtype=np.float64)

    def validate_manifest_alignment(self) -> None:
        """Self-check: ``n_features == len(feature_manifest)`` and names unique.

        Subclass authors should call this in their ``__init__`` (or a test)
        to catch off-by-one drift between ``n_features`` and the manifest
        tuple length, plus duplicate feature names that would corrupt
        downstream LightGBM training.
        """
        manifest = self.feature_manifest
        if len(manifest) != self.n_features:
            raise ValueError(
                f"feature_manifest length {len(manifest)} does not match "
                f"n_features={self.n_features} for scenario {self.name!r}"
            )
        names = [m.name for m in manifest]
        if len(set(names)) != len(names):
            duplicates = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(
                f"feature_manifest has duplicate names {duplicates} for "
                f"scenario {self.name!r}"
            )
