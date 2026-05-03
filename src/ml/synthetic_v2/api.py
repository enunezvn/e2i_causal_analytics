"""Public API surface for the synthetic_v2 generator (shard 01 §B + shard 02 §E).

Exports:

- ``ScenarioMetadata``: frozen audit record carrying solved intercept,
  realized prevalence, feature manifest, and SHA-256 fingerprint.
- ``SyntheticDataset``: frozen container with ``X_*`` / ``y_*`` arrays,
  full-cohort ``stratify`` key, and metadata.
- ``generate_scenario``: end-to-end entry point — registry lookup → feature
  sampling → block correlation → prevalence-calibrated label generation →
  stratified split → train-stats z-score → fingerprint.

Determinism: every random draw flows through a single
``np.random.Generator`` seeded from the public ``seed`` argument.  No
module-level seeding, no ``np.random.seed()`` calls — guaranteed
byte-identical output across processes for the same ``(scenario, seed,
n_total, ratios)`` tuple (shard 01 §C.1).

The registry consumed for dispatch is ``SCENARIO_REGISTRY`` from
``src.ml.synthetic_v2.scenarios`` — populated by commits 07 / 08 / 09 with
the A / B / C scenario factories.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

from src.ml.synthetic_v2.dgp import (
    _sigmoid,
    apply_block_correlation,
    solve_intercept,
    standardize_train_val_test,
)
from src.ml.synthetic_v2.manifest import FeatureManifest, manifest_to_jsonable
from src.ml.synthetic_v2.scenarios import SCENARIO_REGISTRY, ScenarioName
from src.ml.synthetic_v2.splits import stratified_train_val_test_split


@dataclass(frozen=True)
class ScenarioMetadata:
    """Audit-friendly metadata about how a dataset was generated (shard 01 §B.2).

    Frozen so consumers (Phase 1 W4 multi-disease runner, monotone-LightGBM)
    can rely on immutability between scenarios.
    """

    scenario: ScenarioName
    seed: int
    n_total: int
    n_train: int
    n_val: int
    n_test: int
    realized_prevalence: float
    target_prevalence: float
    feature_names: tuple[str, ...]
    monotone_vector: tuple[int, ...]
    feature_manifest: tuple[FeatureManifest, ...]
    target_auc_band: tuple[float, float]
    intercept: float
    slope_multiplier: float
    correlation_strength: float
    audit_fingerprint: str


@dataclass(frozen=True)
class SyntheticDataset:
    """One scenario's full output: arrays + metadata (shard 01 §B.2).

    ``stratify`` is the full-cohort outcome key (pre-split ordering),
    suitable as the ``y`` arg to ``StratifiedKFold(...).split(X, y)``
    per shard 21 §B's ``RepeatedStratifiedSplitter``.
    """

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    stratify: np.ndarray
    metadata: ScenarioMetadata


def _fingerprint(
    scenario: ScenarioName,
    seed: int,
    n_total: int,
    manifest: tuple[FeatureManifest, ...],
) -> str:
    """SHA-256 over (scenario, seed, n_total, manifest) (shard 01 §C.6).

    Stable across processes (uses SHA-256, not Python's salted ``hash()``).
    Two calls with the same inputs produce the same fingerprint; any change
    (new feature, coefficient bump, scenario rename) flips it. Used for
    cache invalidation in ``Phase1MultiDiseaseRunner``.
    """
    payload = {
        "scenario": scenario.value,
        "seed": seed,
        "n_total": n_total,
        "manifest": manifest_to_jsonable(manifest),
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def generate_scenario(
    scenario: ScenarioName,
    *,
    seed: int = 42,
    n_total: int | None = None,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
) -> SyntheticDataset:
    """Generate one scenario's labeled dataset (shard 01 §B.3 + shard 02 §E).

    Parameters
    ----------
    scenario
        Which scenario to generate. See ``ScenarioName`` for choices.
    seed
        Random seed. Same seed → byte-identical output.
    n_total
        Total cohort size. Defaults to the scenario's
        ``builder.default_n_total`` (typically 6000).
    train_ratio, val_ratio, test_ratio
        Must sum to 1.0 within 1e-9. Defaults: 0.6 / 0.2 / 0.2.

    Returns
    -------
    SyntheticDataset
        Frozen dataset with ``X_*`` / ``y_*`` arrays + metadata.

    Raises
    ------
    KeyError
        If ``scenario`` is not registered in ``SCENARIO_REGISTRY`` (i.e.,
        the per-scenario builder commit hasn't landed yet).
    ValueError
        If ratios do not sum to 1.0 or ``n_total`` is below the
        ``min_n_total`` floor of 100 (too small for stratified splits at
        low prevalence).
    """
    if scenario not in SCENARIO_REGISTRY:
        raise KeyError(
            f"scenario {scenario!r} is not registered; "
            f"available: {sorted(s.value for s in SCENARIO_REGISTRY.keys())}"
        )
    builder = SCENARIO_REGISTRY[scenario]()
    builder.validate_manifest_alignment()

    resolved_n_total = n_total if n_total is not None else builder.default_n_total
    if resolved_n_total < 100:
        raise ValueError(
            f"n_total={resolved_n_total} is below the safety floor of 100 "
            "(stratified splits at low prevalence become degenerate)."
        )
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-9:
        raise ValueError(
            "train_ratio + val_ratio + test_ratio must sum to 1.0; got "
            f"{train_ratio} + {val_ratio} + {test_ratio} = "
            f"{train_ratio + val_ratio + test_ratio}"
        )

    rng = np.random.default_rng(seed)

    # 1. Sample independent features per manifest
    X_raw = builder.sample_features(rng, resolved_n_total)
    if X_raw.shape != (resolved_n_total, builder.n_features):
        raise ValueError(
            f"builder.sample_features returned shape {X_raw.shape}; "
            f"expected ({resolved_n_total}, {builder.n_features})"
        )

    # 2. Inject correlation blocks
    X_corr = apply_block_correlation(rng, X_raw, builder.correlation_blocks)

    # 3. Compute logits (with slope multiplier) and solve intercept
    coefs = np.array(
        [m.coefficient for m in builder.feature_manifest],
        dtype=np.float64,
    ) * builder.slope_multiplier
    intercept = solve_intercept(X_corr, coefs, builder.target_prevalence)

    # 4. Sample labels via inverse-CDF on the calibrated sigmoid
    p = _sigmoid(X_corr @ coefs + intercept)
    y = (rng.uniform(size=resolved_n_total) < p).astype(np.int64)

    # 5. Stratified train/val/test split
    X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test = (
        stratified_train_val_test_split(
            X_corr,
            y,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )
    )

    # 6. Z-score standardize using train statistics only (no leakage)
    X_train, X_val, X_test, _, _ = standardize_train_val_test(
        X_train_raw, X_val_raw, X_test_raw
    )

    # 7. Stratify key — full-cohort, pre-split ordering (shard 21 §B contract)
    stratify = y.copy()

    # 8. Audit fingerprint
    fp = _fingerprint(scenario, seed, resolved_n_total, builder.feature_manifest)

    metadata = ScenarioMetadata(
        scenario=scenario,
        seed=seed,
        n_total=resolved_n_total,
        n_train=int(X_train.shape[0]),
        n_val=int(X_val.shape[0]),
        n_test=int(X_test.shape[0]),
        realized_prevalence=float(y.mean()),
        target_prevalence=builder.target_prevalence,
        feature_names=tuple(m.name for m in builder.feature_manifest),
        monotone_vector=tuple(m.monotone_direction for m in builder.feature_manifest),
        feature_manifest=tuple(builder.feature_manifest),
        target_auc_band=builder.target_auc_band,
        intercept=float(intercept),
        slope_multiplier=builder.slope_multiplier,
        correlation_strength=builder.correlation_strength,
        audit_fingerprint=fp,
    )

    return SyntheticDataset(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        stratify=stratify,
        metadata=metadata,
    )
