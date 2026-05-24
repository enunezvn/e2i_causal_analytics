"""Unit tests for the opt-in synthetic-augmentation node.

``augment_training_data`` concatenates a reviewed Phase-3 preview cohort into
the TRAINING split only, with strict schema validation. Covered here:

* opt-in / no-op semantics (no path, upstream error)
* happy path (DataFrame + ndarray training matrices) — counts, audit, and the
  invariant that validation/test/holdout are never in the returned patch
* refusals: feature-name mismatch, feature-count mismatch, missing file,
  missing arrays
* advisory failure (a corrupt cohort never raises)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import pytest

from src.agents.ml_foundation.model_trainer.nodes.augment_training_data import (
    augment_training_data,
)


def _write_cohort(
    out_dir: Path,
    *,
    n: int = 50,
    n_features: int = 3,
    feature_names: Optional[List[str]] = None,
    fingerprint: str = "fp123",
    with_metadata: bool = True,
) -> Path:
    """Write a Phase-3-shaped preview cohort (.npz + sibling metadata.json)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    npz = out_dir / "preview_cohort.npz"
    rng = np.random.default_rng(0)
    np.savez(
        npz,
        X_train=rng.normal(size=(n, n_features)),
        y_train=rng.integers(0, 2, size=n),
        X_val=rng.normal(size=(10, n_features)),
        y_val=rng.integers(0, 2, size=10),
        X_test=rng.normal(size=(10, n_features)),
        y_test=rng.integers(0, 2, size=10),
    )
    if with_metadata:
        names = feature_names if feature_names is not None else [f"f{i}" for i in range(n_features)]
        (out_dir / "preview_metadata.json").write_text(
            json.dumps({"feature_names": names, "audit_fingerprint": fingerprint})
        )
    return npz


def _real_train(n: int = 200, cols: Sequence[str] = ("f0", "f1", "f2"), as_frame: bool = True):
    rng = np.random.default_rng(1)
    X: Any = rng.normal(size=(n, len(cols)))
    if as_frame:
        X = pd.DataFrame(X, columns=list(cols))
        y: Any = pd.Series(rng.integers(0, 2, size=n), name="target")
    else:
        y = rng.integers(0, 2, size=n)
    return {"X": X, "y": y, "row_count": n}


def _state(train_data: Dict[str, Any], path: Optional[Path], **extra: Any) -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "train_data": train_data,
        # Sentinels that MUST survive untouched (never in the returned patch).
        "validation_data": {"X": pd.DataFrame({"f0": [0.0]}), "y": pd.Series([0]), "row_count": 1},
        "test_data": {"X": pd.DataFrame({"f0": [0.0]}), "y": pd.Series([1]), "row_count": 1},
        "holdout_data": {"X": pd.DataFrame({"f0": [0.0]}), "y": pd.Series([0]), "row_count": 1},
        "augmentation_data_path": str(path) if path is not None else None,
    }
    state.update(extra)
    return state


@pytest.mark.asyncio
async def test_no_path_is_noop() -> None:
    patch = await augment_training_data(_state(_real_train(), None))
    assert patch == {}


@pytest.mark.asyncio
async def test_upstream_error_is_noop(tmp_path) -> None:
    cohort = _write_cohort(tmp_path / "prev")
    patch = await augment_training_data(_state(_real_train(), cohort, error="boom"))
    assert patch == {}


@pytest.mark.asyncio
async def test_happy_path_dataframe(tmp_path) -> None:
    cohort = _write_cohort(tmp_path / "prev", n=50, feature_names=["f0", "f1", "f2"])
    patch = await augment_training_data(_state(_real_train(200), cohort))

    assert patch["augmentation_applied"] is True
    assert patch["augmentation_n_original"] == 200
    assert patch["augmentation_n_synthetic"] == 50
    assert patch["train_samples"] == 250
    assert patch["augmentation_fingerprint"] == "fp123"
    assert patch["augmentation_skip_reason"] is None

    aug = patch["train_data"]
    assert aug["row_count"] == 250
    assert isinstance(aug["X"], pd.DataFrame)
    assert list(aug["X"].columns) == ["f0", "f1", "f2"]
    assert len(aug["X"]) == 250
    assert len(aug["y"]) == 250

    # Load-bearing invariant: only the training split is touched.
    assert "validation_data" not in patch
    assert "test_data" not in patch
    assert "holdout_data" not in patch


@pytest.mark.asyncio
async def test_happy_path_ndarray_matrix(tmp_path) -> None:
    cohort = _write_cohort(tmp_path / "prev", n=40, n_features=3)
    patch = await augment_training_data(_state(_real_train(100, as_frame=False), cohort))

    assert patch["augmentation_applied"] is True
    assert patch["train_samples"] == 140
    aug_X = patch["train_data"]["X"]
    assert isinstance(aug_X, np.ndarray)
    assert aug_X.shape == (140, 3)


@pytest.mark.asyncio
async def test_feature_name_mismatch_is_refused(tmp_path) -> None:
    cohort = _write_cohort(tmp_path / "prev", feature_names=["x0", "x1", "x2"])
    patch = await augment_training_data(_state(_real_train(cols=("f0", "f1", "f2")), cohort))

    assert patch["augmentation_applied"] is False
    assert "mismatch" in patch["augmentation_skip_reason"].lower()
    assert "train_data" not in patch  # untouched on refusal


@pytest.mark.asyncio
async def test_feature_count_mismatch_is_refused(tmp_path) -> None:
    cohort = _write_cohort(tmp_path / "prev", n_features=4)  # real has 3
    patch = await augment_training_data(_state(_real_train(cols=("f0", "f1", "f2")), cohort))

    assert patch["augmentation_applied"] is False
    assert "feature-count mismatch" in patch["augmentation_skip_reason"]
    assert "train_data" not in patch


@pytest.mark.asyncio
async def test_missing_file_is_refused(tmp_path) -> None:
    patch = await augment_training_data(_state(_real_train(), tmp_path / "does_not_exist.npz"))
    assert patch["augmentation_applied"] is False
    assert "does not exist" in patch["augmentation_skip_reason"]


@pytest.mark.asyncio
async def test_npz_missing_arrays_is_refused(tmp_path) -> None:
    bad = tmp_path / "bad.npz"
    np.savez(bad, X_val=np.zeros((3, 3)))  # no X_train/y_train
    patch = await augment_training_data(_state(_real_train(), bad))
    assert patch["augmentation_applied"] is False
    assert "missing X_train/y_train" in patch["augmentation_skip_reason"]


@pytest.mark.asyncio
async def test_corrupt_cohort_is_advisory(tmp_path) -> None:
    corrupt = tmp_path / "corrupt.npz"
    corrupt.write_text("this is not a real npz file")
    # Must NOT raise — advisory failure path.
    patch = await augment_training_data(_state(_real_train(), corrupt))
    assert patch["augmentation_applied"] is False
    assert patch["augmentation_skip_reason"]
