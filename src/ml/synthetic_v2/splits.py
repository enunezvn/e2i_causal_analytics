"""Train/val/test stratified splitter (shard 01 §A, shard 02 §D).

Two-stage split that first holds out test, then splits the remainder into
train/val. Stratifies on ``y`` so class balance is preserved.

The stage-2 random seed is ``seed + 1`` (documented child seed) so the same
shuffling pattern doesn't apply to both stages, which would otherwise
correlate train/val internal ordering.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split


def stratified_train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    *,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray,
]:
    """Two-stage stratified split (shard 02 §D).

    Returns ``(X_train, X_val, X_test, y_train, y_val, y_test)``. Same seed
    yields the same split. Stratifies on ``y``.
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-9:
        raise ValueError(
            "train_ratio + val_ratio + test_ratio must sum to 1.0; "
            f"got {train_ratio} + {val_ratio} + {test_ratio} "
            f"= {train_ratio + val_ratio + test_ratio}"
        )
    for label, value in (("train_ratio", train_ratio), ("val_ratio", val_ratio), ("test_ratio", test_ratio)):
        if not 0.0 < value < 1.0:
            raise ValueError(f"{label} must be in (0, 1); got {value}")
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D; got shape {X.shape}")
    if y.ndim != 1 or y.shape[0] != X.shape[0]:
        raise ValueError(
            "y must be 1-D with len(y) == X.shape[0]; got "
            f"y.shape={y.shape}, X.shape={X.shape}"
        )

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_ratio, stratify=y, random_state=seed
    )
    val_size_rebased = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_size_rebased,
        stratify=y_temp,
        random_state=seed + 1,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
