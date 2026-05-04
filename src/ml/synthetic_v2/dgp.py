"""Shared DGP machinery (shard 02).

Hosts the per-scenario-agnostic primitives every ``ScenarioBuilder`` relies on:

- ``sample_one_feature``: per-feature i.i.d. sampling dispatcher (§A.1).
- ``apply_block_correlation``: Cholesky-injected block correlation (§A.2);
  signed Pearson ``r`` is accepted as long as the resulting target matrix is
  positive semi-definite. Non-PSD blocks raise ``ValueError`` with a clear
  message instead of letting NumPy emit a cryptic ``LinAlgError``. The
  implementation is **stricter** than shard 02 §A.2 step 4 prose ("Re-scale
  back to original mean + std"): we re-standardize the Cholesky-transformed
  columns before re-scaling so the input block's marginal mean/std are
  preserved EXACTLY, not just in expectation. Per-column z-scoring is a
  linear transform → it does not change Pearson correlations, so the target
  ``r`` is still hit. Action item: future shard refresh should bring the
  prose in line with this implementation; do NOT regress to the pseudocode
  literal version without coordinating a test-tolerance update.
- ``standardize_train_val_test``: train-stats z-score, no leakage (§A.3).
- ``solve_intercept``: monotone bisection on logistic intercept to hit a
  target prevalence within tolerance (§B.1).

These primitives are deliberately stateless and accept an explicit
``np.random.Generator`` so the determinism contract (shard 01 §C.1) holds:
no module-level seeding, no global RNG state.
"""

from __future__ import annotations

from typing import Any

import numpy as np

_VALID_DISTRIBUTIONS = frozenset({"normal", "uniform", "bernoulli", "categorical"})


def sample_one_feature(
    rng: np.random.Generator,
    n: int,
    distribution: str,
    params: dict[str, Any],
) -> np.ndarray:
    """Sample ``n`` i.i.d. values for one feature (shard 02 §A.1).

    ``categorical`` is intentionally rejected: scenario authors expand
    categorical variables into per-level bernoulli features at the manifest
    level so each level keeps its own coefficient + audit citation (matches
    LightGBM's one-feature-per-column importance surface).
    """
    if n <= 0:
        raise ValueError(f"n must be positive; got {n}")
    if distribution == "normal":
        return np.asarray(
            rng.normal(loc=params["loc"], scale=params["scale"], size=n),
            dtype=np.float64,
        )
    if distribution == "uniform":
        return np.asarray(
            rng.uniform(low=params["low"], high=params["high"], size=n),
            dtype=np.float64,
        )
    if distribution == "bernoulli":
        return rng.binomial(n=1, p=params["p"], size=n).astype(np.float64)
    if distribution == "categorical":
        raise ValueError(
            "categorical not directly samplable; expand into one-hot bernoulli "
            "features in the per-scenario manifest"
        )
    raise ValueError(
        f"Unknown distribution {distribution!r}; "
        f"supported: {sorted(_VALID_DISTRIBUTIONS - {'categorical'})}"
    )


def _validate_correlation_block_psd(n_cols: int, r: float) -> None:
    """Reject non-PSD correlation blocks before Cholesky (shard 02 §A.2 edge case).

    A square ``n_cols × n_cols`` matrix with ``r`` off-diagonal and ``1`` on
    diagonal has eigenvalues ``1 + (n_cols-1)·r`` (once) and ``1 - r``
    (multiplicity ``n_cols - 1``). PSD requires both to be ≥ 0, i.e.
    ``-1/(n_cols-1) <= r <= 1`` for ``n_cols >= 2``.
    """
    if n_cols < 2:
        return
    upper = 1.0
    lower = -1.0 / (n_cols - 1)
    if not (lower <= r <= upper):
        raise ValueError(
            f"correlation block of size {n_cols} requires r in [{lower:.4f}, {upper:.4f}]; "
            f"got r={r}. (Eigenvalues 1 + (n_cols-1)·r and 1 - r must both be >= 0.)"
        )


def apply_block_correlation(
    rng: np.random.Generator,
    base_features: np.ndarray,
    blocks: list[tuple[list[int], float]],
) -> np.ndarray:
    """Inject block correlation via Cholesky (shard 02 §A.2).

    ``rng`` is unused in this implementation but kept on the signature so
    future callers can layer rotation / sign-flip noise without breaking
    the consumer contract.
    """
    del rng  # reserved for future stochastic post-processing
    if base_features.ndim != 2:
        raise ValueError(
            f"base_features must be 2-D (n, n_features); got shape {base_features.shape}"
        )
    out = base_features.astype(np.float64, copy=True)
    seen_indices: set[int] = set()
    n_features = out.shape[1]
    for cols, r in blocks:
        if len(cols) < 2:
            continue
        if any(c < 0 or c >= n_features for c in cols):
            raise ValueError(
                f"correlation block references out-of-range column index in {cols}; "
                f"valid range is [0, {n_features - 1}]"
            )
        if len(set(cols)) != len(cols):
            raise ValueError(f"correlation block has duplicate column indices: {cols}")
        overlap = seen_indices.intersection(cols)
        if overlap:
            raise ValueError(
                f"correlation block columns {cols} overlap previously-correlated "
                f"columns {sorted(overlap)}; each column may belong to at most one block"
            )
        seen_indices.update(cols)
        _validate_correlation_block_psd(len(cols), r)
        block = out[:, cols].copy()
        means = block.mean(axis=0)
        stds = block.std(axis=0, ddof=0)
        stds_safe = np.where(stds < 1e-12, 1.0, stds)
        standardized = (block - means) / stds_safe
        n_cols = len(cols)
        target_corr = np.full((n_cols, n_cols), r, dtype=np.float64)
        np.fill_diagonal(target_corr, 1.0)
        try:
            chol = np.linalg.cholesky(target_corr)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                f"correlation block {cols} target matrix is not positive semi-definite "
                f"(r={r}); pre-validate eigenvalues per shard 02 §A.2 edge case."
            ) from exc
        correlated = standardized @ chol.T
        # Re-standardize the correlated columns so the rescale step exactly
        # preserves the input block's marginal mean/std. Per-column z-scoring
        # is a linear transform → it does NOT change Pearson correlations
        # (only their absolute scale), so the target r is still hit.
        correlated_means = correlated.mean(axis=0)
        correlated_stds = correlated.std(axis=0, ddof=0)
        correlated_stds_safe = np.where(correlated_stds < 1e-12, 1.0, correlated_stds)
        correlated_z = (correlated - correlated_means) / correlated_stds_safe
        out[:, cols] = correlated_z * stds_safe + means
    return out


def standardize_train_val_test(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Z-score using **train** statistics (shard 02 §A.3).

    Returns ``(X_train_z, X_val_z, X_test_z, mean, std)``. Val/test get the
    same mean+std the training pipeline computes — prevents data leakage.

    The returned ``std`` is the **raw** train-set std (without zero-variance
    safe substitution) so callers can detect degenerate columns. Internally
    the division uses a safe-substituted vector to avoid divide-by-zero.
    """
    if X_train.ndim != 2 or X_val.ndim != 2 or X_test.ndim != 2:
        raise ValueError(
            "standardize_train_val_test expects 2-D arrays; got "
            f"shapes train={X_train.shape}, val={X_val.shape}, test={X_test.shape}"
        )
    if X_train.shape[1] != X_val.shape[1] or X_train.shape[1] != X_test.shape[1]:
        raise ValueError(
            "train/val/test must share the same n_features; got "
            f"train={X_train.shape[1]}, val={X_val.shape[1]}, test={X_test.shape[1]}"
        )
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0, ddof=0)
    std_safe = np.where(std < 1e-12, 1.0, std)
    return (
        (X_train - mean) / std_safe,
        (X_val - mean) / std_safe,
        (X_test - mean) / std_safe,
        mean,
        std,
    )


def solve_intercept(
    X: np.ndarray,
    coefficients: np.ndarray,
    target_prevalence: float,
    *,
    tol: float = 1e-4,
    max_iter: int = 100,
    bracket: tuple[float, float] = (-20.0, 20.0),
) -> float:
    """Bisect on the intercept until ``E[sigmoid(X·coef + b)] ≈ target`` (shard 02 §B.1).

    The mean sigmoid is monotone increasing in the intercept, so bisection
    converges in ``O(log((bracket_high - bracket_low) / tol))`` iterations.
    Returns the converged intercept ``b``. Raises ``RuntimeError`` if the
    bracket is too narrow to contain the solution.
    """
    if not 0.0 < target_prevalence < 1.0:
        raise ValueError(f"target_prevalence must be in (0, 1); got {target_prevalence}")
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D; got shape {X.shape}")
    if coefficients.ndim != 1 or coefficients.shape[0] != X.shape[1]:
        raise ValueError(
            "coefficients shape must match X.shape[1]; got "
            f"coefficients.shape={coefficients.shape}, X.shape={X.shape}"
        )
    if bracket[0] >= bracket[1]:
        raise ValueError(f"bracket must be (low, high) with low < high; got {bracket}")

    z = X @ coefficients
    lo, hi = float(bracket[0]), float(bracket[1])
    realized = float("nan")
    mid = 0.5 * (lo + hi)

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        # Sigmoid via numerically-stable form: avoid overflow on extreme z+mid
        p = _sigmoid(z + mid)
        realized = float(p.mean())
        if abs(realized - target_prevalence) < tol:
            return mid
        if realized < target_prevalence:
            lo = mid
        else:
            hi = mid

    raise RuntimeError(
        f"intercept solver failed to converge after {max_iter} iters; "
        f"final realized={realized:.6f}, target={target_prevalence:.6f}, "
        f"bracket=({lo:.4f}, {hi:.4f}), last mid={mid:.4f}. "
        "Widen bracket or check coefficients for saturation."
    )


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically-stable element-wise sigmoid."""
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[neg])
    out[neg] = exp_z / (1.0 + exp_z)
    return out
