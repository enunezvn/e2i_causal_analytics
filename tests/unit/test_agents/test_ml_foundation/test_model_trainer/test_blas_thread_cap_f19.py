"""F19: model_trainer BLAS/OpenMP thread cap bounds peak RSS without changing results.

The cap exists to keep a single agent run memory-bounded (CV/permutation/bootstrap
spiked ~5.9 GB). Thread count does not affect the deterministic fit output of these
estimators — only RSS/wall-clock — so the cap must be result-preserving.
"""

import numpy as np
import threadpoolctl

from src.agents.ml_foundation.model_trainer.agent import _blas_thread_limit


def test_cap_is_result_preserving():
    """A fit under the thread cap must produce bit-for-bit identical coefficients."""
    from sklearn.linear_model import LogisticRegression

    rng = np.random.RandomState(0)
    X = rng.rand(80, 5)
    y = (X[:, 0] + X[:, 1] > 1.0).astype(int)

    base = LogisticRegression(max_iter=300, random_state=0).fit(X, y).coef_.copy()
    with _blas_thread_limit({"blas_thread_cap": 1}):
        capped = LogisticRegression(max_iter=300, random_state=0).fit(X, y).coef_.copy()

    np.testing.assert_array_equal(base, capped)


def test_cap_limits_blas_threads_when_present():
    """Inside the cap context, any loaded BLAS pool must report num_threads == 1."""
    # Ensure a BLAS pool is loaded so the assertion is not vacuous.
    np.dot(np.random.rand(64, 64), np.random.rand(64, 64))
    with _blas_thread_limit({"blas_thread_cap": 1}):
        blas = [p for p in threadpoolctl.threadpool_info() if p.get("user_api") == "blas"]
        assert all(p["num_threads"] == 1 for p in blas), f"BLAS not capped: {blas}"


def test_cap_resolution_is_robust():
    """Default (1), explicit override, and malformed values all yield a usable context."""
    for cfg in ({}, {"blas_thread_cap": 2}, {"blas_thread_cap": "bad"}, None):
        ctx = _blas_thread_limit(cfg)
        with ctx:  # must be a working context manager regardless of input
            pass
