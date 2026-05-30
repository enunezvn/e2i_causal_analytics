"""Tests for src.mlops.shap_explainer_realtime.reset_executor.

Under gunicorn --preload, the module-global ThreadPoolExecutor created at import
time becomes dead in forked workers (threads do not survive fork). reset_executor
recreates a fresh, usable executor in the child and shuts down the inherited one.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import src.mlops.shap_explainer_realtime as shap_rt


def test_reset_executor_returns_live_usable_pool() -> None:
    pool = shap_rt.reset_executor()
    assert isinstance(pool, ThreadPoolExecutor)
    # module global is rebound to the fresh pool
    assert shap_rt._executor is pool
    # the pool actually runs work
    future = pool.submit(lambda: 21 * 2)
    assert future.result(timeout=5) == 42


def test_reset_executor_shuts_down_old_pool() -> None:
    old = shap_rt._executor
    new = shap_rt.reset_executor()
    assert new is not old
    # the old pool is shut down: submitting to it must raise
    try:
        old.submit(lambda: None)
        raised = False
    except RuntimeError:
        raised = True
    assert raised, "old executor should be shut down (RuntimeError on submit)"


def test_reset_executor_idempotent_no_leak() -> None:
    """Calling twice replaces the pool each time; each result still works."""
    first = shap_rt.reset_executor()
    second = shap_rt.reset_executor()
    assert first is not second
    assert shap_rt._executor is second
    assert second.submit(lambda: "ok").result(timeout=5) == "ok"
    # first should now be shut down
    try:
        first.submit(lambda: None)
        raised = False
    except RuntimeError:
        raised = True
    assert raised
