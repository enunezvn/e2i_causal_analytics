"""Regression: hierarchical heavy fits must run OFF the event loop.

Why this exists
---------------
The ``/segment-analysis`` page ran its analysis as a FastAPI ``BackgroundTask``
inside the uvicorn worker's own event loop. Every heterogeneous-optimizer graph
node offloads its heavy fit with ``asyncio.to_thread`` EXCEPT the hierarchical
path, which called the synchronous CausalML uplift fit
(``HierarchicalAnalyzer._compute_uplift_scores`` -> ``UpliftRandomForest.estimate``)
and the per-segment EconML fits (``SegmentCATECalculator._run_econml_estimator``
-> ``_run_causal_forest`` / ``_run_ols`` / ...) INLINE in the coroutine.

A synchronous fit running on the event-loop thread starves uvicorn's heartbeat
task (the one that touches the gunicorn worker's tmpfile), so the gunicorn
master SIGABRTs the worker at ``--timeout 120`` — the chronic
``WORKER TIMEOUT (pid:N) -> code 134`` seen on the loaded box. It also silently
defeats the ``asyncio.wait_for(..., timeout=...)`` guards wrapping ``analyze()``
in BOTH callers (the segments node and ``causal.py`` /hierarchical), because a
coroutine that blocks the loop never lets the timeout timer fire.

These tests assert the fix at the SSOT (the shared ``causal_engine.hierarchical``
library): the heavy fit runs on a DIFFERENT thread than the running event loop
(mechanism), and a concurrent heartbeat keeps ticking while the fit runs
(behavior). Both fail if the fit is executed inline on the loop thread.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
from types import SimpleNamespace
from typing import Any, Coroutine, Tuple

import numpy as np
import pandas as pd
import pytest

# Synthetic "heavy fit" duration. Long relative to the heartbeat interval so an
# inline (loop-blocking) fit would clearly freeze the heartbeat, while an
# off-loop fit lets it keep ticking. time.sleep() releases the GIL, so it
# faithfully models a fit offloaded to a worker thread.
_BLOCK_SECONDS = 0.5
_HEARTBEAT_INTERVAL = 0.02


async def _run_with_heartbeat(
    coro: Coroutine[Any, Any, Any],
) -> Tuple[Any, int]:
    """Await ``coro`` while a concurrent heartbeat counts loop iterations.

    Returns ``(result, tick_count)``. If ``coro`` blocks the event loop for
    ``_BLOCK_SECONDS`` the heartbeat cannot advance and ``tick_count`` stays ~0;
    if the heavy work runs off the loop the heartbeat keeps ticking.
    """
    ticks = 0
    stop = False

    async def _heartbeat() -> None:
        nonlocal ticks
        while not stop:
            await asyncio.sleep(_HEARTBEAT_INTERVAL)
            ticks += 1

    hb_task = asyncio.create_task(_heartbeat())
    try:
        result = await coro
    finally:
        stop = True
        hb_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await hb_task
    return result, ticks


@pytest.mark.asyncio
async def test_segment_cate_fit_runs_off_the_event_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    """The per-segment EconML fit must be offloaded, not run on the loop thread."""
    from src.causal_engine.hierarchical import SegmentCATECalculator, SegmentCATEConfig

    loop_thread = threading.current_thread()
    fit_thread: dict[str, threading.Thread] = {}

    config = SegmentCATEConfig(estimator_type="ols", min_samples=30)
    calculator = SegmentCATECalculator(config)

    # Deterministic, well-balanced segment data (enough treated/control that the
    # calculator's own guards pass and the real OLS fit succeeds).
    rng = np.random.default_rng(0)
    n = 120
    X = pd.DataFrame({"x1": rng.standard_normal(n), "x2": rng.standard_normal(n)})
    treatment = np.array([0, 1] * (n // 2), dtype=int)
    outcome = X["x1"].to_numpy() * 0.3 + treatment * 0.2 + rng.standard_normal(n) * 0.1

    real_run_ols = calculator._run_ols

    def slow_run_ols(*args: Any, **kwargs: Any) -> Any:
        fit_thread["t"] = threading.current_thread()
        time.sleep(_BLOCK_SECONDS)  # stand-in for the heavy synchronous fit
        return real_run_ols(*args, **kwargs)

    monkeypatch.setattr(calculator, "_run_ols", slow_run_ols)

    result, ticks = await _run_with_heartbeat(
        calculator.compute(
            X=X,
            treatment=treatment,
            outcome=outcome,
            segment_id=0,
            segment_name="seg",
        )
    )

    assert result.success is True, result.error_message
    # Mechanism: the fit ran on a worker thread, not the event-loop thread.
    assert fit_thread["t"] is not loop_thread, (
        "segment CATE fit ran on the event-loop thread (inline) — it must be "
        "offloaded via asyncio.to_thread"
    )
    # Behavior: the event loop stayed responsive during the ~0.5s fit.
    assert ticks >= 5, f"event loop was blocked during the segment fit (ticks={ticks})"


@pytest.mark.asyncio
async def test_uplift_scoring_runs_off_the_event_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    """The CausalML uplift fit must be offloaded, not run on the loop thread."""
    import src.causal_engine.uplift as uplift_mod
    from src.causal_engine.hierarchical import HierarchicalAnalyzer, HierarchicalConfig

    loop_thread = threading.current_thread()
    fit_thread: dict[str, threading.Thread] = {}

    n = 200
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"x1": rng.standard_normal(n), "x2": rng.standard_normal(n)})
    treatment = np.array([0, 1] * (n // 2), dtype=int)
    outcome = rng.standard_normal(n)

    def slow_estimate(self: Any, X_: Any, treatment_: Any, outcome_: Any) -> Any:
        fit_thread["t"] = threading.current_thread()
        time.sleep(_BLOCK_SECONDS)  # stand-in for the heavy CausalML fit
        return SimpleNamespace(
            success=True,
            uplift_scores=np.zeros(len(X_)),
            model_type=SimpleNamespace(value="uplift_random_forest"),
            error_message=None,
        )

    monkeypatch.setattr(uplift_mod.UpliftRandomForest, "estimate", slow_estimate)

    analyzer = HierarchicalAnalyzer(HierarchicalConfig(n_segments=3))

    (scores, model_name), ticks = await _run_with_heartbeat(
        analyzer._compute_uplift_scores(X, treatment, outcome)
    )

    assert len(scores) == n
    assert model_name == "uplift_random_forest"
    # Mechanism: the fit ran on a worker thread, not the event-loop thread.
    assert fit_thread["t"] is not loop_thread, (
        "uplift fit ran on the event-loop thread (inline) — it must be offloaded "
        "via asyncio.to_thread"
    )
    # Behavior: the event loop stayed responsive during the ~0.5s fit.
    assert ticks >= 5, f"event loop was blocked during the uplift fit (ticks={ticks})"
