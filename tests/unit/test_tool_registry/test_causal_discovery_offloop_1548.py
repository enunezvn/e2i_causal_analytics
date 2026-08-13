"""#1548 — ``rank_drivers`` SHAP derivation must NOT run on the event loop.

Measured root cause (2026-08-13 live faulthandler dumps): ``rank_drivers``
called ``_compute_shap_from_frame`` (RandomForest fit + TreeExplainer
``shap_values``) SYNCHRONOUSLY inside its coroutine, starving the uvicorn
worker's main event loop for >120s on heavy /chat/stream turns. The loop never
ran uvicorn's ``callback_notify``, so gunicorn's arbiter murdered the worker at
last-notify+120s → mid-stream ``RemoteProtocolError`` tear. The
``asyncio.wait_for`` at ``executor.py:673`` cannot preempt a sync call that
never yields.

The fix routes ``_compute_shap_from_frame`` through the EXISTING bounded
heavy-compute pool (``src.api.dependencies.compute.run_in_bounded_executor``,
prod cap prior art) so the loop keeps ticking while SHAP computes.

Test-design reasoning (per the #1548 brief, stated explicitly):

* ``test_compute_shap_runs_off_event_loop_thread`` uses the FULLY REAL compute
  path (real small frame, real RandomForest, real TreeExplainer) and asserts
  the mechanism: the compute executes on a bounded heavy-compute pool thread,
  not the loop thread. Deterministic — no timing thresholds.
* ``test_rank_drivers_keeps_event_loop_responsive_during_shap`` is the
  behavioral proof (a heartbeat coroutine must keep ticking during SHAP). A
  purely real frame's compute time is box-load-dependent (measured 0.5–3.4s
  for n=120 warm/cold on the dev box, 24s for n=2000 under load), which would
  make any gap threshold flaky in BOTH directions. So the test patches
  ``shap.TreeExplainer.shap_values`` — the EXACT frame named in the
  faulthandler dumps — to ``time.sleep`` a fixed duration and THEN delegate to
  the real implementation: the real path still executes end-to-end, and the
  block duration has a deterministic lower bound regardless of box speed.

Falsifiability: reverting the ``run_in_bounded_executor`` off-load in
``rank_drivers`` (running ``_compute_shap_from_frame`` inline again) makes both
tests fail — the first because the compute runs on the loop thread, the second
because the heartbeat gap reaches the full block duration.
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import pytest

import src.tool_registry.tools.causal_discovery as causal_discovery
from src.api.dependencies.compute import _reset_limiter_cache_for_tests

# Deterministic lower bound on how long the (patched) shap_values blocks.
_BLOCK_SECONDS = 0.75
# Max tolerated heartbeat gap while SHAP computes. Far above a healthy loop's
# tick (~20ms + scheduling noise), far below _BLOCK_SECONDS so old-code failure
# is unambiguous even on a loaded box.
_MAX_GAP_SECONDS = 0.35
_HEARTBEAT_INTERVAL = 0.02


@pytest.fixture(autouse=True)
def _fresh_heavy_compute_pool():
    """Isolate the process-global bounded executor between tests."""
    _reset_limiter_cache_for_tests()
    yield
    _reset_limiter_cache_for_tests()


def _linear_frame(n: int, seed: int = 7) -> pd.DataFrame:
    """A small REAL frame with planted dependencies (not a mock: the real
    RandomForest + TreeExplainer run over it)."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    b = 2.0 * a + rng.normal(size=n)
    c = 1.5 * b + rng.normal(size=n)
    d = rng.normal(size=n)
    return pd.DataFrame({"a": a, "b": b, "c": c, "d": d})


async def test_compute_shap_runs_off_event_loop_thread(monkeypatch) -> None:
    """The SHAP derivation must execute on a bounded heavy-compute pool thread.

    Regression path: running ``_compute_shap_from_frame`` inline in the
    ``rank_drivers`` coroutine puts the RandomForest fit + TreeExplainer
    ``shap_values`` back on the event loop thread (the exact #1548 frame).
    """
    captured: Dict[str, Any] = {}
    real_compute = causal_discovery._compute_shap_from_frame

    def recording_compute(*args: Any, **kwargs: Any) -> Tuple[List[List[float]], List[str]]:
        captured["thread"] = threading.current_thread()
        return real_compute(*args, **kwargs)

    monkeypatch.setattr(causal_discovery, "_compute_shap_from_frame", recording_compute)

    loop_thread = threading.current_thread()
    result = await causal_discovery.rank_drivers(
        dag_edge_list=[],
        target="c",
        estimation_data=_linear_frame(n=120),
    )

    # The real path completed with a real predictive-only ranking.
    assert result["success"] is True
    assert result["n_features"] == 3
    assert {r["feature_name"] for r in result["rankings"]} == {"a", "b", "d"}

    compute_thread = captured["thread"]
    assert compute_thread is not loop_thread, (
        "#1548 regression: _compute_shap_from_frame ran on the event-loop "
        "thread — TreeExplainer.shap_values will starve the loop and gunicorn "
        "will murder the worker at last-notify+120s."
    )
    # Pin the SEAM, not just 'any thread': the shared bounded heavy-compute
    # pool (prod cap prior art, src/api/dependencies/compute.py) — NOT the
    # loop's default (unbounded) executor, which would let N concurrent turns
    # fit N RandomForests inside the 5G cgroup.
    assert compute_thread.name.startswith("heavy-compute"), (
        f"SHAP compute ran on thread {compute_thread.name!r}; expected the "
        "bounded 'heavy-compute' pool from src.api.dependencies.compute."
    )


async def test_rank_drivers_keeps_event_loop_responsive_during_shap(monkeypatch) -> None:
    """A heartbeat coroutine must keep ticking while shap_values computes.

    Regression path: any change that puts the SHAP block back on the loop
    (inline call, or off-loading only part of ``_compute_shap_from_frame``)
    freezes the heartbeat for the full block duration.
    """
    import shap

    real_shap_values = shap.TreeExplainer.shap_values

    def blocking_shap_values(self: Any, *args: Any, **kwargs: Any) -> Any:
        # Deterministic block (see module docstring for why a fixed sleep is
        # prepended instead of relying on real compute time), then the REAL
        # computation so the full path still executes.
        time.sleep(_BLOCK_SECONDS)
        return real_shap_values(self, *args, **kwargs)

    monkeypatch.setattr(shap.TreeExplainer, "shap_values", blocking_shap_values)

    gaps: List[float] = []
    stop = asyncio.Event()

    async def heartbeat() -> None:
        last = time.monotonic()
        while not stop.is_set():
            await asyncio.sleep(_HEARTBEAT_INTERVAL)
            now = time.monotonic()
            gaps.append(now - last)
            last = now

    beat_task = asyncio.create_task(heartbeat())
    # Let the heartbeat establish its baseline before the compute starts.
    await asyncio.sleep(_HEARTBEAT_INTERVAL * 2)
    try:
        result = await causal_discovery.rank_drivers(
            dag_edge_list=[],
            target="c",
            estimation_data=_linear_frame(n=80),
        )
    finally:
        stop.set()
        await beat_task

    assert result["success"] is True
    assert gaps, "heartbeat never ticked — test harness defect"
    max_gap = max(gaps)
    assert max_gap < _MAX_GAP_SECONDS, (
        f"#1548 regression: event loop starved for {max_gap:.3f}s (>= "
        f"{_MAX_GAP_SECONDS}s) while shap_values computed — gunicorn's "
        "arbiter would murder the worker on a real >120s frame."
    )
