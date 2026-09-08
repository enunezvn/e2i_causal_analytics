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
  behavioral proof (a heartbeat coroutine must keep ticking during SHAP). It
  patches ``shap.TreeExplainer.shap_values`` — the EXACT frame named in the
  faulthandler dumps, one level DEEPER than the seam the first test pins, so
  off-loading only PART of ``_compute_shap_from_frame`` is still caught — to
  ``time.sleep`` a fixed duration and THEN delegate to the real
  implementation: the real path still executes end-to-end, and the block has a
  deterministic duration regardless of box speed.

#1963 — why the behavioral proof counts TICKS instead of bounding a gap
-----------------------------------------------------------------------
It originally asserted ``max heartbeat gap < 0.35s``. That measures the wrong
thing. ``shap``'s ``_cext.dense_tree_shap`` holds the GIL for its ENTIRE
monolithic C call (see the ``_SHAP_*`` bounds in ``causal_discovery.py``, which
exist precisely because off-loading alone cannot protect the loop from it), so
the loop is starved for the duration of the REAL ``shap_values`` call even when
the off-load is perfectly correct. Measured on the dev box (n=80 frame, 25
runs) the largest gap on the CORRECT path tracked that C call almost exactly —
0.032–0.144s, unchanged under 2x CPU oversubscription — while CI (``pytest -n
2``, contended runner) produced 0.801s on that same correct code and failed the
gate. The inline regression starves the loop for 0.913–1.323s. The whole
separation between "correct" and "regressed" is therefore the ~0.1s between
CI's noise ceiling and the regression's floor: no wall-clock threshold here is
both flake-free and load-bearing. Widening the bound to 2–3s would be strictly
WORSE than the flake — the regression tops out near 1.0s, so the guard would
never fire again.

What IS deterministic is that ``time.sleep`` RELEASES the GIL. During the block
window the loop can run iff the block is not on the loop thread, so the test
counts heartbeat ticks INSIDE that window: measured 34–37 off-loaded (idle and
loaded alike) versus 0 inline, on every run — categorical, with no threshold to
tune. The sibling guard
``tests/unit/test_causal_engine/test_hierarchical/test_event_loop_offload.py``
uses the same tick-count idiom; this one scopes the count to the block window
rather than the whole coroutine so that ``await`` points added to
``rank_drivers`` later cannot silently inflate it past the floor.

Falsifiability: reverting the ``run_in_bounded_executor`` off-load in
``rank_drivers`` (running ``_compute_shap_from_frame`` inline again) makes both
tests fail — the first because the compute runs on the loop thread, the second
because ``shap_values`` runs on the loop thread AND the heartbeat records zero
ticks during the block (positive-controlled under #1963).
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

# Deterministic block prepended to the (patched) shap_values. ``time.sleep``
# RELEASES the GIL, so this window models an off-loaded compute faithfully: the
# loop is free to tick throughout it iff the block is not on the loop thread.
_BLOCK_SECONDS = 0.75
_HEARTBEAT_INTERVAL = 0.02
# Floor on heartbeat ticks observed INSIDE that GIL-free window. ~37 are
# expected (0.75s / 0.02s); measured 34–37 off-loaded and 0 inline, both idle
# and under 2x CPU oversubscription. 5 leaves a ~7x margin under the healthy
# floor and is unreachable for the regression, which cannot tick at all. See
# the #1963 section of the module docstring for why this replaced a wall-clock
# gap threshold.
_MIN_TICKS_DURING_BLOCK = 5


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
    """A heartbeat coroutine must keep ticking while shap_values blocks.

    Regression path: any change that puts the SHAP block back on the loop
    (inline call, off-loading only part of ``_compute_shap_from_frame``, or
    off-loading but then synchronously waiting on the result) freezes the
    heartbeat for the whole block.
    """
    import shap

    loop_thread = threading.current_thread()
    real_shap_values = shap.TreeExplainer.shap_values
    block: Dict[str, Any] = {}

    def blocking_shap_values(self: Any, *args: Any, **kwargs: Any) -> Any:
        # Deterministic, GIL-RELEASING block (see the module docstring for why
        # a fixed sleep is prepended instead of relying on real compute time),
        # then the REAL computation so the full path still executes.
        block["thread"] = threading.current_thread()
        block["start"] = time.monotonic()
        time.sleep(_BLOCK_SECONDS)
        block["end"] = time.monotonic()
        return real_shap_values(self, *args, **kwargs)

    monkeypatch.setattr(shap.TreeExplainer, "shap_values", blocking_shap_values)

    ticks: List[float] = []
    stop = asyncio.Event()

    async def heartbeat() -> None:
        while not stop.is_set():
            await asyncio.sleep(_HEARTBEAT_INTERVAL)
            ticks.append(time.monotonic())

    beat_task = asyncio.create_task(heartbeat())
    # Let the heartbeat actually start running before the compute does.
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
    assert ticks, "heartbeat never ticked — test harness defect"
    # Positive control on the guard itself: if the patched frame never ran, the
    # tick count below would pass vacuously.
    assert "start" in block, (
        "shap.TreeExplainer.shap_values was never invoked — rank_drivers no "
        "longer routes through TreeExplainer, so this guard is not measuring "
        "the #1548 frame at all."
    )

    # Mechanism: the EXACT frame from the faulthandler dumps ran off the loop,
    # on the bounded heavy-compute pool. Pinned one level deeper than
    # test_compute_shap_runs_off_event_loop_thread, which patches the enclosing
    # _compute_shap_from_frame — a partial off-load passes there but not here.
    assert block["thread"] is not loop_thread, (
        "#1548 regression: shap_values ran on the event-loop thread — its "
        "monolithic GIL-holding C call will starve the loop and gunicorn will "
        "murder the worker at last-notify+120s."
    )
    assert block["thread"].name.startswith("heavy-compute"), (
        f"shap_values ran on thread {block['thread'].name!r}; expected the "
        "bounded 'heavy-compute' pool from src.api.dependencies.compute."
    )

    # Behavior: the loop kept making progress WHILE the block was in flight.
    # Counted only inside the block window, where time.sleep has released the
    # GIL and a healthy loop is therefore unconditionally free to run.
    ticks_during_block = sum(1 for t in ticks if block["start"] < t < block["end"])
    assert ticks_during_block >= _MIN_TICKS_DURING_BLOCK, (
        f"#1548 regression: the event loop advanced only {ticks_during_block} "
        f"time(s) during the {_BLOCK_SECONDS}s SHAP block (expected >= "
        f"{_MIN_TICKS_DURING_BLOCK}, healthy is ~"
        f"{int(_BLOCK_SECONDS / _HEARTBEAT_INTERVAL)}) — the block is running "
        "on the loop thread, and gunicorn's arbiter would murder the worker on "
        "a real >120s frame."
    )
