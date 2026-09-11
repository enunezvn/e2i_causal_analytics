"""The reliability rule turns counts into one verdict word (spec §7.1).

The rule is the single place that decides whether a tool's failures are evidence. Its denominator
is ``n_health = n_succeeded + n_health_failures``: refusals are the tool declining to answer, not
the tool failing, so twenty refusals can never qualify a tool for a verdict.

The calibration test is the point of the whole rule: on planted truth at production-shaped n, a
healthy tool must almost never earn a caveat and a failing one must usually earn it. It runs the
shipped rule itself, not a re-derivation of it.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pytest

from src.agents.tool_composer.reliability import (
    ToolReliability,
    ToolReliabilityReader,
    verdict,
    wilson,
)


def _counts(
    n_succeeded: int = 0,
    n_health_failures: int = 0,
    n_refused: int = 0,
    n_invoked: Optional[int] = None,
) -> Dict[str, int]:
    total = n_invoked if n_invoked is not None else n_succeeded + n_health_failures + n_refused
    return {
        "n_invoked": total,
        "n_succeeded": n_succeeded,
        "n_health_failures": n_health_failures,
        "n_refused": n_refused,
        "n_health": n_succeeded + n_health_failures,
    }


# ---------------------------------------------------------------------------
# Wilson
# ---------------------------------------------------------------------------


def test_wilson_matches_the_known_interval_and_its_edges():
    assert wilson(0, 0) == (0.0, 1.0)  # no evidence spans the whole range
    lo, hi = wilson(0, 40)
    assert lo == 0.0 and 0.0 < hi < 0.12
    lo, hi = wilson(40, 40)
    assert hi == 1.0 and lo > 0.88
    lo, hi = wilson(12, 40)
    assert 0.17 < lo < 0.19 and 0.44 < hi < 0.46  # 0.30 with a 95% Wilson interval


# ---------------------------------------------------------------------------
# The verdict table (spec §7.1)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "counts, expected",
    [
        (_counts(), "no_runs"),
        (_counts(n_succeeded=15, n_health_failures=4), "too_few_runs"),  # n_health 19
        (_counts(n_refused=20), "too_few_runs"),  # refusals are not health evidence
        (_counts(n_succeeded=40, n_health_failures=0), "reliable"),
        (_counts(n_succeeded=28, n_health_failures=12), "caveat"),
        (_counts(n_succeeded=37, n_health_failures=3), "inconclusive"),
    ],
)
def test_verdict_table(counts, expected):
    assert verdict(counts) == expected


def test_refusals_never_enter_the_denominator():
    """A tool that refused a hundred times and never failed is still "too few runs"."""
    assert verdict(_counts(n_succeeded=10, n_health_failures=5, n_refused=100)) == "too_few_runs"


def test_production_volume_reads_too_few_runs():
    """At today's volume (≤ 7 invocations per tool) no tool can earn a verdict."""
    for n in range(0, 8):
        counts = _counts(n_succeeded=n)
        assert verdict(counts) == ("no_runs" if n == 0 else "too_few_runs")


# ---------------------------------------------------------------------------
# Calibration on planted truth
# ---------------------------------------------------------------------------


def _caveat_rate(n: int, p: float, draws: int = 20_000, seed: int = 7) -> float:
    """Fraction of simulated tools the SHIPPED rule caveats, with a planted failure rate p."""
    rng = np.random.default_rng(seed)
    failures = rng.binomial(n, p, size=draws)
    caveats = sum(
        1
        for k in failures.tolist()
        if verdict(_counts(n_succeeded=n - k, n_health_failures=k)) == "caveat"
    )
    return caveats / draws


def test_planted_truth_calibration():
    started = time.monotonic()

    # A healthy tool is almost never caveated, at every production-shaped n.
    for n in (20, 40, 100):
        assert _caveat_rate(n, 0.05) <= 0.01, f"false caveats at n={n}"

    # A badly failing tool is caught.
    assert _caveat_rate(40, 0.30) >= 0.90

    # Below the floor there is no verdict to earn, whatever the draw.
    for n in (5, 10, 19):
        rng = np.random.default_rng(7)
        for k in rng.binomial(n, 0.30, size=200).tolist():
            assert verdict(_counts(n_succeeded=n - k, n_health_failures=k)) == "too_few_runs"

    assert time.monotonic() - started < 5.0


# ---------------------------------------------------------------------------
# Measured latency is shown only when it is measured
# ---------------------------------------------------------------------------


def _row(**over: Any) -> Dict[str, Any]:
    row = {
        "tool_name": "causal_effect_estimator",
        "category": "CAUSAL",
        "source_agent": "causal_impact",
        "version": "1.0.0",
        "declared_latency_ms": 5000.0,
        "n_invoked": 40,
        "n_succeeded": 40,
        "n_refused": 0,
        "n_health_failures": 0,
        "n_health": 40,
        "n_retried": 0,
        "n_synthetic": 0,
        "p50_latency_ms": 1200.0,
        "p95_latency_ms": 3000.0,
        "last_executed_at": "2026-09-11T00:00:00+00:00",
        "most_common_health_error": None,
    }
    row.update(over)
    return row


def test_latency_is_null_below_20_successes_and_declared_is_never_substituted():
    few = ToolReliability.from_row(_row(n_invoked=19, n_succeeded=19, n_health=19))
    assert few.p50_latency_ms is None and few.p95_latency_ms is None
    assert few.declared_latency_ms == 5000.0  # separate, labelled, never a measured value

    enough = ToolReliability.from_row(_row())
    assert enough.p50_latency_ms == 1200.0 and enough.p95_latency_ms == 3000.0


def test_the_row_carries_its_verdict_and_counts():
    tool = ToolReliability.from_row(_row(n_succeeded=28, n_health_failures=12, n_health=40))
    assert tool.verdict == "caveat"
    assert (tool.n_succeeded, tool.n_health_failures, tool.n_health) == (28, 12, 40)
    assert tool.tool_name == "causal_effect_estimator"


# ---------------------------------------------------------------------------
# The reader
# ---------------------------------------------------------------------------


class RecordingPort:
    def __init__(self, rows: Optional[List[Dict[str, Any]]] = None) -> None:
        self.rows = rows if rows is not None else [_row()]
        self.calls: List[Dict[str, Any]] = []

    async def call(self, name: str, params: Dict[str, Any]) -> Any:
        self.calls.append({"name": name, **params})
        return self.rows


class FailingPort:
    async def call(self, name: str, params: Dict[str, Any]) -> Any:
        raise ConnectionError("reliability read is down")


async def test_reader_calls_the_rpc_and_keys_its_cache_by_window_and_provenance(monkeypatch):
    monkeypatch.delenv("E2I_INCLUDE_SYNTHETIC", raising=False)
    port = RecordingPort()
    reader = ToolReliabilityReader(port=port)

    first = await reader.get(30)
    await reader.get(30)
    assert len(port.calls) == 1, "the second read of the same window must come from the cache"
    assert port.calls[0]["name"] == "get_tool_reliability"
    assert port.calls[0]["p_days"] == 30 and port.calls[0]["p_include_synthetic"] is False
    assert first["causal_effect_estimator"].verdict == "reliable"

    await reader.get(60)
    assert len(port.calls) == 2, "a different window is a different key"

    monkeypatch.setenv("E2I_INCLUDE_SYNTHETIC", "1")
    await reader.get(30)
    assert len(port.calls) == 3, "provenance is part of the key"
    assert port.calls[2]["p_include_synthetic"] is True


async def test_reader_expires_its_cache():
    port = RecordingPort()
    reader = ToolReliabilityReader(port=port, ttl_s=0.05)
    await reader.get(30)
    await asyncio.sleep(0.1)
    await reader.get(30)
    assert len(port.calls) == 2


async def test_reader_fails_open():
    reader = ToolReliabilityReader(port=FailingPort())
    assert await reader.get(30) == {}


async def test_reader_tolerates_a_malformed_payload():
    reader = ToolReliabilityReader(port=RecordingPort(rows=[{"nonsense": True}, _row()]))
    verdicts = await reader.get(30)
    assert list(verdicts) == ["causal_effect_estimator"]


def test_the_default_reader_is_one_process_wide_instance():
    """Every consumer reads the same cache, so no two surfaces can disagree within a TTL."""
    from src.agents.tool_composer.reliability import default_reliability_reader

    assert default_reliability_reader() is default_reliability_reader()


class HangingPort:
    def __init__(self) -> None:
        self.calls = 0

    async def call(self, name: str, params: Dict[str, Any]) -> Any:
        self.calls += 1
        await asyncio.sleep(30)
        return []


async def test_reader_bounds_the_read_so_planning_cannot_wait_on_it():
    """The production client allows a 30 s network timeout; planning must not inherit it."""
    port = HangingPort()
    reader = ToolReliabilityReader(port=port, read_timeout_s=0.05)

    started = time.monotonic()
    assert await reader.get(30) == {}
    assert time.monotonic() - started < 5.0

    # A timed-out read is not a reading: the next call tries again rather than serving {}.
    assert await reader.get(30) == {}
    assert port.calls == 2


class FlakyPort:
    """One unusable payload, then a good one."""

    def __init__(self, first: Any) -> None:
        self.payloads: List[Any] = [first, [_row()]]
        self.calls = 0

    async def call(self, name: str, params: Dict[str, Any]) -> Any:
        payload = self.payloads[min(self.calls, len(self.payloads) - 1)]
        self.calls += 1
        return payload


@pytest.mark.parametrize("bad", [{"not": "a list"}, None, "rows"])
async def test_a_payload_that_is_not_rows_is_not_cached(bad):
    port = FlakyPort(bad)
    reader = ToolReliabilityReader(port=port)

    assert await reader.get(30) == {}
    assert list(await reader.get(30)) == ["causal_effect_estimator"]
    assert port.calls == 2


async def test_a_reading_that_dropped_a_row_is_not_cached():
    """A partial decode still serves this call, but must not suppress caveats for 300 s."""
    port = RecordingPort(rows=[{"nonsense": True}, _row()])
    reader = ToolReliabilityReader(port=port)

    assert list(await reader.get(30)) == ["causal_effect_estimator"]
    await reader.get(30)

    assert len(port.calls) == 2
