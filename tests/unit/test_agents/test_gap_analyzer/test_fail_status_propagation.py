"""F2 (HIGH) regression: gap_analyzer must NOT launder a terminal failure into 'completed'.

Audit finding F2: when a real failure occurs in ``gap_detector`` (e.g. a segment/column
mismatch), the node correctly returns ``status='failed'`` and accumulates an entry in
``state['errors']`` -- but the downstream nodes (roi_calculator empty-branch,
prioritizer empty-branch, formatter) used to overwrite the status back to 'completed',
so the graph fail-OPEN'd and returned "No significant performance gaps." with HTTP 200.

The fix principle (REASON-BEFORE-RULES, fail-closed):
  If ANY node accumulated an error in ``state['errors']``, the terminal status must be
  'failed', never 'completed'. A genuinely empty result with NO errors stays 'completed'
  (do NOT turn "no gaps found" into a failure).

These tests drive the REAL gap_detector code path (a real segment/column mismatch raises
a real ``KeyError`` inside ``_detect_segment_gaps``) -- no monkeypatching of status, no
mock that bypasses the real pipeline. They run on tiny in-memory DataFrames (no parquet,
no DB) for OOM safety.
"""

import numpy as np
import pandas as pd
import pytest

from src.agents.gap_analyzer.graph import create_gap_analyzer_graph
from src.agents.gap_analyzer.state import GapAnalyzerState
from src.utils.frame_registry import _clear_all_for_tests, stash_frame


@pytest.fixture(autouse=True)
def _release_stashed_frames():
    """Registry hygiene: _initial_state stashes the tier0 payload (#1743); drop
    every entry after each test so a raising graph run cannot leak it."""
    yield
    _clear_all_for_tests()


def _tier0_malformed_passthrough(n: int = 60):
    """Build a tier0 passthrough payload that triggers a REAL gap_detector failure.

    >= 50 entries so gap_detector takes the tier0 passthrough path, but it is a list
    (not a DataFrame), so ``_derive_performance_from_tier0`` calls ``.select_dtypes``
    on a list -> a REAL ``AttributeError`` inside gap_detector. This drives the genuine
    failure path (gap_detector accumulates an error) without monkeypatching.

    NOTE (#851): a frame that merely LACKS the requested segment column is now handled
    GRACEFULLY (no gaps, status 'completed') — a missing dimension is an unsupported
    segment, not a failure. So this test uses a different, unambiguously-real error to
    exercise the F2 fail-closed principle (errors -> 'failed', never laundered).
    """
    return [{"discontinuation_flag": 0, "tenure_months": 12}] * n


def _initial_state(segments, tier0_data=None) -> GapAnalyzerState:
    state: GapAnalyzerState = {
        "query": "identify trx gaps",
        "metrics": ["trx"],
        "segments": segments,
        "brand": "kisqali",
        "time_period": "current_quarter",
        "filters": None,
        # #1743: these states go into graph.ainvoke, so the tier0 payload must
        # travel as a registry handle — the schema no longer declares tier0_data
        # and LangGraph would silently drop an in-dict frame at the boundary
        # (turning the malformed-passthrough failure test vacuous). The autouse
        # fixture above releases the stash.
        "tier0_frame_ref": (
            stash_frame(tier0_data, label="test-f2") if tier0_data is not None else None
        ),
        "instrument_specs": None,
        "instrument_strength_by_feature": None,
        "gap_type": "vs_target",
        "min_gap_threshold": 5.0,
        "max_opportunities": 10,
        "gaps_detected": None,
        "gaps_by_segment": None,
        "total_gap_value": None,
        "roi_estimates": None,
        "total_addressable_value": None,
        "prioritized_opportunities": None,
        "quick_wins": None,
        "strategic_bets": None,
        "executive_summary": None,
        "key_insights": None,
        "detection_latency_ms": 0,
        "roi_latency_ms": 0,
        "total_latency_ms": 0,
        "segments_analyzed": 0,
        "errors": [],
        "warnings": [],
        "status": "pending",
    }
    return state


@pytest.mark.asyncio
async def test_real_gap_detector_failure_propagates_failed_status():
    """A real segment/column mismatch must end the graph in status='failed' with errors.

    RED before fix: downstream nodes overwrite status to 'completed' so the final
    state laundered the failure. After the fix the failure propagates end-to-end.
    """
    graph = create_gap_analyzer_graph()

    # Drive a REAL gap_detector failure: a malformed (non-DataFrame) tier0 passthrough
    # payload makes _derive_performance_from_tier0 raise inside gap_detector.
    state = _initial_state(
        segments=["region"],
        tier0_data=_tier0_malformed_passthrough(),
    )

    final_state = await graph.ainvoke(state)

    # The error was raised and caught by gap_detector (this is the correct, pre-existing
    # behavior) -- prove the failure was real, not synthetic.
    assert final_state.get("errors"), "expected gap_detector to have accumulated an error"
    assert any(e.get("node") == "gap_detector" for e in final_state["errors"]), (
        f"expected a gap_detector error, got {final_state.get('errors')}"
    )

    # F2 core assertion: the terminal status must be 'failed', NOT laundered to 'completed'.
    assert final_state.get("status") == "failed", (
        "terminal status was laundered to "
        f"{final_state.get('status')!r}; F2 requires 'failed' when errors are present"
    )


@pytest.mark.asyncio
async def test_clean_run_with_no_gaps_stays_completed():
    """Regression guard: a clean run that finds zero gaps must stay 'completed'.

    Do NOT over-correct "no gaps found" into "failed". With matching segment columns
    and a benchmark equal to current (no gap above threshold), the pipeline runs fine
    and should report a genuine 'completed' with no errors.
    """
    graph = create_gap_analyzer_graph()

    # A tier0 frame WITH the requested 'region' segment, but a high threshold so no
    # gap clears it -> zero gaps, no errors -> genuine completed.
    n = 60
    rng = np.random.default_rng(1)
    tier0 = pd.DataFrame(
        {
            "region": ["Northeast"] * (n // 2) + ["West"] * (n - n // 2),
            "discontinuation_flag": rng.integers(0, 2, size=n),
            "tenure_months": rng.integers(1, 48, size=n),
        }
    )

    state = _initial_state(segments=["region"], tier0_data=tier0)
    # Impossibly high threshold so no gap is reported, but the pipeline runs clean.
    state["min_gap_threshold"] = 10_000.0

    final_state = await graph.ainvoke(state)

    assert not final_state.get("errors"), (
        f"clean run should have no errors, got {final_state.get('errors')}"
    )
    assert final_state.get("status") == "completed", (
        f"clean no-gaps run should stay 'completed', got {final_state.get('status')!r}"
    )


@pytest.mark.asyncio
async def test_missing_segment_column_is_graceful_not_a_crash():
    """#851: a requested segment the data does not carry yields NO gaps and stays
    'completed' (unsupported segment), instead of KeyError-crashing the whole node.

    The tier0 frame has >= 50 rows but no 'region' column; the derived current_data
    therefore lacks 'region'. _detect_segment_gaps must short-circuit to no gaps.
    """
    graph = create_gap_analyzer_graph()

    n = 60
    rng = np.random.default_rng(2)
    tier0 = pd.DataFrame(
        {
            "discontinuation_flag": rng.integers(0, 2, size=n),
            "tenure_months": rng.integers(1, 48, size=n),
        }
    )
    state = _initial_state(segments=["region"], tier0_data=tier0)

    final_state = await graph.ainvoke(state)

    assert not final_state.get("errors"), (
        f"missing-segment is unsupported, not an error; got {final_state.get('errors')}"
    )
    assert final_state.get("status") == "completed", (
        f"missing-segment run should stay 'completed', got {final_state.get('status')!r}"
    )
    assert not (final_state.get("gaps_detected") or []), "no gaps for an unsupported segment"
