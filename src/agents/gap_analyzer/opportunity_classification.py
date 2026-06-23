"""Shared opportunity-classification SSOT for the Gap Analyzer (T6).

ONE definition of the 3-bucket scheme + the low-value suppression floor, imported
by BOTH the prioritizer agent (``nodes/prioritizer.py``) and the gaps API route
(``src/api/routes/gaps.py``). A single source means the persisted analysis, the
headline counts, and the per-card category badge can never disagree.

Buckets (``classify_bucket`` is a TOTAL function — every non-suppressed
opportunity is exactly one of these; there is NO residual "other"):

- ``quick_win``     — low effort AND ROI > 1: fast, cheap, profitable.
- ``strategic_bet`` — high effort AND ROI > 2 AND cost > $50k: a big, expensive,
  high-impact play.
- ``steady_play``   — everything else that survives suppression: the meaningful
  middle ground (e.g. a medium-effort solid earner, or a high-effort but modest
  bet that does not clear the strategic-bet bar).

The split deliberately preserves the #1056 "no-phantom-strategic-bet" invariant:
a high-difficulty opportunity that does NOT clear ROI>2 AND cost>$50k is a
``steady_play``, never a ``strategic_bet`` — so the strategic-bet headline count
is never inflated by raw difficulty.

Suppression (``is_low_value``): ``expected_roi == (revenue - cost) / cost``, so
ROI <= 0 means the opportunity does not return its cost (value-destroying or
break-even). Such items are low-value noise and are suppressed rather than dumped
into a junk bucket. The floor is the economic break-even line, not an arbitrary
threshold; marginal-but-profitable opportunities (ROI just above 0) are KEPT.
"""

from typing import Literal

# Break-even floor. An opportunity whose expected ROI is at or below this returns
# no more than its cost and is suppressed as low-value noise. 0.0 == revenue
# equals cost. Deliberately NOT a positive "meaningful" bar: the principled,
# data-grounded boundary is profitability, and marginal earners are still real.
MEANINGFUL_ROI_FLOOR: float = 0.0

# Bucket thresholds (mirror the historical prioritizer definitions so behaviour
# for the curated quick_win / strategic_bet buckets is unchanged).
_QUICK_WIN_MIN_ROI: float = 1.0
_STRATEGIC_BET_MIN_ROI: float = 2.0
_STRATEGIC_BET_MIN_COST: float = 50_000.0

OpportunityBucket = Literal["quick_win", "steady_play", "strategic_bet"]


def is_low_value(expected_roi: float) -> bool:
    """True when an opportunity is low-value noise (does not return its cost).

    ``expected_roi <= MEANINGFUL_ROI_FLOOR`` (== revenue <= cost). Suppressed
    items are hidden from the surfaced opportunity set rather than shown in a
    junk bucket. Marginal-but-profitable opportunities are NOT low-value.
    """
    return expected_roi <= MEANINGFUL_ROI_FLOOR


def classify_bucket(
    difficulty: str,
    expected_roi: float,
    cost: float,
) -> OpportunityBucket:
    """Assign an opportunity to exactly one of the three buckets (TOTAL).

    Args:
        difficulty: Implementation difficulty ("low" | "medium" | "high").
        expected_roi: Base ROI ratio ((revenue - cost) / cost).
        cost: Estimated one-time cost to close (USD).

    Returns:
        ``"quick_win"`` | ``"steady_play"`` | ``"strategic_bet"``. Never "other".

    Note:
        This does not itself suppress money-losers — call ``is_low_value`` first
        and drop those before classifying. A surviving low-ROI opportunity
        (0 < ROI <= 1) correctly classifies as ``steady_play``.
    """
    if difficulty == "low" and expected_roi > _QUICK_WIN_MIN_ROI:
        return "quick_win"
    if (
        difficulty == "high"
        and expected_roi > _STRATEGIC_BET_MIN_ROI
        and cost > _STRATEGIC_BET_MIN_COST
    ):
        return "strategic_bet"
    return "steady_play"
