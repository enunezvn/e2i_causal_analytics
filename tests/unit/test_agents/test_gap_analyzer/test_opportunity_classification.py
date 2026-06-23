"""Tests for the shared opportunity-classification SSOT (T6).

The 3-bucket scheme (Quick Win / Steady Play / Strategic Bet) and the low-value
suppression floor live in ONE module imported by BOTH the prioritizer agent and
the gaps API route, so a single definition drives the persisted analysis, the
headline counts, and the per-card category badge. There is NO residual "other"
bucket: ``classify_bucket`` is a TOTAL function — every non-suppressed
opportunity is exactly one of the three.

Grounding (live data, 2026-06-23): expected_roi == (revenue - cost) / cost, so
ROI <= 0 means the opportunity does not return its cost (value-destroying). The
suppression floor is the economic break-even line, not an arbitrary number.
"""

import pytest

from src.agents.gap_analyzer.opportunity_classification import (
    MEANINGFUL_ROI_FLOOR,
    classify_bucket,
    is_low_value,
)


class TestIsLowValue:
    """Suppression floor = break-even. ROI <= 0 == revenue <= cost == noise."""

    def test_break_even_floor_is_zero(self):
        assert MEANINGFUL_ROI_FLOOR == 0.0

    def test_negative_roi_is_low_value(self):
        # Money-losing (live Fabhalta run was entirely here).
        assert is_low_value(-1.0) is True
        assert is_low_value(-0.43) is True

    def test_break_even_roi_is_low_value(self):
        # Exactly returns cost -> zero net value created -> suppress.
        assert is_low_value(0.0) is True

    def test_marginal_positive_roi_is_kept(self):
        # "report only marginal and above" — marginal-but-profitable survives.
        assert is_low_value(0.13) is False
        assert is_low_value(0.25) is False

    def test_strong_roi_is_kept(self):
        assert is_low_value(4.6) is False


class TestClassifyBucket:
    """Total 3-way partition; preserves the no-phantom-strategic-bet invariant."""

    def test_quick_win_low_difficulty_high_roi(self):
        # low effort AND ROI > 1.
        assert classify_bucket("low", expected_roi=2.0, cost=5_000.0) == "quick_win"

    def test_low_difficulty_but_marginal_roi_is_steady_play(self):
        # low effort but ROI <= 1 is NOT a quick win — it is the meaningful middle.
        assert classify_bucket("low", expected_roi=0.5, cost=5_000.0) == "steady_play"

    def test_strategic_bet_high_difficulty_high_roi_high_cost(self):
        assert classify_bucket("high", expected_roi=3.0, cost=100_000.0) == "strategic_bet"

    def test_high_difficulty_modest_roi_is_steady_play_not_phantom_bet(self):
        # #1056 invariant: a high-difficulty opportunity that does NOT clear
        # ROI>2 is the meaningful middle, NEVER a phantom strategic_bet. (Live
        # Kisqali: ROI 1.8, high, $210k -> steady_play.)
        assert classify_bucket("high", expected_roi=1.8, cost=75_000.0) == "steady_play"

    def test_high_difficulty_high_roi_low_cost_is_steady_play(self):
        # cost <= $50k bars the strategic_bet label even with strong ROI.
        assert classify_bucket("high", expected_roi=3.0, cost=40_000.0) == "steady_play"

    def test_medium_difficulty_strong_roi_is_steady_play(self):
        # Live Kisqali gem: ROI 4.6, medium, $420k was buried in "other"; it is a
        # Steady Play now (surfaced, not lost).
        assert classify_bucket("medium", expected_roi=4.6, cost=75_000.0) == "steady_play"

    @pytest.mark.parametrize(
        "difficulty,roi,cost",
        [
            ("low", 2.0, 5_000.0),
            ("low", 0.5, 5_000.0),
            ("medium", 4.6, 75_000.0),
            ("high", 1.8, 75_000.0),
            ("high", 3.0, 100_000.0),
        ],
    )
    def test_every_opportunity_maps_to_one_of_three_buckets(self, difficulty, roi, cost):
        # Totality: no "other", ever.
        assert classify_bucket(difficulty, roi, cost) in {
            "quick_win",
            "steady_play",
            "strategic_bet",
        }
