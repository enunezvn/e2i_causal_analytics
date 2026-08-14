"""Red-first pins for #1599 — ``gap_calculator`` needs 2 groups with FINITE means.

The #1574 guard counts *groups*: ``len(entity_values) < 2``. A group whose
metric column is entirely null within the group still counts as a group, because
``groupby(...).mean()`` yields ``NaN`` for it — a real value in the dict, just
not a comparable one.

Measured pre-fix (see the module's red run), two distinct failure shapes:

1. Two distinct groups, both all-``NaN`` -> the guard passes and the
   top-minus-bottom arithmetic yields ``gap=nan`` (the shape #1599 describes).

2. **Wider than #1599 describes**: a SINGLE all-``NaN`` group POISONS an
   otherwise-valid comparison. ``max``/``min`` compare with ``>``/``<``, and
   every comparison against ``NaN`` is ``False``, so the first-iterated key wins
   BOTH — with groups ``{'AAA': nan, 'Ibrance': 0.7, 'Kisqali': 0.5}`` the tool
   returned ``top_performer == bottom_performer == 'AAA'`` and ``gap=nan``,
   discarding two perfectly good finite groups. That is the #1574 defect shape
   (top == bottom) reintroduced through a different door, so raising the count
   threshold alone would NOT have fixed it: the non-finite groups must be
   excluded from the comparison basis, not merely counted.

``NaN`` is additionally not JSON-compliant (``json.dumps(allow_nan=False)``
raises), so a ``NaN`` in ``entity_values`` is a serialization hazard for any
strict consumer of the tool's result.

Preserved behavior these tests also pin (the #1574 contract, unchanged):

* Two groups with EQUAL FINITE means are a REAL zero gap and must NOT refuse.
* The refusal remains a ``RuntimeError`` subclass, so every existing
  ``pytest.raises(RuntimeError)`` pin in ``test_gap_comparability_1574.py``
  keeps describing the contract truthfully.
* The singleton and no-match reasons stay byte-identical — the new
  non-finite branch must not leak into them.

Tests build their OWN DataFrames (the anti-mock rule forbids fabricating data
inside tool bodies, not in tests).
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from src.agents.tool_composer import tool_registrations as tr

_COMPARABILITY_PHRASE = "not modeled as comparable brand entities"


def _all_nan_two_group_frame() -> pd.DataFrame:
    """Two genuinely distinct brand groups; the metric is null in both."""
    return pd.DataFrame(
        {
            "brand": ["Kisqali", "Kisqali", "Ibrance", "Ibrance"],
            "market_share": [np.nan, np.nan, np.nan, np.nan],
        }
    )


def _one_finite_one_nan_frame() -> pd.DataFrame:
    """Two distinct groups; only one has any non-null metric value."""
    return pd.DataFrame(
        {
            "brand": ["Kisqali", "Kisqali", "Ibrance", "Ibrance"],
            "market_share": [0.72, 0.70, np.nan, np.nan],
        }
    )


def _nan_group_plus_two_finite_frame() -> pd.DataFrame:
    """Three groups; the all-NaN one sorts FIRST, which is what hijacks max/min."""
    return pd.DataFrame(
        {
            "brand": ["AAA", "AAA", "Kisqali", "Kisqali", "Ibrance", "Ibrance"],
            "market_share": [np.nan, np.nan, 0.50, 0.50, 0.70, 0.70],
        }
    )


# ---------------------------------------------------------------------------
# (a) two distinct groups, both all-NaN -> structured refusal, not a NaN gap
# ---------------------------------------------------------------------------
def test_two_all_nan_groups_fail_closed():
    """Pre-fix: returns GapAnalysis(gap=nan) — the guard counted 2 groups."""
    with pytest.raises(RuntimeError) as exc_info:
        tr.gap_calculator(
            metric="market_share",
            entity_type="brand",
            entities=["Kisqali", "Ibrance"],
            estimation_data=_all_nan_two_group_frame(),
        )
    msg = str(exc_info.value)

    assert "gap_calculator" in msg
    # The refusal names the finite-mean requirement, not just a group count.
    assert "finite" in msg.lower()
    assert "market_share" in msg
    # #1574's disclosure contract still rides the message.
    assert "estimation_data_scope=" in msg
    assert _COMPARABILITY_PHRASE in msg


def test_two_all_nan_groups_refusal_names_the_non_comparable_groups():
    """The user must learn WHICH groups had no usable metric values."""
    with pytest.raises(RuntimeError) as exc_info:
        tr.gap_calculator(
            metric="market_share",
            entity_type="brand",
            entities=["Kisqali", "Ibrance"],
            estimation_data=_all_nan_two_group_frame(),
        )
    msg = str(exc_info.value)

    assert "'entity_groups_non_finite'" in msg
    assert "Ibrance" in msg and "Kisqali" in msg
    assert "'entity_groups_non_finite_count': 2" in msg
    # Never a claim about the wider platform (the #1574 b2 policy).
    assert "no competitor data exists" not in msg.lower()


# ---------------------------------------------------------------------------
# (b) PRESERVED — two groups with equal FINITE means are a REAL zero gap
# ---------------------------------------------------------------------------
def test_equal_finite_means_remain_a_real_zero_gap():
    """A measured zero is a finding; only the fabricated/NaN zero is refused."""
    df = pd.DataFrame(
        {
            "brand": ["Kisqali", "Kisqali", "Ibrance", "Ibrance"],
            "market_share": [0.60, 0.60, 0.60, 0.60],
        }
    )
    result = tr.gap_calculator(
        metric="market_share",
        entity_type="brand",
        entities=["Kisqali", "Ibrance"],
        estimation_data=df,
    )
    assert result.gap == 0.0
    # BOTH groups are present — that is what separates a measured zero from the
    # #1574 fabricated zero, where a single group filled both slots. On an exact
    # tie ``max``/``min`` both return the first key, so top == bottom here is
    # correct and pre-existing; #1574 deliberately leaves it unpinned.
    assert result.entity_values == {"Kisqali": 0.60, "Ibrance": 0.60}


# ---------------------------------------------------------------------------
# (c) one finite group + one all-NaN group -> refusal naming the NaN group
#
# Reasoning for this expectation: after excluding the group with no usable
# metric values, exactly ONE comparable group remains — which is precisely the
# #1574 singleton case. Reporting it against itself would be the fabricated
# zero gap #1574 exists to forbid, and comparing it against NaN yields a NaN
# gap. So: refuse, and name the group that was dropped.
# ---------------------------------------------------------------------------
def test_one_finite_one_all_nan_group_fails_closed():
    """Pre-fix: 2 groups pass the count guard -> gap=nan."""
    with pytest.raises(RuntimeError) as exc_info:
        tr.gap_calculator(
            metric="market_share",
            entity_type="brand",
            entities=["Kisqali", "Ibrance"],
            estimation_data=_one_finite_one_nan_frame(),
        )
    msg = str(exc_info.value)

    assert "'entity_groups_non_finite'" in msg
    assert "'entity_groups_non_finite_count': 1" in msg
    assert "Ibrance" in msg  # the dropped, non-comparable group
    assert "estimation_data_scope=" in msg


# ---------------------------------------------------------------------------
# (d) THE WIDER DEFECT — one all-NaN group must not poison a valid comparison
# ---------------------------------------------------------------------------
def test_nan_group_does_not_hijack_a_valid_two_group_comparison():
    """Pre-fix: top == bottom == 'AAA' and gap = nan, with 0.70/0.50 discarded."""
    result = tr.gap_calculator(
        metric="market_share",
        entity_type="brand",
        entities=[],
        estimation_data=_nan_group_plus_two_finite_frame(),
    )

    assert math.isfinite(result.gap), f"gap must be finite, got {result.gap!r}"
    assert result.gap == pytest.approx(0.20)
    assert result.top_performer == "Ibrance"
    assert result.bottom_performer == "Kisqali"
    assert result.top_performer != result.bottom_performer


def test_returned_entity_values_are_all_finite_and_json_strict():
    """A NaN entity value is uncomparable AND not JSON-compliant."""
    result = tr.gap_calculator(
        metric="market_share",
        entity_type="brand",
        entities=[],
        estimation_data=_nan_group_plus_two_finite_frame(),
    )

    assert all(math.isfinite(v) for v in result.entity_values.values())
    assert "AAA" not in result.entity_values
    assert result.entity_values == {"Kisqali": 0.50, "Ibrance": 0.70}
    # Strict JSON is what a NaN would break for any downstream consumer.
    json.dumps(result.entity_values, allow_nan=False)


# ---------------------------------------------------------------------------
# Contract preservation — the refusal type and the untouched message branches
# ---------------------------------------------------------------------------
def test_refusal_is_a_runtime_error_subclass():
    """The fail-closed contract is documented (and pinned) as RuntimeError."""
    from src.agents.tool_composer.errors import ToolRefusalError

    assert issubclass(ToolRefusalError, RuntimeError)

    with pytest.raises(ToolRefusalError):
        tr.gap_calculator(
            metric="market_share",
            entity_type="brand",
            entities=["Kisqali", "Ibrance"],
            estimation_data=_all_nan_two_group_frame(),
        )


def test_singleton_reason_is_unchanged_by_the_non_finite_branch():
    """#1574's singleton message must not grow non-finite wording or keys."""
    reason = tr._gap_comparability_reason(
        entity_type="brand",
        group_col="brand",
        groups_present=["Kisqali"],
        groups_matched=["Kisqali"],
        entities=["Kisqali", "competitor"],
        row_count=6,
    )
    assert "only 1 distinct brand group ('Kisqali') is present" in reason
    assert "non_finite" not in reason
    assert "reporting the one available group as both the top and the bottom" in reason


def test_no_match_reason_is_unchanged_by_the_non_finite_branch():
    reason = tr._gap_comparability_reason(
        entity_type="brand",
        group_col="brand",
        groups_present=["Kisqali"],
        groups_matched=[],
        entities=["not_a_modeled_brand"],
        row_count=6,
    )
    assert "no brand group matched the requested entities" in reason
    assert "non_finite" not in reason


def test_non_finite_reason_stays_inside_the_composer_carry_limit():
    """The composer truncates from the END, where estimation_data_scope sits.

    The non-finite branch adds a list + a count to that payload, so its
    pathological case needs the same bound #1574 measured for the others.
    """
    reason = tr._gap_comparability_reason(
        entity_type="territory",
        group_col="territory",
        groups_present=[f"territory_{i}_" + "x" * 200 for i in range(400)],
        groups_matched=[f"territory_{i}_" + "x" * 200 for i in range(400)],
        groups_non_finite=[f"territory_{i}_" + "x" * 200 for i in range(400)],
        entities=[f"requested_{i}_" + "y" * 200 for i in range(400)],
        row_count=10**9,
    )
    assert len(reason) < 2_000, f"reason is {len(reason)} chars"
    assert "'entity_groups_non_finite_count': 400" in reason
    assert "estimation_data_scope=" in reason


# ---------------------------------------------------------------------------
# The guard is about the RESOLVED grouping column, not the frame (#1574 pin)
# ---------------------------------------------------------------------------
def test_region_gap_with_a_nan_region_still_compares_the_finite_regions():
    df = pd.DataFrame(
        {
            "brand": ["Kisqali"] * 6,
            "geographic_region": ["blank", "blank", "west", "west", "northeast", "northeast"],
            "market_share": [np.nan, np.nan, 0.60, 0.60, 0.80, 0.80],
        }
    )
    result = tr.gap_calculator(
        metric="market_share",
        entity_type="region",
        entities=[],
        estimation_data=df,
    )
    assert result.gap == pytest.approx(0.20)
    assert result.top_performer == "northeast"
    assert result.bottom_performer == "west"
    assert "blank" not in result.entity_values
