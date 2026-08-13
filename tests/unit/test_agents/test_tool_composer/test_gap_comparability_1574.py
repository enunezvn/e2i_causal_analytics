"""Red-first pins for #1574 — ``gap_calculator`` must fail closed below 2 groups.

A "Kisqali vs competitors" market-share ask reached ``gap_calculator`` with a
SINGLE-brand frame: the orchestrator dispatch resolves one focal-brand-filtered
estimation frame per plan (``dispatcher._extract_brand_region`` ->
``query_entities.brand_from_text`` binds EXACTLY ONE brand, then
``resolve_kpi_frame`` / ``resolve_cohort_frame`` filters to it), so the brand
column carries a single distinct value. The pre-#1574 guard fired only on an
EMPTY match, so the singleton fell through to
``top_performer == bottom_performer == "Kisqali"`` with ``gap=0.0`` — the shape
of a comparison with none of the content.

The fix fails closed at this seam. Two things it must NOT do:

* mirror the focal brand into both slots (the defect), and
* claim "no competitor data exists" — ``Brand.competitor`` is a real enum member
  and ``patient_journeys`` carries competitor RWD rows; it is only
  ``business_metrics`` that deliberately excludes them
  (``business_metrics_generator.py``). The honest statement is about what THIS
  estimation frame models, mirroring the KPI tool's unmatched-brand envelope
  (``chatbot_tools._query_kpis``): name what was requested, enumerate what is
  actually available, never assert non-existence.

Tests build their OWN DataFrames (the anti-mock rule forbids fabricating data
inside tool bodies, not in tests).
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.agents.tool_composer import tool_registrations as tr

# The b2 wording the failure must carry, and the flat non-existence claims it
# must never carry.
_COMPARABILITY_PHRASE = "not modeled as comparable brand entities"
_FORBIDDEN_PHRASES = (
    "no competitor data exists",
    "competitor data does not exist",
    "no such brand exists",
)


def _single_brand_frame() -> pd.DataFrame:
    """The #1574 shape: a focal-brand-filtered frame (one distinct brand)."""
    return pd.DataFrame(
        {
            "brand": ["Kisqali"] * 6,
            "geographic_region": ["west", "west", "northeast", "northeast", "south", "south"],
            "market_share": [0.7265, 0.71, 0.74, 0.73, 0.70, 0.72],
        }
    )


def _multi_brand_frame() -> pd.DataFrame:
    """A genuinely comparable frame: three distinct brand groups."""
    return pd.DataFrame(
        {
            "brand": ["Kisqali", "Kisqali", "Fabhalta", "Fabhalta", "Remibrutinib", "Remibrutinib"],
            "market_share": [0.70, 0.80, 0.40, 0.50, 0.10, 0.20],
        }
    )


# ---------------------------------------------------------------------------
# b1/b2/b3 — single-brand frame + a requested competitor -> fail closed.
# ---------------------------------------------------------------------------
def test_single_brand_frame_with_competitor_entity_fails_closed():
    df = _single_brand_frame()
    with pytest.raises(RuntimeError) as exc_info:
        tr.gap_calculator(
            metric="market_share",
            entity_type="brand",
            entities=["Kisqali", "competitor"],
            estimation_data=df,
        )
    msg = str(exc_info.value)

    # b1 — the reason is the group count, not an empty match.
    assert "at least 2 distinct" in msg
    assert "gap_calculator" in msg

    # b2 — scoped to THIS estimation data; never a flat non-existence claim.
    assert _COMPARABILITY_PHRASE in msg
    lowered = msg.lower()
    for forbidden in _FORBIDDEN_PHRASES:
        assert forbidden not in lowered, f"failure claims non-existence: {forbidden!r}"

    # b3 — the scope the synthesizer needs to disclose what WAS covered.
    assert "estimation_data_scope=" in msg
    assert "'grouping_column': 'brand'" in msg
    assert "'entity_groups_present': ['Kisqali']" in msg
    assert "'entities_requested': ['Kisqali', 'competitor']" in msg
    assert "'row_count': 6" in msg


def test_single_brand_frame_failure_names_no_fabricated_gap():
    """The 0.0-gap / mirrored-performer result must never be produced."""
    df = _single_brand_frame()
    with pytest.raises(RuntimeError) as exc_info:
        tr.gap_calculator(
            metric="market_share",
            entity_type="brand",
            entities=["Kisqali", "competitor"],
            estimation_data=df,
        )
    # The refusal is explicit about what it is refusing to fabricate.
    assert "Refusing to fabricate" in str(exc_info.value)


# ---------------------------------------------------------------------------
# b1 — singleton with NO entity filter (entities=[]) -> same fail-closed.
# ---------------------------------------------------------------------------
def test_singleton_group_without_entity_filter_fails_closed():
    df = _single_brand_frame()
    with pytest.raises(RuntimeError) as exc_info:
        tr.gap_calculator(
            metric="market_share",
            entity_type="brand",
            entities=[],
            estimation_data=df,
        )
    msg = str(exc_info.value)
    assert "at least 2 distinct" in msg
    assert _COMPARABILITY_PHRASE in msg
    assert "'entities_requested': []" in msg
    assert "'entity_groups_present': ['Kisqali']" in msg


# ---------------------------------------------------------------------------
# b1 — filtering a multi-group frame DOWN to one group also fails closed.
# ---------------------------------------------------------------------------
def test_multi_brand_frame_filtered_to_one_entity_fails_closed():
    df = _multi_brand_frame()
    with pytest.raises(RuntimeError) as exc_info:
        tr.gap_calculator(
            metric="market_share",
            entity_type="brand",
            entities=["Kisqali"],
            estimation_data=df,
        )
    msg = str(exc_info.value)
    assert "at least 2 distinct" in msg
    # The scope discloses the groups the FRAME carries, not just the survivor.
    assert "'entity_groups_present': ['Fabhalta', 'Kisqali', 'Remibrutinib']" in msg
    assert "'entity_groups_matched': ['Kisqali']" in msg


# ---------------------------------------------------------------------------
# MUST NOT CHANGE — >= 2 distinct groups still produce the real comparison.
# ---------------------------------------------------------------------------
def test_multi_brand_comparison_unchanged():
    df = _multi_brand_frame()
    out = tr.gap_calculator(
        metric="market_share",
        entity_type="brand",
        entities=[],
        estimation_data=df,
    )
    assert out.entity_values == {
        "Kisqali": pytest.approx(0.75),
        "Fabhalta": pytest.approx(0.45),
        "Remibrutinib": pytest.approx(0.15),
    }
    assert out.top_performer == "Kisqali"
    assert out.bottom_performer == "Remibrutinib"
    assert out.gap == pytest.approx(0.60)


def test_two_entity_filter_unchanged():
    df = _multi_brand_frame()
    out = tr.gap_calculator(
        metric="market_share",
        entity_type="brand",
        entities=["Kisqali", "Fabhalta"],
        estimation_data=df,
    )
    assert set(out.entity_values) == {"Kisqali", "Fabhalta"}
    assert out.top_performer == "Kisqali"
    assert out.bottom_performer == "Fabhalta"
    assert out.gap == pytest.approx(0.30)


def test_region_gap_on_single_brand_frame_unchanged():
    """The single-brand frame is still perfectly comparable ACROSS REGIONS.

    #1574 is about the group count on the resolved grouping column — it must not
    degrade region gaps computed from the same focal-brand frame.
    """
    df = _single_brand_frame()
    out = tr.gap_calculator(
        metric="market_share",
        entity_type="region",
        entities=[],
        estimation_data=df,
        group_by="geographic_region",
    )
    assert set(out.entity_values) == {"west", "northeast", "south"}
    assert out.top_performer == "northeast"
    assert out.bottom_performer == "south"
    assert out.gap == pytest.approx(0.735 - 0.71)


def test_measured_zero_gap_between_two_groups_is_preserved():
    """A REAL zero gap between two distinct groups is a finding, not a defect.

    Only the fabricated zero (one group mirrored into both slots) is refused.
    """
    df = pd.DataFrame(
        {
            "brand": ["Kisqali", "Kisqali", "Fabhalta", "Fabhalta"],
            "market_share": [0.40, 0.60, 0.45, 0.55],
        }
    )
    out = tr.gap_calculator(
        metric="market_share",
        entity_type="brand",
        entities=[],
        estimation_data=df,
    )
    assert set(out.entity_values) == {"Kisqali", "Fabhalta"}
    assert out.gap == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# MUST NOT CHANGE — the pre-existing fail-closed guards.
# ---------------------------------------------------------------------------
def test_empty_after_entity_filter_still_fails_closed():
    df = _multi_brand_frame()
    with pytest.raises(RuntimeError) as exc_info:
        tr.gap_calculator(
            metric="market_share",
            entity_type="brand",
            entities=["not_a_modeled_brand"],
            estimation_data=df,
        )
    msg = str(exc_info.value)
    assert "Refusing to fabricate" in msg
    assert "'entity_groups_matched': []" in msg
    assert "'entities_requested': ['not_a_modeled_brand']" in msg


def test_missing_dataframe_guard_unchanged():
    with pytest.raises(RuntimeError) as exc_info:
        tr.gap_calculator(metric="market_share", entity_type="brand", entities=[])
    msg = str(exc_info.value)
    assert "requires a real DataFrame" in msg
    assert _COMPARABILITY_PHRASE not in msg


def test_missing_metric_column_guard_unchanged():
    df = _multi_brand_frame()
    with pytest.raises(RuntimeError) as exc_info:
        tr.gap_calculator(
            metric="not_a_column",
            entity_type="brand",
            entities=[],
            estimation_data=df,
        )
    msg = str(exc_info.value)
    assert "metric column 'not_a_column' not found" in msg
    assert _COMPARABILITY_PHRASE not in msg


def test_unresolvable_grouping_column_guard_unchanged():
    df = pd.DataFrame({"market_share": [0.1, 0.2], "unrelated": ["a", "b"]})
    with pytest.raises(RuntimeError) as exc_info:
        tr.gap_calculator(
            metric="market_share",
            entity_type="brand",
            entities=[],
            estimation_data=df,
        )
    msg = str(exc_info.value)
    assert "could not resolve a grouping column" in msg
    assert _COMPARABILITY_PHRASE not in msg
