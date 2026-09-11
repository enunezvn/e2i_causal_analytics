"""``roi_estimator`` refuses a ``value_per_unit`` it cannot use instead of substituting 1.0 (#2003).

#2003 declared ``value_per_unit`` to the planner. The body replaced any non-numeric or
non-finite value with 1.0, so a planned ``"50"`` produced an ROI 50x too small — disclosed
only as ``value_per_unit (1)`` in ``assumptions``. A supplied value must be a finite number
above zero; omitting it keeps the documented 1.0.
"""

from __future__ import annotations

import pytest

from src.agents.tool_composer import tool_registrations as tr
from src.agents.tool_composer.errors import ToolRefusalError

GAP = {"gap": 100.0, "entity_values": {"a": 50.0, "b": 150.0, "c": 120.0}}


@pytest.mark.parametrize("value", ["50", float("nan"), float("inf"), 0.0, -2.0, True])
def test_an_unusable_value_per_unit_is_refused(value):
    with pytest.raises(ToolRefusalError, match="value_per_unit"):
        tr.roi_estimator(gap_analysis=GAP, investment=1000.0, value_per_unit=value)


def test_a_valid_value_per_unit_scales_the_roi():
    base = tr.roi_estimator(gap_analysis=GAP, investment=1000.0)
    scaled = tr.roi_estimator(gap_analysis=GAP, investment=1000.0, value_per_unit=50)
    assert scaled.estimated_roi == pytest.approx(50 * base.estimated_roi)


def test_an_omitted_or_null_value_per_unit_is_the_documented_one():
    base = tr.roi_estimator(gap_analysis=GAP, investment=1000.0)
    assert base.estimated_roi == pytest.approx(100.0 * 3 / 1000.0)
    explicit_null = tr.roi_estimator(gap_analysis=GAP, investment=1000.0, value_per_unit=None)
    assert explicit_null.estimated_roi == pytest.approx(base.estimated_roi)
