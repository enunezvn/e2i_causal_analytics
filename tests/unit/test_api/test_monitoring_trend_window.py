"""
TDD Red-first test: /performance/{model_id}/trend route accepts days up to 1825.

This test verifies the route parameter constraint BEFORE the fix (should FAIL
with le=90) and PASSES after widening to le=1825.
"""

import inspect

import pytest
from fastapi import Query


def _get_trend_route_days_param():
    """
    Import the route handler and extract the days Query parameter metadata
    by inspecting the function signature.
    """
    from src.api.routes.monitoring import get_performance_trend

    sig = inspect.signature(get_performance_trend)
    days_param = sig.parameters.get("days")
    assert days_param is not None, "Route handler missing 'days' parameter"
    # The default is a Query(...) object; inspect its metadata
    return days_param.default


class TestMonitoringTrendWindowRoute:
    """Verify the /performance/{model_id}/trend route accepts 5 years of history."""

    def test_trend_route_days_max_is_1825(self):
        """days param le= must be 1825 (5-year cap) not the old 90-day cap."""
        days_query = _get_trend_route_days_param()
        # FastAPI Query objects expose .le via the FieldInfo metadata
        # The le value is stored in the FieldInfo (days_query.le or via gt/lt metadata)
        le_value = None
        if hasattr(days_query, "le"):
            le_value = days_query.le
        elif hasattr(days_query, "metadata"):
            # Pydantic v2 path: metadata contains annotated validators
            for meta in days_query.metadata:
                if hasattr(meta, "le"):
                    le_value = meta.le
                    break
        assert le_value == 1825, (
            f"Expected days le=1825 (5 years) but got le={le_value}. "
            "Change the Query constraint in monitoring.py get_performance_trend."
        )

    def test_trend_route_days_default_is_365(self):
        """days param default= must be 365 (1 year) not the old 30-day default."""
        days_query = _get_trend_route_days_param()
        default_value = None
        if hasattr(days_query, "default"):
            default_value = days_query.default
        assert default_value == 365, (
            f"Expected days default=365 but got default={default_value}. "
            "Change the Query default in monitoring.py get_performance_trend."
        )

    def test_trend_route_days_min_is_1(self):
        """days param ge= must remain 1 (unchanged lower bound)."""
        days_query = _get_trend_route_days_param()
        ge_value = None
        if hasattr(days_query, "ge"):
            ge_value = days_query.ge
        elif hasattr(days_query, "metadata"):
            for meta in days_query.metadata:
                if hasattr(meta, "ge"):
                    ge_value = meta.ge
                    break
        assert ge_value == 1, (
            f"Expected days ge=1 but got ge={ge_value}."
        )
