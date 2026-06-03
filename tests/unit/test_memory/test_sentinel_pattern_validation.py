"""Validation tests for threshold_breach / freshness sentinel configs.

Finding M8 (memory-system-review-20260603): ``_eval_threshold_breach`` and
``_eval_freshness`` interpolate operator-supplied ``table``/``column``/
``ts_column`` straight into a PostgREST projection
(``select(f"{pk}, brand, {column}")``). ``_validate_pattern_config`` only
checked key-presence (+ ``op``) for these patterns — unlike
``invalidation_count`` which allowlists ``table``. PostgREST's projection
mini-language interprets ``*``, ``,``, ``(``, ``)``, ``:`` (resource embedding,
aliasing), so a crafted ``column`` could widen/redirect the projection and an
off-allowlist ``table`` could read PHI tables or escape ``.eq("brand", brand)``
scoping. These tests pin the allowlist + safe-identifier validation.
"""

from __future__ import annotations

import pytest

from src.memory.sentinels.registry import _validate_pattern_config


def test_threshold_breach_rejects_projection_metacharacters_in_column():
    for bad_column in ("*", "related(ssn,dob)", "col:cast", "a,b", "drop table"):
        with pytest.raises(ValueError):
            _validate_pattern_config(
                "threshold_breach",
                {"table": "causal_paths", "column": bad_column, "op": ">", "value": 0},
            )


def test_threshold_breach_rejects_off_allowlist_table():
    with pytest.raises(ValueError):
        _validate_pattern_config(
            "threshold_breach",
            {"table": "users", "column": "causal_effect_size", "op": ">", "value": 0},
        )


def test_freshness_rejects_off_allowlist_table_and_unsafe_ts_column():
    with pytest.raises(ValueError):
        _validate_pattern_config(
            "freshness",
            {"table": "auth.users", "ts_column": "created_at", "max_age_hours": 24},
        )
    with pytest.raises(ValueError):
        _validate_pattern_config(
            "freshness",
            {"table": "causal_paths", "ts_column": "created_at,ssn", "max_age_hours": 24},
        )


def test_threshold_breach_accepts_legitimate_config():
    # Regression guard: a plain identifier on an allowlisted table must pass.
    _validate_pattern_config(
        "threshold_breach",
        {"table": "causal_paths", "column": "causal_effect_size", "op": ">", "value": 0.8},
    )


def test_freshness_accepts_legitimate_config():
    _validate_pattern_config(
        "freshness",
        {"table": "triggers", "ts_column": "created_at", "max_age_hours": 24},
    )
