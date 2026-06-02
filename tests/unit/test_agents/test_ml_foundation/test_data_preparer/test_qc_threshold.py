"""Unit tests for the dynamic QC ``overall_score`` threshold resolver.

The resolver is the single source of truth routed through by all three QC-gate
enforcement points. These tests pin:
- default 0.80 (UNCHANGED baseline),
- the override precedence (state > scope_spec > env > default),
- defensive coercion (malformed / out-of-range overrides are ignored),
so the gate can be made dynamic-capable WITHOUT silently moving the bar.
"""

import pytest

from src.agents.ml_foundation.data_preparer.nodes.qc_threshold import (
    DEFAULT_QC_MIN_OVERALL_SCORE,
    resolve_qc_min_overall_score,
)


class TestDefault:
    def test_default_is_080(self):
        assert DEFAULT_QC_MIN_OVERALL_SCORE == 0.80

    def test_none_state_returns_default(self):
        assert resolve_qc_min_overall_score(None) == 0.80

    def test_empty_state_returns_default(self):
        assert resolve_qc_min_overall_score({}) == 0.80

    def test_state_without_override_returns_default(self):
        state = {"experiment_id": "x", "scope_spec": {"date_column": "created_at"}}
        assert resolve_qc_min_overall_score(state) == 0.80


class TestStateOverride:
    def test_state_override_honored(self):
        assert resolve_qc_min_overall_score({"qc_min_overall_score": 0.70}) == 0.70

    def test_state_override_can_raise_bar(self):
        assert resolve_qc_min_overall_score({"qc_min_overall_score": 0.90}) == 0.90

    def test_state_override_string_coerced(self):
        assert resolve_qc_min_overall_score({"qc_min_overall_score": "0.65"}) == 0.65

    def test_state_override_beats_scope_spec_and_env(self, monkeypatch):
        monkeypatch.setenv("QC_MIN_OVERALL_SCORE", "0.50")
        state = {
            "qc_min_overall_score": 0.72,
            "scope_spec": {"qc_min_overall_score": 0.60},
        }
        assert resolve_qc_min_overall_score(state) == 0.72


class TestScopeSpecOverride:
    def test_scope_spec_override_honored(self):
        state = {"scope_spec": {"qc_min_overall_score": 0.75}}
        assert resolve_qc_min_overall_score(state) == 0.75

    def test_scope_spec_beats_env(self, monkeypatch):
        monkeypatch.setenv("QC_MIN_OVERALL_SCORE", "0.50")
        state = {"scope_spec": {"qc_min_overall_score": 0.66}}
        assert resolve_qc_min_overall_score(state) == 0.66

    def test_scope_spec_none_falls_through(self):
        assert resolve_qc_min_overall_score({"scope_spec": None}) == 0.80


class TestEnvOverride:
    def test_env_override_honored(self, monkeypatch):
        monkeypatch.setenv("QC_MIN_OVERALL_SCORE", "0.70")
        assert resolve_qc_min_overall_score({}) == 0.70

    def test_env_absent_returns_default(self, monkeypatch):
        monkeypatch.delenv("QC_MIN_OVERALL_SCORE", raising=False)
        assert resolve_qc_min_overall_score({}) == 0.80


class TestDefensiveCoercion:
    @pytest.mark.parametrize("bad", ["abc", "", "  ", object()])
    def test_non_numeric_ignored(self, bad):
        assert resolve_qc_min_overall_score({"qc_min_overall_score": bad}) == 0.80

    @pytest.mark.parametrize("bad", [-0.1, 1.5, 2.0, -5])
    def test_out_of_range_ignored(self, bad):
        assert resolve_qc_min_overall_score({"qc_min_overall_score": bad}) == 0.80

    @pytest.mark.parametrize("bad", [True, False])
    def test_bool_ignored(self, bad):
        # bool is an int subclass; must not coerce to 1.0/0.0.
        assert resolve_qc_min_overall_score({"qc_min_overall_score": bad}) == 0.80

    def test_nan_ignored(self):
        assert resolve_qc_min_overall_score({"qc_min_overall_score": float("nan")}) == 0.80

    def test_inf_ignored(self):
        assert resolve_qc_min_overall_score({"qc_min_overall_score": float("inf")}) == 0.80

    def test_boundary_values_allowed(self):
        assert resolve_qc_min_overall_score({"qc_min_overall_score": 0.0}) == 0.0
        assert resolve_qc_min_overall_score({"qc_min_overall_score": 1.0}) == 1.0

    def test_malformed_state_falls_through_to_env(self, monkeypatch):
        monkeypatch.setenv("QC_MIN_OVERALL_SCORE", "0.55")
        # state override is garbage -> ignored -> env wins
        assert resolve_qc_min_overall_score({"qc_min_overall_score": "garbage"}) == 0.55
