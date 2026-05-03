"""Unit tests for ``_resolve_tau_grid_for_metrics`` and ``_resolve_primary_tau``.

Covers shard 20 §F tau-resolver rows of the test plan:
- ``test_resolve_tau_grid_use_case_diagnostic``
- ``test_resolve_tau_grid_disease_override_wins``
- ``test_resolve_tau_grid_custom_with_invalid_bounds_falls_back``
- ``test_resolve_tau_grid_no_schema_returns_legacy``
- ``test_resolve_primary_tau_explicit_wins``
- ``test_resolve_primary_tau_disease_default_when_no_explicit``
- ``test_resolve_primary_tau_returns_none_when_unresolvable``
"""

from __future__ import annotations

import logging

import pytest

from src.agents.ml_foundation.model_trainer.nodes.evaluator import (
    _DISEASE_SPECIFIC_DEFAULTS,
    _USE_CASE_DEFAULTS,
    _V3_NB_GRID_P_T_VALUES,
    _resolve_primary_tau,
    _resolve_tau_grid_for_metrics,
)

# ---------------------------------------------------------------------------
# Tau-grid resolver tests
# ---------------------------------------------------------------------------


def test_resolve_tau_grid_use_case_diagnostic() -> None:
    """``use_case=diagnostic`` resolves to the [0.05, 0.30] grid."""
    sc = {"clinical_threshold_range": {"use_case": "diagnostic"}}
    grid = _resolve_tau_grid_for_metrics(sc, _V3_NB_GRID_P_T_VALUES)
    assert grid is not None
    assert len(grid) == 21
    assert grid[0] == pytest.approx(0.05)
    assert grid[-1] == pytest.approx(0.30)


def test_resolve_tau_grid_disease_override_wins() -> None:
    """A known ``dataset_disease`` overrides ``use_case`` defaults."""
    sc = {
        "dataset_disease": "breast_cancer_recurrence",
        "clinical_threshold_range": {"use_case": "screening"},
    }
    grid = _resolve_tau_grid_for_metrics(sc, _V3_NB_GRID_P_T_VALUES)
    assert grid is not None
    expected = _DISEASE_SPECIFIC_DEFAULTS["breast_cancer_recurrence"]
    assert grid[0] == pytest.approx(expected["tau_low"])
    assert grid[-1] == pytest.approx(expected["tau_high"])


def test_resolve_tau_grid_custom_with_invalid_bounds_falls_back(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Missing ``tau_low`` under ``use_case=custom`` warns + uses legacy."""
    sc = {
        "clinical_threshold_range": {
            "use_case": "custom",
            "tau_high": 0.40,
        }
    }
    with caplog.at_level(logging.WARNING):
        grid = _resolve_tau_grid_for_metrics(sc, _V3_NB_GRID_P_T_VALUES)
    assert grid == list(_V3_NB_GRID_P_T_VALUES)
    assert any(
        "use_case=custom" in rec.getMessage() and "missing or invalid" in rec.getMessage()
        for rec in caplog.records
    )


def test_resolve_tau_grid_custom_with_valid_bounds_returns_grid() -> None:
    """``use_case=custom`` with valid bounds returns the linspace grid."""
    sc = {
        "clinical_threshold_range": {
            "use_case": "custom",
            "tau_low": 0.10,
            "tau_high": 0.40,
        }
    }
    grid = _resolve_tau_grid_for_metrics(sc, _V3_NB_GRID_P_T_VALUES, n_grid_points=11)
    assert grid is not None
    assert len(grid) == 11
    assert grid[0] == pytest.approx(0.10)
    assert grid[-1] == pytest.approx(0.40)


def test_resolve_tau_grid_no_schema_returns_legacy() -> None:
    """Empty / None ``success_criteria`` falls back to ``legacy_grid``."""
    grid_none = _resolve_tau_grid_for_metrics(None, _V3_NB_GRID_P_T_VALUES)
    grid_empty = _resolve_tau_grid_for_metrics({}, _V3_NB_GRID_P_T_VALUES)
    assert grid_none == list(_V3_NB_GRID_P_T_VALUES)
    assert grid_empty == list(_V3_NB_GRID_P_T_VALUES)


def test_resolve_tau_grid_explicit_evaluation_grid_wins() -> None:
    """Caller-supplied ``evaluation_grid`` always wins."""
    sc = {
        "dataset_disease": "breast_cancer_recurrence",  # would otherwise win
        "clinical_threshold_range": {
            "use_case": "diagnostic",
            "evaluation_grid": [0.07, 0.12, 0.18, 0.22],
        },
    }
    grid = _resolve_tau_grid_for_metrics(sc, _V3_NB_GRID_P_T_VALUES)
    assert grid == [0.07, 0.12, 0.18, 0.22]


def test_resolve_tau_grid_unknown_use_case_returns_legacy() -> None:
    """An unknown ``use_case`` key falls back to legacy."""
    sc = {"clinical_threshold_range": {"use_case": "unknown_regime"}}
    grid = _resolve_tau_grid_for_metrics(sc, _V3_NB_GRID_P_T_VALUES)
    assert grid == list(_V3_NB_GRID_P_T_VALUES)


# ---------------------------------------------------------------------------
# Primary-tau resolver tests
# ---------------------------------------------------------------------------


def test_resolve_primary_tau_explicit_wins() -> None:
    """Explicit ``clinical_threshold_range.primary_tau`` beats disease default."""
    sc = {
        "dataset_disease": "cv_risk_10y",  # default primary_tau = 0.075
        "clinical_threshold_range": {"primary_tau": 0.12},
    }
    assert _resolve_primary_tau(sc) == pytest.approx(0.12)


def test_resolve_primary_tau_disease_default_when_no_explicit() -> None:
    """No explicit primary_tau → fall through to disease default."""
    sc = {"dataset_disease": "cv_risk_10y"}
    expected = _DISEASE_SPECIFIC_DEFAULTS["cv_risk_10y"]["primary_tau"]
    assert _resolve_primary_tau(sc) == pytest.approx(expected)


def test_resolve_primary_tau_returns_none_when_unresolvable() -> None:
    """Empty / out-of-range / unknown disease returns ``None``."""
    assert _resolve_primary_tau(None) is None
    assert _resolve_primary_tau({}) is None
    # Out of range explicit value.
    sc_oor = {"clinical_threshold_range": {"primary_tau": 1.5}}
    assert _resolve_primary_tau(sc_oor) is None
    # Unknown disease + no explicit → None.
    sc_unknown = {"dataset_disease": "unknown_indication"}
    assert _resolve_primary_tau(sc_unknown) is None


def test_resolve_primary_tau_handles_disease_in_ctr_subdict() -> None:
    """``dataset_disease`` may live under ``clinical_threshold_range``."""
    sc = {"clinical_threshold_range": {"dataset_disease": "hf_readmission_30d"}}
    expected = _DISEASE_SPECIFIC_DEFAULTS["hf_readmission_30d"]["primary_tau"]
    assert _resolve_primary_tau(sc) == pytest.approx(expected)


def test_use_case_defaults_complete() -> None:
    """All 5 use-case rows are populated with valid τ ordering."""
    expected = {
        "screening",
        "diagnostic",
        "treatment_decision",
        "critical_action",
        "generic_benchmark",
    }
    assert set(_USE_CASE_DEFAULTS) == expected
    for name, row in _USE_CASE_DEFAULTS.items():
        assert 0.0 < row["tau_low"] < row["tau_high"] < 1.0, name


def test_disease_specific_defaults_complete() -> None:
    """All 6 disease rows have valid τ_low < primary_tau < τ_high mostly."""
    assert len(_DISEASE_SPECIFIC_DEFAULTS) == 6
    for name, row in _DISEASE_SPECIFIC_DEFAULTS.items():
        assert 0.0 < row["tau_low"] < row["tau_high"] < 1.0, name
        assert 0.0 < row["primary_tau"] < 1.0, name
