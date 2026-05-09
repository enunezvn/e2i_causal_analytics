"""Unit tests for ``_regime_kwargs`` in scripts/run_tier0_test.py.

Pins backlog #21.3 (ultrareview bug_001): the legacy ``default``/``adverse``/
``clean`` branches must thread ``seed`` into the returned kwargs so the
user-supplied ``--seed`` propagates to ``generate_sample_data`` instead of
silently falling back to its default ``seed=42``.

Pre-fix symptom: ``--regime default --seed 100`` and ``--regime default
--seed 200`` produced byte-identical ``patient_df`` (both ran with
``seed=42`` internally) while the run artifact recorded the user's seed —
silent metadata-vs-data divergence across 3 of 7 regimes.
"""

from __future__ import annotations

import importlib

import pandas as pd
import pytest

_LEGACY_REGIMES = ("default", "adverse", "clean")
_SCENARIO_REGIMES = ("scenario_a", "scenario_a_balanced", "scenario_b", "scenario_c")


def test_legacy_regimes_thread_seed_into_kwargs() -> None:
    """Each legacy regime returns kwargs containing the user-supplied seed."""
    runner = importlib.import_module("scripts.run_tier0_test")
    for regime in _LEGACY_REGIMES:
        kwargs = runner._regime_kwargs(regime, seed=123)
        assert "seed" in kwargs, regime
        assert kwargs["seed"] == 123, regime


def test_legacy_regimes_default_seed_preserved_when_unset() -> None:
    """``seed`` defaults to 42 (matches ``generate_sample_data`` historical default)."""
    runner = importlib.import_module("scripts.run_tier0_test")
    for regime in _LEGACY_REGIMES:
        kwargs = runner._regime_kwargs(regime)
        assert kwargs["seed"] == 42, regime


def test_scenario_regimes_thread_seed_into_kwargs() -> None:
    """Scenario regimes already threaded seed pre-fix; pin that the fix didn't regress."""
    runner = importlib.import_module("scripts.run_tier0_test")
    for regime in _SCENARIO_REGIMES:
        kwargs = runner._regime_kwargs(regime, seed=777)
        assert kwargs == {"_generator": regime, "seed": 777}, regime


def test_unknown_regime_raises() -> None:
    """Unknown labels still raise ``ValueError`` with the valid-regime list."""
    runner = importlib.import_module("scripts.run_tier0_test")
    with pytest.raises(ValueError, match="regime must be one of"):
        runner._regime_kwargs("not_a_regime")


@pytest.mark.parametrize("regime", _LEGACY_REGIMES)
def test_legacy_regimes_different_seeds_produce_different_data(regime: str) -> None:
    """Two distinct ``--seed`` values must yield distinct ``patient_df`` rows.

    This is the falsifiable AC for backlog #21.3 — pre-fix, the assertion
    failed because both invocations dropped seed and ran with ``seed=42``.
    """
    runner = importlib.import_module("scripts.run_tier0_test")

    df_a = runner.generate_sample_data(
        n_samples=100,
        **runner._regime_kwargs(regime, seed=100),
    )
    df_b = runner.generate_sample_data(
        n_samples=100,
        **runner._regime_kwargs(regime, seed=200),
    )

    assert isinstance(df_a, pd.DataFrame)
    assert isinstance(df_b, pd.DataFrame)
    assert len(df_a) == len(df_b) == 100, regime

    target = "discontinuation_flag"
    if target in df_a.columns and target in df_b.columns:
        differs = (df_a[target].to_numpy() != df_b[target].to_numpy()).any()
        assert differs, (
            f"{regime}: target column identical across seeds 100 vs 200 — "
            "seed plumbing regressed (backlog #21.3 / ultrareview bug_001)"
        )
    else:
        differs = not df_a.equals(df_b)
        assert differs, f"{regime}: full dataframe identical across seeds 100 vs 200"
