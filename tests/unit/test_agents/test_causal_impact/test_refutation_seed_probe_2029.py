"""#2029: the per-refit calibration probe passes the run's seed to DoWhy."""

from __future__ import annotations

import inspect

from src.agents.causal_impact.nodes import refutation as node


def _probe_window() -> str:
    """The source window around the throwaway 1-sim calibration refute.

    Anchored on the ``logger.debug`` message that follows the probe (its first
    occurrence in the module); the window reaches back far enough to contain
    the ``refute_estimate`` call itself.
    """
    src = inspect.getsource(node)
    anchor = src.index("per-refit calibration failed")
    return src[anchor - 2000 : anchor + 400]


def test_probe_window_contains_the_calibration_refute():
    # Negative control: the window provably holds the 1-sim probe, not some
    # other refute call in the module.
    probe = _probe_window()
    assert 'method_name="placebo_treatment_refuter"' in probe
    assert "num_simulations=1," in probe


def test_probe_source_passes_random_state():
    probe = _probe_window()
    assert "random_state=" in probe, "the calibration probe must be seeded like the scored refits"
