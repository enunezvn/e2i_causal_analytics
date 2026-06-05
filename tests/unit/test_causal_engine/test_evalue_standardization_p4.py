"""P4 / H3 — E-value must standardize the effect before the RR approximation.

The Chinn(2000)/VanderWeele-Ding ``RR ≈ exp(0.91·d)`` (runner) / ``exp(d)`` (agent)
approximation requires a STANDARDIZED mean difference d. Feeding the raw ATE in
native outcome units makes the E-value scale-dependent — near 1 on a 0–1 outcome,
exploding on a dollar/count outcome — and ``sensitivity_e_value`` is a CRITICAL
gate, so the same finding can hard-BLOCK or wave through depending only on units.
"""

from __future__ import annotations

import pytest

from src.agents.causal_impact.nodes.sensitivity import SensitivityNode
from src.causal_engine.refutation_runner import RefutationRunner


class TestRunnerEngineStandardization:
    def test_evalue_scale_invariant_when_standardized(self):
        runner = RefutationRunner()
        # Scale the effect, CI AND the outcome SD by 1000 — the standardized
        # effect d = ATE/σ_Y is unchanged, so the E-value must be unchanged.
        r1 = runner._run_sensitivity_test(2.0, (1.5, 2.5), outcome_std=1.0)
        r2 = runner._run_sensitivity_test(2000.0, (1500.0, 2500.0), outcome_std=1000.0)
        assert r1.details["standardized"] is True
        assert r1.details["e_value"] == pytest.approx(r2.details["e_value"], rel=1e-6)
        assert r1.details["e_value_ci"] == pytest.approx(r2.details["e_value_ci"], rel=1e-6)

    def test_unstandardized_is_scale_dependent(self):
        runner = RefutationRunner()
        # Without an outcome SD the E-value is computed on the raw effect and
        # explodes with the outcome scale — documenting the bug.
        r1 = runner._run_sensitivity_test(2.0, (1.5, 2.5), outcome_std=None)
        r2 = runner._run_sensitivity_test(20.0, (15.0, 25.0), outcome_std=None)
        assert r1.details["standardized"] is False
        assert r2.details["e_value"] > r1.details["e_value"] * 10


class TestAgentEngineStandardization:
    def test_evalue_scale_invariant_when_standardized(self):
        node = SensitivityNode()
        e1 = node._calculate_e_value(2.0, outcome_std=1.0)
        e2 = node._calculate_e_value(2000.0, outcome_std=1000.0)
        assert e1 == pytest.approx(e2, rel=1e-6)

    def test_unstandardized_is_scale_dependent(self):
        node = SensitivityNode()
        e1 = node._calculate_e_value(2.0)
        e2 = node._calculate_e_value(5.0)
        assert e2 > e1 * 5
