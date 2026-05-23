"""Unit tests for the data_preparer sufficiency_check node.

Covers:
- Verdict classification across problem types (binary, multiclass, regression,
  causal_inference, time_series)
- HARD_FAIL / SOFT_FAIL / PASS branching
- causal-inference SOFT_FAIL blocks unless ``force_low_power_run`` is set
- predictive SOFT_FAIL produces warning only
- ``sufficiency_report`` always emitted
- handles missing target column and missing target_rate gracefully
- doesn't double-clobber existing blocking_issues
"""

from __future__ import annotations

from uuid import uuid4

import numpy as np
import pandas as pd
import pytest

from src.agents.ml_foundation.data_preparer.nodes.sufficiency_check import (
    run_sufficiency_check,
)


def _state(
    *,
    train_df: pd.DataFrame,
    problem_type: str,
    target_column: str = "y",
    target_rate: float | None = None,
    sufficiency: dict | None = None,
    blocking_issues: list[str] | None = None,
) -> dict:
    """Build a minimal data_preparer state dict for the sufficiency check."""
    scope: dict = {
        "problem_type": problem_type,
        "prediction_target": target_column,
        "experiment_id": "test-exp",
    }
    if sufficiency is not None:
        scope["sufficiency"] = sufficiency
    return {
        "audit_workflow_id": uuid4(),
        "experiment_id": "test-exp",
        "scope_spec": scope,
        "train_df": train_df,
        "target_rate": target_rate,
        "blocking_issues": list(blocking_issues or []),
    }


def _binary_df(n: int, prevalence: float = 0.30, n_features: int = 10) -> pd.DataFrame:
    rng = np.random.default_rng(seed=42)
    data: dict = {f"x{i}": rng.normal(size=n) for i in range(n_features)}
    n_pos = int(round(n * prevalence))
    y = np.zeros(n, dtype=int)
    y[:n_pos] = 1
    rng.shuffle(y)
    data["y"] = y
    return pd.DataFrame(data)


def _regression_df(n: int, n_features: int = 10) -> pd.DataFrame:
    rng = np.random.default_rng(seed=42)
    data: dict = {f"x{i}": rng.normal(size=n) for i in range(n_features)}
    data["y"] = rng.normal(loc=5.0, scale=2.5, size=n)
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Binary classification
# ---------------------------------------------------------------------------


class TestBinaryClassification:
    @pytest.mark.asyncio
    async def test_pass_with_ample_data(self):
        # 5000 rows, 30% prevalence, 10 features → EPV = 5000*0.3/10 = 150
        df = _binary_df(n=5000, prevalence=0.30, n_features=10)
        result = await run_sufficiency_check(
            _state(train_df=df, problem_type="binary_classification", target_rate=0.30)
        )
        assert result["sufficiency_report"]["verdict"] == "PASS"

    @pytest.mark.asyncio
    async def test_hard_fail_below_absolute_floor(self):
        # 30 rows is well below the 100 absolute floor
        df = _binary_df(n=30, prevalence=0.30, n_features=10)
        result = await run_sufficiency_check(
            _state(train_df=df, problem_type="binary_classification", target_rate=0.30)
        )
        assert result["sufficiency_report"]["verdict"] == "HARD_FAIL"
        assert any("data_sufficiency" in m for m in result["blocking_issues"])
        assert result["qc_status"] == "failed"

    @pytest.mark.asyncio
    async def test_hard_fail_on_low_epv(self):
        # EPV = 200 * 0.02 / 30 = 0.13 → HARD_FAIL via EPV<2 floor.
        # 200 rows passes absolute floor (100); the per-data n_features*2/prevalence
        # computation raises it to 3000 so EPV is the binding HARD_FAIL trigger.
        df = _binary_df(n=200, prevalence=0.02, n_features=30)
        result = await run_sufficiency_check(
            _state(train_df=df, problem_type="binary_classification", target_rate=0.02)
        )
        assert result["sufficiency_report"]["verdict"] == "HARD_FAIL"

    @pytest.mark.asyncio
    async def test_soft_fail_warns_predictive(self):
        # EPV = 5 with prevalence 0.30 and 10 features → required_n = 5*10/0.3 = 167
        # n=150 is between the absolute floor (100) and required (167) → SOFT_FAIL
        df = _binary_df(n=150, prevalence=0.30, n_features=10)
        result = await run_sufficiency_check(
            _state(train_df=df, problem_type="binary_classification", target_rate=0.30)
        )
        assert result["sufficiency_report"]["verdict"] == "SOFT_FAIL"
        # Predictive paths warn rather than block
        assert "blocking_issues" not in result or not any(
            "data_sufficiency" in m for m in result.get("blocking_issues", [])
        )
        assert any("data_sufficiency" in m for m in result["power_warnings"])

    @pytest.mark.asyncio
    async def test_detectable_mde_present(self):
        df = _binary_df(n=2000, prevalence=0.30, n_features=10)
        result = await run_sufficiency_check(
            _state(train_df=df, problem_type="binary_classification", target_rate=0.30)
        )
        report = result["sufficiency_report"]
        assert report["detectable_mde_at_current_n"] is not None
        assert report["detectable_mde_units"] == "absolute_risk_difference"
        assert report["sensitivity_grid"] is not None
        assert len(report["sensitivity_grid"]["grid"]) == 3

    @pytest.mark.asyncio
    async def test_existing_blocking_issues_preserved(self):
        df = _binary_df(n=30, prevalence=0.30, n_features=10)
        result = await run_sufficiency_check(
            _state(
                train_df=df,
                problem_type="binary_classification",
                target_rate=0.30,
                blocking_issues=["preexisting: foo failure"],
            )
        )
        # Existing issue plus our new one
        assert "preexisting: foo failure" in result["blocking_issues"]
        assert any("data_sufficiency" in m for m in result["blocking_issues"])


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------


class TestRegression:
    @pytest.mark.asyncio
    async def test_pass_with_ample_data(self):
        df = _regression_df(n=2000, n_features=10)
        result = await run_sufficiency_check(_state(train_df=df, problem_type="regression"))
        assert result["sufficiency_report"]["verdict"] == "PASS"

    @pytest.mark.asyncio
    async def test_hard_fail_below_absolute_floor(self):
        df = _regression_df(n=30, n_features=10)
        result = await run_sufficiency_check(_state(train_df=df, problem_type="regression"))
        assert result["sufficiency_report"]["verdict"] == "HARD_FAIL"

    @pytest.mark.asyncio
    async def test_soft_fail_below_ratio_floor(self):
        # ratio_floor=5, n_features=20 → required_n=100; n=80 between abs_floor=50
        # and required=100 → SOFT_FAIL
        df = _regression_df(n=80, n_features=20)
        result = await run_sufficiency_check(_state(train_df=df, problem_type="regression"))
        assert result["sufficiency_report"]["verdict"] == "SOFT_FAIL"
        assert any("data_sufficiency" in m for m in result["power_warnings"])

    @pytest.mark.asyncio
    async def test_mde_uses_data_driven_default(self):
        df = _regression_df(n=2000, n_features=10)
        result = await run_sufficiency_check(_state(train_df=df, problem_type="regression"))
        report = result["sufficiency_report"]
        # MDE assumption should be tagged as computed_from_data (we observed sigma)
        assert report["mde_assumption_used"]["source"] == "computed_from_data"


# ---------------------------------------------------------------------------
# Causal inference
# ---------------------------------------------------------------------------


class TestCausalInference:
    @pytest.mark.asyncio
    async def test_soft_fail_blocks_by_default(self):
        # D5 / D6: causal SOFT_FAIL blocks by default unless force_low_power_run set
        df = _binary_df(n=250, prevalence=0.30, n_features=10)
        result = await run_sufficiency_check(
            _state(train_df=df, problem_type="causal_inference", target_rate=0.30)
        )
        assert result["sufficiency_report"]["verdict"] == "SOFT_FAIL"
        assert any("data_sufficiency" in m for m in result["blocking_issues"])
        assert result["qc_status"] == "failed"

    @pytest.mark.asyncio
    async def test_soft_fail_warns_with_force_low_power(self):
        df = _binary_df(n=250, prevalence=0.30, n_features=10)
        result = await run_sufficiency_check(
            _state(
                train_df=df,
                problem_type="causal_inference",
                target_rate=0.30,
                sufficiency={"force_low_power_run": True},
            )
        )
        assert result["sufficiency_report"]["verdict"] == "SOFT_FAIL"
        # Should warn, not block
        assert "blocking_issues" not in result or not any(
            "data_sufficiency" in m for m in result.get("blocking_issues", [])
        )
        assert any("data_sufficiency" in m for m in result["power_warnings"])

    @pytest.mark.asyncio
    async def test_hard_fail_below_floor(self):
        df = _binary_df(n=100, prevalence=0.30, n_features=10)
        result = await run_sufficiency_check(
            _state(train_df=df, problem_type="causal_inference", target_rate=0.30)
        )
        # Causal absolute floor is 200; n=100 is below
        assert result["sufficiency_report"]["verdict"] == "HARD_FAIL"


# ---------------------------------------------------------------------------
# Time series
# ---------------------------------------------------------------------------


class TestTimeSeries:
    @pytest.mark.asyncio
    async def test_pass_with_two_cycles(self):
        # seasonal_period=7 → required = 2*7 + n_features + 1 = 26, max(100,26)=100
        df = _regression_df(n=300, n_features=5)
        result = await run_sufficiency_check(
            _state(
                train_df=df,
                problem_type="time_series",
                sufficiency={"seasonal_period": 7},
            )
        )
        assert result["sufficiency_report"]["verdict"] == "PASS"

    @pytest.mark.asyncio
    async def test_hard_fail_below_floor(self):
        df = _regression_df(n=50, n_features=5)
        result = await run_sufficiency_check(
            _state(
                train_df=df,
                problem_type="time_series",
                sufficiency={"seasonal_period": 12},
            )
        )
        assert result["sufficiency_report"]["verdict"] == "HARD_FAIL"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_missing_train_df_returns_empty_updates(self):
        result = await run_sufficiency_check(
            {
                "audit_workflow_id": uuid4(),
                "experiment_id": "x",
                "scope_spec": {"problem_type": "binary_classification"},
                "train_df": None,
            }
        )
        assert result == {}

    @pytest.mark.asyncio
    async def test_unknown_problem_type_is_skipped(self):
        df = _binary_df(n=500)
        result = await run_sufficiency_check(
            _state(train_df=df, problem_type="quantum_classification")
        )
        assert result == {}

    @pytest.mark.asyncio
    async def test_resolved_thresholds_present_in_report(self):
        df = _binary_df(n=2000, prevalence=0.30, n_features=10)
        result = await run_sufficiency_check(
            _state(train_df=df, problem_type="binary_classification", target_rate=0.30)
        )
        report = result["sufficiency_report"]
        names = {t["name"] for t in report["resolved_thresholds"]}
        # At a minimum, alpha + power + epv_floor + absolute_floor should appear
        assert {"alpha", "power_target", "epv_floor", "absolute_floor"}.issubset(names)
        # Every resolved threshold must carry source + citation (audit chain need)
        for t in report["resolved_thresholds"]:
            assert t["source"] in {
                "user_override",
                "computed_from_data",
                "literature_default",
            }
            assert t["citation"]

    @pytest.mark.asyncio
    async def test_user_override_propagates(self):
        df = _binary_df(n=2000, prevalence=0.30, n_features=10)
        result = await run_sufficiency_check(
            _state(
                train_df=df,
                problem_type="binary_classification",
                target_rate=0.30,
                sufficiency={"alpha": 0.01, "power_target": 0.90},
            )
        )
        report = result["sufficiency_report"]
        alpha_res = next(t for t in report["resolved_thresholds"] if t["name"] == "alpha")
        assert alpha_res["source"] == "user_override"
        assert alpha_res["value"] == 0.01

    @pytest.mark.asyncio
    async def test_summary_string_present(self):
        df = _binary_df(n=2000, prevalence=0.30, n_features=10)
        result = await run_sufficiency_check(
            _state(train_df=df, problem_type="binary_classification", target_rate=0.30)
        )
        assert result["sufficiency_report"]["human_readable_summary"]

    @pytest.mark.asyncio
    async def test_skips_when_scope_spec_marks_sample_data(self):
        """Skip the check when scope_spec.use_sample_data=True.

        Pins the guard added to unblock ``scripts/run_tier0_test.py`` synthetic
        regimes (and ``MLFoundationPipeline`` when called with sample-data
        mode). The data_preparer is being used as a QC validator on a small
        synthetic cohort; the actual training data is fed independently
        downstream. A HARD_FAIL on a 30-row sample (which would normally
        trip) must NOT propagate when ``use_sample_data=True``.
        """
        df = _binary_df(n=30, prevalence=0.30, n_features=10)
        state = _state(train_df=df, problem_type="binary_classification", target_rate=0.30)
        state["scope_spec"]["use_sample_data"] = True

        result = await run_sufficiency_check(state)
        assert result == {}, (
            "Sufficiency check must short-circuit when scope_spec.use_sample_data=True; "
            f"got result={result!r}"
        )

    @pytest.mark.asyncio
    async def test_runs_when_sample_data_flag_absent(self):
        """Default behavior preserved when use_sample_data is unset or False."""
        df = _binary_df(n=30, prevalence=0.30, n_features=10)
        state = _state(train_df=df, problem_type="binary_classification", target_rate=0.30)
        # use_sample_data not set → check runs as normal
        result = await run_sufficiency_check(state)
        assert result.get("sufficiency_report", {}).get("verdict") == "HARD_FAIL"
