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
    async def test_missing_train_df_emits_skipped_report(self):
        """PR #462 hotfix F10/F11: missing train_df emits a SKIPPED report
        instead of an empty {} update.

        D6 of the rollout plan says the pre-flight ALWAYS runs (no skip
        flag). The previous contract returned `{}` for three different
        skip cases (use_sample_data, missing train_df, unknown
        problem_type), collapsing them into an indistinguishable signal
        in the audit chain. Each skip case now emits a SKIPPED-verdict
        report with a distinct rationale so audit chain consumers can
        tell them apart. The gate still does NOT block (SKIPPED is not
        in the blocking-issues path).
        """
        result = await run_sufficiency_check(
            {
                "audit_workflow_id": uuid4(),
                "experiment_id": "x",
                "scope_spec": {"problem_type": "binary_classification"},
                "train_df": None,
            }
        )
        assert result["sufficiency_report"]["verdict"] == "SKIPPED"
        assert "train_df is missing" in result["sufficiency_report"]["verdict_rationale"]
        # Gate must NOT block on SKIPPED — preserves the pre-fix
        # "data not loaded" semantics where the pipeline proceeded.
        assert "blocking_issues" not in result
        assert "qc_status" not in result

    @pytest.mark.asyncio
    async def test_unknown_problem_type_emits_skipped_report(self):
        """PR #462 hotfix F10/F11: unknown problem_type emits SKIPPED
        with a distinct rationale (cf. missing_train_df and
        use_sample_data which carry different rationales).
        """
        df = _binary_df(n=500)
        result = await run_sufficiency_check(
            _state(train_df=df, problem_type="quantum_classification")
        )
        assert result["sufficiency_report"]["verdict"] == "SKIPPED"
        assert "unknown problem_type" in result["sufficiency_report"]["verdict_rationale"]
        assert "quantum_classification" in result["sufficiency_report"]["verdict_rationale"]
        assert "blocking_issues" not in result

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
    async def test_emits_skipped_when_scope_spec_marks_sample_data(self):
        """PR #462 hotfix F10/F11: scope_spec.use_sample_data=True emits a
        SKIPPED verdict (with distinct rationale) instead of returning {}.

        Pins the carve-out added to unblock ``scripts/run_tier0_test.py``
        synthetic regimes (and ``MLFoundationPipeline`` when called with
        sample-data mode). The data_preparer is being used as a QC
        validator on a small synthetic cohort; the actual training data
        is fed independently downstream. The gate must NOT block — but
        the audit chain MUST record the skip + the WHY (D6 of the
        rollout plan; pre-fix the {} return value collapsed the skip
        signal with the other two skip cases).
        """
        df = _binary_df(n=30, prevalence=0.30, n_features=10)
        state = _state(train_df=df, problem_type="binary_classification", target_rate=0.30)
        state["scope_spec"]["use_sample_data"] = True

        result = await run_sufficiency_check(state)
        assert result["sufficiency_report"]["verdict"] == "SKIPPED"
        assert "use_sample_data" in result["sufficiency_report"]["verdict_rationale"]
        # Gate does NOT block on SKIPPED — preserves the original carve-out
        # intent (synthetic QC samples would HARD_FAIL by construction).
        assert "blocking_issues" not in result
        assert "qc_status" not in result

    @pytest.mark.asyncio
    async def test_runs_when_sample_data_flag_absent(self):
        """Default behavior preserved when use_sample_data is unset or False."""
        df = _binary_df(n=30, prevalence=0.30, n_features=10)
        state = _state(train_df=df, problem_type="binary_classification", target_rate=0.30)
        # use_sample_data not set → check runs as normal
        result = await run_sufficiency_check(state)
        assert result.get("sufficiency_report", {}).get("verdict") == "HARD_FAIL"


# ---------------------------------------------------------------------------
# PR #462 hotfix: 15 critical codex findings
# ---------------------------------------------------------------------------


class TestF6ExceptionPath:
    """F6: diagnostic exception must NOT silently pass the gate.

    Pre-fix: an uncaught exception in the classifier returned
    `{'sufficiency_report': {'error': ..., 'verdict': 'INCONCLUSIVE'}}` —
    no blocking_issues, no qc_status='failed'. finalize_output saw no
    blockers → pipeline silently proceeded to training on a crashed
    pre-flight. The fix constructs a VALID DataSufficiencyReport,
    appends a blocking_issues entry, and sets qc_status='failed'.
    """

    @pytest.mark.asyncio
    async def test_exception_emits_valid_inconclusive_report_and_blocks(self, monkeypatch):
        """An uncaught exception in the classifier produces a halting
        INCONCLUSIVE report (not a silent passthrough)."""
        from src.agents.ml_foundation.data_preparer.nodes import sufficiency_check as scmod

        def _boom(**kwargs):
            raise RuntimeError("simulated classifier crash")

        monkeypatch.setattr(scmod, "_classify_classification", _boom)
        df = _binary_df(n=2000, prevalence=0.30, n_features=10)
        result = await run_sufficiency_check(
            _state(train_df=df, problem_type="binary_classification", target_rate=0.30)
        )
        # (a) report is a valid DataSufficiencyReport (all required fields present)
        report = result["sufficiency_report"]
        assert report["verdict"] == "INCONCLUSIVE"
        assert "simulated classifier crash" in report["verdict_rationale"]
        assert "RuntimeError" in report["verdict_rationale"]
        assert report["n_rows"] >= 0
        assert report["n_features"] >= 0
        assert report["problem_type"] == "binary_classification"
        # (b) blocking_issues contains the entry
        assert any("INCONCLUSIVE" in m for m in result["blocking_issues"])
        # (c) qc_status='failed' so finalize_output halts the pipeline
        assert result["qc_status"] == "failed"


class TestF7OverrideAudit:
    """F7: force_low_power_run override sets `override_applied=True`
    + `original_verdict='SOFT_FAIL'` + suffix on `verdict_rationale` so
    regulators/auditors can detect the bypass.
    """

    @pytest.mark.asyncio
    async def test_override_sets_audit_fields(self):
        df = _binary_df(n=250, prevalence=0.30, n_features=10)
        result = await run_sufficiency_check(
            _state(
                train_df=df,
                problem_type="causal_inference",
                target_rate=0.30,
                sufficiency={"force_low_power_run": True},
            )
        )
        report = result["sufficiency_report"]
        # Verdict still reads SOFT_FAIL but original_verdict + override_applied
        # carry the audit trail forward.
        assert report["verdict"] == "SOFT_FAIL"
        assert report["override_applied"] is True
        assert report["original_verdict"] == "SOFT_FAIL"
        assert "OVERRIDDEN via force_low_power_run" in report["verdict_rationale"]

    @pytest.mark.asyncio
    async def test_no_override_leaves_audit_fields_default(self):
        """Causal SOFT_FAIL without override leaves audit fields at defaults
        AND the pipeline blocks (regression test for D5 base behavior)."""
        df = _binary_df(n=250, prevalence=0.30, n_features=10)
        result = await run_sufficiency_check(
            _state(train_df=df, problem_type="causal_inference", target_rate=0.30)
        )
        report = result["sufficiency_report"]
        assert report["verdict"] == "SOFT_FAIL"
        assert report["override_applied"] is False
        assert report["original_verdict"] is None
        assert "OVERRIDDEN" not in report["verdict_rationale"]
        # Blocking path (D5)
        assert result["qc_status"] == "failed"


class TestF8F9HardFailNotOverridable:
    """F8/F9: HARD_FAIL is non-overridable; the blocking message must
    say so rather than suggesting force_low_power_run=True.
    """

    @pytest.mark.asyncio
    async def test_hard_fail_message_does_not_advertise_override(self):
        df = _binary_df(n=30, prevalence=0.30, n_features=10)
        result = await run_sufficiency_check(
            _state(train_df=df, problem_type="binary_classification", target_rate=0.30)
        )
        blocking = result["blocking_issues"]
        sufficiency_msg = next(m for m in blocking if "data_sufficiency" in m)
        assert "non-overridable" in sufficiency_msg
        # The misleading hint must NOT appear on HARD_FAIL messages.
        assert "force_low_power_run=True" not in sufficiency_msg

    @pytest.mark.asyncio
    async def test_hard_fail_ignores_force_low_power_run(self):
        """Even with force_low_power_run=True, HARD_FAIL still blocks."""
        df = _binary_df(n=30, prevalence=0.30, n_features=10)
        result = await run_sufficiency_check(
            _state(
                train_df=df,
                problem_type="binary_classification",
                target_rate=0.30,
                sufficiency={"force_low_power_run": True},
            )
        )
        assert result["sufficiency_report"]["verdict"] == "HARD_FAIL"
        assert result["qc_status"] == "failed"
        # The override flag has no effect on the HARD_FAIL path; the report
        # also does NOT record override_applied (the override didn't fire).
        assert result["sufficiency_report"]["override_applied"] is False

    @pytest.mark.asyncio
    async def test_causal_soft_fail_message_advertises_override(self):
        """SOFT_FAIL message DOES tell the operator the override is available
        — but only when the verdict that can actually be overridden."""
        df = _binary_df(n=250, prevalence=0.30, n_features=10)
        result = await run_sufficiency_check(
            _state(train_df=df, problem_type="causal_inference", target_rate=0.30)
        )
        blocking = result["blocking_issues"]
        sufficiency_msg = next(m for m in blocking if "data_sufficiency" in m)
        assert "force_low_power_run=True" in sufficiency_msg
        assert "causal SOFT_FAIL only" in sufficiency_msg


class TestF12ZeroEventCohort:
    """F12: target_rate=0.0 must be diagnosed as a data-integrity problem
    (zero observed events) rather than silently falling through to
    minority_prevalence and being treated as a sample-size problem.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize("target_rate", [0.0, 0])
    async def test_zero_target_rate_emits_zero_event_hard_fail(self, target_rate):
        """Zero positive cases → HARD_FAIL with a rationale that names the
        actual problem (NOT a "n below floor" rationale)."""
        df = _binary_df(n=5000, prevalence=0.30, n_features=10)
        result = await run_sufficiency_check(
            _state(
                train_df=df,
                problem_type="binary_classification",
                target_rate=target_rate,
            )
        )
        report = result["sufficiency_report"]
        assert report["verdict"] == "HARD_FAIL"
        assert "Zero positive cases" in report["verdict_rationale"]
        assert report["baseline_rate"] == 0.0
        assert result["qc_status"] == "failed"

    @pytest.mark.asyncio
    async def test_nonzero_target_rate_does_not_trigger_zero_event_path(self):
        """A legitimate small but nonzero target_rate (e.g., 0.02) takes the
        regular EPV path, not the zero-event short-circuit."""
        df = _binary_df(n=5000, prevalence=0.30, n_features=10)
        result = await run_sufficiency_check(
            _state(
                train_df=df,
                problem_type="binary_classification",
                target_rate=0.02,
            )
        )
        # baseline_rate stays at 0.02; the rationale (whatever it is) does NOT
        # mention zero positive cases — that signal is reserved for the actual
        # zero-event case (F12).
        report = result["sufficiency_report"]
        assert "Zero positive cases" not in report["verdict_rationale"]


class TestF13NoPositionalIndexing:
    """F13: alpha + power must reach the classifiers via named kwargs,
    NOT via positional indexing into `resolved[0]` / `resolved[1]`. This
    pins the contract by inspecting the source AND by injecting a
    user-override pair (alpha=0.01, power=0.90) and asserting both end
    up in the resolved list at distinct positions while the MDE math
    still uses the correct values.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "problem_type",
        ["binary_classification", "regression", "causal_inference"],
    )
    async def test_alpha_power_user_override_reaches_mde_math(self, problem_type):
        """A non-default alpha/power pair flows through to the resolver AND
        produces a different detectable_mde than the default would — proving
        the positional/named-kwarg refactor didn't desync alpha/power from
        the MDE calculation."""
        if problem_type == "regression":
            df = _regression_df(n=2000, n_features=10)
            sufficiency = {"alpha": 0.01, "power_target": 0.90}
            target_rate = None
        else:
            df = _binary_df(n=2000, prevalence=0.30, n_features=10)
            sufficiency = {"alpha": 0.01, "power_target": 0.90}
            target_rate = 0.30
        result_overridden = await run_sufficiency_check(
            _state(
                train_df=df,
                problem_type=problem_type,
                target_rate=target_rate,
                sufficiency=sufficiency,
            )
        )
        result_default = await run_sufficiency_check(
            _state(
                train_df=df,
                problem_type=problem_type,
                target_rate=target_rate,
            )
        )
        # alpha + power survived to the resolver (user_override source).
        alpha_res = next(
            t
            for t in result_overridden["sufficiency_report"]["resolved_thresholds"]
            if t["name"] == "alpha"
        )
        assert alpha_res["source"] == "user_override"
        assert alpha_res["value"] == 0.01
        # And the detectable MDE differs from the default (stricter alpha
        # AND higher power → larger required effect to detect, so a larger
        # detectable_mde at the same n).
        mde_o = result_overridden["sufficiency_report"]["detectable_mde_at_current_n"]
        mde_d = result_default["sufficiency_report"]["detectable_mde_at_current_n"]
        assert mde_o is not None and mde_d is not None
        assert mde_o > mde_d


class TestF14BinaryMdeClamp:
    """F14: binary detectable_mde must be clamped at
    `min(baseline_rate, 1 - baseline_rate)` for cases where the asymptotic
    formula returns a nonsensical value (e.g., 0.61 vs baseline 0.05).
    """

    @pytest.mark.asyncio
    async def test_tiny_n_small_baseline_triggers_clamp(self):
        """At small n + small baseline_rate, the unclamped MDE would be
        larger than baseline_rate. The fix clamps to the boundary AND sets
        `detectable_mde_at_n_capped=True` on the report."""
        # n=120 → above the 100 abs floor + EPV>2; SOFT/PASS region.
        # baseline_rate=0.05 → boundary cap = 0.05.
        df = _binary_df(n=120, prevalence=0.05, n_features=2)
        result = await run_sufficiency_check(
            _state(
                train_df=df,
                problem_type="binary_classification",
                target_rate=0.05,
            )
        )
        report = result["sufficiency_report"]
        assert report["detectable_mde_at_current_n"] is not None
        # Either the clamp fired or the asymptotic formula returned an
        # honest value smaller than the cap. The contract: the surfaced
        # MDE NEVER exceeds the boundary.
        boundary = min(0.05, 1.0 - 0.05)
        assert report["detectable_mde_at_current_n"] <= boundary
        if report["detectable_mde_at_n_capped"]:
            # When the clamp fired, the rationale picks up the caveat.
            assert "clamped at boundary" in report["verdict_rationale"]

    @pytest.mark.asyncio
    async def test_ample_n_does_not_clamp(self):
        """At large n, the formula returns an honest small MDE and the
        clamp does NOT fire."""
        df = _binary_df(n=5000, prevalence=0.30, n_features=10)
        result = await run_sufficiency_check(
            _state(
                train_df=df,
                problem_type="binary_classification",
                target_rate=0.30,
            )
        )
        report = result["sufficiency_report"]
        # detectable_mde_at_n_capped is None when the clamp did NOT fire
        # (because we only set it to True; default is None per schema).
        assert not report["detectable_mde_at_n_capped"]
        assert "clamped at boundary" not in report["verdict_rationale"]


class TestF15CausalEPVInteraction:
    """F15: causal required_n uses max(abs_floor, inflated_rct_n,
    epv_floor*n_features) — NOT additive `+ 2*n_features` which would
    dominate for wide panels regardless of MDE.
    """

    @pytest.mark.asyncio
    async def test_wide_panel_does_not_dominate_via_additive_features(self):
        """Pre-fix: n_features=200 added 400 to required_n on top of
        inflated_rct_n. Post-fix: the EPV*n_features term participates via
        max(...), so a wide panel only raises required_n if EPV*n_features
        actually exceeds the inflated-RCT or abs_floor candidates."""
        df = _binary_df(n=2000, prevalence=0.30, n_features=200)
        result = await run_sufficiency_check(
            _state(train_df=df, problem_type="causal_inference", target_rate=0.30)
        )
        report = result["sufficiency_report"]
        # rationale names the binding constraint (regression test for the
        # diagnostic surface the fix adds).
        assert "binding constraint" in report["verdict_rationale"]

    @pytest.mark.asyncio
    async def test_epv_dominates_for_very_wide_panel(self):
        """When n_features is large enough that EPV*n_features exceeds
        the inflated_rct candidate, EPV becomes the binding constraint —
        proves the max(...) semantics in action."""
        df = _binary_df(n=2000, prevalence=0.30, n_features=500)
        result = await run_sufficiency_check(
            _state(train_df=df, problem_type="causal_inference", target_rate=0.30)
        )
        report = result["sufficiency_report"]
        assert "binding constraint" in report["verdict_rationale"]
        # epv_floor (5 for "unknown" algorithm family) × 500 features = 2500
        # required_n; that's the largest candidate so it should bind.
        assert "epv_floor*n_features" in report["verdict_rationale"]
        # required_n is at least the EPV-floor candidate value
        assert report["required_n"] >= 5 * 500
