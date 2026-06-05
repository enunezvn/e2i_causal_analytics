"""P3 (H1/H2) — the tier1-5 smoke gate must accept an HONEST fail-closed causal
verdict and still reject a real execution failure.

After the refutation fail-open remediation, a causal_impact analysis whose
refutation BLOCKS a weak claim correctly returns ``status="failed"`` (a fully
executed analysis with a negative robustness verdict that the orchestrator then
excludes). The smoke harness previously treated ANY ``status="failed"`` as an
agent failure — an assertion that only ever passed because the old fail-open bug
reported ``completed`` regardless of the refutation verdict. These tests pin the
corrected harness semantics:

* an honest block (numeric ATE-in-CI + refutation ran + gate_decision="block")
  → PASS the quality gate + detected as tier0 passthrough;
* a real failure (no ATE, or refutation never ran/errored, or status="error")
  → still FAIL.
"""

from __future__ import annotations

from src.testing.agent_quality_gates import _validate_causal_impact, get_quality_gate
from src.testing.data_source_validator import DataSourceType, DataSourceValidator


def _honest_block_output() -> dict:
    """A fully-executed analysis whose refutation BLOCKED the claim (H1/H2)."""
    return {
        "status": "failed",
        "ate_estimate": -0.0765,
        "confidence_interval": (-0.1266, -0.0264),
        "gate_decision": "block",
        "refutation_passed": False,
        "needs_review": False,
        "refutation_tests_passed": 2,
        "refutation_tests_total": 3,
    }


class TestQualityGateAcceptsHonestBlock:
    def test_fail_on_status_no_longer_blanket_fails_causal_impact(self):
        gate = get_quality_gate("causal_impact")
        assert gate is not None
        # "failed" must NOT be a blanket smoke failure anymore; "error" stays.
        assert "failed" not in gate["fail_on_status"]
        assert "error" in gate["fail_on_status"]

    def test_semantic_validator_accepts_honest_block(self):
        passed, msg = _validate_causal_impact(_honest_block_output())
        assert passed, msg

    def test_semantic_validator_still_passes_completed(self):
        out = _honest_block_output()
        out["status"] = "completed"
        out["gate_decision"] = "proceed"
        passed, msg = _validate_causal_impact(out)
        assert passed, msg


class TestQualityGateStillRejectsRealFailures:
    def test_failed_without_ate_is_rejected(self):
        out = _honest_block_output()
        out["ate_estimate"] = None
        passed, _ = _validate_causal_impact(out)
        assert not passed

    def test_failed_when_refutation_never_ran_is_rejected(self):
        # H1: a refutation that ERRORED (never produced tests) must still fail.
        out = _honest_block_output()
        out["gate_decision"] = None
        out["refutation_tests_total"] = 0
        passed, _ = _validate_causal_impact(out)
        assert not passed

    def test_status_error_is_rejected(self):
        out = _honest_block_output()
        out["status"] = "error"
        passed, _ = _validate_causal_impact(out)
        assert not passed

    def test_ate_outside_ci_is_rejected(self):
        out = _honest_block_output()
        out["confidence_interval"] = (0.10, 0.20)  # ATE -0.0765 is outside
        passed, _ = _validate_causal_impact(out)
        assert not passed


class TestDataSourceDetectionIsVerdictIndependent:
    def test_blocked_analysis_detected_as_tier0(self):
        detected, _ = DataSourceValidator()._detect_data_source(
            agent_name="causal_impact",
            agent_output=_honest_block_output(),
            execution_logs=[],
            agent_instance=None,
        )
        assert detected == DataSourceType.TIER0_PASSTHROUGH

    def test_no_analysis_falls_back_to_unknown(self):
        detected, _ = DataSourceValidator()._detect_data_source(
            agent_name="causal_impact",
            agent_output={"status": "failed"},  # no ATE, no refutation
            execution_logs=[],
            agent_instance=None,
        )
        assert detected == DataSourceType.UNKNOWN
