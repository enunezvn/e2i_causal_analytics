"""Red-first tests proving the QC gate routes through the dynamic resolver.

These exercise the THREE blocking enforcement points reconciled to a single
source of truth (``resolve_qc_min_overall_score``):

1. ``quality_checker.run_quality_checks`` — the blocking-issue append.
2. ``graph.finalize_output`` — the QC-gate ``gate_passed`` decision.
3. ``model_trainer.qc_gate_checker.check_qc_gate`` — surfaces the effective
   bar in its message (it binds on the already-decided ``qc_passed``).

The central guarantees:
- DEFAULT behavior is UNCHANGED at 0.80 (0.79 blocks, 0.81 passes).
- An explicit override (state / scope_spec / env) flows end-to-end and
  changes the gate outcome.
"""

from unittest.mock import patch

import pandas as pd
import pytest

from src.agents.ml_foundation.data_preparer import graph as dp_graph
from src.agents.ml_foundation.data_preparer.nodes.quality_checker import (
    run_quality_checks,
)
from src.agents.ml_foundation.model_trainer.nodes.qc_gate_checker import check_qc_gate


# --------------------------------------------------------------------------- #
# Enforcement point 2: graph.finalize_output gate decision
# --------------------------------------------------------------------------- #
def _gate_state(overall_score, **extra):
    """Minimal state for finalize_output exercising only the score gate."""
    state = {
        "experiment_id": "exp_gate",
        "qc_status": "passed",
        "overall_score": overall_score,
        "blocking_issues": [],
        "train_df": pd.DataFrame({"f1": [1, 2, 3], "target": [0, 1, 0]}),
        "scope_spec": {},
    }
    state.update(extra)
    return state


async def _run_finalize(state):
    # Isolate the gate logic from sidecar I/O / role derivation.
    with (
        patch.object(dp_graph, "write_adaptive_verdicts_sidecar", lambda s: None),
        patch.object(dp_graph, "_derive_role_attributions_safely", lambda s: []),
    ):
        return await dp_graph.finalize_output(state)


class TestFinalizeOutputDefaultUnchanged:
    @pytest.mark.asyncio
    async def test_079_blocks_at_default(self):
        result = await _run_finalize(_gate_state(0.79))
        assert result["gate_passed"] is False

    @pytest.mark.asyncio
    async def test_081_passes_at_default(self):
        result = await _run_finalize(_gate_state(0.81))
        assert result["gate_passed"] is True

    @pytest.mark.asyncio
    async def test_exactly_080_passes_at_default(self):
        # 0.80 is NOT below 0.80 -> passes (boundary preserved).
        result = await _run_finalize(_gate_state(0.80))
        assert result["gate_passed"] is True


class TestFinalizeOutputOverrideFlows:
    @pytest.mark.asyncio
    async def test_state_override_lowers_bar_unblocks(self):
        # 0.72 would block at default 0.80, but passes when bar lowered to 0.70.
        result = await _run_finalize(_gate_state(0.72, qc_min_overall_score=0.70))
        assert result["gate_passed"] is True

    @pytest.mark.asyncio
    async def test_scope_spec_override_lowers_bar_unblocks(self):
        result = await _run_finalize(_gate_state(0.72, scope_spec={"qc_min_overall_score": 0.70}))
        assert result["gate_passed"] is True

    @pytest.mark.asyncio
    async def test_env_override_lowers_bar_unblocks(self, monkeypatch):
        monkeypatch.setenv("QC_MIN_OVERALL_SCORE", "0.70")
        result = await _run_finalize(_gate_state(0.72))
        assert result["gate_passed"] is True

    @pytest.mark.asyncio
    async def test_override_can_raise_bar_blocks(self):
        # 0.85 passes at default, but blocks when bar raised to 0.90.
        result = await _run_finalize(_gate_state(0.85, qc_min_overall_score=0.90))
        assert result["gate_passed"] is False


# --------------------------------------------------------------------------- #
# Enforcement point 1: quality_checker blocking-issue append
# --------------------------------------------------------------------------- #
class TestQualityCheckerOverride:
    @pytest.mark.asyncio
    async def test_override_message_reflects_resolved_threshold(self):
        # Force a low-scoring frame and an override that still blocks, then
        # assert the blocking message cites the RESOLVED bar, not a literal.
        state = {
            "experiment_id": "exp_qc",
            "train_df": pd.DataFrame({"col": [1, None]}),  # low completeness
            "qc_min_overall_score": 0.95,
        }
        result = await run_quality_checks(state)
        if result["overall_score"] < 0.95:
            joined = " ".join(result["blocking_issues"])
            assert "0.95" in joined
            assert result["qc_status"] == "failed"

    @pytest.mark.asyncio
    async def test_low_bar_does_not_block_marginal_score(self):
        # A frame that scores in (0, 0.80) must NOT raise an overall-score
        # blocking issue when the bar is dropped below its score.
        state = {
            "experiment_id": "exp_qc2",
            "train_df": pd.DataFrame({"col": [1, 2]}),
            "qc_min_overall_score": 0.0,
        }
        result = await run_quality_checks(state)
        overall_blocking = [b for b in result.get("blocking_issues", []) if "Overall QC score" in b]
        assert overall_blocking == []


# --------------------------------------------------------------------------- #
# Enforcement point 3: model_trainer.check_qc_gate surfaces resolved bar
# --------------------------------------------------------------------------- #
class TestModelTrainerGateSurfacesThreshold:
    @pytest.mark.asyncio
    async def test_passed_message_includes_resolved_threshold(self):
        # data_preparer carries the effective bar it enforced on the qc_report;
        # the model_trainer gate surfaces THAT bar (not a re-derived one).
        state = {
            "qc_report": {
                "qc_passed": True,
                "overall_score": 0.83,
                "qc_min_overall_score": 0.70,
            },
        }
        result = await check_qc_gate(state)
        assert result["qc_gate_passed"] is True
        assert "0.70" in result["qc_gate_message"]

    @pytest.mark.asyncio
    async def test_blocked_message_includes_resolved_threshold(self):
        state = {
            "qc_report": {
                "qc_passed": False,
                "overall_score": 0.65,
                "qc_min_overall_score": 0.70,
                "qc_errors": ["Overall QC score (0.65) below minimum threshold (0.70)"],
            },
        }
        result = await check_qc_gate(state)
        assert result["qc_gate_passed"] is False
        assert "0.70" in result["qc_gate_message"]

    @pytest.mark.asyncio
    async def test_default_message_includes_080(self):
        state = {"qc_report": {"qc_passed": True, "overall_score": 0.90}}
        result = await check_qc_gate(state)
        assert result["qc_gate_passed"] is True
        assert "0.80" in result["qc_gate_message"]
