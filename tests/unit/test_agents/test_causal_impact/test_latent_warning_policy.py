"""Latent-warning surfacing policy + warnings-accumulator discipline.

Two coupled changes, both measured into existence (2026-09-02):

1. POLICY OVER ALGORITHMS. A full alpha sweep through the production seam
   (``DiscoveryRunner._run_latent_diagnostic`` + the estimand-pair predicate;
   alpha 0.005/0.01/0.02/0.05/0.1/0.2, n=2000, seeds 1-10) showed the shipped
   alpha=0.05 already sits at the knee: detection 10/10 at every level,
   specificity 0/10 through 0.05, and the effectful-frame mark rate is
   MONOTONE DECREASING in alpha with latent==observed at every level — the
   mark on an effectful frame is orientation noise, and no discovery-side
   knob reduces it further without introducing misleading false alarms. So
   fatigue is handled as policy: ``InterpretationNode`` surfaces the warning
   only when the E-value sensitivity analysis corroborates fragility
   (``robust_to_confounding`` False) or could not run (fail-open). The flag
   and payload stay on ``causal_graph`` unconditionally; ``graph_builder``
   annotates and logs but no longer raises the warning itself.

2. ACCUMULATOR DISCIPLINE. LangGraph APPENDS whatever a node returns for an
   ``operator.add`` channel, so a node that ``**state``-spreads the incoming
   state re-submits the already-accumulated ``warnings``/``errors`` lists and
   every entry multiplies (probe pinned below; the pipeline reached 16x by
   the interpretation node). Nodes must spread ``spread_safe(state)`` and
   return ONLY their new entries for accumulator channels.
"""

import operator
import re
from pathlib import Path
from typing import Annotated, Dict, List

import pytest
from typing_extensions import TypedDict

from src.agents.causal_impact.nodes.interpretation import InterpretationNode
from src.agents.causal_impact.state import spread_safe

NODES_DIR = Path("src/agents/causal_impact/nodes")
# handle_workflow_error is a LangGraph node too — it lives outside nodes/ but
# writes to the same operator.add channels, so the source scan must cover it.
SPREAD_SCANNED_SOURCES = (Path("src/agents/causal_impact/graph.py"),)

LATENT_PAYLOAD = {
    "ran": True,
    "converged": True,
    "runtime_seconds": 0.03,
    "bidirected_edges": [["treatment_arm", "persistent_180d"]],
    "treatment": "treatment_arm",
    "outcome": "persistent_180d",
    "flag": True,
}


def _interpretation_state(**overrides) -> Dict:
    """Minimal completed-pipeline state for InterpretationNode.execute."""
    state: Dict = {
        "query": "impact of treatment_arm on persistent_180d?",
        "query_id": "lw-1",
        "treatment_var": "treatment_arm",
        "outcome_var": "persistent_180d",
        "confounders": ["disease_severity"],
        "data_source": "synthetic",
        "causal_graph": {
            "nodes": ["treatment_arm", "persistent_180d", "disease_severity"],
            "edges": [
                ("disease_severity", "treatment_arm"),
                ("treatment_arm", "persistent_180d"),
            ],
            "treatment_nodes": ["treatment_arm"],
            "outcome_nodes": ["persistent_180d"],
            "adjustment_sets": [["disease_severity"]],
            "dag_dot": "digraph { }",
            "confidence": 0.8,
            "latent_diagnostic": dict(LATENT_PAYLOAD),
        },
        "estimation_result": {
            "method": "CausalForestDML",
            "ate": 0.4,
            "ate_ci_lower": 0.3,
            "ate_ci_upper": 0.5,
            "standard_error": 0.05,
            "effect_size": "medium",
            "statistical_significance": True,
            "p_value": 0.01,
            "sample_size": 1000,
            "covariates_adjusted": ["disease_severity"],
            "heterogeneity_detected": False,
        },
        "refutation_results": {
            "tests_passed": 3,
            "tests_failed": 0,
            "total_tests": 3,
            "overall_robust": True,
            "individual_tests": {},
            "confidence_adjustment": 1.0,
        },
        "sensitivity_analysis": {
            "e_value": 2.5,
            "e_value_ci": 2.2,
            "interpretation": "Effect is robust to moderate confounding",
            "robust_to_confounding": True,
            "unmeasured_confounder_strength": "moderate",
        },
        "interpretation_depth": "standard",
        "user_context": {"expertise": "analyst"},
        "status": "pending",
        "errors": [],
        "warnings": [],
    }
    state.update(overrides)
    return state


class TestLatentWarningSurfacingPolicy:
    """The warning surfaces iff flag AND (not robust OR sensitivity failed)."""

    @pytest.mark.asyncio
    async def test_suppressed_when_evalue_says_robust(self):
        """Flag up but robust_to_confounding=True: per the alpha-sweep record
        the mark on a robust estimate is orientation noise — no warning."""
        node = InterpretationNode()
        result = await node.execute(_interpretation_state())
        assert not any("Latent-confounding diagnostic" in w for w in result.get("warnings", []))

    @pytest.mark.asyncio
    async def test_surfaced_when_evalue_corroborates_fragility(self):
        node = InterpretationNode()
        state = _interpretation_state()
        state["sensitivity_analysis"] = {
            **state["sensitivity_analysis"],
            "e_value": 1.3,
            "e_value_ci": 1.1,
            "robust_to_confounding": False,
        }
        result = await node.execute(state)
        latent = [w for w in result.get("warnings", []) if "Latent-confounding diagnostic" in w]
        assert len(latent) == 1
        assert "treatment_arm" in latent[0] and "persistent_180d" in latent[0]
        assert "corroborat" in latent[0]  # names the independent E-value signal

    @pytest.mark.asyncio
    async def test_surfaced_fail_open_when_sensitivity_failed(self):
        """No sensitivity result (node raised): the diagnostic cannot be
        cross-checked, so it surfaces as a precaution."""
        node = InterpretationNode()
        state = _interpretation_state()
        state.pop("sensitivity_analysis")
        state["sensitivity_error"] = "sensitivity blew up"
        result = await node.execute(state)
        latent = [w for w in result.get("warnings", []) if "Latent-confounding diagnostic" in w]
        assert len(latent) == 1
        assert "could not be cross-checked" in latent[0]

    @pytest.mark.asyncio
    async def test_no_warning_without_flag(self):
        node = InterpretationNode()
        state = _interpretation_state()
        state["causal_graph"]["latent_diagnostic"] = {**LATENT_PAYLOAD, "flag": False}
        state["sensitivity_analysis"] = {
            **state["sensitivity_analysis"],
            "robust_to_confounding": False,
        }
        result = await node.execute(state)
        assert not any("Latent-confounding diagnostic" in w for w in result.get("warnings", []))

    @pytest.mark.asyncio
    async def test_exception_path_still_applies_the_policy(self):
        """Interpretation's own crash is a terminal path too: a corroborated
        latent flag must not die with it."""
        node = InterpretationNode()
        state = _interpretation_state(interpretation_depth="unknown-depth")
        state["sensitivity_analysis"] = {
            **state["sensitivity_analysis"],
            "robust_to_confounding": False,
        }
        result = await node.execute(state)
        assert result["status"] == "failed"
        latent = [w for w in result.get("warnings", []) if "Latent-confounding diagnostic" in w]
        assert len(latent) == 1

    @pytest.mark.asyncio
    async def test_depth_none_still_applies_the_policy(self):
        """The skip-interpretation early return completes the analysis; the
        surfacing policy must ride it too."""
        node = InterpretationNode()
        state = _interpretation_state(interpretation_depth="none")
        state["sensitivity_analysis"] = {
            **state["sensitivity_analysis"],
            "robust_to_confounding": False,
        }
        result = await node.execute(state)
        assert any("Latent-confounding diagnostic" in w for w in result.get("warnings", []))


class TestWarningsAccumulatorDiscipline:
    """Nodes must never re-submit accumulated operator.add channels."""

    def test_langgraph_reappends_spread_accumulators(self):
        """The framework behavior that makes **state spreads unsafe, pinned so
        the spread_safe requirement stays explained if LangGraph changes."""
        import asyncio

        from langgraph.graph import END, StateGraph

        class S(TypedDict):
            warnings: Annotated[List[str], operator.add]

        graph = StateGraph(S)
        graph.add_node("first", lambda s: {"warnings": ["w1"]})
        graph.add_node("second", lambda s: {**s})
        graph.set_entry_point("first")
        graph.add_edge("first", "second")
        graph.add_edge("second", END)
        out = asyncio.run(graph.compile().ainvoke({"warnings": []}))
        assert out["warnings"] == ["w1", "w1"]

    def test_spread_safe_strips_accumulator_channels(self):
        state = {"warnings": ["w"], "errors": [{"m": 1}], "status": "ok"}
        assert spread_safe(state) == {"status": "ok"}

    def test_every_state_spread_in_nodes_uses_spread_safe(self):
        """Source pin: a bare ``**state`` spread re-appends warnings/errors.
        New spread sites must go through spread_safe."""
        offenders = []
        for path in sorted(NODES_DIR.glob("*.py")) + list(SPREAD_SCANNED_SOURCES):
            for lineno, line in enumerate(path.read_text().splitlines(), 1):
                if re.search(r"\*\*state\b", line) and "spread_safe" not in line:
                    offenders.append(f"{path.name}:{lineno}: {line.strip()}")
        assert not offenders, (
            "bare **state spread(s) re-append accumulator channels:\n" + "\n".join(offenders)
        )

    @pytest.mark.asyncio
    async def test_interpretation_returns_only_new_warnings(self):
        """Happy path: pre-accumulated entries must NOT come back (LangGraph
        would append them a second time)."""
        node = InterpretationNode()
        state = _interpretation_state(warnings=["pre-existing warning"])
        result = await node.execute(state)
        assert "pre-existing warning" not in result.get("warnings", [])

    @pytest.mark.asyncio
    async def test_interpretation_exception_path_returns_only_new_errors(self):
        """The except path spreads state too: accumulated errors/warnings must
        not be re-submitted alongside the new interpretation error."""
        node = InterpretationNode()
        state = _interpretation_state(
            interpretation_depth="unknown-depth",
            warnings=["pre-existing warning"],
            errors=[{"phase": "earlier", "message": "old"}],
        )
        result = await node.execute(state)
        assert result["status"] == "failed"
        assert "pre-existing warning" not in result.get("warnings", [])
        assert all(e.get("phase") != "earlier" for e in result.get("errors", []))


class TestErrorHandlerTerminalPath:
    """handle_workflow_error ends the graph before sensitivity/interpretation
    run (estimation total failure, refutation error/failure, gate block), so it
    must apply the surfacing policy's fail-open branch itself — and it writes
    to the same operator.add channels, so accumulator discipline applies."""

    def _error_state(self, **overrides) -> Dict:
        state = _interpretation_state(
            status="running",
            current_phase="estimation",
            error_message="estimation produced no ATE",
            errors=[{"phase": "earlier", "message": "old"}],
            warnings=["pre-existing warning"],
        )
        # Sensitivity never ran on these terminal paths.
        state.pop("sensitivity_analysis", None)
        state.update(overrides)
        return state

    def test_surfaces_latent_warning_fail_open(self):
        """A flagged diagnostic can never be cross-checked on a terminal error
        path — it must surface as a precaution, as it did pre-policy."""
        from src.agents.causal_impact.graph import handle_workflow_error

        result = handle_workflow_error(self._error_state())
        latent = [w for w in result.get("warnings", []) if "Latent-confounding diagnostic" in w]
        assert len(latent) == 1
        assert "could not be cross-checked" in latent[0]

    def test_returns_only_new_entries_for_accumulator_channels(self):
        from src.agents.causal_impact.graph import handle_workflow_error

        result = handle_workflow_error(self._error_state())
        assert result["status"] == "failed"
        assert [e["message"] for e in result["errors"]] == ["estimation produced no ATE"]
        assert "pre-existing warning" not in result.get("warnings", [])

    def test_no_flag_no_warning(self):
        from src.agents.causal_impact.graph import handle_workflow_error

        state = self._error_state()
        state["causal_graph"]["latent_diagnostic"] = {**LATENT_PAYLOAD, "flag": False}
        result = handle_workflow_error(state)
        assert "warnings" not in result


class TestMlflowLatentDiagnosticObservability:
    """Item 1 (base rate): the tracker extracts the diagnostic so MLflow keeps
    a durable record (the agent-analyze job store TTL is 8h)."""

    def test_extract_metrics_reads_latent_diagnostic(self):
        from src.agents.causal_impact.mlflow_tracker import CausalImpactMLflowTracker

        tracker = CausalImpactMLflowTracker.__new__(CausalImpactMLflowTracker)
        state = _interpretation_state()
        metrics = tracker._extract_metrics({}, state)  # type: ignore[arg-type]
        assert metrics.latent_diagnostic_ran is True
        assert metrics.latent_diagnostic_flag is True

    def test_extract_metrics_tolerates_absent_payload(self):
        from src.agents.causal_impact.mlflow_tracker import CausalImpactMLflowTracker

        tracker = CausalImpactMLflowTracker.__new__(CausalImpactMLflowTracker)
        state = _interpretation_state()
        del state["causal_graph"]["latent_diagnostic"]
        metrics = tracker._extract_metrics({}, state)  # type: ignore[arg-type]
        assert metrics.latent_diagnostic_ran is None
        assert metrics.latent_diagnostic_flag is None

    def test_log_metrics_emits_latent_diagnostic_metrics(self):
        """Extraction alone is not observability — the values must reach
        mlflow.log_metric."""
        from unittest.mock import patch

        from src.agents.causal_impact.mlflow_tracker import (
            CausalImpactMetrics,
            CausalImpactMLflowTracker,
        )

        tracker = CausalImpactMLflowTracker.__new__(CausalImpactMLflowTracker)
        metrics = CausalImpactMetrics(latent_diagnostic_ran=True, latent_diagnostic_flag=False)
        with patch("mlflow.log_metric") as log_metric:
            tracker._log_metrics(metrics)
        logged = {call.args[0]: call.args[1] for call in log_metric.call_args_list}
        assert logged["latent_diagnostic_ran"] == 1
        assert logged["latent_diagnostic_flag"] == 0

    def test_log_metrics_skips_absent_latent_diagnostic(self):
        from unittest.mock import patch

        from src.agents.causal_impact.mlflow_tracker import (
            CausalImpactMetrics,
            CausalImpactMLflowTracker,
        )

        tracker = CausalImpactMLflowTracker.__new__(CausalImpactMLflowTracker)
        with patch("mlflow.log_metric") as log_metric:
            tracker._log_metrics(CausalImpactMetrics())
        logged = {call.args[0] for call in log_metric.call_args_list}
        assert "latent_diagnostic_ran" not in logged
        assert "latent_diagnostic_flag" not in logged
