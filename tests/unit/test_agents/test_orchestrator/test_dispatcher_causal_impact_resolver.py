"""causal_impact chat input resolver (#1351 — owner ruling: resolvers everywhere).

The 2026-07-29 empirical pass (#1337 Step 0) hit causal_impact with 6 of 22
real chat queries; every one crashed in ~0.4-7.2ms with the raw
``ValueError: Missing required field(s): treatment_var, outcome_var,
confounders, data_source`` — causal_impact was the ONLY dispatched agent with
no input resolver at all (agent.py:181 validates the contract directly).

The resolver mirrors the proven heterogeneous_optimizer template:

(1) an explicit analyst-supplied causal spec in ``dispatch.parameters`` wins;
(2) otherwise BUILD the spec from the real KPI substrate (``KpiFrame``:
    treatment_column / outcome_column / driver_columns over real rows) and
    attach the frame as ``data`` (agent._initialize_state seeds
    ``data_cache['estimation_data']`` from it — the #606 channel);
(3) otherwise FAIL CLOSED gracefully (never the hard raise, never fabricated
    variables), with candidate variables seeded from the curated causal
    knowledge graph (the KG variable-selector infrastructure) so the user can
    re-ask precisely.

Also pinned here: the resolver sets a cooperative ``compute_deadline`` aligned
with the dispatch budget so the refutation suite self-gates instead of
orphaning to_thread compute past the timeout (the orphan-fix contract in
refutation.py).
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import pandas as pd
import pytest

from src.agents.orchestrator.nodes import dispatcher as disp
from src.agents.orchestrator.nodes.dispatcher import DispatcherNode, NeedsStructuredInput


def _agent_input(query: str, *, entities=None, user_context=None, **extra) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "query": query,
        "user_context": user_context if user_context is not None else {},
        "session_id": "sess-ci",
        "parsed_query": {"intent": "causal_effect", "entities": entities or []},
    }
    payload.update(extra)
    return payload


def _dispatch(params: Optional[Dict[str, Any]] = None, timeout_ms: int = 150000) -> Dict[str, Any]:
    return {
        "agent_name": "causal_impact",
        "priority": "critical",
        "parameters": params or {},
        "timeout_ms": timeout_ms,
        "fallback_agent": None,
        "execution_mode": "parallel",
    }


def _kpi_frame(n_rows: int = 500):
    from src.services.kpi_resolution import KpiFrame

    frame = pd.DataFrame(
        {
            "accepted": [0, 1] * (n_rows // 2),
            "acceptance_status": ["rejected", "accepted"] * (n_rows // 2),
            "converted": [0, 1] * (n_rows // 2),
            "trigger_type": ["a", "b"] * (n_rows // 2),
            "priority_score": [0.1, 0.9] * (n_rows // 2),
        }
    )
    return KpiFrame(
        frame=frame,
        outcome_column="converted",
        driver_columns=["accepted", "acceptance_status", "trigger_type", "priority_score"],
        kpi_id="WS3-BI-009",
        kpi_name="Conversion Rate",
        treatment_column="accepted",
        treatment_source_column="acceptance_status",
    )


class TestRegistryShape:
    def test_causal_impact_has_a_resolver(self) -> None:
        assert "causal_impact" in disp.INPUT_RESOLVERS

    def test_causal_impact_fails_closed_on_failed_status(self) -> None:
        # A BLOCK-gated / errored causal run reports status="failed"
        # (_build_output / _build_error_output) and must never be laundered
        # into a successful dispatch.
        assert "causal_impact" in disp._FAIL_CLOSED_ON_FAILED_STATUS

    def test_router_budget_holds_a_real_causal_run(self) -> None:
        # 30s could not hold ANY real DoWhy chain (refutation alone is ~15s on
        # linear estimators, ~60s on meta-learners); pre-resolver this never
        # surfaced because every chat dispatch crashed in <10ms. 150s matches
        # the chat surface budget (#1353 precedent).
        from src.agents.orchestrator.nodes.router import RouterNode

        dispatch = RouterNode.INTENT_TO_AGENTS["causal_effect"][0]
        assert dispatch["agent_name"] == "causal_impact"
        assert dispatch["timeout_ms"] >= 150000


class TestExplicitParams:
    def test_explicit_spec_passes_through(self) -> None:
        params = {
            "treatment_var": "rep_visits",
            "outcome_var": "trx",
            "confounders": ["specialty", "region"],
            "data_source": "analyst_upload",
        }
        resolved = disp.INPUT_RESOLVERS["causal_impact"](
            _agent_input("impact of rep visits on trx"), _dispatch(params)
        )
        assert isinstance(resolved, dict)
        assert resolved["treatment_var"] == "rep_visits"
        assert resolved["outcome_var"] == "trx"
        assert resolved["confounders"] == ["specialty", "region"]
        assert resolved["data_source"] == "analyst_upload"

    def test_explicit_spec_without_data_source_gets_labelled_default(self) -> None:
        params = {
            "treatment_var": "rep_visits",
            "outcome_var": "trx",
            "confounders": ["specialty"],
        }
        resolved = disp.INPUT_RESOLVERS["causal_impact"](
            _agent_input("impact"), _dispatch(params)
        )
        assert isinstance(resolved, dict)
        assert resolved["data_source"] == "router_parameters"

    def test_explicit_causal_path_id_passes_through(self) -> None:
        params = {
            "treatment_var": "t",
            "outcome_var": "y",
            "confounders": ["c"],
            "data_source": "s",
            "causal_path_id": "cp_real_0001",
        }
        resolved = disp.INPUT_RESOLVERS["causal_impact"](
            _agent_input("impact"), _dispatch(params)
        )
        assert isinstance(resolved, dict)
        assert resolved["causal_path_id"] == "cp_real_0001"


class TestKpiSubstrateBuild:
    def test_builds_spec_from_kpi_substrate(self, monkeypatch) -> None:
        kf = _kpi_frame()
        monkeypatch.setattr("src.services.kpi_resolution.recognize_kpi", lambda _q: object())
        monkeypatch.setattr("src.services.kpi_resolution.resolve_kpi_frame", lambda *a, **k: kf)

        resolved = disp.INPUT_RESOLVERS["causal_impact"](
            _agent_input("what drove Kisqali conversion in the west?"), _dispatch()
        )
        assert isinstance(resolved, dict)
        assert resolved["treatment_var"] == "accepted"
        assert resolved["outcome_var"] == "converted"
        # Leak guard: neither the treatment nor its raw source column may be a
        # confounder (deterministic function of the treatment).
        assert "accepted" not in resolved["confounders"]
        assert "acceptance_status" not in resolved["confounders"]
        assert set(resolved["confounders"]) == {"trigger_type", "priority_score"}
        assert resolved["data_source"] == "kpi_substrate:WS3-BI-009"
        assert resolved["data"] is kf.frame

    def test_brand_from_query_text_scopes_the_frame(self, monkeypatch) -> None:
        seen: Dict[str, Any] = {}

        def _resolve(kpi, brand, region, **kwargs):
            seen["brand"] = brand
            seen["region"] = region
            return _kpi_frame()

        monkeypatch.setattr("src.services.kpi_resolution.recognize_kpi", lambda _q: object())
        monkeypatch.setattr("src.services.kpi_resolution.resolve_kpi_frame", _resolve)

        resolved = disp.INPUT_RESOLVERS["causal_impact"](
            # No structured entities, no user_context brand: the ask TEXT is
            # the only carrier (the exact q11-class failure #1356 fixed for
            # cohort_profiler; #1351 lifts it to the shared dispatch path).
            _agent_input("what drove Kisqali conversion in the northeast region?"),
            _dispatch(),
        )
        assert isinstance(resolved, dict)
        assert seen["brand"] == "Kisqali"
        assert seen["region"] == "northeast"
        assert resolved["brand"] == "Kisqali"

    def test_sets_cooperative_compute_deadline_within_budget(self, monkeypatch) -> None:
        kf = _kpi_frame()
        monkeypatch.setattr("src.services.kpi_resolution.recognize_kpi", lambda _q: object())
        monkeypatch.setattr("src.services.kpi_resolution.resolve_kpi_frame", lambda *a, **k: kf)

        before = time.monotonic()
        resolved = disp.INPUT_RESOLVERS["causal_impact"](
            _agent_input("conversion drivers"), _dispatch(timeout_ms=150000)
        )
        after = time.monotonic()
        assert isinstance(resolved, dict)
        deadline = resolved["compute_deadline"]
        # Strictly inside the dispatch budget (self-gating headroom), and in
        # the future.
        assert before < deadline <= after + 150.0

    def test_too_few_rows_fails_closed(self, monkeypatch) -> None:
        kf = _kpi_frame(n_rows=20)
        monkeypatch.setattr("src.services.kpi_resolution.recognize_kpi", lambda _q: object())
        monkeypatch.setattr("src.services.kpi_resolution.resolve_kpi_frame", lambda *a, **k: kf)
        resolved = disp.INPUT_RESOLVERS["causal_impact"](
            _agent_input("conversion drivers"), _dispatch()
        )
        assert isinstance(resolved, NeedsStructuredInput)

    def test_no_treatment_column_fails_closed(self, monkeypatch) -> None:
        kf = _kpi_frame()
        kf.treatment_column = None  # KPI without a defined treatment
        monkeypatch.setattr("src.services.kpi_resolution.recognize_kpi", lambda _q: object())
        monkeypatch.setattr("src.services.kpi_resolution.resolve_kpi_frame", lambda *a, **k: kf)
        resolved = disp.INPUT_RESOLVERS["causal_impact"](
            _agent_input("conversion drivers"), _dispatch()
        )
        assert isinstance(resolved, NeedsStructuredInput)


class TestGracefulFailClosed:
    def test_no_kpi_fails_closed_with_all_missing_fields(self, monkeypatch) -> None:
        monkeypatch.setattr("src.services.kpi_resolution.recognize_kpi", lambda _q: None)
        monkeypatch.setattr(disp, "_kg_causal_variable_candidates", lambda _b: ([], []))
        resolved = disp.INPUT_RESOLVERS["causal_impact"](
            _agent_input("Why did Kisqali TRx drop in Q1 in the northeast region?"),
            _dispatch(),
        )
        assert isinstance(resolved, NeedsStructuredInput)
        assert resolved.agent_name == "causal_impact"
        assert set(resolved.missing) == {
            "treatment_var",
            "outcome_var",
            "confounders",
            "data_source",
        }
        err = resolved.to_error()
        assert "no values were fabricated" in err.lower()

    def test_fail_closed_seeds_kg_variable_candidates(self, monkeypatch) -> None:
        monkeypatch.setattr("src.services.kpi_resolution.recognize_kpi", lambda _q: None)
        monkeypatch.setattr(
            disp,
            "_kg_causal_variable_candidates",
            lambda _b: (["rep_visits_biweekly", "call_frequency"], ["nrx_volume"]),
        )
        resolved = disp.INPUT_RESOLVERS["causal_impact"](
            _agent_input("What is the causal impact of rep visits on TRx for Kisqali?"),
            _dispatch(),
        )
        assert isinstance(resolved, NeedsStructuredInput)
        assert "rep_visits_biweekly" in resolved.reason
        assert "nrx_volume" in resolved.reason

    def test_kg_seed_failure_never_breaks_the_fail_closed_path(self, monkeypatch) -> None:
        monkeypatch.setattr("src.services.kpi_resolution.recognize_kpi", lambda _q: None)

        def _boom(_b):
            raise RuntimeError("falkordb down")

        monkeypatch.setattr(disp, "_kg_causal_variable_candidates", _boom)
        resolved = disp.INPUT_RESOLVERS["causal_impact"](
            _agent_input("causal drivers"), _dispatch()
        )
        assert isinstance(resolved, NeedsStructuredInput)

    @pytest.mark.asyncio
    async def test_dispatch_no_longer_hard_raises_the_contract_error(self, monkeypatch) -> None:
        """The q01-class crash: a bare chat ask through the REAL dispatcher +
        REAL agent must yield the resolver's graceful fail-closed error, never
        the raw ``Missing required field(s)`` ValueError string."""
        monkeypatch.setattr("src.services.kpi_resolution.recognize_kpi", lambda _q: None)
        monkeypatch.setattr(disp, "_kg_causal_variable_candidates", lambda _b: ([], []))

        from src.agents.causal_impact import CausalImpactAgent

        agent = CausalImpactAgent(enable_mlflow=False, enable_memory=False)
        node = DispatcherNode(agent_registry={"causal_impact": agent})
        out = await node.execute(
            {
                "query": "Why did Kisqali TRx drop in Q1 in the northeast region?",
                "user_context": {},
                "session_id": "s1",
                "parsed_query": {},
                "dispatch_plan": [_dispatch()],
                "parallel_groups": [["causal_impact"]],
            }
        )
        res = out["agent_results"][0]
        assert res["success"] is False
        err = res["error"] or ""
        assert "Missing required field(s)" not in err
        assert "fail" in err.lower()
        assert "treatment_var" in err


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-q"])
