"""#2233 defect 1: the backdoor adjustment-set search is bounded (wall-time budget
+ candidate cap) and runs OFF the event loop, so a wide claims-frame DAG can no
longer stall the gunicorn heartbeat until the arbiter aborts the worker.

Measured (docs/demos/results/2026-09-22_optum_biologic_persistence_cert/live_probes,
issue #2233): on the AUGMENT-path DAG of the real 15,209-row persistence frame the
size-<=3 enumeration ran on the loop thread past ``--timeout 120`` -> code 134,
and the analysis row stayed ``running`` forever. The lane's own disproof numbers
are in the PR body.
"""

from __future__ import annotations

import asyncio
import time

import networkx as nx
import pytest

from src.agents.causal_impact.nodes.graph_builder import GraphBuilderNode
from src.ml.causal_role_dgp.backdoor import satisfies_backdoor_criterion

T, Y = "T", "Y"


def _dense_manual_dag(k: int) -> tuple[nx.DiGraph, list[str]]:
    """The manual-DAG shape the AUGMENT / REVIEW / REJECT paths ship: every declared
    covariate a common cause of T and Y. No minimal set of size <= 3 exists, so the
    unbounded search exhausts C(k, <= 3) criterion checks before its full-set
    fallback (76,154 checks at k = 77, measured 242-316 s)."""
    covs = [f"c{i:02d}" for i in range(k)]
    node = GraphBuilderNode()
    return node._construct_dag(T, Y, covs), covs


class TestBoundedSearch:
    def test_small_dag_is_unaffected_and_unbounded(self) -> None:
        from src.agents.causal_impact.nodes.adjustment_search import find_adjustment_sets

        dag = nx.DiGraph([("C", T), ("C", Y), (T, Y)])
        res = find_adjustment_sets(dag, T, Y)
        assert res.adjustment_sets == [["C"]]
        assert res.bound_hit is None
        assert res.n_candidates == 1
        assert res.warning() is None

    def test_time_budget_returns_a_valid_set_and_names_the_bound(self) -> None:
        from src.agents.causal_impact.nodes.adjustment_search import find_adjustment_sets

        dag, covs = _dense_manual_dag(80)
        t0 = time.monotonic()
        res = find_adjustment_sets(dag, T, Y, time_budget_s=0.5, max_candidates=None)
        wall = time.monotonic() - t0
        assert res.bound_hit == "time_budget"
        # Budget + at most one in-flight check + the single full-set fallback check.
        assert wall < 5.0, wall
        assert res.n_candidates == 80
        # A bound never trades validity for speed: the returned set is admissible.
        assert res.adjustment_sets == [sorted(covs)]
        assert satisfies_backdoor_criterion(dag, set(res.adjustment_sets[0]), T, Y)
        assert res.n_checks < 80 + 80 * 79 // 2  # the enumeration was cut short
        warning = res.warning()
        assert warning is not None
        assert "adjustment-set search" in warning and "0.5" in warning

    def test_candidate_cap_skips_the_enumeration_deterministically(self) -> None:
        from src.agents.causal_impact.nodes.adjustment_search import find_adjustment_sets

        dag, covs = _dense_manual_dag(80)
        t0 = time.monotonic()
        res = find_adjustment_sets(dag, T, Y, time_budget_s=None, max_candidates=40)
        wall = time.monotonic() - t0
        assert res.bound_hit == "candidate_cap"
        assert res.n_checks == 1  # only the full-candidate-set check ran
        assert wall < 2.0, wall
        assert res.adjustment_sets == [sorted(covs)]
        warning = res.warning()
        assert warning is not None and "80" in warning and "40" in warning

    def test_budget_exhausted_mid_level_keeps_the_sets_found_so_far(self) -> None:
        """The budget lands after a minimal set was already found: that set ships
        (best-so-far), not the fallback full set, and the bound is still named."""
        from src.agents.causal_impact.nodes.adjustment_search import find_adjustment_sets

        # One true confounder ``a`` plus precision covariates on Y only (degree > 0,
        # so they are candidates, but never needed). Sorted order puts ``a`` first.
        dag = nx.DiGraph([("a", T), ("a", Y), (T, Y)] + [(f"x{i}", Y) for i in range(20)])
        ticks = iter(range(10_000))

        def clock() -> float:  # one "second" per read
            return float(next(ticks))

        # Reads: start, then one per check. Checks: size 0 (empty set, fails), then
        # {a} (valid) -> budget 2.5 s is exhausted at the next check.
        res = find_adjustment_sets(dag, T, Y, time_budget_s=2.5, max_candidates=None, clock=clock)
        assert res.bound_hit == "time_budget"
        assert res.adjustment_sets == [["a"]]

    def test_defaults_are_pinned(self) -> None:
        from src.agents.causal_impact.nodes.adjustment_search import (
            ADJUSTMENT_SEARCH_MAX_CANDIDATES,
            ADJUSTMENT_SEARCH_TIME_BUDGET_S,
        )

        assert ADJUSTMENT_SEARCH_TIME_BUDGET_S == 60.0
        assert ADJUSTMENT_SEARCH_MAX_CANDIDATES == 40

    def test_node_wrapper_keeps_the_legacy_list_contract(self) -> None:
        dag = nx.DiGraph([("C", T), ("C", Y), (T, Y)])
        assert GraphBuilderNode()._find_adjustment_sets(dag, T, Y) == [["C"]]


class TestExecuteRunsTheSearchOffTheLoop:
    @pytest.mark.asyncio
    async def test_dense_manual_dag_finishes_under_budget_with_a_warning(self) -> None:
        covs = [f"c{i:02d}" for i in range(80)]
        state = {
            "query": "q",
            "query_id": "i2233",
            "treatment_var": T,
            "outcome_var": Y,
            "confounders": covs,
            "modeled_confounders": covs,
            "auto_discover": False,
            # Uncapped so the TIME-BUDGET path runs (the cap would short-circuit
            # the search to one check and prove nothing about the loop).
            "adjustment_search_time_budget_s": 0.5,
            "adjustment_search_max_candidates": None,
            "warnings": [],
        }
        ticks = 0

        async def ticker() -> None:
            nonlocal ticks
            while True:
                await asyncio.sleep(0.02)
                ticks += 1

        tick_task = asyncio.create_task(ticker())
        t0 = time.monotonic()
        try:
            out = await GraphBuilderNode().execute(state)  # type: ignore[arg-type]
        finally:
            tick_task.cancel()
        wall = time.monotonic() - t0
        assert wall < 10.0, wall
        assert out.get("status") != "failed", out.get("error_message")
        assert out["causal_graph"]["adjustment_sets"] == [sorted(covs)]
        bound_lines = [w for w in out.get("warnings", []) if "adjustment-set search" in w]
        assert len(bound_lines) == 1, out.get("warnings")
        # The loop kept turning while the search ran (gunicorn's heartbeat lives
        # there): a search on the loop thread would have starved the ticker.
        assert ticks >= 5, ticks

    @pytest.mark.asyncio
    async def test_small_dag_raises_no_bound_warning(self) -> None:
        state = {
            "query": "q",
            "query_id": "i2233-small",
            "treatment_var": T,
            "outcome_var": Y,
            "confounders": ["C"],
            "modeled_confounders": ["C"],
            "auto_discover": False,
            "warnings": [],
        }
        out = await GraphBuilderNode().execute(state)  # type: ignore[arg-type]
        assert out["causal_graph"]["adjustment_sets"] == [["C"]]
        assert not [w for w in out.get("warnings", []) if "adjustment-set search" in w]

    def test_state_declares_the_override_channels(self) -> None:
        from src.agents.causal_impact.state import CausalImpactState

        keys = CausalImpactState.__annotations__
        assert "adjustment_search_time_budget_s" in keys
        assert "adjustment_search_max_candidates" in keys
