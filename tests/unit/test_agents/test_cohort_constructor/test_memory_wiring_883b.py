"""#883 PR B unit tests: CohortConstructorAgent.run must WIRE contribute_to_memory.

The cohort_constructor memory hooks existed since the 4-memory rollout but had
zero production callers, leaving the learning-loop readers (get_prior_cohorts /
get_effective_rules_for_brand) permanently empty. These tests pin the wiring
contract (the faithful real-DB proof lives in
tests/integration/test_agent_memory_wiring_883b.py):

* enable_memory=True (default) -> contribute_to_memory called once post-run
  (graph AND direct modes) with a status-normalized result, the state the
  hook reads, and the caller's session_id;
* enable_memory=False          -> no call attempted;
* 046-trap posture             -> a RAISING contribution never changes the
  run's result or raises to the caller;
* status normalization         -> graph-mode "completed" reaches the hook as
  "success" (the hook's gate/reader vocabulary).

The recorder/raiser patches re-patch the conftest autouse offline guard
(``_no_real_memory_contribution``) — same attribute, test-local behavior.
"""

from typing import Any, Dict, Optional

import pytest

from src.agents.cohort_constructor.agent import CohortConstructorAgent

_AGENT_ATTR = "src.agents.cohort_constructor.agent.contribute_to_memory"


def _make_agent(**kwargs: Any) -> CohortConstructorAgent:
    kwargs.setdefault("enable_observability", False)
    return CohortConstructorAgent(**kwargs)


@pytest.fixture()
def full_patient_df():
    """Frame satisfying the remibrutinib CSU config INCLUDING the exclusion
    fields — graph mode fails the construction when an exclusion criterion's
    field is absent (direct mode tolerates it), and these tests need a
    successful graph run."""
    import pandas as pd

    n = 6
    return pd.DataFrame(
        {
            "patient_journey_id": [f"P88{i}" for i in range(n)],
            "age_at_diagnosis": [25, 45, 62, 19, 33, 55],
            "diagnosis_code": ["L50.1"] * n,
            "diagnosis_date": ["2023-01-15"] * n,
            "urticaria_severity_uas7": [18, 20, 25, 17, 22, 19],
            "prior_antihistamine_therapy": [True] * n,
            "active_autoimmune_condition": [False] * n,
            "concurrent_immunosuppressive": [False] * n,
            "pregnancy_status": [False] * n,
            "severe_hepatic_impairment": [False] * n,
            "first_observation_date": ["2020-01-01"] * n,
            "last_observation_date": ["2024-12-01"] * n,
        }
    )


@pytest.fixture()
def recorder(monkeypatch):
    calls = []

    async def _record(
        result: Dict[str, Any],
        state: Dict[str, Any],
        memory_hooks=None,
        session_id: Optional[str] = None,
        region: Optional[str] = None,
    ) -> Dict[str, int]:
        calls.append(
            {
                "result": result,
                "state": state,
                "session_id": session_id,
            }
        )
        return {
            "episodic_stored": 0,
            "semantic_stored": 0,
            "working_cached": 0,
            "rules_stored": 0,
        }

    monkeypatch.setattr(_AGENT_ATTR, _record)
    return calls


class TestMemoryWiringGraphMode:
    @pytest.mark.asyncio
    async def test_graph_run_contributes_once_with_normalized_status(
        self, recorder, full_patient_df
    ):
        agent = _make_agent(use_graph=True)
        _, result = await agent.run(
            full_patient_df, brand="remibrutinib", session_id="session-883b-cc"
        )

        assert len(recorder) == 1, "run must contribute to memory exactly once"
        call = recorder[0]
        # Graph mode terminates in the state vocabulary ("completed") — the
        # agent normalizes it to the hook's gate/reader vocabulary.
        assert result.status == "completed"
        assert call["result"]["status"] == "success"
        assert call["result"]["cohort_id"] == result.cohort_id
        # The state carries the keys the hook reads.
        assert "config" in call["state"]
        assert "eligibility_stats" in call["state"]
        assert call["session_id"] == "session-883b-cc"

    @pytest.mark.asyncio
    async def test_default_agent_has_memory_enabled(self):
        agent = _make_agent()
        assert agent.enable_memory is True


class TestMemoryWiringDirectMode:
    @pytest.mark.asyncio
    async def test_direct_run_contributes_with_synthesized_state(
        self, recorder, remibrutinib_patient_df
    ):
        agent = _make_agent(use_graph=False)
        _, result = await agent.run(
            remibrutinib_patient_df, brand="remibrutinib", session_id="session-883b-cc-direct"
        )

        assert len(recorder) == 1
        call = recorder[0]
        assert call["result"]["status"] == "success"
        # Direct mode has no graph state — the agent synthesizes the minimal
        # shape the hook reads.
        assert call["state"]["config"].get("brand") == "remibrutinib"
        assert call["state"]["eligibility_stats"] == result.eligibility_stats
        assert call["session_id"] == "session-883b-cc-direct"

    def test_run_sync_does_not_contribute(self, recorder, remibrutinib_patient_df):
        """run_sync has no event loop; memory is async-only by design."""
        agent = _make_agent(use_graph=False)
        agent.run_sync(remibrutinib_patient_df, brand="remibrutinib")
        assert recorder == []


class TestMemoryWiringDisabled:
    @pytest.mark.asyncio
    async def test_run_skips_memory_when_disabled(self, recorder, full_patient_df):
        agent = _make_agent(enable_memory=False)
        _, result = await agent.run(full_patient_df, brand="remibrutinib")

        assert recorder == []
        assert result.status in ("success", "completed")

    def test_memory_hooks_property_returns_none_when_disabled(self):
        agent = _make_agent(enable_memory=False)
        assert agent.memory_hooks is None


class TestMemoryFailureNonBlocking:
    """046-trap posture: a memory failure can never poison the run."""

    @pytest.mark.asyncio
    async def test_raising_contribution_does_not_change_result(self, monkeypatch, full_patient_df):
        async def _boom(*args, **kwargs):
            raise RuntimeError("fabricated memory failure (046-trap probe)")

        monkeypatch.setattr(_AGENT_ATTR, _boom)

        agent = _make_agent(use_graph=True)
        eligible_df, result = await agent.run(
            full_patient_df, brand="remibrutinib", session_id="session-883b-trap"
        )

        assert result.status in ("success", "completed")
        assert result.error_message is None
        assert len(eligible_df) == len(result.eligible_patient_ids)

    @pytest.mark.asyncio
    async def test_failed_construction_not_stored(self, recorder, monkeypatch):
        """The hook's own gate skips non-success constructions; the agent
        still hands them over (the gate lives in one place)."""
        import pandas as pd

        agent = _make_agent(use_graph=True)
        # A frame missing every required field -> failed construction.
        bad_df = pd.DataFrame({"some_column": [1, 2, 3]})
        _, result = await agent.run(bad_df, brand="remibrutinib")

        assert result.status not in ("success", "completed")
        # The contribution fires (single site), but with the failed status —
        # the hook's gate is responsible for skipping storage.
        assert len(recorder) == 1
        assert recorder[0]["result"]["status"] == result.status


class TestMemoryHooksProperty:
    def test_lazy_singleton_when_enabled(self):
        from src.agents.cohort_constructor.memory_hooks import (
            CohortConstructorMemoryHooks,
        )

        agent = _make_agent()
        hooks = agent.memory_hooks
        assert isinstance(hooks, CohortConstructorMemoryHooks)
        assert agent.memory_hooks is hooks

    def test_init_failure_is_swallowed(self, monkeypatch):
        def _boom():
            raise RuntimeError("hooks factory down")

        monkeypatch.setattr(
            "src.agents.cohort_constructor.agent.get_cohort_constructor_memory_hooks", _boom
        )
        agent = _make_agent()
        assert agent.memory_hooks is None
