"""#879 unit tests: check_health must WIRE contribute_to_memory (gated, non-blocking).

PR #878 (#876) proved the health_score memory HOOK persists; #879 is about the
CALLER — ``HealthScoreAgent.check_health`` never invoked it, so production
wrote no episodic memories despite a working hook. These tests pin the wiring
contract (the faithful real-DB persistence proof lives in
``tests/integration/test_health_score_memory_wiring_879.py``):

* enable_memory=True (default)  -> contribute_to_memory called once post-run
  with the output dict, the FINAL graph state, and the caller's session_id;
* enable_memory=False           -> no call attempted at all;
* 046-trap posture              -> a RAISING contribution (fabricated failure,
  not a fabricated result) never changes the run's status/errors;
* memory_hooks property         -> lazy, gated, init-failure-swallowing
  (mirrors experiment_monitor).

The recorder/raiser patches below re-patch the conftest autouse offline guard
(``_no_real_memory_contribution``) — same attribute, test-local behavior.
"""

from typing import Any, Dict, Optional

import pytest

from src.agents.health_score.agent import HealthScoreAgent

_AGENT_ATTR = "src.agents.health_score.agent.contribute_to_memory"


def _make_agent(**kwargs: Any) -> HealthScoreAgent:
    """Default test agent: trackers off (no MLflow/Opik I/O in unit scope)."""
    kwargs.setdefault("enable_mlflow", False)
    kwargs.setdefault("enable_opik", False)
    return HealthScoreAgent(**kwargs)


@pytest.fixture()
def recorder(monkeypatch):
    """Record the kwargs check_health hands to contribute_to_memory."""
    calls = []

    async def _record(
        result: Dict[str, Any],
        state: Dict[str, Any],
        memory_hooks=None,
        session_id: Optional[str] = None,
    ) -> Dict[str, int]:
        calls.append(
            {
                "result": result,
                "state": state,
                "memory_hooks": memory_hooks,
                "session_id": session_id,
            }
        )
        return {"episodic_stored": 0, "working_cached": 0}

    monkeypatch.setattr(_AGENT_ATTR, _record)
    return calls


class TestMemoryWiringEnabled:
    """enable_memory=True (the default): the post-run contribution fires."""

    @pytest.mark.asyncio
    async def test_check_health_contributes_once_with_result_state_session(self, recorder):
        agent = _make_agent()
        output = await agent.check_health(scope="quick", session_id="session-879")

        assert len(recorder) == 1, "check_health must contribute to memory exactly once"
        call = recorder[0]
        # The OUTPUT the caller returns is what gets stored.
        assert call["result"]["overall_health_score"] == output.overall_health_score
        assert call["result"]["health_grade"] == output.health_grade
        assert call["result"]["status"] == output.status
        # The FINAL graph state (not the initial state) rides along: the
        # compose node has run, so status is terminal and check_scope is set.
        assert call["state"].get("status") == "completed"
        assert call["state"].get("check_scope") == "quick"
        # Session plumbing (#879): the caller's session id reaches the hook.
        assert call["session_id"] == "session-879"

    @pytest.mark.asyncio
    async def test_full_scope_also_contributes(self, recorder):
        agent = _make_agent()
        await agent.check_health(scope="full")

        assert len(recorder) == 1
        assert recorder[0]["state"].get("check_scope") == "full"
        # No session passed -> None reaches the hook, which generates a UUID.
        assert recorder[0]["session_id"] is None

    @pytest.mark.asyncio
    async def test_default_agent_has_memory_enabled(self):
        """Production default (the route constructs without the flag): on."""
        agent = HealthScoreAgent(enable_mlflow=False, enable_opik=False)
        assert agent.enable_memory is True


class TestMemoryWiringDisabled:
    """enable_memory=False: no contribution is attempted at all."""

    @pytest.mark.asyncio
    async def test_check_health_skips_memory_when_disabled(self, recorder):
        agent = _make_agent(enable_memory=False)
        output = await agent.check_health(scope="quick", session_id="session-879-off")

        assert recorder == [], "enable_memory=False must not attempt any contribution"
        assert output.status == "completed"

    def test_memory_hooks_property_returns_none_when_disabled(self):
        agent = _make_agent(enable_memory=False)
        assert agent.memory_hooks is None


class TestMemoryFailureNonBlocking:
    """046-trap posture: a memory failure can never poison the run."""

    @pytest.mark.asyncio
    async def test_raising_contribution_does_not_change_status_or_errors(self, monkeypatch):
        async def _boom(*args, **kwargs):
            raise RuntimeError("fabricated memory failure (046-trap probe)")

        monkeypatch.setattr(_AGENT_ATTR, _boom)

        agent = _make_agent()
        output = await agent.check_health(scope="quick", session_id="session-879-trap")

        # The run completed; the swallow happened caller-side.
        assert output.status == "completed"
        assert all("046-trap probe" not in str(e) for e in output.errors), (
            f"memory failure leaked into agent errors: {output.errors}"
        )

    @pytest.mark.asyncio
    async def test_mlflow_failure_after_graph_completion_still_contributes(self, recorder):
        """codex r2: the contribution is keyed to the GRAPH outcome, not the
        telemetry wrappers'. An MLflow logging failure AFTER the graph completed
        (fabricated FAILURE, not a fabricated result) must not suppress the real
        measurement's trend datapoint — even though the caller receives the
        failed fallback for the escaped telemetry exception."""
        from contextlib import asynccontextmanager

        class _ExplodingMlflowTracker:
            @asynccontextmanager
            async def start_health_run(self, experiment_name, check_scope):
                yield

            async def log_health_result(self, output, result):
                raise RuntimeError("fabricated mlflow logging failure (post-graph)")

        agent = HealthScoreAgent(enable_mlflow=True, enable_opik=False)
        agent._mlflow_tracker = _ExplodingMlflowTracker()  # bypass lazy init

        output = await agent.check_health(scope="quick", session_id="session-879-mlflow")

        # The escaped telemetry exception keeps the existing caller contract:
        # the hard-failure fallback is returned.
        assert output.status == "failed"
        assert any("mlflow logging failure" in str(e) for e in output.errors)

        # ...but the BUILT output (the real, completed measurement) was still
        # contributed exactly once, with the completed graph state.
        assert len(recorder) == 1
        call = recorder[0]
        assert call["result"]["status"] == "completed"
        assert call["result"]["health_grade"] == "F"  # unmeasured-dims run
        assert call["state"].get("status") == "completed"
        assert call["session_id"] == "session-879-mlflow"

    @pytest.mark.asyncio
    async def test_graph_failure_contributes_nothing(self, recorder, monkeypatch):
        """When the GRAPH itself raises there is no trustworthy measurement:
        no output is built and nothing is contributed."""

        class _ExplodingGraph:
            async def ainvoke(self, state):
                raise RuntimeError("fabricated graph failure")

        agent = _make_agent()
        agent._quick_graph = _ExplodingGraph()

        output = await agent.check_health(scope="quick", session_id="session-879-graphfail")

        assert output.status == "failed"
        assert recorder == [], "a failed graph must not contribute a fabricated measurement"

    @pytest.mark.asyncio
    async def test_baseline_output_identical_with_and_without_memory(self, recorder):
        """The contribution is observability-only: the agent's output is
        byte-equivalent whether memory is on or off (timestamps/latency aside)."""
        on = await _make_agent().check_health(scope="quick")
        off = await _make_agent(enable_memory=False).check_health(scope="quick")

        for field in (
            "overall_health_score",
            "health_grade",
            "status",
            "data_provenance",
            "critical_issues",
            "recommendations",
        ):
            assert getattr(on, field) == getattr(off, field), field


class TestMemoryHooksProperty:
    """Lazy hooks accessor mirrors the experiment_monitor convention."""

    def test_lazy_singleton_when_enabled(self):
        from src.agents.health_score.memory_hooks import HealthScoreMemoryHooks

        agent = _make_agent()
        hooks = agent.memory_hooks
        assert isinstance(hooks, HealthScoreMemoryHooks)
        # Lazy-cached: same instance on re-access.
        assert agent.memory_hooks is hooks

    def test_init_failure_is_swallowed(self, monkeypatch):
        def _boom():
            raise RuntimeError("hooks factory down")

        monkeypatch.setattr("src.agents.health_score.agent.get_health_score_memory_hooks", _boom)
        agent = _make_agent()
        # Never raises; degrades to None (contribute_to_memory would then
        # resolve its own singleton, or fail non-blockingly downstream).
        assert agent.memory_hooks is None
