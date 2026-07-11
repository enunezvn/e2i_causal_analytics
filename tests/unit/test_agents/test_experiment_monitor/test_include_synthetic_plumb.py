"""Red-first tests: include_synthetic plumb (route -> input -> state -> query).

Issue (live audit): the experiment-monitor path never threaded the
``include_synthetic`` opt-in from the API request through to the state the
nodes read, so the monitor always ran real-mode default-exclude. With the live
substrate (360 synthetic "running" A/B experiments + 216k synthetic
assignments, and only 693 zero-enrollment real training rows) the Overview /
cards / health-pie surfaced the empty training rows and HID the real A/B data.

PR #894 already plumbed ``state["include_synthetic"]`` into every node read
(health_checker / srm_detector). The GAP these tests pin is the upstream
handoff that #894 missed:

    TriggerMonitorRequest.include_synthetic
        -> ExperimentMonitorInput.include_synthetic
        -> ExperimentMonitorAgent initial_state["include_synthetic"]
        -> HealthCheckerNode._get_experiments provenance predicate

These tests reuse the shared default-exclude resolver convention (gap_analyzer
#877 / het #880 / #894): the opt-in defaults False (real mode); only an
explicit truthy flag surfaces the synthetic substrate. No mocks return
plausible-but-fake monitor results — the synthetic vs real rows are tagged via
``is_synthetic`` exactly as the live ``ml_experiments`` table is.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.experiment_monitor import ExperimentMonitorAgent, ExperimentMonitorInput
from src.agents.experiment_monitor.nodes.health_checker import HealthCheckerNode

from .conftest import MockSupabaseClient


def _provenance_tagged_client() -> MockSupabaseClient:
    """Client whose ml_experiments carries one real + one synthetic running row.

    One real + one synthetic A/B experiment (both carry an
    intervention_channel — migration 102's genuine-A/B predicate excludes
    channel-less lineage rows, so provenance stays the only variable under
    test here). ``apply_provenance_filter`` appends
    ``.eq('is_synthetic', False)`` in real mode, which the mock query honours via
    its eq-filter, so the synthetic row is excluded unless opted in.
    """
    client = MockSupabaseClient()
    now = datetime.now(timezone.utc)
    client.set_mock_data(
        "ml_experiments",
        [
            # brand: the unscoped sweep restricts to the 3-brand portfolio
            # (2026-07-11); live training + A/B rows carry a platform brand,
            # so the fixture mirrors that shape.
            # intervention_channel: the sweep excludes channel-less lineage
            # rows (migration 102's genuine-A/B predicate) — these rows are
            # A/B experiments so provenance stays the only variable under test.
            {
                "id": "exp-real-693",
                "experiment_name": "Real A/B Experiment",
                "status": "running",
                "prediction_target": "conversion",
                "created_at": (now - timedelta(days=7)).isoformat(),
                "is_synthetic": False,
                "brand": "Kisqali",
                "intervention_channel": "email_campaign",
            },
            {
                "id": "exp-synth-360",
                "experiment_name": "Synthetic A/B Experiment",
                "status": "running",
                "prediction_target": "conversion",
                "created_at": (now - timedelta(days=7)).isoformat(),
                "is_synthetic": True,
                "brand": "Fabhalta",
                "intervention_channel": "email_campaign",
            },
        ],
    )
    return client


class TestInputCarriesIncludeSynthetic:
    """ExperimentMonitorInput must expose the opt-in (default False)."""

    def test_input_has_include_synthetic_field_default_false(self):
        # RED until ExperimentMonitorInput gains the field.
        assert ExperimentMonitorInput().include_synthetic is False

    def test_input_accepts_explicit_opt_in(self):
        assert ExperimentMonitorInput(include_synthetic=True).include_synthetic is True


class TestHealthCheckerProvenancePlumb:
    """_get_experiments must honour state['include_synthetic']."""

    @pytest.mark.asyncio
    async def test_excludes_synthetic_by_default(self):
        node = HealthCheckerNode()
        client = _provenance_tagged_client()
        state = {"check_all_active": True, "include_synthetic": False}
        rows = await node._get_experiments(client, state)  # type: ignore[arg-type]
        ids = {r["id"] for r in rows}
        assert ids == {"exp-real-693"}, ids

    @pytest.mark.asyncio
    async def test_includes_synthetic_when_opted_in(self):
        node = HealthCheckerNode()
        client = _provenance_tagged_client()
        state = {"check_all_active": True, "include_synthetic": True}
        rows = await node._get_experiments(client, state)  # type: ignore[arg-type]
        ids = {r["id"] for r in rows}
        assert ids == {"exp-real-693", "exp-synth-360"}, ids


class TestAgentInitialStatePlumb:
    """The agent must thread the input flag into the graph initial_state."""

    @pytest.mark.asyncio
    async def test_agent_threads_opt_in_to_node_query(self, monkeypatch):
        """End-to-end: ExperimentMonitorInput(include_synthetic=True) must reach
        the health_checker query so the synthetic A/B experiments are surfaced.

        RED until the agent's initial_state sets include_synthetic from the
        input. We patch the node's async client resolver to the tagged client and
        disable memory so the run is offline and deterministic.
        """
        client = _provenance_tagged_client()

        async def _get_client():
            return client

        monkeypatch.setattr(HealthCheckerNode, "_get_client", lambda self: _get_client())

        agent = ExperimentMonitorAgent(enable_memory=False)

        opted_in = await agent.run_async(
            ExperimentMonitorInput(check_all_active=True, include_synthetic=True)
        )
        opted_in_ids = {e["experiment_id"] for e in opted_in.experiments}
        assert "exp-synth-360" in opted_in_ids, opted_in_ids

        default = await agent.run_async(
            ExperimentMonitorInput(check_all_active=True, include_synthetic=False)
        )
        default_ids = {e["experiment_id"] for e in default.experiments}
        assert "exp-synth-360" not in default_ids, default_ids
        assert "exp-real-693" in default_ids, default_ids


class TestAsyncSweepTaskPlumb:
    """The async /monitor path's Celery task must honour the opt-in (codex R1)."""

    def _run_task_and_capture_opt_in(self, include_synthetic):
        """Run check_all_active_experiments with a tagged client; return the
        provenance opt-in that reached the ml_experiments query."""
        from src.tasks import ab_testing_tasks

        captured: dict = {}

        def _fake_apply(query, opt_in=False):
            captured["opt_in"] = opt_in
            # apply_provenance_filter(query, opt_in).execute() is awaited; an
            # empty roster short-circuits before any interim queueing.
            stub = MagicMock()
            stub.execute = AsyncMock(return_value=MagicMock(data=[]))
            return stub

        client = MockSupabaseClient()

        async def _async_client():
            return client

        with (
            patch.object(ab_testing_tasks, "get_async_supabase_client", _async_client, create=True),
            patch(
                "src.memory.services.factories.get_async_supabase_client",
                _async_client,
            ),
            patch("src.repositories.provenance.apply_provenance_filter", _fake_apply),
        ):
            ab_testing_tasks.check_all_active_experiments.run(include_synthetic=include_synthetic)
        return captured.get("opt_in")

    def test_async_sweep_forwards_opt_in_true(self):
        # RED until the task accepts include_synthetic and forwards it.
        assert self._run_task_and_capture_opt_in(True) is True

    def test_async_sweep_defaults_real_mode(self):
        assert self._run_task_and_capture_opt_in(False) is False

    def test_synthetic_discovered_but_not_interim_enqueued(self):
        """codex R2 boundary: opted-in discovery surfaces a synthetic experiment
        but does NOT enqueue an interim-analysis child for it (that child is
        deliberately real-mode and would no-op). The synthetic row is reported
        under synthetic_skipped instead."""
        from src.tasks import ab_testing_tasks

        rows = [
            {"id": "exp-real", "experiment_name": "Real", "is_synthetic": False},
            {"id": "exp-synth", "experiment_name": "Synth", "is_synthetic": True},
        ]

        def _fake_apply(query, opt_in=False):
            # apply_provenance_filter(query, opt_in).execute() is awaited, so the
            # returned object needs an async execute() yielding a .data result.
            stub = MagicMock()
            stub.execute = AsyncMock(return_value=MagicMock(data=rows))
            return stub

        client = MockSupabaseClient()

        async def _async_client():
            return client

        with (
            patch.object(ab_testing_tasks, "get_async_supabase_client", _async_client, create=True),
            patch(
                "src.memory.services.factories.get_async_supabase_client",
                _async_client,
            ),
            patch("src.repositories.provenance.apply_provenance_filter", _fake_apply),
            patch.object(ab_testing_tasks.scheduled_interim_analysis, "delay") as mock_delay,
        ):
            mock_delay.return_value.id = "queued-task-id"
            result = ab_testing_tasks.check_all_active_experiments.run(include_synthetic=True)

        # Only the real experiment is interim-enqueued.
        enqueued_ids = [c.kwargs.get("experiment_id") for c in mock_delay.call_args_list]
        assert enqueued_ids == ["exp-real"], enqueued_ids
        # The synthetic experiment is honestly reported, not silently dropped.
        skipped_ids = {s["experiment_id"] for s in result["synthetic_skipped"]}
        assert skipped_ids == {"exp-synth"}, result["synthetic_skipped"]
        assert result["experiments_found"] == 2
        assert result["tasks_queued"] == 1
