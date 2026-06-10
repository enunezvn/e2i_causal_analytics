"""Unit tests (no DB) for issue #825 lower-priority item: the experiment-monitor
selective-check flags (check_srm / check_enrollment / check_fidelity) defined on
the FE request + BE request model but never threaded into the agent. These prove
the flags reach the agent input/state and that a disabled check short-circuits
the node BEFORE it acquires a DB client (decisive: ``_client is None``)."""


def test_experiment_monitor_input_has_selective_check_flags():
    from src.agents.experiment_monitor.agent import ExperimentMonitorInput

    default = ExperimentMonitorInput()
    assert default.check_srm is True
    assert default.check_enrollment is True
    assert default.check_fidelity is True

    off = ExperimentMonitorInput(check_srm=False, check_enrollment=False, check_fidelity=False)
    assert (off.check_srm, off.check_enrollment, off.check_fidelity) == (False, False, False)


def test_state_typeddict_carries_selective_check_flags():
    from src.agents.experiment_monitor.state import ExperimentMonitorState

    keys = set(ExperimentMonitorState.__annotations__)
    assert {"check_srm", "check_enrollment", "check_fidelity"} <= keys


class _DBTouched(BaseException):
    """Raised if a gated node reaches the DB factory. A BaseException subclass so
    the node's ``except Exception`` cannot mask it (decisive gate proof)."""


async def _factory_must_not_be_called(*args, **kwargs):
    raise _DBTouched("a disabled check must not acquire a DB client")


async def test_srm_detector_skips_without_acquiring_client_when_disabled(monkeypatch):
    """Decisive: patch the async-client factory to raise; the gated node must
    short-circuit before reaching it (without the gate it acquires a client)."""
    from src.agents.experiment_monitor.nodes.srm_detector import SRMDetectorNode

    monkeypatch.setattr(
        "src.memory.services.factories.get_async_supabase_client",
        _factory_must_not_be_called,
    )
    node = SRMDetectorNode()
    state = {
        "check_srm": False,
        "experiments": [{"experiment_id": "00000000-0000-0000-0000-000000000000"}],
        "srm_threshold": 0.001,
        "errors": [],
    }

    result = await node.execute(state)  # type: ignore[arg-type]

    assert result["srm_issues"] == []


async def test_fidelity_checker_skips_without_acquiring_client_when_disabled(monkeypatch):
    """Decisive: patch the async-client factory to raise; the gated node must
    short-circuit before reaching it."""
    from src.agents.experiment_monitor.nodes.fidelity_checker import FidelityCheckerNode

    monkeypatch.setattr(
        "src.memory.services.factories.get_async_supabase_client",
        _factory_must_not_be_called,
    )
    node = FidelityCheckerNode()
    state = {
        "check_fidelity": False,
        "experiments": [{"experiment_id": "00000000-0000-0000-0000-000000000000"}],
        "fidelity_threshold": 0.2,
        "errors": [],
    }

    result = await node.execute(state)  # type: ignore[arg-type]

    assert result["fidelity_issues"] == []


def test_health_checker_enrollment_gate_is_wired():
    """The enrollment-rate check inside health_checker.execute is gated on
    check_enrollment (read from state) so the FE toggle actually suppresses it."""
    import inspect

    from src.agents.experiment_monitor.nodes.health_checker import HealthCheckerNode

    source = inspect.getsource(HealthCheckerNode.execute)
    assert 'state.get("check_enrollment"' in source, "enrollment check is not gated on the flag"
