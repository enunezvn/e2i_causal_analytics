"""Faithful (real-DB, NO mocks) regression tests for the experiment-monitor
await-sync-client bug (H3b).

WHY: the pre-existing tests/unit/.../test_health_checker_node.py::test_get_client_lazy_loads
mocked `get_supabase_client` as an AsyncMock, making `await get_supabase_client()`
succeed in the test and HIDING the real bug -- the factory is SYNC, so awaiting it
raises `TypeError: object Client can't be used in 'await' expression`. execute()'s
except swallowed it -> empty experiments masking a crash, while 621 real running
ml_experiments existed.

Opt-in (real docker supabase-db required), skipped in CI by default:
    E2I_DB_INTEGRATION=1 .venv/bin/pytest tests/integration/test_experiment_monitor_async_realdb.py -p no:cacheprovider
"""

import os

import pytest

from src.agents.experiment_monitor.nodes.fidelity_checker import FidelityCheckerNode
from src.agents.experiment_monitor.nodes.health_checker import HealthCheckerNode
from src.agents.experiment_monitor.nodes.interim_analyzer import InterimAnalyzerNode
from src.agents.experiment_monitor.nodes.srm_detector import SRMDetectorNode

pytestmark = pytest.mark.skipif(
    os.getenv("E2I_DB_INTEGRATION") != "1",
    reason="real-DB integration; set E2I_DB_INTEGRATION=1 with docker supabase-db reachable",
)

ALL_DB_NODES = [HealthCheckerNode, FidelityCheckerNode, InterimAnalyzerNode, SRMDetectorNode]


@pytest.fixture(autouse=True)
def _fresh_async_supabase_client():
    """Reset the cached async client so each test builds a fresh one on its OWN
    event loop. The global cache binds the httpx.AsyncClient to the loop that
    created it; pytest-asyncio's per-test loops would otherwise reuse a client
    from a closed loop. (Prod has a single long-lived loop, so this is a test-only
    isolation concern.)"""
    import src.memory.services.factories as factories

    factories._async_supabase_client = None
    yield
    factories._async_supabase_client = None


@pytest.mark.parametrize("node_cls", ALL_DB_NODES)
@pytest.mark.asyncio
async def test_monitor_node_get_client_returns_awaitable_async_client(node_cls):
    """Every DB-backed monitor node must lazily load an ASYNC Supabase client
    whose query .execute() is awaitable. The buggy code awaited the SYNC
    get_supabase_client() and raised TypeError."""
    node = node_cls()

    client = await node._get_client()
    assert client is not None, f"{node_cls.__name__}._get_client() returned None"

    # Decisive faithful check: a real async client lets us AWAIT a query.
    result = await client.table("ml_experiments").select("id").limit(1).execute()
    assert hasattr(result, "data"), f"{node_cls.__name__} client query returned no response"


@pytest.mark.asyncio
async def test_health_checker_surfaces_real_running_experiments_not_empty():
    """check_all_active must return the REAL running experiments, not an empty list
    masking a crash. The buggy code returned experiments=[] + a swallowed TypeError."""
    node = HealthCheckerNode()
    state = {
        "check_all_active": True,
        "experiment_ids": None,
        "warnings": [],
        "errors": [],
    }

    result = await node.execute(state)  # type: ignore[arg-type]

    errors = result.get("errors", []) or []
    assert not any("await" in str(e.get("error", "")).lower() for e in errors), (
        f"await-on-sync error leaked into errors: {errors}"
    )
    assert result.get("experiments_checked", 0) > 0, (
        f"expected real running experiments, got "
        f"experiments_checked={result.get('experiments_checked')} errors={errors}"
    )


@pytest.mark.asyncio
async def test_monitor_node_does_not_fabricate_mock_experiments_in_prod():
    """No-mocking guard: the health checker must never surface the hardcoded
    mock experiment IDs (exp-001/exp-002) in a production (client-available) run."""
    node = HealthCheckerNode()
    state = {"check_all_active": True, "experiment_ids": None, "warnings": [], "errors": []}

    result = await node.execute(state)  # type: ignore[arg-type]

    ids = {str(e.get("experiment_id", "")) for e in result.get("experiments", [])}
    assert not ({"exp-001", "exp-002"} & ids), f"mock experiments leaked into prod: {ids}"
    warnings = result.get("warnings", []) or []
    assert not any("mock data" in str(w).lower() for w in warnings), (
        f"mock-data warning in prod: {warnings}"
    )
