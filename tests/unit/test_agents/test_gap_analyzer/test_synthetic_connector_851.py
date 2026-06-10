"""Issue #851 — CI-safe unit tests for the gap_analyzer synthetic-connector fix.

No DB required. Cover the three plumbed gaps and the fail-closed default:
  1. ``include_synthetic`` is threaded factory -> connector -> repository.
  2. The provenance filter default-EXCLUDES synthetic and opts in only on request.
  3. The connector/benchmark store FAIL-CLOSED (re-raise ServiceConnectionError)
     instead of laundering a missing backend into an empty frame.
  4. Region discovery is data-driven (no hardcoded title-case list).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# 1. Provenance filter — default-exclude vs explicit opt-in.
# ---------------------------------------------------------------------------
def test_provenance_filter_default_excludes_synthetic():
    from src.repositories.provenance import apply_provenance_filter

    query = MagicMock()
    apply_provenance_filter(query, include_synthetic=False)
    query.eq.assert_called_once_with("is_synthetic", False)


def test_provenance_filter_opt_in_is_noop():
    from src.repositories.provenance import apply_provenance_filter

    query = MagicMock()
    out = apply_provenance_filter(query, include_synthetic=True)
    query.eq.assert_not_called()
    assert out is query


# ---------------------------------------------------------------------------
# 2. Factory threads include_synthetic into the production connector / store.
# ---------------------------------------------------------------------------
def test_factory_threads_include_synthetic_into_connector():
    from src.agents.gap_analyzer.connectors import get_benchmark_store, get_data_connector

    conn = get_data_connector(include_synthetic=True)
    assert conn.include_synthetic is True

    store = get_benchmark_store(include_synthetic=True)
    assert store.include_synthetic is True


def test_factory_default_is_fail_closed_real_mode():
    from src.agents.gap_analyzer.connectors import get_benchmark_store, get_data_connector

    assert get_data_connector().include_synthetic is False
    assert get_benchmark_store().include_synthetic is False


def test_agent_threads_include_synthetic():
    from src.agents.gap_analyzer.agent import GapAnalyzerAgent

    agent = GapAnalyzerAgent(enable_mlflow=False, enable_opik=False, include_synthetic=True)
    assert agent.include_synthetic is True
    # default real-mode isolation
    assert GapAnalyzerAgent(enable_mlflow=False, enable_opik=False).include_synthetic is False


# ---------------------------------------------------------------------------
# 3. include_synthetic reaches the repository call (connector forwards it).
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_connector_forwards_include_synthetic_to_repo():
    from src.agents.gap_analyzer.connectors.supabase_connector import SupabaseDataConnector

    fake_repo = MagicMock()
    fake_repo.get_time_series = AsyncMock(return_value=[])
    connector = SupabaseDataConnector(supabase_client=MagicMock(), include_synthetic=True)
    connector._repository = fake_repo  # bypass lazy client resolution

    await connector.fetch_performance_data(
        brand="Kisqali",
        metrics=["trx"],
        segments=["region"],
        time_period="2012-01-01_2026-12-31",
    )
    assert fake_repo.get_time_series.await_args.kwargs["include_synthetic"] is True


@pytest.mark.asyncio
async def test_connector_default_forwards_real_mode_to_repo():
    from src.agents.gap_analyzer.connectors.supabase_connector import SupabaseDataConnector

    fake_repo = MagicMock()
    fake_repo.get_time_series = AsyncMock(return_value=[])
    connector = SupabaseDataConnector(supabase_client=MagicMock())  # default
    connector._repository = fake_repo

    await connector.fetch_performance_data(
        brand="Kisqali",
        metrics=["trx"],
        segments=["region"],
        time_period="2012-01-01_2026-12-31",
    )
    assert fake_repo.get_time_series.await_args.kwargs["include_synthetic"] is False


# ---------------------------------------------------------------------------
# 4. FAIL-CLOSED: an unconfigured Supabase must NOT become an empty frame.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_connector_fails_closed_when_client_unresolvable(monkeypatch):
    from src.agents.gap_analyzer.connectors.supabase_connector import SupabaseDataConnector
    from src.memory.services.factories import ServiceConnectionError

    async def _raise():
        raise ServiceConnectionError("Supabase", "SUPABASE_URL not set")

    monkeypatch.setattr("src.memory.services.factories.get_async_supabase_client", _raise)
    connector = SupabaseDataConnector()  # no client injected -> must resolve, then raise
    with pytest.raises(ServiceConnectionError):
        await connector.fetch_performance_data(
            brand="Kisqali",
            metrics=["trx"],
            segments=["region"],
            time_period="2012-01-01_2026-12-31",
        )


@pytest.mark.asyncio
async def test_connector_propagates_unknown_read_error(monkeypatch):
    """An operational read error (e.g. a transport timeout) must PROPAGATE, not be
    laundered into an empty frame (round-3 HIGH: no broad except → empty)."""
    from src.agents.gap_analyzer.connectors.supabase_connector import SupabaseDataConnector

    class _Boom(Exception):
        pass

    fake_repo = MagicMock()
    fake_repo.get_time_series = AsyncMock(side_effect=_Boom("connection reset"))
    connector = SupabaseDataConnector(supabase_client=MagicMock(), include_synthetic=True)
    connector._repository = fake_repo

    with pytest.raises(_Boom):
        await connector.fetch_performance_data(
            brand="Kisqali",
            metrics=["trx"],
            segments=["region"],
            time_period="2012-01-01_2026-12-31",
        )


@pytest.mark.asyncio
async def test_get_by_id_applies_provenance_for_provenance_repo():
    """get_by_id on a HAS_PROVENANCE repo must default-exclude synthetic (round-3 MED)."""
    from src.repositories.business_metric import BusinessMetricRepository

    client = MagicMock()
    result = MagicMock()
    result.data = []
    chain = MagicMock()
    chain.select.return_value = chain
    chain.eq.return_value = chain
    chain.execute = AsyncMock(return_value=result)
    client.table.return_value = chain

    repo = BusinessMetricRepository(client)
    await repo.get_by_id("some-id")  # default real-mode
    chain.eq.assert_any_call("is_synthetic", False)


@pytest.mark.asyncio
async def test_benchmark_store_fails_closed_when_client_unresolvable(monkeypatch):
    from src.agents.gap_analyzer.connectors.benchmark_store import BenchmarkStore
    from src.memory.services.factories import ServiceConnectionError

    async def _raise():
        raise ServiceConnectionError("Supabase", "SUPABASE_URL not set")

    monkeypatch.setattr("src.memory.services.factories.get_async_supabase_client", _raise)
    store = BenchmarkStore()
    with pytest.raises(ServiceConnectionError):
        await store.get_targets(brand="Kisqali", metrics=["trx"], segments=["region"])


def test_repository_property_fails_closed_without_client():
    """The sync `repository` property must NOT build a client-less no-op repo (the
    #845 fail-OPEN escape codex flagged). It raises instead."""
    from src.agents.gap_analyzer.connectors.benchmark_store import BenchmarkStore
    from src.agents.gap_analyzer.connectors.supabase_connector import SupabaseDataConnector
    from src.memory.services.factories import ServiceConnectionError

    with pytest.raises(ServiceConnectionError):
        _ = SupabaseDataConnector().repository
    with pytest.raises(ServiceConnectionError):
        _ = BenchmarkStore().repository


def test_repository_property_serves_injected_client():
    """With an injected client the property builds a real repo (backward compat)."""
    from src.agents.gap_analyzer.connectors.supabase_connector import SupabaseDataConnector

    conn = SupabaseDataConnector(supabase_client=MagicMock())
    repo = conn.repository
    assert repo is not None
    assert repo.client is not None


# ---------------------------------------------------------------------------
# 5. Benchmark store returns the per-region WIDE shape _calculate_gap consumes.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_benchmark_peer_returns_per_segment_wide_frame():
    from src.agents.gap_analyzer.connectors.benchmark_store import BenchmarkStore

    store = BenchmarkStore(supabase_client=MagicMock())

    fake_repo = MagicMock()
    fake_repo.get_distinct_values = AsyncMock(return_value=["northeast", "south", "west"])

    def _records(region, brand, limit, include_synthetic):
        base = {"northeast": 100.0, "south": 200.0, "west": 300.0}[region]
        return [{"metric_name": "trx", "value": base}]

    fake_repo.get_by_region = AsyncMock(side_effect=_records)
    store._repository = fake_repo

    peers = await store.get_peer_benchmarks(brand="Kisqali", metrics=["trx"], segments=["region"])
    # Per-segment wide: a `region` column + the metric column, one row per value.
    assert "region" in peers.columns
    assert "trx" in peers.columns
    assert set(peers["region"]) == {"northeast", "south", "west"}
    # The cross-segment P75 bar is broadcast identically to every segment value.
    assert peers["trx"].nunique() == 1


@pytest.mark.asyncio
async def test_benchmark_unsupported_segment_returns_empty_no_fabrication():
    """specialty/hcp_tier are NOT columns on business_metrics → empty frame, NOT
    fabricated benchmarks. _calculate_gap then yields no gaps for that segment."""
    from src.agents.gap_analyzer.connectors.benchmark_store import BenchmarkStore

    store = BenchmarkStore(supabase_client=MagicMock())
    fake_repo = MagicMock()
    fake_repo.get_distinct_values = AsyncMock(return_value=[])  # unsupported segment
    store._repository = fake_repo

    peers = await store.get_peer_benchmarks(
        brand="Kisqali", metrics=["trx"], segments=["specialty"]
    )
    assert peers.empty


def test_broadcast_cross_segment_stat_shape():
    from src.agents.gap_analyzer.connectors.benchmark_store import BenchmarkStore

    frame = pd.DataFrame({"region": ["northeast", "south", "west"], "trx": [100.0, 200.0, 300.0]})
    out = BenchmarkStore._broadcast_cross_segment_stat(frame, ["trx"], quantile=0.90)
    assert list(out["region"]) == ["northeast", "south", "west"]
    assert out["trx"].nunique() == 1  # single broadcast bar


# ---------------------------------------------------------------------------
# 6. Segment-value discovery is data-driven (no hardcoded title-case list).
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_get_distinct_values_returns_data_values():
    from src.repositories.business_metric import BusinessMetricRepository

    client = MagicMock()
    result = MagicMock()
    result.data = [
        {"region": "northeast"},
        {"region": "south"},
        {"region": "northeast"},
        {"region": None},
    ]
    chain = MagicMock()
    chain.select.return_value = chain
    chain.eq.return_value = chain
    chain.limit.return_value = chain
    chain.execute = AsyncMock(return_value=result)
    client.table.return_value = chain

    repo = BusinessMetricRepository(client)
    regions = await repo.get_distinct_values("region", brand="Kisqali", include_synthetic=True)
    assert regions == ["northeast", "south"]  # sorted, de-duped, None dropped


@pytest.mark.asyncio
async def test_get_distinct_values_swallows_only_undefined_column():
    """A 42703 (undefined column, e.g. specialty) fails soft to []; any OTHER
    operational error MUST surface (not laundered into 'no data')."""
    from postgrest.exceptions import APIError

    from src.repositories.business_metric import BusinessMetricRepository

    # 42703 -> []
    client = MagicMock()
    chain = MagicMock()
    chain.select.return_value = chain
    chain.eq.return_value = chain
    chain.limit.return_value = chain
    chain.execute = AsyncMock(side_effect=APIError({"code": "42703", "message": "no column"}))
    client.table.return_value = chain
    repo = BusinessMetricRepository(client)
    assert await repo.get_distinct_values("specialty", include_synthetic=True) == []

    # Any other operational error -> raise
    chain.execute = AsyncMock(
        side_effect=APIError({"code": "08006", "message": "connection failure"})
    )
    with pytest.raises(APIError):
        await repo.get_distinct_values("region", include_synthetic=True)


@pytest.mark.asyncio
async def test_detect_segment_gaps_no_crash_on_missing_segment_column():
    """_detect_segment_gaps must return no gaps (not KeyError-crash) when the
    current-data frame lacks the requested segment column (#851 MED)."""
    from src.agents.gap_analyzer.nodes.gap_detector import GapDetectorNode

    node = GapDetectorNode(use_mock=True)
    current = pd.DataFrame({"region": ["northeast"], "trx": [100.0]})
    comparison = {"vs_target": pd.DataFrame({"region": ["northeast"], "trx": [120.0]})}

    seg, gaps = await node._detect_segment_gaps(
        current_data=current,
        comparison_data=comparison,
        segment="specialty",  # absent from current_data
        metrics=["trx"],
        gap_type="vs_target",
        min_gap_threshold=5.0,
    )
    assert seg == "specialty"
    assert gaps == []


# ---------------------------------------------------------------------------
# 7. Agent must SURFACE a failed graph, not launder it into empty success.
# ---------------------------------------------------------------------------
def test_build_output_surfaces_failed_status_and_errors():
    """A gap_detector failure must propagate status='failed' + errors to the output
    (the HIGH codex flagged: a missing backend looked like empty success)."""
    from src.agents.gap_analyzer.agent import GapAnalyzerAgent

    agent = GapAnalyzerAgent(enable_mlflow=False, enable_opik=False)
    failed_state = {
        "status": "failed",
        "errors": [{"node": "gap_detector", "error": "[Supabase] not configured"}],
        "segments": ["region"],
    }
    out = agent._build_output(failed_state)  # type: ignore[arg-type]
    assert out["status"] == "failed"
    assert out["errors"] and out["errors"][0]["node"] == "gap_detector"
    # Empty opportunities are present but the status makes the failure unambiguous.
    assert out["prioritized_opportunities"] == []


def test_build_output_marks_completed_on_clean_state():
    from src.agents.gap_analyzer.agent import GapAnalyzerAgent

    agent = GapAnalyzerAgent(enable_mlflow=False, enable_opik=False)
    clean_state = {"status": "completed", "errors": [], "segments": ["region"]}
    out = agent._build_output(clean_state)  # type: ignore[arg-type]
    assert out["status"] == "completed"
    assert out["errors"] == []
