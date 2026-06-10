"""Offline guard: the corpus indexer never embeds synthetic business_metrics
(Shard 07 R12).

``_fetch_brand_rows`` (and the brand-discovery select in ``index_business_metrics``)
read the live ``business_metrics`` fact table. Shard 02 stamps synthetic rows with
``is_synthetic=true``; the prod RAG corpus must default-exclude them so the chatbot
never surfaces synthetic KPI prose. These tests assert the ``.eq('is_synthetic',
False)`` predicate is appended to the source query, using a fluent fake that records
every ``.eq`` call (no DB, no mock of the unit under test).
"""

from typing import Any

from src.rag import corpus_ingestion as ci


class _RecordingQuery:
    """Fluent supabase-py query stub that records every ``.eq`` call."""

    def __init__(self, calls: list[tuple[Any, ...]]):
        self._calls = calls

    def select(self, *_a: Any, **_k: Any) -> "_RecordingQuery":
        return self

    def eq(self, *a: Any) -> "_RecordingQuery":
        self._calls.append(a)
        return self

    @property
    def not_(self) -> "_RecordingQuery":
        return self

    def is_(self, *_a: Any) -> "_RecordingQuery":
        return self

    def order(self, *_a: Any, **_k: Any) -> "_RecordingQuery":
        return self

    def limit(self, *_a: Any) -> "_RecordingQuery":
        return self

    def range(self, *_a: Any) -> "_RecordingQuery":
        return self

    def execute(self) -> Any:
        class _R:
            data: list[Any] = []

        return _R()


class _RecordingClient:
    def __init__(self, calls: list[tuple[Any, ...]]):
        self._calls = calls

    def table(self, _name: str) -> _RecordingQuery:
        return _RecordingQuery(self._calls)


def test_fetch_brand_rows_excludes_synthetic_business_metrics() -> None:
    calls: list[tuple[Any, ...]] = []
    ci._fetch_brand_rows(_RecordingClient(calls), "Kisqali", 50, latest_per_combo=False)
    assert ("is_synthetic", False) in calls


def test_fetch_brand_rows_excludes_synthetic_latest_per_combo() -> None:
    calls: list[tuple[Any, ...]] = []
    ci._fetch_brand_rows(_RecordingClient(calls), "Kisqali", 50, latest_per_combo=True)
    assert ("is_synthetic", False) in calls


def test_index_business_metrics_brand_discovery_excludes_synthetic() -> None:
    import asyncio

    calls: list[tuple[Any, ...]] = []
    # brands=None triggers the brand-discovery select; empty data -> early return
    # (no _existing_corpus_descriptions / insert path), so this isolates the
    # discovery query predicate.
    asyncio.run(ci.index_business_metrics(supabase_client=_RecordingClient(calls)))
    assert ("is_synthetic", False) in calls


def test_existing_corpus_descriptions_excludes_synthetic() -> None:
    """Shard 07 R15: the dedup read must not let a synthetic episodic
    description suppress ingesting a real business_metrics row."""
    calls: list[tuple[Any, ...]] = []
    ci._existing_corpus_descriptions(_RecordingClient(calls), "corpus_ingestion")
    assert ("is_synthetic", False) in calls
    # the agent_name filter must still be present (real reader, not vacuous).
    assert ("agent_name", "corpus_ingestion") in calls
