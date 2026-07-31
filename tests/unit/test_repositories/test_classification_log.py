"""Tests for ClassificationLogRepository."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.orchestrator.classifier.schemas import (
    ClassificationResult,
    ClassificationStages,
    Dependency,
    DependencyAnalysis,
    DependencyType,
    Domain,
    DomainMapping,
    DomainMatch,
    EntityFeatures,
    ExtractedFeatures,
    IntentSignals,
    RoutingPattern,
    StructuralFeatures,
    SubQuestion,
    TemporalFeatures,
)
from src.repositories.classification_log import (
    ClassificationLogRepository,
    get_classification_log_repository,
)


def _result(with_stages: bool = True) -> ClassificationResult:
    sub = SubQuestion(id="Q1", text="part", domains=[], primary_domain=Domain.CAUSAL_ANALYSIS)
    dep = Dependency(
        **{"from": "Q1", "to": "Q2"},
        dependency_type=DependencyType.CONDITIONAL,
        reason="test",
    )
    stages = None
    if with_stages:
        stages = ClassificationStages(
            features=ExtractedFeatures(
                structural=StructuralFeatures(word_count=5),
                temporal=TemporalFeatures(),
                entities=EntityFeatures(),
                intent_signals=IntentSignals(causal_keywords=["impact"]),
                raw_query="q",
            ),
            domain_mapping=DomainMapping(
                domains_detected=[
                    DomainMatch(domain=Domain.CAUSAL_ANALYSIS, confidence=0.7, evidence=["impact"])
                ],
                domain_count=1,
                primary_domain=Domain.CAUSAL_ANALYSIS,
            ),
            dependency_analysis=DependencyAnalysis(sub_questions=[sub], dependencies=[dep]),
        )
    return ClassificationResult(
        routing_pattern=RoutingPattern.SINGLE_AGENT,
        target_agents=["causal_impact"],
        sub_questions=[sub],
        dependencies=[dep],
        confidence=0.7,
        reasoning="test",
        is_followup=True,
        classification_latency_ms=2.5,
        used_llm_layer=False,
        stages=stages,
    )


def _mock_client(insert_raises: bool = False):
    client = MagicMock()
    execute = AsyncMock()
    if insert_raises:
        execute.side_effect = RuntimeError("db down")
    else:
        execute.return_value = MagicMock(data=[{"classification_id": "abc"}])
    client.table.return_value.insert.return_value.execute = execute
    return client


class TestClassificationLogRepository:
    async def test_row_shape_matches_ddl(self):
        client = _mock_client()
        repo = ClassificationLogRepository(client)
        row = await repo.record_classification(
            query_text="What is the impact of rep visits?",
            result=_result(),
            session_id="user-1~sess-1",
            user_id="user-1",
        )
        assert row == {"classification_id": "abc"}
        inserted = client.table.return_value.insert.call_args[0][0]
        # Columns from database/ml/013_tool_composer_tables.sql
        assert inserted["routing_pattern"] == "SINGLE_AGENT"
        assert inserted["target_agents"] == ["causal_impact"]
        assert inserted["confidence"] == 0.7
        assert inserted["used_llm_layer"] is False
        assert inserted["classification_latency_ms"] == 2.5
        assert inserted["is_followup"] is True
        assert inserted["session_id"] == "user-1~sess-1"
        assert inserted["user_id"] == "user-1"
        assert len(inserted["query_hash"]) == 64
        # JSONB stage columns populated from result.stages
        assert inserted["features_extracted"]["structural"]["word_count"] == 5
        assert inserted["domain_mapping"]["domain_count"] == 1
        assert inserted["dependency_analysis"]["sub_questions"]
        # Dependency dumps use field names (from_id/to_id), not aliases
        assert inserted["dependencies"][0]["from_id"] == "Q1"
        assert inserted["dependencies"][0]["to_id"] == "Q2"

    async def test_missing_stages_defaults_empty(self):
        client = _mock_client()
        repo = ClassificationLogRepository(client)
        await repo.record_classification(query_text="q", result=_result(with_stages=False))
        inserted = client.table.return_value.insert.call_args[0][0]
        assert inserted["features_extracted"] == {}
        assert inserted["domain_mapping"] == {}
        assert inserted["dependency_analysis"] == {}

    async def test_session_id_truncated_to_column_limit(self):
        client = _mock_client()
        repo = ClassificationLogRepository(client)
        await repo.record_classification(
            query_text="q", result=_result(), session_id="s" * 150, user_id="u" * 150
        )
        inserted = client.table.return_value.insert.call_args[0][0]
        assert len(inserted["session_id"]) == 100
        assert len(inserted["user_id"]) == 100

    async def test_insert_failure_swallowed(self):
        repo = ClassificationLogRepository(_mock_client(insert_raises=True))
        row = await repo.record_classification(query_text="q", result=_result())
        assert row is None

    async def test_no_client_returns_none(self):
        repo = get_classification_log_repository(None)
        row = await repo.record_classification(query_text="q", result=_result())
        assert row is None


@pytest.mark.parametrize("field", ["session_id", "user_id"])
async def test_optional_context_omitted_when_none(field):
    client = _mock_client()
    repo = ClassificationLogRepository(client)
    await repo.record_classification(query_text="q", result=_result())
    inserted = client.table.return_value.insert.call_args[0][0]
    assert field not in inserted


def _mock_select_client(rows=None, raises: bool = False):
    """Mock supporting the chained select().is_().gte().order().limit().execute()."""
    client = MagicMock()
    execute = AsyncMock()
    if raises:
        execute.side_effect = RuntimeError("db down")
    else:
        execute.return_value = MagicMock(data=rows or [])
    query = client.table.return_value.select.return_value
    query.is_.return_value.gte.return_value.order.return_value.limit.return_value.execute = execute
    return client


def _mock_update_client(rows=None, raises: bool = False):
    client = MagicMock()
    execute = AsyncMock()
    if raises:
        execute.side_effect = RuntimeError("db down")
    else:
        execute.return_value = MagicMock(data=rows if rows is not None else [{"was_correct": True}])
    client.table.return_value.update.return_value.eq.return_value.execute = execute
    return client


class TestFetchUnlabeled:
    async def test_filters_null_was_correct_within_lookback(self):
        rows = [{"classification_id": "abc", "was_correct": None}]
        client = _mock_select_client(rows)
        repo = ClassificationLogRepository(client)
        result = await repo.fetch_unlabeled(lookback_days=7, limit=100)
        assert result == rows
        query = client.table.return_value.select.return_value
        assert query.is_.call_args[0] == ("was_correct", "null")
        query.is_.return_value.gte.return_value.order.return_value.limit.assert_called_once_with(
            100
        )

    async def test_no_client_returns_empty(self):
        repo = ClassificationLogRepository(None)
        assert await repo.fetch_unlabeled() == []

    async def test_failure_fails_open(self):
        repo = ClassificationLogRepository(_mock_select_client(raises=True))
        assert await repo.fetch_unlabeled() == []


def _mock_metrics_select_client(rows=None, raises: bool = False):
    """Mock for the chained select().gte().order().limit().execute() (no is_)."""
    client = MagicMock()
    execute = AsyncMock()
    if raises:
        execute.side_effect = RuntimeError("db down")
    else:
        execute.return_value = MagicMock(data=rows or [])
    query = client.table.return_value.select.return_value
    query.gte.return_value.order.return_value.limit.return_value.execute = execute
    return client


class TestFetchForMetrics:
    async def test_returns_labeled_and_unlabeled_rows(self):
        rows = [
            {"routing_pattern": "SINGLE_AGENT", "was_correct": True},
            {"routing_pattern": "CLARIFICATION_NEEDED", "was_correct": None},
        ]
        client = _mock_metrics_select_client(rows)
        repo = ClassificationLogRepository(client)
        assert await repo.fetch_for_metrics(lookback_days=30, limit=2000) == rows
        # No was_correct filter (unlike fetch_unlabeled) — labeled rows included.
        query = client.table.return_value.select.return_value
        query.gte.return_value.order.return_value.limit.assert_called_once_with(2000)

    async def test_no_client_returns_empty(self):
        assert await ClassificationLogRepository(None).fetch_for_metrics() == []

    async def test_failure_fails_open(self):
        repo = ClassificationLogRepository(_mock_metrics_select_client(raises=True))
        assert await repo.fetch_for_metrics() == []


class TestRecordMetricsSnapshot:
    _METRICS = {
        "total": 12,
        "labeled": 8,
        "overall_accuracy_pct": 75.0,
        "engagement_rate": 0.5,
        "active_floor": 0.5,
        "llm_layer_share": 0.25,
        "abstention": {"total": 3, "judged_correct": 1, "judged_incorrect": 2},
        "per_pattern": {"SINGLE_AGENT": {"total": 6}},
        "label_sources": {"llm_judge": 5},
    }

    async def test_insert_flattens_abstention_and_keeps_window(self):
        client = _mock_client()  # insert().execute() -> truthy data
        repo = ClassificationLogRepository(client)
        ok = await repo.record_metrics_snapshot(self._METRICS, task_id="t-1", window_days=30)
        assert ok is True
        client.table.assert_called_with("routing_classifier_metrics")
        data = client.table.return_value.insert.call_args[0][0]
        assert data["window_days"] == 30
        assert data["abstention_total"] == 3
        assert data["abstention_correct"] == 1
        assert data["abstention_incorrect"] == 2
        assert data["overall_accuracy_pct"] == 75.0
        assert data["per_pattern"] == {"SINGLE_AGENT": {"total": 6}}

    async def test_missing_table_fails_open(self):
        # migration 032 not applied yet -> insert raises -> False, never raises.
        repo = ClassificationLogRepository(_mock_client(insert_raises=True))
        assert (
            await repo.record_metrics_snapshot(self._METRICS, task_id="t", window_days=30) is False
        )

    async def test_no_client_returns_false(self):
        repo = ClassificationLogRepository(None)
        assert await repo.record_metrics_snapshot({}, task_id="t", window_days=30) is False


class TestApplyLabel:
    async def test_sends_label_update_keyed_on_classification_id(self):
        client = _mock_update_client()
        repo = ClassificationLogRepository(client)
        ok = await repo.apply_label(
            "abc", was_correct=False, correct_pattern="TOOL_COMPOSER", feedback_notes="{}"
        )
        assert ok is True
        update = client.table.return_value.update.call_args[0][0]
        assert update == {
            "was_correct": False,
            "correct_pattern": "TOOL_COMPOSER",
            "feedback_notes": "{}",
        }
        eq_args = client.table.return_value.update.return_value.eq.call_args[0]
        assert eq_args == ("classification_id", "abc")

    async def test_abstention_writes_notes_only(self):
        client = _mock_update_client()
        repo = ClassificationLogRepository(client)
        ok = await repo.apply_label("abc", feedback_notes='{"source": "llm_judge_abstain"}')
        assert ok is True
        update = client.table.return_value.update.call_args[0][0]
        assert "was_correct" not in update

    async def test_empty_update_is_noop(self):
        client = _mock_update_client()
        repo = ClassificationLogRepository(client)
        assert await repo.apply_label("abc") is False
        client.table.return_value.update.assert_not_called()

    async def test_no_client_returns_false(self):
        repo = ClassificationLogRepository(None)
        assert await repo.apply_label("abc", was_correct=True) is False

    async def test_failure_fails_open(self):
        repo = ClassificationLogRepository(_mock_update_client(raises=True))
        assert await repo.apply_label("abc", was_correct=True) is False
