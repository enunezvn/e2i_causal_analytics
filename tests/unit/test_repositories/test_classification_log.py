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
