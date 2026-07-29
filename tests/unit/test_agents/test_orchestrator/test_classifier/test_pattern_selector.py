"""Tests for Stage 4: PatternSelector."""

import time

from src.agents.orchestrator.classifier.pattern_selector import PatternSelector
from src.agents.orchestrator.classifier.schemas import (
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


def _features(word_count: int = 8) -> ExtractedFeatures:
    return ExtractedFeatures(
        structural=StructuralFeatures(word_count=word_count),
        temporal=TemporalFeatures(),
        entities=EntityFeatures(),
        intent_signals=IntentSignals(),
        raw_query="test query",
    )


def _mapping(*matches: tuple[Domain, float]) -> DomainMapping:
    detected = [DomainMatch(domain=d, confidence=c, evidence=["kw"]) for d, c in matches]
    return DomainMapping(
        domains_detected=detected,
        domain_count=len(detected),
        primary_domain=detected[0].domain if detected else None,
        is_multi_domain=len(detected) > 1,
    )


def _analysis(
    n_subq: int = 1, dependencies: list[Dependency] | None = None, depth: int = 0
) -> DependencyAnalysis:
    subs = [
        SubQuestion(
            id=f"Q{i + 1}", text=f"part {i + 1}", domains=[], primary_domain=Domain.EXPLANATION
        )
        for i in range(n_subq)
    ]
    deps = dependencies or []
    return DependencyAnalysis(
        sub_questions=subs,
        dependencies=deps,
        has_dependencies=bool(deps),
        is_parallelizable=not deps,
        dependency_depth=depth,
    )


def _dep(frm: str = "Q1", to: str = "Q2") -> Dependency:
    return Dependency(
        **{"from": frm, "to": to},
        dependency_type=DependencyType.REFERENCE_CHAIN,
        reason="test",
    )


class TestPatternSelector:
    def setup_method(self):
        self.selector = PatternSelector()

    def test_no_domains_clarification(self):
        result = self.selector.select(_features(), _mapping(), _analysis())
        assert result.routing_pattern == RoutingPattern.CLARIFICATION_NEEDED
        assert result.target_agents == []
        assert result.confidence == 0.0

    def test_low_confidence_clarification(self):
        result = self.selector.select(
            _features(), _mapping((Domain.CAUSAL_ANALYSIS, 0.4)), _analysis()
        )
        assert result.routing_pattern == RoutingPattern.CLARIFICATION_NEEDED

    def test_explanation_override_with_consultation_hints(self):
        result = self.selector.select(
            _features(),
            _mapping((Domain.EXPLANATION, 0.8), (Domain.CAUSAL_ANALYSIS, 0.5)),
            _analysis(),
        )
        assert result.routing_pattern == RoutingPattern.SINGLE_AGENT
        assert result.target_agents == ["explainer"]
        assert "causal_impact" in result.consultation_hints

    def test_single_domain_single_agent(self):
        result = self.selector.select(
            _features(), _mapping((Domain.CAUSAL_ANALYSIS, 0.7)), _analysis()
        )
        assert result.routing_pattern == RoutingPattern.SINGLE_AGENT
        assert result.target_agents == ["causal_impact"]
        assert result.confidence == 0.7

    def test_multi_domain_parallel_delegation(self):
        result = self.selector.select(
            _features(),
            _mapping((Domain.CAUSAL_ANALYSIS, 0.7), (Domain.GAP_ANALYSIS, 0.6)),
            _analysis(n_subq=2),
        )
        assert result.routing_pattern == RoutingPattern.PARALLEL_DELEGATION
        assert result.target_agents == ["causal_impact", "gap_analyzer"]

    def test_multi_domain_with_dependencies_tool_composer(self):
        result = self.selector.select(
            _features(),
            _mapping((Domain.CAUSAL_ANALYSIS, 0.7), (Domain.PREDICTION, 0.6)),
            _analysis(n_subq=2, dependencies=[_dep()], depth=1),
        )
        assert result.routing_pattern == RoutingPattern.TOOL_COMPOSER
        assert result.target_agents == ["tool_composer"]
        assert len(result.dependencies) == 1

    def test_deep_dependency_chain_complexity_warning(self):
        deps = [_dep("Q1", "Q2"), _dep("Q2", "Q3"), _dep("Q3", "Q4"), _dep("Q4", "Q5")]
        result = self.selector.select(
            _features(),
            _mapping((Domain.CAUSAL_ANALYSIS, 0.7), (Domain.PREDICTION, 0.6)),
            _analysis(n_subq=5, dependencies=deps, depth=4),
        )
        assert result.complexity_warning is not None

    def test_latency_and_llm_passthrough(self):
        start = time.time() - 0.005
        result = self.selector.select(
            _features(),
            _mapping((Domain.CAUSAL_ANALYSIS, 0.7)),
            _analysis(),
            classification_start_time=start,
            used_llm=True,
        )
        assert result.classification_latency_ms >= 5.0
        assert result.used_llm_layer is True

    def test_clarification_keeps_followup_metadata(self):
        """Regression: the CLARIFICATION branch dropped is_followup/
        context_source — the rows most valuable to the agreement analysis."""
        result = self.selector.select(
            _features(),
            _mapping(),
            _analysis(),
            is_followup=True,
            context_source="conversation_history",
        )
        assert result.routing_pattern == RoutingPattern.CLARIFICATION_NEEDED
        assert result.is_followup is True
        assert result.context_source == "conversation_history"

    def test_followup_metadata_passthrough(self):
        result = self.selector.select(
            _features(),
            _mapping((Domain.CAUSAL_ANALYSIS, 0.7)),
            _analysis(),
            is_followup=True,
            context_source="conversation_history",
        )
        assert result.is_followup is True
        assert result.context_source == "conversation_history"
