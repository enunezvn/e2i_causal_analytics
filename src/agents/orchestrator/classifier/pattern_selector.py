# src/e2i/agents/orchestrator/classifier/pattern_selector.py
"""
Stage 4: Select routing pattern based on analysis.

This module takes the outputs from Stages 1-3 and determines the
appropriate routing pattern:
- SINGLE_AGENT: Route to one primary agent
- PARALLEL_DELEGATION: Route to multiple independent agents
- TOOL_COMPOSER: Use Tool Composer for dependent multi-domain queries
- CLARIFICATION_NEEDED: Query too ambiguous
"""

import time
from typing import Optional

from .schemas import (
    ClassificationResult,
    DependencyAnalysis,
    Domain,
    DomainMapping,
    ExtractedFeatures,
    RoutingPattern,
)

# Agent mapping by domain
DOMAIN_TO_AGENT = {
    Domain.CAUSAL_ANALYSIS: "causal_impact",
    Domain.HETEROGENEITY: "heterogeneous_optimizer",
    Domain.GAP_ANALYSIS: "gap_analyzer",
    Domain.EXPERIMENTATION: "experiment_designer",
    Domain.PREDICTION: "prediction_synthesizer",
    Domain.MONITORING: "drift_monitor",
    Domain.EXPLANATION: "explainer",
    Domain.COHORT_DEFINITION: "cohort_constructor",
}


class PatternSelector:
    """
    Selects the appropriate routing pattern based on classification stages.
    """

    # Thresholds
    MIN_CONFIDENCE = 0.5
    MAX_COMPLEXITY_DOMAINS = 5
    MAX_DEPENDENCY_DEPTH = 3

    def select(
        self,
        features: ExtractedFeatures,
        domain_mapping: DomainMapping,
        dependency_analysis: DependencyAnalysis,
        is_followup: bool = False,
        context_source: Optional[str] = None,
        classification_start_time: Optional[float] = None,
        used_llm: bool = False,
    ) -> ClassificationResult:
        """
        Select routing pattern and build final classification result.

        Args:
            features: Extracted features from Stage 1
            domain_mapping: Domain mapping from Stage 2
            dependency_analysis: Dependency analysis from Stage 3
            is_followup: Whether this is a follow-up query
            context_source: Source of context if follow-up
            classification_start_time: Start time for latency tracking
            used_llm: Whether LLM layer was used

        Returns:
            ClassificationResult with routing decision
        """
        # Calculate latency
        latency_ms = 0.0
        if classification_start_time:
            latency_ms = (time.time() - classification_start_time) * 1000

        # =====================================================================
        # Rule 1: Check for low confidence (needs clarification)
        # =====================================================================
        if (
            not domain_mapping.domains_detected
            or domain_mapping.domains_detected[0].confidence < self.MIN_CONFIDENCE
        ):
            return ClassificationResult(
                routing_pattern=RoutingPattern.CLARIFICATION_NEEDED,
                target_agents=[],
                confidence=0.0,
                reasoning="Unable to determine query intent with sufficient confidence",
                classification_latency_ms=latency_ms,
                used_llm_layer=used_llm,
            )

        # =====================================================================
        # Rule 2: Explanation override
        # =====================================================================
        if (
            domain_mapping.primary_domain == Domain.EXPLANATION
            and domain_mapping.domains_detected[0].confidence > 0.7
        ):
            consultation_hints = []
            # Check if other domains present that might need consultation
            for dm in domain_mapping.domains_detected[1:]:
                if dm.confidence > 0.4:
                    consultation_hints.append(DOMAIN_TO_AGENT[dm.domain])

            return ClassificationResult(
                routing_pattern=RoutingPattern.SINGLE_AGENT,
                target_agents=["explainer"],
                sub_questions=dependency_analysis.sub_questions,
                confidence=domain_mapping.domains_detected[0].confidence,
                reasoning="Primary intent is explanation",
                consultation_hints=consultation_hints,
                is_followup=is_followup,
                context_source=context_source,
                classification_latency_ms=latency_ms,
                used_llm_layer=used_llm,
            )

        # =====================================================================
        # Rule 3: Single domain → Single agent
        # =====================================================================
        if not domain_mapping.is_multi_domain:
            primary_domain = domain_mapping.primary_domain
            target_agent = DOMAIN_TO_AGENT.get(primary_domain, "explainer")

            return ClassificationResult(
                routing_pattern=RoutingPattern.SINGLE_AGENT,
                target_agents=[target_agent],
                sub_questions=dependency_analysis.sub_questions,
                confidence=domain_mapping.domains_detected[0].confidence,
                reasoning=f"Single domain query: {primary_domain.value}",
                is_followup=is_followup,
                context_source=context_source,
                classification_latency_ms=latency_ms,
                used_llm_layer=used_llm,
            )

        # =====================================================================
        # Rule 4: Multi-domain, check complexity limits
        # =====================================================================
        complexity_warning = None
        if domain_mapping.domain_count > self.MAX_COMPLEXITY_DOMAINS:
            complexity_warning = (
                f"Query spans {domain_mapping.domain_count} domains. "
                "Consider breaking into multiple queries for better results."
            )
        if dependency_analysis.dependency_depth > self.MAX_DEPENDENCY_DEPTH:
            complexity_warning = (
                f"Query has {dependency_analysis.dependency_depth}-level dependency chain. "
                "Response may be lengthy."
            )

        # =====================================================================
        # Rule 5: Multi-domain without dependencies → Parallel
        # =====================================================================
        if dependency_analysis.is_parallelizable:
            target_agents = [
                DOMAIN_TO_AGENT[dm.domain]
                for dm in domain_mapping.domains_detected
                if dm.domain in DOMAIN_TO_AGENT
            ]
            # Deduplicate while preserving order
            seen = set()
            unique_agents = []
            for agent in target_agents:
                if agent not in seen:
                    seen.add(agent)
                    unique_agents.append(agent)

            return ClassificationResult(
                routing_pattern=RoutingPattern.PARALLEL_DELEGATION,
                target_agents=unique_agents,
                sub_questions=dependency_analysis.sub_questions,
                confidence=self._calculate_multi_confidence(domain_mapping),
                reasoning="Multi-domain query with independent sub-tasks",
                complexity_warning=complexity_warning,
                is_followup=is_followup,
                context_source=context_source,
                classification_latency_ms=latency_ms,
                used_llm_layer=used_llm,
            )

        # =====================================================================
        # Rule 6: Multi-domain with dependencies → Tool Composer
        # =====================================================================
        return ClassificationResult(
            routing_pattern=RoutingPattern.TOOL_COMPOSER,
            target_agents=["tool_composer"],  # Special routing target
            sub_questions=dependency_analysis.sub_questions,
            dependencies=dependency_analysis.dependencies,
            confidence=self._calculate_multi_confidence(domain_mapping),
            reasoning=(
                f"Multi-domain query with {len(dependency_analysis.dependencies)} "
                "data dependencies requiring tool composition"
            ),
            complexity_warning=complexity_warning,
            is_followup=is_followup,
            context_source=context_source,
            classification_latency_ms=latency_ms,
            used_llm_layer=used_llm,
        )

    def _calculate_multi_confidence(self, domain_mapping: DomainMapping) -> float:
        """Calculate confidence for multi-domain classification."""
        if not domain_mapping.domains_detected:
            return 0.0

        # Average of top domain confidences, weighted by position
        confidences = [dm.confidence for dm in domain_mapping.domains_detected[:3]]
        weights = [0.5, 0.3, 0.2][: len(confidences)]

        weighted_sum = sum(c * w for c, w in zip(confidences, weights, strict=False))
        weight_sum = sum(weights)

        return round(weighted_sum / weight_sum, 3)
