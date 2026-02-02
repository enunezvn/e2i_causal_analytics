# src/e2i/agents/orchestrator/classifier/domain_mapper.py
"""
Stage 2: Map extracted features to agent domains.

This module maps the features extracted in Stage 1 to agent capability
domains using weighted scoring. Each domain has specific signals that
indicate relevance.
"""

from .schemas import (
    Domain,
    DomainMapping,
    DomainMatch,
    EntityFeatures,
    ExtractedFeatures,
    IntentSignals,
    StructuralFeatures,
    TemporalFeatures,
)


class DomainMapper:
    """
    Maps extracted features to agent capability domains.
    Uses weighted scoring based on feature signals.
    """

    # =========================================================================
    # DOMAIN SCORING WEIGHTS
    # =========================================================================

    DOMAIN_WEIGHTS = {
        Domain.CAUSAL_ANALYSIS: {
            "intent_keywords": 0.5,  # causal_keywords presence
            "structural_conditional": 0.2,  # has_conditional
            "temporal_past": 0.1,  # has_past (analyzing what happened)
            "base": 0.2,
        },
        Domain.HETEROGENEITY: {
            "intent_keywords": 0.4,  # exploration_keywords
            "entity_segments": 0.3,  # segment-related entities
            "structural_comparison": 0.2,
            "base": 0.1,
        },
        Domain.GAP_ANALYSIS: {
            "intent_keywords": 0.4,
            "entity_regions": 0.3,  # region entities
            "structural_comparison": 0.2,
            "base": 0.1,
        },
        Domain.EXPERIMENTATION: {
            "intent_keywords": 0.6,  # design_keywords are strong signal
            "structural_conditional": 0.2,
            "temporal_future": 0.1,
            "base": 0.1,
        },
        Domain.PREDICTION: {
            "intent_keywords": 0.5,  # prediction_keywords
            "structural_conditional": 0.2,
            "temporal_future": 0.2,
            "base": 0.1,
        },
        Domain.MONITORING: {
            "intent_keywords": 0.6,  # monitoring_keywords
            "base": 0.4,
        },
        Domain.EXPLANATION: {
            "intent_keywords": 0.7,  # explanation_keywords
            "base": 0.3,
        },
        Domain.COHORT_DEFINITION: {
            "intent_keywords": 0.7,  # cohort_keywords are strong signal
            "entity_segments": 0.2,  # segment/cohort entities boost
            "base": 0.1,
        },
    }

    # Minimum confidence to include a domain
    CONFIDENCE_THRESHOLD = 0.3

    # =========================================================================
    # MAIN MAPPING METHOD
    # =========================================================================

    def map_domains(self, features: ExtractedFeatures) -> DomainMapping:
        """
        Map features to domains with confidence scores.

        Args:
            features: Extracted features from Stage 1

        Returns:
            DomainMapping with detected domains and confidences
        """
        domain_scores: list[DomainMatch] = []

        for domain in Domain:
            confidence, evidence = self._score_domain(domain, features)
            if confidence >= self.CONFIDENCE_THRESHOLD:
                domain_scores.append(
                    DomainMatch(
                        domain=domain,
                        confidence=round(confidence, 3),
                        evidence=evidence,
                    )
                )

        # Sort by confidence descending
        domain_scores.sort(key=lambda x: x.confidence, reverse=True)

        # Determine primary domain
        primary_domain = domain_scores[0].domain if domain_scores else None

        return DomainMapping(
            domains_detected=domain_scores,
            domain_count=len(domain_scores),
            primary_domain=primary_domain,
            is_multi_domain=len(domain_scores) > 1,
        )

    # =========================================================================
    # DOMAIN SCORING
    # =========================================================================

    def _score_domain(self, domain: Domain, features: ExtractedFeatures) -> tuple[float, list[str]]:
        """
        Calculate confidence score for a domain.

        Args:
            domain: Domain to score
            features: Extracted features

        Returns:
            Tuple of (confidence score, evidence list)
        """
        weights = self.DOMAIN_WEIGHTS[domain]
        score = 0.0
        evidence = []

        # Score based on intent keywords
        keyword_score, keyword_evidence = self._score_intent_keywords(
            domain, features.intent_signals
        )
        score += keyword_score * weights.get("intent_keywords", 0)
        evidence.extend(keyword_evidence)

        # Score based on structural features
        structural_score, structural_evidence = self._score_structural(domain, features.structural)
        score += structural_score * weights.get("structural_conditional", 0)
        score += structural_score * weights.get("structural_comparison", 0)
        evidence.extend(structural_evidence)

        # Score based on temporal features
        temporal_score, temporal_evidence = self._score_temporal(domain, features.temporal)
        score += temporal_score * weights.get("temporal_past", 0)
        score += temporal_score * weights.get("temporal_future", 0)
        evidence.extend(temporal_evidence)

        # Score based on entity features
        entity_score, entity_evidence = self._score_entities(domain, features.entities)
        score += entity_score * weights.get("entity_segments", 0)
        score += entity_score * weights.get("entity_regions", 0)
        evidence.extend(entity_evidence)

        # Add base score
        score += weights.get("base", 0)

        # Normalize to [0, 1]
        score = min(score, 1.0)

        return score, evidence

    def _score_intent_keywords(
        self, domain: Domain, signals: IntentSignals
    ) -> tuple[float, list[str]]:
        """Score based on intent keywords."""

        keyword_map = {
            Domain.CAUSAL_ANALYSIS: signals.causal_keywords,
            Domain.HETEROGENEITY: signals.exploration_keywords,
            Domain.GAP_ANALYSIS: signals.exploration_keywords,  # Overlap
            Domain.EXPERIMENTATION: signals.design_keywords,
            Domain.PREDICTION: signals.prediction_keywords,
            Domain.MONITORING: signals.monitoring_keywords,
            Domain.EXPLANATION: signals.explanation_keywords,
            Domain.COHORT_DEFINITION: signals.cohort_keywords,
        }

        keywords = keyword_map.get(domain, [])
        if keywords:
            # More keywords = higher confidence (diminishing returns)
            score = min(len(keywords) * 0.3, 1.0)
            return score, keywords
        return 0.0, []

    def _score_structural(
        self, domain: Domain, structural: StructuralFeatures
    ) -> tuple[float, list[str]]:
        """Score based on structural features."""

        evidence = []
        score = 0.0

        if domain in {Domain.CAUSAL_ANALYSIS, Domain.PREDICTION, Domain.EXPERIMENTATION}:
            if structural.has_conditional:
                score += 0.5
                evidence.append("conditional_structure")

        if domain in {Domain.HETEROGENEITY, Domain.GAP_ANALYSIS}:
            if structural.has_comparison:
                score += 0.5
                evidence.append("comparison_structure")

        return min(score, 1.0), evidence

    def _score_temporal(
        self, domain: Domain, temporal: TemporalFeatures
    ) -> tuple[float, list[str]]:
        """Score based on temporal features."""

        evidence = []
        score = 0.0

        if domain == Domain.CAUSAL_ANALYSIS and temporal.has_past:
            score += 0.5
            evidence.append("past_tense_analysis")

        if domain in {Domain.PREDICTION, Domain.EXPERIMENTATION}:
            if temporal.has_future:
                score += 0.5
                evidence.append("future_orientation")

        return min(score, 1.0), evidence

    def _score_entities(self, domain: Domain, entities: EntityFeatures) -> tuple[float, list[str]]:
        """Score based on entity features."""

        evidence = []
        score = 0.0

        if domain == Domain.HETEROGENEITY:
            if "segment" in entities.entity_types or "HCP" in entities.entity_types:
                score += 0.5
                evidence.extend([f"entity:{t}" for t in entities.entity_types])

        if domain == Domain.GAP_ANALYSIS:
            if "region" in entities.entity_types:
                score += 0.5
                evidence.append("entity:region")

        if domain == Domain.COHORT_DEFINITION:
            # Boost cohort scoring for cohort-relevant entities
            cohort_entities = {"cohort", "patient", "population", "HCP", "segment"}
            matched_entities = cohort_entities.intersection(entities.entity_types)
            if matched_entities:
                score += 0.5
                evidence.extend([f"entity:{t}" for t in matched_entities])

        return min(score, 1.0), evidence
