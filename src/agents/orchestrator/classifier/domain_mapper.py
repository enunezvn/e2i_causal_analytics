# src/e2i/agents/orchestrator/classifier/domain_mapper.py
"""
Stage 2: Map extracted features to agent domains.

This module maps the features extracted in Stage 1 to agent capability
domains using weighted scoring. Each domain has specific signals that
indicate relevance.
"""

import re

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

# =============================================================================
# KPI VALUE-LOOKUP FAST PATH (#1593)
# =============================================================================
# Stage 2 had no notion of a KPI value lookup, so "what is the TRx for Kisqali"
# scored no domain above CONFIDENCE_THRESHOLD and the whole pipeline abstained
# — 46 of the 54 KPI-lookup rows in the 337-row #1337 gold set, the largest
# gold class. Teaching the mapper the pattern the intent classifier already
# routes on takes that subset from 7/54 to 46/54 agents-exact.
#
# The pattern is IMPORTED, never copied: ``KPI_VALUE_LOOKUP_RE`` (#1475) is the
# SSOT shared with the orchestrator dispatcher's explainer resolver, which
# binds a REAL KPI value for exactly the shape it selects. A forked copy would
# let "routes to explainer" and "explainer can answer it" drift apart. The
# import is function-local because ``nodes/intent_classifier.py`` imports THIS
# package (``from ..classifier import ClassificationPipeline``) — a
# module-level import here is a genuine import cycle.

# Clears PatternSelector's explanation-override floor (> 0.7). The DECISION is
# deterministic — a vetted regex hit, not a scored guess — so it is not hedged
# below the floor that would send it back to the ambiguity path.
KPI_LOOKUP_CONFIDENCE = 0.85
KPI_LOOKUP_EVIDENCE = "kpi_value_lookup"

# --- narrowing: shapes the fast path must NOT claim ---------------------------
# In active mode the classifier's decision IS the dispatch plan
# (RouterNode._dispatch_from_classification), so the fast path may only claim a
# query when the KPI value lookup is the WHOLE ask. Two shapes measured as
# active-mode degradations on the gold set are handed back to legacy routing.
#
# (a) Population breakdown. "NRx broken down by patient segment" asks for a
#     DECOMPOSITION; the explainer resolver binds one scalar, so the fast
#     path's own answerability premise fails. cohort_profiler owns per-segment
#     counts (see PatternSelector.DOMAIN_TO_AGENT). Gold bench-0008 / 0133 /
#     0139 / 0140 / 0141 are all gold ``cohort_profiler``.
_POPULATION_AXIS_RE = re.compile(r"\bpatients?\b", re.IGNORECASE)
_DECOMPOSITION_RE = re.compile(
    r"\bsegments?\b|\bsegmented\b|\bbreak\s*downs?\b|\bbroken\s+down\b|\bsplit\b|\bcohorts?\b",
    re.IGNORECASE,
)


def _kpi_value_lookup_re() -> re.Pattern[str]:
    """The #1475 KPI-value-lookup SSOT (function-local: import cycle, above)."""
    from ..nodes.intent_classifier import KPI_VALUE_LOOKUP_RE

    return KPI_VALUE_LOOKUP_RE


def _is_population_breakdown(query: str) -> bool:
    """A per-patient-population decomposition, not a single-figure lookup."""
    return bool(_POPULATION_AXIS_RE.search(query) and _DECOMPOSITION_RE.search(query))


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

        if self._takes_kpi_lookup_fast_path(features):
            # Promote EXPLANATION to first so PatternSelector's explanation
            # override fires. FLOOR, not overwrite: a query that also scored
            # EXPLANATION above KPI_LOOKUP_CONFIDENCE on real keyword evidence
            # ("whats TRx mean? explain how") keeps its higher score and its
            # evidence — the fast path may only ever raise this domain.
            scored = next(
                (dm for dm in domain_scores if dm.domain is Domain.EXPLANATION),
                None,
            )
            domain_scores = [
                DomainMatch(
                    domain=Domain.EXPLANATION,
                    confidence=max(KPI_LOOKUP_CONFIDENCE, scored.confidence if scored else 0.0),
                    evidence=[KPI_LOOKUP_EVIDENCE] + (scored.evidence if scored else []),
                )
            ] + [dm for dm in domain_scores if dm.domain is not Domain.EXPLANATION]

        # Determine primary domain
        primary_domain = domain_scores[0].domain if domain_scores else None

        return DomainMapping(
            domains_detected=domain_scores,
            domain_count=len(domain_scores),
            primary_domain=primary_domain,
            is_multi_domain=len(domain_scores) > 1,
        )

    def _takes_kpi_lookup_fast_path(self, features: ExtractedFeatures) -> bool:
        """Whether this query is a KPI value lookup the explainer can answer whole.

        The SSOT pattern carries its own vetoes (notably the whole-query
        forecast guard, so a forecast ask is never answered with a
        current-period figure). The two added here are the NARROWING: shapes
        where a lone explainer would be a measured active-mode degradation.
        """
        query = features.raw_query
        if not _kpi_value_lookup_re().search(query):
            return False
        if _is_population_breakdown(query):
            return False
        # (b) Compound ask — a second wh-clause after a connector is a second
        # facet a lone explainer silently drops (gold bench-0143, gold
        # PARALLEL_DELEGATION[explainer, gap_analyzer]). Deliberately
        # conservative: it also yields on compound asks the explainer COULD
        # cover (gold bench-0064 / 0135), because the two are not separable
        # without fitting the gold set — the DomainMapper scores GAP_ANALYSIS
        # for both (0.49 vs 0.34) purely off the shared "what"/"which"
        # exploration keywords. Yielding costs a forgone win; claiming costs a
        # dropped facet, and the second is the one that reaches a user.
        if features.structural.has_compound_question:
            return False
        return True

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

        # Add base score only when the domain has at least one piece of real
        # evidence. An unconditional base lets evidence-free domains clear
        # CONFIDENCE_THRESHOLD (MONITORING base 0.4, EXPLANATION base 0.3 vs
        # threshold 0.3), which made every query classify as multi-domain.
        if evidence:
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
