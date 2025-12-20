# src/e2i/agents/orchestrator/classifier/feature_extractor.py
"""
Stage 1: Extract features from raw query text.

This module provides rule-based feature extraction for query classification.
It operates quickly (~10ms) without LLM calls, extracting:
- Structural features (questions, clauses, conditionals)
- Temporal features (time references, tense)
- Entity features (HCPs, regions, drugs, etc.)
- Intent signals (keywords indicating query type)
"""

import re
from typing import Optional

from .schemas import (
    ExtractedFeatures,
    StructuralFeatures,
    TemporalFeatures,
    EntityFeatures,
    IntentSignals,
)


class FeatureExtractor:
    """
    Extracts classification features from query text.
    Pure rule-based for speed.
    """

    # =========================================================================
    # KEYWORD DICTIONARIES
    # =========================================================================

    CONDITIONAL_MARKERS = {
        "if", "would", "what if", "assuming", "suppose", "hypothetically",
        "in case", "should we", "could we"
    }

    COMPARISON_MARKERS = {
        "vs", "versus", "compared to", "relative to", "against",
        "better than", "worse than", "difference between"
    }

    SEQUENCE_MARKERS = {
        "then", "after", "next", "followed by", "subsequently",
        "first", "second", "finally", "before"
    }

    CONNECTORS = {"and", "but", "also", "additionally", "moreover", "plus"}

    TIME_PATTERNS = [
        r"\bQ[1-4]\b",                          # Q1, Q2, Q3, Q4
        r"\b20[0-9]{2}\b",                      # Years
        r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b",
        r"\b(last|this|next)\s+(week|month|quarter|year)\b",
        r"\b(yesterday|today|tomorrow)\b",
        r"\b(recent|current|previous|upcoming)\b",
    ]

    FUTURE_MARKERS = {"would", "will", "predict", "forecast", "expect", "if we"}
    PAST_MARKERS = {"was", "were", "did", "showed", "had", "resulted"}

    # Intent keyword dictionaries
    CAUSAL_KEYWORDS = {
        "impact", "effect", "caused", "drove", "attributed", "due to",
        "resulted in", "led to", "influenced", "affected"
    }

    EXPLORATION_KEYWORDS = {
        "show", "list", "which", "what", "display", "find", "identify",
        "who", "where", "how many"
    }

    PREDICTION_KEYWORDS = {
        "predict", "forecast", "will", "would", "expect", "likelihood",
        "probability", "risk", "chance", "future"
    }

    DESIGN_KEYWORDS = {
        "design", "create", "plan", "test", "experiment", "A/B",
        "trial", "validate", "hypothesis", "setup"
    }

    EXPLANATION_KEYWORDS = {
        "explain", "why", "how", "clarify", "simplify", "summarize",
        "elaborate", "describe", "understand", "mean"
    }

    MONITORING_KEYWORDS = {
        "drift", "shift", "change", "anomaly", "data quality", "issue",
        "problem", "error", "missing", "outlier"
    }

    # Entity patterns (simplified - would use NER in production)
    ENTITY_PATTERNS = {
        "HCP": [
            r"\bHCP[s]?\b", 
            r"\bphysician[s]?\b", 
            r"\bdoctor[s]?\b",
            r"\boncologist[s]?\b", 
            r"\brheumatologist[s]?\b"
        ],
        "region": [
            r"\b(Northeast|Midwest|South|West|Southeast|Northwest)\b",
            r"\bregion[s]?\b", 
            r"\bterritor(y|ies)\b"
        ],
        "drug": [
            r"\b(Kisqali|Fabhalta|Remibrutinib)\b", 
            r"\bbrand[s]?\b"
        ],
        "campaign": [
            r"\b(Q[1-4]\s+)?campaign[s]?\b", 
            r"\bmessaging\b",
            r"\bprogram[s]?\b", 
            r"\bintervention[s]?\b"
        ],
        "segment": [
            r"\bsegment[s]?\b", 
            r"\bcohort[s]?\b", 
            r"\bgroup[s]?\b"
        ],
        "time_period": [
            r"\bQ[1-4]\b", 
            r"\b20[0-9]{2}\b"
        ],
    }

    # =========================================================================
    # MAIN EXTRACTION METHOD
    # =========================================================================

    def extract(self, query: str, context: Optional[dict] = None) -> ExtractedFeatures:
        """
        Extract all features from query text.
        
        Args:
            query: Raw user query
            context: Optional conversation context
            
        Returns:
            ExtractedFeatures with all feature categories
        """
        query_lower = query.lower()

        return ExtractedFeatures(
            structural=self._extract_structural(query, query_lower),
            temporal=self._extract_temporal(query, query_lower),
            entities=self._extract_entities(query, query_lower),
            intent_signals=self._extract_intent_signals(query_lower),
            raw_query=query,
        )

    # =========================================================================
    # STRUCTURAL FEATURES
    # =========================================================================

    def _extract_structural(
        self, query: str, query_lower: str
    ) -> StructuralFeatures:
        """Extract structural features from query."""
        
        # Count questions (? marks + implied questions with "and")
        question_marks = query.count("?")
        # Detect compound questions: "X, and Y?" or "X and what Y"
        compound_pattern = r",?\s+and\s+(what|which|how|why|where|who)"
        compound_matches = len(re.findall(compound_pattern, query_lower))
        question_count = max(question_marks, 1) + compound_matches

        # Count clauses (split by major conjunctions)
        clause_splits = re.split(r"\s+(and|but|or)\s+", query_lower)
        clause_count = len([c for c in clause_splits if len(c.strip()) > 5])

        # Check for conditional markers
        has_conditional = any(
            marker in query_lower for marker in self.CONDITIONAL_MARKERS
        )

        # Check for comparison markers
        has_comparison = any(
            marker in query_lower for marker in self.COMPARISON_MARKERS
        )

        # Check for sequence markers
        has_sequence = any(
            marker in query_lower for marker in self.SEQUENCE_MARKERS
        )

        # Word count
        words = query.split()
        word_count = len(words)

        # Connector density
        connector_count = sum(
            1 for word in words if word.lower() in self.CONNECTORS
        )
        connector_density = connector_count / max(word_count, 1)

        return StructuralFeatures(
            question_count=question_count,
            clause_count=max(clause_count, 1),
            has_conditional=has_conditional,
            has_comparison=has_comparison,
            has_sequence=has_sequence,
            word_count=word_count,
            connector_density=round(connector_density, 3),
        )

    # =========================================================================
    # TEMPORAL FEATURES
    # =========================================================================

    def _extract_temporal(
        self, query: str, query_lower: str
    ) -> TemporalFeatures:
        """Extract temporal features from query."""
        
        time_references = []
        for pattern in self.TIME_PATTERNS:
            matches = re.findall(pattern, query_lower, re.IGNORECASE)
            time_references.extend(matches)

        # Deduplicate while preserving order
        seen = set()
        unique_refs = []
        for ref in time_references:
            ref_lower = ref.lower() if isinstance(ref, str) else str(ref).lower()
            if ref_lower not in seen:
                seen.add(ref_lower)
                unique_refs.append(ref)

        return TemporalFeatures(
            time_references=unique_refs,
            time_span_count=len(unique_refs),
            has_future=any(m in query_lower for m in self.FUTURE_MARKERS),
            has_past=any(m in query_lower for m in self.PAST_MARKERS),
        )

    # =========================================================================
    # ENTITY FEATURES
    # =========================================================================

    def _extract_entities(
        self, query: str, query_lower: str
    ) -> EntityFeatures:
        """Extract entity features from query."""
        
        entity_types = []
        entity_mentions = []

        for entity_type, patterns in self.ENTITY_PATTERNS.items():
            for pattern in patterns:
                matches = re.findall(pattern, query, re.IGNORECASE)
                if matches:
                    if entity_type not in entity_types:
                        entity_types.append(entity_type)
                    entity_mentions.extend(matches)

        return EntityFeatures(
            entity_types=entity_types,
            entity_mentions=list(set(entity_mentions)),
            entity_type_count=len(entity_types),
        )

    # =========================================================================
    # INTENT SIGNALS
    # =========================================================================

    def _extract_intent_signals(self, query_lower: str) -> IntentSignals:
        """Extract intent signal keywords from query."""
        
        def find_matches(keywords: set) -> list[str]:
            return [kw for kw in keywords if kw in query_lower]

        return IntentSignals(
            causal_keywords=find_matches(self.CAUSAL_KEYWORDS),
            exploration_keywords=find_matches(self.EXPLORATION_KEYWORDS),
            prediction_keywords=find_matches(self.PREDICTION_KEYWORDS),
            design_keywords=find_matches(self.DESIGN_KEYWORDS),
            explanation_keywords=find_matches(self.EXPLANATION_KEYWORDS),
            monitoring_keywords=find_matches(self.MONITORING_KEYWORDS),
        )
