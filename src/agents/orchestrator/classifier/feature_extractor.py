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
    EntityFeatures,
    ExtractedFeatures,
    IntentSignals,
    StructuralFeatures,
    TemporalFeatures,
)

# A second ASK hanging off a connector — "what is the total TRx AND WHICH
# region has the largest gap". Module-level so Stage 2 can gate on the same
# signal Stage 1 counts with (#1593); a forked copy would let the compound
# veto and the compound count disagree.
#
# The imperative heads (compare/show/list/...) matter as much as the wh-words:
# "what is TRx and compare it to last quarter" is just as compound as
# "... and which region leads", and a wh-only pattern silently misses that
# whole family (codex iter-1 MEDIUM). ``question_count`` has no behavioural
# consumer, so widening here only makes the compound count more accurate.
COMPOUND_QUESTION_RE = re.compile(
    r",?\s+and\s+(?:also\s+|then\s+)?"
    r"(what|which|how|why|where|who|whose"
    r"|compare|contrast|show|list|display|give|tell|find|identify|rank|break)\b",
    re.IGNORECASE,
)

# ``COMPOUND_QUESTION_RE`` above enumerates connector FORMS, which is a losing
# game: it missed imperative heads (codex iter-1), then whole second SENTENCES
# with no connector at all (iter-4), then polite/modal interposition —
# "and please show me...", "and can you rank..." (iter-5). Each fix closed one
# spelling and left the next.
#
# So the veto's real test is structural and enumerates nothing: cut the query
# at every clause boundary (sentence terminators AND "and"), then count the
# segments that OPEN AN ASK. Two or more asks is a compound query however it is
# punctuated or padded. This subsumes the connector pattern; the two are OR-ed
# so ``question_count`` keeps its original connector-based meaning.
#
# The ask-head list is what keeps it safe in the other direction: a second
# segment that is not itself an ask does not count, so "whats TRx mean? total
# rx's?" (gold bench-0253) and "what is the TRx for kisqali and remibrutinib"
# (one ask, two entities) stay single lookups.
_CLAUSE_SPLIT_RE = re.compile(r"[?;.!,\n]+|\band\b", re.IGNORECASE)
# ANCHORED at the segment start (after at most a short politeness/modal
# lead-in), because the test is whether a segment OPENS an ask — a head buried
# mid-clause ("the TRx broken down by what measure") is not a second ask.
_ASK_HEAD_RE = re.compile(
    r"^\s*(?:(?:also|then|please|kindly|can|could|would|will|you|i|we|let|me)\s+){0,3}"
    r"(what'?s?|which|how|why|where|who|whose|when"
    r"|compare|contrast|show|list|display|give|tell|find|identify|rank|break)\b",
    re.IGNORECASE,
)


def has_second_ask(query: str) -> bool:
    """Two or more clause-like segments that each OPEN an ask.

    Known residual: a leading subordinate clause that itself starts with a
    wh-word ("When looking at Kisqali, what is the TRx?") counts as an ask, so
    such queries are treated as compound and fall through to legacy routing.
    Separating those from real interrogatives needs syntax, not shape — and the
    error direction is a forgone improvement, never a dropped facet. Zero of
    the 337 #1337 gold queries take that form.
    """
    asks = 0
    for segment in _CLAUSE_SPLIT_RE.split(query):
        if len(segment.split()) >= 2 and _ASK_HEAD_RE.search(segment):
            asks += 1
            if asks >= 2:
                return True
    return False


class FeatureExtractor:
    """
    Extracts classification features from query text.
    Pure rule-based for speed.
    """

    # =========================================================================
    # KEYWORD DICTIONARIES
    # =========================================================================

    CONDITIONAL_MARKERS = {
        "if",
        "would",
        "what if",
        "assuming",
        "suppose",
        "hypothetically",
        "in case",
        "should we",
        "could we",
    }

    COMPARISON_MARKERS = {
        "vs",
        "versus",
        "compared to",
        "relative to",
        "against",
        "better than",
        "worse than",
        "difference between",
    }

    SEQUENCE_MARKERS = {
        "then",
        "after",
        "next",
        "followed by",
        "subsequently",
        "first",
        "second",
        "finally",
        "before",
    }

    CONNECTORS = {"and", "but", "also", "additionally", "moreover", "plus"}

    TIME_PATTERNS = [
        r"\bQ[1-4]\b",  # Q1, Q2, Q3, Q4
        r"\b20[0-9]{2}\b",  # Years
        r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b",
        r"\b(last|this|next)\s+(week|month|quarter|year)\b",
        r"\b(yesterday|today|tomorrow)\b",
        r"\b(recent|current|previous|upcoming)\b",
    ]

    FUTURE_MARKERS = {"would", "will", "predict", "forecast", "expect", "if we"}
    PAST_MARKERS = {"was", "were", "did", "showed", "had", "resulted"}

    # Intent keyword dictionaries
    CAUSAL_KEYWORDS = {
        "impact",
        "effect",
        "caused",
        "drove",
        "attributed",
        "due to",
        "resulted in",
        "led to",
        "influenced",
        "affected",
    }

    EXPLORATION_KEYWORDS = {
        "show",
        "list",
        "which",
        "what",
        "display",
        "find",
        "identify",
        "who",
        "where",
        "how many",
    }

    PREDICTION_KEYWORDS = {
        "predict",
        "forecast",
        "will",
        "would",
        "expect",
        "likelihood",
        "probability",
        "risk",
        "chance",
        "future",
    }

    DESIGN_KEYWORDS = {
        "design",
        "create",
        "plan",
        "test",
        "experiment",
        "A/B",
        "trial",
        "validate",
        "hypothesis",
        "setup",
    }

    EXPLANATION_KEYWORDS = {
        "explain",
        "why",
        "how",
        "clarify",
        "simplify",
        "summarize",
        "elaborate",
        "describe",
        "understand",
        "mean",
    }

    MONITORING_KEYWORDS = {
        "drift",
        "shift",
        "change",
        "anomaly",
        "data quality",
        "issue",
        "problem",
        "error",
        "missing",
        "outlier",
    }

    COHORT_KEYWORDS = {
        "cohort",
        "define cohort",
        "define a cohort",
        "build cohort",
        "build a cohort",
        "construct cohort",
        "create cohort",
        "create a cohort",
        "patient population",
        "eligible patients",
        "eligibility",
        "inclusion",
        "exclusion",
        "criteria",
        "filter patients",
        "patient selection",
        "target population",
        "patient segment",
        "patient group",
        "patients with",  # Common pattern: "patients with [condition]"
        "hcp cohort",
        "hcp selection",
    }

    # Entity patterns (simplified - would use NER in production)
    ENTITY_PATTERNS = {
        "HCP": [
            r"\bHCP[s]?\b",
            r"\bphysician[s]?\b",
            r"\bdoctor[s]?\b",
            r"\boncologist[s]?\b",
            r"\brheumatologist[s]?\b",
        ],
        "region": [
            r"\b(Northeast|Midwest|South|West|Southeast|Northwest)\b",
            r"\bregion[s]?\b",
            r"\bterritor(y|ies)\b",
        ],
        "drug": [r"\b(Kisqali|Fabhalta|Remibrutinib)\b", r"\bbrand[s]?\b"],
        "campaign": [
            r"\b(Q[1-4]\s+)?campaign[s]?\b",
            r"\bmessaging\b",
            r"\bprogram[s]?\b",
            r"\bintervention[s]?\b",
        ],
        "segment": [r"\bsegment[s]?\b", r"\bcohort[s]?\b", r"\bgroup[s]?\b"],
        "time_period": [r"\bQ[1-4]\b", r"\b20[0-9]{2}\b"],
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

    def _extract_structural(self, query: str, query_lower: str) -> StructuralFeatures:
        """Extract structural features from query."""

        # Count questions (? marks + implied questions with "and")
        question_marks = query.count("?")
        # Detect compound questions: "X, and Y?" or "X and what Y"
        compound_matches = len(COMPOUND_QUESTION_RE.findall(query_lower))
        question_count = max(question_marks, 1) + compound_matches

        # Count clauses (split by major conjunctions)
        clause_splits = re.split(r"\s+(and|but|or)\s+", query_lower)
        clause_count = len([c for c in clause_splits if len(c.strip()) > 5])

        # Check for conditional markers
        has_conditional = any(marker in query_lower for marker in self.CONDITIONAL_MARKERS)

        # Check for comparison markers
        has_comparison = any(marker in query_lower for marker in self.COMPARISON_MARKERS)

        # Check for sequence markers
        has_sequence = any(marker in query_lower for marker in self.SEQUENCE_MARKERS)

        # Word count
        words = query.split()
        word_count = len(words)

        # Connector density
        connector_count = sum(1 for word in words if word.lower() in self.CONNECTORS)
        connector_density = connector_count / max(word_count, 1)

        return StructuralFeatures(
            question_count=question_count,
            has_compound_question=compound_matches > 0 or has_second_ask(query),
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

    def _extract_temporal(self, query: str, query_lower: str) -> TemporalFeatures:
        """Extract temporal features from query."""

        # group(0) = full matched text; findall would return capture-group
        # tuples for multi-group patterns (e.g. "next quarter") and fail
        # TemporalFeatures validation.
        time_references = []
        for pattern in self.TIME_PATTERNS:
            for match in re.finditer(pattern, query_lower, re.IGNORECASE):
                time_references.append(match.group(0))

        # Deduplicate while preserving order
        seen = set()
        unique_refs = []
        for ref in time_references:
            ref_lower = ref.lower()
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

    def _extract_entities(self, query: str, query_lower: str) -> EntityFeatures:
        """Extract entity features from query."""

        entity_types = []
        entity_mentions = []

        for entity_type, patterns in self.ENTITY_PATTERNS.items():
            for pattern in patterns:
                # group(0) keeps the full mention ("territory", not the
                # capture-group fragment "y" that findall would return)
                matches = [m.group(0) for m in re.finditer(pattern, query, re.IGNORECASE)]
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
            cohort_keywords=find_matches(self.COHORT_KEYWORDS),
        )
