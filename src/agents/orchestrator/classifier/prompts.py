# src/e2i/agents/orchestrator/classifier/prompts.py
"""
LLM prompts for classifier components.

These prompts are used by the LLM layer for complex classification tasks
that cannot be handled by rule-based approaches alone.
"""

# =============================================================================
# DEPENDENCY DETECTION PROMPT
# =============================================================================

DEPENDENCY_DETECTION_PROMPT = """Analyze this query and identify data dependencies between sub-questions.

QUERY: {query}

SUB-QUESTIONS:
{sub_questions}

For each pair of sub-questions, determine if there's a dependency where one requires output from another.

DEPENDENCY TYPES:
- REFERENCE_CHAIN: Later question references results from earlier (pronouns, "the result", etc.)
- CONDITIONAL: "if X then Y" structure where Y depends on X's outcome
- LOGICAL_SEQUENCE: Natural ordering required (must know cause before effect)
- ENTITY_TRANSFORMATION: Entity set filtered/transformed by earlier step

Respond in JSON format:
{{
    "dependencies": [
        {{
            "from": "Q1",
            "to": "Q2",
            "type": "REFERENCE_CHAIN",
            "reason": "Q2 uses 'those HCPs' referring to Q1 results"
        }}
    ],
    "parallelizable_pairs": ["Q1-Q3"],
    "reasoning": "Brief explanation of dependency structure"
}}

If no dependencies exist, return empty dependencies array.
Analyze carefully - missing a dependency leads to incorrect results, while false positives just reduce parallelism."""


# =============================================================================
# QUERY DECOMPOSITION PROMPT (for complex queries)
# =============================================================================

QUERY_DECOMPOSITION_PROMPT = """Decompose this multi-part query into distinct sub-questions.

QUERY: {query}

AVAILABLE DOMAINS:
- CAUSAL_ANALYSIS: Cause-effect relationships, impact measurement
- HETEROGENEITY: Segment variation, who responds best
- GAP_ANALYSIS: Performance gaps, opportunities
- EXPERIMENTATION: Test design, what-if scenarios
- PREDICTION: Future outcomes, forecasts
- MONITORING: Data quality, drift detection
- EXPLANATION: Clarification, summarization

For each sub-question, identify:
1. The text of the sub-question
2. Which domain(s) it belongs to
3. The primary domain

Respond in JSON format:
{{
    "sub_questions": [
        {{
            "id": "Q1",
            "text": "What was the impact of Q3 speaker programs on Kisqali adoption?",
            "domains": ["CAUSAL_ANALYSIS"],
            "primary_domain": "CAUSAL_ANALYSIS"
        }},
        {{
            "id": "Q2",
            "text": "Which HCP segments responded best?",
            "domains": ["HETEROGENEITY", "CAUSAL_ANALYSIS"],
            "primary_domain": "HETEROGENEITY"
        }}
    ],
    "reasoning": "Brief explanation of decomposition"
}}

Keep sub-questions atomic - each should map to a single primary analytical task."""


# =============================================================================
# AMBIGUITY RESOLUTION PROMPT
# =============================================================================

AMBIGUITY_RESOLUTION_PROMPT = """The following query is ambiguous. Generate clarifying questions.

QUERY: {query}

DETECTED DOMAINS (with confidence):
{domain_scores}

AMBIGUITY REASONS:
- Multiple domains with similar confidence scores
- Missing key context (time period, entity specification, etc.)
- Unclear analytical intent

Generate 1-3 clarifying questions that would help determine:
1. The user's primary analytical intent
2. Any missing entity specifications
3. Time period or scope if unclear

Respond in JSON format:
{{
    "clarifying_questions": [
        "Are you asking about the causal impact or just descriptive trends?",
        "Which time period should I focus on?"
    ],
    "assumed_interpretation": "If no clarification provided, I would interpret this as...",
    "confidence_if_assumed": 0.6
}}"""


# =============================================================================
# DOMAIN CONFIDENCE CALIBRATION PROMPT
# =============================================================================

DOMAIN_CALIBRATION_PROMPT = """Review and calibrate domain confidence scores for this query.

QUERY: {query}

RULE-BASED SCORES:
{rule_based_scores}

EXTRACTED FEATURES:
- Intent keywords: {intent_keywords}
- Structural: conditional={has_conditional}, comparison={has_comparison}
- Entities: {entities}

Review the rule-based scores and adjust if needed. Consider:
1. Are any domains over/under-weighted based on the actual query intent?
2. Should any domains be added that weren't detected?
3. Are the relative rankings correct?

Respond in JSON format:
{{
    "calibrated_scores": [
        {{"domain": "CAUSAL_ANALYSIS", "confidence": 0.85, "adjustment_reason": "Strong causal language"}},
        {{"domain": "HETEROGENEITY", "confidence": 0.45, "adjustment_reason": "Segment mentioned but secondary"}}
    ],
    "domains_to_add": [],
    "domains_to_remove": [],
    "reasoning": "Brief explanation of calibration"
}}"""
