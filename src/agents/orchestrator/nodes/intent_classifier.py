"""Intent classification node for orchestrator agent.

Fast intent classification optimized for <500ms:
- Pattern matching first (fastest)
- LLM fallback for ambiguous cases (Haiku)
"""

import re
import time
from typing import Any, Dict

from langchain_anthropic import ChatAnthropic

from ..state import IntentClassification, OrchestratorState


class IntentClassifierNode:
    """Fast intent classification - optimized for <500ms.

    Uses pattern matching first, LLM only for ambiguous cases.
    """

    # Pattern-based classification for common queries
    INTENT_PATTERNS = {
        "causal_effect": [
            r"what.*(caus|impact|effect|driv|lead|result)",
            r"why.*(increase|decrease|change|drop|rise)",
            r"how does.*affect",
            r"what drives",
            r"attribution",
        ],
        "performance_gap": [
            r"(gap|opportunit|underperform|potential|improve)",
            r"roi.*(opportun|analys)",
            r"where.*underperform",
            r"untapped",
        ],
        "segment_analysis": [
            r"(segment|cohort|group|heterogen)",
            r"which.*(respond|perform).*(best|better)",
            r"\bcate\b|treatment effect.*by",
            r"differentiat.*strategy",
        ],
        "experiment_design": [
            r"(design|run|plan).*(experiment|test|trial)",
            r"a/b test",
            r"sample size",
            r"hypothesis.*test",
        ],
        "prediction": [
            r"predict|forecast|project",
            r"what will|expected",
            r"likelihood|probability",
        ],
        "resource_allocation": [
            r"(allocat|optimi|distribut).*(resource|budget|rep)",
            r"where.*invest",
            r"prioriti",
        ],
        "explanation": [
            r"explain|clarify|what does.*mean",
            r"help.*understand",
            r"break down",
        ],
        "system_health": [
            r"system.*(health|status)",
            r"model.*perform",
            r"pipeline.*status",
        ],
        "drift_check": [
            r"drift|shift|distribution.*change",
            r"data quality",
            r"model.*degrad",
        ],
        "feedback": [
            r"feedback|learn.*from",
            r"improve.*based on",
        ],
    }

    def __init__(self):
        """Initialize intent classifier with Haiku for fast classification."""
        # Use Haiku for fast classification
        self.llm = ChatAnthropic(model="claude-haiku-4-20250414", max_tokens=256, timeout=2)

    async def execute(self, state: OrchestratorState) -> OrchestratorState:
        """Execute intent classification.

        Args:
            state: Current orchestrator state

        Returns:
            Updated state with intent classification
        """
        start_time = time.time()

        query = state.get("query", "").lower()

        # Try pattern matching first (fastest)
        pattern_result = self._pattern_classify(query)

        if pattern_result["confidence"] >= 0.8:
            intent = pattern_result
        else:
            # Fall back to LLM for ambiguous cases
            intent = await self._llm_classify(state.get("query", ""))

        classification_time = int((time.time() - start_time) * 1000)

        return {
            **state,
            "intent": intent,
            "classification_latency_ms": classification_time,
            "current_phase": "routing",
        }

    def _pattern_classify(self, query: str) -> IntentClassification:
        """Fast pattern-based classification.

        Args:
            query: User query (lowercased)

        Returns:
            Intent classification result
        """
        scores = {}

        for intent, patterns in self.INTENT_PATTERNS.items():
            matched_count = 0
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    matched_count += 1
            # Any pattern match gives high confidence for that intent
            # More matches = higher confidence
            if matched_count > 0:
                scores[intent] = 0.8 + (0.2 * min(matched_count, 3) / 3)
            else:
                scores[intent] = 0.0

        if not scores or max(scores.values()) == 0:
            return IntentClassification(
                primary_intent="general",
                confidence=0.5,
                secondary_intents=[],
                requires_multi_agent=False,
            )

        primary = max(scores, key=scores.get)
        confidence = scores[primary]

        # Get secondary intents (those with matches but lower score)
        secondary = [
            k for k, v in sorted(scores.items(), key=lambda x: -x[1]) if v > 0 and k != primary
        ]

        return IntentClassification(
            primary_intent=primary,
            confidence=confidence,
            secondary_intents=secondary[:2],
            requires_multi_agent=len(secondary) > 0 and scores.get(secondary[0], 0) > 0.8,
        )

    async def _llm_classify(self, query: str) -> IntentClassification:
        """LLM-based classification for ambiguous cases.

        Args:
            query: User query

        Returns:
            Intent classification result
        """
        prompt = f"""Classify this pharmaceutical analytics query into ONE primary intent.

Query: "{query}"

Intents:
- causal_effect: Questions about cause and effect, impact, attribution
- performance_gap: ROI opportunities, underperformance, potential improvements
- segment_analysis: Segment-specific effects, CATE, cohort analysis
- experiment_design: A/B tests, experiment planning, sample size
- prediction: Forecasting, projections, likelihood estimates
- resource_allocation: Budget/resource optimization, prioritization
- explanation: Clarifying results, interpreting findings
- system_health: Model/pipeline status, system performance
- drift_check: Data/model drift, distribution changes
- feedback: Learning from outcomes, improvement suggestions
- general: Other/unclear

Respond with ONLY a JSON object:
{{"primary_intent": "<intent>", "confidence": <0.0-1.0>, "requires_multi_agent": <bool>}}"""

        try:
            response = await self.llm.ainvoke(prompt)
            import json

            result = json.loads(response.content)
            return IntentClassification(
                primary_intent=result.get("primary_intent", "general"),
                confidence=result.get("confidence", 0.5),
                secondary_intents=[],
                requires_multi_agent=result.get("requires_multi_agent", False),
            )
        except Exception:
            return IntentClassification(
                primary_intent="general",
                confidence=0.3,
                secondary_intents=[],
                requires_multi_agent=False,
            )


# Export for use in graph
async def classify_intent(state: Dict[str, Any]) -> Dict[str, Any]:
    """Node function for intent classification.

    Args:
        state: Current state

    Returns:
        Updated state
    """
    classifier = IntentClassifierNode()
    return await classifier.execute(state)
