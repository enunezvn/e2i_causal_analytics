"""
E2I Feedback Learner Agent - Learning Extractor Node
Version: 4.2
Purpose: Extract actionable learnings from detected patterns
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

from ..state import FeedbackLearnerState, LearningRecommendation

logger = logging.getLogger(__name__)


def _get_opik_connector():
    """Lazy import of OpikConnector to avoid circular imports."""
    try:
        from src.mlops.opik_connector import get_opik_connector

        return get_opik_connector()
    except ImportError:
        logger.debug("OpikConnector not available")
        return None
    except Exception as e:
        logger.warning(f"Failed to get OpikConnector: {e}")
        return None


class LearningExtractorNode:
    """
    Extract actionable learnings from detected patterns.
    Generate improvement recommendations.
    """

    def __init__(self, use_llm: bool = False, llm: Optional[Any] = None):
        """
        Initialize learning extractor.

        Args:
            use_llm: Whether to use LLM for extraction
            llm: Optional LLM instance
        """
        self.use_llm = use_llm
        self.llm = llm

    async def execute(self, state: FeedbackLearnerState) -> FeedbackLearnerState:
        """Execute learning extraction."""
        start_time = time.time()

        # Check if already failed
        if state.get("status") == "failed":
            return state

        try:
            patterns = state.get("detected_patterns") or []

            if not patterns:
                return {
                    **state,
                    "learning_recommendations": [],
                    "priority_improvements": [],
                    "extraction_latency_ms": 0,
                    "status": "updating",
                }

            # Extract learnings
            if self.use_llm and self.llm:
                result = await self._extract_with_llm(state)
            else:
                result = self._extract_deterministic(state)

            extraction_time = int((time.time() - start_time) * 1000)

            logger.info(
                f"Learning extraction complete: {len(result['recommendations'])} recommendations"
            )

            return {
                **state,
                "learning_recommendations": result["recommendations"],
                "priority_improvements": result["priorities"],
                "extraction_latency_ms": extraction_time,
                "status": "updating",
            }

        except Exception as e:
            logger.error(f"Learning extraction failed: {e}")
            return {
                **state,
                "errors": [{"node": "learning_extractor", "error": str(e)}],
                "status": "failed",
            }

    def _extract_deterministic(self, state: FeedbackLearnerState) -> Dict[str, Any]:
        """Deterministic learning extraction using heuristics."""
        patterns = state.get("detected_patterns") or []

        recommendations: List[LearningRecommendation] = []
        rec_id = 1

        for pattern in patterns:
            pattern_type = pattern["pattern_type"]
            severity = pattern["severity"]
            affected_agents = pattern["affected_agents"]

            # Generate recommendations based on pattern type
            if pattern_type == "accuracy_issue":
                recommendations.append(
                    LearningRecommendation(
                        recommendation_id=f"R{rec_id}",
                        category="data_update",
                        description=f"Review and update training data for {', '.join(affected_agents)}",
                        affected_agents=affected_agents,
                        expected_impact="Improved accuracy and reduced errors",
                        implementation_effort="medium",
                        priority=1 if severity in ["high", "critical"] else 2,
                        proposed_change="Update knowledge base with corrected information",
                    )
                )
                rec_id += 1

                if severity in ["high", "critical"]:
                    recommendations.append(
                        LearningRecommendation(
                            recommendation_id=f"R{rec_id}",
                            category="model_retrain",
                            description=f"Consider model retraining for {', '.join(affected_agents)}",
                            affected_agents=affected_agents,
                            expected_impact="Significant accuracy improvement",
                            implementation_effort="high",
                            priority=2,
                            proposed_change=None,
                        )
                    )
                    rec_id += 1

            elif pattern_type == "latency_issue":
                recommendations.append(
                    LearningRecommendation(
                        recommendation_id=f"R{rec_id}",
                        category="config_change",
                        description="Optimize agent response latency",
                        affected_agents=affected_agents,
                        expected_impact="Faster response times",
                        implementation_effort="low",
                        priority=2,
                        proposed_change="Increase timeout thresholds or optimize query processing",
                    )
                )
                rec_id += 1

            elif pattern_type == "relevance_issue":
                recommendations.append(
                    LearningRecommendation(
                        recommendation_id=f"R{rec_id}",
                        category="prompt_update",
                        description=f"Improve response relevance for {', '.join(affected_agents)}",
                        affected_agents=affected_agents,
                        expected_impact="More relevant and focused responses",
                        implementation_effort="medium",
                        priority=1 if severity in ["high", "critical"] else 2,
                        proposed_change="Update system prompts to better guide response generation",
                    )
                )
                rec_id += 1

            elif pattern_type == "format_issue":
                recommendations.append(
                    LearningRecommendation(
                        recommendation_id=f"R{rec_id}",
                        category="prompt_update",
                        description="Improve response formatting and clarity",
                        affected_agents=affected_agents,
                        expected_impact="Clearer and better structured responses",
                        implementation_effort="low",
                        priority=3,
                        proposed_change="Add formatting guidelines to prompts",
                    )
                )
                rec_id += 1

            elif pattern_type == "coverage_gap":
                recommendations.append(
                    LearningRecommendation(
                        recommendation_id=f"R{rec_id}",
                        category="new_capability",
                        description="Expand knowledge coverage in identified areas",
                        affected_agents=affected_agents,
                        expected_impact="Broader query coverage",
                        implementation_effort="high",
                        priority=2,
                        proposed_change="Add new training data for uncovered topics",
                    )
                )
                rec_id += 1

        # Prioritize recommendations
        priorities = self._prioritize(recommendations)

        return {
            "recommendations": recommendations,
            "priorities": priorities,
        }

    async def _extract_with_llm(self, state: FeedbackLearnerState) -> Dict[str, Any]:
        """Use LLM for sophisticated learning extraction."""
        if not self.llm:
            return self._extract_deterministic(state)

        try:
            prompt = self._build_extraction_prompt(state)

            # Get OpikConnector for LLM call tracing
            opik = _get_opik_connector()
            model_name = getattr(self.llm, "model", "claude")

            if opik and opik.is_enabled:
                # Trace the LLM call
                async with opik.trace_llm_call(
                    model=model_name,
                    provider="anthropic",
                    prompt_template="learning_extraction",
                    input_data={"prompt": prompt[:500]},
                    metadata={"agent": "feedback_learner", "operation": "learning_extraction"},
                ) as llm_span:
                    response = await self.llm.ainvoke(prompt)
                    # Log tokens from response metadata
                    usage = response.response_metadata.get("usage", {})
                    llm_span.log_tokens(
                        input_tokens=usage.get("input_tokens", 0),
                        output_tokens=usage.get("output_tokens", 0),
                    )
            else:
                # Fallback: no tracing
                response = await self.llm.ainvoke(prompt)

            recommendations = self._parse_recommendations(response.content)
            priorities = self._prioritize(recommendations)

            return {
                "recommendations": recommendations,
                "priorities": priorities,
            }
        except Exception as e:
            logger.warning(f"LLM extraction failed, using fallback: {e}")
            return self._extract_deterministic(state)

    def _build_extraction_prompt(self, state: FeedbackLearnerState) -> str:
        """Build learning extraction prompt for LLM."""
        patterns = state.get("detected_patterns") or []

        patterns_str = "\n\n".join(
            [
                f"**{p['pattern_id']}**: {p['description']}\n"
                f"Type: {p['pattern_type']}, Severity: {p['severity']}\n"
                f"Affected agents: {', '.join(p['affected_agents'])}\n"
                f"Root cause: {p['root_cause_hypothesis']}"
                for p in patterns
            ]
        )

        return f"""Generate improvement recommendations for these patterns.

## Detected Patterns

{patterns_str}

---

Categories: prompt_update, model_retrain, data_update, config_change, new_capability

Output JSON:
```json
{{
  "recommendations": [
    {{
      "recommendation_id": "R1",
      "category": "...",
      "description": "...",
      "affected_agents": ["..."],
      "expected_impact": "...",
      "implementation_effort": "low|medium|high",
      "priority": 1-5,
      "proposed_change": "..."
    }}
  ]
}}
```"""

    def _parse_recommendations(self, content: str) -> List[LearningRecommendation]:
        """Parse recommendations from response."""
        json_match = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                recommendations = []
                for r in data.get("recommendations", []):
                    recommendations.append(
                        LearningRecommendation(
                            recommendation_id=r.get("recommendation_id", "R?"),
                            category=r.get("category", "data_update"),
                            description=r.get("description", ""),
                            affected_agents=r.get("affected_agents", []),
                            expected_impact=r.get("expected_impact", ""),
                            implementation_effort=r.get("implementation_effort", "medium"),
                            priority=r.get("priority", 3),
                            proposed_change=r.get("proposed_change"),
                        )
                    )
                return recommendations
            except (json.JSONDecodeError, TypeError):
                pass

        return []

    def _prioritize(self, recommendations: List[LearningRecommendation]) -> List[str]:
        """Get prioritized list of improvements."""
        effort_order = {"low": 1, "medium": 2, "high": 3}

        sorted_recs = sorted(
            recommendations,
            key=lambda r: (
                r["priority"],
                effort_order.get(r["implementation_effort"], 2),
            ),
        )

        return [r["description"] for r in sorted_recs[:5]]
