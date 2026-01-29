"""
E2I Explainer Agent - Deep Reasoner Node
Version: 4.2
Purpose: Extended reasoning for insight extraction and narrative planning
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

from ..state import ExplainerState, Insight

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


class DeepReasonerNode:
    """
    Deep reasoning for insight extraction and narrative planning.

    Can operate in two modes:
    - LLM mode: Uses Claude for sophisticated reasoning
    - Deterministic mode: Uses heuristics for fast processing
    """

    def __init__(self, use_llm: bool = False, llm: Optional[Any] = None):
        """
        Initialize deep reasoner.

        Args:
            use_llm: Whether to use LLM for reasoning
            llm: Optional LLM instance to use
        """
        self.use_llm = use_llm
        self.llm = llm

    async def execute(self, state: ExplainerState) -> ExplainerState:
        """Execute deep reasoning."""
        start_time = time.time()

        # Check if already failed
        if state.get("status") == "failed":
            return state

        try:
            analysis_context = state.get("analysis_context", [])

            if not analysis_context:
                return {
                    **state,
                    "errors": [{"node": "deep_reasoner", "error": "No analysis context available"}],
                    "status": "failed",
                }

            # Extract insights
            if self.use_llm and self.llm:
                result = await self._reason_with_llm(state)
            else:
                result = self._reason_deterministic(state)

            reasoning_time = int((time.time() - start_time) * 1000)

            logger.info(
                f"Reasoning complete: {len(result.get('insights', []))} insights, "
                f"{len(result.get('themes', []))} themes"
            )

            return {
                **state,
                "extracted_insights": result.get("insights", []),
                "narrative_structure": result.get("structure", []),
                "key_themes": result.get("themes", []),
                "reasoning_latency_ms": reasoning_time,
                "model_used": result.get("model_used", "deterministic"),
                "status": "generating",
            }

        except Exception as e:
            logger.error(f"Deep reasoning failed: {e}")
            return {
                **state,
                "errors": [{"node": "deep_reasoner", "error": str(e)}],
                "extracted_insights": [],  # Required output default
                "status": "failed",
            }

    def _reason_deterministic(self, state: ExplainerState) -> Dict[str, Any]:
        """Deterministic reasoning using heuristics."""
        analysis_context = state.get("analysis_context") or []
        user_expertise = state.get("user_expertise", "analyst")
        focus_areas = state.get("focus_areas") or []

        insights = []
        insight_id = 1

        # Extract insights from each context
        for ctx in analysis_context:
            findings = ctx.get("key_findings", [])
            source_agent = ctx.get("source_agent", "unknown")
            confidence = ctx.get("confidence", 0.5)

            for finding in findings:
                # Categorize the finding
                category = self._categorize_finding(finding)
                actionability = self._assess_actionability(finding, category)
                priority = self._assess_priority(finding, focus_areas)

                insight = Insight(
                    insight_id=str(insight_id),
                    category=category,
                    statement=finding if isinstance(finding, str) else str(finding),
                    supporting_evidence=[f"Source: {source_agent}"],
                    confidence=confidence,
                    priority=priority,
                    actionability=actionability,
                )
                insights.append(insight)
                insight_id += 1

        # Sort by priority
        insights.sort(key=lambda x: x["priority"])

        # Limit to top insights
        insights = insights[:10]

        # Determine narrative structure based on expertise
        structure = self._determine_structure(user_expertise, insights)

        # Extract key themes
        themes = self._extract_themes(insights)

        return {
            "insights": insights,
            "structure": structure,
            "themes": themes,
            "model_used": "deterministic",
        }

    def _categorize_finding(self, finding: str) -> str:
        """Categorize a finding based on content."""
        finding_lower = finding.lower() if isinstance(finding, str) else ""

        if any(word in finding_lower for word in ["recommend", "should", "suggest", "consider"]):
            return "recommendation"
        elif any(
            word in finding_lower for word in ["warning", "caution", "risk", "concern", "note"]
        ):
            return "warning"
        elif any(
            word in finding_lower
            for word in ["opportunity", "potential", "could increase", "possible gain"]
        ):
            return "opportunity"
        else:
            return "finding"

    def _assess_actionability(self, finding: str, category: str) -> str:
        """Assess actionability of a finding."""
        finding_lower = finding.lower() if isinstance(finding, str) else ""

        if category == "recommendation":
            if any(word in finding_lower for word in ["immediately", "urgent", "now"]):
                return "immediate"
            elif any(word in finding_lower for word in ["soon", "next", "short"]):
                return "short_term"
            else:
                return "long_term"
        elif category == "warning":
            return "immediate"
        elif category == "opportunity":
            return "short_term"
        else:
            return "informational"

    def _assess_priority(self, finding: str, focus_areas: List[str]) -> int:
        """Assess priority of a finding (1=highest, 5=lowest)."""
        finding_lower = finding.lower() if isinstance(finding, str) else ""

        # Check if matches focus areas
        for area in focus_areas:
            if area.lower() in finding_lower:
                return 1

        # Priority based on keywords
        if any(word in finding_lower for word in ["significant", "major", "critical", "important"]):
            return 2
        elif any(word in finding_lower for word in ["moderate", "notable", "substantial"]):
            return 3
        elif any(word in finding_lower for word in ["minor", "small", "slight"]):
            return 4
        else:
            return 3  # Default to moderate priority

    def _determine_structure(self, expertise: str, insights: List[Insight]) -> List[str]:
        """Determine narrative structure based on expertise level."""
        has_recommendations = any(i["category"] == "recommendation" for i in insights)
        has_warnings = any(i["category"] == "warning" for i in insights)

        if expertise == "executive":
            structure = [
                "Executive Summary",
                "Key Findings",
                "Business Impact",
            ]
            if has_recommendations:
                structure.append("Recommended Actions")
            structure.append("Next Steps")
        elif expertise == "data_scientist":
            structure = [
                "Summary",
                "Methodology Overview",
                "Detailed Findings",
                "Statistical Validity",
            ]
            if has_warnings:
                structure.append("Caveats and Limitations")
            if has_recommendations:
                structure.append("Recommendations")
            structure.append("Further Analysis Suggestions")
        else:  # analyst
            structure = [
                "Summary",
                "Key Findings",
            ]
            if has_warnings:
                structure.append("Considerations")
            if has_recommendations:
                structure.append("Recommendations")
            structure.append("Action Items")

        return structure

    def _extract_themes(self, insights: List[Insight]) -> List[str]:
        """Extract key themes from insights."""
        themes = []

        # Count categories
        categories = {}
        for insight in insights:
            cat = insight["category"]
            categories[cat] = categories.get(cat, 0) + 1

        # Generate themes based on category distribution
        if categories.get("finding", 0) > 0:
            themes.append(f"Analysis revealed {categories['finding']} key findings")

        if categories.get("recommendation", 0) > 0:
            themes.append(f"{categories['recommendation']} actionable recommendations identified")

        if categories.get("warning", 0) > 0:
            themes.append(f"{categories['warning']} areas requiring attention")

        if categories.get("opportunity", 0) > 0:
            themes.append(f"{categories['opportunity']} opportunities for improvement")

        # Add confidence-based theme
        avg_confidence = sum(i["confidence"] for i in insights) / max(len(insights), 1)
        if avg_confidence >= 0.8:
            themes.append("High confidence in analysis results")
        elif avg_confidence < 0.5:
            themes.append("Results should be interpreted with caution")

        return themes[:3]  # Return top 3 themes

    async def _reason_with_llm(self, state: ExplainerState) -> Dict[str, Any]:
        """Use LLM for sophisticated reasoning."""
        if not self.llm:
            return self._reason_deterministic(state)

        prompt = self._build_reasoning_prompt(state)

        try:
            # Get OpikConnector for LLM call tracing
            opik = _get_opik_connector()
            model_name = getattr(self.llm, "model", "claude")

            if opik and opik.is_enabled:
                # Trace the LLM call
                async with opik.trace_llm_call(
                    model=model_name,
                    provider="anthropic",
                    prompt_template="deep_reasoning",
                    input_data={"prompt": prompt[:500]},
                    metadata={"agent": "explainer", "operation": "deep_reasoning"},
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

            parsed = self._parse_reasoning(response.content)
            parsed["model_used"] = model_name
            return parsed
        except Exception as e:
            logger.warning(f"LLM reasoning failed, using fallback: {e}")
            return self._reason_deterministic(state)

    def _build_reasoning_prompt(self, state: ExplainerState) -> str:
        """Build reasoning prompt for LLM."""
        contexts_str = "\n\n".join(
            [
                f"### Analysis from {ctx['source_agent']}\n"
                f"Type: {ctx['analysis_type']}\n"
                f"Confidence: {ctx['confidence']}\n"
                f"Key Findings:\n" + "\n".join(f"- {f}" for f in ctx["key_findings"])
                for ctx in state.get("analysis_context", [])
            ]
        )

        return f"""Analyze these results and extract insights:

{contexts_str}

Target Audience: {state.get('user_expertise', 'analyst')}
Focus Areas: {state.get('focus_areas', ['General overview'])}

Provide JSON with:
- insights: List of insights with statement, category, confidence, priority
- structure: Suggested narrative sections
- themes: 2-3 overarching themes

```json
{{
  "insights": [...],
  "structure": [...],
  "themes": [...]
}}
```"""

    def _parse_reasoning(self, content: str) -> Dict[str, Any]:
        """Parse LLM reasoning output."""
        json_match = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Fallback: return empty structure
        return {
            "insights": [],
            "structure": ["Summary", "Findings", "Recommendations"],
            "themes": [],
        }
