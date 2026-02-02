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

        # P2 Enhancement: Generate recommendations from findings
        # This ensures we always have actionable recommendations,
        # even when upstream agents don't provide explicit ones
        generated_recs = self._generate_recommendations(insights, analysis_context)
        if generated_recs:
            logger.info(f"Generated {len(generated_recs)} recommendations from findings")
            insights.extend(generated_recs)

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

Target Audience: {state.get("user_expertise", "analyst")}
Focus Areas: {state.get("focus_areas", ["General overview"])}

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

    def _generate_recommendations(
        self, insights: List[Insight], context: List[Dict[str, Any]]
    ) -> List[Insight]:
        """Generate actionable recommendations from findings.

        Transforms findings into recommendations when upstream agents
        don't provide explicit recommendations.

        Args:
            insights: List of extracted insights
            context: Original analysis context from agents

        Returns:
            List of generated recommendation insights
        """
        recommendations = []
        rec_id_start = len(insights) + 1

        # First, check if we already have recommendations
        existing_recs = [i for i in insights if i["category"] == "recommendation"]
        if len(existing_recs) >= 3:
            # Already have enough recommendations
            return []

        # Transform high-priority findings into recommendations
        findings = [i for i in insights if i["category"] == "finding"]
        opportunities = [i for i in insights if i["category"] == "opportunity"]

        for insight in findings + opportunities:
            rec = self._finding_to_recommendation(insight, context, rec_id_start)
            if rec:
                recommendations.append(rec)
                rec_id_start += 1

        # Prioritize and deduplicate
        recommendations = self._prioritize_recommendations(recommendations)

        # Limit to 5 generated recommendations
        return recommendations[:5]

    def _finding_to_recommendation(
        self, insight: Insight, context: List[Dict[str, Any]], rec_id: int
    ) -> Optional[Insight]:
        """Transform a finding into an actionable recommendation.

        Uses pattern matching to generate appropriate action statements
        based on the finding content.

        Args:
            insight: The finding insight to transform
            context: Original analysis context for additional info
            rec_id: ID to assign to the recommendation

        Returns:
            Recommendation insight or None if not transformable
        """
        statement = insight["statement"].lower()
        original_statement = insight["statement"]
        confidence = insight["confidence"]
        priority = insight["priority"]

        # Pattern: Improvement/increase findings
        if any(word in statement for word in ["improvement", "increase", "higher", "better"]):
            # Extract what improved if possible
            action = self._extract_improvement_action(original_statement)
            if action:
                return Insight(
                    insight_id=f"rec_{rec_id}",
                    category="recommendation",
                    statement=action,
                    supporting_evidence=[f"Based on: {original_statement}"],
                    confidence=confidence * 0.9,  # Slightly lower confidence for generated
                    priority=priority,
                    actionability="short_term",
                )

        # Pattern: Performance/efficiency findings
        if any(word in statement for word in ["performance", "efficiency", "optimize", "latency"]):
            return Insight(
                insight_id=f"rec_{rec_id}",
                category="recommendation",
                statement="ACTION: Investigate and scale the factors driving this performance pattern",
                supporting_evidence=[f"Based on: {original_statement}"],
                confidence=confidence * 0.85,
                priority=priority,
                actionability="short_term",
            )

        # Pattern: Segment/cohort findings
        if any(word in statement for word in ["segment", "cohort", "group", "subset"]):
            return Insight(
                insight_id=f"rec_{rec_id}",
                category="recommendation",
                statement="ACTION: Develop targeted strategy for this segment based on identified characteristics",
                supporting_evidence=[f"Based on: {original_statement}"],
                confidence=confidence * 0.85,
                priority=priority,
                actionability="short_term",
            )

        # Pattern: Causal/correlation findings
        if any(word in statement for word in ["causal", "effect", "impact", "drives", "caused by"]):
            return Insight(
                insight_id=f"rec_{rec_id}",
                category="recommendation",
                statement="ACTION: Design intervention targeting the identified causal mechanism",
                supporting_evidence=[f"Based on: {original_statement}"],
                confidence=confidence * 0.8,
                priority=priority,
                actionability="long_term",
            )

        # Pattern: Risk/decline findings
        if any(word in statement for word in ["risk", "decline", "decrease", "drop", "churn"]):
            return Insight(
                insight_id=f"rec_{rec_id}",
                category="recommendation",
                statement="ACTION: Implement mitigation strategy to address identified risk factors",
                supporting_evidence=[f"Based on: {original_statement}"],
                confidence=confidence * 0.9,
                priority=max(1, priority - 1),  # Increase priority for risks
                actionability="immediate",
            )

        # Pattern: Opportunity findings
        if any(word in statement for word in ["opportunity", "potential", "could", "possible"]):
            return Insight(
                insight_id=f"rec_{rec_id}",
                category="recommendation",
                statement="ACTION: Evaluate and prioritize this opportunity for resource allocation",
                supporting_evidence=[f"Based on: {original_statement}"],
                confidence=confidence * 0.85,
                priority=priority,
                actionability="short_term",
            )

        # Pattern: Significant/notable findings (generic but important)
        if any(word in statement for word in ["significant", "notable", "important", "key"]):
            return Insight(
                insight_id=f"rec_{rec_id}",
                category="recommendation",
                statement="ACTION: Conduct deeper analysis to determine actionable next steps",
                supporting_evidence=[f"Based on: {original_statement}"],
                confidence=confidence * 0.7,
                priority=priority + 1,  # Lower priority for generic
                actionability="long_term",
            )

        # No pattern matched - don't generate recommendation
        return None

    def _extract_improvement_action(self, statement: str) -> Optional[str]:
        """Extract specific action from improvement finding."""
        statement_lower = statement.lower()

        # Look for percentage improvements
        import re

        pct_match = re.search(r"(\d+(?:\.\d+)?)\s*%", statement)
        if pct_match:
            pct = pct_match.group(1)
            if "territory" in statement_lower or "region" in statement_lower:
                return f"ACTION: Expand successful tactics from high-performing territories showing {pct}% improvement"
            elif "conversion" in statement_lower:
                return f"ACTION: Scale conversion optimization tactics that achieved {pct}% improvement"
            elif "response" in statement_lower or "engagement" in statement_lower:
                return f"ACTION: Replicate engagement strategy showing {pct}% response improvement"
            else:
                return f"ACTION: Investigate and scale the intervention driving {pct}% improvement"

        # Generic improvement action
        if "improvement" in statement_lower:
            return "ACTION: Analyze root causes of improvement and develop scaling strategy"

        return None

    def _prioritize_recommendations(self, recommendations: List[Insight]) -> List[Insight]:
        """Prioritize and deduplicate recommendations.

        Args:
            recommendations: List of generated recommendations

        Returns:
            Sorted and deduplicated list
        """
        if not recommendations:
            return []

        # Remove duplicates by comparing statements (fuzzy)
        seen_patterns = set()
        unique_recs = []

        for rec in recommendations:
            # Create pattern signature from first few words
            words = rec["statement"].split()[:4]
            pattern = " ".join(words).lower()

            if pattern not in seen_patterns:
                seen_patterns.add(pattern)
                unique_recs.append(rec)

        # Sort by priority (ascending - 1 is highest) and confidence (descending)
        unique_recs.sort(key=lambda x: (x["priority"], -x["confidence"]))

        return unique_recs
