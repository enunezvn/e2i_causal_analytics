"""
E2I Explainer Agent - Narrative Generator Node
Version: 4.2
Purpose: Generate final narrative explanations

Memory Integration:
- Working Memory (Redis): Cache generated explanations with 24h TTL
- Episodic Memory (Supabase): Store explanations for future retrieval
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ..state import ExplainerState, Insight, NarrativeSection

if TYPE_CHECKING:
    from ..memory_hooks import ExplanationMemoryHooks

logger = logging.getLogger(__name__)


class NarrativeGeneratorNode:
    """
    Generate final narrative explanations.
    Uses structured insights from reasoning phase.

    Memory Integration:
    - Caches generated explanations in working memory (24h TTL)
    - Stores explanations in episodic memory for future retrieval
    """

    def __init__(
        self,
        use_llm: bool = False,
        llm: Optional[Any] = None,
        memory_hooks: Optional["ExplanationMemoryHooks"] = None,
    ):
        """
        Initialize narrative generator.

        Args:
            use_llm: Whether to use LLM for generation
            llm: Optional LLM instance to use
            memory_hooks: Memory hooks for tri-memory integration
        """
        self.use_llm = use_llm
        self.llm = llm
        self._memory_hooks = memory_hooks

    @property
    def memory_hooks(self) -> Optional["ExplanationMemoryHooks"]:
        """Lazy-load memory hooks if not provided."""
        if self._memory_hooks is None:
            try:
                from ..memory_hooks import get_explanation_memory_hooks

                self._memory_hooks = get_explanation_memory_hooks()
            except Exception as e:
                logger.warning(f"Failed to initialize memory hooks: {e}")
                return None
        return self._memory_hooks

    async def execute(self, state: ExplainerState) -> ExplainerState:
        """Execute narrative generation."""
        start_time = time.time()

        # Check if already failed
        if state.get("status") == "failed":
            return state

        try:
            output_format = state.get("output_format", "narrative")

            if output_format == "brief":
                result = self._generate_brief(state)
            elif output_format == "structured":
                result = self._generate_structured(state)
            elif output_format == "presentation":
                result = self._generate_presentation(state)
            else:
                result = self._generate_narrative(state)

            # Generate supplementary content
            visuals = self._suggest_visuals(state)
            follow_ups = self._generate_follow_ups(state)

            generation_time = int((time.time() - start_time) * 1000)
            total_time = (
                state.get("assembly_latency_ms", 0)
                + state.get("reasoning_latency_ms", 0)
                + generation_time
            )

            logger.info(
                f"Narrative generated: format={output_format}, "
                f"sections={len(result.get('narrative_sections', []))}"
            )

            # === MEMORY STORAGE ===
            session_id = state.get("session_id")
            if session_id and self.memory_hooks:
                await self._store_explanation_in_memory(
                    session_id=session_id,
                    state=state,
                    result=result,
                )

            return {
                **state,
                **result,
                "visual_suggestions": visuals,
                "follow_up_questions": follow_ups,
                "generation_latency_ms": generation_time,
                "total_latency_ms": total_time,
                "status": "completed",
            }

        except Exception as e:
            logger.error(f"Narrative generation failed: {e}")
            return {
                **state,
                "errors": [{"node": "narrative_generator", "error": str(e)}],
                "status": "failed",
            }

    def _generate_narrative(self, state: ExplainerState) -> Dict[str, Any]:
        """Generate full narrative explanation."""
        insights = state.get("extracted_insights") or []
        state.get("narrative_structure") or []
        themes = state.get("key_themes") or []
        expertise = state.get("user_expertise", "analyst")

        sections = []

        # Executive Summary
        exec_summary = self._create_executive_summary(insights, themes, expertise)
        sections.append(
            NarrativeSection(
                section_type="summary",
                title="Executive Summary",
                content=exec_summary,
                supporting_data=None,
            )
        )

        # Key Findings section
        findings = [i for i in insights if i.get("category") == "finding"]
        if findings:
            sections.append(
                NarrativeSection(
                    section_type="findings",
                    title="Key Findings",
                    content=self._format_insights_section(findings, expertise),
                    supporting_data={"count": len(findings)},
                )
            )

        # Recommendations section
        recommendations = [i for i in insights if i.get("category") == "recommendation"]
        if recommendations:
            sections.append(
                NarrativeSection(
                    section_type="recommendations",
                    title="Recommendations",
                    content=self._format_insights_section(recommendations, expertise),
                    supporting_data={"count": len(recommendations)},
                )
            )

        # Warnings/Caveats section
        warnings = [i for i in insights if i.get("category") == "warning"]
        if warnings:
            sections.append(
                NarrativeSection(
                    section_type="caveats",
                    title="Considerations & Caveats",
                    content=self._format_insights_section(warnings, expertise),
                    supporting_data={"count": len(warnings)},
                )
            )

        # Opportunities section
        opportunities = [i for i in insights if i.get("category") == "opportunity"]
        if opportunities:
            sections.append(
                NarrativeSection(
                    section_type="opportunities",
                    title="Opportunities",
                    content=self._format_insights_section(opportunities, expertise),
                    supporting_data={"count": len(opportunities)},
                )
            )

        # Next Steps
        next_steps = self._create_next_steps(insights, expertise)
        sections.append(
            NarrativeSection(
                section_type="next_steps",
                title="Next Steps",
                content=next_steps,
                supporting_data=None,
            )
        )

        # Combine into detailed explanation
        detailed = self._combine_sections(sections)

        return {
            "executive_summary": exec_summary,
            "detailed_explanation": detailed,
            "narrative_sections": sections,
        }

    def _generate_brief(self, state: ExplainerState) -> Dict[str, Any]:
        """Generate brief summary."""
        insights = state.get("extracted_insights") or []
        top_insights = sorted(insights, key=lambda x: x.get("priority", 5))[:3]

        brief = "**Key Findings:**\n\n"
        for insight in top_insights:
            brief += f"- {insight['statement']}\n"

        return {
            "executive_summary": brief,
            "detailed_explanation": brief,
            "narrative_sections": [],
        }

    def _generate_structured(self, state: ExplainerState) -> Dict[str, Any]:
        """Generate structured output with clear sections."""
        insights = state.get("extracted_insights") or []

        sections = []

        # Group insights by category
        categories = {
            "finding": "Key Findings",
            "recommendation": "Recommendations",
            "warning": "Caveats & Considerations",
            "opportunity": "Opportunities",
        }

        for cat_key, cat_title in categories.items():
            cat_insights = [i for i in insights if i.get("category") == cat_key]
            if cat_insights:
                sections.append(
                    NarrativeSection(
                        section_type=cat_key,
                        title=cat_title,
                        content="\n".join(f"- {i['statement']}" for i in cat_insights),
                        supporting_data={"count": len(cat_insights)},
                    )
                )

        # Count summary
        finding_count = len([i for i in insights if i.get("category") == "finding"])
        rec_count = len([i for i in insights if i.get("category") == "recommendation"])

        combined = "\n\n".join(f"## {s['title']}\n{s['content']}" for s in sections)

        return {
            "executive_summary": f"Analysis complete with {finding_count} findings and {rec_count} recommendations.",
            "detailed_explanation": combined,
            "narrative_sections": sections,
        }

    def _generate_presentation(self, state: ExplainerState) -> Dict[str, Any]:
        """Generate presentation-style output with bullet points."""
        insights = state.get("extracted_insights") or []
        themes = state.get("key_themes") or []

        # Title slide content
        title_content = "**Analysis Results**\n\n"
        title_content += "Key Themes:\n"
        for theme in themes:
            title_content += f"- {theme}\n"

        sections = [
            NarrativeSection(
                section_type="title",
                title="Overview",
                content=title_content,
                supporting_data=None,
            )
        ]

        # Insights slides
        for i, insight in enumerate(insights[:5], 1):
            sections.append(
                NarrativeSection(
                    section_type="insight",
                    title=f"Insight {i}",
                    content=(
                        f"**{insight['statement']}**\n\n"
                        f"Category: {insight['category'].title()}\n"
                        f"Confidence: {insight['confidence']:.0%}\n"
                        f"Actionability: {insight['actionability'].replace('_', ' ').title()}"
                    ),
                    supporting_data={"priority": insight.get("priority", 3)},
                )
            )

        combined = "\n\n---\n\n".join(f"### {s['title']}\n\n{s['content']}" for s in sections)

        return {
            "executive_summary": title_content,
            "detailed_explanation": combined,
            "narrative_sections": sections,
        }

    def _create_executive_summary(
        self, insights: List[Insight], themes: List[str], expertise: str
    ) -> str:
        """Create executive summary based on expertise level."""
        finding_count = len([i for i in insights if i.get("category") == "finding"])
        rec_count = len([i for i in insights if i.get("category") == "recommendation"])
        warning_count = len([i for i in insights if i.get("category") == "warning"])

        # Get top priority insights
        top_insights = sorted(insights, key=lambda x: x.get("priority", 5))[:3]

        if expertise == "executive":
            summary = f"This analysis identified {finding_count} key findings"
            if rec_count > 0:
                summary += f" with {rec_count} actionable recommendations"
            summary += ".\n\n"

            summary += "**Bottom Line:**\n"
            for insight in top_insights:
                summary += f"- {insight['statement']}\n"

        elif expertise == "data_scientist":
            summary = f"Analysis yielded {finding_count} findings, {rec_count} recommendations"
            if warning_count > 0:
                summary += f", and {warning_count} methodological considerations"
            summary += ".\n\n"

            # Include confidence assessment
            avg_conf = sum(i.get("confidence", 0.5) for i in insights) / max(len(insights), 1)
            summary += f"**Overall confidence:** {avg_conf:.0%}\n\n"

            summary += "**Key Results:**\n"
            for insight in top_insights:
                summary += f"- {insight['statement']} (conf: {insight['confidence']:.0%})\n"

        else:  # analyst
            summary = f"Analysis complete with {finding_count} findings"
            if rec_count > 0:
                summary += f" and {rec_count} recommendations"
            summary += ".\n\n"

            summary += "**Highlights:**\n"
            for insight in top_insights:
                summary += f"- {insight['statement']}\n"

        return summary

    def _format_insights_section(self, insights: List[Insight], expertise: str) -> str:
        """Format insights for a section."""
        lines = []

        for insight in insights:
            if expertise == "executive":
                lines.append(f"- {insight['statement']}")
            elif expertise == "data_scientist":
                lines.append(
                    f"- {insight['statement']} "
                    f"(confidence: {insight['confidence']:.0%}, "
                    f"priority: {insight['priority']})"
                )
            else:
                lines.append(
                    f"- {insight['statement']} "
                    f"[{insight['actionability'].replace('_', ' ').title()}]"
                )

        return "\n".join(lines)

    def _create_next_steps(self, insights: List[Insight], expertise: str) -> str:
        """Create next steps section."""
        # Filter to actionable insights
        immediate = [i for i in insights if i.get("actionability") == "immediate"]
        short_term = [i for i in insights if i.get("actionability") == "short_term"]

        steps = []

        if immediate:
            steps.append("**Immediate Actions:**")
            for i in immediate[:3]:
                steps.append(f"- {i['statement']}")
            steps.append("")

        if short_term:
            steps.append("**Short-Term Actions:**")
            for i in short_term[:3]:
                steps.append(f"- {i['statement']}")

        if not steps:
            steps.append(
                "Review the findings above and determine appropriate actions based on your strategic priorities."
            )

        return "\n".join(steps)

    def _combine_sections(self, sections: List[NarrativeSection]) -> str:
        """Combine sections into full narrative."""
        parts = []
        for section in sections:
            parts.append(f"## {section['title']}\n\n{section['content']}")
        return "\n\n".join(parts)

    def _suggest_visuals(self, state: ExplainerState) -> List[Dict[str, Any]]:
        """Suggest visualizations based on analysis."""
        suggestions = []

        for ctx in state.get("analysis_context") or []:
            analysis_type = ctx.get("analysis_type", "")

            if "causal" in analysis_type.lower():
                suggestions.append(
                    {
                        "type": "effect_plot",
                        "title": "Causal Effect Estimate",
                        "description": "Bar chart showing treatment effect with confidence interval",
                    }
                )
            elif "roi" in analysis_type.lower() or "gap" in analysis_type.lower():
                suggestions.append(
                    {
                        "type": "opportunity_matrix",
                        "title": "ROI Opportunity Matrix",
                        "description": "Scatter plot of ROI potential vs. implementation effort",
                    }
                )
            elif "segment" in analysis_type.lower() or "heterogeneous" in analysis_type.lower():
                suggestions.append(
                    {
                        "type": "segment_effects",
                        "title": "Effects by Segment",
                        "description": "Grouped bar chart showing effects by segment",
                    }
                )
            elif "trend" in analysis_type.lower() or "time" in analysis_type.lower():
                suggestions.append(
                    {
                        "type": "trend_line",
                        "title": "Trend Analysis",
                        "description": "Line chart showing trend over time",
                    }
                )

        # Default suggestion if none matched
        if not suggestions:
            suggestions.append(
                {
                    "type": "summary_table",
                    "title": "Key Metrics Summary",
                    "description": "Table summarizing key metrics from the analysis",
                }
            )

        return suggestions[:3]  # Return top 3 suggestions

    def _generate_follow_ups(self, state: ExplainerState) -> List[str]:
        """Generate follow-up questions."""
        follow_ups = []
        insights = state.get("extracted_insights") or []

        # Based on insight categories
        has_finding = any(i.get("category") == "finding" for i in insights)
        has_recommendation = any(i.get("category") == "recommendation" for i in insights)
        has_warning = any(i.get("category") == "warning" for i in insights)
        has_opportunity = any(i.get("category") == "opportunity" for i in insights)

        if has_finding:
            follow_ups.append("What's driving these findings at the segment level?")

        if has_recommendation:
            follow_ups.append("How do we prioritize these recommendations?")
            follow_ups.append("What resources are needed to implement these changes?")

        if has_warning:
            follow_ups.append("What additional data would strengthen confidence in these results?")

        if has_opportunity:
            follow_ups.append("What's the expected ROI for pursuing these opportunities?")

        # Default questions
        if not follow_ups:
            follow_ups = [
                "Can you explain the methodology behind this analysis?",
                "What are the key assumptions in this analysis?",
                "How confident are we in these results?",
            ]

        return follow_ups[:5]

    # =========================================================================
    # MEMORY STORAGE
    # =========================================================================

    async def _store_explanation_in_memory(
        self,
        session_id: str,
        state: ExplainerState,
        result: Dict[str, Any],
    ) -> None:
        """Store generated explanation in memory systems.

        Args:
            session_id: Session identifier for memory correlation
            state: Current explainer state
            result: Generated narrative result

        Memory Storage:
        - Working Memory (Redis): 24-hour cache for quick retrieval
        - Episodic Memory (Supabase): Permanent storage with embeddings
        """
        if not self.memory_hooks:
            return

        try:
            # Build explanation data
            explanation_data = {
                "query": state.get("query", ""),
                "executive_summary": result.get("executive_summary", ""),
                "detailed_explanation": result.get("detailed_explanation", ""),
                "insights": state.get("extracted_insights", []),
                "audience": state.get("user_expertise", "analyst"),
                "output_format": state.get("output_format", "narrative"),
                "themes": state.get("key_themes", []),
                "visual_suggestions": result.get("visual_suggestions", []),
            }

            # Cache in working memory (24h TTL)
            cache_success = await self.memory_hooks.cache_explanation(
                session_id=session_id,
                explanation=explanation_data,
            )

            # Store in episodic memory for future retrieval
            brand = state.get("memory_config", {}).get("brand")
            region = state.get("memory_config", {}).get("region")

            memory_id = await self.memory_hooks.store_explanation(
                session_id=session_id,
                explanation=explanation_data,
                brand=brand,
                region=region,
            )

            logger.debug(
                f"Explanation stored: cache={cache_success}, "
                f"episodic_id={memory_id}"
            )

        except Exception as e:
            # Non-fatal: log warning but don't fail the generation
            logger.warning(f"Memory storage failed (non-fatal): {e}")
