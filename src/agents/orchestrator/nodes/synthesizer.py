"""Synthesizer node for orchestrator agent.

Combine multi-agent results into coherent response.
Uses fast model for synthesis.
"""

import logging
import time
from typing import Any, Dict, List

from src.utils.llm_factory import get_fast_llm, get_llm_provider

from ..state import AgentResult, OrchestratorState

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


class SynthesizerNode:
    """Combine multi-agent results into coherent response.

    Uses fast model (Haiku) for synthesis.
    """

    def __init__(self):
        """Initialize synthesizer with fast LLM for synthesis."""
        # Use fast LLM (Haiku or gpt-4o-mini based on LLM_PROVIDER env var)
        self.llm = get_fast_llm(max_tokens=1024, timeout=5)
        self._provider = get_llm_provider()

    async def execute(self, state: OrchestratorState) -> OrchestratorState:
        """Execute response synthesis.

        Args:
            state: Current orchestrator state

        Returns:
            Updated state with synthesized response
        """
        start_time = time.time()

        results = state.get("agent_results", [])
        successful_results = [r for r in results if r.get("success")]
        failed_results = [r for r in results if not r.get("success")]

        # Single agent - return directly
        if len(successful_results) == 1:
            synthesized = self._extract_response(successful_results[0])
        elif len(successful_results) > 1:
            # Multi-agent synthesis
            synthesized = await self._synthesize_multiple(successful_results)
        else:
            # All failed
            synthesized = self._generate_error_response(failed_results)

        synthesis_time = int((time.time() - start_time) * 1000)
        total_latency = (
            state.get("classification_latency_ms", 0)
            + state.get("rag_latency_ms", 0)
            + state.get("routing_latency_ms", 0)
            + state.get("dispatch_latency_ms", 0)
            + synthesis_time
        )

        return {
            **state,
            "synthesized_response": synthesized["response"],
            "response_confidence": synthesized.get("confidence", 0.5),
            "recommendations": synthesized.get("recommendations", []),
            "follow_up_suggestions": synthesized.get("follow_ups", []),
            "synthesis_latency_ms": synthesis_time,
            "total_latency_ms": total_latency,
            "current_phase": "complete",
            "status": "completed" if successful_results else "failed",
        }

    def _extract_response(self, result: AgentResult) -> Dict[str, Any]:
        """Extract response from single agent result.

        Args:
            result: Agent result

        Returns:
            Extracted response
        """
        agent_output = result.get("result", {})

        return {
            "response": agent_output.get(
                "narrative", agent_output.get("response", str(agent_output))
            ),
            "confidence": agent_output.get("confidence", 0.5),
            "recommendations": agent_output.get("recommendations", []),
            "follow_ups": agent_output.get("follow_up_suggestions", []),
        }

    async def _synthesize_multiple(self, results: List[AgentResult]) -> Dict[str, Any]:
        """Synthesize multiple agent results.

        Args:
            results: List of successful agent results

        Returns:
            Synthesized response
        """
        summaries = []
        all_recommendations = []
        confidences = []

        for result in results:
            agent_output = result.get("result", {})
            narrative = agent_output.get("narrative", "")[:500]
            summaries.append(f"**{result['agent_name']}**: {narrative}")
            all_recommendations.extend(agent_output.get("recommendations", []))
            confidences.append(agent_output.get("confidence", 0.5))

        synthesis_prompt = f"""Synthesize these analysis results into a coherent response.
Be concise and actionable.

{chr(10).join(summaries)}

Provide a unified 2-3 paragraph response that:
1. Highlights the key findings
2. Connects insights across analyses
3. Provides clear recommendations"""

        try:
            # Get OpikConnector for LLM call tracing
            opik = _get_opik_connector()

            if opik and opik.is_enabled:
                # Trace the LLM call with dynamic provider info
                model_name = "gpt-4o-mini" if self._provider == "openai" else "claude-haiku-4-20250414"
                async with opik.trace_llm_call(
                    model=model_name,
                    provider=self._provider,
                    prompt_template="multi_agent_synthesis",
                    input_data={"summaries": summaries, "prompt": synthesis_prompt},
                    metadata={"agent": "orchestrator", "operation": "response_synthesis"},
                ) as llm_span:
                    response = await self.llm.ainvoke(synthesis_prompt)
                    # Log tokens from response metadata
                    usage = response.response_metadata.get("usage", {})
                    llm_span.log_tokens(
                        input_tokens=usage.get("input_tokens", 0),
                        output_tokens=usage.get("output_tokens", 0),
                    )
            else:
                # Fallback: no tracing
                response = await self.llm.ainvoke(synthesis_prompt)

            # Calculate weighted confidence (round to avoid floating point precision issues)
            avg_confidence = round(sum(confidences) / len(confidences) if confidences else 0.5, 2)

            return {
                "response": response.content,
                "confidence": avg_confidence,
                "recommendations": all_recommendations[:5],
                "follow_ups": [
                    "Explore segment-specific effects",
                    "Design follow-up experiment",
                ],
            }
        except Exception:
            # Fallback: concatenate responses
            return {
                "response": "\n\n".join([s.split(": ", 1)[1] for s in summaries]),
                "confidence": round(sum(confidences) / len(confidences) if confidences else 0.5, 2),
                "recommendations": all_recommendations[:5],
                "follow_ups": [],
            }

    def _generate_error_response(self, failed_results: List[AgentResult]) -> Dict[str, Any]:
        """Generate error response when all agents fail.

        Args:
            failed_results: List of failed agent results

        Returns:
            Error response
        """
        errors = [f"- {r['agent_name']}: {r.get('error')}" for r in failed_results]

        return {
            "response": f"I was unable to complete the analysis due to the following errors:\n{chr(10).join(errors)}\n\nPlease try again or rephrase your question.",
            "confidence": 0.0,
            "recommendations": [],
            "follow_ups": ["Simplify your question", "Check system health"],
        }


# Export for use in graph
async def synthesize_response(state: Dict[str, Any]) -> Dict[str, Any]:
    """Node function for response synthesis.

    Args:
        state: Current state

    Returns:
        Updated state
    """
    synthesizer = SynthesizerNode()
    return await synthesizer.execute(state)
