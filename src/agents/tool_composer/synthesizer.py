"""
E2I Tool Composer - Phase 4: Synthesizer
Version: 4.2
Purpose: Synthesize tool outputs into a coherent natural language response
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .models.composition_models import (
    ComposedResponse,
    DecompositionResult,
    ExecutionTrace,
    SynthesisInput,
)

logger = logging.getLogger(__name__)


# ============================================================================
# SYNTHESIS PROMPT
# ============================================================================

SYNTHESIS_SYSTEM_PROMPT = """You are a pharmaceutical analytics response synthesizer.

Your task is to combine the results from multiple analytical tools into a single, coherent response that answers the user's original question.

## Guidelines:
1. Address the original question directly
2. Integrate insights from all successful tool outputs
3. Present numerical results with appropriate context and caveats
4. Acknowledge any failed components and their impact
5. Maintain a professional, confident tone
6. Structure the response logically (don't just list results)

## Response Quality:
- Lead with the key insight/answer
- Support with specific data points
- Include confidence levels where relevant
- Note any limitations or caveats
- Suggest follow-up actions if appropriate

## Output Format:
Return a JSON object with:
{{
  "answer": "The synthesized natural language response",
  "confidence": 0.85,  // Overall confidence in the response
  "supporting_data": {{
    "key_metric_1": "value",
    "key_metric_2": "value"
  }},
  "citations": ["step_1", "step_2"],  // Which steps contributed
  "caveats": ["Any important caveats"],
  "failed_components": ["Any sub-questions that couldn't be fully answered"],
  "reasoning": "Your synthesis reasoning"
}}"""


SYNTHESIS_USER_TEMPLATE = """Synthesize the following into a response to the original query:

ORIGINAL QUERY:
{query}

SUB-QUESTIONS AND RESULTS:
{results}

Create a coherent response that directly answers the original query.
Return valid JSON only."""


# ============================================================================
# SYNTHESIZER CLASS
# ============================================================================

class ResponseSynthesizer:
    """
    Synthesizes tool outputs into coherent responses.
    
    This is Phase 4 of the Tool Composer pipeline.
    """
    
    def __init__(
        self,
        llm_client: Any,
        model: str = "claude-sonnet-4-20250514",
        temperature: float = 0.4,
        max_tokens: int = 2000
    ):
        self.llm_client = llm_client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    async def synthesize(
        self,
        synthesis_input: SynthesisInput
    ) -> ComposedResponse:
        """
        Synthesize tool outputs into a response.
        
        Args:
            synthesis_input: Contains original query, decomposition, and execution trace
            
        Returns:
            ComposedResponse with the synthesized answer
        """
        logger.info(f"Synthesizing response for query: {synthesis_input.original_query[:50]}...")
        
        try:
            # Format results for the prompt
            results_text = self._format_results(synthesis_input)
            
            # Call LLM for synthesis
            response = await self._call_llm(
                synthesis_input.original_query,
                results_text
            )
            
            # Parse response
            parsed = self._parse_response(response)
            
            # Build ComposedResponse
            composed = ComposedResponse(
                answer=parsed["answer"],
                confidence=parsed.get("confidence", 0.8),
                supporting_data=parsed.get("supporting_data", {}),
                citations=parsed.get("citations", []),
                caveats=parsed.get("caveats", []),
                failed_components=parsed.get("failed_components", []),
                synthesis_reasoning=parsed.get("reasoning", ""),
                timestamp=datetime.now(timezone.utc)
            )
            
            logger.info(f"Synthesis complete, confidence: {composed.confidence}")
            return composed
            
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            # Return a fallback response
            return self._create_fallback_response(synthesis_input, str(e))
    
    def _format_results(self, synthesis_input: SynthesisInput) -> str:
        """Format execution results for the synthesis prompt"""
        lines = []
        
        # Get sub-questions
        sq_map = {sq.id: sq for sq in synthesis_input.decomposition.sub_questions}
        
        for result in synthesis_input.execution_trace.step_results:
            sq = sq_map.get(result.sub_question_id)
            sq_text = sq.question if sq else "Unknown question"
            
            lines.append(f"## Sub-Question: {sq_text}")
            lines.append(f"Tool: {result.tool_name}")
            lines.append(f"Status: {'SUCCESS' if result.output.is_success else 'FAILED'}")
            
            if result.output.is_success and result.output.result:
                # Format the output nicely
                output_str = json.dumps(result.output.result, indent=2, default=str)
                # Truncate if too long
                if len(output_str) > 1000:
                    output_str = output_str[:1000] + "\n... (truncated)"
                lines.append(f"Output:\n{output_str}")
            elif result.output.error:
                lines.append(f"Error: {result.output.error}")
            
            lines.append("")
        
        return "\n".join(lines)
    
    async def _call_llm(self, query: str, results: str) -> str:
        """Call the LLM for synthesis"""
        user_message = SYNTHESIS_USER_TEMPLATE.format(
            query=query,
            results=results
        )
        
        response = await self.llm_client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=SYNTHESIS_SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": user_message}
            ]
        )
        
        return response.content[0].text
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON from LLM response"""
        if "```json" in response:
            start = response.find("```json") + 7
            end = response.find("```", start)
            response = response[start:end].strip()
        elif "```" in response:
            start = response.find("```") + 3
            end = response.find("```", start)
            response = response[start:end].strip()
        
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse synthesis JSON, using raw response: {e}")
            # Return the raw response as the answer
            return {
                "answer": response,
                "confidence": 0.6,
                "reasoning": "JSON parsing failed, using raw response"
            }
    
    def _create_fallback_response(
        self,
        synthesis_input: SynthesisInput,
        error: str
    ) -> ComposedResponse:
        """Create a fallback response when synthesis fails"""
        # Try to extract key results
        successful_results = [
            r for r in synthesis_input.execution_trace.step_results
            if r.output.is_success
        ]
        
        if successful_results:
            # Build a basic response from successful results
            answer_parts = ["Based on the analysis:"]
            for r in successful_results:
                if r.output.result:
                    # Extract key values
                    result_dict = r.output.result
                    key_values = []
                    for k, v in result_dict.items():
                        if isinstance(v, (int, float)):
                            key_values.append(f"{k}: {v}")
                        elif isinstance(v, str) and len(v) < 100:
                            key_values.append(f"{k}: {v}")
                    if key_values:
                        answer_parts.append(f"- {r.tool_name}: {', '.join(key_values[:3])}")
            
            answer = "\n".join(answer_parts)
        else:
            answer = f"Unable to fully answer the query. Error: {error}"
        
        return ComposedResponse(
            answer=answer,
            confidence=0.3,
            caveats=[f"Synthesis encountered an error: {error}"],
            failed_components=[
                r.sub_question_id 
                for r in synthesis_input.execution_trace.step_results
                if not r.output.is_success
            ],
            synthesis_reasoning="Fallback response due to synthesis error"
        )


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

async def synthesize_results(
    query: str,
    decomposition: DecompositionResult,
    execution_trace: ExecutionTrace,
    llm_client: Any,
    **kwargs
) -> ComposedResponse:
    """
    Convenience function to synthesize results.
    
    Args:
        query: Original user query
        decomposition: Decomposition result from Phase 1
        execution_trace: Execution trace from Phase 3
        llm_client: LLM client for synthesis
        **kwargs: Additional arguments for ResponseSynthesizer
        
    Returns:
        ComposedResponse with the synthesized answer
    """
    synthesizer = ResponseSynthesizer(llm_client=llm_client, **kwargs)
    
    synthesis_input = SynthesisInput(
        original_query=query,
        decomposition=decomposition,
        execution_trace=execution_trace
    )
    
    return await synthesizer.synthesize(synthesis_input)


# ============================================================================
# SYNC WRAPPER
# ============================================================================

def synthesize_sync(
    synthesis_input: SynthesisInput,
    llm_client: Any,
    **kwargs
) -> ComposedResponse:
    """
    Synchronous wrapper for synthesis.
    """
    import asyncio
    
    synthesizer = ResponseSynthesizer(llm_client=llm_client, **kwargs)
    return asyncio.run(synthesizer.synthesize(synthesis_input))
