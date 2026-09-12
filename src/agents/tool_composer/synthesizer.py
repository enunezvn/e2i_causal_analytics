"""
E2I Tool Composer - Phase 4: Synthesizer
Version: 4.2
Purpose: Synthesize tool outputs into a coherent natural language response
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, cast

from langchain_core.messages import HumanMessage, SystemMessage

from src.utils.llm_content import normalize_llm_content
from src.utils.redaction import redact_query

from .models.composition_models import (
    ComposedResponse,
    DecompositionResult,
    ExecutionTrace,
    SynthesisInput,
)

logger = logging.getLogger(__name__)


# ============================================================================
# TOOL-OUTPUT PROJECTION (#2019)
# ============================================================================
#
# Until #2019 the synthesis prompt carried ``json.dumps(result)[:1000]`` -- a
# byte cut through whatever happened to sit at offset 1,000. Every tool in this
# registry declares its bulky containers BEFORE its short fields, so the cut
# removed exactly what the answer has to carry. Measured on this branch by
# calling the real tool callables on real frames:
#
#   gap_calculator[150 territories]  4,925 chars -> top_performer / bottom_performer cut
#   cohort_builder[800 patients]    11,003 chars -> every count cut; the prompt was
#                                                   1,000 chars of raw patient IDs
#   risk_scorer[1,200 patients]    131,546 chars -> model_version / scored_at cut
#   cate_analyzer[16 payer tiers]    3,279 chars -> excluded_segments (#1610) cut
#   segment_ranker[16 tiers]         1,189 chars -> recommended_targets cut
#
# So the cap was not merely dropping disclosures; it was dropping verdicts. The
# fix is to spend the budget structurally rather than to raise it blindly:
#
#   * every SCALAR field survives -- that is where the verdicts and the short
#     disclosures live, and the scalar skeleton of every registered tool's output
#     measures 55-598 chars, so keeping all of them is nearly free;
#   * CONTAINER fields (lists / dicts) are filled round-robin from their head,
#     disclosure-ish containers first, so no single bulky array can starve the
#     rest the way ``entity_values`` starved ``top_performer``;
#   * every trim is stated inline with a count, so a trimmed array can never be
#     read as a complete one.
#
# The budget is pinned to the composer's existing ``_MAX_FAILURE_REASON_CHARS``
# carry limit: a tool's FAILURE reason already gets 2,000 characters carried into
# the user-visible answer (#1574 / #1599 / #1610 all bound their reasons against
# it), so its SUCCESS output getting 1,000 was the incoherent half. With the
# decomposer's ``max_sub_questions=6`` ceiling the whole results block is bounded
# at 6 x 2,000 chars. The budget covers the WHOLE rendered string -- body plus
# the trailing summary line -- since that is what costs prompt budget.
SYNTHESIS_OUTPUT_BUDGET_CHARS = 2000

# Longest string measured in a real tool output is 269 chars (the
# ``sensitivity_analyzer`` interpretation), so this clip never fires on today's
# tools; it exists so a pathological prose field cannot eat the whole budget.
_MAX_STRING_CHARS = 400
_MIN_STRING_CHARS = 80

# A disclosure entry's load-bearing parts are its stable ``reason`` code and its
# ``n``; the ``detail`` beside them is templated prose that repeats almost
# verbatim per entry (measured: 7 ``cate_analyzer`` exclusions spend 917 of their
# 1,672 chars on the SAME sentence). So when disclosures would otherwise lose
# entries, their prose is compressed to this floor first -- an exclusion present
# with a shortened detail beats an exclusion the answer never mentions.
_DISCLOSURE_STRING_FLOOR = 48

# Containers whose name suggests a disclosure, a limitation or the actionable
# recommendation are filled BEFORE bulky data containers. This is a PRIORITY
# hint, not a whitelist: a container that matches nothing is still rendered and
# still trimmed with an explicit count, so a missing token degrades the ordering,
# never the disclosure.
_DISCLOSURE_KEY_TOKENS = (
    "assumption",
    "breakdown",
    "caveat",
    "dropped",
    "error",
    "exclud",
    "fail",
    "limitat",
    "missing",
    "reason",
    "recommend",
    "refus",
    "refut",
    "responder",
    "review",
    "skipped",
    "uncertain",
    "validation",
    "warn",
)

# Share of the budget the non-disclosure containers keep while the disclosure
# containers are filled first, so keeping the caveat never costs the verdict.
_NON_DISCLOSURE_BUDGET_SHARE = 0.2

_OMISSION_PREFIX = "_omitted"


def _render_json(node: Any) -> str:
    """The serialization the synthesis prompt has always used."""
    return json.dumps(node, indent=2, default=str)


def _is_container(value: Any) -> bool:
    return isinstance(value, (list, tuple, dict))


def _is_disclosure_key(key: str) -> bool:
    lowered = str(key).lower()
    return any(token in lowered for token in _DISCLOSURE_KEY_TOKENS)


def _clip_strings(node: Any, limit: int) -> Any:
    """Bound every string in ``node``, marking the elision with its size."""
    if isinstance(node, dict):
        return {k: _clip_strings(v, limit) for k, v in node.items()}
    if isinstance(node, (list, tuple)):
        return [_clip_strings(v, limit) for v in node]
    if isinstance(node, str) and len(node) > limit:
        return node[:limit] + f"… (+{len(node) - limit} chars)"
    return node


def _build_projection(
    result: Dict[str, Any],
    containers: Dict[str, Any],
    counts: Dict[str, int],
    string_limits: Dict[str, int],
) -> Dict[str, Any]:
    """Render ``result`` keeping ``counts[key]`` head entries of each container.

    Key order is the tool model's own, so the projection reads like the output it
    stands for. A container that lost entries carries an explicit ``_omitted``
    note naming how many of how many went. ``string_limits`` is per top-level key
    so a disclosure's prose can be compressed without touching anything else.
    """
    projected: Dict[str, Any] = {}
    for key, value in result.items():
        limit = string_limits[key]
        if key not in containers:
            projected[key] = _clip_strings(value, limit)
            continue

        shown = counts[key]
        if isinstance(value, dict):
            entries = list(value.items())
            kept: Dict[Any, Any] = {k: _clip_strings(v, limit) for k, v in entries[:shown]}
            if shown < len(entries):
                kept[_OMISSION_PREFIX] = f"{len(entries) - shown} of {len(entries)} keys omitted"
            projected[key] = kept
        else:
            items = list(value)
            kept_list = [_clip_strings(v, limit) for v in items[:shown]]
            if shown < len(items):
                kept_list.append(
                    f"{_OMISSION_PREFIX}: {len(items) - shown} of {len(items)} items omitted"
                )
            projected[key] = kept_list
    return projected


def project_tool_output(result: Dict[str, Any], budget: int = SYNTHESIS_OUTPUT_BUDGET_CHARS) -> str:
    """Render a tool result for the synthesis prompt within ``budget`` chars.

    An output that already fits is returned byte-identically to the pre-#2019
    dump, so nothing changes for the tools that were never truncated.

    ``budget`` bounds the WHOLE returned string -- JSON body plus the trailing
    summary line -- because the whole string is what costs synthesis prompt
    budget. The footer is sized first and its cost is deducted before any
    container is filled.

    The real bound is::

        len(project_tool_output(x, n)) <= max(n, scalar_floor(x))

    NOT ``<= n``. ``scalar_floor(x)`` is the incompressible part: every scalar
    field rendered at its hardest clip (``_MIN_STRING_CHARS``), every container
    emptied to its ``_omitted`` note, plus the footer. Below that floor this
    function OVERRUNS the budget rather than meeting it, deliberately: scalars
    are where the verdicts live (``top_performer``, ``gate_decision``,
    ``total_eligible``), they are never trimmed, and dropping one to hit a byte
    target is precisely the defect #2019 removed. A caller that lowers ``budget``
    below the floor gets a complete answer that is too big, never a small answer
    that is missing the finding.

    Measured floors: ``gap_calculator``-shaped output 340 chars; the worst
    scalars-only size across every registered tool is 598. So at the production
    budget of 2,000 no registered tool can reach the floor, and the only caller
    today is ``ResponseSynthesizer.output_budget_chars`` (default 2,000). The
    overrun is reachable only by a future caller passing a much smaller budget --
    which is why it is documented and pinned rather than silently true.
    """
    full = _render_json(result)
    if len(full) <= budget:
        return full

    # Size the footer up front and charge it to the budget, so the caller's cap
    # covers what the LLM actually receives rather than only the JSON body.
    footer = (
        f"\n(structured summary of a {len(full)}-char tool output: every scalar field is "
        f"kept in full, longer lists and dicts are trimmed from the tail and each trim is "
        f"marked '{_OMISSION_PREFIX}' with its count)"
    )
    budget = max(budget - len(footer), 0)

    containers = {k: v for k, v in result.items() if _is_container(v)}
    counts: Dict[str, int] = dict.fromkeys(containers, 0)
    limits: Dict[str, int] = dict.fromkeys(result, _MAX_STRING_CHARS)
    scalar_keys = [k for k in result if k not in containers]

    def rendered_size(candidate: Dict[str, int]) -> int:
        return len(_render_json(_build_projection(result, containers, candidate, limits)))

    def fill(keys: List[str], cap: int) -> None:
        """Round-robin one entry at a time, so no container starves another."""
        progressed = True
        while progressed:
            progressed = False
            for key in keys:
                if counts[key] >= len(containers[key]):
                    continue
                counts[key] += 1
                if rendered_size(counts) > cap:
                    counts[key] -= 1
                else:
                    progressed = True

    def incomplete(keys: List[str]) -> bool:
        return any(counts[k] < len(containers[k]) for k in keys)

    # 1. Scalars always survive. Clip their prose only if the skeleton alone
    #    overruns -- no measured tool output gets here (worst scalar skeleton is
    #    598 chars), so this is the pathological-prose guard.
    while rendered_size(counts) > budget and any(
        limits[k] > _MIN_STRING_CHARS for k in scalar_keys
    ):
        for key in scalar_keys:
            limits[key] = max(_MIN_STRING_CHARS, limits[key] // 2)

    disclosure_keys = [k for k in containers if _is_disclosure_key(k)]
    plain_keys = [k for k in containers if not _is_disclosure_key(k)]

    # 2. Disclosures first, but only up to the reserve -- keeping the caveat must
    #    not cost the headline numbers it qualifies.
    if disclosure_keys and plain_keys:
        reserve = int(budget * (1 - _NON_DISCLOSURE_BUDGET_SHARE))
        fill(disclosure_keys, reserve)
        # 3. Compress templated disclosure prose rather than drop an entry: the
        #    ``reason`` code and ``n`` beside it are what the answer must carry.
        while incomplete(disclosure_keys) and any(
            limits[k] > _DISCLOSURE_STRING_FLOOR for k in disclosure_keys
        ):
            for key in disclosure_keys:
                limits[key] = max(_DISCLOSURE_STRING_FLOOR, limits[key] // 2)
            fill(disclosure_keys, reserve)
    # 4. Then everything competes for what is left.
    fill(disclosure_keys + plain_keys, budget)

    projected = _build_projection(result, containers, counts, limits)
    return _render_json(projected) + footer


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
        model: str = "claude-sonnet-4-6",
        temperature: float = 0.4,
        max_tokens: int = 2000,
        output_budget_chars: int = SYNTHESIS_OUTPUT_BUDGET_CHARS,
    ):
        self.llm_client = llm_client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        # Per-tool-output budget for the synthesis prompt (#2019).
        self.output_budget_chars = output_budget_chars

    async def synthesize(self, synthesis_input: SynthesisInput) -> ComposedResponse:
        """
        Synthesize tool outputs into a response.

        Args:
            synthesis_input: Contains original query, decomposition, and execution trace

        Returns:
            ComposedResponse with the synthesized answer
        """
        logger.info(
            f"Synthesizing response for query: {redact_query(synthesis_input.original_query)}"
        )

        try:
            # Format results for the prompt
            results_text = self._format_results(synthesis_input)

            # Call LLM for synthesis
            response = await self._call_llm(synthesis_input.original_query, results_text)

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
                timestamp=datetime.now(timezone.utc),
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
                # Structured projection, not a byte cut (#2019): scalars always
                # survive, containers are trimmed from the tail with an explicit
                # count. The old ``[:1000]`` removed whole verdict and disclosure
                # fields because every tool declares its bulky containers first.
                output_str = project_tool_output(result.output.result, self.output_budget_chars)
                lines.append(f"Output:\n{output_str}")
            elif result.output.error:
                lines.append(f"Error: {result.output.error}")

            lines.append("")

        return "\n".join(lines)

    async def _call_llm(self, query: str, results: str) -> str:
        """Call the LLM for synthesis using LangChain interface"""
        user_message = SYNTHESIS_USER_TEMPLATE.format(query=query, results=results)

        # Using LangChain's message format (works with ChatAnthropic/ChatOpenAI)
        messages = [
            SystemMessage(content=SYNTHESIS_SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ]

        response = await self.llm_client.ainvoke(messages)

        # AIMessage.content is str | list of content blocks (#1350)
        return normalize_llm_content(response.content)

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
            return cast(Dict[str, Any], json.loads(response))
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"Failed to parse synthesis JSON, using raw response: {e}")
            # Return the raw response as the answer
            return {
                "answer": response,
                "confidence": 0.6,
                "reasoning": "JSON parsing failed, using raw response",
            }

    def _create_fallback_response(
        self, synthesis_input: SynthesisInput, error: str
    ) -> ComposedResponse:
        """Create a fallback response when synthesis fails"""
        # Try to extract key results
        successful_results = [
            r for r in synthesis_input.execution_trace.step_results if r.output.is_success
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
            synthesis_reasoning="Fallback response due to synthesis error",
        )


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================


async def synthesize_results(
    query: str,
    decomposition: DecompositionResult,
    execution_trace: ExecutionTrace,
    llm_client: Any,
    **kwargs,
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
        original_query=query, decomposition=decomposition, execution_trace=execution_trace
    )

    return await synthesizer.synthesize(synthesis_input)


# ============================================================================
# SYNC WRAPPER
# ============================================================================


def synthesize_sync(synthesis_input: SynthesisInput, llm_client: Any, **kwargs) -> ComposedResponse:
    """
    Synchronous wrapper for synthesis.

    Handles event loop conflicts when called from async contexts.
    """
    import asyncio

    synthesizer = ResponseSynthesizer(llm_client=llm_client, **kwargs)

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import nest_asyncio

        nest_asyncio.apply()
        return loop.run_until_complete(synthesizer.synthesize(synthesis_input))
    else:
        return asyncio.run(synthesizer.synthesize(synthesis_input))
