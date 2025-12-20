# src/e2i/agents/tool_composer/prompts.py
"""
LLM prompts for Tool Composer components.

These prompts are used by the decomposer and synthesizer
for LLM-based processing.
"""

# =============================================================================
# DECOMPOSITION PROMPT (Phase 1)
# =============================================================================

DECOMPOSITION_PROMPT = """Decompose this multi-part query into distinct, atomic sub-questions.

QUERY: {query}

AVAILABLE DOMAINS:
- CAUSAL_ANALYSIS: Cause-effect relationships, impact measurement, treatment effects
- HETEROGENEITY: Segment variation, who responds best, CATE analysis
- GAP_ANALYSIS: Performance gaps, opportunities, benchmarking
- EXPERIMENTATION: Test design, what-if scenarios, power calculations
- PREDICTION: Future outcomes, risk scores, forecasts
- MONITORING: Data quality, drift detection, anomalies
- EXPLANATION: Clarification, summarization, interpretation

RULES:
1. Each sub-question should be answerable by a single analytical tool
2. Preserve the logical flow and any dependencies between questions
3. Keep sub-questions specific and actionable
4. Assign domains based on the analytical capability needed

Respond in JSON format:
{{
    "sub_questions": [
        {{
            "id": "Q1",
            "text": "What was the causal impact of Q3 speaker programs on Kisqali adoption?",
            "domains": ["CAUSAL_ANALYSIS"],
            "primary_domain": "CAUSAL_ANALYSIS"
        }},
        {{
            "id": "Q2",
            "text": "Which HCP segments showed the strongest response to Q3 speaker programs?",
            "domains": ["HETEROGENEITY", "CAUSAL_ANALYSIS"],
            "primary_domain": "HETEROGENEITY"
        }}
    ],
    "reasoning": "Brief explanation of decomposition logic"
}}"""


# =============================================================================
# SYNTHESIS PROMPT (Phase 4)
# =============================================================================

SYNTHESIS_PROMPT = """Synthesize the following tool outputs into a coherent response.

ORIGINAL QUERY: {query}

SUB-QUESTIONS AND TOOL OUTPUTS:
{tool_outputs_formatted}

INSTRUCTIONS:
1. Address the original query comprehensively using all tool outputs
2. Maintain logical flow connecting the sub-question answers
3. Highlight key insights and actionable findings
4. Note any caveats or limitations from the analysis
5. If outputs conflict, explain the discrepancy
6. Use specific numbers and evidence from the tools
7. Structure the response clearly but conversationally

FORMAT GUIDELINES:
- Lead with the most important finding
- Use clear transitions between topics
- Bold key metrics and recommendations
- Keep technical details accessible
- End with actionable next steps if applicable

Generate a response that directly answers the user's question using the evidence from the tools."""


# =============================================================================
# PARAMETER INFERENCE PROMPT
# =============================================================================

PARAMETER_INFERENCE_PROMPT = """Determine the input parameters for this tool based on the query context.

TOOL: {tool_name}
TOOL DESCRIPTION: {tool_description}
TOOL INPUT SCHEMA:
{input_schema}

QUERY CONTEXT:
- Original query: {query}
- Sub-question being answered: {sub_question}
- Available context: {context}
- Outputs from previous steps: {previous_outputs}

INSTRUCTIONS:
1. Extract parameter values from the query context
2. Use outputs from previous steps if this tool depends on them
3. Apply reasonable defaults for optional parameters
4. Flag any required parameters that cannot be determined

Respond in JSON format:
{{
    "parameters": {{
        "param_name": "value",
        ...
    }},
    "parameter_sources": {{
        "param_name": "source description"
    }},
    "missing_required": [],
    "assumptions": ["Any assumptions made about parameter values"]
}}"""


# =============================================================================
# ERROR RECOVERY PROMPT
# =============================================================================

ERROR_RECOVERY_PROMPT = """A tool execution failed. Determine recovery strategy.

FAILED TOOL: {tool_name}
ERROR: {error_message}
ERROR TYPE: {error_type}

CONTEXT:
- Original query: {query}
- Sub-question: {sub_question}
- Input parameters: {input_params}
- Retry count: {retry_count}

AVAILABLE ACTIONS:
1. RETRY - Retry with same parameters (transient error)
2. RETRY_MODIFIED - Retry with modified parameters
3. SKIP - Skip this tool and note limitation
4. FALLBACK - Use alternative tool
5. ABORT - Cannot recover, abort composition

Respond in JSON format:
{{
    "action": "RETRY_MODIFIED",
    "modified_params": {{}},
    "fallback_tool": null,
    "explanation": "Why this recovery strategy",
    "user_message": "Message to include in final response if step is skipped"
}}"""


# =============================================================================
# RESPONSE QUALITY CHECK PROMPT
# =============================================================================

RESPONSE_QUALITY_PROMPT = """Review this synthesized response for quality and completeness.

ORIGINAL QUERY: {query}

SUB-QUESTIONS:
{sub_questions}

SYNTHESIZED RESPONSE:
{response}

TOOL OUTPUTS USED:
{tool_outputs_summary}

CHECK FOR:
1. Completeness: Does response address all parts of the query?
2. Accuracy: Is the response faithful to tool outputs?
3. Coherence: Does the response flow logically?
4. Actionability: Are findings actionable?
5. Caveats: Are limitations properly noted?

Respond in JSON format:
{{
    "quality_score": 0.85,
    "issues_found": [
        {{"type": "incomplete", "description": "Q2 not fully addressed"}}
    ],
    "suggested_improvements": ["Add specific numbers from CATE analysis"],
    "missing_from_outputs": ["Any tool output not incorporated"],
    "pass": true
}}"""
