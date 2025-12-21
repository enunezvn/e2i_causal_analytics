"""
E2I Tool Composer - Phase 1: Decomposer
Version: 4.2
Purpose: Decompose complex queries into atomic sub-questions
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List

from .models.composition_models import (
    DecompositionResult,
    SubQuestion,
)

logger = logging.getLogger(__name__)


# ============================================================================
# DECOMPOSITION PROMPT
# ============================================================================

DECOMPOSITION_SYSTEM_PROMPT = """You are a query decomposition specialist for a pharmaceutical analytics platform.

Your task is to break down complex, multi-faceted queries into atomic sub-questions that can each be answered by a single analytical tool.

## Guidelines:
1. Each sub-question should be answerable by ONE tool/computation
2. Identify dependencies between sub-questions (which needs results from which)
3. Extract key entities (brands, regions, HCPs, time periods)
4. Classify each sub-question's intent:
   - CAUSAL: Questions about cause-and-effect relationships
   - COMPARATIVE: Questions comparing entities or time periods
   - PREDICTIVE: Questions about future outcomes
   - DESCRIPTIVE: Questions about current state or metrics
   - EXPERIMENTAL: Questions about test design or simulation

## Output Format:
Return a JSON object with:
{
  "reasoning": "Your step-by-step reasoning for the decomposition",
  "sub_questions": [
    {
      "id": "sq_1",
      "question": "The atomic sub-question",
      "intent": "CAUSAL|COMPARATIVE|PREDICTIVE|DESCRIPTIVE|EXPERIMENTAL",
      "entities": ["entity1", "entity2"],
      "depends_on": []  // IDs of prerequisite sub-questions
    }
  ]
}

## Important:
- Generate 2-6 sub-questions (no more, no fewer)
- Ensure dependencies form a valid DAG (no cycles)
- Root questions (no dependencies) should be answerable independently
- Be specific about what each sub-question is asking"""


DECOMPOSITION_USER_TEMPLATE = """Decompose the following query into atomic sub-questions:

QUERY: {query}

Remember:
- Each sub-question should be answerable by a single tool
- Identify dependencies between sub-questions
- Return valid JSON only"""


# ============================================================================
# DECOMPOSER CLASS
# ============================================================================


class QueryDecomposer:
    """
    Decomposes complex queries into atomic sub-questions.

    This is Phase 1 of the Tool Composer pipeline.
    """

    def __init__(
        self,
        llm_client: Any,  # Anthropic client or compatible
        model: str = "claude-sonnet-4-20250514",
        temperature: float = 0.3,
        max_sub_questions: int = 6,
        min_sub_questions: int = 2,
    ):
        self.llm_client = llm_client
        self.model = model
        self.temperature = temperature
        self.max_sub_questions = max_sub_questions
        self.min_sub_questions = min_sub_questions

    async def decompose(self, query: str) -> DecompositionResult:
        """
        Decompose a query into sub-questions.

        Args:
            query: The original user query

        Returns:
            DecompositionResult with sub-questions and reasoning
        """
        logger.info(f"Decomposing query: {query[:100]}...")

        try:
            # Call LLM for decomposition
            response = await self._call_llm(query)

            # Parse response
            parsed = self._parse_response(response)

            # Validate and build result
            sub_questions = self._build_sub_questions(parsed)

            # Validate dependency graph
            self._validate_dependencies(sub_questions)

            result = DecompositionResult(
                original_query=query,
                sub_questions=sub_questions,
                decomposition_reasoning=parsed.get("reasoning", ""),
                timestamp=datetime.now(timezone.utc),
            )

            logger.info(f"Decomposed into {len(sub_questions)} sub-questions")
            return result

        except Exception as e:
            logger.error(f"Decomposition failed: {e}")
            raise DecompositionError(f"Failed to decompose query: {e}") from e

    async def _call_llm(self, query: str) -> str:
        """Call the LLM for decomposition"""
        user_message = DECOMPOSITION_USER_TEMPLATE.format(query=query)

        # Using Anthropic's message API format
        response = await self.llm_client.messages.create(
            model=self.model,
            max_tokens=2000,
            temperature=self.temperature,
            system=DECOMPOSITION_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

        return response.content[0].text

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON from LLM response"""
        # Handle markdown code blocks
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
            logger.error(f"Failed to parse JSON: {response[:200]}...")
            raise DecompositionError(f"Invalid JSON in LLM response: {e}") from e

    def _build_sub_questions(self, parsed: Dict[str, Any]) -> List[SubQuestion]:
        """Build SubQuestion objects from parsed response"""
        raw_questions = parsed.get("sub_questions", [])

        if len(raw_questions) < self.min_sub_questions:
            raise DecompositionError(
                f"Too few sub-questions: {len(raw_questions)} < {self.min_sub_questions}"
            )

        if len(raw_questions) > self.max_sub_questions:
            logger.warning(
                f"Too many sub-questions ({len(raw_questions)}), truncating to {self.max_sub_questions}"
            )
            raw_questions = raw_questions[: self.max_sub_questions]

        sub_questions = []
        for i, sq in enumerate(raw_questions):
            sub_questions.append(
                SubQuestion(
                    id=sq.get("id", f"sq_{i+1}"),
                    question=sq["question"],
                    intent=sq.get("intent", "DESCRIPTIVE"),
                    entities=sq.get("entities", []),
                    depends_on=sq.get("depends_on", []),
                )
            )

        return sub_questions

    def _validate_dependencies(self, sub_questions: List[SubQuestion]) -> None:
        """Validate that dependencies form a valid DAG"""
        sq_ids = {sq.id for sq in sub_questions}

        for sq in sub_questions:
            for dep in sq.depends_on:
                if dep not in sq_ids:
                    raise DecompositionError(
                        f"Invalid dependency: {sq.id} depends on unknown {dep}"
                    )

        # Check for cycles using DFS
        visited = set()
        rec_stack = set()

        def has_cycle(node_id: str) -> bool:
            visited.add(node_id)
            rec_stack.add(node_id)

            node = next((sq for sq in sub_questions if sq.id == node_id), None)
            if node:
                for dep in node.depends_on:
                    if dep not in visited:
                        if has_cycle(dep):
                            return True
                    elif dep in rec_stack:
                        return True

            rec_stack.remove(node_id)
            return False

        for sq in sub_questions:
            if sq.id not in visited:
                if has_cycle(sq.id):
                    raise DecompositionError("Dependency cycle detected in sub-questions")


# ============================================================================
# EXCEPTIONS
# ============================================================================


class DecompositionError(Exception):
    """Error during query decomposition"""

    pass


# ============================================================================
# SYNC WRAPPER
# ============================================================================


def decompose_sync(query: str, llm_client: Any, **kwargs) -> DecompositionResult:
    """
    Synchronous wrapper for decomposition.

    Useful for non-async contexts.
    """
    import asyncio

    decomposer = QueryDecomposer(llm_client=llm_client, **kwargs)
    return asyncio.run(decomposer.decompose(query))
