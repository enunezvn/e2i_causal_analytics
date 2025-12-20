# src/e2i/agents/orchestrator/router.py
"""
Query Router for the Orchestrator.

Routes classified queries to appropriate handlers based on the
routing pattern determined by the classifier:
- SINGLE_AGENT: Direct to one agent
- PARALLEL_DELEGATION: Fan out to multiple agents
- TOOL_COMPOSER: Use Tool Composer for dependent multi-domain
- CLARIFICATION_NEEDED: Request clarification
"""

import asyncio
from typing import Any, Optional

from .classifier import ClassificationResult, RoutingPattern


class QueryRouter:
    """
    Routes queries based on classification results.
    """

    def __init__(
        self,
        agents: dict[str, Any],  # Agent name -> Agent instance
        tool_composer: Any,       # ToolComposer instance
        redis_client: Any = None,
    ):
        """
        Initialize router.
        
        Args:
            agents: Dictionary of agent instances by name
            tool_composer: Tool Composer instance
            redis_client: Redis client for working memory
        """
        self.agents = agents
        self.tool_composer = tool_composer
        self.redis_client = redis_client

    async def route(
        self,
        query: str,
        classification: ClassificationResult,
        context: Optional[dict] = None,
    ) -> dict[str, Any]:
        """
        Route query based on classification.
        
        Args:
            query: Original query
            classification: Classification result from pipeline
            context: Optional conversation context
            
        Returns:
            Response dictionary with result and metadata
        """
        pattern = classification.routing_pattern

        if pattern == RoutingPattern.SINGLE_AGENT:
            return await self._route_single_agent(
                query, classification, context
            )

        if pattern == RoutingPattern.PARALLEL_DELEGATION:
            return await self._route_parallel(
                query, classification, context
            )

        if pattern == RoutingPattern.TOOL_COMPOSER:
            return await self._route_tool_composer(
                query, classification, context
            )

        if pattern == RoutingPattern.CLARIFICATION_NEEDED:
            return self._request_clarification(classification)

        # Fallback
        return {
            "response": "I'm not sure how to handle this query. Could you rephrase?",
            "routing_pattern": pattern.value,
            "error": "Unknown routing pattern",
        }

    # =========================================================================
    # SINGLE AGENT ROUTING
    # =========================================================================

    async def _route_single_agent(
        self,
        query: str,
        classification: ClassificationResult,
        context: Optional[dict],
    ) -> dict[str, Any]:
        """Route to a single agent."""
        
        if not classification.target_agents:
            return {
                "response": "Unable to determine appropriate agent.",
                "routing_pattern": RoutingPattern.SINGLE_AGENT.value,
                "error": "No target agent",
            }

        agent_name = classification.target_agents[0]
        agent = self.agents.get(agent_name)

        if not agent:
            return {
                "response": f"Agent '{agent_name}' not available.",
                "routing_pattern": RoutingPattern.SINGLE_AGENT.value,
                "error": f"Agent not found: {agent_name}",
            }

        try:
            # Build agent context
            agent_context = {
                **(context or {}),
                "classification": classification.model_dump(),
                "consultation_hints": classification.consultation_hints,
            }

            # Call agent
            result = await agent.process(query, agent_context)

            return {
                "response": result.get("response", str(result)),
                "routing_pattern": RoutingPattern.SINGLE_AGENT.value,
                "agent": agent_name,
                "classification_confidence": classification.confidence,
                "metadata": result.get("metadata", {}),
            }

        except Exception as e:
            return {
                "response": f"Error processing query: {str(e)}",
                "routing_pattern": RoutingPattern.SINGLE_AGENT.value,
                "agent": agent_name,
                "error": str(e),
            }

    # =========================================================================
    # PARALLEL DELEGATION
    # =========================================================================

    async def _route_parallel(
        self,
        query: str,
        classification: ClassificationResult,
        context: Optional[dict],
    ) -> dict[str, Any]:
        """Route to multiple agents in parallel."""
        
        if not classification.target_agents:
            return {
                "response": "Unable to determine appropriate agents.",
                "routing_pattern": RoutingPattern.PARALLEL_DELEGATION.value,
                "error": "No target agents",
            }

        # Map sub-questions to agents
        agent_tasks = []
        sub_question_map = {
            sq.id: sq for sq in classification.sub_questions
        }

        for i, agent_name in enumerate(classification.target_agents):
            agent = self.agents.get(agent_name)
            if not agent:
                continue

            # Get sub-question for this agent if available
            sq_id = f"Q{i + 1}"
            sub_question = sub_question_map.get(sq_id)
            sq_text = sub_question.text if sub_question else query

            agent_context = {
                **(context or {}),
                "sub_question_id": sq_id,
                "is_parallel": True,
            }

            agent_tasks.append(
                self._call_agent_safe(agent, agent_name, sq_text, agent_context)
            )

        # Execute in parallel
        results = await asyncio.gather(*agent_tasks)

        # Merge results
        merged_response = self._merge_parallel_results(
            results, classification.target_agents
        )

        return {
            "response": merged_response,
            "routing_pattern": RoutingPattern.PARALLEL_DELEGATION.value,
            "agents": classification.target_agents,
            "classification_confidence": classification.confidence,
            "sub_results": results,
        }

    async def _call_agent_safe(
        self,
        agent: Any,
        agent_name: str,
        query: str,
        context: dict,
    ) -> dict[str, Any]:
        """Call agent with error handling."""
        try:
            result = await agent.process(query, context)
            return {
                "agent": agent_name,
                "success": True,
                "response": result.get("response", str(result)),
                "metadata": result.get("metadata", {}),
            }
        except Exception as e:
            return {
                "agent": agent_name,
                "success": False,
                "error": str(e),
            }

    def _merge_parallel_results(
        self,
        results: list[dict],
        agent_names: list[str],
    ) -> str:
        """Merge results from parallel agent calls."""
        
        successful = [r for r in results if r.get("success")]
        failed = [r for r in results if not r.get("success")]

        if not successful:
            return "Unable to get results from any agent."

        # Simple merge: concatenate with headers
        parts = []
        for result in successful:
            agent = result.get("agent", "Unknown")
            response = result.get("response", "")
            parts.append(f"**{agent.replace('_', ' ').title()}:**\n{response}")

        merged = "\n\n".join(parts)

        if failed:
            failed_agents = [r.get("agent") for r in failed]
            merged += f"\n\n*Note: Unable to get results from: {', '.join(failed_agents)}*"

        return merged

    # =========================================================================
    # TOOL COMPOSER ROUTING
    # =========================================================================

    async def _route_tool_composer(
        self,
        query: str,
        classification: ClassificationResult,
        context: Optional[dict],
    ) -> dict[str, Any]:
        """Route to Tool Composer for dependent multi-domain queries."""
        
        from ..tool_composer.schemas import (
            CompositionRequest,
            SubQuestionInput,
            DependencyInput,
        )

        # Build composition request
        sub_questions = [
            SubQuestionInput(
                id=sq.id,
                text=sq.text,
                primary_domain=sq.primary_domain.value if hasattr(sq.primary_domain, 'value') else str(sq.primary_domain),
                domains=[d.value if hasattr(d, 'value') else str(d) for d in sq.domains],
            )
            for sq in classification.sub_questions
        ]

        dependencies = [
            DependencyInput(
                **{
                    "from": dep.from_id,
                    "to": dep.to_id,
                },
                dependency_type=dep.dependency_type.value if hasattr(dep.dependency_type, 'value') else str(dep.dependency_type),
                reason=dep.reason,
            )
            for dep in classification.dependencies
        ]

        request = CompositionRequest(
            query=query,
            sub_questions=sub_questions,
            dependencies=dependencies,
            context=context or {},
        )

        try:
            result = await self.tool_composer.compose(request)

            return {
                "response": result.response,
                "routing_pattern": RoutingPattern.TOOL_COMPOSER.value,
                "composition_id": result.composition_id,
                "status": result.status.value,
                "total_latency_ms": result.total_latency_ms,
                "tool_outputs": result.tool_outputs,
                "classification_confidence": classification.confidence,
            }

        except Exception as e:
            # Fallback to parallel if Tool Composer fails
            return {
                "response": f"Tool composition failed: {str(e)}. Falling back to parallel execution.",
                "routing_pattern": RoutingPattern.TOOL_COMPOSER.value,
                "error": str(e),
                "fallback_attempted": True,
            }

    # =========================================================================
    # CLARIFICATION
    # =========================================================================

    def _request_clarification(
        self, classification: ClassificationResult
    ) -> dict[str, Any]:
        """Request clarification for ambiguous queries."""
        
        response = (
            "I'd like to make sure I understand your question correctly. "
            "Could you please clarify:\n\n"
        )

        # Add specific clarifying questions based on classification
        questions = []
        
        if classification.reasoning:
            questions.append(
                f"• {classification.reasoning}"
            )
        else:
            questions.append(
                "• What specific aspect would you like me to focus on?"
            )
            questions.append(
                "• Are you looking for causal analysis, predictions, or general exploration?"
            )

        response += "\n".join(questions)

        return {
            "response": response,
            "routing_pattern": RoutingPattern.CLARIFICATION_NEEDED.value,
            "needs_clarification": True,
            "classification_confidence": classification.confidence,
        }

    # =========================================================================
    # CONTEXT MANAGEMENT
    # =========================================================================

    async def _store_routing_context(
        self,
        session_id: str,
        classification: ClassificationResult,
    ) -> None:
        """Store routing context in Redis for follow-up handling."""
        if not self.redis_client or not session_id:
            return

        key = f"routing:{session_id}:last"
        await self.redis_client.set(
            key,
            classification.model_dump_json(),
            ex=1800,  # 30 minute TTL
        )

    async def _get_routing_context(
        self, session_id: str
    ) -> Optional[ClassificationResult]:
        """Retrieve last routing context for follow-up handling."""
        if not self.redis_client or not session_id:
            return None

        key = f"routing:{session_id}:last"
        data = await self.redis_client.get(key)
        
        if data:
            return ClassificationResult.model_validate_json(data)
        return None
