"""
E2I Explainer Agent - Context Assembler Node
Version: 4.2
Purpose: Gather and assemble context from multiple analysis results
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from ..state import AnalysisContext, ExplainerState

logger = logging.getLogger(__name__)


@runtime_checkable
class ConversationStoreProtocol(Protocol):
    """Protocol for conversation store."""

    async def get_recent(self, session_id: str, limit: int) -> List[Dict[str, Any]]:
        """Get recent conversation history."""
        ...


class ContextAssemblerNode:
    """
    Assemble context from multiple analysis results.
    Prepares input for deep reasoning.
    """

    def __init__(self, conversation_store: Optional[ConversationStoreProtocol] = None):
        """Initialize context assembler."""
        self.conversation_store = conversation_store

    async def execute(self, state: ExplainerState) -> ExplainerState:
        """Execute context assembly."""
        start_time = time.time()

        # Check if already failed
        if state.get("status") == "failed":
            return state

        try:
            analysis_results = state.get("analysis_results", [])

            if not analysis_results:
                return {
                    **state,
                    "errors": [
                        {"node": "context_assembler", "error": "No analysis results provided"}
                    ],
                    "status": "failed",
                }

            # Extract context from each analysis
            contexts = []
            for result in analysis_results:
                context = self._extract_context(result)
                if context:
                    contexts.append(context)

            if not contexts:
                return {
                    **state,
                    "errors": [
                        {"node": "context_assembler", "error": "No valid contexts extracted"}
                    ],
                    "status": "failed",
                }

            # Get user context
            user_context = self._get_user_context(state)

            # Get conversation history if available
            history = await self._get_conversation_history(state)

            assembly_time = int((time.time() - start_time) * 1000)

            logger.info(
                f"Context assembled: {len(contexts)} analysis contexts, "
                f"focus_areas={state.get('focus_areas', [])}"
            )

            return {
                **state,
                "analysis_context": contexts,
                "user_context": user_context,
                "conversation_history": history,
                "assembly_latency_ms": assembly_time,
                "status": "reasoning",
            }

        except Exception as e:
            logger.error(f"Context assembly failed: {e}")
            return {
                **state,
                "errors": [{"node": "context_assembler", "error": str(e)}],
                "status": "failed",
            }

    def _extract_context(self, result: Dict[str, Any]) -> Optional[AnalysisContext]:
        """Extract standardized context from analysis result."""
        try:
            # Extract key findings from various possible formats
            key_findings = result.get("key_findings", [])
            if isinstance(key_findings, dict):
                # If key_findings is a dict, convert to list of strings
                key_findings = [f"{k}: {v}" for k, v in key_findings.items()]
            elif not isinstance(key_findings, list):
                key_findings = [str(key_findings)] if key_findings else []

            # Build data summary excluding standard fields
            excluded_keys = {
                "agent",
                "analysis_type",
                "key_findings",
                "narrative",
                "confidence",
                "warnings",
            }
            data_summary = {k: v for k, v in result.items() if k not in excluded_keys}

            return AnalysisContext(
                source_agent=result.get("agent", "unknown"),
                analysis_type=result.get("analysis_type", "unknown"),
                key_findings=key_findings,
                data_summary=data_summary,
                confidence=result.get("confidence", 0.5),
                warnings=result.get("warnings", []),
            )
        except Exception as e:
            logger.warning(f"Failed to extract context from result: {e}")
            return None

    def _get_user_context(self, state: ExplainerState) -> Dict[str, Any]:
        """Get user context for personalization."""
        return {
            "expertise": state.get("user_expertise", "analyst"),
            "focus_areas": state.get("focus_areas", []),
            "output_format": state.get("output_format", "narrative"),
        }

    async def _get_conversation_history(self, state: ExplainerState) -> List[Dict[str, Any]]:
        """Get relevant conversation history."""
        if self.conversation_store:
            try:
                return await self.conversation_store.get_recent(
                    session_id=state.get("session_id", "default"),
                    limit=5,
                )
            except Exception as e:
                logger.warning(f"Failed to get conversation history: {e}")
        return []
