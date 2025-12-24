"""
E2I Tool Composer Agent - DSPy Integration Module
Version: 4.2
Purpose: DSPy signatures and training signals for tool_composer Hybrid role

The Tool Composer is a DSPy Hybrid agent that:
1. Generates training signals from composition executions
2. Consumes optimized prompts for decomposition and synthesis
3. Routes optimization requests to feedback_learner
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# 1. TRAINING SIGNAL STRUCTURE
# =============================================================================


@dataclass
class CompositionTrainingSignal:
    """
    Training signal for Tool Composer DSPy optimization.

    Captures composition decisions and their outcomes to train:
    - QueryDecompositionSignature: Breaking queries into sub-questions
    - ToolMappingSignature: Mapping sub-questions to tools
    - ResponseSynthesisSignature: Combining tool outputs
    """

    # === Input Context ===
    signal_id: str = ""
    session_id: str = ""
    query: str = ""
    query_complexity: str = ""  # simple, moderate, complex
    entity_count: int = 0
    domain_count: int = 0

    # === Decomposition Phase ===
    sub_questions_count: int = 0
    decomposition_method: str = ""  # pre_decomposed, llm, rule_based
    decomposition_quality: Optional[float] = None  # 0-1 score

    # === Planning Phase ===
    tools_planned: List[str] = field(default_factory=list)
    parallel_groups_count: int = 0
    plan_used_episodic: bool = False  # Used similar past composition

    # === Execution Phase ===
    tools_succeeded: int = 0
    tools_failed: int = 0
    total_execution_latency_ms: float = 0.0

    # === Synthesis Phase ===
    synthesis_confidence: float = 0.0
    response_length: int = 0
    sources_cited_count: int = 0

    # === Outcome Metrics ===
    total_latency_ms: float = 0.0
    user_satisfaction: Optional[float] = None  # 1-5 rating
    answer_quality: Optional[float] = None  # 0-1 score

    # === Timestamp ===
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def compute_reward(self) -> float:
        """
        Compute reward for MIPROv2 optimization.

        Weighting:
        - execution_success: 0.30 (tools succeeded / planned)
        - efficiency: 0.20 (latency penalty)
        - decomposition_quality: 0.20 (if available)
        - synthesis_confidence: 0.15
        - user_satisfaction: 0.15 (if available)
        """
        reward = 0.0

        # Execution success rate
        if self.tools_planned:
            success_rate = self.tools_succeeded / len(self.tools_planned)
            reward += 0.30 * success_rate

        # Efficiency (target < 10s for complex compositions)
        target_latency = 10000
        if self.total_latency_ms > 0:
            efficiency = min(1.0, target_latency / self.total_latency_ms)
            reward += 0.20 * efficiency
        else:
            reward += 0.20

        # Decomposition quality
        if self.decomposition_quality is not None:
            reward += 0.20 * self.decomposition_quality
        else:
            # Proxy: sub_questions count vs domain count ratio
            if self.domain_count > 0 and self.sub_questions_count > 0:
                # Ideal: 1-2 sub-questions per domain
                ratio = self.sub_questions_count / self.domain_count
                quality = 1.0 if 1 <= ratio <= 2 else max(0.5, 1.0 - abs(ratio - 1.5) * 0.2)
                reward += 0.20 * quality
            else:
                reward += 0.10

        # Synthesis confidence
        reward += 0.15 * self.synthesis_confidence

        # User satisfaction
        if self.user_satisfaction is not None:
            satisfaction_score = (self.user_satisfaction - 1) / 4  # 1-5 to 0-1
            reward += 0.15 * satisfaction_score
        else:
            reward += 0.075  # Partial credit if no feedback

        return round(min(1.0, reward), 4)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "signal_id": self.signal_id or f"tc_{self.session_id}_{self.created_at}",
            "source_agent": "tool_composer",
            "dspy_type": "hybrid",
            "timestamp": self.created_at,
            "input_context": {
                "query": self.query[:500] if self.query else "",
                "query_complexity": self.query_complexity,
                "entity_count": self.entity_count,
                "domain_count": self.domain_count,
            },
            "decomposition": {
                "sub_questions_count": self.sub_questions_count,
                "method": self.decomposition_method,
                "quality": self.decomposition_quality,
            },
            "planning": {
                "tools_planned": self.tools_planned[:10],
                "parallel_groups_count": self.parallel_groups_count,
                "used_episodic": self.plan_used_episodic,
            },
            "execution": {
                "tools_succeeded": self.tools_succeeded,
                "tools_failed": self.tools_failed,
                "execution_latency_ms": self.total_execution_latency_ms,
            },
            "synthesis": {
                "confidence": self.synthesis_confidence,
                "response_length": self.response_length,
                "sources_cited_count": self.sources_cited_count,
            },
            "outcome": {
                "total_latency_ms": self.total_latency_ms,
                "user_satisfaction": self.user_satisfaction,
                "answer_quality": self.answer_quality,
            },
            "reward": self.compute_reward(),
        }


# =============================================================================
# 2. DSPy SIGNATURES
# =============================================================================

try:
    import dspy

    class QueryDecompositionSignature(dspy.Signature):
        """
        Decompose complex queries into atomic sub-questions.

        Given a multi-faceted query, break it down into simpler questions
        that can each be answered by a single analytical tool.
        """

        query: str = dspy.InputField(desc="Complex multi-faceted query to decompose")
        entities_detected: str = dspy.InputField(desc="Entities extracted from query")
        domains_detected: str = dspy.InputField(desc="Analytical domains in query")

        sub_questions: list = dspy.OutputField(
            desc="List of atomic sub-questions, each answerable by one tool"
        )
        dependencies: list = dspy.OutputField(
            desc="Dependencies between sub-questions (from_id, to_id, reason)"
        )
        decomposition_rationale: str = dspy.OutputField(
            desc="Brief explanation of decomposition logic"
        )

    class ToolMappingSignature(dspy.Signature):
        """
        Map sub-questions to available tools.

        Given atomic sub-questions and available tools, determine the best
        tool to answer each sub-question.
        """

        sub_questions: str = dspy.InputField(desc="Atomic sub-questions to map")
        available_tools: str = dspy.InputField(desc="Available tools with descriptions")
        past_mappings: str = dspy.InputField(desc="Similar past mappings (if any)")

        tool_assignments: list = dspy.OutputField(
            desc="Tool assignment for each sub-question"
        )
        execution_order: list = dspy.OutputField(desc="Ordered list of tool executions")
        parallel_groups: list = dspy.OutputField(
            desc="Groups of tools that can run in parallel"
        )

    class ResponseSynthesisSignature(dspy.Signature):
        """
        Synthesize tool outputs into coherent response.

        Given outputs from multiple tools, create a unified natural language
        response that addresses the original query.
        """

        original_query: str = dspy.InputField(desc="Original user query")
        tool_outputs: str = dspy.InputField(desc="Outputs from executed tools")
        failed_tools: str = dspy.InputField(desc="Tools that failed (if any)")

        response: str = dspy.OutputField(
            desc="Coherent response addressing the original query"
        )
        key_insights: list = dspy.OutputField(
            desc="Key insights extracted from tool outputs"
        )
        confidence: float = dspy.OutputField(
            desc="Confidence in the synthesized response (0-1)"
        )
        caveats: list = dspy.OutputField(
            desc="Limitations or caveats to mention"
        )

    DSPY_AVAILABLE = True
    logger.info("DSPy signatures loaded for Tool Composer agent")

except ImportError:
    DSPY_AVAILABLE = False
    logger.warning("DSPy not available - using deterministic composition")
    QueryDecompositionSignature = None
    ToolMappingSignature = None
    ResponseSynthesisSignature = None


# =============================================================================
# 3. SIGNAL COLLECTOR
# =============================================================================


class ToolComposerSignalCollector:
    """
    Collects training signals from tool composition executions.

    The Tool Composer is a Hybrid agent that both generates signals
    (from composition outcomes) and consumes optimized prompts
    (for decomposition and synthesis).
    """

    def __init__(self):
        self._signals_buffer: List[CompositionTrainingSignal] = []
        self._buffer_limit = 100

    def collect_composition_signal(
        self,
        session_id: str,
        query: str,
        query_complexity: str,
        entity_count: int,
        domain_count: int,
    ) -> CompositionTrainingSignal:
        """
        Initialize training signal at composition start.

        Call this when starting a new composition.
        """
        signal = CompositionTrainingSignal(
            session_id=session_id,
            query=query,
            query_complexity=query_complexity,
            entity_count=entity_count,
            domain_count=domain_count,
        )
        return signal

    def update_decomposition(
        self,
        signal: CompositionTrainingSignal,
        sub_questions_count: int,
        decomposition_method: str,
        decomposition_quality: Optional[float] = None,
    ) -> CompositionTrainingSignal:
        """Update signal with decomposition phase results."""
        signal.sub_questions_count = sub_questions_count
        signal.decomposition_method = decomposition_method
        signal.decomposition_quality = decomposition_quality
        return signal

    def update_planning(
        self,
        signal: CompositionTrainingSignal,
        tools_planned: List[str],
        parallel_groups_count: int,
        used_episodic: bool = False,
    ) -> CompositionTrainingSignal:
        """Update signal with planning phase results."""
        signal.tools_planned = tools_planned
        signal.parallel_groups_count = parallel_groups_count
        signal.plan_used_episodic = used_episodic
        return signal

    def update_execution(
        self,
        signal: CompositionTrainingSignal,
        tools_succeeded: int,
        tools_failed: int,
        execution_latency_ms: float,
    ) -> CompositionTrainingSignal:
        """Update signal with execution phase results."""
        signal.tools_succeeded = tools_succeeded
        signal.tools_failed = tools_failed
        signal.total_execution_latency_ms = execution_latency_ms
        return signal

    def update_synthesis(
        self,
        signal: CompositionTrainingSignal,
        synthesis_confidence: float,
        response_length: int,
        sources_cited_count: int,
        total_latency_ms: float,
    ) -> CompositionTrainingSignal:
        """Update signal with synthesis phase results."""
        signal.synthesis_confidence = synthesis_confidence
        signal.response_length = response_length
        signal.sources_cited_count = sources_cited_count
        signal.total_latency_ms = total_latency_ms

        # Add to buffer
        self._signals_buffer.append(signal)
        if len(self._signals_buffer) > self._buffer_limit:
            self._signals_buffer.pop(0)

        return signal

    def update_with_feedback(
        self,
        signal: CompositionTrainingSignal,
        user_satisfaction: Optional[float] = None,
        answer_quality: Optional[float] = None,
    ) -> CompositionTrainingSignal:
        """Update signal with user feedback (delayed)."""
        signal.user_satisfaction = user_satisfaction
        signal.answer_quality = answer_quality
        return signal

    def get_signals_for_training(
        self,
        min_reward: float = 0.0,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Get signals suitable for DSPy training."""
        signals = [
            s.to_dict()
            for s in self._signals_buffer
            if s.compute_reward() >= min_reward
        ]
        return signals[-limit:]

    def get_high_quality_examples(
        self,
        min_reward: float = 0.7,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Get high-quality examples for few-shot prompting."""
        signals = sorted(
            [s for s in self._signals_buffer if s.compute_reward() >= min_reward],
            key=lambda s: s.compute_reward(),
            reverse=True,
        )
        return [s.to_dict() for s in signals[:limit]]

    def clear_buffer(self):
        """Clear the signals buffer."""
        self._signals_buffer.clear()


# =============================================================================
# 4. HYBRID DSPy INTEGRATION
# =============================================================================


class ToolComposerDSPyIntegration:
    """
    DSPy Hybrid integration for the Tool Composer.

    As a Hybrid agent, the Tool Composer:
    1. Generates training signals from successful compositions
    2. Consumes optimized prompts for decomposition and synthesis
    3. Requests optimization from feedback_learner when signal quality is high
    """

    def __init__(self):
        self.dspy_type: Literal["hybrid"] = "hybrid"
        self._optimized_prompts: Dict[str, str] = {}
        self._optimization_requests: List[Dict[str, Any]] = []

    async def get_optimized_decomposition_prompt(
        self,
        default_prompt: str,
    ) -> str:
        """
        Get DSPy-optimized prompt for query decomposition.

        Returns the optimized prompt if available, otherwise the default.
        """
        return self._optimized_prompts.get("decomposition", default_prompt)

    async def get_optimized_synthesis_prompt(
        self,
        default_prompt: str,
    ) -> str:
        """
        Get DSPy-optimized prompt for response synthesis.

        Returns the optimized prompt if available, otherwise the default.
        """
        return self._optimized_prompts.get("synthesis", default_prompt)

    def update_optimized_prompt(
        self,
        prompt_type: Literal["decomposition", "synthesis", "tool_mapping"],
        optimized_prompt: str,
    ) -> None:
        """
        Update with a DSPy-optimized prompt from feedback_learner.

        Called when feedback_learner completes optimization.
        """
        self._optimized_prompts[prompt_type] = optimized_prompt
        logger.info(f"Updated {prompt_type} prompt with DSPy-optimized version")

    async def request_optimization(
        self,
        signature_name: str,
        training_signals: List[Dict[str, Any]],
        priority: Literal["low", "medium", "high"] = "medium",
    ) -> str:
        """
        Request prompt optimization from feedback_learner.

        Routes to feedback_learner for actual MIPROv2 optimization.
        """
        request = {
            "request_id": f"tc_opt_{signature_name}_{datetime.now(timezone.utc).isoformat()}",
            "agent_name": "tool_composer",
            "signature_name": signature_name,
            "signal_count": len(training_signals),
            "priority": priority,
            "status": "pending",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        self._optimization_requests.append(request)
        logger.info(f"Optimization requested for tool_composer.{signature_name}")

        return request["request_id"]

    def get_pending_requests(self) -> List[Dict[str, Any]]:
        """Get pending optimization requests."""
        return [r for r in self._optimization_requests if r["status"] == "pending"]

    def has_optimized_prompts(self) -> bool:
        """Check if any optimized prompts are available."""
        return bool(self._optimized_prompts)


# =============================================================================
# 5. SINGLETON ACCESS
# =============================================================================

_signal_collector: Optional[ToolComposerSignalCollector] = None
_dspy_integration: Optional[ToolComposerDSPyIntegration] = None


def get_tool_composer_signal_collector() -> ToolComposerSignalCollector:
    """Get or create signal collector singleton."""
    global _signal_collector
    if _signal_collector is None:
        _signal_collector = ToolComposerSignalCollector()
    return _signal_collector


def get_tool_composer_dspy_integration() -> ToolComposerDSPyIntegration:
    """Get or create DSPy integration singleton."""
    global _dspy_integration
    if _dspy_integration is None:
        _dspy_integration = ToolComposerDSPyIntegration()
    return _dspy_integration


def reset_dspy_integration() -> None:
    """Reset singletons (for testing)."""
    global _signal_collector, _dspy_integration
    _signal_collector = None
    _dspy_integration = None


# =============================================================================
# 6. EXPORTS
# =============================================================================

__all__ = [
    # Training Signals
    "CompositionTrainingSignal",
    # DSPy Signatures
    "QueryDecompositionSignature",
    "ToolMappingSignature",
    "ResponseSynthesisSignature",
    "DSPY_AVAILABLE",
    # Collectors
    "ToolComposerSignalCollector",
    "ToolComposerDSPyIntegration",
    # Access
    "get_tool_composer_signal_collector",
    "get_tool_composer_dspy_integration",
    "reset_dspy_integration",
]
