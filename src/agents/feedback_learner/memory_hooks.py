"""
Feedback Learner Memory Hooks
=============================

Hooks for integrating the Feedback Learner agent with the memory system.

The Feedback Learner uses these hooks to:
1. Receive learning signals from cognitive cycles
2. Collect training examples for DSPy optimization
3. Update procedural memory with successful patterns
4. Monitor agent performance metrics

Author: E2I Causal Analytics Team
Version: 1.0.0
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class LearningSignal:
    """A learning signal from a cognitive cycle."""

    signal_id: str
    signal_type: str  # outcome_success, outcome_partial, rating, thumbs_up, thumbs_down
    signal_value: float
    applies_to_type: str  # query, procedure, agent
    applies_to_id: str
    rated_agent: Optional[str] = None
    session_id: Optional[str] = None
    cycle_id: Optional[str] = None
    signal_details: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class TrainingExample:
    """A training example for DSPy optimization."""

    example_id: str
    query: str
    query_type: str
    context: Dict[str, Any]
    expected_output: str
    actual_output: str
    agent_name: str
    score: float
    is_positive: bool  # Good example (score > 0.8) or negative
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class AgentPerformanceMetric:
    """Performance metric for an agent."""

    agent_name: str
    metric_name: str
    metric_value: float
    window_size: str  # hour, day, week
    sample_count: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# =============================================================================
# MEMORY HOOKS
# =============================================================================


class MemoryHooks:
    """
    Memory integration hooks for the Feedback Learner.

    Provides methods to:
    - Receive and process learning signals
    - Build training datasets for DSPy
    - Update procedural memory patterns
    - Calculate agent performance metrics
    """

    def __init__(self):
        """Initialize memory hooks."""
        self._signal_buffer: List[LearningSignal] = []
        self._training_examples: List[TrainingExample] = []
        self._signal_handlers: List[Callable[[LearningSignal], None]] = []
        self._buffer_lock = asyncio.Lock()

    def register_handler(self, handler: Callable[[LearningSignal], None]) -> None:
        """Register a handler for learning signals."""
        self._signal_handlers.append(handler)

    async def receive_signal(self, signal: LearningSignal) -> None:
        """
        Receive a learning signal from a cognitive cycle.

        Signals are buffered and processed in batches.
        """
        async with self._buffer_lock:
            self._signal_buffer.append(signal)

        # Notify handlers
        for handler in self._signal_handlers:
            try:
                handler(signal)
            except Exception as e:
                logger.error(f"Signal handler failed: {e}")

        # Auto-process if buffer is large
        if len(self._signal_buffer) >= 100:
            await self.process_signal_buffer()

    async def process_signal_buffer(self) -> Dict[str, int]:
        """
        Process buffered learning signals.

        Groups signals by type and updates:
        - Procedural memory success rates
        - Training example pool
        - Performance metrics
        """
        async with self._buffer_lock:
            signals = self._signal_buffer.copy()
            self._signal_buffer.clear()

        if not signals:
            return {"processed": 0}

        # Group by agent
        agent_signals: Dict[str, List[LearningSignal]] = {}
        for signal in signals:
            agent = signal.rated_agent or "unknown"
            if agent not in agent_signals:
                agent_signals[agent] = []
            agent_signals[agent].append(signal)

        # Calculate metrics per agent
        metrics_updated = 0
        for agent_name, agent_sigs in agent_signals.items():
            try:
                await self._update_agent_metrics(agent_name, agent_sigs)
                metrics_updated += 1
            except Exception as e:
                logger.error(f"Failed to update metrics for {agent_name}: {e}")

        # Extract training examples from high-quality signals
        examples_created = 0
        for signal in signals:
            if signal.signal_value > 0.8:  # High quality
                try:
                    example = await self._create_training_example(signal)
                    if example:
                        self._training_examples.append(example)
                        examples_created += 1
                except Exception as e:
                    logger.error(f"Failed to create training example: {e}")

        return {
            "processed": len(signals),
            "metrics_updated": metrics_updated,
            "examples_created": examples_created,
        }

    async def _update_agent_metrics(self, agent_name: str, signals: List[LearningSignal]) -> None:
        """Update performance metrics for an agent."""
        from src.memory.procedural_memory import update_procedure_outcome

        # Calculate success rate
        success_count = sum(1 for s in signals if s.signal_value > 0.7)
        total_count = len(signals)

        if total_count > 0:
            success_rate = success_count / total_count
            logger.info(
                f"Agent {agent_name}: {success_count}/{total_count} "
                f"success rate ({success_rate:.1%})"
            )

        # Update procedural memory for each signal
        for signal in signals:
            if signal.applies_to_type == "procedure" and signal.applies_to_id:
                try:
                    await update_procedure_outcome(
                        procedure_id=signal.applies_to_id, success=signal.signal_value > 0.7
                    )
                except Exception as e:
                    logger.warning(f"Failed to update procedure outcome: {e}")

    async def _create_training_example(self, signal: LearningSignal) -> Optional[TrainingExample]:
        """Create a training example from a high-quality signal."""
        import uuid

        # Only create examples from query signals with details
        if signal.applies_to_type != "query":
            return None

        details = signal.signal_details or {}
        if not details.get("is_training_example"):
            return None

        return TrainingExample(
            example_id=str(uuid.uuid4()),
            query=details.get("query", ""),
            query_type=details.get("query_type", "general"),
            context=details.get("context", {}),
            expected_output=details.get("expected_output", ""),
            actual_output=details.get("actual_output", ""),
            agent_name=signal.rated_agent or "unknown",
            score=signal.signal_value,
            is_positive=signal.signal_value > 0.8,
            metadata={
                "session_id": signal.session_id,
                "cycle_id": signal.cycle_id,
                "signal_type": signal.signal_type,
            },
        )

    async def get_training_examples(
        self,
        agent_name: Optional[str] = None,
        query_type: Optional[str] = None,
        min_score: float = 0.8,
        limit: int = 100,
    ) -> List[TrainingExample]:
        """
        Get training examples for DSPy optimization.

        Filters by agent name, query type, and minimum score.
        """
        examples = self._training_examples

        if agent_name:
            examples = [e for e in examples if e.agent_name == agent_name]

        if query_type:
            examples = [e for e in examples if e.query_type == query_type]

        examples = [e for e in examples if e.score >= min_score]

        # Sort by score descending
        examples = sorted(examples, key=lambda x: x.score, reverse=True)

        return examples[:limit]

    async def get_few_shot_examples(
        self, query: str, query_type: str, agent_name: str, max_examples: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Get few-shot examples for prompt engineering.

        Finds the most similar high-quality examples for a given query.
        """
        from src.memory.procedural_memory import get_few_shot_examples_by_text

        try:
            examples = await get_few_shot_examples_by_text(
                query_text=query,
                intent=query_type,
                brand=None,
                max_examples=max_examples,
            )
            return examples
        except Exception as e:
            logger.warning(f"Failed to get few-shot examples: {e}")
            return []

    async def get_agent_performance(self, agent_name: str, window: str = "day") -> Dict[str, Any]:
        """
        Get performance metrics for an agent.

        Returns success rate, response quality, and trend.
        """
        from src.memory.procedural_memory import get_training_examples_for_agent

        try:
            examples = await get_training_examples_for_agent(
                agent_name=agent_name, brand=None, min_score=0.0, limit=100
            )

            if not examples:
                return {
                    "agent_name": agent_name,
                    "sample_count": 0,
                    "success_rate": None,
                    "average_score": None,
                    "window": window,
                }

            scores = [e.get("score", 0.5) for e in examples]
            success_count = sum(1 for s in scores if s > 0.7)

            return {
                "agent_name": agent_name,
                "sample_count": len(examples),
                "success_rate": success_count / len(examples) if examples else 0,
                "average_score": sum(scores) / len(scores) if scores else 0,
                "min_score": min(scores) if scores else 0,
                "max_score": max(scores) if scores else 0,
                "window": window,
            }
        except Exception as e:
            logger.error(f"Failed to get agent performance: {e}")
            return {"agent_name": agent_name, "error": str(e), "window": window}


# =============================================================================
# SINGLETON ACCESS
# =============================================================================

_memory_hooks: Optional[MemoryHooks] = None


def get_memory_hooks() -> MemoryHooks:
    """Get or create memory hooks singleton."""
    global _memory_hooks
    if _memory_hooks is None:
        _memory_hooks = MemoryHooks()
    return _memory_hooks


# Alias for more specific naming (used by prediction_synthesizer.dspy_integration)
get_feedback_learner_memory_hooks = get_memory_hooks


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


async def on_learning_signal(
    signal_type: str,
    signal_value: float,
    applies_to_type: str,
    applies_to_id: str,
    rated_agent: Optional[str] = None,
    session_id: Optional[str] = None,
    cycle_id: Optional[str] = None,
    signal_details: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Convenience function to record a learning signal.

    Called by the cognitive workflow reflector phase.
    """
    import uuid

    hooks = get_memory_hooks()

    signal = LearningSignal(
        signal_id=str(uuid.uuid4()),
        signal_type=signal_type,
        signal_value=signal_value,
        applies_to_type=applies_to_type,
        applies_to_id=applies_to_id,
        rated_agent=rated_agent,
        session_id=session_id,
        cycle_id=cycle_id,
        signal_details=signal_details or {},
    )

    await hooks.receive_signal(signal)


async def get_dspy_training_set(
    agent_name: str, min_score: float = 0.8, limit: int = 50
) -> List[Dict[str, Any]]:
    """
    Get training examples formatted for DSPy.

    Returns examples in the format expected by DSPy's BootstrapFewShot.
    """
    hooks = get_memory_hooks()
    examples = await hooks.get_training_examples(
        agent_name=agent_name, min_score=min_score, limit=limit
    )

    # Format for DSPy
    dspy_examples = []
    for ex in examples:
        dspy_examples.append(
            {
                "question": ex.query,
                "context": ex.context,
                "answer": ex.actual_output,
                "score": ex.score,
            }
        )

    return dspy_examples
