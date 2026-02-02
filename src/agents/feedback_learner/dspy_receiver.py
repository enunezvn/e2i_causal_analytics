"""
E2I Feedback Learner Agent - DSPy Signal Receiver
Version: 4.3
Purpose: Receive training signals from Tier 2 agents for DSPy optimization

This module implements the DSPy Receiver pattern for the Feedback Learner agent.
It receives training signals from:
- causal_impact (Sender)
- gap_analyzer (Sender)
- heterogeneous_optimizer (Sender)

Signals are stored and made available for:
1. Pattern detection in feedback learning cycle
2. MIPROv2/GEPA prompt optimization training data
3. Knowledge graph updates

Usage:
    from src.agents.feedback_learner.dspy_receiver import receive_training_signals

    # Receive batch from Tier2SignalRouter
    await receive_training_signals({
        "causal_impact": [signal1, signal2],
        "gap_analyzer": [signal3],
    })

    # Retrieve for feedback learning
    signals = get_pending_training_signals(limit=50)
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Literal, Optional

from .state import FeedbackItem

logger = logging.getLogger(__name__)


# =============================================================================
# 1. TYPE DEFINITIONS
# =============================================================================

Tier2Agent = Literal["causal_impact", "gap_analyzer", "heterogeneous_optimizer"]


@dataclass
class ReceivedSignal:
    """A training signal received from a Tier 2 agent."""

    source_agent: Tier2Agent
    signal_data: Dict[str, Any]
    received_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    processed: bool = False


# =============================================================================
# 2. SIGNAL RECEIVER
# =============================================================================


class TrainingSignalReceiver:
    """
    Receives and buffers training signals from Tier 2 agents.

    Implements the DSPy Receiver pattern for the feedback_learner.
    Signals are buffered and made available for:
    - Feedback learning cycle (pattern detection)
    - DSPy optimization (training data)
    """

    def __init__(
        self,
        max_buffer_size: int = 1000,
        max_signals_per_agent: int = 200,
    ):
        """
        Initialize signal receiver.

        Args:
            max_buffer_size: Maximum total signals to buffer
            max_signals_per_agent: Maximum signals per agent type
        """
        self.max_buffer_size = max_buffer_size
        self.max_signals_per_agent = max_signals_per_agent

        # Signal buffers by agent
        self._buffers: Dict[Tier2Agent, Deque[ReceivedSignal]] = {
            "causal_impact": deque(maxlen=max_signals_per_agent),
            "gap_analyzer": deque(maxlen=max_signals_per_agent),
            "heterogeneous_optimizer": deque(maxlen=max_signals_per_agent),
        }

        # Lock for thread-safety
        self._lock = asyncio.Lock()

        # Statistics
        self._stats = {
            "signals_received": 0,
            "signals_processed": 0,
            "signals_dropped": 0,
        }

    async def receive_signals(
        self,
        signals_by_agent: Dict[str, List[Dict[str, Any]]],
    ) -> int:
        """
        Receive batch of training signals from Tier2SignalRouter.

        Args:
            signals_by_agent: Dict mapping agent names to lists of signals

        Returns:
            Number of signals received
        """
        received = 0

        async with self._lock:
            for agent_name, signals in signals_by_agent.items():
                if agent_name not in self._buffers:
                    logger.warning(f"Unknown agent {agent_name}, skipping signals")
                    continue

                buffer = self._buffers[agent_name]

                for signal_data in signals:
                    # Create received signal
                    received_signal = ReceivedSignal(
                        source_agent=agent_name,
                        signal_data=signal_data,
                    )

                    # Add to buffer (deque handles maxlen)
                    buffer.append(received_signal)
                    received += 1
                    self._stats["signals_received"] += 1

                logger.debug(
                    f"Received {len(signals)} signals from {agent_name}, buffer_size={len(buffer)}"
                )

        logger.info(f"Received {received} training signals from {len(signals_by_agent)} agents")
        return received

    async def get_pending_signals(
        self,
        limit: int = 100,
        agent: Optional[Tier2Agent] = None,
        min_reward: float = 0.0,
    ) -> List[ReceivedSignal]:
        """
        Get pending training signals for processing.

        Args:
            limit: Maximum signals to retrieve
            agent: Optional filter by agent type
            min_reward: Minimum reward threshold

        Returns:
            List of ReceivedSignal objects
        """
        async with self._lock:
            signals = []

            # Collect from specified agent or all agents
            buffers = {agent: self._buffers[agent]} if agent else self._buffers

            for _agent_name, buffer in buffers.items():
                for signal in buffer:
                    if signal.processed:
                        continue

                    # Check reward threshold
                    reward = signal.signal_data.get("reward", 0.0)
                    if reward < min_reward:
                        continue

                    signals.append(signal)

                    if len(signals) >= limit:
                        break

                if len(signals) >= limit:
                    break

            return signals

    async def mark_processed(
        self,
        signals: List[ReceivedSignal],
    ) -> int:
        """
        Mark signals as processed.

        Args:
            signals: List of signals to mark

        Returns:
            Number of signals marked
        """
        async with self._lock:
            count = 0
            for signal in signals:
                signal.processed = True
                count += 1
                self._stats["signals_processed"] += 1
            return count

    def get_signals_as_feedback_items(
        self,
        limit: int = 50,
    ) -> List[FeedbackItem]:
        """
        Convert pending signals to FeedbackItem format.

        This is used by the FeedbackCollectorNode to include
        training signals in the feedback learning cycle.

        Args:
            limit: Maximum items to return

        Returns:
            List of FeedbackItem dictionaries
        """
        feedback_items = []

        for agent_name, buffer in self._buffers.items():
            for signal in buffer:
                if signal.processed:
                    continue

                # Convert to FeedbackItem format
                signal_data = signal.signal_data

                feedback_item: FeedbackItem = {
                    "feedback_id": signal_data.get(
                        "signal_id",
                        f"{agent_name}_{signal.received_at}",
                    ),
                    "timestamp": signal.received_at,
                    "feedback_type": "training_signal",
                    "source_agent": agent_name,
                    "query": signal_data.get("input_context", {}).get("query", ""),
                    "agent_response": str(signal_data.get("output", {})),
                    "user_feedback": {
                        "reward": signal_data.get("reward", 0.0),
                        "outcome": signal_data.get("outcome_observed", {}),
                    },
                    "metadata": {
                        "dspy_type": signal_data.get("dspy_type", "sender"),
                        "latency_ms": signal_data.get("latency_ms", 0),
                        "raw_signal": signal_data,
                    },
                }

                feedback_items.append(feedback_item)

                if len(feedback_items) >= limit:
                    break

            if len(feedback_items) >= limit:
                break

        return feedback_items

    def get_training_data_for_optimization(
        self,
        agent: Tier2Agent,
        min_reward: float = 0.5,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get high-quality training data for DSPy optimization.

        Args:
            agent: Agent to get training data for
            min_reward: Minimum reward threshold
            limit: Maximum examples

        Returns:
            List of signal dictionaries suitable for DSPy training
        """
        if agent not in self._buffers:
            return []

        training_data = []
        buffer = self._buffers[agent]

        for signal in buffer:
            reward = signal.signal_data.get("reward", 0.0)
            if reward >= min_reward:
                training_data.append(signal.signal_data)

        # Sort by reward (best first) and limit
        training_data.sort(key=lambda x: x.get("reward", 0.0), reverse=True)
        return training_data[:limit]

    def get_statistics(self) -> Dict[str, Any]:
        """Get receiver statistics."""
        buffer_sizes = {agent: len(buffer) for agent, buffer in self._buffers.items()}

        return {
            **self._stats,
            "buffer_sizes": buffer_sizes,
            "total_buffered": sum(buffer_sizes.values()),
        }

    def clear_processed(self) -> int:
        """
        Remove processed signals from buffers.

        Returns:
            Number of signals removed
        """
        removed = 0

        for buffer in self._buffers.values():
            # Create new deque without processed signals
            unprocessed = [s for s in buffer if not s.processed]
            removed += len(buffer) - len(unprocessed)
            buffer.clear()
            buffer.extend(unprocessed)

        return removed


# =============================================================================
# 3. SINGLETON ACCESS
# =============================================================================

_receiver_instance: Optional[TrainingSignalReceiver] = None


def get_signal_receiver() -> TrainingSignalReceiver:
    """Get or create the singleton receiver."""
    global _receiver_instance
    if _receiver_instance is None:
        _receiver_instance = TrainingSignalReceiver()
    return _receiver_instance


def reset_signal_receiver() -> None:
    """Reset the singleton receiver (for testing)."""
    global _receiver_instance
    _receiver_instance = None


# =============================================================================
# 4. CONVENIENCE FUNCTIONS
# =============================================================================


async def receive_training_signals(
    signals_by_agent: Dict[str, List[Dict[str, Any]]],
) -> int:
    """
    Receive training signals from Tier2SignalRouter.

    This is the main entry point called by the router.

    Args:
        signals_by_agent: Dict mapping agent names to signal lists

    Returns:
        Number of signals received
    """
    receiver = get_signal_receiver()
    return await receiver.receive_signals(signals_by_agent)


def get_pending_training_signals(
    limit: int = 100,
    agent: Optional[Tier2Agent] = None,
) -> List[Dict[str, Any]]:
    """
    Get pending training signals as dictionaries.

    Args:
        limit: Maximum signals to retrieve
        agent: Optional filter by agent

    Returns:
        List of signal dictionaries
    """
    receiver = get_signal_receiver()

    # Use synchronous access for compatibility
    signals = []
    buffers = (
        {agent: receiver._buffers[agent]}
        if agent and agent in receiver._buffers
        else receiver._buffers
    )

    for _agent_name, buffer in buffers.items():
        for signal in buffer:
            if not signal.processed:
                signals.append(signal.signal_data)
                if len(signals) >= limit:
                    break
        if len(signals) >= limit:
            break

    return signals


def get_feedback_items_from_signals(limit: int = 50) -> List[FeedbackItem]:
    """
    Get training signals as FeedbackItem format.

    Used by FeedbackCollectorNode for feedback learning.

    Args:
        limit: Maximum items

    Returns:
        List of FeedbackItem dictionaries
    """
    receiver = get_signal_receiver()
    return receiver.get_signals_as_feedback_items(limit)


def get_training_data(
    agent: Tier2Agent,
    min_reward: float = 0.5,
    limit: int = 100,
) -> List[Dict[str, Any]]:
    """
    Get training data for DSPy optimization.

    Args:
        agent: Agent to get data for
        min_reward: Minimum reward
        limit: Maximum examples

    Returns:
        List of high-quality signal dictionaries
    """
    receiver = get_signal_receiver()
    return receiver.get_training_data_for_optimization(agent, min_reward, limit)


# =============================================================================
# 5. EXPORTS
# =============================================================================

__all__ = [
    # Types
    "Tier2Agent",
    "ReceivedSignal",
    # Receiver
    "TrainingSignalReceiver",
    # Access
    "get_signal_receiver",
    "reset_signal_receiver",
    # Functions
    "receive_training_signals",
    "get_pending_training_signals",
    "get_feedback_items_from_signals",
    "get_training_data",
]
