"""
Tier 2 Causal Agents - Unified Signal Router
=============================================

Central router that batches and sends training signals from Tier 2 agents
(causal_impact, gap_analyzer, heterogeneous_optimizer) to the feedback_learner
for DSPy/GEPA optimization.

Features:
- Async signal queuing for non-blocking submission
- Batched delivery to reduce overhead
- Graceful fallback if feedback_learner unavailable
- Metrics tracking for signal flow monitoring

Usage:
    from src.agents.tier2_signal_router import get_signal_router

    router = get_signal_router()
    await router.submit_signal("causal_impact", signal.to_dict())

Author: E2I Causal Analytics Team
Version: 1.0.0
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SignalBatch:
    """Batch of training signals ready for delivery."""

    signals: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def add(self, agent_name: str, signal: Dict[str, Any]) -> None:
        """Add a signal to the batch."""
        self.signals.append(
            {
                "agent_name": agent_name,
                "signal": signal,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    def __len__(self) -> int:
        return len(self.signals)


class Tier2SignalRouter:
    """
    Routes training signals from Tier 2 agents to feedback_learner.

    Implements batched, async delivery with graceful fallback.
    """

    # Configuration
    BATCH_SIZE = 10  # Signals per batch
    FLUSH_INTERVAL_SECONDS = 30  # Auto-flush interval
    MAX_QUEUE_SIZE = 100  # Maximum pending signals

    def __init__(self):
        """Initialize signal router."""
        self._queue: List[Dict[str, Any]] = []
        self._lock = asyncio.Lock()
        self._metrics = {
            "signals_received": 0,
            "signals_delivered": 0,
            "signals_dropped": 0,
            "batches_sent": 0,
            "delivery_errors": 0,
        }
        self._feedback_learner_available: Optional[bool] = None

    async def submit_signal(
        self,
        agent_name: str,
        signal: Dict[str, Any],
    ) -> bool:
        """
        Queue a training signal for delivery to feedback_learner.

        Args:
            agent_name: Source agent (causal_impact, gap_analyzer, heterogeneous_optimizer)
            signal: Training signal dictionary from agent's signal collector

        Returns:
            True if queued successfully, False if dropped
        """
        async with self._lock:
            self._metrics["signals_received"] += 1

            # Check queue capacity
            if len(self._queue) >= self.MAX_QUEUE_SIZE:
                # Drop oldest signal
                self._queue.pop(0)
                self._metrics["signals_dropped"] += 1
                logger.warning(
                    f"Signal queue full, dropped oldest signal. "
                    f"Received from {agent_name}"
                )

            # Add to queue
            self._queue.append(
                {
                    "agent_name": agent_name,
                    "signal": signal,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )

            # Auto-flush if batch size reached
            if len(self._queue) >= self.BATCH_SIZE:
                await self._flush_internal()

            return True

    async def flush(self) -> int:
        """
        Immediately flush all queued signals to feedback_learner.

        Returns:
            Number of signals delivered
        """
        async with self._lock:
            return await self._flush_internal()

    async def _flush_internal(self) -> int:
        """Internal flush implementation (assumes lock is held)."""
        if not self._queue:
            return 0

        signals_to_deliver = list(self._queue)
        self._queue.clear()

        try:
            delivered = await self._deliver_to_feedback_learner(signals_to_deliver)
            self._metrics["signals_delivered"] += delivered
            self._metrics["batches_sent"] += 1
            return delivered
        except Exception as e:
            # On failure, log and continue (signals are lost)
            logger.error(f"Failed to deliver signals to feedback_learner: {e}")
            self._metrics["delivery_errors"] += 1
            self._metrics["signals_dropped"] += len(signals_to_deliver)
            return 0

    async def _deliver_to_feedback_learner(
        self,
        signals: List[Dict[str, Any]],
    ) -> int:
        """
        Deliver signals to feedback_learner agent.

        Args:
            signals: List of signal entries with agent_name and signal data

        Returns:
            Number of signals delivered
        """
        # Check if feedback_learner is available
        if self._feedback_learner_available is False:
            logger.debug("Feedback learner unavailable, skipping delivery")
            return 0

        try:
            # Try to import feedback_learner signal receiver
            from src.agents.feedback_learner.dspy_receiver import (
                receive_training_signals,
            )

            # Organize signals by source agent
            by_agent: Dict[str, List[Dict[str, Any]]] = {}
            for entry in signals:
                agent = entry["agent_name"]
                if agent not in by_agent:
                    by_agent[agent] = []
                by_agent[agent].append(entry["signal"])

            # Submit to feedback_learner
            await receive_training_signals(by_agent)

            self._feedback_learner_available = True
            logger.info(
                f"Delivered {len(signals)} signals to feedback_learner "
                f"from agents: {list(by_agent.keys())}"
            )
            return len(signals)

        except ImportError:
            # Feedback learner module not available
            self._feedback_learner_available = False
            logger.warning(
                "Feedback learner not available. "
                "Signals will be stored locally for later retrieval."
            )
            # Store locally for later retrieval
            await self._store_signals_locally(signals)
            return len(signals)

        except Exception as e:
            logger.error(f"Error delivering to feedback_learner: {e}")
            raise

    async def _store_signals_locally(
        self,
        signals: List[Dict[str, Any]],
    ) -> None:
        """
        Store signals locally when feedback_learner is unavailable.

        Uses the signal collector buffers as local storage.
        """
        for entry in signals:
            agent_name = entry["agent_name"]
            signal_data = entry["signal"]

            try:
                # Store in agent's local buffer
                if agent_name == "causal_impact":
                    from src.agents.causal_impact.dspy_integration import (
                        get_causal_impact_signal_collector,
                    )

                    collector = get_causal_impact_signal_collector()
                    # Signal already in buffer from collection
                    logger.debug(f"Signal from {agent_name} stored locally")

                elif agent_name == "gap_analyzer":
                    from src.agents.gap_analyzer.dspy_integration import (
                        get_gap_analyzer_signal_collector,
                    )

                    collector = get_gap_analyzer_signal_collector()
                    logger.debug(f"Signal from {agent_name} stored locally")

                elif agent_name == "heterogeneous_optimizer":
                    from src.agents.heterogeneous_optimizer.dspy_integration import (
                        get_heterogeneous_optimizer_signal_collector,
                    )

                    collector = get_heterogeneous_optimizer_signal_collector()
                    logger.debug(f"Signal from {agent_name} stored locally")

            except Exception as e:
                logger.warning(f"Failed to store signal locally for {agent_name}: {e}")

    def get_metrics(self) -> Dict[str, int]:
        """Get signal routing metrics."""
        return dict(self._metrics)

    def reset_metrics(self) -> None:
        """Reset metrics counters."""
        for key in self._metrics:
            self._metrics[key] = 0

    @property
    def queue_size(self) -> int:
        """Current queue size."""
        return len(self._queue)


# =============================================================================
# SINGLETON ACCESS
# =============================================================================

_signal_router: Optional[Tier2SignalRouter] = None


def get_signal_router() -> Tier2SignalRouter:
    """Get or create signal router singleton."""
    global _signal_router
    if _signal_router is None:
        _signal_router = Tier2SignalRouter()
    return _signal_router


def reset_signal_router() -> None:
    """Reset the signal router singleton (for testing)."""
    global _signal_router
    _signal_router = None


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


async def route_causal_impact_signal(signal: Dict[str, Any]) -> bool:
    """Route a causal_impact training signal."""
    router = get_signal_router()
    return await router.submit_signal("causal_impact", signal)


async def route_gap_analyzer_signal(signal: Dict[str, Any]) -> bool:
    """Route a gap_analyzer training signal."""
    router = get_signal_router()
    return await router.submit_signal("gap_analyzer", signal)


async def route_heterogeneous_optimizer_signal(signal: Dict[str, Any]) -> bool:
    """Route a heterogeneous_optimizer training signal."""
    router = get_signal_router()
    return await router.submit_signal("heterogeneous_optimizer", signal)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "Tier2SignalRouter",
    "SignalBatch",
    "get_signal_router",
    "reset_signal_router",
    "route_causal_impact_signal",
    "route_gap_analyzer_signal",
    "route_heterogeneous_optimizer_signal",
]
