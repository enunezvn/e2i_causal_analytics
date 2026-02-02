"""
E2I Orchestrator Agent - DSPy Integration Module
Version: 4.2
Purpose: DSPy signatures and training signals for orchestrator Hub role

The Orchestrator is a DSPy Hub agent that:
1. Coordinates DSPy optimization across all agents
2. Collects training signals for AgentRoutingSignature optimization
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
class RoutingTrainingSignal:
    """
    Training signal for AgentRoutingSignature optimization.

    Captures routing decisions and their outcomes to train the
    DSPy signature that routes queries to appropriate agents.
    """

    # === Input Context ===
    signal_id: str = ""
    session_id: str = ""
    query: str = ""
    query_pattern: str = ""  # Classified query type
    intent: str = ""
    entities_extracted: List[str] = field(default_factory=list)

    # === Routing Decision ===
    agents_selected: List[str] = field(default_factory=list)
    routing_confidence: float = 0.0
    routing_rationale: str = ""

    # === Execution Outcome ===
    agents_succeeded: int = 0
    agents_failed: int = 0
    total_latency_ms: float = 0.0

    # === Quality Metrics (Delayed) ===
    user_satisfaction: Optional[float] = None  # 1-5 rating
    answer_quality: Optional[float] = None  # 0-1 score
    was_rerouted: bool = False  # Had to try different agents

    # === Timestamp ===
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def compute_reward(self) -> float:
        """
        Compute reward for MIPROv2 optimization.

        Weighting:
        - routing_accuracy: 0.35 (agents succeeded / selected)
        - efficiency: 0.25 (latency penalty)
        - no_rerouting: 0.20 (got it right first time)
        - user_satisfaction: 0.20 (if available)
        """
        reward = 0.0

        # Routing accuracy
        if self.agents_selected:
            accuracy = self.agents_succeeded / len(self.agents_selected)
            reward += 0.35 * accuracy

        # Efficiency (target < 5s total)
        target_latency = 5000
        if self.total_latency_ms > 0:
            efficiency = min(1.0, target_latency / self.total_latency_ms)
            reward += 0.25 * efficiency
        else:
            reward += 0.25  # No latency recorded

        # No rerouting bonus
        if not self.was_rerouted:
            reward += 0.20

        # User satisfaction
        if self.user_satisfaction is not None:
            satisfaction_score = (self.user_satisfaction - 1) / 4  # 1-5 to 0-1
            reward += 0.20 * satisfaction_score
        else:
            reward += 0.10  # Partial credit if no feedback

        return round(min(1.0, reward), 4)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "signal_id": self.signal_id or f"orch_{self.session_id}_{self.created_at}",
            "source_agent": "orchestrator",
            "dspy_type": "hub",
            "timestamp": self.created_at,
            "input_context": {
                "query": self.query[:500] if self.query else "",
                "query_pattern": self.query_pattern,
                "intent": self.intent,
                "entities_extracted": self.entities_extracted[:10],
            },
            "routing_decision": {
                "agents_selected": self.agents_selected,
                "routing_confidence": self.routing_confidence,
                "routing_rationale": self.routing_rationale[:200] if self.routing_rationale else "",
            },
            "outcome": {
                "agents_succeeded": self.agents_succeeded,
                "agents_failed": self.agents_failed,
                "total_latency_ms": self.total_latency_ms,
                "was_rerouted": self.was_rerouted,
            },
            "quality_metrics": {
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

    class AgentRoutingSignature(dspy.Signature):
        """
        Route queries to appropriate E2I agents.

        Given a user query and context, determine which agent(s) should
        handle the request and in what order.
        """

        query: str = dspy.InputField(desc="User query to route")
        query_pattern: str = dspy.InputField(
            desc="Classified query type: CAUSAL, COMPARATIVE, PREDICTIVE, etc."
        )
        entities: str = dspy.InputField(desc="Extracted entities from query")
        available_agents: str = dspy.InputField(desc="List of available agents with capabilities")

        primary_agent: str = dspy.OutputField(desc="Primary agent to handle query")
        secondary_agents: list = dspy.OutputField(
            desc="Secondary agents for additional context (may be empty)"
        )
        routing_confidence: float = dspy.OutputField(desc="Confidence in routing decision (0-1)")
        routing_rationale: str = dspy.OutputField(desc="Brief explanation of routing decision")

    class IntentClassificationSignature(dspy.Signature):
        """
        Classify query intent for routing.

        Determines the type of analysis or action the user is requesting.
        """

        query: str = dspy.InputField(desc="User query")
        conversation_context: str = dspy.InputField(desc="Recent conversation history")

        intent: str = dspy.OutputField(
            desc="Intent: CAUSAL_ANALYSIS, GAP_ANALYSIS, PREDICTION, COMPARISON, EXPERIMENT_DESIGN, HEALTH_CHECK, EXPLANATION"
        )
        sub_intent: str = dspy.OutputField(desc="More specific intent classification")
        confidence: float = dspy.OutputField(desc="Confidence in classification (0-1)")

    DSPY_AVAILABLE = True
    logger.info("DSPy signatures loaded for Orchestrator agent")

except ImportError:
    DSPY_AVAILABLE = False
    logger.warning("DSPy not available - using deterministic routing")
    AgentRoutingSignature = None
    IntentClassificationSignature = None


# =============================================================================
# 3. SIGNAL COLLECTOR
# =============================================================================


class OrchestratorSignalCollector:
    """
    Collects training signals from orchestrator routing decisions.

    The orchestrator is a Hub agent that coordinates DSPy optimization.
    """

    def __init__(self):
        self._signals_buffer: List[RoutingTrainingSignal] = []
        self._buffer_limit = 100

    def collect_routing_signal(
        self,
        session_id: str,
        query: str,
        query_pattern: str,
        intent: str,
        entities: List[str],
        agents_selected: List[str],
        routing_confidence: float,
        routing_rationale: str,
    ) -> RoutingTrainingSignal:
        """
        Collect training signal at routing decision time.

        Call this when the orchestrator makes a routing decision.
        """
        signal = RoutingTrainingSignal(
            session_id=session_id,
            query=query,
            query_pattern=query_pattern,
            intent=intent,
            entities_extracted=entities,
            agents_selected=agents_selected,
            routing_confidence=routing_confidence,
            routing_rationale=routing_rationale,
        )
        return signal

    def update_with_outcome(
        self,
        signal: RoutingTrainingSignal,
        agents_succeeded: int,
        agents_failed: int,
        total_latency_ms: float,
        was_rerouted: bool = False,
    ) -> RoutingTrainingSignal:
        """
        Update signal with execution outcome.

        Call this after agents have finished processing.
        """
        signal.agents_succeeded = agents_succeeded
        signal.agents_failed = agents_failed
        signal.total_latency_ms = total_latency_ms
        signal.was_rerouted = was_rerouted

        # Add to buffer
        self._signals_buffer.append(signal)
        if len(self._signals_buffer) > self._buffer_limit:
            self._signals_buffer.pop(0)

        return signal

    def update_with_feedback(
        self,
        signal: RoutingTrainingSignal,
        user_satisfaction: Optional[float] = None,
        answer_quality: Optional[float] = None,
    ) -> RoutingTrainingSignal:
        """
        Update signal with user feedback (delayed).

        Call this when user provides feedback on the response.
        """
        signal.user_satisfaction = user_satisfaction
        signal.answer_quality = answer_quality
        return signal

    def get_signals_for_training(
        self,
        min_reward: float = 0.0,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Get signals suitable for DSPy training."""
        signals = [s.to_dict() for s in self._signals_buffer if s.compute_reward() >= min_reward]
        return signals[-limit:]

    def clear_buffer(self):
        """Clear the signals buffer."""
        self._signals_buffer.clear()


# =============================================================================
# 4. HUB COORDINATION
# =============================================================================


class OrchestratorDSPyHub:
    """
    DSPy Hub coordination for the Orchestrator.

    Manages optimization requests and coordinates with feedback_learner.
    """

    def __init__(self):
        self.dspy_type: Literal["hub"] = "hub"
        self._pending_optimization_requests: List[Dict[str, Any]] = []

    async def request_optimization(
        self,
        agent_name: str,
        signature_name: str,
        training_signals: List[Dict[str, Any]],
        priority: Literal["low", "medium", "high"] = "medium",
    ) -> str:
        """
        Request optimization for an agent's DSPy signature.

        Routes to feedback_learner for actual optimization.
        """
        request = {
            "request_id": f"opt_{agent_name}_{datetime.now(timezone.utc).isoformat()}",
            "agent_name": agent_name,
            "signature_name": signature_name,
            "signal_count": len(training_signals),
            "priority": priority,
            "status": "pending",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        self._pending_optimization_requests.append(request)
        logger.info(f"Optimization requested for {agent_name}.{signature_name}")

        return request["request_id"]

    def get_pending_requests(self) -> List[Dict[str, Any]]:
        """Get pending optimization requests."""
        return [r for r in self._pending_optimization_requests if r["status"] == "pending"]


# =============================================================================
# 5. SINGLETON ACCESS
# =============================================================================

_signal_collector: Optional[OrchestratorSignalCollector] = None
_dspy_hub: Optional[OrchestratorDSPyHub] = None


def get_orchestrator_signal_collector() -> OrchestratorSignalCollector:
    """Get or create signal collector singleton."""
    global _signal_collector
    if _signal_collector is None:
        _signal_collector = OrchestratorSignalCollector()
    return _signal_collector


def get_orchestrator_dspy_hub() -> OrchestratorDSPyHub:
    """Get or create DSPy hub singleton."""
    global _dspy_hub
    if _dspy_hub is None:
        _dspy_hub = OrchestratorDSPyHub()
    return _dspy_hub


def reset_dspy_integration() -> None:
    """Reset singletons (for testing)."""
    global _signal_collector, _dspy_hub
    _signal_collector = None
    _dspy_hub = None


# =============================================================================
# 6. EXPORTS
# =============================================================================

__all__ = [
    # Training Signals
    "RoutingTrainingSignal",
    # DSPy Signatures
    "AgentRoutingSignature",
    "IntentClassificationSignature",
    "DSPY_AVAILABLE",
    # Collectors
    "OrchestratorSignalCollector",
    "OrchestratorDSPyHub",
    # Access
    "get_orchestrator_signal_collector",
    "get_orchestrator_dspy_hub",
    "reset_dspy_integration",
]
