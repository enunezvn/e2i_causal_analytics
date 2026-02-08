"""
E2I Experiment Designer Agent - DSPy Integration Module
Version: 4.2
Purpose: DSPy signatures and training signals for experiment_designer Sender role

The Experiment Designer agent is a DSPy Sender agent that:
1. Generates training signals from experiment design sessions
2. Provides InvestigationPlanSignature training examples
3. Routes high-quality signals to feedback_learner for optimization
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
class ExperimentDesignTrainingSignal:
    """
    Training signal for Experiment Designer DSPy optimization.

    Captures experiment design decisions and their outcomes to train:
    - InvestigationPlanSignature: Planning investigation approach
    - DesignReasoningSignature: Choosing experiment design type
    - ValidityAssessmentSignature: Identifying threats to validity
    """

    # === Input Context ===
    signal_id: str = ""
    session_id: str = ""
    business_question: str = ""
    preregistration_formality: str = ""  # light, medium, heavy
    max_redesign_iterations: int = 3

    # === Design Reasoning Phase ===
    design_type_chosen: str = ""  # RCT, quasi_experiment, etc.
    treatments_count: int = 0
    outcomes_count: int = 0
    randomization_unit: str = ""  # individual, cluster, etc.

    # === Power Analysis Phase ===
    required_sample_size: int = 0
    achieved_power: float = 0.0
    minimum_detectable_effect: float = 0.0
    duration_estimate_days: int = 0

    # === Validity Audit Phase ===
    validity_threats_identified: int = 0
    critical_threats: int = 0
    mitigations_proposed: int = 0
    overall_validity_score: float = 0.0
    redesign_iterations: int = 0

    # === Template Generation ===
    template_generated: bool = False
    causal_graph_generated: bool = False
    analysis_code_generated: bool = False

    # === Outcome Metrics ===
    total_llm_tokens_used: int = 0
    total_latency_ms: float = 0.0
    experiment_approved: Optional[bool] = None  # Later validation
    user_satisfaction: Optional[float] = None  # 1-5 rating

    # === Timestamp ===
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def compute_reward(self) -> float:
        """
        Compute reward for MIPROv2 optimization.

        Weighting:
        - design_quality: 0.25 (appropriate design for question)
        - power_quality: 0.25 (adequate power with reasonable sample)
        - validity_handling: 0.20 (threats identified and mitigated)
        - completeness: 0.15 (all artifacts generated)
        - approval/satisfaction: 0.15 (if available)
        """
        reward = 0.0

        # Design quality
        design_score = 0.0
        if self.design_type_chosen:
            design_score += 0.4  # Has a design
            # Appropriate number of treatments and outcomes
            if 1 <= self.treatments_count <= 4:
                design_score += 0.3
            if 1 <= self.outcomes_count <= 5:
                design_score += 0.3
        reward += 0.25 * design_score

        # Power quality
        power_score = 0.0
        if self.achieved_power >= 0.80:
            power_score += 0.5  # Adequate power
        elif self.achieved_power >= 0.70:
            power_score += 0.3
        # Reasonable sample size (not extreme)
        if 50 <= self.required_sample_size <= 10000:
            power_score += 0.3
        # Reasonable duration
        if 7 <= self.duration_estimate_days <= 365:
            power_score += 0.2
        reward += 0.25 * power_score

        # Validity handling
        validity_score = 0.0
        if self.validity_threats_identified > 0:
            # Found threats
            validity_score += 0.3
            # Mitigated critical threats
            if self.critical_threats > 0:
                mitigation_rate = self.mitigations_proposed / max(1, self.critical_threats)
                validity_score += 0.4 * min(1.0, mitigation_rate)
            else:
                validity_score += 0.4  # No critical threats is good
            # Good validity score
            if self.overall_validity_score >= 0.7:
                validity_score += 0.3
        else:
            validity_score = 0.5  # May have missed threats
        reward += 0.20 * validity_score

        # Completeness
        completeness_score = 0.0
        if self.template_generated:
            completeness_score += 0.4
        if self.causal_graph_generated:
            completeness_score += 0.3
        if self.analysis_code_generated:
            completeness_score += 0.3
        reward += 0.15 * completeness_score

        # Approval / Satisfaction
        if self.experiment_approved is not None:
            reward += 0.15 if self.experiment_approved else 0.0
        elif self.user_satisfaction is not None:
            satisfaction_score = (self.user_satisfaction - 1) / 4  # 1-5 to 0-1
            reward += 0.15 * satisfaction_score
        else:
            reward += 0.075  # Partial credit

        return round(min(1.0, reward), 4)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "signal_id": self.signal_id or f"ed_{self.session_id}_{self.created_at}",
            "source_agent": "experiment_designer",
            "dspy_type": "sender",
            "timestamp": self.created_at,
            "input_context": {
                "business_question": self.business_question[:500] if self.business_question else "",
                "preregistration_formality": self.preregistration_formality,
                "max_redesign_iterations": self.max_redesign_iterations,
            },
            "design_reasoning": {
                "design_type_chosen": self.design_type_chosen,
                "treatments_count": self.treatments_count,
                "outcomes_count": self.outcomes_count,
                "randomization_unit": self.randomization_unit,
            },
            "power_analysis": {
                "required_sample_size": self.required_sample_size,
                "achieved_power": self.achieved_power,
                "minimum_detectable_effect": self.minimum_detectable_effect,
                "duration_estimate_days": self.duration_estimate_days,
            },
            "validity_audit": {
                "validity_threats_identified": self.validity_threats_identified,
                "critical_threats": self.critical_threats,
                "mitigations_proposed": self.mitigations_proposed,
                "overall_validity_score": self.overall_validity_score,
                "redesign_iterations": self.redesign_iterations,
            },
            "template_generation": {
                "template_generated": self.template_generated,
                "causal_graph_generated": self.causal_graph_generated,
                "analysis_code_generated": self.analysis_code_generated,
            },
            "outcome": {
                "total_llm_tokens_used": self.total_llm_tokens_used,
                "total_latency_ms": self.total_latency_ms,
                "experiment_approved": self.experiment_approved,
                "user_satisfaction": self.user_satisfaction,
            },
            "reward": self.compute_reward(),
        }


# =============================================================================
# 2. DSPy SIGNATURES
# =============================================================================

try:
    import dspy

    class DesignReasoningSignature(dspy.Signature):
        """
        Reason about appropriate experiment design type.

        Given a business question and constraints, determine the best
        experimental design approach.
        """

        business_question: str = dspy.InputField(desc="The causal question to answer")
        constraints: str = dspy.InputField(desc="Budget, timeline, and ethical constraints")
        available_data: str = dspy.InputField(desc="Available data sources and variables")
        historical_experiments: str = dspy.InputField(desc="Similar past experiments")

        design_type: str = dspy.OutputField(
            desc="RCT, quasi_experiment, difference_in_differences, etc."
        )
        design_rationale: str = dspy.OutputField(desc="Reasoning for chosen design")
        randomization_unit: str = dspy.OutputField(
            desc="individual, cluster, time_period, geography"
        )
        key_assumptions: list = dspy.OutputField(desc="Assumptions required for causal inference")

    class InvestigationPlanSignature(dspy.Signature):
        """
        Plan a multi-hop investigation for experiment design.

        Determine what context and data are needed to design
        a rigorous experiment.
        """

        business_question: str = dspy.InputField(desc="The causal question")
        initial_context: str = dspy.InputField(desc="What we know so far")
        formality_level: str = dspy.InputField(desc="light, medium, or heavy preregistration")

        investigation_steps: list = dspy.OutputField(desc="Ordered steps for investigation")
        data_requirements: list = dspy.OutputField(desc="Data needed for each step")
        expected_outputs: list = dspy.OutputField(desc="What each step should produce")
        parallel_possible: list = dspy.OutputField(desc="Steps that can run in parallel")

    class ValidityAssessmentSignature(dspy.Signature):
        """
        Assess threats to experimental validity.

        Given an experiment design, identify potential threats
        to internal and external validity.
        """

        design_summary: str = dspy.InputField(desc="Summary of experiment design")
        treatments: str = dspy.InputField(desc="Treatment arms description")
        outcomes: str = dspy.InputField(desc="Primary and secondary outcomes")
        randomization_method: str = dspy.InputField(desc="How randomization is done")

        validity_threats: list = dspy.OutputField(
            desc="List of threats with type, severity, and description"
        )
        mitigations: list = dspy.OutputField(desc="Proposed mitigations for each threat")
        overall_validity_score: float = dspy.OutputField(desc="0-1 validity confidence")
        redesign_needed: bool = dspy.OutputField(desc="Whether design should be revised")

    DSPY_AVAILABLE = True
    logger.info("DSPy signatures loaded for Experiment Designer agent")

except ImportError:
    DSPY_AVAILABLE = False
    logger.warning("DSPy not available - using deterministic design")
    DesignReasoningSignature = None  # type: ignore[assignment,misc]
    InvestigationPlanSignature = None  # type: ignore[assignment,misc]
    ValidityAssessmentSignature = None  # type: ignore[assignment,misc]


# =============================================================================
# 3. SIGNAL COLLECTOR
# =============================================================================


class ExperimentDesignerSignalCollector:
    """
    Collects training signals from experiment design sessions.

    The Experiment Designer agent is a Sender that generates signals
    for InvestigationPlanSignature optimization.
    """

    def __init__(self):
        self.dspy_type: Literal["sender"] = "sender"
        self._signals_buffer: List[ExperimentDesignTrainingSignal] = []
        self._buffer_limit = 100

    def collect_design_signal(
        self,
        session_id: str,
        business_question: str,
        preregistration_formality: str,
        max_redesign_iterations: int,
    ) -> ExperimentDesignTrainingSignal:
        """
        Initialize training signal at design session start.

        Call this when starting a new experiment design.
        """
        signal = ExperimentDesignTrainingSignal(
            session_id=session_id,
            business_question=business_question,
            preregistration_formality=preregistration_formality,
            max_redesign_iterations=max_redesign_iterations,
        )
        return signal

    def update_design_reasoning(
        self,
        signal: ExperimentDesignTrainingSignal,
        design_type_chosen: str,
        treatments_count: int,
        outcomes_count: int,
        randomization_unit: str,
    ) -> ExperimentDesignTrainingSignal:
        """Update signal with design reasoning results."""
        signal.design_type_chosen = design_type_chosen
        signal.treatments_count = treatments_count
        signal.outcomes_count = outcomes_count
        signal.randomization_unit = randomization_unit
        return signal

    def update_power_analysis(
        self,
        signal: ExperimentDesignTrainingSignal,
        required_sample_size: int,
        achieved_power: float,
        minimum_detectable_effect: float,
        duration_estimate_days: int,
    ) -> ExperimentDesignTrainingSignal:
        """Update signal with power analysis results."""
        signal.required_sample_size = required_sample_size
        signal.achieved_power = achieved_power
        signal.minimum_detectable_effect = minimum_detectable_effect
        signal.duration_estimate_days = duration_estimate_days
        return signal

    def update_validity_audit(
        self,
        signal: ExperimentDesignTrainingSignal,
        validity_threats_identified: int,
        critical_threats: int,
        mitigations_proposed: int,
        overall_validity_score: float,
        redesign_iterations: int,
    ) -> ExperimentDesignTrainingSignal:
        """Update signal with validity audit results."""
        signal.validity_threats_identified = validity_threats_identified
        signal.critical_threats = critical_threats
        signal.mitigations_proposed = mitigations_proposed
        signal.overall_validity_score = overall_validity_score
        signal.redesign_iterations = redesign_iterations
        return signal

    def update_template_generation(
        self,
        signal: ExperimentDesignTrainingSignal,
        template_generated: bool,
        causal_graph_generated: bool,
        analysis_code_generated: bool,
        total_llm_tokens_used: int,
        total_latency_ms: float,
    ) -> ExperimentDesignTrainingSignal:
        """Update signal with template generation results."""
        signal.template_generated = template_generated
        signal.causal_graph_generated = causal_graph_generated
        signal.analysis_code_generated = analysis_code_generated
        signal.total_llm_tokens_used = total_llm_tokens_used
        signal.total_latency_ms = total_latency_ms

        # Add to buffer
        self._signals_buffer.append(signal)
        if len(self._signals_buffer) > self._buffer_limit:
            self._signals_buffer.pop(0)

        return signal

    def update_with_approval(
        self,
        signal: ExperimentDesignTrainingSignal,
        experiment_approved: bool,
        user_satisfaction: Optional[float] = None,
    ) -> ExperimentDesignTrainingSignal:
        """Update signal with approval decision (delayed feedback)."""
        signal.experiment_approved = experiment_approved
        signal.user_satisfaction = user_satisfaction
        return signal

    def get_signals_for_training(
        self,
        min_reward: float = 0.0,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Get signals suitable for DSPy training."""
        signals = [s.to_dict() for s in self._signals_buffer if s.compute_reward() >= min_reward]
        return signals[-limit:]

    def get_approved_examples(
        self,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Get examples from approved experiments (highest quality)."""
        signals = [s for s in self._signals_buffer if s.experiment_approved is True]
        sorted_signals = sorted(signals, key=lambda s: s.compute_reward(), reverse=True)
        return [s.to_dict() for s in sorted_signals[:limit]]

    def clear_buffer(self):
        """Clear the signals buffer."""
        self._signals_buffer.clear()


# =============================================================================
# 4. SINGLETON ACCESS
# =============================================================================

_signal_collector: Optional[ExperimentDesignerSignalCollector] = None


def get_experiment_designer_signal_collector() -> ExperimentDesignerSignalCollector:
    """Get or create signal collector singleton."""
    global _signal_collector
    if _signal_collector is None:
        _signal_collector = ExperimentDesignerSignalCollector()
    return _signal_collector


def reset_dspy_integration() -> None:
    """Reset singletons (for testing)."""
    global _signal_collector
    _signal_collector = None


# =============================================================================
# 5. EXPORTS
# =============================================================================

__all__ = [
    # Training Signals
    "ExperimentDesignTrainingSignal",
    # DSPy Signatures
    "DesignReasoningSignature",
    "InvestigationPlanSignature",
    "ValidityAssessmentSignature",
    "DSPY_AVAILABLE",
    # Collectors
    "ExperimentDesignerSignalCollector",
    # Access
    "get_experiment_designer_signal_collector",
    "reset_dspy_integration",
]
