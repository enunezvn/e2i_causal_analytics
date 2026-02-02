"""
E2I Causal Impact Agent - DSPy Integration Module
Version: 4.2
Purpose: DSPy signatures and training signals for causal_impact Sender role

The Causal Impact agent is a DSPy Sender agent that:
1. Generates training signals from causal analysis executions
2. Provides EvidenceSynthesisSignature training examples
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
class CausalAnalysisTrainingSignal:
    """
    Training signal for Causal Impact DSPy optimization.

    Captures causal analysis decisions and their outcomes to train:
    - EvidenceSynthesisSignature: Synthesizing causal evidence
    - CausalInterpretationSignature: Generating interpretations
    """

    # === Input Context ===
    signal_id: str = ""
    session_id: str = ""
    query: str = ""
    treatment_var: str = ""
    outcome_var: str = ""
    confounders_count: int = 0

    # === Graph Building Phase ===
    dag_nodes_count: int = 0
    dag_edges_count: int = 0
    adjustment_sets_found: int = 0
    graph_confidence: float = 0.0

    # === Estimation Phase ===
    estimation_method: str = ""
    ate_estimate: float = 0.0
    ate_ci_width: float = 0.0  # Width of confidence interval
    statistical_significance: bool = False
    effect_size: str = ""  # small, medium, large
    sample_size: int = 0

    # === V4.2 Enhancement: Energy Score Selection ===
    energy_score_enabled: bool = False
    selection_strategy: str = ""  # first_success, best_energy, ensemble
    selected_estimator: str = ""  # causal_forest, linear_dml, drlearner, ols
    energy_score: float = 0.0  # Selected estimator's energy score (0-1)
    energy_score_gap: float = 0.0  # Gap between best and second-best
    n_estimators_evaluated: int = 0
    n_estimators_succeeded: int = 0

    # === Refutation Phase ===
    refutation_tests_passed: int = 0
    refutation_tests_failed: int = 0
    overall_robust: bool = False

    # === Sensitivity Phase ===
    e_value: float = 0.0
    robust_to_confounding: bool = False

    # === Interpretation Phase ===
    interpretation_depth: str = ""  # none, minimal, standard, deep
    narrative_length: int = 0
    key_findings_count: int = 0
    recommendations_count: int = 0

    # === Outcome Metrics ===
    total_latency_ms: float = 0.0
    confidence_score: float = 0.0
    user_satisfaction: Optional[float] = None  # 1-5 rating

    # === Timestamp ===
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def compute_reward(self) -> float:
        """
        Compute reward for MIPROv2 optimization.

        Weighting (V4.2 Updated):
        - refutation_robustness: 0.28 (passed / total)
        - estimation_quality: 0.24 (significance + CI width)
        - interpretation_quality: 0.18 (depth + findings)
        - efficiency: 0.15 (latency)
        - user_satisfaction: 0.10 (if available)
        - energy_score_quality: 0.05 (lower energy score = better)
        """
        reward = 0.0

        # Refutation robustness (28%)
        total_tests = self.refutation_tests_passed + self.refutation_tests_failed
        if total_tests > 0:
            robustness = self.refutation_tests_passed / total_tests
            reward += 0.28 * robustness

        # Estimation quality (24%)
        estimation_score = 0.0
        if self.statistical_significance:
            estimation_score += 0.5
        # Narrower CI is better (target CI width < 20% of estimate)
        if self.ate_estimate != 0 and self.ate_ci_width > 0:
            relative_width = abs(self.ate_ci_width / self.ate_estimate)
            ci_quality = min(1.0, 0.2 / relative_width) if relative_width > 0 else 0.5
            estimation_score += 0.5 * ci_quality
        else:
            estimation_score += 0.25
        reward += 0.24 * estimation_score

        # Interpretation quality (18%)
        interp_score = 0.0
        depth_scores = {"none": 0.0, "minimal": 0.3, "standard": 0.7, "deep": 1.0}
        interp_score += 0.5 * depth_scores.get(self.interpretation_depth, 0.0)
        # Quality proxy: findings + recommendations
        finding_quality = min(1.0, (self.key_findings_count + self.recommendations_count) / 8)
        interp_score += 0.5 * finding_quality
        reward += 0.18 * interp_score

        # Efficiency (15%, target < 15s for full analysis)
        target_latency = 15000
        if self.total_latency_ms > 0:
            efficiency = min(1.0, target_latency / self.total_latency_ms)
            reward += 0.15 * efficiency
        else:
            reward += 0.15

        # User satisfaction (10%)
        if self.user_satisfaction is not None:
            satisfaction_score = (self.user_satisfaction - 1) / 4  # 1-5 to 0-1
            reward += 0.10 * satisfaction_score
        else:
            reward += 0.05  # Partial credit

        # V4.2 Enhancement: Energy Score Quality (5%)
        # Lower energy score = better quality (0-1 scale, lower is better)
        if self.energy_score_enabled and self.energy_score > 0:
            energy_quality = 1.0 - min(1.0, self.energy_score)
            reward += 0.05 * energy_quality
        elif not self.energy_score_enabled:
            # Partial credit for legacy path
            reward += 0.025

        return round(min(1.0, reward), 4)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "signal_id": self.signal_id or f"ci_{self.session_id}_{self.created_at}",
            "source_agent": "causal_impact",
            "dspy_type": "sender",
            "timestamp": self.created_at,
            "input_context": {
                "query": self.query[:500] if self.query else "",
                "treatment_var": self.treatment_var,
                "outcome_var": self.outcome_var,
                "confounders_count": self.confounders_count,
            },
            "graph_building": {
                "dag_nodes_count": self.dag_nodes_count,
                "dag_edges_count": self.dag_edges_count,
                "adjustment_sets_found": self.adjustment_sets_found,
                "graph_confidence": self.graph_confidence,
            },
            "estimation": {
                "method": self.estimation_method,
                "ate_estimate": self.ate_estimate,
                "ate_ci_width": self.ate_ci_width,
                "statistical_significance": self.statistical_significance,
                "effect_size": self.effect_size,
                "sample_size": self.sample_size,
                # V4.2 Enhancement: Energy Score Selection
                "energy_score_enabled": self.energy_score_enabled,
                "selection_strategy": self.selection_strategy,
                "selected_estimator": self.selected_estimator,
                "energy_score": self.energy_score,
                "energy_score_gap": self.energy_score_gap,
                "n_estimators_evaluated": self.n_estimators_evaluated,
                "n_estimators_succeeded": self.n_estimators_succeeded,
            },
            "refutation": {
                "tests_passed": self.refutation_tests_passed,
                "tests_failed": self.refutation_tests_failed,
                "overall_robust": self.overall_robust,
            },
            "sensitivity": {
                "e_value": self.e_value,
                "robust_to_confounding": self.robust_to_confounding,
            },
            "interpretation": {
                "depth": self.interpretation_depth,
                "narrative_length": self.narrative_length,
                "key_findings_count": self.key_findings_count,
                "recommendations_count": self.recommendations_count,
            },
            "outcome": {
                "total_latency_ms": self.total_latency_ms,
                "confidence_score": self.confidence_score,
                "user_satisfaction": self.user_satisfaction,
            },
            "reward": self.compute_reward(),
        }


# =============================================================================
# 2. DSPy SIGNATURES
# =============================================================================

try:
    import dspy

    class CausalGraphSignature(dspy.Signature):
        """
        Construct a causal DAG from domain knowledge and data.

        Given treatment and outcome variables, construct a plausible
        causal graph including confounders and adjustment sets.
        """

        treatment_var: str = dspy.InputField(desc="Treatment variable name")
        outcome_var: str = dspy.InputField(desc="Outcome variable name")
        available_vars: str = dspy.InputField(desc="All available variables in data")
        domain_context: str = dspy.InputField(desc="Domain knowledge context")

        confounders: list = dspy.OutputField(desc="Variables that confound treatment-outcome")
        mediators: list = dspy.OutputField(desc="Variables that mediate the effect")
        adjustment_set: list = dspy.OutputField(desc="Recommended adjustment set for estimation")
        graph_rationale: str = dspy.OutputField(desc="Explanation of graph structure")

    class EvidenceSynthesisSignature(dspy.Signature):
        """
        Synthesize causal evidence into an interpretation.

        Given estimation results, refutation tests, and sensitivity analysis,
        produce a coherent interpretation of the causal effect.
        """

        estimation_result: str = dspy.InputField(desc="ATE estimate with CI and significance")
        refutation_summary: str = dspy.InputField(desc="Summary of refutation test results")
        sensitivity_summary: str = dspy.InputField(desc="E-value and sensitivity interpretation")
        user_expertise: str = dspy.InputField(desc="User expertise level")

        narrative: str = dspy.OutputField(desc="Natural language interpretation")
        key_findings: list = dspy.OutputField(desc="3-5 key findings as bullet points")
        confidence_level: str = dspy.OutputField(desc="low, medium, or high")
        recommendations: list = dspy.OutputField(desc="Actionable recommendations")
        limitations: list = dspy.OutputField(desc="Important caveats and limitations")

    class CausalInterpretationSignature(dspy.Signature):
        """
        Generate deep causal interpretation for domain experts.

        Provides detailed explanation of the causal mechanism,
        assumptions, and implications.
        """

        treatment_var: str = dspy.InputField(desc="Treatment variable")
        outcome_var: str = dspy.InputField(desc="Outcome variable")
        ate_estimate: float = dspy.InputField(desc="Point estimate of causal effect")
        effect_size: str = dspy.InputField(desc="Effect magnitude category")
        refutation_passed: bool = dspy.InputField(desc="Whether refutation tests passed")

        mechanism_explanation: str = dspy.OutputField(desc="Explanation of the causal mechanism")
        assumption_warnings: list = dspy.OutputField(desc="Key assumptions that may be violated")
        practical_significance: str = dspy.OutputField(desc="What this means in practice")

    class CausalImpactModule(dspy.Module):
        """
        DSPy Module for Causal Impact agent optimization.

        Wraps the EvidenceSynthesisSignature for GEPA optimization,
        enabling prompt tuning for improved causal interpretation quality.
        """

        def __init__(self):
            super().__init__()
            self.evidence_synthesis = dspy.ChainOfThought(EvidenceSynthesisSignature)
            self.causal_interpretation = dspy.ChainOfThought(CausalInterpretationSignature)

        def forward(
            self,
            estimation_result: str,
            refutation_summary: str,
            sensitivity_summary: str,
            user_expertise: str = "intermediate",
            treatment_var: str = "",
            outcome_var: str = "",
            ate_estimate: float = 0.0,
            effect_size: str = "medium",
            refutation_passed: bool = True,
        ) -> dspy.Prediction:
            """
            Generate causal interpretation from analysis results.

            Args:
                estimation_result: ATE estimate with CI and significance
                refutation_summary: Summary of refutation test results
                sensitivity_summary: E-value and sensitivity interpretation
                user_expertise: User expertise level (novice, intermediate, expert)
                treatment_var: Treatment variable name (for deep interpretation)
                outcome_var: Outcome variable name (for deep interpretation)
                ate_estimate: Point estimate of causal effect
                effect_size: Effect magnitude category
                refutation_passed: Whether refutation tests passed

            Returns:
                DSPy Prediction with narrative, findings, recommendations
            """
            # Primary synthesis using ChainOfThought
            synthesis = self.evidence_synthesis(
                estimation_result=estimation_result,
                refutation_summary=refutation_summary,
                sensitivity_summary=sensitivity_summary,
                user_expertise=user_expertise,
            )

            # Add deep interpretation for experts if context provided
            if user_expertise == "expert" and treatment_var and outcome_var:
                interpretation = self.causal_interpretation(
                    treatment_var=treatment_var,
                    outcome_var=outcome_var,
                    ate_estimate=ate_estimate,
                    effect_size=effect_size,
                    refutation_passed=refutation_passed,
                )
                synthesis.mechanism_explanation = interpretation.mechanism_explanation
                synthesis.assumption_warnings = interpretation.assumption_warnings
                synthesis.practical_significance = interpretation.practical_significance

            return synthesis

    DSPY_AVAILABLE = True
    logger.info("DSPy signatures loaded for Causal Impact agent")

except ImportError:
    DSPY_AVAILABLE = False
    logger.warning("DSPy not available - using deterministic interpretation")
    CausalGraphSignature = None
    EvidenceSynthesisSignature = None
    CausalInterpretationSignature = None
    CausalImpactModule = None


# Module singleton for optimization
_causal_impact_module: Optional["CausalImpactModule"] = None


def get_causal_impact_module() -> "CausalImpactModule":
    """
    Get or create CausalImpactModule singleton for GEPA optimization.

    Returns:
        CausalImpactModule instance ready for optimization

    Raises:
        RuntimeError: If DSPy is not available
    """
    global _causal_impact_module

    if not DSPY_AVAILABLE:
        raise RuntimeError("DSPy is not available. Install with: pip install dspy-ai")

    if _causal_impact_module is None:
        _causal_impact_module = CausalImpactModule()

    return _causal_impact_module


# =============================================================================
# 3. SIGNAL COLLECTOR
# =============================================================================


class CausalImpactSignalCollector:
    """
    Collects training signals from causal analysis executions.

    The Causal Impact agent is a Sender that generates signals
    for EvidenceSynthesisSignature optimization.
    """

    def __init__(self):
        self.dspy_type: Literal["sender"] = "sender"
        self._signals_buffer: List[CausalAnalysisTrainingSignal] = []
        self._buffer_limit = 100

    def collect_analysis_signal(
        self,
        session_id: str,
        query: str,
        treatment_var: str,
        outcome_var: str,
        confounders_count: int,
    ) -> CausalAnalysisTrainingSignal:
        """
        Initialize training signal at analysis start.

        Call this when starting a new causal analysis.
        """
        signal = CausalAnalysisTrainingSignal(
            session_id=session_id,
            query=query,
            treatment_var=treatment_var,
            outcome_var=outcome_var,
            confounders_count=confounders_count,
        )
        return signal

    def update_graph_building(
        self,
        signal: CausalAnalysisTrainingSignal,
        dag_nodes_count: int,
        dag_edges_count: int,
        adjustment_sets_found: int,
        graph_confidence: float,
    ) -> CausalAnalysisTrainingSignal:
        """Update signal with graph building phase results."""
        signal.dag_nodes_count = dag_nodes_count
        signal.dag_edges_count = dag_edges_count
        signal.adjustment_sets_found = adjustment_sets_found
        signal.graph_confidence = graph_confidence
        return signal

    def update_estimation(
        self,
        signal: CausalAnalysisTrainingSignal,
        method: str,
        ate_estimate: float,
        ate_ci_lower: float,
        ate_ci_upper: float,
        statistical_significance: bool,
        effect_size: str,
        sample_size: int,
    ) -> CausalAnalysisTrainingSignal:
        """Update signal with estimation phase results."""
        signal.estimation_method = method
        signal.ate_estimate = ate_estimate
        signal.ate_ci_width = ate_ci_upper - ate_ci_lower
        signal.statistical_significance = statistical_significance
        signal.effect_size = effect_size
        signal.sample_size = sample_size
        return signal

    def update_energy_score(
        self,
        signal: CausalAnalysisTrainingSignal,
        energy_score_enabled: bool,
        selection_strategy: str = "",
        selected_estimator: str = "",
        energy_score: float = 0.0,
        energy_score_gap: float = 0.0,
        n_estimators_evaluated: int = 0,
        n_estimators_succeeded: int = 0,
    ) -> CausalAnalysisTrainingSignal:
        """Update signal with V4.2 energy score selection results."""
        signal.energy_score_enabled = energy_score_enabled
        signal.selection_strategy = selection_strategy
        signal.selected_estimator = selected_estimator
        signal.energy_score = energy_score
        signal.energy_score_gap = energy_score_gap
        signal.n_estimators_evaluated = n_estimators_evaluated
        signal.n_estimators_succeeded = n_estimators_succeeded
        return signal

    def update_refutation(
        self,
        signal: CausalAnalysisTrainingSignal,
        tests_passed: int,
        tests_failed: int,
        overall_robust: bool,
    ) -> CausalAnalysisTrainingSignal:
        """Update signal with refutation phase results."""
        signal.refutation_tests_passed = tests_passed
        signal.refutation_tests_failed = tests_failed
        signal.overall_robust = overall_robust
        return signal

    def update_sensitivity(
        self,
        signal: CausalAnalysisTrainingSignal,
        e_value: float,
        robust_to_confounding: bool,
    ) -> CausalAnalysisTrainingSignal:
        """Update signal with sensitivity phase results."""
        signal.e_value = e_value
        signal.robust_to_confounding = robust_to_confounding
        return signal

    def update_interpretation(
        self,
        signal: CausalAnalysisTrainingSignal,
        interpretation_depth: str,
        narrative_length: int,
        key_findings_count: int,
        recommendations_count: int,
        total_latency_ms: float,
        confidence_score: float,
    ) -> CausalAnalysisTrainingSignal:
        """Update signal with interpretation phase results."""
        signal.interpretation_depth = interpretation_depth
        signal.narrative_length = narrative_length
        signal.key_findings_count = key_findings_count
        signal.recommendations_count = recommendations_count
        signal.total_latency_ms = total_latency_ms
        signal.confidence_score = confidence_score

        # Add to buffer
        self._signals_buffer.append(signal)
        if len(self._signals_buffer) > self._buffer_limit:
            self._signals_buffer.pop(0)

        return signal

    def update_with_feedback(
        self,
        signal: CausalAnalysisTrainingSignal,
        user_satisfaction: Optional[float] = None,
    ) -> CausalAnalysisTrainingSignal:
        """Update signal with user feedback (delayed)."""
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

    def get_robust_examples(
        self,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Get examples where refutation passed (high-quality causal evidence)."""
        signals = [s for s in self._signals_buffer if s.overall_robust]
        sorted_signals = sorted(signals, key=lambda s: s.compute_reward(), reverse=True)
        return [s.to_dict() for s in sorted_signals[:limit]]

    def clear_buffer(self):
        """Clear the signals buffer."""
        self._signals_buffer.clear()


# =============================================================================
# 4. SINGLETON ACCESS
# =============================================================================

_signal_collector: Optional[CausalImpactSignalCollector] = None


def get_causal_impact_signal_collector() -> CausalImpactSignalCollector:
    """Get or create signal collector singleton."""
    global _signal_collector
    if _signal_collector is None:
        _signal_collector = CausalImpactSignalCollector()
    return _signal_collector


def reset_dspy_integration() -> None:
    """Reset singletons (for testing)."""
    global _signal_collector, _causal_impact_module
    _signal_collector = None
    _causal_impact_module = None


# =============================================================================
# 5. EXPORTS
# =============================================================================

__all__ = [
    # Training Signals
    "CausalAnalysisTrainingSignal",
    # DSPy Signatures & Module
    "CausalGraphSignature",
    "EvidenceSynthesisSignature",
    "CausalInterpretationSignature",
    "CausalImpactModule",
    "DSPY_AVAILABLE",
    # Collectors
    "CausalImpactSignalCollector",
    # Access
    "get_causal_impact_signal_collector",
    "get_causal_impact_module",
    "reset_dspy_integration",
]
