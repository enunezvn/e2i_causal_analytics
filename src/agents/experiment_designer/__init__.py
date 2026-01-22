"""Experiment Designer Agent.

Tier: 3 (Monitoring & Design)
Agent Type: Hybrid (Deep Reasoning + Computation)
Performance Target: <60s total latency

This agent designs rigorous experiments for causal inference in pharmaceutical
commercial operations. It combines:
1. Deep reasoning for strategic experiment design (LLM)
2. Statistical power analysis for sample sizing
3. Adversarial validity audit for threat identification (LLM)
4. DoWhy code generation for analysis templates

Key Features:
- Multi-design exploration (RCT, cluster RCT, quasi-experimental)
- Power analysis with sensitivity analysis
- Validity threat identification and mitigation
- Pre-registration document generation
- DoWhy/EconML integration

Usage:
    from src.agents.experiment_designer import ExperimentDesignerAgent, ExperimentDesignerInput

    agent = ExperimentDesignerAgent()
    result = agent.run(ExperimentDesignerInput(
        business_question="Does increasing rep visit frequency improve HCP engagement?",
        constraints={
            "expected_effect_size": 0.25,
            "power": 0.80,
            "weekly_accrual": 50
        },
        available_data={
            "variables": ["hcp_id", "territory", "visit_count", "engagement_score"]
        }
    ))

    print(f"Design: {result.design_type}")
    print(f"Sample Size: {result.power_analysis.required_sample_size}")
    print(f"Validity Score: {result.overall_validity_score}")
    print(f"\\nAnalysis Code:")
    print(result.analysis_code)

Documentation:
    - Specialist: .claude/specialists/Agent_Specialists_Tiers 1-5/experiment-designer.md
    - Contract: .claude/contracts/tier3-contracts.md lines 82-142
"""

from src.agents.experiment_designer.agent import (
    ExperimentDesignerAgent,
    ExperimentDesignerInput,
    ExperimentDesignerOutput,
    OutcomeOutput,
    PowerAnalysisOutput,
    TreatmentOutput,
    ValidityThreatOutput,
)
from src.agents.experiment_designer.graph import (
    create_experiment_designer_graph,
    create_initial_state,
    experiment_designer_graph,
)
from src.agents.experiment_designer.mlflow_tracker import (
    ExperimentDesignerMLflowTracker,
    ExperimentDesignerMetrics,
    DesignContext,
    create_tracker as create_mlflow_tracker,
)
from src.agents.experiment_designer.state import (
    DesignIteration,
    DoWhySpec,
    ErrorDetails,
    ExperimentDesignState,
    ExperimentTemplate,
    MitigationRecommendation,
    OutcomeDefinition,
    PowerAnalysisResult,
    TreatmentDefinition,
    ValidityThreat,
)

__all__ = [
    # Main agent
    "ExperimentDesignerAgent",
    # Input/Output models
    "ExperimentDesignerInput",
    "ExperimentDesignerOutput",
    "TreatmentOutput",
    "OutcomeOutput",
    "ValidityThreatOutput",
    "PowerAnalysisOutput",
    # State types
    "ExperimentDesignState",
    "TreatmentDefinition",
    "OutcomeDefinition",
    "ValidityThreat",
    "MitigationRecommendation",
    "PowerAnalysisResult",
    "DoWhySpec",
    "ExperimentTemplate",
    "DesignIteration",
    "ErrorDetails",
    # Graph
    "create_experiment_designer_graph",
    "experiment_designer_graph",
    "create_initial_state",
    # MLflow tracking
    "ExperimentDesignerMLflowTracker",
    "ExperimentDesignerMetrics",
    "DesignContext",
    "create_mlflow_tracker",
]
