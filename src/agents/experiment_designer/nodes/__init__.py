"""Experiment Designer Agent Nodes.

This package contains the individual nodes that make up the experiment designer agent's workflow.

Nodes:
    - ContextLoaderNode: Loads organizational learning context
    - TwinSimulationNode: Digital twin pre-screening (Phase 15)
    - DesignReasoningNode: Deep reasoning for experiment design (LLM)
    - PowerAnalysisNode: Statistical power calculations
    - ValidityAuditNode: Adversarial validity assessment (LLM)
    - RedesignNode: Incorporates audit feedback for redesign
    - TemplateGeneratorNode: Generates DoWhy code and pre-registration docs

Workflow:
    context_loader → twin_simulation (optional) → design_reasoning → power_analysis →
    validity_audit → (conditional redesign) → template_generator
"""

from src.agents.experiment_designer.nodes.context_loader import ContextLoaderNode
from src.agents.experiment_designer.nodes.design_reasoning import DesignReasoningNode
from src.agents.experiment_designer.nodes.power_analysis import PowerAnalysisNode
from src.agents.experiment_designer.nodes.redesign import RedesignNode
from src.agents.experiment_designer.nodes.template_generator import TemplateGeneratorNode
from src.agents.experiment_designer.nodes.twin_simulation import TwinSimulationNode
from src.agents.experiment_designer.nodes.validity_audit import ValidityAuditNode

__all__ = [
    "ContextLoaderNode",
    "TwinSimulationNode",
    "DesignReasoningNode",
    "PowerAnalysisNode",
    "ValidityAuditNode",
    "RedesignNode",
    "TemplateGeneratorNode",
]
