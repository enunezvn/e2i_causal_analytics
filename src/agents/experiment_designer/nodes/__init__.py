"""Experiment Designer Agent Nodes.

This package contains the individual nodes that make up the experiment designer agent's workflow.

Nodes:
    - ContextLoaderNode: Loads organizational learning context
    - DesignReasoningNode: Deep reasoning for experiment design (LLM)
    - PowerAnalysisNode: Statistical power calculations
    - ValidityAuditNode: Adversarial validity assessment (LLM)
    - RedesignNode: Incorporates audit feedback for redesign
    - TemplateGeneratorNode: Generates DoWhy code and pre-registration docs

Workflow:
    context_loader → design_reasoning → power_analysis → validity_audit →
    (conditional redesign) → template_generator
"""

from src.agents.experiment_designer.nodes.context_loader import ContextLoaderNode
from src.agents.experiment_designer.nodes.design_reasoning import DesignReasoningNode
from src.agents.experiment_designer.nodes.power_analysis import PowerAnalysisNode
from src.agents.experiment_designer.nodes.validity_audit import ValidityAuditNode
from src.agents.experiment_designer.nodes.redesign import RedesignNode
from src.agents.experiment_designer.nodes.template_generator import TemplateGeneratorNode

__all__ = [
    "ContextLoaderNode",
    "DesignReasoningNode",
    "PowerAnalysisNode",
    "ValidityAuditNode",
    "RedesignNode",
    "TemplateGeneratorNode",
]
