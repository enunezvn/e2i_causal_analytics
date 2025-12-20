# src/e2i/agents/tool_composer/tool_registry.py
"""
Central registry for composable tools.

This module manages the registration and lookup of tools that can
be used by the Tool Composer. Tools are exposed by agents and
registered at startup.
"""

from typing import Optional, Callable
from .schemas import ToolSchema, ToolCategory


# Domain to category mapping
DOMAIN_TO_CATEGORY = {
    "CAUSAL_ANALYSIS": ToolCategory.CAUSAL,
    "HETEROGENEITY": ToolCategory.SEGMENTATION,
    "GAP_ANALYSIS": ToolCategory.GAP,
    "EXPERIMENTATION": ToolCategory.EXPERIMENT,
    "PREDICTION": ToolCategory.PREDICTION,
    "MONITORING": ToolCategory.MONITORING,
}


class ToolRegistry:
    """
    Central registry for composable tools.
    
    Provides registration, lookup, and dependency information
    for tools exposed by agents.
    """

    def __init__(self):
        """Initialize empty registry."""
        self._tools: dict[str, ToolSchema] = {}
        self._by_category: dict[ToolCategory, list[str]] = {
            cat: [] for cat in ToolCategory
        }
        self._by_agent: dict[str, list[str]] = {}
        self._dependencies: dict[str, list[str]] = {}  # tool -> can_consume_from

    # =========================================================================
    # REGISTRATION
    # =========================================================================

    def register(self, tool: ToolSchema) -> None:
        """
        Register a tool in the registry.
        
        Args:
            tool: Tool schema to register
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' already registered")

        self._tools[tool.name] = tool
        
        # Index by category
        self._by_category[tool.category].append(tool.name)
        
        # Index by agent
        if tool.source_agent not in self._by_agent:
            self._by_agent[tool.source_agent] = []
        self._by_agent[tool.source_agent].append(tool.name)
        
        # Store dependencies
        self._dependencies[tool.name] = tool.can_consume_from

    def register_many(self, tools: list[ToolSchema]) -> None:
        """Register multiple tools."""
        for tool in tools:
            self.register(tool)

    def unregister(self, tool_name: str) -> None:
        """Remove a tool from the registry."""
        if tool_name not in self._tools:
            return
            
        tool = self._tools[tool_name]
        
        # Remove from indices
        self._by_category[tool.category].remove(tool_name)
        self._by_agent[tool.source_agent].remove(tool_name)
        del self._dependencies[tool_name]
        del self._tools[tool_name]

    # =========================================================================
    # LOOKUP
    # =========================================================================

    def get(self, tool_name: str) -> Optional[ToolSchema]:
        """Get tool by name."""
        return self._tools.get(tool_name)

    def get_by_category(self, category: ToolCategory) -> list[ToolSchema]:
        """Get all tools in a category."""
        return [self._tools[name] for name in self._by_category.get(category, [])]

    def get_by_agent(self, agent_name: str) -> list[ToolSchema]:
        """Get all tools from an agent."""
        return [self._tools[name] for name in self._by_agent.get(agent_name, [])]

    def get_by_domain(self, domain: str) -> list[ToolSchema]:
        """Get tools by domain name (maps to category)."""
        category = DOMAIN_TO_CATEGORY.get(domain)
        if category:
            return self.get_by_category(category)
        return []

    def list_all(self) -> list[ToolSchema]:
        """List all registered tools."""
        return list(self._tools.values())

    def list_composable(self) -> list[ToolSchema]:
        """List only composable tools."""
        return [t for t in self._tools.values() if t.composable]

    # =========================================================================
    # DEPENDENCY QUERIES
    # =========================================================================

    def get_consumers(self, tool_name: str) -> list[str]:
        """Get tools that can consume output from this tool."""
        consumers = []
        for name, deps in self._dependencies.items():
            if tool_name in deps:
                consumers.append(name)
        return consumers

    def get_producers(self, tool_name: str) -> list[str]:
        """Get tools whose output this tool can consume."""
        return self._dependencies.get(tool_name, [])

    def can_chain(self, producer: str, consumer: str) -> bool:
        """Check if producer output can feed consumer input."""
        return producer in self._dependencies.get(consumer, [])

    # =========================================================================
    # TOOL SELECTION
    # =========================================================================

    def select_for_domains(self, domains: list[str]) -> list[ToolSchema]:
        """
        Select best tools for given domains.
        
        Args:
            domains: List of domain names
            
        Returns:
            List of tools that cover the domains
        """
        selected = []
        seen = set()
        
        for domain in domains:
            tools = self.get_by_domain(domain)
            for tool in tools:
                if tool.name not in seen and tool.composable:
                    selected.append(tool)
                    seen.add(tool.name)
        
        return selected

    def get_execution_order(self, tool_names: list[str]) -> list[list[str]]:
        """
        Get tools in execution order based on dependencies.
        Returns groups that can be executed in parallel.
        
        Args:
            tool_names: Tools to order
            
        Returns:
            List of groups (each group can run in parallel)
        """
        # Build dependency graph
        in_degree = {name: 0 for name in tool_names}
        for name in tool_names:
            for dep in self._dependencies.get(name, []):
                if dep in tool_names:
                    in_degree[name] += 1

        # Topological sort with level tracking
        levels = []
        remaining = set(tool_names)
        
        while remaining:
            # Find tools with no dependencies in remaining set
            level = [
                name for name in remaining
                if all(
                    dep not in remaining
                    for dep in self._dependencies.get(name, [])
                )
            ]
            
            if not level:
                # Circular dependency detected
                raise ValueError(
                    f"Circular dependency detected among: {remaining}"
                )
            
            levels.append(level)
            remaining -= set(level)
        
        return levels


# =============================================================================
# DEFAULT TOOL DEFINITIONS
# =============================================================================

def create_default_tools() -> list[ToolSchema]:
    """
    Create default tool definitions.
    
    Note: fn (callable) is set to None here and must be
    populated when agents register their actual implementations.
    """
    return [
        # =====================================================================
        # CAUSAL IMPACT TOOLS
        # =====================================================================
        ToolSchema(
            name="causal_effect_estimator",
            description="Estimates Average Treatment Effect (ATE) with confidence intervals",
            category=ToolCategory.CAUSAL,
            source_agent="causal_impact",
            input_schema={
                "type": "object",
                "properties": {
                    "treatment_col": {"type": "string"},
                    "outcome_col": {"type": "string"},
                    "confounders": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["treatment_col", "outcome_col"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "ate": {"type": "number"},
                    "ci_lower": {"type": "number"},
                    "ci_upper": {"type": "number"},
                    "p_value": {"type": "number"},
                    "method": {"type": "string"},
                },
            },
            can_consume_from=[],  # Root tool
        ),
        ToolSchema(
            name="refutation_runner",
            description="Runs DoWhy refutation tests on causal estimates",
            category=ToolCategory.CAUSAL,
            source_agent="causal_impact",
            input_schema={
                "type": "object",
                "properties": {
                    "causal_result": {"type": "object"},
                    "tests": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["causal_result"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "passed": {"type": "boolean"},
                    "confidence_score": {"type": "number"},
                    "test_results": {"type": "array"},
                    "gate_decision": {"type": "string"},
                },
            },
            can_consume_from=["causal_effect_estimator"],
        ),
        ToolSchema(
            name="sensitivity_analyzer",
            description="Performs sensitivity analysis on causal estimates",
            category=ToolCategory.CAUSAL,
            source_agent="causal_impact",
            input_schema={
                "type": "object",
                "properties": {
                    "causal_result": {"type": "object"},
                    "gamma_range": {"type": "array", "items": {"type": "number"}},
                },
                "required": ["causal_result"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "e_value": {"type": "number"},
                    "robustness_value": {"type": "number"},
                    "sensitivity_plot_data": {"type": "array"},
                },
            },
            can_consume_from=["causal_effect_estimator"],
        ),

        # =====================================================================
        # HETEROGENEOUS OPTIMIZER TOOLS
        # =====================================================================
        ToolSchema(
            name="cate_analyzer",
            description="Computes Conditional Average Treatment Effects by segment",
            category=ToolCategory.SEGMENTATION,
            source_agent="heterogeneous_optimizer",
            input_schema={
                "type": "object",
                "properties": {
                    "causal_result": {"type": "object"},
                    "segment_cols": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["segment_cols"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "segments": {"type": "array"},
                    "effect_by_segment": {"type": "object"},
                    "high_responders": {"type": "array"},
                    "feature_importance": {"type": "object"},
                },
            },
            can_consume_from=["causal_effect_estimator"],
        ),
        ToolSchema(
            name="segment_ranker",
            description="Ranks segments by treatment effect or ROI",
            category=ToolCategory.SEGMENTATION,
            source_agent="heterogeneous_optimizer",
            input_schema={
                "type": "object",
                "properties": {
                    "cate_result": {"type": "object"},
                    "rank_by": {"type": "string", "enum": ["effect", "roi", "volume", "uplift"]},
                    "top_n": {"type": "integer", "default": 10},
                },
                "required": ["cate_result"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "ranked_segments": {"type": "array"},
                    "targeting_recommendations": {"type": "array"},
                },
            },
            can_consume_from=["cate_analyzer"],
        ),

        # =====================================================================
        # GAP ANALYZER TOOLS
        # =====================================================================
        ToolSchema(
            name="gap_calculator",
            description="Calculates performance gaps between entities",
            category=ToolCategory.GAP,
            source_agent="gap_analyzer",
            input_schema={
                "type": "object",
                "properties": {
                    "entity_type": {"type": "string"},
                    "metric": {"type": "string"},
                    "group_by": {"type": "string"},
                    "benchmark": {"type": "string", "enum": ["mean", "median", "top_decile", "target"]},
                },
                "required": ["entity_type", "metric"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "gaps": {"type": "array"},
                    "top_performers": {"type": "array"},
                    "bottom_performers": {"type": "array"},
                    "total_opportunity": {"type": "number"},
                },
            },
            can_consume_from=[],
        ),
        ToolSchema(
            name="roi_estimator",
            description="Estimates ROI of closing performance gaps",
            category=ToolCategory.GAP,
            source_agent="gap_analyzer",
            input_schema={
                "type": "object",
                "properties": {
                    "gap_result": {"type": "object"},
                    "intervention_cost": {"type": "number"},
                    "time_horizon_months": {"type": "integer", "default": 12},
                },
                "required": ["gap_result"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "estimated_roi": {"type": "number"},
                    "breakeven_months": {"type": "number"},
                    "npv": {"type": "number"},
                    "confidence_range": {"type": "array"},
                },
            },
            can_consume_from=["gap_calculator"],
        ),

        # =====================================================================
        # EXPERIMENT DESIGNER TOOLS
        # =====================================================================
        ToolSchema(
            name="power_calculator",
            description="Calculates statistical power and required sample size",
            category=ToolCategory.EXPERIMENT,
            source_agent="experiment_designer",
            input_schema={
                "type": "object",
                "properties": {
                    "effect_size": {"type": "number"},
                    "baseline_rate": {"type": "number"},
                    "sample_size": {"type": "integer"},
                    "alpha": {"type": "number", "default": 0.05},
                    "power_target": {"type": "number", "default": 0.8},
                },
                "required": ["effect_size"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "power": {"type": "number"},
                    "required_sample_size": {"type": "integer"},
                    "minimum_detectable_effect": {"type": "number"},
                    "duration_estimate_weeks": {"type": "integer"},
                },
            },
            can_consume_from=["causal_effect_estimator", "cate_analyzer"],
        ),
        ToolSchema(
            name="counterfactual_simulator",
            description="Simulates counterfactual outcomes for what-if scenarios",
            category=ToolCategory.EXPERIMENT,
            source_agent="experiment_designer",
            input_schema={
                "type": "object",
                "properties": {
                    "baseline": {"type": "object"},
                    "intervention": {"type": "object"},
                    "target_population": {"type": "object"},
                    "causal_model": {"type": "object"},
                },
                "required": ["baseline", "intervention"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "predicted_outcome": {"type": "number"},
                    "confidence_interval": {"type": "array"},
                    "lift_percentage": {"type": "number"},
                    "assumptions": {"type": "array"},
                    "caveats": {"type": "array"},
                },
            },
            can_consume_from=["causal_effect_estimator", "cate_analyzer", "gap_calculator"],
        ),

        # =====================================================================
        # PREDICTION SYNTHESIZER TOOLS
        # =====================================================================
        ToolSchema(
            name="risk_scorer",
            description="Scores entities by predicted risk",
            category=ToolCategory.PREDICTION,
            source_agent="prediction_synthesizer",
            input_schema={
                "type": "object",
                "properties": {
                    "entity_type": {"type": "string"},
                    "risk_type": {"type": "string"},
                    "time_horizon_days": {"type": "integer", "default": 90},
                },
                "required": ["entity_type", "risk_type"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "scores": {"type": "array"},
                    "high_risk_entities": {"type": "array"},
                    "risk_factors": {"type": "object"},
                    "model_performance": {"type": "object"},
                },
            },
            can_consume_from=[],
        ),
        ToolSchema(
            name="propensity_estimator",
            description="Estimates propensity scores for treatment assignment",
            category=ToolCategory.PREDICTION,
            source_agent="prediction_synthesizer",
            input_schema={
                "type": "object",
                "properties": {
                    "treatment_col": {"type": "string"},
                    "covariates": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["treatment_col", "covariates"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "propensity_scores": {"type": "array"},
                    "balance_statistics": {"type": "object"},
                    "overlap_assessment": {"type": "object"},
                },
            },
            can_consume_from=[],
        ),

        # =====================================================================
        # DRIFT MONITOR TOOLS
        # =====================================================================
        ToolSchema(
            name="psi_calculator",
            description="Calculates Population Stability Index for drift detection",
            category=ToolCategory.MONITORING,
            source_agent="drift_monitor",
            input_schema={
                "type": "object",
                "properties": {
                    "feature_name": {"type": "string"},
                    "reference_period": {"type": "string"},
                    "comparison_period": {"type": "string"},
                },
                "required": ["feature_name"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "psi_value": {"type": "number"},
                    "drift_severity": {"type": "string"},
                    "bin_contributions": {"type": "array"},
                },
            },
            can_consume_from=[],
        ),
        ToolSchema(
            name="distribution_comparator",
            description="Compares distributions between periods or segments",
            category=ToolCategory.MONITORING,
            source_agent="drift_monitor",
            input_schema={
                "type": "object",
                "properties": {
                    "columns": {"type": "array", "items": {"type": "string"}},
                    "group_a": {"type": "object"},
                    "group_b": {"type": "object"},
                    "tests": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["columns"],
            },
            output_schema={
                "type": "object",
                "properties": {
                    "test_results": {"type": "array"},
                    "significant_differences": {"type": "array"},
                    "visualization_data": {"type": "object"},
                },
            },
            can_consume_from=[],
        ),
    ]


# Global registry instance
_global_registry: Optional[ToolRegistry] = None


def get_registry() -> ToolRegistry:
    """Get or create the global tool registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = ToolRegistry()
        # Register default tools
        _global_registry.register_many(create_default_tools())
    return _global_registry


def reset_registry() -> None:
    """Reset the global registry (for testing)."""
    global _global_registry
    _global_registry = None
