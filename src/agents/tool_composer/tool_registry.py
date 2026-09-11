# src/e2i/agents/tool_composer/tool_registry.py
"""
Central registry for composable tools.

This module manages the registration and lookup of tools that can
be used by the Tool Composer. Tools are exposed by agents and
registered at startup.
"""

from typing import Any, Optional

from .schemas import ToolCategory, ToolSchema

# Domain to category mapping
DOMAIN_TO_CATEGORY = {
    "COHORT_CONSTRUCTION": ToolCategory.COHORT,
    "COHORT_DEFINITION": ToolCategory.COHORT,
    "PATIENT_ELIGIBILITY": ToolCategory.COHORT,
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
        self._by_category: dict[ToolCategory, list[str]] = {cat: [] for cat in ToolCategory}
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
        in_degree = dict.fromkeys(tool_names, 0)
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
                name
                for name in remaining
                if all(dep not in remaining for dep in self._dependencies.get(name, []))
            ]

            if not level:
                # Circular dependency detected
                raise ValueError(f"Circular dependency detected among: {remaining}")

            levels.append(level)
            remaining -= set(level)

        return levels


# =============================================================================
# DEFAULT TOOL DEFINITIONS
# =============================================================================
#
# The planner, the DSPy planning signature and the executor read the live registry in
# ``src.tool_registry.registry`` (filled by ``@composable_tool`` / ``registry.register``).
# This module keeps what only it carries -- each tool's category and the tools whose
# output it can consume -- and DERIVES everything else from the live registry, so its
# input/output schemas cannot drift from the registered callables again (#2003: the
# hand-written copies had drifted on 14 of 16 tools). The DB ``tool_registry`` /
# ``tool_dependencies`` rows are synced from these definitions by
# ``scripts/generate_tool_registry_sync_migration.py``.

# name -> (category, tools whose output it can consume). Covers every live tool.
TOOL_METADATA: dict[str, tuple[ToolCategory, list[str]]] = {
    # Cohort constructor (Tier 0)
    "cohort_builder": (ToolCategory.COHORT, []),
    "cohort_validator": (ToolCategory.COHORT, ["cohort_builder"]),
    "cohort_statistics": (ToolCategory.COHORT, ["cohort_builder"]),
    # Causal impact
    "causal_effect_estimator": (ToolCategory.CAUSAL, []),
    "refutation_runner": (ToolCategory.CAUSAL, ["causal_effect_estimator"]),
    "sensitivity_analyzer": (ToolCategory.CAUSAL, ["causal_effect_estimator"]),
    "discover_dag": (ToolCategory.CAUSAL, []),
    "rank_drivers": (ToolCategory.CAUSAL, ["discover_dag"]),
    # Heterogeneous optimizer
    "cate_analyzer": (ToolCategory.SEGMENTATION, ["causal_effect_estimator"]),
    "segment_ranker": (ToolCategory.SEGMENTATION, ["cate_analyzer"]),
    # Gap analyzer
    "gap_calculator": (ToolCategory.GAP, []),
    "roi_estimator": (ToolCategory.GAP, ["gap_calculator"]),
    # Experiment designer
    "power_calculator": (ToolCategory.EXPERIMENT, ["causal_effect_estimator", "cate_analyzer"]),
    # #2015: no longer consumes causal_effect_estimator -- the twin engine estimates the
    # effect from the brand's cohort, so an upstream effect is not an input.
    "counterfactual_simulator": (ToolCategory.EXPERIMENT, ["cate_analyzer", "gap_calculator"]),
    # Prediction synthesizer
    "risk_scorer": (ToolCategory.PREDICTION, []),
    "propensity_estimator": (ToolCategory.PREDICTION, []),
    "model_inference": (ToolCategory.PREDICTION, []),
    # Drift monitor
    "psi_calculator": (ToolCategory.MONITORING, []),
    "distribution_comparator": (ToolCategory.MONITORING, []),
    "detect_structural_drift": (ToolCategory.MONITORING, []),
}

# (consumer, producer) -> (producer output field, consumer input field), one entry per
# ``can_consume_from`` pair. ``None`` output field = the whole output; ``None`` input
# field = mapped by matching field names (the DB column semantics). Both are REAL keys
# of the registered tools -- the drift test checks them.
DEPENDENCY_FIELD_MAPPINGS: dict[tuple[str, str], tuple[Optional[str], Optional[str]]] = {
    ("cohort_validator", "cohort_builder"): (None, "cohort_result"),
    ("cohort_statistics", "cohort_builder"): (None, "cohort_result"),
    # Refutation re-estimates on the same treatment/outcome/confounders; no output
    # field of the estimate is an input of the refutation suite.
    ("refutation_runner", "causal_effect_estimator"): (None, None),
    # ate / ci_lower / ci_upper map by name.
    ("sensitivity_analyzer", "causal_effect_estimator"): (None, None),
    # Ordering only: CATE re-estimates per segment from the data.
    ("cate_analyzer", "causal_effect_estimator"): (None, None),
    ("segment_ranker", "cate_analyzer"): (None, "cate_results"),
    ("roi_estimator", "gap_calculator"): (None, "gap_analysis"),
    # Ordering only (#2015): the estimate's ``ate`` is in OUTCOME units, while
    # ``effect_size`` is a Cohen's d, a relative change or a hazard ratio depending on the
    # design, so passing it through would size the study for the wrong effect.
    ("power_calculator", "causal_effect_estimator"): (None, None),
    # No direct field: effect_by_segment is a per-segment dict and effect_size one number,
    # so the planner has to pick the segment's effect.
    ("power_calculator", "cate_analyzer"): (None, None),
    ("counterfactual_simulator", "cate_analyzer"): ("high_responders", "target_entities"),
    # No direct field: bottom_performer is one entity name and target_entities a list,
    # so the planner has to build the list.
    ("counterfactual_simulator", "gap_calculator"): (None, None),
    ("rank_drivers", "discover_dag"): ("edge_list", "dag_edge_list"),
}

_SCALAR_JSON_TYPES = {"str": "string", "float": "number", "int": "integer", "bool": "boolean"}


def _json_type(type_hint: str) -> dict[str, Any]:
    """JSON Schema for a registry parameter's Python type string."""
    hint = type_hint.strip()
    if hint in _SCALAR_JSON_TYPES:
        return {"type": _SCALAR_JSON_TYPES[hint]}
    if hint == "Any":
        return {}
    if hint == "dict" or (hint.startswith("Dict[") and hint.endswith("]")):
        return {"type": "object"}
    if hint == "list":
        return {"type": "array"}
    if hint.startswith("List[") and hint.endswith("]"):
        return {"type": "array", "items": _json_type(hint[len("List[") : -1])}
    raise ValueError(f"no JSON Schema mapping for registry parameter type {type_hint!r}")


def _input_schema(parameters: list[Any]) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    for param in parameters:
        prop = {**_json_type(param.type), "description": param.description}
        if param.default is not None:
            prop["default"] = param.default
        properties[param.name] = prop
    return {
        "type": "object",
        "properties": properties,
        "required": [param.name for param in parameters if param.required],
    }


def create_default_tools() -> list[ToolSchema]:
    """
    Create default tool definitions from the live registry.

    Name, description, source agent, input schema, output schema, latency and the
    callable (``fn``) come from the registered tool; category and ``can_consume_from``
    from ``TOOL_METADATA``. Raises ``LookupError`` when a tool in ``TOOL_METADATA`` is
    not registered or registered no output model.
    """
    # Function-local: registering the tools imports the causal engine, which this
    # module's other users do not need.
    from src.agents.tool_composer import tool_registrations  # noqa: F401
    from src.tool_registry.registry import get_registry as get_live_registry
    from src.tool_registry.tools.causal_discovery import register_all_discovery_tools
    from src.tool_registry.tools.model_inference import register_model_inference_tool
    from src.tool_registry.tools.structural_drift import register_structural_drift_tool

    live = get_live_registry()
    register_all_discovery_tools()
    register_model_inference_tool()
    register_structural_drift_tool()

    tools = []
    for name, (category, can_consume_from) in TOOL_METADATA.items():
        registered = live.get(name)
        if registered is None:
            raise LookupError(f"tool '{name}' is not registered in the live tool registry")
        if registered.pydantic_output_model is None:
            raise LookupError(f"tool '{name}' registered no output model")
        tools.append(
            ToolSchema(
                name=name,
                description=registered.schema.description,
                category=category,
                source_agent=registered.schema.source_agent,
                input_schema=_input_schema(registered.schema.input_parameters),
                output_schema=registered.pydantic_output_model.model_json_schema(),
                fn=registered.callable,
                avg_latency_ms=float(registered.schema.avg_execution_ms),
                can_consume_from=list(can_consume_from),
            )
        )
    return tools


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
