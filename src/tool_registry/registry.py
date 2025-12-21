"""
E2I Tool Registry
Version: 4.2
Purpose: Centralized registration and lookup of composable tools from agents
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar

from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Type variable for tool functions
T = TypeVar("T", bound=Callable[..., Any])


# ============================================================================
# TOOL SCHEMA DEFINITIONS
# ============================================================================


@dataclass
class ToolParameter:
    """Schema for a single tool parameter"""

    name: str
    type: str  # Python type as string
    description: str
    required: bool = True
    default: Any = None


@dataclass
class ToolSchema:
    """Complete schema for a composable tool"""

    name: str
    description: str
    source_agent: str
    tier: int

    # Input/output schemas
    input_parameters: List[ToolParameter] = field(default_factory=list)
    output_schema: str = "Dict[str, Any]"  # Pydantic model name or type hint

    # Execution characteristics
    avg_execution_ms: int = 1000
    is_async: bool = False
    supports_batch: bool = False

    # Dependencies
    requires_tools: List[str] = field(default_factory=list)

    # Metadata
    version: str = "1.0.0"
    registered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "name": self.name,
            "description": self.description,
            "source_agent": self.source_agent,
            "tier": self.tier,
            "input_parameters": [
                {
                    "name": p.name,
                    "type": p.type,
                    "description": p.description,
                    "required": p.required,
                    "default": p.default,
                }
                for p in self.input_parameters
            ],
            "output_schema": self.output_schema,
            "avg_execution_ms": self.avg_execution_ms,
            "is_async": self.is_async,
            "version": self.version,
        }


@dataclass
class RegisteredTool:
    """A registered tool with its callable and schema"""

    schema: ToolSchema
    callable: Callable[..., Any]
    pydantic_input_model: Optional[Type[BaseModel]] = None
    pydantic_output_model: Optional[Type[BaseModel]] = None


# ============================================================================
# TOOL REGISTRY SINGLETON
# ============================================================================


class ToolRegistry:
    """
    Centralized registry for composable tools.

    Tools are registered by agents at startup and can be looked up
    by the Tool Composer for dynamic composition.
    """

    _instance: Optional[ToolRegistry] = None

    def __new__(cls) -> ToolRegistry:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._tools: Dict[str, RegisteredTool] = {}
        self._by_agent: Dict[str, List[str]] = {}
        self._by_tier: Dict[int, List[str]] = {}
        self._initialized = True

        logger.info("ToolRegistry initialized")

    def register(
        self,
        schema: ToolSchema,
        callable: Callable[..., Any],
        input_model: Optional[Type[BaseModel]] = None,
        output_model: Optional[Type[BaseModel]] = None,
    ) -> None:
        """
        Register a tool for composition.

        Args:
            schema: Tool schema with metadata
            callable: The actual function to execute
            input_model: Optional Pydantic model for input validation
            output_model: Optional Pydantic model for output validation
        """
        tool_name = schema.name

        if tool_name in self._tools:
            logger.warning(f"Tool '{tool_name}' already registered, overwriting")

        registered = RegisteredTool(
            schema=schema,
            callable=callable,
            pydantic_input_model=input_model,
            pydantic_output_model=output_model,
        )

        self._tools[tool_name] = registered

        # Index by agent
        if schema.source_agent not in self._by_agent:
            self._by_agent[schema.source_agent] = []
        if tool_name not in self._by_agent[schema.source_agent]:
            self._by_agent[schema.source_agent].append(tool_name)

        # Index by tier
        if schema.tier not in self._by_tier:
            self._by_tier[schema.tier] = []
        if tool_name not in self._by_tier[schema.tier]:
            self._by_tier[schema.tier].append(tool_name)

        logger.info(f"Registered tool: {tool_name} from {schema.source_agent} (Tier {schema.tier})")

    def get(self, tool_name: str) -> Optional[RegisteredTool]:
        """Get a registered tool by name"""
        return self._tools.get(tool_name)

    def get_callable(self, tool_name: str) -> Optional[Callable[..., Any]]:
        """Get just the callable for a tool"""
        tool = self.get(tool_name)
        return tool.callable if tool else None

    def get_schema(self, tool_name: str) -> Optional[ToolSchema]:
        """Get just the schema for a tool"""
        tool = self.get(tool_name)
        return tool.schema if tool else None

    def list_tools(self) -> List[str]:
        """List all registered tool names"""
        return list(self._tools.keys())

    def list_by_agent(self, agent_name: str) -> List[str]:
        """List tools from a specific agent"""
        return self._by_agent.get(agent_name, [])

    def list_by_tier(self, tier: int) -> List[str]:
        """List tools from a specific tier"""
        return self._by_tier.get(tier, [])

    def get_all_schemas(self) -> List[ToolSchema]:
        """Get schemas for all registered tools"""
        return [t.schema for t in self._tools.values()]

    def get_schemas_for_planning(self) -> List[Dict[str, Any]]:
        """
        Get simplified schemas suitable for LLM planning.

        Returns a list of tool descriptions that can be included
        in a prompt for the planning phase.
        """
        return [
            {
                "name": t.schema.name,
                "description": t.schema.description,
                "source": t.schema.source_agent,
                "inputs": [
                    f"{p.name}: {p.type} - {p.description}" for p in t.schema.input_parameters
                ],
                "output": t.schema.output_schema,
                "avg_ms": t.schema.avg_execution_ms,
            }
            for t in self._tools.values()
        ]

    def validate_tool_exists(self, tool_name: str) -> bool:
        """Check if a tool is registered"""
        return tool_name in self._tools

    def clear(self) -> None:
        """Clear all registered tools (useful for testing)"""
        self._tools.clear()
        self._by_agent.clear()
        self._by_tier.clear()
        logger.info("ToolRegistry cleared")

    @property
    def tool_count(self) -> int:
        """Number of registered tools"""
        return len(self._tools)

    @property
    def agent_count(self) -> int:
        """Number of agents with registered tools"""
        return len(self._by_agent)


# ============================================================================
# DECORATOR FOR TOOL REGISTRATION
# ============================================================================


def composable_tool(
    name: str,
    description: str,
    source_agent: str,
    tier: int,
    input_parameters: Optional[List[Dict[str, Any]]] = None,
    output_schema: str = "Dict[str, Any]",
    avg_execution_ms: int = 1000,
    input_model: Optional[Type[BaseModel]] = None,
    output_model: Optional[Type[BaseModel]] = None,
) -> Callable[[T], T]:
    """
    Decorator to register a function as a composable tool.

    Usage:
        @composable_tool(
            name="causal_effect_estimator",
            description="Estimate ATE/ATT with DoWhy",
            source_agent="causal_impact",
            tier=2,
            input_parameters=[
                {"name": "treatment", "type": "str", "description": "Treatment variable"},
                {"name": "outcome", "type": "str", "description": "Outcome variable"},
            ],
            output_schema="EffectEstimate"
        )
        def estimate_effect(treatment: str, outcome: str, data: pd.DataFrame) -> EffectEstimate:
            ...
    """

    def decorator(func: T) -> T:
        # Build parameter list
        params = []
        if input_parameters:
            for p in input_parameters:
                params.append(
                    ToolParameter(
                        name=p["name"],
                        type=p.get("type", "Any"),
                        description=p.get("description", ""),
                        required=p.get("required", True),
                        default=p.get("default"),
                    )
                )

        # Create schema
        schema = ToolSchema(
            name=name,
            description=description,
            source_agent=source_agent,
            tier=tier,
            input_parameters=params,
            output_schema=output_schema,
            avg_execution_ms=avg_execution_ms,
            is_async=asyncio_iscoroutinefunction(func) if "asyncio" in dir() else False,
        )

        # Register the tool
        registry = ToolRegistry()
        registry.register(
            schema=schema, callable=func, input_model=input_model, output_model=output_model
        )

        # Preserve the original function
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        # Attach schema to function for introspection
        wrapper._tool_schema = schema

        return wrapper

    return decorator


# Helper to check for async functions
def asyncio_iscoroutinefunction(func: Callable) -> bool:
    """Check if function is async"""
    try:
        import asyncio

        return asyncio.iscoroutinefunction(func)
    except ImportError:
        return False


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def get_registry() -> ToolRegistry:
    """Get the global tool registry instance"""
    return ToolRegistry()


def register_tool(
    name: str,
    callable: Callable[..., Any],
    description: str,
    source_agent: str,
    tier: int,
    **kwargs,
) -> None:
    """
    Convenience function to register a tool without decorator.

    Useful for registering existing functions or methods.
    """
    schema = ToolSchema(
        name=name,
        description=description,
        source_agent=source_agent,
        tier=tier,
        **{k: v for k, v in kwargs.items() if k in ToolSchema.__dataclass_fields__},
    )

    registry = get_registry()
    registry.register(schema=schema, callable=callable)


def list_available_tools() -> List[Dict[str, Any]]:
    """List all available tools with their schemas"""
    registry = get_registry()
    return registry.get_schemas_for_planning()
