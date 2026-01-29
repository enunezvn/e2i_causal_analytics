"""
E2I Tool Registry
Version: 4.3
Purpose: Centralized registration and lookup of composable tools from agents

Features:
- In-memory registry with singleton pattern
- Decorator-based tool registration
- Database synchronization via Supabase
- Schema validation before registration
- Tool performance tracking support
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Type, TypeVar

from pydantic import BaseModel, ValidationError, validator
from pydantic import BaseModel as PydanticModel

logger = logging.getLogger(__name__)


# ============================================================================
# EXCEPTIONS
# ============================================================================


class ToolRegistryError(Exception):
    """Base exception for tool registry errors."""

    pass


class ToolValidationError(ToolRegistryError):
    """Raised when tool schema validation fails."""

    def __init__(self, tool_name: str, errors: List[str]):
        self.tool_name = tool_name
        self.errors = errors
        super().__init__(f"Tool '{tool_name}' validation failed: {'; '.join(errors)}")


class ToolNotFoundError(ToolRegistryError):
    """Raised when a tool is not found in the registry."""

    def __init__(self, tool_name: str):
        self.tool_name = tool_name
        super().__init__(f"Tool '{tool_name}' not found in registry")


class ToolCategory(str, Enum):
    """Tool category enum matching database constraint."""

    CAUSAL = "causal"
    COMPARATIVE = "comparative"
    PREDICTIVE = "predictive"
    DESCRIPTIVE = "descriptive"
    EXPERIMENTAL = "experimental"
    MONITORING = "monitoring"

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

    # ========================================================================
    # DATABASE SYNCHRONIZATION (G3: Dynamic Tool Registration)
    # ========================================================================

    def validate_tool_schema(self, schema: ToolSchema) -> List[str]:
        """
        Validate a tool schema against constraints.

        Args:
            schema: The tool schema to validate

        Returns:
            List of validation error messages (empty if valid)
        """
        errors: List[str] = []

        # Name validation
        if not schema.name or len(schema.name) < 2:
            errors.append("Tool name must be at least 2 characters")
        if not schema.name.replace("_", "").isalnum():
            errors.append("Tool name must be alphanumeric with underscores only")

        # Description validation
        if not schema.description or len(schema.description) < 10:
            errors.append("Tool description must be at least 10 characters")

        # Tier validation
        if schema.tier < 0 or schema.tier > 5:
            errors.append(f"Tool tier must be 0-5, got {schema.tier}")

        # Source agent validation
        if not schema.source_agent:
            errors.append("Tool must have a source_agent")

        # Execution time validation
        if schema.avg_execution_ms < 0:
            errors.append("avg_execution_ms cannot be negative")

        # Dependency validation
        for dep in schema.requires_tools:
            if dep not in self._tools and dep != schema.name:
                errors.append(f"Required tool '{dep}' not found in registry")

        return errors

    def register_validated(
        self,
        schema: ToolSchema,
        callable: Callable[..., Any],
        input_model: Optional[Type[BaseModel]] = None,
        output_model: Optional[Type[BaseModel]] = None,
        strict: bool = True,
    ) -> bool:
        """
        Register a tool with validation.

        Args:
            schema: Tool schema with metadata
            callable: The actual function to execute
            input_model: Optional Pydantic model for input validation
            output_model: Optional Pydantic model for output validation
            strict: If True, raise on validation errors; if False, log warnings

        Returns:
            True if registration succeeded

        Raises:
            ToolValidationError: If validation fails and strict=True
        """
        errors = self.validate_tool_schema(schema)

        if errors:
            if strict:
                raise ToolValidationError(schema.name, errors)
            else:
                for err in errors:
                    logger.warning(f"Tool '{schema.name}' validation: {err}")

        self.register(
            schema=schema,
            callable=callable,
            input_model=input_model,
            output_model=output_model,
        )
        return True

    async def register_from_database(
        self,
        db_client,
        table_name: str = "tool_registry",
        category_filter: Optional[ToolCategory] = None,
    ) -> int:
        """
        Load and register tools from database.

        Args:
            db_client: Supabase client with execute method
            table_name: Name of the tool registry table
            category_filter: Optional filter by tool category

        Returns:
            Number of tools registered

        Note:
            This registers tool metadata only. Actual callables must be
            provided separately via register() or @composable_tool decorator.
        """
        # Build query
        query = f"SELECT * FROM {table_name} WHERE composable = true"
        if category_filter:
            query += f" AND category = '{category_filter.value}'"

        try:
            result = await db_client.execute(query)
            rows = result.data if hasattr(result, "data") else result

            registered_count = 0
            for row in rows:
                # Convert DB row to ToolSchema
                schema = ToolSchema(
                    name=row["name"],
                    description=row.get("description", ""),
                    source_agent=row.get("source_agent", "unknown"),
                    tier=row.get("tier", 0),
                    avg_execution_ms=row.get("avg_latency_ms", 1000),
                    version=row.get("version", "1.0.0"),
                )

                # Store schema without callable (placeholder)
                # Actual callable must be registered separately
                if schema.name not in self._tools:
                    self._tools[schema.name] = RegisteredTool(
                        schema=schema,
                        callable=self._create_placeholder_callable(schema.name),
                    )

                    # Index by agent
                    if schema.source_agent not in self._by_agent:
                        self._by_agent[schema.source_agent] = []
                    if schema.name not in self._by_agent[schema.source_agent]:
                        self._by_agent[schema.source_agent].append(schema.name)

                    # Index by tier
                    if schema.tier not in self._by_tier:
                        self._by_tier[schema.tier] = []
                    if schema.name not in self._by_tier[schema.tier]:
                        self._by_tier[schema.tier].append(schema.name)

                    registered_count += 1
                    logger.info(f"Registered tool from DB: {schema.name}")

            logger.info(f"Registered {registered_count} tools from database")
            return registered_count

        except Exception as e:
            logger.error(f"Failed to register tools from database: {e}")
            raise ToolRegistryError(f"Database sync failed: {e}") from e

    def _create_placeholder_callable(self, tool_name: str) -> Callable[..., Any]:
        """Create a placeholder callable for DB-registered tools."""

        def placeholder(*args, **kwargs):
            raise ToolNotFoundError(
                f"Tool '{tool_name}' was loaded from database but has no implementation. "
                "Register the actual callable using registry.register() or @composable_tool."
            )

        return placeholder

    async def sync_to_database(
        self,
        db_client,
        table_name: str = "tool_registry",
        update_existing: bool = True,
    ) -> Dict[str, int]:
        """
        Sync registered tools to database.

        Args:
            db_client: Supabase client with execute method
            table_name: Name of the tool registry table
            update_existing: If True, update existing records; if False, skip

        Returns:
            Dict with counts: {"inserted": N, "updated": N, "skipped": N}
        """
        stats = {"inserted": 0, "updated": 0, "skipped": 0}

        for tool_name, registered_tool in self._tools.items():
            schema = registered_tool.schema

            # Prepare record
            record = {
                "name": schema.name,
                "description": schema.description,
                "source_agent": schema.source_agent,
                "tier": schema.tier,
                "input_schema": {"parameters": [p.__dict__ for p in schema.input_parameters]},
                "output_schema": {"type": schema.output_schema},
                "composable": True,
                "avg_latency_ms": schema.avg_execution_ms,
                "version": schema.version,
            }

            try:
                # Check if exists
                check_query = f"SELECT tool_id FROM {table_name} WHERE name = '{schema.name}'"
                result = await db_client.execute(check_query)
                exists = bool(result.data if hasattr(result, "data") else result)

                if exists:
                    if update_existing:
                        # Update existing
                        update_query = f"""
                            UPDATE {table_name}
                            SET description = '{schema.description}',
                                source_agent = '{schema.source_agent}',
                                tier = {schema.tier},
                                avg_latency_ms = {schema.avg_execution_ms},
                                version = '{schema.version}',
                                updated_at = NOW()
                            WHERE name = '{schema.name}'
                        """
                        await db_client.execute(update_query)
                        stats["updated"] += 1
                        logger.debug(f"Updated tool in DB: {schema.name}")
                    else:
                        stats["skipped"] += 1
                else:
                    # Insert new
                    import json

                    insert_query = f"""
                        INSERT INTO {table_name} (name, description, source_agent, tier,
                            input_schema, output_schema, composable, avg_latency_ms, version)
                        VALUES (
                            '{schema.name}',
                            '{schema.description}',
                            '{schema.source_agent}',
                            {schema.tier},
                            '{json.dumps(record["input_schema"])}',
                            '{json.dumps(record["output_schema"])}',
                            true,
                            {schema.avg_execution_ms},
                            '{schema.version}'
                        )
                    """
                    await db_client.execute(insert_query)
                    stats["inserted"] += 1
                    logger.debug(f"Inserted tool to DB: {schema.name}")

            except Exception as e:
                logger.error(f"Failed to sync tool '{schema.name}' to database: {e}")
                stats["skipped"] += 1

        logger.info(
            f"Database sync complete: {stats['inserted']} inserted, "
            f"{stats['updated']} updated, {stats['skipped']} skipped"
        )
        return stats

    def get_tools_by_category(self, category: ToolCategory) -> List[str]:
        """
        Get tools filtered by category.

        Note: This requires tools to have category metadata set.
        For now, returns empty list as category is not in ToolSchema.
        """
        # Category filtering would require schema enhancement
        # This is a placeholder for future enhancement
        logger.warning("Category filtering not yet implemented in ToolSchema")
        return []


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
# GLOBAL REGISTRY SINGLETON
# ============================================================================

_global_registry: Optional[ToolRegistry] = None


def get_registry() -> ToolRegistry:
    """Get the global tool registry singleton.

    Returns the same ToolRegistry instance across all calls, ensuring
    tools registered via @composable_tool decorator are available.
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = ToolRegistry()
    return _global_registry


def reset_registry() -> None:
    """Reset the global registry (for testing)."""
    global _global_registry
    _global_registry = None


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


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


__all__ = [
    # Exceptions
    "ToolRegistryError",
    "ToolValidationError",
    "ToolNotFoundError",
    # Enums
    "ToolCategory",
    # Schema classes
    "ToolParameter",
    "ToolSchema",
    "RegisteredTool",
    # Registry
    "ToolRegistry",
    # Decorator
    "composable_tool",
    # Convenience functions
    "get_registry",
    "register_tool",
    "list_available_tools",
]
