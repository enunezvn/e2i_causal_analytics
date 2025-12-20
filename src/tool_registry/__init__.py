"""
E2I Tool Registry
Version: 4.2

Centralized registration and lookup of composable tools from agents.

Usage:
    from src.tool_registry import (
        ToolRegistry,
        composable_tool,
        register_tool,
        get_registry,
    )
    
    # Using decorator
    @composable_tool(
        name="my_tool",
        description="Does something useful",
        source_agent="my_agent",
        tier=2
    )
    def my_tool(x: int) -> dict:
        return {"result": x * 2}
    
    # Manual registration
    register_tool(
        name="another_tool",
        callable=some_function,
        description="Another useful tool",
        source_agent="another_agent",
        tier=3
    )
    
    # Lookup
    registry = get_registry()
    tool = registry.get("my_tool")
    result = tool.callable(x=5)
"""

from .registry import (
    # Classes
    ToolRegistry,
    RegisteredTool,
    ToolSchema,
    ToolParameter,
    # Decorator
    composable_tool,
    # Functions
    register_tool,
    get_registry,
    list_available_tools,
)

__all__ = [
    "ToolRegistry",
    "RegisteredTool",
    "ToolSchema",
    "ToolParameter",
    "composable_tool",
    "register_tool",
    "get_registry",
    "list_available_tools",
]

__version__ = "4.2.0"
