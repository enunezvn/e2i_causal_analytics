"""Unit tests for Tool Registry.

Tests cover:
- ToolRegistry singleton pattern
- Tool registration and lookup
- Schema validation (G3)
- Database synchronization (G3)
- Exception handling
- Category filtering
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.tool_registry.registry import (
    RegisteredTool,
    ToolCategory,
    ToolNotFoundError,
    ToolParameter,
    ToolRegistry,
    ToolRegistryError,
    ToolSchema,
    ToolValidationError,
    composable_tool,
    get_registry,
    list_available_tools,
    register_tool,
)


# =============================================================================
# EXCEPTION TESTS
# =============================================================================


class TestToolRegistryExceptions:
    """Tests for registry exceptions."""

    def test_tool_registry_error_base(self):
        """Test base exception."""
        exc = ToolRegistryError("Test error")
        assert str(exc) == "Test error"

    def test_tool_validation_error(self):
        """Test validation error with details."""
        errors = ["Name too short", "Missing description"]
        exc = ToolValidationError("test_tool", errors)

        assert exc.tool_name == "test_tool"
        assert exc.errors == errors
        assert "Name too short" in str(exc)
        assert "Missing description" in str(exc)

    def test_tool_not_found_error(self):
        """Test not found error."""
        exc = ToolNotFoundError("missing_tool")

        assert exc.tool_name == "missing_tool"
        assert "missing_tool" in str(exc)
        assert "not found" in str(exc)


# =============================================================================
# TOOL CATEGORY TESTS
# =============================================================================


class TestToolCategory:
    """Tests for ToolCategory enum."""

    def test_all_categories_exist(self):
        """Test all expected categories exist."""
        assert ToolCategory.CAUSAL == "causal"
        assert ToolCategory.COMPARATIVE == "comparative"
        assert ToolCategory.PREDICTIVE == "predictive"
        assert ToolCategory.DESCRIPTIVE == "descriptive"
        assert ToolCategory.EXPERIMENTAL == "experimental"
        assert ToolCategory.MONITORING == "monitoring"

    def test_category_is_string_enum(self):
        """Test category is string-compatible."""
        # String enums use .value for the string representation
        assert ToolCategory.CAUSAL.value == "causal"
        assert ToolCategory.PREDICTIVE.value == "predictive"
        # Also check they can be used in string comparisons
        assert ToolCategory.CAUSAL == "causal"


# =============================================================================
# TOOL SCHEMA TESTS
# =============================================================================


class TestToolSchema:
    """Tests for ToolSchema dataclass."""

    def test_minimal_schema(self):
        """Test minimal valid schema."""
        schema = ToolSchema(
            name="test_tool",
            description="A test tool",
            source_agent="test_agent",
            tier=1,
        )

        assert schema.name == "test_tool"
        assert schema.tier == 1
        assert schema.avg_execution_ms == 1000  # default
        assert schema.is_async is False  # default

    def test_full_schema(self):
        """Test schema with all fields."""
        params = [
            ToolParameter(name="x", type="int", description="Input x"),
            ToolParameter(name="y", type="float", description="Input y", required=False),
        ]

        schema = ToolSchema(
            name="full_tool",
            description="A comprehensive test tool",
            source_agent="causal_impact",
            tier=2,
            input_parameters=params,
            output_schema="ResultModel",
            avg_execution_ms=500,
            is_async=True,
            supports_batch=True,
            requires_tools=["helper_tool"],
            version="2.0.0",
        )

        assert len(schema.input_parameters) == 2
        assert schema.output_schema == "ResultModel"
        assert schema.avg_execution_ms == 500
        assert schema.is_async is True
        assert "helper_tool" in schema.requires_tools

    def test_schema_to_dict(self):
        """Test schema serialization."""
        schema = ToolSchema(
            name="serialized_tool",
            description="Tool for serialization test",
            source_agent="test_agent",
            tier=1,
            input_parameters=[
                ToolParameter(name="input", type="str", description="Input param")
            ],
        )

        result = schema.to_dict()

        assert result["name"] == "serialized_tool"
        assert result["tier"] == 1
        assert len(result["input_parameters"]) == 1
        assert result["input_parameters"][0]["name"] == "input"


# =============================================================================
# TOOL REGISTRY TESTS
# =============================================================================


class TestToolRegistry:
    """Tests for ToolRegistry singleton."""

    @pytest.fixture(autouse=True)
    def clear_registry(self):
        """Clear registry before each test."""
        registry = get_registry()
        registry.clear()
        yield
        registry.clear()

    def test_singleton_pattern(self):
        """Test registry is singleton."""
        reg1 = ToolRegistry()
        reg2 = ToolRegistry()
        reg3 = get_registry()

        assert reg1 is reg2
        assert reg2 is reg3

    def test_register_tool(self):
        """Test basic tool registration."""
        registry = get_registry()

        schema = ToolSchema(
            name="basic_tool",
            description="A basic test tool",
            source_agent="test_agent",
            tier=1,
        )

        def tool_callable(x: int) -> int:
            return x * 2

        registry.register(schema, tool_callable)

        assert registry.validate_tool_exists("basic_tool")
        assert registry.tool_count == 1

    def test_get_tool(self):
        """Test retrieving registered tool."""
        registry = get_registry()

        schema = ToolSchema(
            name="retrievable_tool",
            description="A tool to retrieve",
            source_agent="test_agent",
            tier=2,
        )

        def tool_func():
            return "result"

        registry.register(schema, tool_func)

        tool = registry.get("retrievable_tool")
        assert tool is not None
        assert tool.schema.name == "retrievable_tool"
        assert tool.callable() == "result"

    def test_get_nonexistent_tool(self):
        """Test getting non-existent tool returns None."""
        registry = get_registry()
        assert registry.get("nonexistent") is None

    def test_get_callable(self):
        """Test get_callable retrieves just the function."""
        registry = get_registry()

        schema = ToolSchema(
            name="callable_tool",
            description="Test callable retrieval",
            source_agent="test_agent",
            tier=1,
        )

        def my_func(a, b):
            return a + b

        registry.register(schema, my_func)

        callable_func = registry.get_callable("callable_tool")
        assert callable_func is not None
        assert callable_func(2, 3) == 5

    def test_get_schema(self):
        """Test get_schema retrieves just the schema."""
        registry = get_registry()

        schema = ToolSchema(
            name="schema_tool",
            description="Test schema retrieval",
            source_agent="test_agent",
            tier=3,
        )

        registry.register(schema, lambda: None)

        retrieved_schema = registry.get_schema("schema_tool")
        assert retrieved_schema is not None
        assert retrieved_schema.tier == 3

    def test_list_tools(self):
        """Test listing all tool names."""
        registry = get_registry()

        for i in range(3):
            schema = ToolSchema(
                name=f"tool_{i}",
                description=f"Tool number {i}",
                source_agent="test_agent",
                tier=1,
            )
            registry.register(schema, lambda: None)

        tools = registry.list_tools()
        assert len(tools) == 3
        assert "tool_0" in tools
        assert "tool_2" in tools

    def test_list_by_agent(self):
        """Test listing tools by agent."""
        registry = get_registry()

        # Register tools for different agents
        for agent in ["agent_a", "agent_b"]:
            for i in range(2):
                schema = ToolSchema(
                    name=f"{agent}_tool_{i}",
                    description=f"Tool for {agent}",
                    source_agent=agent,
                    tier=1,
                )
                registry.register(schema, lambda: None)

        agent_a_tools = registry.list_by_agent("agent_a")
        assert len(agent_a_tools) == 2
        assert all("agent_a" in t for t in agent_a_tools)

        agent_b_tools = registry.list_by_agent("agent_b")
        assert len(agent_b_tools) == 2

    def test_list_by_tier(self):
        """Test listing tools by tier."""
        registry = get_registry()

        # Register tools at different tiers
        for tier in [1, 2, 2, 3]:
            schema = ToolSchema(
                name=f"tier_{tier}_tool_{registry.tool_count}",
                description=f"Tier {tier} tool",
                source_agent="test_agent",
                tier=tier,
            )
            registry.register(schema, lambda: None)

        tier_2_tools = registry.list_by_tier(2)
        assert len(tier_2_tools) == 2

        tier_1_tools = registry.list_by_tier(1)
        assert len(tier_1_tools) == 1

    def test_get_schemas_for_planning(self):
        """Test getting simplified schemas for LLM planning."""
        registry = get_registry()

        schema = ToolSchema(
            name="planning_tool",
            description="Tool for planning test",
            source_agent="planner",
            tier=1,
            input_parameters=[
                ToolParameter(name="query", type="str", description="Search query")
            ],
            avg_execution_ms=250,
        )
        registry.register(schema, lambda: None)

        planning_schemas = registry.get_schemas_for_planning()

        assert len(planning_schemas) == 1
        ps = planning_schemas[0]
        assert ps["name"] == "planning_tool"
        assert ps["source"] == "planner"
        assert ps["avg_ms"] == 250
        assert "query: str" in ps["inputs"][0]

    def test_clear(self):
        """Test clearing the registry."""
        registry = get_registry()

        schema = ToolSchema(
            name="clearable_tool",
            description="Tool to be cleared",
            source_agent="test_agent",
            tier=1,
        )
        registry.register(schema, lambda: None)

        assert registry.tool_count == 1

        registry.clear()

        assert registry.tool_count == 0
        assert registry.agent_count == 0


# =============================================================================
# VALIDATION TESTS (G3)
# =============================================================================


class TestToolValidation:
    """Tests for tool schema validation."""

    @pytest.fixture(autouse=True)
    def clear_registry(self):
        """Clear registry before each test."""
        registry = get_registry()
        registry.clear()
        yield
        registry.clear()

    def test_validate_valid_schema(self):
        """Test validation of valid schema."""
        registry = get_registry()

        schema = ToolSchema(
            name="valid_tool",
            description="A perfectly valid tool",
            source_agent="test_agent",
            tier=2,
        )

        errors = registry.validate_tool_schema(schema)
        assert errors == []

    def test_validate_name_too_short(self):
        """Test validation catches short name."""
        registry = get_registry()

        schema = ToolSchema(
            name="x",  # Too short
            description="Valid description here",
            source_agent="test_agent",
            tier=1,
        )

        errors = registry.validate_tool_schema(schema)
        assert any("name" in e.lower() for e in errors)

    def test_validate_description_too_short(self):
        """Test validation catches short description."""
        registry = get_registry()

        schema = ToolSchema(
            name="valid_name",
            description="Short",  # Too short
            source_agent="test_agent",
            tier=1,
        )

        errors = registry.validate_tool_schema(schema)
        assert any("description" in e.lower() for e in errors)

    def test_validate_invalid_tier(self):
        """Test validation catches invalid tier."""
        registry = get_registry()

        schema = ToolSchema(
            name="tier_test",
            description="Testing tier validation",
            source_agent="test_agent",
            tier=10,  # Invalid - should be 0-5
        )

        errors = registry.validate_tool_schema(schema)
        assert any("tier" in e.lower() for e in errors)

    def test_validate_negative_execution_time(self):
        """Test validation catches negative execution time."""
        registry = get_registry()

        schema = ToolSchema(
            name="time_test",
            description="Testing execution time validation",
            source_agent="test_agent",
            tier=1,
            avg_execution_ms=-100,  # Invalid
        )

        errors = registry.validate_tool_schema(schema)
        assert any("execution" in e.lower() or "negative" in e.lower() for e in errors)

    def test_validate_missing_dependency(self):
        """Test validation catches missing dependency."""
        registry = get_registry()

        schema = ToolSchema(
            name="dependent_tool",
            description="Tool with missing dependency",
            source_agent="test_agent",
            tier=1,
            requires_tools=["nonexistent_tool"],  # Not in registry
        )

        errors = registry.validate_tool_schema(schema)
        assert any("nonexistent_tool" in e for e in errors)

    def test_register_validated_strict(self):
        """Test register_validated in strict mode."""
        registry = get_registry()

        invalid_schema = ToolSchema(
            name="x",  # Too short
            description="Short",  # Too short
            source_agent="test_agent",
            tier=1,
        )

        with pytest.raises(ToolValidationError) as exc_info:
            registry.register_validated(invalid_schema, lambda: None, strict=True)

        assert exc_info.value.tool_name == "x"
        assert len(exc_info.value.errors) >= 2

    def test_register_validated_non_strict(self):
        """Test register_validated in non-strict mode."""
        registry = get_registry()

        schema = ToolSchema(
            name="x",  # Too short but allowed in non-strict
            description="Short",
            source_agent="test_agent",
            tier=1,
        )

        result = registry.register_validated(schema, lambda: None, strict=False)
        assert result is True
        assert registry.validate_tool_exists("x")

    def test_register_validated_success(self):
        """Test successful validated registration."""
        registry = get_registry()

        schema = ToolSchema(
            name="validated_tool",
            description="This is a fully valid tool description",
            source_agent="test_agent",
            tier=2,
        )

        result = registry.register_validated(schema, lambda: "result")
        assert result is True
        assert registry.get_callable("validated_tool")() == "result"


# =============================================================================
# DATABASE SYNCHRONIZATION TESTS (G3)
# =============================================================================


class TestDatabaseSync:
    """Tests for database synchronization methods."""

    @pytest.fixture(autouse=True)
    def clear_registry(self):
        """Clear registry before each test."""
        registry = get_registry()
        registry.clear()
        yield
        registry.clear()

    @pytest.mark.asyncio
    async def test_register_from_database(self):
        """Test loading tools from database."""
        registry = get_registry()

        # Mock database client
        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.data = [
            {
                "name": "db_tool_1",
                "description": "First database tool",
                "source_agent": "db_agent",
                "tier": 1,
                "avg_latency_ms": 100,
                "version": "1.0.0",
            },
            {
                "name": "db_tool_2",
                "description": "Second database tool",
                "source_agent": "db_agent",
                "tier": 2,
                "avg_latency_ms": 200,
                "version": "1.1.0",
            },
        ]
        mock_db.execute = AsyncMock(return_value=mock_result)

        count = await registry.register_from_database(mock_db)

        assert count == 2
        assert registry.validate_tool_exists("db_tool_1")
        assert registry.validate_tool_exists("db_tool_2")

        # Verify schema loaded correctly
        schema = registry.get_schema("db_tool_1")
        assert schema.avg_execution_ms == 100
        assert schema.version == "1.0.0"

    @pytest.mark.asyncio
    async def test_register_from_database_with_category_filter(self):
        """Test loading tools with category filter."""
        registry = get_registry()

        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.data = [
            {
                "name": "causal_tool",
                "description": "Causal analysis tool",
                "source_agent": "causal_agent",
                "tier": 2,
            }
        ]
        mock_db.execute = AsyncMock(return_value=mock_result)

        await registry.register_from_database(mock_db, category_filter=ToolCategory.CAUSAL)

        # Verify query included category filter
        call_args = mock_db.execute.call_args[0][0]
        assert "causal" in call_args.lower()

    @pytest.mark.asyncio
    async def test_register_from_database_error_handling(self):
        """Test error handling during database load."""
        registry = get_registry()

        mock_db = AsyncMock()
        mock_db.execute = AsyncMock(side_effect=Exception("Database connection failed"))

        with pytest.raises(ToolRegistryError) as exc_info:
            await registry.register_from_database(mock_db)

        assert "Database sync failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_placeholder_callable_raises(self):
        """Test that placeholder callable raises appropriate error."""
        registry = get_registry()

        mock_db = AsyncMock()
        mock_result = MagicMock()
        mock_result.data = [
            {
                "name": "placeholder_tool",
                "description": "Tool without implementation",
                "source_agent": "test_agent",
                "tier": 1,
            }
        ]
        mock_db.execute = AsyncMock(return_value=mock_result)

        await registry.register_from_database(mock_db)

        tool = registry.get("placeholder_tool")
        with pytest.raises(ToolNotFoundError) as exc_info:
            tool.callable()

        assert "no implementation" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_sync_to_database_insert(self):
        """Test syncing new tools to database."""
        registry = get_registry()

        # Register a tool
        schema = ToolSchema(
            name="new_tool",
            description="Brand new tool to sync",
            source_agent="test_agent",
            tier=1,
            input_parameters=[
                ToolParameter(name="x", type="int", description="Input")
            ],
        )
        registry.register(schema, lambda x: x)

        # Mock database client
        mock_db = AsyncMock()
        # First call checks existence, second inserts
        mock_check_result = MagicMock()
        mock_check_result.data = []  # Not exists
        mock_insert_result = MagicMock()

        mock_db.execute = AsyncMock(
            side_effect=[mock_check_result, mock_insert_result]
        )

        stats = await registry.sync_to_database(mock_db)

        assert stats["inserted"] == 1
        assert stats["updated"] == 0

    @pytest.mark.asyncio
    async def test_sync_to_database_update(self):
        """Test syncing existing tools to database."""
        registry = get_registry()

        schema = ToolSchema(
            name="existing_tool",
            description="Tool already in database",
            source_agent="test_agent",
            tier=2,
        )
        registry.register(schema, lambda: None)

        mock_db = AsyncMock()
        # Check returns existing record
        mock_check_result = MagicMock()
        mock_check_result.data = [{"tool_id": 1}]
        mock_update_result = MagicMock()

        mock_db.execute = AsyncMock(
            side_effect=[mock_check_result, mock_update_result]
        )

        stats = await registry.sync_to_database(mock_db, update_existing=True)

        assert stats["updated"] == 1
        assert stats["inserted"] == 0

    @pytest.mark.asyncio
    async def test_sync_to_database_skip_existing(self):
        """Test skipping existing tools when update_existing=False."""
        registry = get_registry()

        schema = ToolSchema(
            name="skip_tool",
            description="Tool to skip during sync",
            source_agent="test_agent",
            tier=1,
        )
        registry.register(schema, lambda: None)

        mock_db = AsyncMock()
        mock_check_result = MagicMock()
        mock_check_result.data = [{"tool_id": 1}]

        mock_db.execute = AsyncMock(return_value=mock_check_result)

        stats = await registry.sync_to_database(mock_db, update_existing=False)

        assert stats["skipped"] == 1
        assert stats["updated"] == 0
        # Only one call - the check
        assert mock_db.execute.call_count == 1


# =============================================================================
# DECORATOR TESTS
# =============================================================================


class TestComposableToolDecorator:
    """Tests for @composable_tool decorator."""

    @pytest.fixture(autouse=True)
    def clear_registry(self):
        """Clear registry before each test."""
        registry = get_registry()
        registry.clear()
        yield
        registry.clear()

    def test_decorator_registers_tool(self):
        """Test decorator registers tool in registry."""

        @composable_tool(
            name="decorated_tool",
            description="A decorated test tool",
            source_agent="decorator_test",
            tier=1,
        )
        def my_tool(x: int) -> int:
            return x * 2

        registry = get_registry()
        assert registry.validate_tool_exists("decorated_tool")

    def test_decorator_preserves_function(self):
        """Test decorator preserves original function behavior."""

        @composable_tool(
            name="preserved_tool",
            description="Tool with preserved behavior",
            source_agent="test_agent",
            tier=1,
        )
        def double(x: int) -> int:
            return x * 2

        assert double(5) == 10

    def test_decorator_attaches_schema(self):
        """Test decorator attaches schema to function."""

        @composable_tool(
            name="schema_attached",
            description="Tool with attached schema",
            source_agent="test_agent",
            tier=2,
            input_parameters=[
                {"name": "x", "type": "int", "description": "Input value"}
            ],
        )
        def tool_func(x: int) -> int:
            return x

        assert hasattr(tool_func, "_tool_schema")
        assert tool_func._tool_schema.name == "schema_attached"
        assert len(tool_func._tool_schema.input_parameters) == 1


# =============================================================================
# CONVENIENCE FUNCTION TESTS
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    @pytest.fixture(autouse=True)
    def clear_registry(self):
        """Clear registry before each test."""
        registry = get_registry()
        registry.clear()
        yield
        registry.clear()

    def test_register_tool_function(self):
        """Test register_tool convenience function."""

        def simple_func(x):
            return x

        register_tool(
            name="convenience_tool",
            callable=simple_func,
            description="Registered via convenience function",
            source_agent="convenience_test",
            tier=1,
        )

        registry = get_registry()
        assert registry.validate_tool_exists("convenience_tool")

    def test_list_available_tools_function(self):
        """Test list_available_tools convenience function."""
        register_tool(
            name="listed_tool",
            callable=lambda: None,
            description="Tool to be listed in planning",
            source_agent="listing_test",
            tier=1,
        )

        tools = list_available_tools()

        assert len(tools) >= 1
        tool_names = [t["name"] for t in tools]
        assert "listed_tool" in tool_names

    def test_get_registry_returns_singleton(self):
        """Test get_registry returns singleton instance."""
        reg1 = get_registry()
        reg2 = get_registry()

        assert reg1 is reg2


# =============================================================================
# CATEGORY FILTERING TESTS
# =============================================================================


class TestCategoryFiltering:
    """Tests for category-based tool filtering."""

    @pytest.fixture(autouse=True)
    def clear_registry(self):
        """Clear registry before each test."""
        registry = get_registry()
        registry.clear()
        yield
        registry.clear()

    def test_get_tools_by_category_placeholder(self):
        """Test category filtering returns placeholder result."""
        registry = get_registry()

        # Category filtering not yet implemented
        result = registry.get_tools_by_category(ToolCategory.CAUSAL)

        # Should return empty list with warning (see implementation)
        assert result == []
