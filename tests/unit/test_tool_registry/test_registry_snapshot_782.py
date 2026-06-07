"""Unit tests for ToolRegistry.snapshot() / restore_snapshot() (issue #782).

These power the root-conftest autouse finalizer that keeps the process-wide
ToolRegistry singleton from leaking a cleared/shadowed state across tests.
"""

from __future__ import annotations

# Register the real composable tools into the singleton so the snapshot has a
# realistic, non-empty set to protect.
import src.agents.tool_composer.tool_registrations  # noqa: F401
from src.tool_registry.registry import ToolParameter, ToolSchema, get_registry


def _mock_schema(name: str) -> ToolSchema:
    return ToolSchema(
        name=name,
        description="mock",
        source_agent="test",
        tier=2,
        input_parameters=[ToolParameter("x", "str", "x", True)],
        output_schema="Mock",
        avg_execution_ms=1,
    )


def test_restore_snapshot_re_adds_tools_after_clear():
    reg = get_registry()
    snap = reg.snapshot()
    assert "causal_effect_estimator" in snap["tools"]

    reg.clear()
    assert reg.get("causal_effect_estimator") is None
    assert reg.tool_count == 0

    reg.restore_snapshot(snap)
    assert reg.get("causal_effect_estimator") is not None
    assert reg.tool_count >= len(snap["tools"])


def test_restore_snapshot_unshadows_same_name_mock():
    reg = get_registry()
    snap = reg.snapshot()
    real = reg.get("causal_effect_estimator")
    assert real is not None

    # Shadow the real tool with a same-name mock (what the conftest fixtures do).
    reg.clear()
    reg.register(schema=_mock_schema("causal_effect_estimator"), callable=lambda **k: {})
    assert reg.get("causal_effect_estimator").callable is not real.callable

    reg.restore_snapshot(snap)
    # The original RegisteredTool is restored (mock un-shadowed).
    assert reg.get("causal_effect_estimator") is real


def test_restore_snapshot_removes_tools_added_after_snapshot():
    reg = get_registry()
    snap = reg.snapshot()

    reg.register(schema=_mock_schema("_test_only_tool_782"), callable=lambda **k: {})
    assert reg.get("_test_only_tool_782") is not None

    reg.restore_snapshot(snap)
    # Exact-reset: a stray tool registered AFTER the snapshot is removed, while
    # the snapshot's real tools remain. No cumulative leak across tests.
    assert reg.get("_test_only_tool_782") is None
    assert reg.get("causal_effect_estimator") is not None


def test_restore_snapshot_leaves_indexes_consistent():
    # Pollute with a clear + a stray registration, then restore. Every tool that
    # ends up in the registry must also be reachable via its agent and tier
    # indexes (no orphan _tools entry) -- the byte-consistency invariant.
    reg = get_registry()
    snap = reg.snapshot()

    reg.clear()
    reg.register(schema=_mock_schema("_stray_782"), callable=lambda **k: {})
    reg.restore_snapshot(snap)

    assert reg.get("_stray_782") is None
    for schema in reg.get_all_schemas():
        assert schema.name in reg.list_by_agent(schema.source_agent), (
            f"{schema.name} missing from agent index after restore"
        )
        assert schema.name in reg.list_by_tier(schema.tier), (
            f"{schema.name} missing from tier index after restore"
        )
