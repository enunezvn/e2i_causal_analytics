"""Regression test for #782: global ToolRegistry isolation across tests.

``ToolRegistry`` is a process-wide singleton (``registry.__new__`` returns
``cls._instance``). Several tool_composer / test_tool_registry fixtures and tests
call ``ToolRegistry().clear()`` / ``get_registry().clear()`` on it, which wipes
the REAL import-time-registered tools (e.g. ``causal_effect_estimator``). When a
later test in the same process depends on those tools — notably the integration
functional gate ``test_canonical_query_produces_real_tool_successes_stub_planner``
— it fails with ``Unknown tool in plan: causal_effect_estimator`` and 0 tools
succeeding.

The fix is an autouse, function-scoped finalizer in the root ``tests/conftest.py``
(``_restore_tool_registry_after_pollution``) that snapshots the global registry
before each test and restores it afterwards, mirroring the established
``_restore_asyncio_run_after_pollution`` pattern (#218).

These two ordered tests prove the cross-test restoration: ``test_a`` clears the
global singleton; ``test_b`` (which runs next in this module) asserts the real
tools are back. Without the finalizer, ``test_b`` fails.
"""

from __future__ import annotations

# Importing tool_registrations registers the real composable tools into the
# global singleton at collection time.
import src.agents.tool_composer.tool_registrations  # noqa: F401
from src.tool_registry.registry import ToolRegistry, get_registry


def test_a_clears_the_global_tool_registry():
    # Exactly what the conftest mock_tool_registry / empty_registry fixtures and
    # the direct registry.clear() calls in sibling test files do to the singleton.
    registry = ToolRegistry()  # the global singleton
    registry.clear()
    assert registry.tool_count == 0


def test_b_global_registry_is_restored_after_pollution():
    # Without the #782 autouse restore finalizer, the global registry would still
    # be empty here (test_a cleared it and nothing restored it). With the fix, the
    # real import-time tools are back.
    registry = get_registry()
    assert registry.get("causal_effect_estimator") is not None, (
        "global tool registry leaked a cleared state across tests (#782): "
        "causal_effect_estimator is missing after a prior test cleared the singleton"
    )
    # The full canonical tool_composer set (not 0, not just a few mocks).
    assert registry.tool_count >= 10
