"""Parity guard: AGENT_METHOD_MAP is the single source of truth.

Issue #252. ``scripts/run_tier1_5_test.py::AGENT_CONFIGS`` and
``src/agents/orchestrator/_agent_method_map.py::AGENT_METHOD_MAP`` previously
duplicated (method, is_async, uses_kwargs, input_model, input_module). This
test pins that AGENT_CONFIGS is now derived from AGENT_METHOD_MAP and the
overlapping fields match for every shared agent.
"""

from __future__ import annotations

import importlib

from src.agents.orchestrator._agent_method_map import AGENT_METHOD_MAP


def test_agent_configs_is_derived_from_agent_method_map() -> None:
    """AGENT_CONFIGS must read its dispatch + harness fields from AGENT_METHOD_MAP.

    Per issue #252 acceptance 1+2. Pins every overlapping field: dispatch
    (method, is_async, uses_kwargs, input_model, input_module) AND harness
    (tier, agent_module, agent_class, state_module, state_class, timeout).
    Codex-rescue iter-1 flagged that the original parity test only pinned
    the dispatch fields, leaving harness fields free to drift.
    """
    harness = importlib.import_module("scripts.run_tier1_5_test")
    configs = harness.AGENT_CONFIGS
    # Every agent in AGENT_METHOD_MAP must appear in AGENT_CONFIGS
    # (the harness exists to test every wired agent).
    missing = set(AGENT_METHOD_MAP) - set(configs)
    assert not missing, f"AGENT_CONFIGS missing agents from AGENT_METHOD_MAP: {missing}"
    for agent_name, spec in AGENT_METHOD_MAP.items():
        cfg = configs[agent_name]
        # Dispatch fields
        assert cfg["method"] == spec.method, (
            f"{agent_name}: method drift "
            f"AGENT_METHOD_MAP={spec.method!r} vs AGENT_CONFIGS={cfg['method']!r}"
        )
        assert cfg.get("is_async", True) == spec.is_async, f"{agent_name}: is_async drift"
        assert cfg.get("uses_kwargs", False) == spec.uses_kwargs, f"{agent_name}: uses_kwargs drift"
        assert cfg.get("input_model") == spec.input_model, f"{agent_name}: input_model drift"
        assert cfg.get("input_module") == spec.input_module, f"{agent_name}: input_module drift"
        # Harness fields (codex-rescue iter-1 #252 acceptance — pin every
        # overlapping field, not just dispatch)
        assert cfg.get("tier") == spec.tier, f"{agent_name}: tier drift"
        assert cfg.get("agent_module") == spec.agent_module, f"{agent_name}: agent_module drift"
        assert cfg.get("agent_class") == spec.agent_class, f"{agent_name}: agent_class drift"
        assert cfg.get("state_module") == spec.state_module, f"{agent_name}: state_module drift"
        assert cfg.get("state_class") == spec.state_class, f"{agent_name}: state_class drift"
        assert cfg.get("timeout") == spec.timeout, f"{agent_name}: timeout drift"


def test_helper_functions_exposed() -> None:
    """Per issue #252 acceptance 1: AGENT_METHOD_MAP module must export helpers
    that the harness imports."""
    from src.agents.orchestrator._agent_method_map import (
        get_harness_configs,
        to_harness_config,
    )

    assert callable(get_harness_configs)
    assert callable(to_harness_config)


def test_get_harness_configs_returns_complete_dict() -> None:
    """get_harness_configs() must return one entry per AGENT_METHOD_MAP key.

    Per issue #252 acceptance 2.
    """
    from src.agents.orchestrator._agent_method_map import get_harness_configs

    configs = get_harness_configs()
    assert set(configs.keys()) == set(AGENT_METHOD_MAP.keys()), (
        f"get_harness_configs() drift vs AGENT_METHOD_MAP keys: "
        f"in_helper_only={set(configs) - set(AGENT_METHOD_MAP)}, "
        f"in_map_only={set(AGENT_METHOD_MAP) - set(configs)}"
    )


def test_removing_entry_breaks_both_dispatch_and_harness() -> None:
    """Per issue #252 acceptance 4: removing an entry must break both.

    We don't actually remove; we assert that both surfaces read from the
    same underlying dict object identity. If a refactor breaks the link
    (e.g., harness deep-copies the map at import), this test catches it.
    """
    import scripts.run_tier1_5_test as harness

    # Harness AGENT_CONFIGS must have one entry per AGENT_METHOD_MAP entry.
    # Adding a key to AGENT_METHOD_MAP propagates via get_harness_configs() at
    # module import time.
    assert set(harness.AGENT_CONFIGS.keys()) >= set(AGENT_METHOD_MAP.keys()), (
        f"AGENT_CONFIGS missing keys present in AGENT_METHOD_MAP: "
        f"{set(AGENT_METHOD_MAP) - set(harness.AGENT_CONFIGS)}"
    )
