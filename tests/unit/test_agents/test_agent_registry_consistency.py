"""Drift-guard: the runtime agent registries must agree on the 21-agent roster.

Issue #607 (follow-up to #601). #601 fixed a doc/UI/API drift where the agent
count had splintered into "12 / 18 / 19 / 20" across many surfaces, all omitting
``experiment_monitor`` (and sometimes ``cohort_constructor`` / ``tool_composer``).
Nothing enforced agreement *between* the registries, so the drift could recur.

This test pins the "Family A" runtime registries to a single source of truth —
``src/agents/factory.py::AGENT_REGISTRY_CONFIG`` (the registry that actually
instantiates every agent) — so any future addition/removal of an agent that is
not propagated consistently fails fast in CI.

Deliberately NOT asserted here (separate, era-layered "Family B" vocabularies
tracked in #607 / a follow-up): the DB ``agent_name_type_v3`` enum,
``config/domain_vocabulary.yaml`` / ontology, and memory/observability Pydantic
enums. ``AGENT_METHOD_MAP`` is the *Tier 1-5 dispatcher contract* (13 agents by
design) and is asserted to equal factory's Tier 1-5 subset — it must NOT be
"fixed" to 21.
"""

from src.agents.factory import AGENT_REGISTRY_CONFIG
from src.agents.orchestrator._agent_method_map import AGENT_METHOD_MAP
from src.api.routes.agents import AGENT_REGISTRY

# Single source of truth.
FACTORY_IDS = set(AGENT_REGISTRY_CONFIG)
FACTORY_TIER_1_5 = {name for name, cfg in AGENT_REGISTRY_CONFIG.items() if cfg["tier"] != 0}
# The three agents whose omission caused the #601/#607 drift.
CONTESTED = {"experiment_monitor", "cohort_constructor", "tool_composer"}


def test_factory_registry_has_21_agents():
    """factory.AGENT_REGISTRY_CONFIG is the de-facto runtime source of truth (21 agents)."""
    assert len(AGENT_REGISTRY_CONFIG) == 21


def test_contested_agents_present_in_factory():
    """The three agents that drifted out of the docs/UI must exist in the SoT."""
    assert CONTESTED <= FACTORY_IDS


def test_api_registry_agrees_with_factory():
    """GET /api/agents/status (AGENT_REGISTRY) must expose exactly the factory roster.

    The API uses kebab-case ids; normalise to snake_case for comparison.
    """
    api_ids = {a.id.replace("-", "_") for a in AGENT_REGISTRY}
    assert api_ids == FACTORY_IDS, (
        f"API registry diverged from factory SoT: "
        f"only-in-api={api_ids - FACTORY_IDS}, only-in-factory={FACTORY_IDS - api_ids}"
    )
    assert len(AGENT_REGISTRY) == len(AGENT_REGISTRY_CONFIG)


def test_method_map_equals_factory_tier_1_5():
    """AGENT_METHOD_MAP is the Tier 1-5 dispatcher contract (13) = factory's non-Tier-0 set.

    This pins the by-design distinction: the orchestrator dispatches 13 Tier 1-5
    agents; the full platform is 21 (incl. 8 Tier-0 ML-foundation agents).
    """
    assert set(AGENT_METHOD_MAP) == FACTORY_TIER_1_5
    assert len(AGENT_METHOD_MAP) == 13


def test_tier_0_count_is_8():
    """Tier 0 (ML Foundation) must be 8 agents (incl. cohort_constructor)."""
    tier_0 = {name for name, cfg in AGENT_REGISTRY_CONFIG.items() if cfg["tier"] == 0}
    assert len(tier_0) == 8
    assert "cohort_constructor" in tier_0
