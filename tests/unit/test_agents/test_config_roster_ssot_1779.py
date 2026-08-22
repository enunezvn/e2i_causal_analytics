"""#1779 / #1791: every agent roster must EQUAL the registry.

#1779 covers the five **YAML** rosters (set equality). #1791 adds the two
**Python** rosters in ``src/mlops/``, which need membership *and* value pinned
-- see the section header lower down for why one guard cannot do both here.

#1779: the YAML agent rosters must EQUAL the registry, as sets.

``cohort_profiler`` has been a real, enabled, dispatched Tier-0 agent since
d841d87b2, and four separate YAML rosters never learned about it. Every "21
agents" comment sitting above those rosters was *accurate for its own file* —
which is exactly why no reviewer caught the gap and why a find-and-replace of
21 → 22 would have been the wrong fix: it would have made five correct comments
wrong and hidden a missing agent behind a green docs diff.

So this file asserts nothing about prose. It asserts **set equality** between
``factory.AGENT_REGISTRY_CONFIG`` — the registry that actually instantiates
agents — and each YAML roster. A count in a comment can only ever be checked
against the file it sits in; set equality is checked against the SSOT, so it
fails on the commit that adds the 23rd agent rather than N months later.

The five rosters:

* ``config/agent_config.yaml`` ``agents:``            — per-agent tuning blocks
* ``config/agent_config.yaml`` ``routing.priority_order`` — dispatch ordering
* ``config/domain_vocabulary.yaml`` SECTION 2 tier lists — loaded at runtime by
  ``VocabularyRegistry`` / ``E2IQueryExtractor``
* ``config/ontology/node_types.yaml`` ``Agent.agent_name.values`` — the graph
  node enum, whose own comment claims alignment with SECTION 2
* ``config/observability.yaml`` ``agent_tiers.*.agents`` — per-tier Opik
  sampling policy, parsed by ``ObservabilityConfig.from_yaml``

The fifth was found by the codex audit of this change, and it had drifted
further than the others: three names missing (``cohort_constructor``,
``cohort_profiler``, ``experiment_monitor``), which is why it is pinned here
rather than patched for this one agent and left to drift again.

Deliberately NOT asserted here: the DB ``agent_name_type_v3`` enum
(``database/core/029_update_agent_enums_v4.sql``). It is a frozen 18-name
migration artifact from an era with agents that no longer exist
(``model_evaluator``, ``risk_assessor``, ...); ``scripts/validate_vocabulary_enum_sync.py``
already reports that mismatch and has done so since long before this issue.
Pinning it here would either fail on arrival or force a rewrite of a shipped
migration — see #607, which layered it out for the same reason.

**Known blind spot, stated rather than papered over.** ``backend-tests.yml``
gates its whole matrix on a path list that covers ``src/`` and ``tests/`` but
NOT ``config/``. So this guard runs on the change that matters — registering a
new agent necessarily edits ``src/agents/factory.py`` — but a *config-only*
edit that deletes an agent from a YAML roster would not trigger it. Closing
that means adding ``config/`` to the ``changes`` job pattern (and the mirrored
``push:`` filter), which makes every config edit run the full backend matrix:
a cost trade-off, not a defect fix, so it is deliberately left to a decision
rather than smuggled in here.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Set

import pytest
import yaml

from src.agents.factory import AGENT_REGISTRY_CONFIG, AGENT_TIER_NAMES
from src.mlops.opik_connector import OpikConnector
from src.mlops.slo_monitor import AGENT_TIER_MAP, AgentTier, get_agent_tier

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]

AGENT_CONFIG_PATH = REPO_ROOT / "config" / "agent_config.yaml"
DOMAIN_VOCAB_PATH = REPO_ROOT / "config" / "domain_vocabulary.yaml"
NODE_TYPES_PATH = REPO_ROOT / "config" / "ontology" / "node_types.yaml"
OBSERVABILITY_PATH = REPO_ROOT / "config" / "observability.yaml"

#: ``tier_0`` -> 0, for ``config/observability.yaml``'s flatter tier keys.
_OBS_TIER_KEY = re.compile(r"^tier_(\d+)$")

#: ``tier_0_ml_foundation`` -> 0. SECTION 2 spells the tier suffixes differently
#: from every other surface (``tier_2_causal`` vs ``causal_analytics``,
#: ``tier_4_prediction`` vs ``ml_predictions``), so the NUMBER is the only part
#: worth parsing -- which is also what ``VocabularyRegistry.get_agents(tier=N)``
#: keys off.
_VOCAB_TIER_KEY = re.compile(r"^tier_(\d+)_")

REGISTRY_IDS: Set[str] = set(AGENT_REGISTRY_CONFIG)


def _load(path: Path) -> Dict[str, Any]:
    with path.open() as fh:
        data: Dict[str, Any] = yaml.safe_load(fh)
    return data


def _registry_by_tier() -> Dict[int, Set[str]]:
    by_tier: Dict[int, Set[str]] = {}
    for name, cfg in AGENT_REGISTRY_CONFIG.items():
        by_tier.setdefault(int(cfg["tier"]), set()).add(name)
    return by_tier


def _tier_slug(tier: int) -> str:
    """``0`` -> ``ml_foundation``, derived from the registry's own tier labels.

    ``agent_config.yaml`` writes the tier as a slug rather than a number. Deriving
    the slug from :data:`AGENT_TIER_NAMES` keeps this test honest when a tier is
    added or renamed, instead of encoding a second copy of the mapping here.
    """
    return AGENT_TIER_NAMES[tier].lower().replace(" ", "_").replace("-", "_")


def _diff(actual: Set[str], label: str) -> str:
    return (
        f"{label} does not equal factory.AGENT_REGISTRY_CONFIG.\n"
        f"  missing from {label}: {sorted(REGISTRY_IDS - actual) or 'none'}\n"
        f"  present in {label} but NOT in the registry: {sorted(actual - REGISTRY_IDS) or 'none'}\n"
        f"  ({len(actual)} in the file vs {len(REGISTRY_IDS)} in the registry)"
    )


class TestAgentConfigRoster:
    """``config/agent_config.yaml`` holds two independent rosters; both must match."""

    @pytest.fixture(scope="class")
    def agent_config(self) -> Dict[str, Any]:
        return _load(AGENT_CONFIG_PATH)

    def test_agents_mapping_equals_registry(self, agent_config: Dict[str, Any]) -> None:
        keys: Set[str] = set(agent_config["agents"])
        assert keys == REGISTRY_IDS, _diff(keys, "config/agent_config.yaml agents:")

    def test_priority_order_equals_registry(self, agent_config: Dict[str, Any]) -> None:
        order: List[str] = agent_config["routing"]["priority_order"]
        assert len(order) == len(set(order)), f"duplicate entries in priority_order: {order}"
        assert set(order) == REGISTRY_IDS, _diff(
            set(order), "config/agent_config.yaml routing.priority_order"
        )

    def test_each_agent_declares_the_registry_tier(self, agent_config: Dict[str, Any]) -> None:
        """A roster can be complete and still put an agent in the wrong tier."""
        expected = {
            name: _tier_slug(int(cfg["tier"])) for name, cfg in AGENT_REGISTRY_CONFIG.items()
        }
        wrong = {
            name: (block.get("tier"), expected[name])
            for name, block in agent_config["agents"].items()
            if name in expected and block.get("tier") != expected[name]
        }
        assert not wrong, f"agents whose yaml tier disagrees with the registry (got, want): {wrong}"


class TestDomainVocabularyRoster:
    """SECTION 2 of ``config/domain_vocabulary.yaml`` -- a RUNTIME input.

    ``VocabularyRegistry.load()`` and ``E2IQueryExtractor`` both read this file,
    and ``get_agents(tier=N)`` / ``get_agent_names()`` serve these very lists, so
    an omission here is a hole in a live lookup rather than a documentation typo.
    """

    @pytest.fixture(scope="class")
    def tier_lists(self) -> Dict[int, Set[str]]:
        agents = _load(DOMAIN_VOCAB_PATH)["agents"]
        lists: Dict[int, Set[str]] = {}
        for key, value in agents.items():
            match = _VOCAB_TIER_KEY.match(key)
            if match and isinstance(value, list):
                lists[int(match.group(1))] = set(value)
        return lists

    def test_flattened_roster_equals_registry(self, tier_lists: Dict[int, Set[str]]) -> None:
        flat: Set[str] = set().union(*tier_lists.values()) if tier_lists else set()
        assert flat == REGISTRY_IDS, _diff(flat, "config/domain_vocabulary.yaml SECTION 2")

    def test_each_tier_list_equals_its_registry_tier(self, tier_lists: Dict[int, Set[str]]) -> None:
        expected = _registry_by_tier()
        assert set(tier_lists) == set(expected), (
            f"SECTION 2 declares tiers {sorted(tier_lists)}; the registry has {sorted(expected)}"
        )
        wrong = {
            tier: {
                "missing": sorted(expected[tier] - tier_lists[tier]),
                "unexpected": sorted(tier_lists[tier] - expected[tier]),
            }
            for tier in expected
            if tier_lists[tier] != expected[tier]
        }
        assert not wrong, f"SECTION 2 tier lists disagree with the registry: {wrong}"

    def test_agent_tiers_enum_covers_every_tier_list(self) -> None:
        """``agent_tiers.values`` is the DB-facing tier enum; it must name each list."""
        data = _load(DOMAIN_VOCAB_PATH)
        declared = set(data["agent_tiers"]["values"])
        used = {key for key in data["agents"] if _VOCAB_TIER_KEY.match(key)}
        assert used == declared, (
            f"agent_tiers.values {sorted(declared)} does not match the tier lists {sorted(used)}"
        )


class TestNodeTypesRoster:
    """``config/ontology/node_types.yaml`` ``Agent.agent_name`` is a name enum.

    Its own comment claims "aligned with domain_vocabulary.yaml Section 2", and
    ``tests/integration/test_ontology/test_vocabulary_enum_sync.py`` asserts that
    alignment -- but only between the two YAML files. Chained equality lets both
    drift away from the registry together, so this pins it to the registry.
    """

    def test_agent_name_values_equal_registry(self) -> None:
        values = _load(NODE_TYPES_PATH)["node_types"]["Agent"]["properties"]["agent_name"]["values"]
        assert len(values) == len(set(values)), (
            f"duplicate agent names in node_types.yaml: {values}"
        )
        assert set(values) == REGISTRY_IDS, _diff(
            set(values), "config/ontology/node_types.yaml Agent.agent_name.values"
        )


class TestObservabilityRoster:
    """``config/observability.yaml`` ``agent_tiers`` — the per-tier Opik policy.

    Found by the codex audit of #1779, and the worst-drifted of the five: it was
    short three agents. Note what is NOT claimed here — that the drift was
    changing sampling. ``ObservabilityConfig.get_agent_tier()`` /
    ``.get_sample_rate()`` have no production call sites, and ``OpikConnector``
    reads only ``opik:`` and ``sampling:`` from this file
    (``src/mlops/opik_connector.py:129-140``). The block is parsed and then
    unread, so this pins a roster that is *documentation with a parser*, and the
    test says so rather than implying a behavioural fix.
    """

    @pytest.fixture(scope="class")
    def tier_lists(self) -> Dict[int, Set[str]]:
        data = _load(OBSERVABILITY_PATH)["agent_tiers"]
        lists: Dict[int, Set[str]] = {}
        for key, block in data.items():
            match = _OBS_TIER_KEY.match(key)
            if match:
                lists[int(match.group(1))] = set(block.get("agents") or [])
        return lists

    def test_flattened_roster_equals_registry(self, tier_lists: Dict[int, Set[str]]) -> None:
        flat: Set[str] = set().union(*tier_lists.values()) if tier_lists else set()
        assert flat == REGISTRY_IDS, _diff(flat, "config/observability.yaml agent_tiers")

    def test_each_tier_list_equals_its_registry_tier(self, tier_lists: Dict[int, Set[str]]) -> None:
        expected = _registry_by_tier()
        assert set(tier_lists) == set(expected), (
            f"observability.yaml declares tiers {sorted(tier_lists)}; "
            f"the registry has {sorted(expected)}"
        )
        wrong = {
            tier: {
                "missing": sorted(expected[tier] - tier_lists[tier]),
                "unexpected": sorted(tier_lists[tier] - expected[tier]),
            }
            for tier in expected
            if tier_lists[tier] != expected[tier]
        }
        assert not wrong, f"observability.yaml tier lists disagree with the registry: {wrong}"


# =============================================================================
# #1791: the two PYTHON rosters in ``src/mlops/``
# =============================================================================
#
# #1779 (above) pinned five YAML rosters. Two more live in Python, and they are
# a different shape of wrong: they do not merely lag the registry, they
# *contradict* it, and one has done so since the commit that created it.
#
#   src/mlops/slo_monitor.py    AGENT_TIER_MAP                      20 entries
#       absent (2): cohort_profiler, experiment_monitor
#       wrong VALUE (1): cohort_constructor -> TIER_2_CAUSAL, registry says 0
#   src/mlops/opik_connector.py _get_agent_tier's tier_mapping       19 entries
#       absent (3): cohort_constructor, cohort_profiler, experiment_monitor
#       wrong VALUE (0)
#
# Note the asymmetry, because it is the whole argument for testing membership
# and value SEPARATELY: the opik roster is missing three names yet returns the
# right number for two of them, purely because ``.get(name, 0)`` defaults to 0
# and both cohort agents happen to be tier 0. A value-only guard sees 1 of 3
# defects there; a membership-only guard sees 0 of 1 in ``AGENT_TIER_MAP``.
#
# Both maps also convert an ABSENCE into a confident wrong VALUE
# (``AGENT_TIER_MAP.get(name, AgentTier.TIER_2_CAUSAL)`` /
# ``tier_mapping.get(name, 0)``), so "an agent I have never heard of" and "an
# agent deliberately placed in that tier" are indistinguishable downstream --
# the same fail-open shape as a probe whose failure path and negative result
# produce identical output. That is pinned here too.

#: ``{agent_name: tier NUMBER}`` straight off the registry -- the SSOT both
#: Python rosters are a projection of.
REGISTRY_TIERS: Dict[str, int] = {
    name: int(cfg["tier"]) for name, cfg in AGENT_REGISTRY_CONFIG.items()
}

#: The tier numbers that actually exist. An unknown agent must not be reported
#: as one of these.
REAL_TIER_NUMBERS: Set[int] = set(REGISTRY_TIERS.values())

#: ``AgentTier.TIER_0_FOUNDATION.value`` is the string ``"tier_0"``.
_AGENT_TIER_VALUE = re.compile(r"^tier_(\d+)$")

_NOT_AN_AGENT = "definitely_not_a_registered_agent_1791"


def _tier_number(tier: AgentTier) -> int:
    """``AgentTier.TIER_0_FOUNDATION`` -> ``0``.

    TRAP, and the reason the coercion is spelled out rather than inlined:
    ``AgentTier`` is a ``str`` enum whose members are ``"tier_0"`` .. ``"tier_5"``
    while the registry stores ``0`` .. ``5`` as ``int``. Comparing the two
    directly makes all 22 agents look wrong -- a probe that did exactly that
    reported "20 of 22" while this issue was being measured.
    :meth:`TestRegistryTierReadIsNotVacuous.test_agent_tier_members_are_strings`
    pins the premise this rests on, so the coercion cannot quietly become a no-op.
    """
    match = _AGENT_TIER_VALUE.match(tier.value)
    assert match is not None, f"AgentTier member {tier!r} does not look like 'tier_N'"
    return int(match.group(1))


def _roster_defects(roster: Dict[str, int]) -> Dict[str, Any]:
    """Diff a ``{agent_name: tier NUMBER}`` roster against the registry.

    Membership defects and value defects come back under *separate* keys, and
    callers assert on them separately on purpose -- see the note above on why
    collapsing them hides two thirds of this issue.
    """
    names = set(roster)
    return {
        "absent": sorted(REGISTRY_IDS - names),
        "unexpected": sorted(names - REGISTRY_IDS),
        "wrong_value": {
            name: (roster[name], REGISTRY_TIERS[name])
            for name in sorted(names & REGISTRY_IDS)
            if roster[name] != REGISTRY_TIERS[name]
        },
    }


#: ``AGENT_TIER_MAP`` exactly as it stood before this fix (f272766ae ..
#: 492118701), as tier NUMBERS. Frozen on purpose: it is the fixture that keeps
#: :class:`TestRosterComparatorIsNotVacuous` honest once both live rosters are
#: derived from the registry and every real comparison necessarily passes.
_HISTORICAL_SLO_ROSTER_1791: Dict[str, int] = {
    "scope_definer": 0,
    "data_preparer": 0,
    "feature_analyzer": 0,
    "model_selector": 0,
    "model_trainer": 0,
    "model_deployer": 0,
    "observability_connector": 0,
    "orchestrator": 1,
    "tool_composer": 1,
    "causal_impact": 2,
    "gap_analyzer": 2,
    "heterogeneous_optimizer": 2,
    "cohort_constructor": 2,  # <- the day-one contradiction; the registry says 0
    "drift_monitor": 3,
    "experiment_designer": 3,
    "health_score": 3,
    "prediction_synthesizer": 4,
    "resource_optimizer": 4,
    "explainer": 5,
    "feedback_learner": 5,
}

#: ``OpikConnector._get_agent_tier``'s function-local ``tier_mapping`` exactly
#: as it stood before this fix. Same purpose as the fixture above.
_HISTORICAL_OPIK_ROSTER_1791: Dict[str, int] = {
    "scope_definer": 0,
    "data_preparer": 0,
    "feature_analyzer": 0,
    "model_selector": 0,
    "model_trainer": 0,
    "model_deployer": 0,
    "observability_connector": 0,
    "orchestrator": 1,
    "tool_composer": 1,
    "causal_impact": 2,
    "gap_analyzer": 2,
    "heterogeneous_optimizer": 2,
    "drift_monitor": 3,
    "experiment_designer": 3,
    "health_score": 3,
    "prediction_synthesizer": 4,
    "resource_optimizer": 4,
    "explainer": 5,
    "feedback_learner": 5,
}


class TestRegistryTierReadIsNotVacuous:
    """Positive controls for the *probe*, before anything is asserted with it.

    Three separate probes written against this registry returned three
    different confident wrong answers (7 of 22, 20 of 22, 22 of 22). Every one
    was a bug in how the registry was READ, not in the registry. So the read is
    pinned before it is used.
    """

    def test_every_registry_agent_carries_an_int_tier(self) -> None:
        """If this ever reads 0, every roster assertion below is comparing to nothing."""
        assert len(REGISTRY_TIERS) > 0, "registry is empty -- every roster test below is vacuous"
        assert len(REGISTRY_TIERS) == len(AGENT_REGISTRY_CONFIG)
        non_int = {
            name: type(cfg.get("tier")).__name__
            for name, cfg in AGENT_REGISTRY_CONFIG.items()
            if not isinstance(cfg.get("tier"), int)
        }
        assert not non_int, f"registry entries whose tier is not an int: {non_int}"

    def test_registry_values_are_dicts_so_getattr_is_the_wrong_read(self) -> None:
        """TRAP: ``getattr(cfg, "tier", None)`` returns None for EVERY agent.

        It reads exactly like "the registry carries no tier data" and is the
        probe bug that reported 7 of 22. Pinned so nobody re-derives it.
        """
        assert all(isinstance(cfg, dict) for cfg in AGENT_REGISTRY_CONFIG.values())
        assert all(getattr(cfg, "tier", None) is None for cfg in AGENT_REGISTRY_CONFIG.values())

    def test_agent_tier_members_are_strings(self) -> None:
        """TRAP: ``AgentTier`` values are ``"tier_N"`` strings; registry tiers are ints."""
        assert all(isinstance(member.value, str) for member in AgentTier)
        assert AgentTier.TIER_0_FOUNDATION.value == "tier_0"
        assert _tier_number(AgentTier.TIER_0_FOUNDATION) == 0
        assert _tier_number(AgentTier.TIER_5_LEARNING) == 5
        # The int the registry stores is NOT equal to the enum member's value.
        assert AgentTier.TIER_0_FOUNDATION.value != 0

    def test_every_registry_tier_has_an_agent_tier_member(self) -> None:
        """A tier 6 in the registry with no ``AgentTier`` member must not pass silently."""
        members = {_tier_number(member) for member in AgentTier}
        assert REAL_TIER_NUMBERS <= members, (
            f"registry uses tiers {sorted(REAL_TIER_NUMBERS)} but AgentTier only "
            f"defines {sorted(members)}"
        )


class TestRosterComparatorIsNotVacuous:
    """The comparator must SEE a disagreement, not merely fail to find one.

    Once ``AGENT_TIER_MAP`` and the opik mapping are derived from the registry,
    every live comparison in this module passes *by construction* -- a guard
    comparing the registry to itself. These two tests are what stop that from
    being worthless: they run the same comparator over the frozen pre-fix
    rosters and assert it reports the exact defects #1791 measured.
    """

    def test_comparator_detects_the_historical_slo_defects(self) -> None:
        defects = _roster_defects(_HISTORICAL_SLO_ROSTER_1791)
        assert defects["absent"] == ["cohort_profiler", "experiment_monitor"]
        assert defects["unexpected"] == []
        assert defects["wrong_value"] == {"cohort_constructor": (2, 0)}

    def test_comparator_detects_the_historical_opik_defects(self) -> None:
        defects = _roster_defects(_HISTORICAL_OPIK_ROSTER_1791)
        assert defects["absent"] == [
            "cohort_constructor",
            "cohort_profiler",
            "experiment_monitor",
        ]
        assert defects["unexpected"] == []
        # The asymmetry that makes a value-only guard useless here: three names
        # are missing, yet NONE of the present ones carries a wrong number.
        assert defects["wrong_value"] == {}


class TestSLOMonitorTierRoster:
    """``src/mlops/slo_monitor.py`` ``AGENT_TIER_MAP`` -- membership AND value.

    Re-exported from ``src/mlops/__init__.py`` and read by
    ``SLOMonitor.get_tier_compliance`` (via ``.items()``) and by
    ``get_agent_tier``, which feeds ``get_slo_target`` ->
    ``DEFAULT_SLO_TARGETS``. A name missing here does not raise; it silently
    selects TIER_2 SLO targets.
    """

    def test_membership_equals_registry(self) -> None:
        assert set(AGENT_TIER_MAP) == REGISTRY_IDS, _diff(
            set(AGENT_TIER_MAP), "src/mlops/slo_monitor.py AGENT_TIER_MAP"
        )

    def test_every_value_equals_the_registry_tier(self) -> None:
        """Complete and still wrong: ``cohort_constructor`` was tier 2 here, 0 in the registry."""
        defects = _roster_defects({n: _tier_number(t) for n, t in AGENT_TIER_MAP.items()})
        assert not defects["wrong_value"], (
            "AGENT_TIER_MAP entries whose tier disagrees with the registry "
            f"(got, want): {defects['wrong_value']}"
        )

    def test_get_agent_tier_agrees_with_the_registry(self) -> None:
        """The accessor, not just the literal -- ``get_slo_target`` reads through it."""
        wrong = {
            name: (_tier_number(get_agent_tier(name)), tier)
            for name, tier in REGISTRY_TIERS.items()
            if _tier_number(get_agent_tier(name)) != tier
        }
        assert not wrong, f"get_agent_tier disagrees with the registry (got, want): {wrong}"

    def test_unknown_agent_is_reported_rather_than_defaulted(self) -> None:
        """An absence must not become a confident wrong answer.

        ``AGENT_TIER_MAP.get(name, AgentTier.TIER_2_CAUSAL)`` cannot say "I do
        not know this agent": it hands back real TIER_2 SLO targets for a typo.
        Once the map is complete by construction, a name that is not in it is a
        caller error, so it has to surface as one.
        """
        with pytest.raises(KeyError):
            get_agent_tier(_NOT_AN_AGENT)


class TestOpikConnectorTierRoster:
    """``OpikConnector._get_agent_tier`` -- a seventh roster, independent of both.

    Returns a plain ``int`` (not an ``AgentTier``) and is reachable from live
    import paths (``src/api/main.py``, ``src/api/routes/chatbot_tracer.py``,
    ``src/agents/base/audit_chain_mixin.py``, ...). It is dormant only because
    OPIK is switched off by env -- one flag from live, not one missing caller --
    so wrong tier metadata would start landing on spans the moment it flips.
    """

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """``OpikConnector`` is a singleton; don't leak one across tests."""
        OpikConnector._instance = None
        OpikConnector._initialized = False
        yield
        OpikConnector._instance = None
        OpikConnector._initialized = False

    @pytest.fixture()
    def connector(self) -> OpikConnector:
        return OpikConnector()

    def test_every_registry_agent_gets_its_registry_tier(self, connector: OpikConnector) -> None:
        """NOTE: this deliberately does NOT test membership.

        The roster is built by *asking* for each registry name, so ``absent``
        is empty by construction and asserting on it would be vacuous.
        Membership is covered behaviourally by
        :meth:`test_unknown_agent_is_not_reported_as_a_real_tier` -- an absent
        name is exactly one that falls through to the default.
        """
        roster = {name: connector._get_agent_tier(name) for name in REGISTRY_TIERS}
        defects = _roster_defects(roster)
        assert not defects["wrong_value"], (
            f"_get_agent_tier disagrees with the registry (got, want): {defects['wrong_value']}"
        )

    def test_returns_a_plain_int(self, connector: OpikConnector) -> None:
        """Guards the return type -- it lands in Opik span metadata, not an enum."""
        tier = connector._get_agent_tier("orchestrator")
        assert isinstance(tier, int)
        assert not isinstance(tier, AgentTier)

    def test_unknown_agent_is_not_reported_as_a_real_tier(self, connector: OpikConnector) -> None:
        """``tier_mapping.get(name, 0)`` labels every unknown agent Tier 0.

        Tier 0 is a REAL tier (ML Foundation), so an agent the mapping has
        never heard of is stamped onto the span as a plausible, wrong,
        indistinguishable value. This is also the only observable trace of the
        three missing names: two of them are tier 0 in the registry, so the
        default returns the right number for the wrong reason and no
        value-based assertion can see them.
        """
        assert connector._get_agent_tier(_NOT_AN_AGENT) not in REAL_TIER_NUMBERS
