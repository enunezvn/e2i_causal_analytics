"""Knowledge-graph writer integrity against a REAL FalkorDB (#2174, #2175, #2176, #2177).

The graph separates *curated* nodes (seed + causal-path sync) from *runtime*
nodes by one convention: a node is curated when it has no ``agent`` property
(``src/tasks/graph_reseed_tasks.py`` CURATED_COUNT_QUERY). These tests drive the
real writers — ``FalkorDBSemanticMemory``, the agent memory hooks, the seed's
Cypher and the causal-path sync — into a throwaway graph and read the result
back with Cypher. Nothing here is faked: a writer that does not execute, or
executes the wrong Cypher, fails the read-back.

Each test gets its own graph ``e2i_test_kg_<uuid>`` which is deleted on
teardown, so the production graph ``e2i_causal`` is never touched. Skipped when
``FALKORDB_URL`` is unset (CI lanes without a FalkorDB service).
"""

from __future__ import annotations

import importlib.util
import os
import sys
import uuid
from pathlib import Path
from typing import Any, List
from urllib.parse import urlparse

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.requires_falkordb]

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"_kg_{name}", _SCRIPTS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def graph():
    url = os.environ.get("FALKORDB_URL", "")
    if not url:
        pytest.skip("FALKORDB_URL unset — no FalkorDB in this lane")
    from falkordb import FalkorDB

    parsed = urlparse(url)
    db = FalkorDB(host=parsed.hostname, port=parsed.port or 6379, password=parsed.password)
    g = db.select_graph(f"e2i_test_kg_{uuid.uuid4().hex[:12]}")
    try:
        g.query("RETURN 1")
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"FalkorDB unreachable: {exc}")
    yield g
    try:
        g.delete()
    except Exception:
        pass


@pytest.fixture
def semantic(graph):
    from src.memory.semantic_memory import FalkorDBSemanticMemory

    sm = FalkorDBSemanticMemory()
    sm._graph = graph
    return sm


def rows(graph, cypher: str, params: dict | None = None) -> List[List[Any]]:
    return graph.query(cypher, params or {}).result_set


# ---------------------------------------------------------------------------
# #2174 — ownership properties are create-only
# ---------------------------------------------------------------------------


def test_create_only_properties_never_override_an_existing_node(graph, semantic):
    graph.query(
        "CREATE (:Variable {id: 'var:treatment_arm', name: 'treatment_arm', role: 'driver'})"
    )

    semantic.add_e2i_entity(
        entity_type="Variable",
        entity_id="var:treatment_arm",
        properties={"name": "treatment_arm"},
        create_only_properties={"agent": "causal_impact", "role": "treatment"},
    )
    semantic.add_e2i_entity(
        entity_type="Variable",
        entity_id="var:fresh",
        properties={"name": "fresh"},
        create_only_properties={"agent": "causal_impact", "role": "treatment"},
    )

    assert rows(graph, "MATCH (v:Variable {id:'var:treatment_arm'}) RETURN v.agent, v.role") == [
        [None, "driver"]
    ]
    assert rows(graph, "MATCH (v:Variable {id:'var:fresh'}) RETURN v.agent, v.role") == [
        ["causal_impact", "treatment"]
    ]


@pytest.mark.asyncio
async def test_causal_impact_path_leaves_a_curated_variable_curated(graph, semantic):
    from src.agents.causal_impact.memory_hooks import CausalImpactMemoryHooks

    graph.query(
        "CREATE (:Variable {id: 'var:treatment_arm', name: 'treatment_arm', role: 'driver'})"
    )
    hooks = CausalImpactMemoryHooks()
    hooks._semantic_memory = semantic

    stored = await hooks.store_causal_path(
        treatment_var="treatment_arm",
        outcome_var="brand_new_outcome",
        confounders=["region"],
        ate_estimate=0.1,
        confidence=0.9,
        refutation_passed=True,
        effect_size="small",
        brand="Kisqali",
    )

    assert stored is True
    assert rows(graph, "MATCH (v:Variable {id:'var:treatment_arm'}) RETURN v.agent, v.role") == [
        [None, "driver"]
    ]
    assert rows(
        graph, "MATCH (v:Variable {id:'var:brand_new_outcome'}) RETURN v.agent, v.role"
    ) == [["causal_impact", "outcome"]]
    assert rows(
        graph,
        "MATCH (:Variable {id:'var:treatment_arm'})-[r:CAUSES]->(:Variable {id:'var:brand_new_outcome'}) "
        "RETURN r.agent, r.brand",
    ) == [["causal_impact", "Kisqali"]]


@pytest.mark.asyncio
async def test_scope_definer_stamps_problem_type_and_keeps_curated_target(graph, semantic):
    from src.agents.ml_foundation.scope_definer.memory_hooks import ScopeDefinerMemoryHooks

    graph.query(
        "CREATE (:Variable {id: 'var:persistent_180d', name: 'persistent_180d', role: 'outcome'})"
    )
    hooks = ScopeDefinerMemoryHooks()
    hooks._semantic_memory = semantic

    assert await hooks.store_experiment_pattern(
        experiment_id="exp_a",
        experiment_name="a",
        problem_type="binary_classification",
        target_variable="persistent_180d",
        features=["f1"],
        success_criteria={},
    )

    assert rows(graph, "MATCH (p:ProblemType) RETURN p.id, p.agent") == [
        ["ptype:binary_classification", "scope_definer"]
    ]
    assert rows(graph, "MATCH (v:Variable {id:'var:persistent_180d'}) RETURN v.agent, v.role") == [
        [None, "outcome"]
    ]
    assert rows(
        graph,
        "MATCH (:Experiment {id:'exp:exp_a'})-[:TARGETS]->(v:Variable) RETURN v.id",
    ) == [["var:persistent_180d"]]


# ---------------------------------------------------------------------------
# #2175 — no nameless target Variable
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("missing_target", ["", "   ", None])
async def test_scope_definer_without_target_writes_no_target_variable(
    graph, semantic, missing_target
):
    from src.agents.ml_foundation.scope_definer.memory_hooks import ScopeDefinerMemoryHooks

    hooks = ScopeDefinerMemoryHooks()
    hooks._semantic_memory = semantic

    assert await hooks.store_experiment_pattern(
        experiment_id="exp_b",
        experiment_name="b",
        problem_type="regression",
        target_variable=missing_target,  # type: ignore[arg-type]
        features=[],
        success_criteria={},
    )

    assert rows(graph, "MATCH (v:Variable) RETURN count(v)") == [[0]]
    assert rows(graph, "MATCH ()-[r:TARGETS]->() RETURN count(r)") == [[0]]
    # The rest of the pattern still lands.
    assert rows(
        graph, "MATCH (e:Experiment {id:'exp:exp_b'})-[:HAS_TYPE]->(:ProblemType) RETURN count(e)"
    ) == [[1]]
    assert rows(
        graph, "MATCH (:Experiment {id:'exp:exp_b'})-[:DEFINED_BY]->(s:ScopeSpec) RETURN count(s)"
    ) == [[1]]


# ---------------------------------------------------------------------------
# #2174 — the curated sync claims the Variables it writes
# ---------------------------------------------------------------------------


def test_causal_path_sync_claims_variables_as_curated(graph):
    sync = _load_script("sync_causal_paths_to_falkordb")
    graph.query(
        "CREATE (:Variable {id: 'var:a', name: 'a', role: 'treatment', agent: 'causal_impact'})"
    )
    row = {
        "confidence_level": 0.8,
        "causal_effect_size": 0.2,
        "method_used": "dowhy",
        "brand": "Kisqali",
        "region": "northeast",
        "validation_status": "validated",
        "confirmation_count": 2,
        "discovery_date": None,
    }
    edges = sync._mediation_edges("a", [], "b")
    roles = sync._variable_roles(edges)

    counts = sync.write_chains(graph, [(row, edges)], roles)

    assert counts["written"] == 1
    assert rows(graph, "MATCH (v:Variable) RETURN v.id, v.agent, v.role ORDER BY v.id") == [
        ["var:a", None, roles["a"]],
        ["var:b", None, roles["b"]],
    ]


# ---------------------------------------------------------------------------
# #2176 — seeded Brands carry the canonical id the agents MERGE on
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cohort_brand_write_merges_onto_the_seeded_brand(graph, semantic):
    seed = _load_script("seed_falkordb")
    from src.agents.cohort_constructor.memory_hooks import CohortConstructorMemoryHooks

    for q in seed.generate_brand_queries():
        graph.query(q)
    hooks = CohortConstructorMemoryHooks()
    hooks._semantic_memory = semantic

    assert await hooks.store_eligibility_rule(
        rule_name="age",
        criterion={"field": "age", "operator": "gte", "value": 18},
        brand="kisqali",
        effectiveness_score=0.5,
        cohort_id="c1",
    )

    brands = rows(graph, "MATCH (b:Brand) RETURN b.name, b.id ORDER BY b.name")
    assert brands == [
        ["Fabhalta", "brand:Fabhalta"],
        ["Kisqali", "brand:Kisqali"],
        ["Remibrutinib", "brand:Remibrutinib"],
    ]
    assert rows(graph, "MATCH (b:Brand {name:'Kisqali'}) RETURN b.agent, b.indication") == [
        [None, "HR_HER2_breast_cancer"]
    ]


# ---------------------------------------------------------------------------
# #2177 — drift patterns land, stamped as runtime
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_drift_pattern_lands_and_is_runtime_owned(graph, semantic):
    from src.agents.drift_monitor.memory_hooks import DriftMonitorMemoryHooks

    graph.query("CREATE (:Feature {name: 'trx_volume'})")  # a curated feature node
    hooks = DriftMonitorMemoryHooks()
    hooks._semantic_memory = semantic
    result = {
        "features_with_drift": ["trx_volume", "nrx_o'brien"],
        "data_drift_results": [
            {
                "feature": "trx_volume",
                "drift_detected": True,
                "test_statistic": 0.4,
                "p_value": 0.01,
            },
            {
                "feature": "nrx_o'brien",
                "drift_detected": True,
                "test_statistic": 0.3,
                "p_value": 0.02,
            },
        ],
    }
    state = {"model_id": "model_1", "brand": "Kisqali"}

    assert await hooks.store_drift_pattern("trx_volume", "data", "high", result, state)
    assert await hooks.store_drift_pattern("nrx_o'brien", "data", "medium", result, state)

    assert rows(
        graph,
        "MATCH (f:Feature)-[:HAS_DRIFT]->(d:DriftPattern) "
        "RETURN f.name, d.drift_type, d.severity, d.agent ORDER BY f.name",
    ) == [
        ["nrx_o'brien", "data", "medium", "drift_monitor"],
        ["trx_volume", "data", "high", "drift_monitor"],
    ]
    # The pre-existing curated Feature is not taken over; the new one is runtime-owned.
    assert rows(graph, "MATCH (f:Feature) RETURN f.name, f.agent ORDER BY f.name") == [
        ["nrx_o'brien", "drift_monitor"],
        ["trx_volume", None],
    ]
    assert rows(graph, "MATCH ()-[r:CO_DRIFTS_WITH]->() RETURN count(r)") == [[2]]
    assert rows(
        graph,
        "MATCH (:DriftPattern)-[:AFFECTS_MODEL]->(m:Model {model_id:'model_1'}) RETURN count(m)",
    ) == [[2]]


# ---------------------------------------------------------------------------
# Codex R1 — parameter namespace, legacy ProblemType, one-off repair
# ---------------------------------------------------------------------------


def test_create_only_values_cannot_collide_with_caller_keys(graph, semantic):
    semantic.add_e2i_entity(
        entity_type="Variable",
        entity_id="var:x",
        properties={"name": "x", "co_agent": "caller-metadata"},
        create_only_properties={"agent": "scope_definer"},
    )

    assert rows(graph, "MATCH (v:Variable {id:'var:x'}) RETURN v.agent, v.co_agent") == [
        ["scope_definer", "caller-metadata"]
    ]


@pytest.mark.asyncio
async def test_scope_definer_repairs_a_pre_fix_problem_type(graph, semantic):
    from src.agents.ml_foundation.scope_definer.memory_hooks import ScopeDefinerMemoryHooks

    # Written by the pre-#2174 hook: no agent, so it counted as curated.
    graph.query("CREATE (:ProblemType {id: 'ptype:regression', name: 'regression'})")
    hooks = ScopeDefinerMemoryHooks()
    hooks._semantic_memory = semantic

    assert await hooks.store_experiment_pattern(
        experiment_id="exp_c",
        experiment_name="c",
        problem_type="regression",
        target_variable="",
        features=[],
        success_criteria={},
    )

    assert rows(graph, "MATCH (p:ProblemType) RETURN p.id, p.agent") == [
        ["ptype:regression", "scope_definer"]
    ]


def _seed_pre_fix_damage(graph) -> None:
    graph.query(
        "CREATE (:Experiment {id:'exp:1', agent:'scope_definer'})-[:TARGETS {agent:'scope_definer'}]->"
        "(:Variable {id:'var:', name:'', role:'target', agent:'scope_definer'})"
    )
    graph.query(
        "MATCH (e:Experiment {id:'exp:1'}), (v:Variable {id:'var:'}) CREATE (:Experiment {id:'exp:2', agent:'scope_definer'})-[:TARGETS]->(v)"
    )
    graph.query("CREATE (:ProblemType {id:'ptype:regression', name:'regression'})")
    # A validated synced edge whose endpoint an old hook stamped.
    graph.query(
        "CREATE (:Variable {id:'var:treatment_arm', name:'treatment_arm', role:'treatment', agent:'causal_impact'})"
        "-[:CAUSES {validation_status:'validated', brand:'Kisqali', region:'northeast'}]->"
        "(:Variable {id:'var:persistent_180d', name:'persistent_180d', role:'outcome'})"
    )
    # A genuinely agent-owned causal_impact pair must stay agent-owned.
    graph.query(
        "CREATE (:Variable {id:'var:accepted', agent:'causal_impact'})"
        "-[:CAUSES {agent:'causal_impact'}]->(:Variable {id:'var:converted', agent:'causal_impact'})"
    )


def test_repair_script_dry_run_changes_nothing(graph):
    repair = _load_script("repair_kg_ownership")
    _seed_pre_fix_damage(graph)

    report = repair.repair(graph, execute=False)

    assert report == {
        "empty_target_variables": 1,
        "unowned_problem_types": 1,
        "stamped_synced_variables": 1,
    }
    assert rows(graph, "MATCH (v:Variable {id:'var:'}) RETURN count(v)") == [[1]]
    assert rows(graph, "MATCH (p:ProblemType) RETURN p.agent") == [[None]]


def test_repair_script_execute_fixes_pre_fix_damage_and_is_idempotent(graph):
    repair = _load_script("repair_kg_ownership")
    _seed_pre_fix_damage(graph)

    repair.repair(graph, execute=True)
    second = repair.repair(graph, execute=False)

    assert second == {
        "empty_target_variables": 0,
        "unowned_problem_types": 0,
        "stamped_synced_variables": 0,
    }
    assert rows(graph, "MATCH (v:Variable {id:'var:'}) RETURN count(v)") == [[0]]
    assert rows(graph, "MATCH (e:Experiment) RETURN count(e)") == [[2]]  # experiments kept
    assert rows(graph, "MATCH (p:ProblemType) RETURN p.agent") == [["scope_definer"]]
    assert rows(graph, "MATCH (v:Variable {id:'var:treatment_arm'}) RETURN v.agent") == [[None]]
    assert rows(
        graph,
        "MATCH (v:Variable) WHERE v.id IN ['var:accepted','var:converted'] RETURN count(v.agent)",
    ) == [[2]]


@pytest.mark.asyncio
async def test_cohort_pattern_brand_write_keeps_the_seeded_brand_curated(graph, semantic):
    """Codex R3: both cohort Brand writers MERGE onto the seeded Brand without owning it
    (``agent`` rides the APPLIES_TO / FOR_BRAND edges, never the Brand node)."""
    seed = _load_script("seed_falkordb")
    from src.agents.cohort_constructor.memory_hooks import CohortConstructorMemoryHooks

    for q in seed.generate_brand_queries():
        graph.query(q)
    hooks = CohortConstructorMemoryHooks()
    hooks._semantic_memory = semantic

    assert await hooks.store_cohort_pattern(
        cohort_id="c2",
        cohort_name="pnh",
        brand="FABHALTA",
        indication="PNH",
        criteria_summary={},
        eligibility_rate=0.4,
    )

    assert rows(graph, "MATCH (b:Brand {name:'Fabhalta'}) RETURN count(b), collect(b.agent)") == [
        [1, []]
    ]
    assert rows(
        graph, "MATCH (:CohortConfig)-[r:FOR_BRAND]->(:Brand {id:'brand:Fabhalta'}) RETURN r.agent"
    ) == [["cohort_constructor"]]
