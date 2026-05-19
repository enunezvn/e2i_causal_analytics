"""S12a synthetic golden-set DGP — unit tests (TDD red-first).

Plan: ``.claude/plans/s12_synthetic_golden_set_plan.md`` §6.1.

Tests the mechanical role extractor, scenario builders, golden-set JSON
schema, and fixture-pin determinism for the synthetic precision/recall
baseline that gates S12 Option C's CausalRoleClassifier (recompiled in
PR #371, ``84c7adbc``).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import networkx as nx
import pytest

# Imports from the not-yet-written package — these import errors make every
# test RED at collection time per TDD red-first. The plan §3.5 specifies
# the module layout.
from src.ml.causal_role_dgp.extractor import extract_role
from src.ml.causal_role_dgp.golden_set import (
    DATASET_CONTEXT_REGEX,
    DERIVATION_PSEUDOCODE_REGEX,
    GOLDEN_SET_VERSION,
    build_golden_set,
)
from src.ml.causal_role_dgp.scenarios import (
    SCENARIO_BUILDERS,
    SCENARIO_NAMES,
    build_scenario,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_PATH = PROJECT_ROOT / "tests" / "fixtures" / "causal_role_golden_set_synthetic.json"


# ---------------------------------------------------------------------------
# §3.3 role extractor — one test per role + exogeneity edge case
# ---------------------------------------------------------------------------


def test_extract_role_confounder() -> None:
    """On {Z→T, Z→Y, T→Y}, Z is a confounder per Pearl-Lauritzen."""
    G = nx.DiGraph()
    G.add_edges_from([("Z", "T"), ("Z", "Y"), ("T", "Y")])
    assert extract_role("Z", "T", "Y", G) == "confounder"


def test_extract_role_collider() -> None:
    """On {T→V, Y→V}, V is a collider (common descendant of T and Y)."""
    G = nx.DiGraph()
    G.add_edges_from([("T", "V"), ("Y", "V")])
    assert extract_role("V", "T", "Y", G) == "collider"


def test_extract_role_mediator() -> None:
    """On {T→M→Y}, M is a mediator (on directed T→Y path)."""
    G = nx.DiGraph()
    G.add_edges_from([("T", "M"), ("M", "Y")])
    assert extract_role("M", "T", "Y", G) == "mediator"


def test_extract_role_descendant() -> None:
    """On {T→D} (no D→Y), D is a descendant of T (not on T→Y path)."""
    G = nx.DiGraph()
    G.add_edges_from([("T", "D"), ("T", "Y")])
    assert extract_role("D", "T", "Y", G) == "descendant"


def test_extract_role_instrument() -> None:
    """On {Z→T, T→Y} (no Z→Y, no common ancestor of Z and Y), Z is an instrument.

    Per §3.3 two-condition IV check: (i) no path Z→Y after removing T,
    AND (ii) no common ancestor with Y in G.
    """
    G = nx.DiGraph()
    G.add_edges_from([("Z", "T"), ("T", "Y")])
    assert extract_role("Z", "T", "Y", G) == "instrument"


def test_extract_role_instrument_fails_exogeneity_falls_to_ancestor() -> None:
    """On {U→Z, U→Y, Z→T, T→Y}, Z fails exogeneity and demotes to ancestor.

    Z has Z→T (relevance) and no path Z→Y after T removed (exclusion),
    but Z and Y share common ancestor U → violates Pearl IV exogeneity.
    Per §3.3 the extractor falls through to ancestor (next-best graph
    property: still pre-treatment-causal-of-Y, just not IV-clean).
    """
    G = nx.DiGraph()
    G.add_edges_from([("U", "Z"), ("U", "Y"), ("Z", "T"), ("T", "Y")])
    assert extract_role("Z", "T", "Y", G) == "ancestor"


def test_extract_role_ancestor() -> None:
    """On {A→Y} (no A→T, no path A→T), A is an ancestor of Y."""
    G = nx.DiGraph()
    G.add_edges_from([("A", "Y"), ("T", "Y")])
    assert extract_role("A", "T", "Y", G) == "ancestor"


def test_extract_role_unclassified_raises() -> None:
    """An isolated node with no relation to T or Y raises ValueError."""
    G = nx.DiGraph()
    G.add_edges_from([("T", "Y")])
    G.add_node("X")
    with pytest.raises(ValueError, match="unclassified"):
        extract_role("X", "T", "Y", G)


# ---------------------------------------------------------------------------
# §3.2 scenario builders — distribution / structure invariants
# ---------------------------------------------------------------------------


def test_scenario_A1_role_distribution() -> None:
    """A1 realized role counts match plan §3.2: Confounder=3, Ancestor=2,
    Descendant=2, Collider=1, Instrument=1."""
    scenario = build_scenario("A1_confounder_heavy")
    by_role: dict[str, int] = {}
    for entry in scenario.entries:
        by_role[entry.ground_truth_role] = by_role.get(entry.ground_truth_role, 0) + 1
    assert by_role.get("confounder") == 3, by_role
    assert by_role.get("ancestor") == 2, by_role
    assert by_role.get("descendant") == 2, by_role
    assert by_role.get("collider") == 1, by_role
    assert by_role.get("instrument") == 1, by_role


def test_scenario_A2_mediators_have_T_M_Y_path() -> None:
    """All A2 mediators must have a directed T→M→Y path in the DAG."""
    scenario = build_scenario("A2_mediator_heavy")
    G = scenario.dag
    T, Y = scenario.treatment_node, scenario.outcome_node
    mediators = [e.node_name for e in scenario.entries if e.ground_truth_role == "mediator"]
    assert len(mediators) >= 3, f"A2 expected ≥3 mediators; got {mediators}"
    for m in mediators:
        assert nx.has_path(G, T, m), f"no T→{m} path in A2 DAG"
        assert nx.has_path(G, m, Y), f"no {m}→Y path in A2 DAG"


def test_scenario_A3_colliders_have_T_V_Y_parent_pair() -> None:
    """All A3 colliders must have BOTH T and Y as direct parents in the DAG."""
    scenario = build_scenario("A3_descendant_collider_rich")
    G = scenario.dag
    T, Y = scenario.treatment_node, scenario.outcome_node
    colliders = [e.node_name for e in scenario.entries if e.ground_truth_role == "collider"]
    assert len(colliders) >= 3, f"A3 expected ≥3 colliders; got {colliders}"
    for v in colliders:
        assert G.has_edge(T, v), f"missing T→{v} in A3 DAG"
        assert G.has_edge(Y, v), f"missing Y→{v} in A3 DAG"


def test_scenario_A4_instruments_have_no_Y_path_after_T_removed() -> None:
    """All A4 instruments must satisfy the §3.3 two-condition IV check."""
    scenario = build_scenario("A4_instrument_rich")
    G = scenario.dag
    T, Y = scenario.treatment_node, scenario.outcome_node
    instruments = [e.node_name for e in scenario.entries if e.ground_truth_role == "instrument"]
    assert len(instruments) >= 3, f"A4 expected ≥3 instruments; got {instruments}"
    G_no_T = G.copy()
    G_no_T.remove_node(T)
    for iv in instruments:
        # Condition (i): no directed path IV→Y after T removed
        assert not (Y in G_no_T and nx.has_path(G_no_T, iv, Y)), (
            f"instrument {iv} has a path to Y bypassing T"
        )
        # Condition (ii): no common ancestor with Y in G
        common = nx.ancestors(G, iv) & nx.ancestors(G, Y)
        assert not common, f"instrument {iv} shares common ancestor with Y: {common}"


# ---------------------------------------------------------------------------
# §3.4 derivation_pseudocode regex — production exemplar coverage
# ---------------------------------------------------------------------------


def _render_production_derivation(
    *,
    source: str,
    derivation_inputs: tuple[str, ...],
    aggregation: Any,
    window_days: Any,
    knowable_at_str: str,
) -> str:
    """Reproduce the production f-string at adaptive_validity_check.py:879-885."""
    return (
        f"source={source}; "
        f"derivation_inputs={list(derivation_inputs)}; "
        f"aggregation={aggregation}; "
        f"window_days={window_days}; "
        f"knowable_at={knowable_at_str}"
    )


def test_derivation_regex_matches_production_exemplars() -> None:
    """The §4 regex must match ≥3 production-shape strings.

    Exemplars cover: event-typed contract with full fields, static
    contract with aggregation=None + window_days=None, date-offset
    knowable_at (index_date+180d).
    """
    pattern = re.compile(DERIVATION_PSEUDOCODE_REGEX)
    exemplars = [
        # Event-typed: medication count with 180d window
        _render_production_derivation(
            source="medication_events",
            derivation_inputs=("medication_date",),
            aggregation="count",
            window_days=180,
            knowable_at_str="index_date",
        ),
        # Static / demographic: aggregation=None, window_days=None
        _render_production_derivation(
            source="demographics",
            derivation_inputs=("date_of_birth",),
            aggregation=None,
            window_days=None,
            knowable_at_str="enrollment_date",
        ),
        # Date-offset knowable_at: post-index window
        _render_production_derivation(
            source="lab_events",
            derivation_inputs=("lab_value", "lab_date"),
            aggregation="max",
            window_days=180,
            knowable_at_str="index_date+180d",
        ),
    ]
    for ex in exemplars:
        assert pattern.match(ex), f"regex rejected production exemplar: {ex!r}"


# ---------------------------------------------------------------------------
# §4 golden-set JSON schema invariants
# ---------------------------------------------------------------------------


def test_golden_set_schema() -> None:
    """Each entry of the golden set must satisfy the §4 schema invariants."""
    golden = build_golden_set()
    deriv_re = re.compile(DERIVATION_PSEUDOCODE_REGEX)
    ctx_re = re.compile(DATASET_CONTEXT_REGEX)
    valid_roles = {"ancestor", "confounder", "mediator", "collider", "descendant", "instrument"}
    scenario_names = {s["name"] for s in golden["scenarios"]}

    assert golden["version"] == GOLDEN_SET_VERSION
    for entry in golden["entries"]:
        assert entry["scenario"] in scenario_names, entry
        assert entry["ground_truth_role"] in valid_roles, entry
        assert deriv_re.match(entry["derivation_pseudocode"]), entry["derivation_pseudocode"]
        assert ctx_re.match(entry["dataset_context"]), entry["dataset_context"]
        assert entry["treatment_explicit"] == ("treatment=" in entry["dataset_context"])


def test_golden_set_role_coverage() -> None:
    """All 6 classifier roles must appear in the golden set."""
    golden = build_golden_set()
    roles_seen = {e["ground_truth_role"] for e in golden["entries"]}
    assert roles_seen == {
        "ancestor",
        "confounder",
        "mediator",
        "collider",
        "descendant",
        "instrument",
    }, roles_seen


def test_golden_set_size_floor() -> None:
    """Family A (cohort-only) must have ≥30 entries per §3.2."""
    golden = build_golden_set()
    family_a = [e for e in golden["entries"] if not e["treatment_explicit"]]
    assert len(family_a) >= 30, len(family_a)


def test_golden_set_family_split_invariant() -> None:
    """Every Family B entry mirrors a Family A entry (same scenario+feature)."""
    golden = build_golden_set()
    family_a = {
        (e["scenario"], e["feature_name"]) for e in golden["entries"] if not e["treatment_explicit"]
    }
    family_b = {
        (e["scenario"], e["feature_name"]) for e in golden["entries"] if e["treatment_explicit"]
    }
    # Family B is a strict mirror of Family A: same rows, same feature/
    # derivation/role; only dataset_context differs. Equality catches both
    # B-missing-entries-from-A AND B-having-entries-not-in-A regressions.
    assert family_b == family_a, {
        "missing_b": family_a - family_b,
        "extra_b": family_b - family_a,
    }


def test_golden_set_fixture_pin() -> None:
    """The checked-in fixture must match the regenerated golden set.

    Compares dict-equality via json.loads after excluding volatile fields
    (per §4, the persisted fixture itself omits generated_at/generator_commit
    so semantic compare suffices).
    """
    if not FIXTURE_PATH.exists():
        pytest.fail(
            f"fixture missing at {FIXTURE_PATH}; "
            f"run `python scripts/build_causal_role_golden_set.py` to write it"
        )
    on_disk = json.loads(FIXTURE_PATH.read_text())
    regenerated = build_golden_set()
    assert on_disk == regenerated, (
        "fixture drift: regenerated golden set does not match the checked-in "
        f"fixture at {FIXTURE_PATH}. Diff the two and either (a) update the "
        f"fixture to match the new scenarios or (b) revert the scenario "
        f"changes if unintended."
    )


# ---------------------------------------------------------------------------
# §3.5 reuse-vs-new — sanity invariants on the API surface
# ---------------------------------------------------------------------------


def test_scenarios_registry_completeness() -> None:
    """SCENARIO_NAMES enumerates all 4 plan §3.2 scenarios."""
    expected = {
        "A1_confounder_heavy",
        "A2_mediator_heavy",
        "A3_descendant_collider_rich",
        "A4_instrument_rich",
    }
    assert set(SCENARIO_NAMES) == expected, set(SCENARIO_NAMES)
    assert set(SCENARIO_BUILDERS.keys()) == expected, set(SCENARIO_BUILDERS.keys())


def test_scenario_dag_is_acyclic() -> None:
    """Every scenario DAG must be acyclic (DAG invariant of §3.3)."""
    for name in SCENARIO_NAMES:
        s = build_scenario(name)
        assert nx.is_directed_acyclic_graph(s.dag), f"{name} DAG has a cycle"


def test_scenario_dag_contains_T_and_Y() -> None:
    """Every scenario DAG must contain its declared T and Y nodes."""
    for name in SCENARIO_NAMES:
        s = build_scenario(name)
        assert s.treatment_node in s.dag.nodes, f"{name} missing T node"
        assert s.outcome_node in s.dag.nodes, f"{name} missing Y node"
