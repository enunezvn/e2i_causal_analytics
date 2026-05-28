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
# Issue #501 — confounder-collider M-structure T → V ← U → Y (RED-first).
# Covers #242 correlated-failure cases 1,2,7,8,9 (the dominant pharmacoepi
# collider mode the current extractor mis-returns as ``descendant``).
# Plan .claude/plans/501_ac35_gate_implementation_plan.md §8.1.
# ---------------------------------------------------------------------------


def test_extract_role_m_structure_confounder_collider() -> None:
    """On {T→V, U→V, U→Y}, V is a confounder-collider M-structure → collider.

    U is an independent second parent (NOT T-downstream) with its own arrow into
    Y; conditioning on V opens the backdoor ``T → V ← U → Y``. The pre-#501
    extractor returns ``descendant`` here (RED first). Cases 1,2,7,8,9.
    """
    G = nx.DiGraph()
    G.add_edges_from([("T", "V"), ("U", "V"), ("U", "Y")])
    assert extract_role("V", "T", "Y", G) == "collider"


def test_extract_role_m_structure_does_not_fire_on_mediator_via_U() -> None:
    """On {T→V, U→V, V→Y}, U reaches Y ONLY through V → V is a mediator.

    The corrected predicate requires U to reach Y on a path that BYPASSES V; here
    no such path exists (U → V → Y), so the M-structure rule must NOT fire and V
    stays a mediator (on the T→V→Y path). Regression guard for the naive
    predicate that would have wrongly returned collider.
    """
    G = nx.DiGraph()
    G.add_edges_from([("T", "V"), ("U", "V"), ("V", "Y")])
    assert extract_role("V", "T", "Y", G) == "mediator"


def test_extract_role_m_structure_plus_on_path_is_collider() -> None:
    """On {T→V, U→V, U→Y, V→Y}, V is both on-path AND M-structure → collider.

    Pearl priority is collider > mediator: the independent U→Y arrow (bypassing
    V) makes V a collider even though V also lies on a T→V→Y path.
    """
    G = nx.DiGraph()
    G.add_edges_from([("T", "V"), ("U", "V"), ("U", "Y"), ("V", "Y")])
    assert extract_role("V", "T", "Y", G) == "collider"


def test_extract_role_literal_collider_still_fires_post_extension() -> None:
    """On {T→V, Y→V}, the literal common-descendant collider survives (cases 10,11).

    The M-structure rule's condition (b) excludes ``V ∈ descendants(Y)``, so the
    literal shape is still handled by Step 1, not double-handled.
    """
    G = nx.DiGraph()
    G.add_edges_from([("T", "V"), ("Y", "V")])
    assert extract_role("V", "T", "Y", G) == "collider"


def test_extract_role_m_structure_off_path_descendant_unaffected() -> None:
    """On {T→V, T→Y} (no independent second parent), V stays a descendant (cases 3,4,5).

    No M-structure parent exists, so the rule does not fire and the off-path
    descendant classification is preserved.
    """
    G = nx.DiGraph()
    G.add_edges_from([("T", "V"), ("T", "Y")])
    assert extract_role("V", "T", "Y", G) == "descendant"


def test_extract_role_m_structure_does_not_misfire_on_confounder() -> None:
    """On {Z→T, Z→Y}, Z is a parent of T (not a T-descendant) → confounder, not collider.

    Condition (a) requires V ∈ descendants(T); a confounder fails it, so the
    M-structure rule must not fire and Step 2 (confounder) governs.
    """
    G = nx.DiGraph()
    G.add_edges_from([("Z", "T"), ("Z", "Y"), ("T", "Y")])
    assert extract_role("Z", "T", "Y", G) == "confounder"


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


# ---------------------------------------------------------------------------
# Plan v4 Layer B / Phase 2 — pure ``derive_structural_role`` helper (Task 1).
# Lifts the graph-building + ``extract_role`` call out of the post-LLM telemetry
# helper into a pure, deterministic, zero-LLM-cost function the decider shares.
# ---------------------------------------------------------------------------
from src.data.feature_contract import CausalStructureAttestation, FeatureContract, KnowableAt
from src.ml.causal_role_dgp.extractor import derive_structural_role


def _contract_with(edges, *, role=None):
    return FeatureContract(
        name="V",
        knowable_at=KnowableAt(reference="index_date"),
        source="derived",
        causal_role=role,
        causal_structure=CausalStructureAttestation(
            treatment_node="T", outcome_node="Y", feature_node="V", edges=edges
        ),
    )


def test_derive_structural_role_confounder():
    role, err = derive_structural_role(_contract_with((("V", "T"), ("V", "Y"))))
    assert role == "confounder" and err is None


def test_derive_structural_role_collider():
    role, err = derive_structural_role(_contract_with((("T", "V"), ("Y", "V"))))
    assert role == "collider" and err is None


def test_derive_structural_role_none_when_unattested():
    role, err = derive_structural_role(
        FeatureContract(name="V", knowable_at=KnowableAt(reference="index_date"), source="demo")
    )
    assert role is None and err is None


def test_derive_structural_role_returns_error_on_unclassifiable():
    # T and Y ARE in the graph, but V has no classifiable relation to either →
    # extract_role's own ValueError ("unclassified node ...") fires. (The plan's
    # original fixture ``(("V","Z"),)`` instead omits T/Y entirely → a networkx
    # "node not in digraph" error whose message lacks "unclassified"; that path
    # is covered by the next test.)
    role, err = derive_structural_role(_contract_with((("T", "Y"), ("V", "Q"))))
    assert role is None
    assert err is not None and "unclassified" in err.lower()


def test_derive_structural_role_captures_malformed_graph_error():
    # Edges omit the declared T/Y nodes (an author typo) → networkx raises before
    # extract_role can classify. derive_structural_role must CAPTURE it as
    # (None, message) and never crash the calling node.
    role, err = derive_structural_role(_contract_with((("V", "Z"),)))
    assert role is None and err is not None
