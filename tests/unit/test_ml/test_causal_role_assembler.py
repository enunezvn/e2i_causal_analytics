"""Lane B (real-data causal estimation, 2026-09-22) — cohort-DAG assembler on
hand-built fragments (spec §6: "assembler on hand-built fragments (latent,
M-structure)"). No LM, no network: every record is built the way the author
would build it, with the fragment role derived by ``extract_role``.
"""

from __future__ import annotations

import json

import networkx as nx
import pytest

from src.data.kg.structural_author import AuthoredAttestation, EdgeProvenance
from src.ml.causal_role_dgp.assembler import ADJUSTABLE_ROLES, assemble_cohort_dag
from src.ml.causal_role_dgp.backdoor import satisfies_backdoor_criterion
from src.ml.causal_role_dgp.extractor import extract_role

T, Y = "treatment_dupixent", "persistent_at_180d_g28"


def _att(
    feature,
    edges,
    *,
    grades=None,
    panel=None,
    violation_edge=None,
    review=False,
    reasons=(),
    ambiguous=False,
):
    """An ``AuthoredAttestation`` as the author emits it for ``edges`` over
    {feature, "T", "Y", U_*}; the fragment role comes from the extractor."""
    g = nx.DiGraph(edges)
    try:
        role = extract_role(feature, "T", "Y", g)
    except (ValueError, nx.NodeNotFound):
        role = None
    grades = grades or {}
    prov = []
    for a, b in edges:
        grade = "estimand" if (a, b) == ("T", "Y") else grades.get((a, b), "unsupported")
        ep = EdgeProvenance(from_node=a, to_node=b, grade=grade, rationale="", citations=[])
        if violation_edge == (a, b):
            ep.constraint_violation = "layer_1_post_index_forbids_feature_to_T"
        prov.append(ep)
    return AuthoredAttestation(
        feature_name=feature,
        treatment_node="T",
        outcome_node="Y",
        feature_node=feature,
        edges=[list(e) for e in edges],
        edge_provenance=prov,
        derived_role=role,
        expected_role=role,
        ambiguous=ambiguous,
        review_required=review or role is None,
        review_reasons=list(reasons),
        cross_check={"derived_role": role, "panel_final_role": None, "agrees": None},
        constraint_violations=(
            ["layer_1_post_index_forbids_feature_to_T"] if violation_edge else []
        ),
        panel_summary=panel or {"present": False},
        latents=sorted({n for e in edges for n in e if n.startswith("U_")}),
        provenance="machine",
        model_id="dummy",
        prompt_hash="p" * 64,
        guide_hash="g" * 64,
        authored_at="2026-09-22T00:00:00+00:00",
    )


def _latent_confounder():
    # B is a direct parent of T and Y, with an unmeasured common cause U_sev
    # into both B and Y: a confounder whose latent parent is blocked BY B.
    return _att(
        "baseline_uas7",
        [
            ("baseline_uas7", "T"),
            ("baseline_uas7", "Y"),
            ("U_sev", "baseline_uas7"),
            ("U_sev", "Y"),
            ("T", "Y"),
        ],
        grades={("baseline_uas7", "T"): "direct", ("baseline_uas7", "Y"): "family"},
    )


def _ancestor():
    return _att("family_atopy", [("family_atopy", "Y"), ("T", "Y")])


def _instrument():
    return _att("prescriber_pref", [("prescriber_pref", "T"), ("T", "Y")])


def _m_structure():
    # T -> V <- U_dis -> Y : collider (M-structure), a leak role.
    return _att(
        "on_therapy_90d",
        [("T", "on_therapy_90d"), ("U_dis", "on_therapy_90d"), ("U_dis", "Y"), ("T", "Y")],
    )


def test_latent_and_m_structure_fragments_assemble_with_a_valid_adjustment_set():
    dag = assemble_cohort_dag(
        [_latent_confounder(), _ancestor(), _instrument(), _m_structure()],
        treatment=T,
        outcome=Y,
    )
    assert dag.is_dag is True
    # Anchors mapped to the cohort's column names; latents kept by name.
    assert dag.nodes[:2] == [T, Y]
    assert dag.latent_nodes == ["U_dis", "U_sev"]
    assert [T, Y] in dag.active_edges() and ["baseline_uas7", T] in dag.active_edges()
    assert not any("T" == e[0] or "Y" == e[1] for e in dag.active_edges())
    # Fragment roles are re-derived on the union without drift here.
    roles = {f.feature: (f.fragment_role, f.cohort_role, f.role_drift) for f in dag.features}
    assert roles == {
        "baseline_uas7": ("confounder", "confounder", False),
        "family_atopy": ("ancestor", "ancestor", False),
        "prescriber_pref": ("instrument", "instrument", False),
        "on_therapy_90d": ("collider", "collider", False),
    }
    # Adjustment: full observed admissible set = confounder + ancestor; the
    # greedy-minimal set is the confounder alone (it blocks T <- B <- U_sev -> Y).
    assert dag.adjustment_valid is True
    assert dag.adjustment_set == ["baseline_uas7", "family_atopy"]
    assert dag.minimal_adjustment_set == ["baseline_uas7"]
    g = nx.DiGraph(dag.active_edges())
    assert satisfies_backdoor_criterion(g, dag.minimal_adjustment_set, T, Y)
    excl = {f.feature: f.adjustment_exclusion for f in dag.features}
    assert excl["prescriber_pref"] == "cohort role instrument is not a backdoor variable"
    assert excl["on_therapy_90d"] == "cohort role collider is not a backdoor variable"
    assert excl["baseline_uas7"] is None
    assert "instrument" not in ADJUSTABLE_ROLES
    # The estimand edge is shared by every fragment; the best grade wins per edge.
    by_key = {(e.from_node, e.to_node): e for e in dag.edges}
    assert sorted(by_key[(T, Y)].authored_by) == sorted(
        ["baseline_uas7", "family_atopy", "prescriber_pref", "on_therapy_90d"]
    )
    assert by_key[("baseline_uas7", T)].grade == "direct"
    assert by_key[("U_sev", Y)].grade == "unsupported"


def test_dag_structure_json_matches_the_dag_panel_snapshot_shape():
    dag = assemble_cohort_dag([_latent_confounder(), _ancestor()], treatment=T, outcome=Y)
    snap = dag.dag_structure_json()
    assert set(snap) >= {"nodes", "edges", "treatment_nodes", "outcome_nodes", "adjustment_sets"}
    assert snap["treatment_nodes"] == [T] and snap["outcome_nodes"] == [Y]
    assert all(isinstance(e, list) and len(e) == 2 for e in snap["edges"])
    assert snap["adjustment_sets"] == [["baseline_uas7"], ["baseline_uas7", "family_atopy"]]
    assert snap["latent_nodes"] == ["U_sev"]
    # Every node an edge names is in nodes (DagPanel maps ids -> nodes).
    named = {n for e in snap["edges"] for n in e}
    assert named <= set(snap["nodes"])
    # Serialisable end to end, and the evidence table has one row per feature.
    payload = json.loads(json.dumps(dag.to_dict()))
    assert [row["feature"] for row in payload["features"]] == ["baseline_uas7", "family_atopy"]
    assert payload["features"][0]["in_minimal_adjustment_set"] is True
    # The ancestor is admissible (in the full set) but not needed (not minimal).
    assert payload["features"][1]["in_adjustment_set"] is True
    assert payload["features"][1]["in_minimal_adjustment_set"] is False
    assert payload["features"][1]["adjustment_exclusion"] is None
    # Records may also arrive as their JSON dicts (attestations.json).
    again = assemble_cohort_dag(
        [_latent_confounder().to_dict(), _ancestor().to_dict()], treatment=T, outcome=Y
    )
    assert again.to_dict() == dag.to_dict()


def test_leak_verdict_feature_is_excluded_from_adjustment_whatever_its_edges():
    """Spec §3 Lane E 3(b): a Layer-3 high veto excludes the feature from any
    adjustment set — the authored confounder edges stay in the DAG."""
    leaky = _att(
        "post_index_visits",
        [("post_index_visits", "T"), ("post_index_visits", "Y"), ("T", "Y")],
        panel={
            "present": True,
            "leak_verdict": True,
            "leak_source": "layer_3_high",
            "ensemble_final_role": None,
        },
        review=True,
        reasons=["panel leak verdict (layer_3_high)"],
    )
    dag = assemble_cohort_dag([leaky, _ancestor()], treatment=T, outcome=Y)
    row = next(f for f in dag.features if f.feature == "post_index_visits")
    assert row.cohort_role == "confounder"  # the authored structure says so
    assert row.leak_verdict is True and row.leak_source == "layer_3_high"
    assert row.in_adjustment_set is False
    assert row.adjustment_exclusion == "panel leak verdict (layer_3_high)"
    assert ["post_index_visits", T] in dag.active_edges()
    # Without the leaky confounder the observed set cannot block T <- V -> Y.
    assert dag.adjustment_valid is False
    assert any("no admissible OBSERVED adjustment set" in w for w in dag.warnings)
    assert dag.dag_structure_json()["adjustment_sets"] == []


def test_layer_1_constraint_drops_the_forbidden_edge_but_keeps_its_provenance():
    att = _att(
        "post_index_flag",
        [("post_index_flag", "T"), ("post_index_flag", "Y"), ("T", "Y")],
        violation_edge=("post_index_flag", "T"),
        panel={"present": True, "leak_verdict": True, "leak_source": "layer_1_post_index"},
        review=True,
    )
    dag = assemble_cohort_dag([att, _ancestor()], treatment=T, outcome=Y)
    assert ["post_index_flag", T] not in dag.active_edges()
    dropped = next(e for e in dag.edges if (e.from_node, e.to_node) == ("post_index_flag", T))
    assert dropped.dropped is True
    assert dropped.dropped_reason == "layer_1_post_index_forbids_feature_to_T"
    assert dropped.authored_by == ["post_index_flag"]
    assert any("dropped: layer_1_post_index_forbids_feature_to_T" in w for w in dag.warnings)
    # With the edge gone the feature is a pure ancestor on the union: drift is
    # reported, never silently accepted.
    row = next(f for f in dag.features if f.feature == "post_index_flag")
    assert (row.fragment_role, row.cohort_role, row.role_drift) == ("confounder", "ancestor", True)
    assert row.review_required is True


def test_unblockable_latent_backdoor_is_a_finding_not_an_empty_set():
    # U_h -> T and U_h -> Y with no observed child on that path: nothing observed
    # blocks it. P is an honest ancestor; the full set {P} still fails.
    p = _att("prognostic", [("prognostic", "Y"), ("U_h", "T"), ("U_h", "Y"), ("T", "Y")])
    dag = assemble_cohort_dag([p], treatment=T, outcome=Y)
    assert dag.is_dag is True
    assert dag.adjustment_valid is False
    assert dag.adjustment_set == [] and dag.minimal_adjustment_set == []
    assert any("U_h" in w for w in dag.warnings)
    assert dag.dag_structure_json()["adjustment_sets"] == []


def test_shared_latent_can_close_a_cycle_and_is_reported_not_broken():
    a = _att("feat_a", [("feat_a", "U_x"), ("U_z", "feat_a"), ("feat_a", "Y"), ("T", "Y")])
    b = _att("feat_b", [("U_x", "U_z"), ("feat_b", "T"), ("T", "Y")])
    dag = assemble_cohort_dag([a, b], treatment=T, outcome=Y)
    assert dag.is_dag is False
    assert any("not acyclic" in w for w in dag.warnings)
    rows = {f.feature: f for f in dag.features}
    assert rows["feat_a"].review_required is True
    assert rows["feat_a"].cohort_role is None
    assert dag.adjustment_valid is False
    # Namespacing the latents per feature removes the shared node and the cycle.
    split = assemble_cohort_dag([a, b], treatment=T, outcome=Y, merge_latents=False)
    assert split.is_dag is True
    assert split.latent_nodes == ["U_x@feat_a", "U_x@feat_b", "U_z@feat_a", "U_z@feat_b"]


def test_role_drift_via_a_shared_latent_is_flagged_and_excluded():
    # Alone, F is an ancestor (U_q -> F -> Y). G says the treatment CAUSES U_q
    # (T -> U_q -> G -> Y). On the union T -> U_q -> F -> Y makes F a mediator.
    f = _att("feat_f", [("U_q", "feat_f"), ("feat_f", "Y"), ("T", "Y")])
    g = _att("feat_g", [("T", "U_q"), ("U_q", "feat_g"), ("feat_g", "Y"), ("T", "Y")])
    dag = assemble_cohort_dag([f, g], treatment=T, outcome=Y)
    row = next(r for r in dag.features if r.feature == "feat_f")
    assert (row.fragment_role, row.cohort_role, row.role_drift) == ("ancestor", "mediator", True)
    assert row.review_required is True
    assert any("role drift" in r for r in row.review_reasons)
    assert row.in_adjustment_set is False
    assert row.adjustment_exclusion == "cohort role mediator is not a backdoor variable"


def test_review_only_record_without_edges_is_listed_and_excluded():
    empty = _att("unauthored", [("T", "Y")])  # placeholder to build, then blank it
    empty.edges = []
    empty.edge_provenance = []
    empty.derived_role = None
    empty.review_required = True
    empty.review_reasons = ["LM call failed: provider down"]
    dag = assemble_cohort_dag([empty, _ancestor()], treatment=T, outcome=Y)
    row = next(r for r in dag.features if r.feature == "unauthored")
    assert row.n_edges == 0 and row.in_adjustment_set is False
    assert row.adjustment_exclusion == "no authored fragment (review only)"
    assert "unauthored" not in dag.nodes


def test_empty_input_yields_the_estimand_only_graph():
    dag = assemble_cohort_dag([], treatment=T, outcome=Y)
    assert dag.active_edges() == [[T, Y]]
    assert dag.adjustment_valid is True and dag.adjustment_set == []
    assert dag.warnings == []


def test_duplicate_feature_or_anchor_collision_is_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        assemble_cohort_dag([_ancestor(), _ancestor()], treatment=T, outcome=Y)
    with pytest.raises(ValueError, match="collides"):
        assemble_cohort_dag([_att(T, [(T, "Y"), ("T", "Y")])], treatment=T, outcome=Y)


def test_full_candidate_set_failure_means_no_observed_subset_is_admissible():
    """codex r2 MED 1 asked for a fallback search on the grounds that
    d-separation is not monotone. In the fragment vocabulary the case cannot
    arise (enumerated: docs/demos/results/2026-09-22_lane_b_structural_author_scaffold/
    nonmonotone_search.txt, 19,521 unions, 0 counter-examples): an ancestor
    candidate that two latents point into (one reaching T) needs itself to
    block the chain T <- U1 -> F -> Y, and conditioning on it opens a path only
    latents could block. The assembler must report that honestly rather than
    an empty set that reads as "nothing to adjust"."""
    c = _att("severity", [("severity", "T"), ("severity", "Y"), ("T", "Y")])
    f = _att(
        "atopy_marker",
        [
            ("U1", "T"),
            ("U1", "atopy_marker"),
            ("U2", "atopy_marker"),
            ("U2", "Y"),
            ("atopy_marker", "Y"),
            ("T", "Y"),
        ],
    )
    dag = assemble_cohort_dag([c, f], treatment=T, outcome=Y)
    g = nx.DiGraph(dag.active_edges())
    assert satisfies_backdoor_criterion(g, ["severity", "atopy_marker"], T, Y) is False
    assert satisfies_backdoor_criterion(g, ["severity"], T, Y) is False
    assert satisfies_backdoor_criterion(g, [], T, Y) is False
    assert dag.adjustment_valid is False
    assert dag.adjustment_set == [] and dag.minimal_adjustment_set == []
    assert any("no admissible OBSERVED adjustment set" in w for w in dag.warnings)
    assert dag.dag_structure_json()["adjustment_sets"] == []
