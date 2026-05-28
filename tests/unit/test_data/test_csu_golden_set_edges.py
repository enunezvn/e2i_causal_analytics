"""Pin the CSU edge-augmented golden fixture (Layer-4 Track-2B-v3 Phase 1).

``tests/fixtures/causal_role_golden_set_csu_edges.json`` carries, for the 31
``CSU_remibrutinib`` literature features, a domain-authored DAG (``edges`` +
``feature_node``/``treatment_node``/``outcome_node``) authored BLIND to the
independent literature ``ground_truth_role`` (which is preserved verbatim for
scoring). These tests pin the fixture's shape AND the leak-vs-accept decision
gate it was built to demonstrate: ``missed_leaks == 0`` — no LEAK feature
(mediator/collider/descendant) is ever placed in ACCEPT
(ancestor/confounder/instrument) by the deterministic ``extract_role`` decider.

The decider remains DARK in production; this fixture is the labeled CSU-cohort
acceptance evidence that gates a future production-authoring phase.
"""

import importlib.util
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
_FIXTURE = _REPO / "tests" / "fixtures" / "causal_role_golden_set_csu_edges.json"

_VALID_ROLES = frozenset(
    {"ancestor", "confounder", "instrument", "mediator", "collider", "descendant"}
)


def _load_fixture() -> dict:
    data: dict = json.loads(_FIXTURE.read_text())
    return data


def _load_measure_module():
    spec = importlib.util.spec_from_file_location(
        "measure_layer4_precision", _REPO / "scripts" / "measure_layer4_precision.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    # Register before exec: module-level @dataclass resolves itself in sys.modules.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_fixture_has_31_csu_entries():
    data = _load_fixture()
    entries = data["entries"]
    assert data["total_entries"] == 31
    assert len(entries) == 31
    assert all(e["cohort"] == "CSU_remibrutinib" for e in entries)


def test_every_entry_has_structural_keys_and_anchors():
    entries = _load_fixture()["entries"]
    for e in entries:
        # required structural keys for --decider structural scoring
        assert e["feature_node"] == e["feature_name"], e["feature_name"]
        assert e["treatment_node"] == "T", e["feature_name"]
        assert e["outcome_node"] == "Y", e["feature_name"]
        assert e["edges"], f"empty edges: {e['feature_name']}"
        # both T and Y must appear as nodes (the cohort frame anchors); otherwise
        # extract_role raises on nx.descendants(graph, missing_node).
        nodes = {n for edge in e["edges"] for n in edge}
        assert "T" in nodes and "Y" in nodes, e["feature_name"]
        # independent literature label preserved + valid
        assert e["ground_truth_role"] in _VALID_ROLES, e["feature_name"]


def test_no_prose_leaks_into_fixture():
    # Non-circularity hygiene: the blind authoring rationale/provenance must NOT
    # be carried into the scoring fixture (only the independent label + edges).
    entries = _load_fixture()["entries"]
    for e in entries:
        assert "rationale" not in e, e["feature_name"]
        assert "provenance" not in e, e["feature_name"]


def test_label_distribution_matches_literature_cohort():
    # The 31-entry CSU literature cohort distribution (independent labels).
    from collections import Counter

    entries = _load_fixture()["entries"]
    dist = Counter(e["ground_truth_role"] for e in entries)
    assert dist == {
        "confounder": 6,
        "instrument": 6,
        "ancestor": 5,
        "collider": 5,
        "mediator": 5,
        "descendant": 4,
    }


def test_every_dag_is_classifiable_by_extract_role():
    # Each authored DAG must yield a valid role (no unclassifiable/raise) via the
    # SAME path the eval uses (_structural_predict).
    mod = _load_measure_module()
    entries = _load_fixture()["entries"]
    for e in entries:
        role = mod._structural_predict(e)
        assert role in _VALID_ROLES, f"{e['feature_name']} -> {role}"


def test_leak_decision_gate_zero_missed_leaks():
    """THE GATE: scoring the CSU fixture yields zero missed leaks.

    A missed leak = a feature whose independent label is a LEAK role
    (mediator/collider/descendant) but whose authored DAG derives an ACCEPT role
    (ancestor/confounder/instrument). This is the safety-critical failure mode the
    structural decider must never exhibit before activation.
    """
    mod = _load_measure_module()
    entries = _load_fixture()["entries"]

    cm = mod.CohortMetrics(cohort="CSU_remibrutinib", gate="ungated")
    for e in entries:
        predicted = mod._structural_predict(e)
        truth = str(e["ground_truth_role"])
        cm.confusion.setdefault(truth, {})
        cm.confusion[truth][predicted] = cm.confusion[truth].get(predicted, 0) + 1

    metrics = mod._leak_decision_metrics({("CSU_remibrutinib", "ungated"): cm})
    assert metrics["missed_leaks"] == 0, metrics
    assert metrics["false_alarms"] == 0, metrics
    assert metrics["decided"] == 31, metrics
    assert metrics["leak_decision_accuracy"] == 1.0, metrics
