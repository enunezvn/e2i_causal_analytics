"""Track-2B authoring-yield harness (Plan v4 Layer B / Phase 2, Tasks 7-8).

The Track-2B experiment samples literature golden-set features across the three
temporal regimes, has their DAG edges authored *blind to the role label*, and
scores ``extract_role`` over those edges against the independent literature
``ground_truth_role``. This module is the deterministic, testable harness around
that experiment (sampling + edge-augmented fixture assembly); the authoring
itself is the experiment's data, not code.
"""

import importlib.util
import json
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
_MOD_PATH = _REPO / "scripts" / "layer4_trackb_authoring.py"


def _load_mod():
    spec = importlib.util.spec_from_file_location("layer4_trackb_authoring", _MOD_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _golden_entries() -> list[dict]:
    data = json.loads((_REPO / "tests" / "fixtures" / "causal_role_golden_set.json").read_text())
    return data["entries"]


def test_regime_of_classifies_three_regimes():
    mod = _load_mod()
    assert mod.regime_of({"derivation_pseudocode": "knowable_at=preindex"}) == "pre"
    assert mod.regime_of({"derivation_pseudocode": "knowable_at=index_date_minus_1"}) == "index"
    assert mod.regime_of({"derivation_pseudocode": "knowable_at=index_date"}) == "index"
    assert mod.regime_of({"derivation_pseudocode": "knowable_at=t_plus_180d"}) == "post"
    assert mod.regime_of({"derivation_pseudocode": "knowable_at=index_date_plus_90"}) == "post"


def test_sample_returns_requested_count():
    mod = _load_mod()
    sample = mod.stratified_sample(_golden_entries(), n=10, seed=0)
    assert len(sample) == 10


def test_sample_is_deterministic_for_seed():
    mod = _load_mod()
    entries = _golden_entries()
    a = [e["feature_name"] for e in mod.stratified_sample(entries, n=10, seed=0)]
    b = [e["feature_name"] for e in mod.stratified_sample(entries, n=10, seed=0)]
    assert a == b


def test_sample_covers_all_three_regimes():
    mod = _load_mod()
    sample = mod.stratified_sample(_golden_entries(), n=10, seed=0)
    regimes = {mod.regime_of(e) for e in sample}
    assert regimes == {"pre", "index", "post"}


def test_sample_covers_all_six_roles():
    mod = _load_mod()
    sample = mod.stratified_sample(_golden_entries(), n=10, seed=0)
    roles = {e["ground_truth_role"] for e in sample}
    assert roles == {
        "ancestor",
        "collider",
        "confounder",
        "descendant",
        "instrument",
        "mediator",
    }


# --- build_edge_augmented_fixture (Task 8 fixture assembly) ---

_GOLDEN = [
    {"feature_name": "x", "ground_truth_role": "confounder", "cohort": "C1", "rationale": "secret"},
    {"feature_name": "z", "ground_truth_role": "collider", "cohort": "C2", "rationale": "secret"},
]
_AUTHORED = {
    "x": {
        "feature_node": "x",
        "treatment_node": "T",
        "outcome_node": "Y",
        "edges": [["x", "T"], ["x", "Y"]],
    }
}


def test_build_preserves_independent_label_and_cohort():
    mod = _load_mod()
    fx = mod.build_edge_augmented_fixture(_GOLDEN, _AUTHORED)
    assert fx["total_entries"] == 1
    e = fx["entries"][0]
    # the INDEPENDENT literature label + cohort come from the golden entry,
    # never from the (blind) authored payload — that is what keeps it non-circular.
    assert e["ground_truth_role"] == "confounder"
    assert e["cohort"] == "C1"
    assert e["edges"] == [["x", "T"], ["x", "Y"]]
    assert e["feature_node"] == "x"


def test_build_only_includes_authored_features():
    mod = _load_mod()
    fx = mod.build_edge_augmented_fixture(_GOLDEN, _AUTHORED)
    assert [e["feature_name"] for e in fx["entries"]] == ["x"]


def test_build_entry_has_all_structural_keys():
    mod = _load_mod()
    e = mod.build_edge_augmented_fixture(_GOLDEN, _AUTHORED)["entries"][0]
    assert {
        "feature_name",
        "ground_truth_role",
        "cohort",
        "feature_node",
        "treatment_node",
        "outcome_node",
        "edges",
    } <= set(e)
    # no leakage of the prose rationale into the structural fixture
    assert "rationale" not in e


def test_build_rejects_unknown_authored_feature():
    mod = _load_mod()
    import pytest

    with pytest.raises((KeyError, ValueError)):
        mod.build_edge_augmented_fixture(_GOLDEN, {"nonexistent": _AUTHORED["x"]})
